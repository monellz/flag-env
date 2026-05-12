#!/usr/bin/env python3
"""
Standalone profiler/benchmark driver for block-wise FP8 ``bhr,hdr->bhd`` einsum.

Targets:
  - DeepGEMM ``fp8_einsum("bhr,hdr->bhd", ...)``        (FP8 W8A8, per-token A x per-block B)
  - FlagGems ``w8a8_block_fp8_matmul`` invoked once per head (FP8 W8A8)
  - DeepGEMM ``einsum(..., use_cublaslt=True)``         (BF16 cuBLASLt baseline)

Examples:
  python profile_fp8_bhr_hdr_bhd.py --shape-preset default --backends all
  python profile_fp8_bhr_hdr_bhd.py --mode profile --backends deepgemm,flaggems --batch 4096 --heads 8 --r 4096 --d 1024
  nsys profile -o fp8_bmm --capture-range=nvtx --capture-range-end=stop python profile_fp8_bhr_hdr_bhd.py --mode profile --backends deepgemm
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
FLAGGEMS_ROOT = ROOT / "FlagGems"
DEEPGEMM_ROOT = ROOT / "DeepGEMM"


DEFAULT_BLOCK_SHAPE = (128, 128)
DEFAULT_SEED = 20260512

torch = None
triton = None
flag_gems = None
deep_gemm = None
per_token_cast_to_fp8 = None
per_block_cast_to_fp8 = None
ceil_div = None
calc_diff = None
count_bytes = None


# (b, h, r, d) tuples. h/r/d groups follow the user-defined model classes;
# b sweeps mirror DeepGEMM/tests/test_einsum.py::test_fp8_bhr_hdr_bhd.
_BATCH_SIZES = (1, 4, 32, 128, 4096, 8192, 16384, 32768)
_HRD_GROUPS = {
    "small": (128, 512, 128),
    "flash": (8, 4096, 1024),
    "pro":   (16, 7168, 1024),
}
BHRD_SHAPES: Dict[str, List[Tuple[int, int, int, int]]] = {
    name: [(b, h, r, d) for b in _BATCH_SIZES]
    for name, (h, r, d) in _HRD_GROUPS.items()
}


@dataclass
class EinsumCase:
    shape_name: str
    b: int
    h: int
    r: int
    d: int

    # BF16 references (used by cuBLASLt baseline + correctness reference)
    x_bf16: "torch.Tensor"
    y_bf16: "torch.Tensor"

    # FP8 quantised inputs (DeepGEMM packing): (data, scale)
    x_fp8: Tuple["torch.Tensor", "torch.Tensor"]   # (b, h, r) FP8, (b, h, r/128) FP32
    y_fp8: Tuple["torch.Tensor", "torch.Tensor"]   # (h, d, r) FP8, (h, d/128, r/128) FP32

    # Reusable output buffers (one per backend, so backends don't fight over the same buffer)
    z_deepgemm: "torch.Tensor"   # (b, h, d) BF16
    z_cublaslt: "torch.Tensor"   # (b, h, d) BF16
    z_flaggems: "torch.Tensor"   # (b, h, d) BF16

    # Pre-split per-head contiguous tensors for FlagGems (set in __post_init__-like hook)
    fg_a:   List["torch.Tensor"] = field(default_factory=list)  # h x (b, r)        FP8
    fg_as:  List["torch.Tensor"] = field(default_factory=list)  # h x (b, r/128)    FP32
    fg_b:   List["torch.Tensor"] = field(default_factory=list)  # h x (d, r)        FP8
    fg_bs:  List["torch.Tensor"] = field(default_factory=list)  # h x (d/128, r/128) FP32


def load_runtime_deps() -> None:
    global torch
    global triton
    global flag_gems
    global deep_gemm
    global per_token_cast_to_fp8
    global per_block_cast_to_fp8
    global ceil_div
    global calc_diff
    global count_bytes

    if torch is not None:
        return

    # deep_gemm is installed in the venv (its `_C` ext lives in site-packages);
    # flag_gems is installed in editable mode. Don't inject the source dirs onto
    # sys.path or we shadow the installed packages and break the C extension import.
    import torch as torch_mod
    import triton as triton_mod
    import flag_gems as flag_gems_mod
    import deep_gemm as deep_gemm_mod
    from deep_gemm.testing import (
        calc_diff as calc_diff_mod,
        count_bytes as count_bytes_mod,
    )
    from deep_gemm.utils.math import (
        ceil_div as ceil_div_mod,
        per_block_cast_to_fp8 as per_block_cast_to_fp8_mod,
        per_token_cast_to_fp8 as per_token_cast_to_fp8_mod,
    )

    torch = torch_mod
    triton = triton_mod
    flag_gems = flag_gems_mod
    deep_gemm = deep_gemm_mod
    per_token_cast_to_fp8 = per_token_cast_to_fp8_mod
    per_block_cast_to_fp8 = per_block_cast_to_fp8_mod
    ceil_div = ceil_div_mod
    calc_diff = calc_diff_mod
    count_bytes = count_bytes_mod


def get_tabulate() -> Callable:
    from tabulate import tabulate

    return tabulate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark/profile DeepGEMM and FlagGems on "bhr,hdr->bhd" block-wise FP8 GEMM.'
    )
    parser.add_argument(
        "--mode",
        choices=("bench", "profile"),
        default="bench",
        help="bench: repeated timing, profile: warmup then marked single/limited calls",
    )
    parser.add_argument(
        "--backends",
        default="all",
        help="comma-separated: deepgemm,flaggems,cublaslt or all",
    )
    parser.add_argument(
        "--shape-preset",
        default="flash",
        choices=tuple(BHRD_SHAPES.keys()) + ("all",),
        help="Preset shape family: small (h=128,r=512,d=128), flash (h=8,r=4096,d=1024), pro (h=16,r=7168,d=1024)",
    )
    parser.add_argument("--batch", type=int, default=None, help="Custom b")
    parser.add_argument("--heads", type=int, default=None, help="Custom h")
    parser.add_argument("--r", type=int, default=None, help="Custom r (reduction dim)")
    parser.add_argument("--d", type=int, default=None, help="Custom d (output dim)")
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_SHAPE[0])
    parser.add_argument("--block-k", type=int, default=DEFAULT_BLOCK_SHAPE[1])
    parser.add_argument(
        "--no-ue8m0",
        dest="use_ue8m0",
        action="store_false",
        help="Disable UE8M0 rounding of FP32 scales (test_fp8_bhr_hdr_bhd defaults to UE8M0 on)",
    )
    parser.set_defaults(use_ue8m0=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--use-cudagraph",
        action="store_true",
        help="Use triton.testing.do_bench_cudagraph in bench mode",
    )
    parser.add_argument(
        "--profile-repeat",
        type=int,
        default=1,
        help="Measured calls per backend/case in profile mode",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate FP8 backends against the BF16 torch.einsum reference (uses calc_diff)",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=1e-3,
        help="calc_diff threshold for --check (matches test_fp8_bhr_hdr_bhd's 1e-3)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Optional path to dump structured JSON results",
    )
    parser.add_argument(
        "--empty-cache",
        action="store_true",
        help="Call torch.cuda.empty_cache() before each backend/case",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this script.")


def synchronize() -> None:
    torch.cuda.synchronize()


def nvtx_range(name: str):
    class _Range:
        def __enter__(self):
            torch.cuda.nvtx.range_push(name)

        def __exit__(self, exc_type, exc, tb):
            torch.cuda.nvtx.range_pop()

    return _Range()


def parse_backends(raw: str) -> List[str]:
    if raw == "all":
        return ["deepgemm", "flaggems", "cublaslt"]
    backends = [item.strip().lower() for item in raw.split(",") if item.strip()]
    valid = {"deepgemm", "flaggems", "cublaslt"}
    invalid = sorted(set(backends) - valid)
    if invalid:
        raise ValueError(f"Unsupported backends: {', '.join(invalid)}")
    return backends


def resolve_shapes(args: argparse.Namespace) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    custom_fields = [args.batch, args.heads, args.r, args.d]
    if any(value is not None for value in custom_fields):
        if not all(value is not None for value in custom_fields):
            raise ValueError("Custom shape requires --batch --heads --r --d together.")
        shape = (args.batch, args.heads, args.r, args.d)
        return [("custom", shape)]

    if args.shape_preset == "all":
        resolved: List[Tuple[str, Tuple[int, int, int, int]]] = []
        for preset_name, shapes in BHRD_SHAPES.items():
            for idx, shape in enumerate(shapes):
                resolved.append((f"{preset_name}_{idx}", shape))
        return resolved

    return [
        (f"{args.shape_preset}_{idx}", shape)
        for idx, shape in enumerate(BHRD_SHAPES[args.shape_preset])
    ]


def generate_fp8_bhrd_case(
    shape_name: str,
    config: Tuple[int, int, int, int],
    block_shape: Tuple[int, int],
    use_ue8m0: bool,
) -> EinsumCase:
    b, h, r, d = config
    block_n, block_k = block_shape
    assert block_n == 128 and block_k == 128, "FP8 bhr,hdr->bhd path expects (128, 128) blocks."
    assert r % block_k == 0, f"r={r} must be divisible by block_k={block_k}"
    assert d % block_n == 0, f"d={d} must be divisible by block_n={block_n}"
    device = torch.device("cuda")

    # Mirrors DeepGEMM/tests/test_einsum.py::test_fp8_bhr_hdr_bhd.
    x = torch.randn((b, h, r), device=device, dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device=device, dtype=torch.bfloat16)

    # Per-token quant on A (1D: scale is per (b*h)-row, blocked along K=r).
    x_fp8 = per_token_cast_to_fp8(x.view(-1, r), use_ue8m0=use_ue8m0)
    x_data  = x_fp8[0].view(b, h, r)
    x_scale = x_fp8[1].view(b, h, ceil_div(r, block_k))

    # Per-block quant on B (2D: scale is per (block_n=128, block_k=128) tile); done per-head.
    y_data  = torch.empty_like(y, dtype=torch.float8_e4m3fn)
    y_scale = torch.empty((h, ceil_div(d, block_n), ceil_div(r, block_k)),
                          device=device, dtype=torch.float32)
    for i in range(h):
        y_data[i], y_scale[i] = per_block_cast_to_fp8(y[i], use_ue8m0=use_ue8m0)

    z_dg = torch.empty((b, h, d), device=device, dtype=torch.bfloat16)
    z_cb = torch.empty((b, h, d), device=device, dtype=torch.bfloat16)
    z_fg = torch.empty((b, h, d), device=device, dtype=torch.bfloat16)

    # Pre-split contiguous per-head views for FlagGems (single matmul per call).
    fg_a, fg_as, fg_b, fg_bs = [], [], [], []
    for i in range(h):
        fg_a.append(x_data[:, i, :].contiguous())
        fg_as.append(x_scale[:, i, :].contiguous())
        fg_b.append(y_data[i].contiguous())
        fg_bs.append(y_scale[i].contiguous())

    return EinsumCase(
        shape_name=shape_name,
        b=b, h=h, r=r, d=d,
        x_bf16=x, y_bf16=y,
        x_fp8=(x_data, x_scale),
        y_fp8=(y_data, y_scale),
        z_deepgemm=z_dg, z_cublaslt=z_cb, z_flaggems=z_fg,
        fg_a=fg_a, fg_as=fg_as, fg_b=fg_b, fg_bs=fg_bs,
    )


def load_deepgemm_backend() -> Callable[[EinsumCase], "torch.Tensor"]:
    def _run(case: EinsumCase) -> "torch.Tensor":
        deep_gemm.fp8_einsum("bhr,hdr->bhd", case.x_fp8, case.y_fp8, case.z_deepgemm)
        return case.z_deepgemm

    return _run


def load_cublaslt_backend() -> Callable[[EinsumCase], "torch.Tensor"]:
    def _run(case: EinsumCase) -> "torch.Tensor":
        deep_gemm.einsum(
            "bhr,hdr->bhd", case.x_bf16, case.y_bf16, case.z_cublaslt, use_cublaslt=True
        )
        return case.z_cublaslt

    return _run


def load_flaggems_backend() -> Callable[[EinsumCase], "torch.Tensor"]:
    block_shape = list(DEFAULT_BLOCK_SHAPE)

    def _run(case: EinsumCase) -> "torch.Tensor":
        # FlagGems' w8a8_block_fp8_matmul handles one (M, K) @ (N, K).T at a time.
        # Loop over heads and write per-head output back into the (b, h, d) buffer.
        for i in range(case.h):
            out = flag_gems.w8a8_block_fp8_matmul(
                case.fg_a[i],
                case.fg_b[i],
                case.fg_as[i],
                case.fg_bs[i],
                block_size=block_shape,
                output_dtype=torch.bfloat16,
            )
            case.z_flaggems[:, i, :].copy_(out)
        return case.z_flaggems

    return _run


def load_backend_runners(
    backends: Sequence[str],
) -> Dict[str, Callable[[EinsumCase], "torch.Tensor"]]:
    runners: Dict[str, Callable[[EinsumCase], "torch.Tensor"]] = {}
    for backend in backends:
        try:
            if backend == "deepgemm":
                runners[backend] = load_deepgemm_backend()
            elif backend == "flaggems":
                runners[backend] = load_flaggems_backend()
            elif backend == "cublaslt":
                runners[backend] = load_cublaslt_backend()
        except Exception as exc:
            print(f"[skip] backend={backend} unavailable: {exc}", file=sys.stderr)
    if not runners:
        raise RuntimeError("No requested backends are available.")
    return runners


def run_once(
    backend: str,
    runner: Callable[[EinsumCase], "torch.Tensor"],
    case: EinsumCase,
) -> "torch.Tensor":
    return runner(case)


def case_flops(case: EinsumCase) -> float:
    return 2.0 * case.b * case.h * case.r * case.d


def case_bytes(case: EinsumCase, backend: str) -> int:
    # FP8 backends touch FP8 data + FP32 scales + BF16 output;
    # cuBLASLt operates on BF16 directly.
    if backend == "cublaslt":
        return count_bytes((case.x_bf16, case.y_bf16, case.z_cublaslt))
    out_buf = case.z_flaggems if backend == "flaggems" else case.z_deepgemm
    return count_bytes((case.x_fp8, case.y_fp8, out_buf))


def bench_backend(
    backend: str,
    runner: Callable[[EinsumCase], "torch.Tensor"],
    case: EinsumCase,
    warmup: int,
    repeat: int,
    use_cudagraph: bool,
) -> dict:
    fn = lambda: run_once(backend, runner, case)

    if use_cudagraph:
        for _ in range(5):
            fn()
        synchronize()
        median_ms = triton.testing.do_bench_cudagraph(fn=fn, rep=repeat, return_mode="median")
        synchronize()
    else:
        synchronize()
        median_ms = triton.testing.do_bench(
            fn=fn, warmup=warmup, rep=repeat, return_mode="median"
        )
        synchronize()

    sec = median_ms * 1e-3
    tflops = case_flops(case) / sec / 1e12
    gbps = case_bytes(case, backend) / sec / 1e9
    return {
        "backend": backend,
        "shape_name": case.shape_name,
        "shape": [case.b, case.h, case.r, case.d],
        "median_ms": float(median_ms),
        "tflops": float(tflops),
        "gbps": float(gbps),
        "warmup": warmup,
        "repeat": repeat,
        "use_cudagraph": use_cudagraph,
    }


def profile_backend(
    backend: str,
    runner: Callable[[EinsumCase], "torch.Tensor"],
    case: EinsumCase,
    warmup: int,
    repeat: int,
) -> dict:
    for _ in range(warmup):
        run_once(backend, runner, case)
    synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    try:
        latencies_ms: List[float] = []
        for _ in range(repeat):
            start = time.perf_counter_ns()
            with nvtx_range(f"fp8_einsum::{backend}::{case.shape_name}"):
                run_once(backend, runner, case)
            synchronize()
            end = time.perf_counter_ns()
            latencies_ms.append((end - start) / 1e6)
    finally:
        torch.cuda.cudart().cudaProfilerStop()

    median_ms = statistics.median(latencies_ms)
    sec = median_ms * 1e-3
    tflops = case_flops(case) / sec / 1e12
    gbps = case_bytes(case, backend) / sec / 1e9
    return {
        "backend": backend,
        "shape_name": case.shape_name,
        "shape": [case.b, case.h, case.r, case.d],
        "profile_repeat": repeat,
        "warmup": warmup,
        "median_ms": float(median_ms),
        "mean_ms": float(statistics.fmean(latencies_ms)),
        "tflops": float(tflops),
        "gbps": float(gbps),
    }


def maybe_check_outputs(
    case: EinsumCase,
    runners: Dict[str, Callable[[EinsumCase], "torch.Tensor"]],
    diff_threshold: float,
) -> List[dict]:
    # Reference: BF16 torch.einsum on the original (un-quantised) inputs.
    reference = torch.einsum("bhr,hdr->bhd", case.x_bf16, case.y_bf16)
    synchronize()

    checks: List[dict] = []
    for backend, runner in runners.items():
        output = run_once(f"{backend}_check", runner, case).clone()
        synchronize()
        diff = float(calc_diff(output, reference))
        max_abs = (output.float() - reference.float()).abs().max().item()
        checks.append(
            {
                "backend": backend,
                "shape_name": case.shape_name,
                "calc_diff": diff,
                "passed": bool(diff < diff_threshold),
                "max_abs_diff": float(max_abs),
                "threshold": diff_threshold,
            }
        )
    return checks


def _shape_text(shape: Sequence[int]) -> str:
    return f"b={shape[0]} h={shape[1]} r={shape[2]} d={shape[3]}"


def print_results_table(results: Sequence[dict], mode: str) -> None:
    if not results:
        return

    tabulate = get_tabulate()

    if mode == "bench":
        rows = [
            [
                r["backend"],
                r["shape_name"],
                _shape_text(r["shape"]),
                f"{r['median_ms']:.3f}",
                f"{r['tflops']:.1f}",
                f"{r['gbps']:.0f}",
                r["warmup"],
                r["repeat"],
                r["use_cudagraph"],
            ]
            for r in results
        ]
        headers = [
            "backend", "shape_name", "shape",
            "median_ms", "TFLOPS", "GB/s", "warmup", "repeat", "cudagraph",
        ]
    else:
        rows = [
            [
                r["backend"],
                r["shape_name"],
                _shape_text(r["shape"]),
                f"{r['median_ms']:.3f}",
                f"{r['mean_ms']:.3f}",
                f"{r['tflops']:.1f}",
                f"{r['gbps']:.0f}",
                r["warmup"],
                r["profile_repeat"],
            ]
            for r in results
        ]
        headers = [
            "backend", "shape_name", "shape",
            "median_ms", "mean_ms", "TFLOPS", "GB/s", "warmup", "profile_repeat",
        ]

    print(tabulate(rows, headers=headers, tablefmt="github"))


def print_checks_table(checks: Sequence[dict]) -> None:
    if not checks:
        return

    tabulate = get_tabulate()
    rows = [
        [
            c["backend"],
            c["shape_name"],
            c["passed"],
            f"{c['calc_diff']:.2e}",
            f"{c['max_abs_diff']:.6f}",
            c["threshold"],
        ]
        for c in checks
    ]
    headers = ["backend", "shape_name", "passed", "calc_diff", "max_abs_diff", "threshold"]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def main() -> int:
    args = parse_args()
    load_runtime_deps()
    ensure_cuda()
    seed_everything(args.seed)

    backends = parse_backends(args.backends)
    block_shape = (args.block_n, args.block_k)

    if block_shape != DEFAULT_BLOCK_SHAPE:
        raise ValueError(
            f"Only block shape {DEFAULT_BLOCK_SHAPE} is supported by this script "
            "(FlagGems' w8a8_block_fp8_matmul + DeepGEMM's default fp8_einsum recipe)."
        )

    runners = load_backend_runners(backends)
    shapes = resolve_shapes(args)

    print(
        f"# mode={args.mode} backends={','.join(runners.keys())} "
        f"block_shape={list(block_shape)} use_ue8m0={args.use_ue8m0}"
    )

    all_results: List[dict] = []
    all_checks: List[dict] = []

    for shape_name, config in shapes:
        case = generate_fp8_bhrd_case(
            shape_name=shape_name,
            config=config,
            block_shape=block_shape,
            use_ue8m0=args.use_ue8m0,
        )
        if args.check:
            all_checks.extend(
                maybe_check_outputs(
                    case=case,
                    runners=runners,
                    diff_threshold=args.diff_threshold,
                )
            )
        for backend, runner in runners.items():
            if args.empty_cache:
                torch.cuda.empty_cache()
            result = (
                bench_backend(
                    backend=backend,
                    runner=runner,
                    case=case,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    use_cudagraph=args.use_cudagraph,
                )
                if args.mode == "bench"
                else profile_backend(
                    backend=backend,
                    runner=runner,
                    case=case,
                    warmup=args.warmup,
                    repeat=args.profile_repeat,
                )
            )
            all_results.append(result)

    print_results_table(all_results, args.mode)

    if all_checks:
        print_checks_table(all_checks)

    if args.json_path:
        output = {
            "mode": args.mode,
            "block_shape": list(block_shape),
            "use_ue8m0": args.use_ue8m0,
            "results": all_results,
            "checks": all_checks,
        }
        Path(args.json_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"# wrote json to {args.json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
