#!/usr/bin/env python3
"""
Standalone profiler/benchmark driver for block-wise FP8 W8A8 fused MoE.

Targets:
  - FlagGems fused_experts_impl
  - vLLM fused_experts_impl
  - hpc-ops fuse_moe_blockwise_fp8

Examples:
  python profile_fused_moe_fp8_blockwise.py --shape-preset mixtral --backends all
  python profile_fused_moe_fp8_blockwise.py --mode profile --backends flaggems --tokens 128 --experts 8 --hidden 4096 --intermediate 14336 --topk 2
  nsys profile -o fg_moe --capture-range=nvtx --capture-range-end=stop python profile_fused_moe_fp8_blockwise.py --mode profile --backends hpc
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parent
FLAGGEMS_ROOT = ROOT / "FlagGems"
HPC_ROOT = ROOT / "hpc-ops"


DEFAULT_BLOCK_SHAPE = (128, 128)
DEFAULT_SEED = 20260329

torch = None
flag_gems = None
per_token_group_quant_fp8 = None
triton = None

MOE_SHAPES: Dict[str, List[tuple[int, int, int, int, int]]] = {
    "mixtral": [
        (1, 8, 4096, 14336, 2),
        (4, 8, 4096, 14336, 2),
        (16, 8, 4096, 14336, 2),
        (64, 8, 4096, 14336, 2),
        (128, 8, 4096, 14336, 2),
        (256, 8, 4096, 14336, 2),
        (512, 8, 4096, 14336, 2),
    ],
    "deepseek_v3_tp8": [
        (1, 256, 7168, 2048, 8),
        (4, 256, 7168, 2048, 8),
        (16, 256, 7168, 2048, 8),
        (64, 256, 7168, 2048, 8),
        (128, 256, 7168, 2048, 8),
        (256, 256, 7168, 2048, 8),
    ],
    "qwen3_5_397b_a17b": [
        (1, 512, 4096, 1024, 10),
        (4, 512, 4096, 1024, 10),
        (16, 512, 4096, 1024, 10),
        (64, 512, 4096, 1024, 10),
        (128, 512, 4096, 1024, 10),
        (256, 512, 4096, 1024, 10),
    ],
}


@dataclass
class MoECase:
    shape_name: str
    num_tokens: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    topk: int
    hidden_states: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: torch.Tensor
    w2_scale: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor


def load_runtime_deps() -> None:
    global torch
    global flag_gems
    global per_token_group_quant_fp8
    global triton

    if torch is not None:
        return

    for path in (FLAGGEMS_ROOT / "src", FLAGGEMS_ROOT):
        sys.path.insert(0, str(path))

    for build_dir in sorted(HPC_ROOT.glob("build/lib.*/")):
        sys.path.insert(0, str(build_dir))

    import torch as torch_mod
    import triton as triton_mod
    import flag_gems as flag_gems_mod
    from flag_gems.ops.per_token_group_quant_fp8 import (
        per_token_group_quant_fp8 as per_token_group_quant_fp8_mod,
    )

    torch = torch_mod
    triton = triton_mod
    flag_gems = flag_gems_mod
    per_token_group_quant_fp8 = per_token_group_quant_fp8_mod


def get_tabulate() -> Callable:
    from tabulate import tabulate

    return tabulate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark/profile FlagGems/vLLM/hpc block-wise FP8 W8A8 fused MoE."
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
        help="comma-separated: flaggems,vllm,hpc or all",
    )
    parser.add_argument(
        "--shape-preset",
        default="mixtral",
        choices=tuple(MOE_SHAPES.keys()) + ("all",),
        help="Preset shape family to run when custom shape is not provided",
    )
    parser.add_argument("--tokens", type=int, default=256, help="Custom num_tokens")
    parser.add_argument("--experts", type=int, default=512, help="Custom num_experts")
    parser.add_argument("--hidden", type=int, default=4096, help="Custom hidden_size")
    parser.add_argument("--intermediate", type=int, default=1024, help="Custom intermediate_size")
    parser.add_argument("--topk", type=int, default=10, help="Custom topk")
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_SHAPE[0])
    parser.add_argument("--block-k", type=int, default=DEFAULT_BLOCK_SHAPE[1])
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
        "--sorted-topk-ids",
        action="store_true",
        help="Force sorted topk ids. Required for hpc fairness/compatibility.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate outputs against FlagGems for the same case",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=2e-2,
        help="Relative tolerance for --check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-2,
        help="Absolute tolerance for --check",
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
        return ["flaggems", "vllm", "hpc"]
    backends = [item.strip().lower() for item in raw.split(",") if item.strip()]
    valid = {"flaggems", "vllm", "hpc"}
    invalid = sorted(set(backends) - valid)
    if invalid:
        raise ValueError(f"Unsupported backends: {', '.join(invalid)}")
    return backends


def resolve_shapes(args: argparse.Namespace) -> List[tuple[str, tuple[int, int, int, int, int]]]:
    custom_fields = [args.tokens, args.experts, args.hidden, args.intermediate, args.topk]
    if any(value is not None for value in custom_fields):
        if not all(value is not None for value in custom_fields):
            raise ValueError("Custom shape requires --tokens --experts --hidden --intermediate --topk together.")
        shape = (args.tokens, args.experts, args.hidden, args.intermediate, args.topk)
        return [("custom", shape)]

    if args.shape_preset == "all":
        resolved: List[tuple[str, tuple[int, int, int, int, int]]] = []
        for preset_name, shapes in MOE_SHAPES.items():
            for idx, shape in enumerate(shapes):
                resolved.append((f"{preset_name}_{idx}", shape))
        return resolved

    return [
        (f"{args.shape_preset}_{idx}", shape)
        for idx, shape in enumerate(MOE_SHAPES[args.shape_preset])
    ]


def generate_fp8_blockwise_case(
    shape_name: str,
    config: tuple[int, int, int, int, int],
    block_shape: tuple[int, int],
    sorted_topk_ids: bool,
) -> MoECase:
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    block_n, block_k = block_shape
    device = torch.device("cuda")

    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )
    w1 = (
        torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * (1.0 / hidden_size**0.5)
    ).to(torch.float8_e4m3fn)
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * (1.0 / intermediate_size**0.5)
    ).to(torch.float8_e4m3fn)
    w1_scale = torch.rand(
        num_experts,
        ceil(intermediate_size * 2 / block_n),
        ceil(hidden_size / block_k),
        device=device,
        dtype=torch.float32,
    ) + 0.01
    w2_scale = torch.rand(
        num_experts,
        ceil(hidden_size / block_n),
        ceil(intermediate_size / block_k),
        device=device,
        dtype=torch.float32,
    ) + 0.01

    if sorted_topk_ids:
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device
        )
        topk_ids, _ = torch.sort(topk_ids, dim=1)
        topk_weights = torch.softmax(
            torch.randn((num_tokens, topk), device=device, dtype=torch.float32),
            dim=-1,
        )
    else:
        gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return MoECase(
        shape_name=shape_name,
        num_tokens=num_tokens,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        topk=topk,
        hidden_states=hidden_states.contiguous(),
        w1=w1.contiguous(),
        w2=w2.contiguous(),
        w1_scale=w1_scale.contiguous(),
        w2_scale=w2_scale.contiguous(),
        topk_weights=topk_weights.to(torch.float32).contiguous(),
        topk_ids=topk_ids.to(torch.int32).contiguous(),
    )


def load_vllm_backend() -> Callable[[MoECase, tuple[int, int]], torch.Tensor]:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )

    def _run(case: MoECase, block_shape: tuple[int, int]) -> torch.Tensor:
        return vllm_fused_experts_impl(
            case.hidden_states,
            case.w1,
            case.w2,
            case.topk_weights,
            case.topk_ids,
            inplace=False,
            activation="silu",
            use_fp8_w8a8=True,
            w1_scale=case.w1_scale,
            w2_scale=case.w2_scale,
            block_shape=list(block_shape),
        )

    return _run


def load_flaggems_backend() -> Callable[[MoECase, tuple[int, int]], torch.Tensor]:
    def _run(case: MoECase, block_shape: tuple[int, int]) -> torch.Tensor:
        return flag_gems.fused_experts_impl(
            case.hidden_states,
            case.w1,
            case.w2,
            case.topk_weights,
            case.topk_ids,
            global_num_experts=case.num_experts,
            use_fp8_w8a8=True,
            w1_scale=case.w1_scale,
            w2_scale=case.w2_scale,
            block_shape=list(block_shape),
        )

    return _run


def load_hpc_backend() -> Callable[[MoECase, tuple[int, int]], torch.Tensor]:
    import hpc

    def _run(case: MoECase, block_shape: tuple[int, int]) -> torch.Tensor:
        _, block_k = block_shape
        hidden_states_q, a1_scale = per_token_group_quant_fp8(
            case.hidden_states,
            group_size=block_k,
            dtype=torch.float8_e4m3fn,
            column_major_scales=False,
            scale_ue8m0=False,
        )
        return hpc.fuse_moe_blockwise_fp8(
            hidden_states_q,
            a1_scale,
            case.w1,
            case.w1_scale,
            case.w2,
            case.w2_scale,
            case.topk_ids,
            case.topk_weights,
            0,
            case.num_experts,
        )

    return _run


def load_backend_runners(backends: Sequence[str]) -> Dict[str, Callable[[MoECase, tuple[int, int]], torch.Tensor]]:
    runners: Dict[str, Callable[[MoECase, tuple[int, int]], torch.Tensor]] = {}
    for backend in backends:
        try:
            if backend == "flaggems":
                runners[backend] = load_flaggems_backend()
            elif backend == "vllm":
                runners[backend] = load_vllm_backend()
            elif backend == "hpc":
                runners[backend] = load_hpc_backend()
        except Exception as exc:
            print(f"[skip] backend={backend} unavailable: {exc}", file=sys.stderr)
    if not runners:
        raise RuntimeError("No requested backends are available.")
    return runners


def run_once(
    backend: str,
    runner: Callable[[MoECase, tuple[int, int]], torch.Tensor],
    case: MoECase,
    block_shape: tuple[int, int],
) -> torch.Tensor:
    # with nvtx_range(f"fused_moe_fp8_blockwise::{backend}::{case.shape_name}"):
    #     return runner(case, block_shape)
    return runner(case, block_shape)


def bench_backend(
    backend: str,
    runner: Callable[[MoECase, tuple[int, int]], torch.Tensor],
    case: MoECase,
    block_shape: tuple[int, int],
    warmup: int,
    repeat: int,
    use_cudagraph: bool,
) -> dict:
    fn = lambda: run_once(backend, runner, case, block_shape)

    if use_cudagraph:
        # Trigger lazy init/JIT before graph capture, matching FlagGems benchmark behavior.
        for _ in range(5):
            fn()
        synchronize()
        median_ms = triton.testing.do_bench_cudagraph(fn=fn, rep=repeat, return_mode="median")
        synchronize()
        mean_ms = median_ms
        min_ms = median_ms
        max_ms = median_ms
    else:
        synchronize()
        median_ms = triton.testing.do_bench(
            fn=fn,
            warmup=warmup,
            rep=repeat,
            return_mode="median",
        )
        synchronize()
        mean_ms = median_ms
        min_ms = median_ms
        max_ms = median_ms

    return {
        "backend": backend,
        "shape_name": case.shape_name,
        "shape": [
            case.num_tokens,
            case.num_experts,
            case.hidden_size,
            case.intermediate_size,
            case.topk,
        ],
        "median_ms": float(median_ms),
        "mean_ms": float(mean_ms),
        "min_ms": float(min_ms),
        "max_ms": float(max_ms),
        "repeat": repeat,
        "warmup": warmup,
        "use_cudagraph": use_cudagraph,
    }


def profile_backend(
    backend: str,
    runner: Callable[[MoECase, tuple[int, int]], torch.Tensor],
    case: MoECase,
    block_shape: tuple[int, int],
    warmup: int,
    repeat: int,
) -> dict:
    for _ in range(warmup):
        run_once(backend, runner, case, block_shape)
    synchronize()

    # Let external profilers capture only the marked region.
    torch.cuda.cudart().cudaProfilerStart()
    try:
        latencies_ms: List[float] = []
        for _ in range(repeat):
            start = time.perf_counter_ns()
            run_once(backend, runner, case, block_shape)
            synchronize()
            end = time.perf_counter_ns()
            latencies_ms.append((end - start) / 1e6)
    finally:
        torch.cuda.cudart().cudaProfilerStop()

    return {
        "backend": backend,
        "shape_name": case.shape_name,
        "shape": [
            case.num_tokens,
            case.num_experts,
            case.hidden_size,
            case.intermediate_size,
            case.topk,
        ],
        "profile_repeat": repeat,
        "warmup": warmup,
        "median_ms": statistics.median(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
    }


def maybe_check_outputs(
    case: MoECase,
    block_shape: tuple[int, int],
    runners: Dict[str, Callable[[MoECase, tuple[int, int]], torch.Tensor]],
    atol: float,
    rtol: float,
) -> List[dict]:
    if "flaggems" not in runners or len(runners) <= 1:
        return []

    reference = run_once("flaggems_check", runners["flaggems"], case, block_shape)
    synchronize()
    checks: List[dict] = []

    for backend, runner in runners.items():
        if backend == "flaggems":
            continue
        output = run_once(f"{backend}_check", runner, case, block_shape)
        synchronize()
        max_abs = (output - reference).abs().max().item()
        ok = torch.allclose(output, reference, atol=atol, rtol=rtol)
        checks.append(
            {
                "backend": backend,
                "shape_name": case.shape_name,
                "allclose_to_flaggems": bool(ok),
                "max_abs_diff": float(max_abs),
                "atol": atol,
                "rtol": rtol,
            }
        )
    return checks


def _shape_text(shape: Sequence[int]) -> str:
    return f"M={shape[0]} E={shape[1]} K={shape[2]} N={shape[3]} topk={shape[4]}"


def print_results_table(results: Sequence[dict], mode: str) -> None:
    if not results:
        return

    tabulate = get_tabulate()

    if mode == "bench":
        rows = [
            [
                result["backend"],
                result["shape_name"],
                _shape_text(result["shape"]),
                f"{result['median_ms']:.3f}",
                f"{result['mean_ms']:.3f}",
                f"{result['min_ms']:.3f}",
                f"{result['max_ms']:.3f}",
                result["warmup"],
                result["repeat"],
                result["use_cudagraph"],
            ]
            for result in results
        ]
        headers = [
            "backend",
            "shape_name",
            "shape",
            "median_ms",
            "mean_ms",
            "min_ms",
            "max_ms",
            "warmup",
            "repeat",
            "cudagraph",
        ]
    else:
        rows = [
            [
                result["backend"],
                result["shape_name"],
                _shape_text(result["shape"]),
                f"{result['median_ms']:.3f}",
                f"{result['mean_ms']:.3f}",
                result["warmup"],
                result["profile_repeat"],
            ]
            for result in results
        ]
        headers = [
            "backend",
            "shape_name",
            "shape",
            "median_ms",
            "mean_ms",
            "warmup",
            "profile_repeat",
        ]

    print(tabulate(rows, headers=headers, tablefmt="github"))


def print_checks_table(checks: Sequence[dict]) -> None:
    if not checks:
        return

    tabulate = get_tabulate()

    rows = [
        [
            check["backend"],
            check["shape_name"],
            check["allclose_to_flaggems"],
            f"{check['max_abs_diff']:.6f}",
            check["atol"],
            check["rtol"],
        ]
        for check in checks
    ]
    headers = ["backend", "shape_name", "allclose_to_flaggems", "max_abs_diff", "atol", "rtol"]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def main() -> int:
    args = parse_args()
    load_runtime_deps()
    ensure_cuda()
    seed_everything(args.seed)

    backends = parse_backends(args.backends)
    sorted_topk_ids = args.sorted_topk_ids or ("hpc" in backends)
    block_shape = (args.block_n, args.block_k)

    if block_shape != DEFAULT_BLOCK_SHAPE:
        raise ValueError(
            f"Only block shape {DEFAULT_BLOCK_SHAPE} is supported by the current hpc path."
        )

    runners = load_backend_runners(backends)
    shapes = resolve_shapes(args)

    print(
        f"# mode={args.mode} backends={','.join(runners.keys())} "
        f"block_shape={list(block_shape)} sorted_topk_ids={sorted_topk_ids}"
    )

    all_results: List[dict] = []
    all_checks: List[dict] = []

    for shape_name, config in shapes:
        case = generate_fp8_blockwise_case(
            shape_name=shape_name,
            config=config,
            block_shape=block_shape,
            sorted_topk_ids=sorted_topk_ids,
        )
        if args.check:
            all_checks.extend(
                maybe_check_outputs(
                    case=case,
                    block_shape=block_shape,
                    runners=runners,
                    atol=args.atol,
                    rtol=args.rtol,
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
                    block_shape=block_shape,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    use_cudagraph=args.use_cudagraph,
                )
                if args.mode == "bench"
                else profile_backend(
                    backend=backend,
                    runner=runner,
                    case=case,
                    block_shape=block_shape,
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
            "sorted_topk_ids": sorted_topk_ids,
            "results": all_results,
            "checks": all_checks,
        }
        Path(args.json_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"# wrote json to {args.json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
