#!/usr/bin/env python3
"""
Standalone profiler/benchmark driver for fused_marlin_moe (W4A16 INT4 MoE).

Targets:
  - FlagGems fused_marlin_moe (Triton wna16, group_size=128 GPTQ uint4b8)
  - vLLM fused_marlin_moe (CUDA Marlin, group_size=128 GPTQ uint4b8)

Both backends consume the SAME FP source weights, quantized into their
respective layouts (FlagGems wna16: packed uint8 [E,N,K/2]; vLLM Marlin:
int32 block-permuted [E, ...]) using vLLM's quantize_weights / marlin_quantize
so the numerics are matched.

Examples:
  python profile_fused_marlin_moe.py --shape-preset mixtral
  python profile_fused_marlin_moe.py --shape-preset deepseek_v3_tp8 --use-cudagraph
  python profile_fused_marlin_moe.py --tokens 128 --experts 8 --hidden 4096 --intermediate 14336 --topk 2
  nsys profile -o fg_marlin_moe --capture-range=nvtx --capture-range-end=stop \\
      python profile_fused_marlin_moe.py --mode profile --backends flaggems
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import gc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
FLAGGEMS_ROOT = ROOT / "FlagGems"


DEFAULT_GROUP_SIZE = 128
DEFAULT_SEED = 20260521

torch = None
triton = None
flag_gems = None
gems_fused_marlin_moe = None
GEMS_QUANT_TYPE_UINT4B8 = None
vllm_fused_marlin_moe = None
vllm_quantize_weights = None
vllm_marlin_quantize = None
VLLM_QUANT_TYPE_UINT4B8 = None  # vllm.scalar_types.uint4b8
my_fused_moe_w4a16 = None      # flag_gems.fused.fused_moe_w4a16.fused_moe_w4a16_gptq


# Standard token-count sweep for all production presets.
_TOKEN_SWEEP = (1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)

# (num_tokens, num_experts, hidden_size, intermediate_size, topk)
MOE_SHAPES: Dict[str, List[Tuple[int, int, int, int, int]]] = {
    # Mirrors FlagGems/tests/test_fused_marlin_moe.py QUICK_CONFIGS for sanity.
    "smoke": [
        (1, 8, 128, 256, 2),
        (4, 8, 128, 256, 2),
        (16, 8, 256, 512, 2),
        (32, 8, 128, 256, 4),
    ],
    # E=8, hidden=4096, intermediate=14336, topk=2
    "mixtral":          [(M, 8,   4096, 14336, 2)  for M in _TOKEN_SWEEP],
    # E=256, hidden=7168, intermediate=2048, topk=8 (DeepSeek-V3 TP=8 shard)
    "deepseek_v3_tp8":  [(M, 256, 7168, 2048,  8)  for M in _TOKEN_SWEEP],
    # E=512, hidden=4096, intermediate=1024, topk=10
    "qwen3_5_397b_a17b": [(M, 512, 4096, 1024, 10) for M in _TOKEN_SWEEP],
}


@dataclass
class MarlinMoECase:
    shape_name: str
    num_tokens: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    topk: int
    dtype: "torch.dtype"

    hidden_states: "torch.Tensor"     # (M, K)              fp16/bf16
    topk_weights: "torch.Tensor"      # (M, topk)           fp32
    topk_ids: "torch.Tensor"          # (M, topk)           int32

    # Dequantized FP weights for the torch SwiGLU reference (same numerics as
    # what the kernel's wna16-format weights represent, modulo quant rounding).
    # Heavy: skipped when not needed (no --check) to avoid OOM at large E/K.
    w1_ref: "Optional[torch.Tensor]"   # (E, 2*intermediate, hidden) fp16/bf16
    w2_ref: "Optional[torch.Tensor]"   # (E, hidden, intermediate)   fp16/bf16

    # FlagGems wna16 layout (per-expert GPTQ uint4b8 packed two-per-byte)
    w1_q_wna16: "torch.Tensor"        # (E, 2N, K/2)        uint8
    w2_q_wna16: "torch.Tensor"        # (E, K, N/2)         uint8
    w1_scale_wna16: "torch.Tensor"    # (E, 2N, K/gs)       fp16/bf16
    w2_scale_wna16: "torch.Tensor"    # (E, K, N/gs)        fp16/bf16

    # vLLM Marlin layout (CUDA-kernel-specific block-permuted int32 weights)
    w1_q_marlin: "torch.Tensor"
    w2_q_marlin: "torch.Tensor"
    w1_scale_marlin: "torch.Tensor"
    w2_scale_marlin: "torch.Tensor"


def load_runtime_deps(want_vllm: bool) -> None:
    global torch, triton, flag_gems, gems_fused_marlin_moe, GEMS_QUANT_TYPE_UINT4B8
    global vllm_fused_marlin_moe, vllm_quantize_weights, vllm_marlin_quantize
    global VLLM_QUANT_TYPE_UINT4B8
    global my_fused_moe_w4a16

    if torch is not None:
        return

    for path in (FLAGGEMS_ROOT / "src", FLAGGEMS_ROOT):
        sys.path.insert(0, str(path))

    import torch as torch_mod
    import triton as triton_mod
    import flag_gems as flag_gems_mod
    from flag_gems.fused.fused_marlin_moe import (
        fused_marlin_moe as _gems_fmm,
        QUANT_TYPE_UINT4B8 as _gems_qt,
    )
    from flag_gems.fused.fused_marlin_moe import (
        fused_moe_w4a16_gptq as _my_fmm,
    )

    torch = torch_mod
    triton = triton_mod
    flag_gems = flag_gems_mod
    gems_fused_marlin_moe = _gems_fmm
    GEMS_QUANT_TYPE_UINT4B8 = _gems_qt
    my_fused_moe_w4a16 = _my_fmm

    if want_vllm:
        # vllm helpers are only loaded if the user asked for the vllm backend.
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            fused_marlin_moe as _vllm_fmm,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
            marlin_quantize as _marlin_quantize,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            quantize_weights as _quantize_weights,
        )
        from vllm.scalar_type import scalar_types

        vllm_fused_marlin_moe = _vllm_fmm
        vllm_quantize_weights = _quantize_weights
        vllm_marlin_quantize = _marlin_quantize
        VLLM_QUANT_TYPE_UINT4B8 = scalar_types.uint4b8


def get_tabulate() -> Callable:
    from tabulate import tabulate
    return tabulate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark/profile FlagGems & vLLM fused_marlin_moe (W4A16 INT4 MoE)."
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
        help="comma-separated: flaggems,vllm or all",
    )
    parser.add_argument(
        "--shape-preset",
        default="qwen3_5_397b_a17b",
        choices=tuple(MOE_SHAPES.keys()) + ("all",),
        help="Preset shape family to run when custom shape is not provided",
    )
    parser.add_argument("--tokens", type=int, default=None, help="Custom num_tokens")
    parser.add_argument("--experts", type=int, default=None, help="Custom num_experts")
    parser.add_argument("--hidden", type=int, default=None, help="Custom hidden_size")
    parser.add_argument("--intermediate", type=int, default=None, help="Custom intermediate_size")
    parser.add_argument("--topk", type=int, default=None, help="Custom topk")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16"),
        help="Activation/scale dtype (kernel accumulates in fp32 regardless)",
    )
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
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
        help="Validate FlagGems against vLLM (when both are loaded)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-1,
        help="Relative tolerance for --check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=5e-2,
        help="Absolute tolerance floor (atol = max(this, ref.max()*1e-3))",
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
        return ["flaggems", "vllm", "my"]
    backends = [item.strip().lower() for item in raw.split(",") if item.strip()]
    valid = {"flaggems", "vllm", "my"}
    invalid = sorted(set(backends) - valid)
    if invalid:
        raise ValueError(f"Unsupported backends: {', '.join(invalid)}")
    return backends


def resolve_shapes(args: argparse.Namespace) -> List[Tuple[str, Tuple[int, int, int, int, int]]]:
    custom_fields = [args.tokens, args.experts, args.hidden, args.intermediate, args.topk]
    if any(v is not None for v in custom_fields):
        if not all(v is not None for v in custom_fields):
            raise ValueError(
                "Custom shape requires --tokens --experts --hidden --intermediate --topk together."
            )
        return [("custom", tuple(custom_fields))]

    if args.shape_preset == "all":
        resolved: List[Tuple[str, Tuple[int, int, int, int, int]]] = []
        for preset_name, shapes in MOE_SHAPES.items():
            for idx, shape in enumerate(shapes):
                resolved.append((f"{preset_name}_{idx}", shape))
        return resolved

    return [
        (f"{args.shape_preset}_{idx}", shape)
        for idx, shape in enumerate(MOE_SHAPES[args.shape_preset])
    ]


# ---------- Quantization helpers (port from FlagGems/benchmark/test_fused_marlin_moe.py) ----------


def _wna16_quantize_per_expert(
    w_fp: "torch.Tensor", group_size: int, want_ref: bool = True
) -> Tuple["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
    """FlagGems wna16 layout: (E, out_dim, in_dim) -> packed uint8 (E, out_dim, in_dim/2) + scales [+ ref]."""
    E, out_dim, in_dim = w_fp.shape
    assert in_dim % group_size == 0
    w_q = torch.empty(E, out_dim, in_dim // 2, device=w_fp.device, dtype=torch.uint8)
    w_ref = torch.empty_like(w_fp) if want_ref else None
    scales = torch.empty(
        E, out_dim, in_dim // group_size, device=w_fp.device, dtype=w_fp.dtype
    )
    for e in range(E):
        # quantize_weights expects (in_dim, out_dim) — that's the GPTQ convention.
        # Returns (w_ref, w_q_unsigned, scales, zp); w_ref is the dequantized FP
        # tensor — the ground truth that any GPTQ kernel should reproduce.
        ref_e, q_e, sc_e, _ = vllm_quantize_weights(
            w_fp[e].T, VLLM_QUANT_TYPE_UINT4B8, group_size, False, False
        )
        q_e = q_e.T.contiguous().to(torch.uint8)
        sc_e = sc_e.T
        # low nibble = even k, high nibble = odd k
        w_q[e] = q_e[:, 1::2] * 16 + q_e[:, ::2]
        scales[e] = sc_e
        if want_ref:
            w_ref[e] = ref_e.T.contiguous().to(w_fp.dtype)
    return w_q, scales, w_ref


def _marlin_quantize_per_expert(
    w_fp: "torch.Tensor", group_size: int
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """vLLM Marlin layout: per-expert marlin_quantize -> stacked int32 qweight + scales."""
    qweight_l, scales_l = [], []
    E = w_fp.shape[0]
    for e in range(E):
        # marlin_quantize expects (in_dim, out_dim)
        _, qw, sc, _, _, _ = vllm_marlin_quantize(
            w_fp[e].T.contiguous(), VLLM_QUANT_TYPE_UINT4B8, group_size, act_order=False
        )
        qweight_l.append(qw)
        scales_l.append(sc)
    qweight = torch.stack(qweight_l, dim=0).contiguous()
    scales = torch.stack(scales_l, dim=0).contiguous()
    return qweight, scales


def generate_case(
    shape_name: str,
    config: Tuple[int, int, int, int, int],
    dtype: "torch.dtype",
    group_size: int,
    want_vllm_layout: bool,
    want_ref: bool = True,
) -> MarlinMoECase:
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = torch.device("cuda")

    # Match vLLM's test_fused_marlin_moe input distribution:
    # tests/kernels/moe/test_moe.py::test_fused_marlin_moe scales A, w1, w2 all by 1/10
    # so output magnitudes stay small enough for the fixed atol=4e-2 check.
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) / 10.0

    w1_fp = (
        torch.randn(
            num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
        )
        / 10.0
    )
    w2_fp = (
        torch.randn(
            num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
        )
        / 10.0
    )

    # FlagGems wna16 layout — always built (used by either flaggems or as cross-check)
    w1_q_wna16, w1_scale_wna16, w1_ref = _wna16_quantize_per_expert(w1_fp, group_size, want_ref)
    w2_q_wna16, w2_scale_wna16, w2_ref = _wna16_quantize_per_expert(w2_fp, group_size, want_ref)

    # vLLM Marlin layout — heavy CPU work, skip when not needed
    if want_vllm_layout:
        w1_q_marlin, w1_scale_marlin = _marlin_quantize_per_expert(w1_fp, group_size)
        w2_q_marlin, w2_scale_marlin = _marlin_quantize_per_expert(w2_fp, group_size)
    else:
        empty = torch.empty(0, device=device)
        w1_q_marlin = w2_q_marlin = w1_scale_marlin = w2_scale_marlin = empty

    del w1_fp, w2_fp
    torch.cuda.empty_cache()

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    return MarlinMoECase(
        shape_name=shape_name,
        num_tokens=num_tokens,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        topk=topk,
        dtype=dtype,
        hidden_states=hidden_states.contiguous(),
        topk_weights=topk_weights.contiguous(),
        topk_ids=topk_ids.contiguous(),
        w1_q_wna16=w1_q_wna16.contiguous(),
        w2_q_wna16=w2_q_wna16.contiguous(),
        w1_scale_wna16=w1_scale_wna16.contiguous(),
        w2_scale_wna16=w2_scale_wna16.contiguous(),
        w1_q_marlin=w1_q_marlin,
        w2_q_marlin=w2_q_marlin,
        w1_scale_marlin=w1_scale_marlin,
        w2_scale_marlin=w2_scale_marlin,
        w1_ref=w1_ref.contiguous() if w1_ref is not None else None,
        w2_ref=w2_ref.contiguous() if w2_ref is not None else None,
    )


# ---------- Backends ----------


def load_flaggems_backend(group_size: int) -> Callable[[MarlinMoECase], "torch.Tensor"]:
    def _run(case: MarlinMoECase) -> "torch.Tensor":
        return gems_fused_marlin_moe(
            hidden_states=case.hidden_states,
            w1=case.w1_q_wna16,
            w2=case.w2_q_wna16,
            bias1=None,
            bias2=None,
            w1_scale=case.w1_scale_wna16,
            w2_scale=case.w2_scale_wna16,
            topk_weights=case.topk_weights,
            topk_ids=case.topk_ids,
            quant_type_id=GEMS_QUANT_TYPE_UINT4B8,
            group_size=group_size,
        )

    return _run


def load_vllm_backend() -> Callable[[MarlinMoECase], "torch.Tensor"]:
    def _run(case: MarlinMoECase) -> "torch.Tensor":
        return vllm_fused_marlin_moe(
            hidden_states=case.hidden_states,
            w1=case.w1_q_marlin,
            w2=case.w2_q_marlin,
            bias1=None,
            bias2=None,
            w1_scale=case.w1_scale_marlin,
            w2_scale=case.w2_scale_marlin,
            topk_weights=case.topk_weights,
            topk_ids=case.topk_ids,
            quant_type_id=VLLM_QUANT_TYPE_UINT4B8.id,
        )

    return _run


def load_my_backend(group_size: int) -> Callable[[MarlinMoECase], "torch.Tensor"]:
    """My (tile-B + nibble interleaved + magic-trick SIMD dequant) fused MoE W4A16."""
    def _run(case: MarlinMoECase) -> "torch.Tensor":
        # topk_weights from the fixture is fp32 for vLLM compatibility; my impl
        # accepts any FP dtype (kernel just `tl.load`s and mul's against fp32 acc).
        return my_fused_moe_w4a16(
            hidden_states=case.hidden_states,
            w1=case.w1_q_wna16,
            w2=case.w2_q_wna16,
            w1_scale=case.w1_scale_wna16,
            w2_scale=case.w2_scale_wna16,
            topk_weights=case.topk_weights,
            topk_ids=case.topk_ids,
            group_size=group_size,
        )

    return _run


def load_backend_runners(
    backends: Sequence[str], group_size: int
) -> Dict[str, Callable[[MarlinMoECase], "torch.Tensor"]]:
    runners: Dict[str, Callable[[MarlinMoECase], "torch.Tensor"]] = {}
    for backend in backends:
        try:
            if backend == "flaggems":
                runners[backend] = load_flaggems_backend(group_size)
            elif backend == "vllm":
                runners[backend] = load_vllm_backend()
            elif backend == "my":
                runners[backend] = load_my_backend(group_size)
        except Exception as exc:
            print(f"[skip] backend={backend} unavailable: {exc}", file=sys.stderr)
    if not runners:
        raise RuntimeError("No requested backends are available.")
    return runners


def run_once(
    backend: str,
    runner: Callable[[MarlinMoECase], "torch.Tensor"],
    case: MarlinMoECase,
) -> "torch.Tensor":
    return runner(case)


# ---------- Counters ----------


def case_flops(case: MarlinMoECase) -> float:
    # GEMM1: gate+up = 2 * (M*topk) * (2N) * K = 4*M*topk*N*K MACs -> 8*M*topk*N*K flops
    # GEMM2:        = 2 * (M*topk) * N * K = 2*M*topk*N*K MACs    -> 4*M*topk*N*K flops
    M = case.num_tokens
    K = case.hidden_size
    N = case.intermediate_size
    topk = case.topk
    return 2.0 * M * topk * (2 * N * K) + 2.0 * M * topk * (N * K)


def case_bytes(case: MarlinMoECase) -> int:
    """Approximate HBM traffic.

    MoE only touches weights for experts that receive at least one routed token,
    so weight bytes scale with the actual number of unique routed experts (read
    directly from ``topk_ids``), not the total ``E``. The whole expert weight
    tensor is loaded once per visited expert (assumes large enough per-expert
    tile to load weights only once across all its tokens — true here since
    block_m ≥ avg tokens-per-expert and the kernel iterates K once per N-block).
    """
    elem_act = case.hidden_states.element_size()      # 2 (bf16/fp16)
    elem_scale = case.w1_scale_wna16.element_size()   # 2
    K = case.hidden_size
    N = case.intermediate_size
    M = case.num_tokens
    topk = case.topk

    # Exact: count unique routed experts in this batch.
    active_experts = int(case.topk_ids.unique().numel())

    # Per-expert footprint (kernel-visible bytes).
    w1_per = (2 * N) * K // 2                                            # uint8 packed
    w2_per = K * N // 2
    w1_sc_per = int(case.w1_scale_wna16[0].numel()) * elem_scale         # (2N, K/gs)
    w2_sc_per = int(case.w2_scale_wna16[0].numel()) * elem_scale         # (K,  N/gs)

    w_bytes = active_experts * (w1_per + w2_per + w1_sc_per + w2_sc_per)
    hs_bytes = M * K * elem_act
    out_bytes = M * K * elem_act
    tk_bytes = M * topk * (4 + 4)  # fp32 weights + int32 ids
    return int(w_bytes + hs_bytes + out_bytes + tk_bytes)


# ---------- Bench / profile ----------


def bench_backend(
    backend: str,
    runner: Callable[[MarlinMoECase], "torch.Tensor"],
    case: MarlinMoECase,
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

    sec = max(median_ms * 1e-3, 1e-12)
    return {
        "backend": backend,
        "shape_name": case.shape_name,
        "shape": [
            case.num_tokens, case.num_experts, case.hidden_size,
            case.intermediate_size, case.topk,
        ],
        "dtype": str(case.dtype).replace("torch.", ""),
        "median_ms": float(median_ms),
        "tflops": float(case_flops(case) / sec / 1e12),
        "gbps": float(case_bytes(case) / sec / 1e9),
        "warmup": warmup,
        "repeat": repeat,
        "use_cudagraph": use_cudagraph,
    }


def profile_backend(
    backend: str,
    runner: Callable[[MarlinMoECase], "torch.Tensor"],
    case: MarlinMoECase,
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
            with nvtx_range(f"fused_marlin_moe::{backend}::{case.shape_name}"):
                run_once(backend, runner, case)
            synchronize()
            end = time.perf_counter_ns()
            latencies_ms.append((end - start) / 1e6)
    finally:
        torch.cuda.cudart().cudaProfilerStop()

    median_ms = statistics.median(latencies_ms)
    sec = max(median_ms * 1e-3, 1e-12)
    return {
        "backend": backend,
        "shape_name": case.shape_name,
        "shape": [
            case.num_tokens, case.num_experts, case.hidden_size,
            case.intermediate_size, case.topk,
        ],
        "dtype": str(case.dtype).replace("torch.", ""),
        "profile_repeat": repeat,
        "warmup": warmup,
        "median_ms": float(median_ms),
        "mean_ms": float(statistics.fmean(latencies_ms)),
        "tflops": float(case_flops(case) / sec / 1e12),
        "gbps": float(case_bytes(case) / sec / 1e9),
    }


# ---------- Correctness ----------


def _torch_swiglu_moe_reference(case: MarlinMoECase) -> "torch.Tensor":
    """Obviously-correct dequantized SwiGLU MoE, computed in fp32.

    Ported from FlagGems/tests/test_fused_marlin_moe.py::_reference_swiglu_moe,
    but accumulating in fp32. The kernels accumulate their GEMMs in fp32, so a
    bf16 reference (per-token bf16 matmuls + bf16 intermediates) is actually
    *less* accurate than the kernel under test — at large K (e.g. deepseek
    K=7168) the bf16 reference alone carries ~0.047 error, pushing every backend
    (flaggems/vllm/my produce bit-identical output) past the fixed atol=4e-2.
    Computing the reference in fp32 makes it a true ground truth: the kernels'
    bf16 round-off (~0.017 here) then sits comfortably inside atol.
    """
    hs = case.hidden_states.float()
    w1_ref = case.w1_ref.float()
    w2_ref = case.w2_ref.float()
    tw = case.topk_weights.float()
    ti = case.topk_ids
    M, _ = hs.shape
    _, two_N, _ = w1_ref.shape
    N = two_N // 2
    out = torch.zeros_like(hs)
    for m in range(M):
        for k in range(ti.size(1)):
            e = int(ti[m, k].item())
            w_topk = tw[m, k]
            x = hs[m]
            gate_up = w1_ref[e] @ x
            gate, up = gate_up[:N], gate_up[N:]
            act = torch.nn.functional.silu(gate) * up
            y = w2_ref[e] @ act
            out[m] = out[m] + w_topk * y
    # Kept in fp32: maybe_check_outputs compares via .float(), so returning the
    # un-rounded fp32 ground truth avoids re-introducing bf16 round-off.
    return out


def maybe_check_outputs(
    case: MarlinMoECase,
    runners: Dict[str, Callable[[MarlinMoECase], "torch.Tensor"]],
    atol_floor: float,  # unused; kept for arg-compat
    rtol: float,        # unused; kept for arg-compat
) -> List[dict]:
    """Check every backend against the dequantized torch SwiGLU MoE reference.

    Tolerance follows vLLM's test_fused_marlin_moe (atol=4e-2, rtol=0):
        vllm/tests/kernels/moe/test_moe.py::test_fused_marlin_moe (line ~1007)
    vLLM scales A and w by 1/10 to keep output magnitudes small; we match that
    in generate_case (hidden_states / 10.0) so the same fixed atol applies.
    """
    VLLM_ATOL = 4e-2
    VLLM_RTOL = 0.0

    reference = _torch_swiglu_moe_reference(case)
    synchronize()

    checks: List[dict] = []
    for backend, runner in runners.items():
        output = run_once(f"{backend}_check", runner, case).clone()
        synchronize()
        diff = (output.float() - reference.float()).abs()
        max_abs = diff.max().item()
        ok = torch.allclose(output.float(), reference.float(),
                            atol=VLLM_ATOL, rtol=VLLM_RTOL)
        checks.append({
            "backend": backend,
            "ref": "torch",
            "shape_name": case.shape_name,
            "passed": bool(ok),
            "max_abs_diff": float(max_abs),
            "atol": float(VLLM_ATOL),
            "rtol": float(VLLM_RTOL),
        })
    return checks


# ---------- Reporting ----------


def _shape_text(shape: Sequence[int]) -> str:
    return f"M={shape[0]} E={shape[1]} K={shape[2]} N={shape[3]} topk={shape[4]}"


def print_results_table(results: Sequence[dict], mode: str) -> None:
    if not results:
        return
    tabulate = get_tabulate()

    if mode == "bench":
        rows = [
            [
                r["backend"], r["shape_name"], _shape_text(r["shape"]), r["dtype"],
                f"{r['median_ms']:.3f}", f"{r['tflops']:.1f}", f"{r['gbps']:.0f}",
                r["warmup"], r["repeat"], r["use_cudagraph"],
            ]
            for r in results
        ]
        headers = [
            "backend", "shape_name", "shape", "dtype",
            "median_ms", "TFLOPS", "GB/s", "warmup", "repeat", "cudagraph",
        ]
    else:
        rows = [
            [
                r["backend"], r["shape_name"], _shape_text(r["shape"]), r["dtype"],
                f"{r['median_ms']:.3f}", f"{r['mean_ms']:.3f}",
                f"{r['tflops']:.1f}", f"{r['gbps']:.0f}",
                r["warmup"], r["profile_repeat"],
            ]
            for r in results
        ]
        headers = [
            "backend", "shape_name", "shape", "dtype",
            "median_ms", "mean_ms", "TFLOPS", "GB/s", "warmup", "profile_repeat",
        ]

    print(tabulate(rows, headers=headers, tablefmt="github"))


def print_checks_table(checks: Sequence[dict]) -> None:
    if not checks:
        return
    tabulate = get_tabulate()
    rows = [
        [
            c["backend"], c["ref"], c["shape_name"], c["passed"],
            f"{c['max_abs_diff']:.6f}", f"{c['atol']:.4f}", f"{c['rtol']:.4f}",
        ]
        for c in checks
    ]
    headers = ["backend", "ref", "shape_name", "passed", "max_abs_diff", "atol", "rtol"]
    print(tabulate(rows, headers=headers, tablefmt="github"))


# ---------- Driver ----------


def main() -> int:
    args = parse_args()
    backends = parse_backends(args.backends)
    load_runtime_deps(want_vllm="vllm" in backends or args.check)
    ensure_cuda()
    seed_everything(args.seed)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    runners = load_backend_runners(backends, args.group_size)
    shapes = resolve_shapes(args)

    want_vllm_layout = "vllm" in runners or args.check

    print(
        f"# mode={args.mode} backends={','.join(runners.keys())} dtype={args.dtype} "
        f"group_size={args.group_size}"
    )

    all_results: List[dict] = []
    all_checks: List[dict] = []

    for shape_name, config in shapes:
        case = generate_case(
            shape_name, config, dtype, args.group_size,
            want_vllm_layout, want_ref=args.check,
        )
        if args.check:
            all_checks.extend(
                maybe_check_outputs(case, runners, args.atol, args.rtol)
            )
        for backend, runner in runners.items():
            if args.empty_cache:
                torch.cuda.empty_cache()
            result = (
                bench_backend(
                    backend=backend, runner=runner, case=case,
                    warmup=args.warmup, repeat=args.repeat,
                    use_cudagraph=args.use_cudagraph,
                )
                if args.mode == "bench"
                else profile_backend(
                    backend=backend, runner=runner, case=case,
                    warmup=args.warmup, repeat=args.profile_repeat,
                )
            )
            all_results.append(result)
        # End-of-case cleanup: drop case (weights/scales/etc.) before allocating next one.
        # Heavy tensors held in `case` are GC'd here; gc.collect() + empty_cache() then
        # actually returns the memory to the CUDA allocator.
        del case
        gc.collect()
        if args.empty_cache:
            torch.cuda.empty_cache()

    print_results_table(all_results, args.mode)
    if all_checks:
        print_checks_table(all_checks)

    if args.json_path:
        output = {
            "mode": args.mode,
            "dtype": args.dtype,
            "group_size": args.group_size,
            "results": all_results,
            "checks": all_checks,
        }
        Path(args.json_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"# wrote json to {args.json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
