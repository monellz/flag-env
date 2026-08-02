"""Benchmark the MODEL1 sparse decode path on the REAL DeepSeek-V4-Flash trace
shapes, comparing three implementations and dumping one markdown report:

    cuda   - vLLM CUDA FlashMLA (reference)
    base   - FlagGems Triton, MODEL1 TLE fast path DISABLED (portable kernel)
    tle    - FlagGems Triton, MODEL1 TLE fast path ENABLED

All three are driven through the same public entry points, so `base` and `tle`
differ only by the `HAS_TLE_MODEL1` gate in flag_gems.fused.flash_mla_with_kvcache.

Shapes come from the DeepSeek-V4-Flash traces: every (extra config, batch) pair
that actually appears, with its call count. Trace files record shapes only, so
the dynamic-length arguments (topk_length / extra_topk_length) are not known
per call; `--length` selects what to assume:

    auto     (default) what the FlagGems harness generates by default, i.e. no
             topk_length, and extra_topk_length only when an extra cache is
             present. Matches the older dump_perf_markdown.py runs.
    dynamic  every config also gets a per-request topk_length, matching the
             MODEL1 cases in FlagGems' own benchmark.
    static   no dynamic lengths at all (every padded block is valid).

usage:
    source env.sh && $H python bench_model1_perf_table.py [out.md] [options]

options:
    --length {auto,dynamic,static}  length assumption (default auto)
    --batches N                     sample at most N trace batch values (default: all)
    --rep N                         cudagraph rep per measurement (default 50)
"""

import argparse
import glob
import math
import re
import sys
from collections import defaultdict

FLAGGEMS_DIR = "/home/zhongrx/dev/flag-env-mla/FlagGems"
sys.path.insert(0, FLAGGEMS_DIR)

TRACE_GLOB = "/home/zhongrx/dev/FlagOSTune/shape-config/DeepSeek-V4-Flash-*.txt"
PERF_S_K = 66560  # max serving context over the trace files
EXTRA_PAGES = 88357  # the trace's real extra_k_cache page pool

# (label, extra_topk, extra_page_block_size, extra_num_pages, trace_key)
# no-extra is real in the trace too (extra fields = -1), just a small share.
EXTRA_CONFIGS = [
    ("—", 0, 0, 0, (-1, -1)),
    ("512/64", 512, 64, EXTRA_PAGES, (512, 64)),
    ("8192/2", 8192, 2, EXTRA_PAGES, (8192, 2)),
]


def load_trace_counts():
    """Parse the traces -> {(extra_topk, extra_page, batch): total_count} plus
    the sorted set of batch values that actually occur."""
    counts = defaultdict(int)
    batches = set()
    for fn in glob.glob(TRACE_GLOB):
        for line in open(fn):
            if "flash_mla_with_kvcache.flash_mla_with_kvcache" not in line:
                continue
            m = re.search(r"\[shape info\]:\s*\[([^\]]+)\]", line)
            c = re.search(r"\[count\]:\s*(\d+)", line)
            if not m or not c:
                continue
            f = [int(x) for x in m.group(1).split(",")]
            batch, epage, etopk = f[0], f[8], f[13]
            counts[(etopk, epage, batch)] += int(c.group(1))
            batches.add(batch)
    return counts, sorted(batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outfile", nargs="?", default=None)
    ap.add_argument(
        "--length", choices=["auto", "dynamic", "static"], default="auto"
    )
    ap.add_argument("--batches", type=int, default=0)
    ap.add_argument("--rep", type=int, default=50)
    args = ap.parse_args()

    trace_counts, all_batches = load_trace_counts()
    batches = all_batches
    if args.batches and args.batches < len(all_batches):
        step = len(all_batches) / args.batches
        batches = [all_batches[int(i * step)] for i in range(args.batches)]

    import torch
    import triton
    from benchmark.test_flash_mla_with_kvcache import (
        FlashMLAWithKVCacheBenchmark,
        TestParam,
        _cuda_wrapper,
        _triton_wrapper,
    )

    # flag_gems.fused.__init__ rebinds the attribute to the function, so the
    # module (whose HAS_TLE_MODEL1 gate we flip) must come from sys.modules
    import flag_gems.fused.flash_mla_with_kvcache  # noqa: F401

    fmk = sys.modules["flag_gems.fused.flash_mla_with_kvcache"]
    assert fmk.HAS_TLE_MODEL1, "MODEL1 TLE fast path unavailable in this env"

    # count fast-path entries so the table can assert the gate really fired
    hits = {"n": 0}
    _orig_tle = fmk.sparse_decode_model1_tle

    def counting(*a, **kw):
        hits["n"] += 1
        return _orig_tle(*a, **kw)

    fmk.sparse_decode_model1_tle = counting

    torch.manual_seed(0)
    device_name = torch.cuda.get_device_name(0)

    def build(batch, extra_topk, extra_page, extra_pages):
        kw = dict(
            batch=batch,
            topk=128,
            h_q=64,
            d_qk=512,
            page_block_size=64,
            num_pages=math.ceil(PERF_S_K / 64),
            is_fp8=True,
            have_attn_sink=True,
            have_topk_length=(args.length == "dynamic"),
        )
        if extra_topk > 0:
            kw.update(
                extra_topk=extra_topk,
                extra_page_block_size=extra_page,
                extra_num_pages=extra_pages,
            )
        q, k, bt, cs, dv, kwargs = next(
            iter(FlashMLAWithKVCacheBenchmark.make_input(TestParam(**kw)))
        )
        kwargs.pop("out", None)
        if args.length == "static":
            kwargs.pop("topk_length", None)
            kwargs.pop("extra_topk_length", None)
        return q, k, bt, cs, dv, kwargs

    def cudagraph_ms(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        return triton.testing.do_bench_cudagraph(fn, rep=args.rep, return_mode="median")

    rows = []
    fast_path_ok = True
    for label, extra_topk, extra_page, extra_pages, trace_key in EXTRA_CONFIGS:
        for batch in batches:
            count = trace_counts.get((trace_key[0], trace_key[1], batch), 0)
            q, k, bt, cs, dv, kwargs = build(batch, extra_topk, extra_page, extra_pages)
            kw = {kk: vv for kk, vv in kwargs.items() if kk != "is_fp8_kvcache"}

            t_cuda = cudagraph_ms(
                lambda: _cuda_wrapper(q, k, bt, cs, dv, is_fp8_kvcache=True, **kw)
            )
            fmk.HAS_TLE_MODEL1 = False
            t_base = cudagraph_ms(
                lambda: _triton_wrapper(q, k, bt, cs, dv, is_fp8_kvcache=True, **kw)
            )
            fmk.HAS_TLE_MODEL1 = True
            before = hits["n"]
            t_tle = cudagraph_ms(
                lambda: _triton_wrapper(q, k, bt, cs, dv, is_fp8_kvcache=True, **kw)
            )
            fast_path_ok &= hits["n"] > before

            rows.append((label, batch, t_cuda, t_base, t_tle, count))
            print(
                f"  ...{label} b={batch}: cuda={t_cuda:.4f} base={t_base:.4f} "
                f"tle={t_tle:.4f}",
                file=sys.stderr,
            )
            del q, k, kwargs
            torch.cuda.empty_cache()

    # ---------------- markdown ----------------
    def stats(vals):
        """(avg, min, max) with the batch each extremum came from."""
        avg = sum(v for v, _ in vals) / len(vals)
        lo = min(vals, key=lambda x: x[0])
        hi = max(vals, key=lambda x: x[0])
        return avg, lo, hi

    per_config = defaultdict(lambda: {"cuda": [], "base": [], "count": 0})
    for label, batch, t_cuda, t_base, t_tle, count in rows:
        per_config[label]["cuda"].append((t_cuda / t_tle, batch))
        per_config[label]["base"].append((t_base / t_tle, batch))
        per_config[label]["count"] += count

    length_note = {
        "auto": "harness default — no `topk_length`; `extra_topk_length` "
        "(randomised) only where an extra cache exists, so no-extra runs the "
        "static schedule and the extra configs run the dynamic one",
        "dynamic": "every config gets a randomised per-request `topk_length` "
        "(and `extra_topk_length` where applicable), as in the FlagGems MODEL1 "
        "benchmark cases",
        "static": "no dynamic lengths (every padded block is valid)",
    }[args.length]
    out = []
    out.append(
        f"MODEL1 sparse decode on {device_name} — real DeepSeek-V4-Flash trace "
        f"shapes (h_q=64, topk=128, page=64, s_k={PERF_S_K}); ms, cudagraph median."
    )
    out.append("")
    out.append("- `cuda` — vLLM CUDA FlashMLA (reference)")
    out.append("- `base` — FlagGems Triton before the MODEL1 TLE fast path (portable kernel)")
    out.append("- `tle` — FlagGems Triton with the MODEL1 TLE fast path")
    out.append(f"- `extra` = extra_topk/extra_page_size; `count` = trace call count")
    out.append(f"- lengths: {length_note}")
    out.append("")

    out.append("### Summary")
    out.append("")
    out.append(
        "| extra | tle/cuda avg | tle/cuda min | tle/cuda max | "
        "tle/base avg | tle/base min | tle/base max | trace calls |"
    )
    out.append("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for label, _, _, _, _ in EXTRA_CONFIGS:
        d = per_config[label]
        c_avg, c_lo, c_hi = stats(d["cuda"])
        b_avg, b_lo, b_hi = stats(d["base"])
        out.append(
            f"| {label} | {c_avg:.2f}x | {c_lo[0]:.2f}x (b={c_lo[1]}) | "
            f"{c_hi[0]:.2f}x (b={c_hi[1]}) | {b_avg:.2f}x | "
            f"{b_lo[0]:.2f}x (b={b_lo[1]}) | {b_hi[0]:.2f}x (b={b_hi[1]}) | "
            f"{d['count']} |"
        )
    out.append("")

    out.append("### Per-shape")
    out.append("")
    out.append("| extra | b | cuda | base | tle | tle/cuda | tle/base | count |")
    out.append("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for label, batch, t_cuda, t_base, t_tle, count in rows:
        out.append(
            f"| {label} | {batch} | {t_cuda:.4f} | {t_base:.4f} | **{t_tle:.4f}** | "
            f"{t_cuda / t_tle:.2f}x | {t_base / t_tle:.2f}x | {count} |"
        )

    if not fast_path_ok:
        out.append("")
        out.append("> WARNING: the TLE fast path was not taken for every shape.")

    md = "\n".join(out)
    if args.outfile:
        with open(args.outfile, "w") as f:
            f.write(md + "\n")
        print(f"wrote {args.outfile}", file=sys.stderr)
    print(md)


if __name__ == "__main__":
    main()
