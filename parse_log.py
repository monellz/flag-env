#!/usr/bin/env python3
"""
Parse FlagGems fused_moe benchmark output into Feishu-pasteable Markdown table.

Usage:
    python parse_bench.py bench_output.txt
    cat bench_output.txt | python parse_bench.py
"""

import re
import sys

import pandas as pd
from tabulate import tabulate


def parse_line(line):
    m = re.match(
        r"(SUCCESS|FAILED)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[(.+)\]", line.strip()
    )
    if not m:
        return None

    status, torch_lat, gems_lat, speedup, size_str = m.groups()
    sizes = [
        tuple(int(part.strip()) for part in size.split(","))
        for size in re.findall(r"torch\.Size\(\[([^\]]+)\]\)", size_str)
    ]
    if len(sizes) < 5:
        return None

    tokens, hidden = sizes[0]
    num_experts, intermediate_x2, hidden_from_w1 = sizes[1]
    _, hidden_from_w2, intermediate = sizes[2]
    if hidden != hidden_from_w1 or hidden != hidden_from_w2:
        return None

    topk_shape = sizes[-1]
    if len(topk_shape) != 2:
        return None
    _, topk = topk_shape

    return {
        "Model Config": f"E={num_experts},H={hidden},I={intermediate},TopK={topk}",
        "Tokens": tokens,
        "Torch Latency (ms)": float(torch_lat),
        "Gems Latency (ms)": float(gems_lat),
        "Speedup": float(speedup),
    }


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    rows = [r for line in lines if (r := parse_line(line))]

    if not rows:
        print("No benchmark data found.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    print(tabulate(df, headers="keys", tablefmt="pipe", showindex=False))


if __name__ == "__main__":
    main()
