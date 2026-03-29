#!/usr/bin/env python3
"""
Parse FlagGems fused_moe benchmark output into Feishu-pasteable Markdown table.

Usage:
    python parse_bench.py bench_output.txt
    cat bench_output.txt | python parse_bench.py
"""

import sys
import re

import pandas as pd
from tabulate import tabulate


def parse_line(line):
    m = re.match(
        r"(SUCCESS|FAILED)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[(.+)\]", line.strip()
    )
    if not m:
        return None

    status, torch_lat, gems_lat, speedup, size_str = m.groups()
    sizes = re.findall(r"torch\.Size\(\[([^\]]+)\]\)", size_str)
    if len(sizes) < 5:
        return None

    tokens, hidden = sizes[0].split(", ")
    E = sizes[1].split(", ")[0]
    topk = sizes[3].split(", ")[1]

    return {
        "Model Config": f"E={E},H={hidden},TopK={topk}",
        "Tokens": int(tokens),
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
