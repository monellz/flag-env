#!/usr/bin/env bash
set -euo pipefail

echo "=== CPU ==="
lscpu | awk -F: '
/Architecture:/ {print $1 ":" sprintf("%*s", 35 - length($1), "") $2}
/^CPU\(s\):/ {print $1 ":" sprintf("%*s", 35 - length($1), "") $2}
/On-line CPU\(s\) list:/ {print $1 ":" sprintf("%*s", 35 - length($1), "") $2}
/Model name:/ {print $1 ":" sprintf("%*s", 35 - length($1), "") $2}
/Thread\(s\) per core:/ {print $1 ":" sprintf("%*s", 35 - length($1), "") $2}
/Core\(s\) per socket:/ {print $1 ":" sprintf("%*s", 35 - length($1), "") $2}
/Socket\(s\):/ {print $1 ":" sprintf("%*s", 35 - length($1), "") $2}
/NUMA node0 CPU\(s\):/ {print $1 ":" sprintf("%*s", 35 - length($1), "") $2}
/NUMA node1 CPU\(s\):/ {print $1 ":" sprintf("%*s", 35 - length($1), "") $2}
'

echo "=== Memory ==="
free -h | sed -n '1,2p'

echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits \
    | head -n 1 \
    | awk -F', ' '{printf "%s, %s MiB, %s\n", $1, $2, $3}'
else
  echo "nvidia-smi not found"
fi

echo "=== CUDA ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | sed -n '/Driver Version/ {p; q;}'
else
  echo "nvidia-smi not found"
fi

echo "=== OS ==="
if [[ -f /etc/os-release ]]; then
  grep -E '^(NAME|VERSION)=' /etc/os-release
else
  uname -o
fi

echo "=== Kernel ==="
uname -r

echo "=== GLIBC ==="
ldd --version | head -n 1
