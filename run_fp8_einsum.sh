#!/bin/bash
set -ex

date=$(date +%Y%m%d_%H%M%S)

log_dir=$(pwd)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gems_dir="${ROOT_DIR}/FlagGems"
bench_dir="${gems_dir}/benchmark"

# 可选：TAG=myrun → ..._myrun.log；未设置则不带后缀
acc_log_file=${log_dir}/${date}_fp8_einsum_accuracy${tag:+_$tag}.log
perf_log_file=${log_dir}/${date}_fp8_einsum_perf${tag:+_$tag}.log
perf_cudagraph_log_file=${log_dir}/${date}_fp8_einsum_perf_cudagraph${tag:+_$tag}.log

# Accuracy: FlagGems fp8_einsum vs dequantized PyTorch reference
{
  echo "$date"
  echo "FlagGems commit: $(git -C "${gems_dir}" rev-parse HEAD)"
  echo "FlagGems: $(git -C "${gems_dir}" log -1 --oneline)"
  cd "${gems_dir}"
  $H pytest -x -vv -s tests/test_fp8_einsum.py::test_accuracy_fp8_einsum
} 2>&1 | tee "${acc_log_file}"

# Performance: FlagGems fp8_einsum vs DeepGEMM fp8_einsum
{
  echo "$date"
  echo "FlagGems commit: $(git -C "${gems_dir}" rev-parse HEAD)"
  echo "FlagGems: $(git -C "${gems_dir}" log -1 --oneline)"
  cd "${bench_dir}"
  $H pytest -x -vv -s test_fp8_einsum.py::test_perf_fp8_einsum_gems_vs_deepgemm
} 2>&1 | tee "${perf_log_file}"

# Performance with CUDA Graph timing
{
  echo "$date"
  echo "FlagGems commit: $(git -C "${gems_dir}" rev-parse HEAD)"
  echo "FlagGems: $(git -C "${gems_dir}" log -1 --oneline)"
  cd "${bench_dir}"
  $H pytest -x -vv -s --use_cudagraph test_fp8_einsum.py::test_perf_fp8_einsum_gems_vs_deepgemm
} 2>&1 | tee "${perf_cudagraph_log_file}"
