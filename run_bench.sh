#!/bin/bash
set -ex

date=$(date +%Y%m%d_%H%M%S)

log_dir=$(pwd)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="${ROOT_DIR}/FlagGems/benchmark"
# 可选：TAG=myrun → ..._cudagraph_myrun.log；未设置则不带后缀
log_file=${log_dir}/${date}_vllm_fp8_w8a8_${tag:+_$tag}.log
cudagraph_log_file=${log_dir}/${date}_vllm_fp8_w8a8_cudagraph${tag:+_$tag}.log
hpc_cudagraph_log_file=${log_dir}/${date}_hpc_fp8_w8a8_cudagraph${tag:+_$tag}.log

# $H pytest --use_cudagraph -x -vv -s test_vllm_perf.py::test_perf_fused_moe_gems_vs_vllm 2>&1 | tee ${date}_vllm.log
# $H pytest -x -vv -s test_vllm_perf.py::test_perf_fused_moe_gems_vs_sonicmoe 2>&1 | tee ${date}_sonicmoe.log

# cd ${bench_dir}
# $H pytest --use_cudagraph -x -vv -s test_fused_moe_perf.py::test_perf_fused_moe_fp8_blockwise_gems_vs_hpc 2>&1 | tee ${log_dir}/${date}_hpc.log
# $H pytest --use_cudagraph -x -vv -s test_fused_moe_perf.py::test_perf_fused_moe_fp8_blockwise_gems_vs_vllm 2>&1 | tee ${log_dir}/${date}_vllm_fp8_w8a8.log

{
  echo "$date"
  echo "FlagGems commit: $(git -C "${ROOT_DIR}/FlagGems" rev-parse HEAD)"
  echo "FlagGems: $(git -C "${ROOT_DIR}/FlagGems" log -1 --oneline)"
  cd "${bench_dir}"
  $H pytest -x -vv -s --use_cudagraph test_fused_moe_perf.py::test_perf_fused_moe_fp8_blockwise_gems_vs_hpc
} 2>&1 | tee "${hpc_cudagraph_log_file}"

{
  echo "$date"
  echo "FlagGems commit: $(git -C "${ROOT_DIR}/FlagGems" rev-parse HEAD)"
  echo "FlagGems: $(git -C "${ROOT_DIR}/FlagGems" log -1 --oneline)"
  cd "${bench_dir}"
  $H pytest -x -vv -s test_vllm_perf.py::test_perf_fused_moe_fp8_blockwise_gems_vs_vllm
} 2>&1 | tee "${log_file}"


{
  echo "$date"
  echo "FlagGems commit: $(git -C "${ROOT_DIR}/FlagGems" rev-parse HEAD)"
  echo "FlagGems: $(git -C "${ROOT_DIR}/FlagGems" log -1 --oneline)"
  cd "${bench_dir}"
  $H pytest -x -vv -s --use_cudagraph test_vllm_perf.py::test_perf_fused_moe_fp8_blockwise_gems_vs_vllm 
} 2>&1 | tee "${cudagraph_log_file}"