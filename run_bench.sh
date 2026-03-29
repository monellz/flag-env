#!/bin/bash
set -ex

date=$(date +%Y%m%d_%H%M%S)
echo $date

log_dir=$(pwd)
bench_dir=FlagGems/benchmark

# $H pytest --use_cudagraph -x -vv -s test_vllm_perf.py::test_perf_fused_moe_gems_vs_vllm 2>&1 | tee ${date}_vllm.log
# $H pytest -x -vv -s test_vllm_perf.py::test_perf_fused_moe_gems_vs_sonicmoe 2>&1 | tee ${date}_sonicmoe.log

cd ${bench_dir}
$H pytest --use_cudagraph -x -vv -s test_fused_moe_perf.py::test_perf_fused_moe_fp8_blockwise_gems_vs_hpc 2>&1 | tee ${log_dir}/${date}_hpc.log
$H pytest --use_cudagraph -x -vv -s test_fused_moe_perf.py::test_perf_fused_moe_fp8_blockwise_gems_vs_vllm 2>&1 | tee ${log_dir}/${date}_vllm_fp8_w8a8.log
