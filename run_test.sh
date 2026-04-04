#!/bin/bash
set -ex

gems_dir="${ROOT_DIR}/FlagGems"
echo "FlagGems commit: $(git -C "${gems_dir}" rev-parse HEAD)"
echo "FlagGems: $(git -C "${gems_dir}" log -1 --oneline)"
cd "${gems_dir}"
$H pytest -x -vv -s tests/test_vllm_ops.py::test_accuracy_fused_moe_fp8_blockwise
