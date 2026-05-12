#!/usr/bin/env bash
set -euo pipefail

script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir/DeepGEMM"

rm -rf build dist *.egg-info

# --no-build-isolation: reuse torch from the active uv venv (setup.py imports torch at top level)
# --reinstall: matches the original install.sh's --force-reinstall behavior
uv pip install --no-build-isolation --reinstall .
