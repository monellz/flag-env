#!/bin/bash
set -ex
cd hpc-ops
make wheel -j4
uv pip install --no-deps dist/*.whl
