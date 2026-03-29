# h20
# uv venv --python 3.12 .venv
# source .venv/bin/activate
# spack load cuda@12.9.1
# uv pip install torch --index-url https://download.pytorch.org/whl/cu129
# uv pip install pytest vllm==0.17.0 # 20260307 更新

spack load cuda@12.9.1
source .venv/bin/activate
export H="srun --gres=gpu:1"
