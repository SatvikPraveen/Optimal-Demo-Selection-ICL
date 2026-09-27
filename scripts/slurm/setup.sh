#!/bin/bash
# ============================================================================
# One-time setup on a Slurm cluster's login node.
#
#   bash scripts/slurm/setup.sh
#
# Creates a Python 3.11 venv with uv, installs the package, pre-downloads the
# datasets, the embedding model and the local LM checkpoints into the shared
# HuggingFace cache (compute nodes may not have internet access), and runs
# the smoke benchmark. Gated checkpoints (LLaMA, Gemma) need HF_TOKEN or a
# prior `huggingface-cli login`. If the login node cannot run NumPy 2 (a CPU
# without x86-64-v2) or the GPU driver predates the newest CUDA, run it on a
# compute node with a driver-matched torch, e.g. for a CUDA 12.1 driver:
#
#   srun -p Quick --gres=gpu:1 --time=02:00:00 \
#     env TORCH_CUDA=cu121 TORCH_VERSION=2.5.1 bash scripts/slurm/setup.sh
# ============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/general/satvikpraveen/Optimal-Demo-Selection-ICL}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
export HF_HOME="${HF_HOME:-/general/satvikpraveen/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=0

cd "$PROJECT_DIR"
mkdir -p "$HF_HOME" logs results/raw results/processed

# huggingface_hub reads its token from $HF_HOME/token, so a token stored by
# `huggingface-cli login` in the default location must be mirrored there.
if [ ! -f "$HF_HOME/token" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  install -m 600 "$HOME/.cache/huggingface/token" "$HF_HOME/token"
fi

if [ ! -x venv/bin/python ]; then
  echo ">> creating venv (python $PYTHON_VERSION)"
  uv venv --python "$PYTHON_VERSION" venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
echo ">> installing package"
uv pip install -q -e ".[dev]"

# PyPI's torch wheels track the newest CUDA toolkit, which can be newer than
# the cluster's NVIDIA driver (driver 530 supports CUDA <= 12.1). Install a
# driver-compatible build from the PyTorch index when TORCH_CUDA is set,
# e.g. TORCH_CUDA=cu121 TORCH_VERSION=2.5.1.
if [ -n "${TORCH_CUDA:-}" ]; then
  echo ">> installing torch ${TORCH_VERSION:-} for ${TORCH_CUDA}"
  uv pip install -q "torch${TORCH_VERSION:+==$TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi
python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"

echo ">> pre-downloading datasets and models into $HF_HOME"
python - <<'PY'
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

load_dataset("SetFit/sst5", "default")
load_dataset("tau/commonsense_qa")
for split in ("train", "test"):
    pd.read_parquet(f"hf://datasets/wangrongsheng/ag_news/data/{split}-00000-of-00001.parquet")
SentenceTransformer("all-MiniLM-L6-v2")
for name in os.environ.get("MODELS", "gpt2 meta-llama/Llama-3.2-3B-Instruct google/gemma-2b").split():
    print("  ", name)
    AutoTokenizer.from_pretrained(name)
    AutoModelForCausalLM.from_pretrained(name)
print("downloads complete")
PY

echo ">> smoke benchmark"
python experiments/run_benchmark.py --benchmark smoke --output-dir results/raw/smoke
echo ">> setup complete"
