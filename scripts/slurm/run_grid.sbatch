#!/bin/bash
# ============================================================================
# Slurm array job: the full benchmark grid on the cluster.
#
# One array task per (dataset, model, method); each task runs every seed of
# that cell sequentially with --resume, so a killed task can be resubmitted
# without redoing finished runs. Submit with:
#
#   sbatch --array=0-47 scripts/slurm/run_grid.sbatch
#   sbatch --array=0-47 --export=ALL,BENCHMARK=full_benchmark,SEEDS="0 1 2" scripts/slurm/run_grid.sbatch
#
# Grid indices are enumerated by scripts/slurm/grid.py (dataset-major), so
# `python scripts/slurm/grid.py` prints the mapping.
# ============================================================================
#SBATCH --job-name=icl-grid
#SBATCH --partition=general
#SBATCH --gres=gpu:A100:1   # override on the command line, e.g. sbatch --gres=gpu:A40:1 ...
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/general/satvikpraveen/Optimal-Demo-Selection-ICL}"
BENCHMARK="${BENCHMARK:-full_benchmark}"
SEEDS="${SEEDS:-0 1 2}"
export HF_HOME="${HF_HOME:-/general/satvikpraveen/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"
# shellcheck disable=SC1091
source venv/bin/activate

read -r DATASET MODEL METHOD < <(python scripts/slurm/grid.py --benchmark "$BENCHMARK" --index "${SLURM_ARRAY_TASK_ID:-0}")
echo "task ${SLURM_ARRAY_TASK_ID:-0}: dataset=$DATASET model=$MODEL method=$METHOD seeds=$SEEDS"
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# shellcheck disable=SC2086
python experiments/run_benchmark.py \
  --benchmark "$BENCHMARK" \
  --datasets "$DATASET" --models "$MODEL" --methods "$METHOD" \
  --seeds $SEEDS \
  --resume \
  --log-file "logs/${DATASET}__${MODEL}__${METHOD}.log"
