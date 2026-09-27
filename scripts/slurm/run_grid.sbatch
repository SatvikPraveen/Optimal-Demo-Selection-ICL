#!/bin/bash
# ============================================================================
# Slurm array job: the full benchmark grid, several cells per job.
#
# Clusters often cap the number of concurrently running jobs per user, so
# each array task requests several GPUs and runs one (dataset, model, method)
# cell per GPU in parallel. Task t handles cells [t*G, t*G+G) where G is the
# number of GPUs allocated to the task (SLURM_GPUS_ON_NODE). Every cell runs
# all seeds sequentially with --resume, so a killed task can be resubmitted
# without redoing finished runs.
#
#   python scripts/slurm/grid.py --models gemma-2b        # 24 cells
#   sbatch --array=0-11 --gres=gpu:2 \
#          --export=ALL,MODELS="gemma-2b" scripts/slurm/run_grid.sbatch
#   sbatch --array=0-5 --gres=gpu:4 --exclude=<pascal/turing nodes> ...
#
# Recommended submission on a cluster with a per-user job cap: one cell per
# task under a high-job-count QOS, excluding nodes the torch build cannot use
# (Pascal/Turing GPUs lack bf16; Blackwell GPUs have no cu121 kernels) and
# nodes with known-faulty GPUs:
#
#   sbatch --array=0-23 --gres=gpu:1 --qos=maxjobs50 --partition=Contributors \
#          --exclude=GPU22,GPU3,GPU4,GPU42,GPU6,GPU8,GPU9,GPU49,GPU55,GPU56,GPU57 \
#          --export=ALL,MODELS="gemma-2b" scripts/slurm/run_grid.sbatch
#
# Environment overrides: BENCHMARK (full_benchmark), SEEDS ("0 1 2"),
# MODELS (subset of the benchmark's models), HF_HOME, HF_HUB_OFFLINE (0).
# ============================================================================
#SBATCH --job-name=icl-grid
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/general/satvikpraveen/Optimal-Demo-Selection-ICL}"
BENCHMARK="${BENCHMARK:-full_benchmark}"
SEEDS="${SEEDS:-0 1 2}"
MODELS="${MODELS:-}"
export HF_HOME="${HF_HOME:-/general/satvikpraveen/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"
# shellcheck disable=SC1091
source venv/bin/activate

# GPUs visible to this task, as indices into CUDA_VISIBLE_DEVICES.
NGPU="${SLURM_GPUS_ON_NODE:-1}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
# shellcheck disable=SC2086
NCELLS=$(python scripts/slurm/grid.py --benchmark "$BENCHMARK" ${MODELS:+--models $MODELS} | tail -1 | sed 's/.*--array=0-//')
NCELLS=$((NCELLS + 1))
echo "task $TASK on $(hostname): $NGPU GPU(s) ($(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,)), $NCELLS cells total"

pids=()
for ((g = 0; g < NGPU; g++)); do
  cell=$((TASK * NGPU + g))
  [ "$cell" -ge "$NCELLS" ] && break
  # shellcheck disable=SC2086
  read -r DATASET MODEL METHOD < <(python scripts/slurm/grid.py --benchmark "$BENCHMARK" --index "$cell" ${MODELS:+--models $MODELS})
  echo "  gpu $g -> cell $cell: $DATASET / $MODEL / $METHOD (seeds: $SEEDS)"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$g python experiments/run_benchmark.py \
    --benchmark "$BENCHMARK" \
    --datasets "$DATASET" --models "$MODEL" --methods "$METHOD" \
    --seeds $SEEDS --resume \
    --log-file "logs/${DATASET}__${MODEL}__${METHOD}.log" \
    > "logs/cell_${DATASET}__${MODEL}__${METHOD}.out" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
echo "task $TASK finished with status $status"
exit $status
