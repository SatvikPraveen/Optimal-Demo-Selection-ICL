#!/bin/bash
# ============================================================================
# Delete what is finished and recorded, keep everything still needed.
#
#   bash scripts/slurm/prune_finished.sh            # prune
#   bash scripts/slurm/prune_finished.sh --dry-run  # only report
#
# A cell (dataset, model, method) is "done" when all SEEDS have a result JSON
# in results/raw. For done cells the per-cell logs are removed. When every
# cell of a model is done, that model's weights are removed from the caches
# (they are the only large artefacts). Result JSONs are never touched.
# ============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/general/satvikpraveen/Optimal-Demo-Selection-ICL}"
SEEDS="${SEEDS:-0 1 2}"
DATASETS="${DATASETS:-sst5 agnews csqa}"
METHODS="${METHODS:-random topk bm25 topk_cone ids rdes se2 influence}"
MODELS="${MODELS:-gemma-2b llama-3.2-3b qwen2.5-7b}"
declare -A MODEL_CACHE=(
  [gemma-2b]="/general/satvikpraveen/hf_cache/hub/models--google--gemma-2b"
  [llama-3.2-3b]="/general/satvikpraveen/hf_cache/hub/models--meta-llama--Llama-3.2-3B-Instruct"
  [qwen2.5-7b]="$HOME/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct"
)
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
run() { if [ "$DRY" = 1 ]; then echo "  would: $*"; else "$@"; fi; }

cd "$PROJECT_DIR"
nseeds=$(wc -w <<<"$SEEDS")
total=0; done_cells=0
for m in $MODELS; do
  model_done=1
  for d in $DATASETS; do for me in $METHODS; do
    total=$((total + 1))
    have=0
    for s in $SEEDS; do [ -f "results/raw/${d}__${m}__${me}__seed${s}.json" ] && have=$((have + 1)); done
    if [ "$have" -eq "$nseeds" ]; then
      done_cells=$((done_cells + 1))
      for f in "logs/cell_${d}__${m}__${me}.out" "logs/${d}__${m}__${me}.log"; do
        [ -f "$f" ] && run rm -f "$f"
      done
    else
      model_done=0
    fi
  done; done
  if [ "$model_done" = 1 ] && [ -d "${MODEL_CACHE[$m]}" ]; then
    echo "model $m complete: removing weights ${MODEL_CACHE[$m]}"
    run rm -rf "${MODEL_CACHE[$m]}"
  fi
done

# Slurm task logs whose task has finished (no longer in the queue).
# %A = array master job id, %K = array index (matches the %A_%a output name).
active=$(squeue -u "$USER" -h -o "%A_%K" 2>/dev/null | tr '\n' ' ')
for f in logs/icl-grid_*_*.out logs/icl-grid_*_*.err; do
  [ -f "$f" ] || continue
  id=$(basename "$f" | sed -E 's/^icl-grid_([0-9]+)_([0-9]+)\.(out|err)$/\1_\2/')
  case " $active " in *" $id "*) ;; *) run rm -f "$f" ;; esac
done

echo "cells done: $done_cells / $total  (results: $(ls results/raw/*.json 2>/dev/null | wc -l))"
echo "logs left: $(ls logs | wc -l) files  ($(du -sh logs | cut -f1))"
