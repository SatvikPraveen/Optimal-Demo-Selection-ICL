#!/bin/bash
# Progress of the grid: finished result files vs expected, plus queue state.
PROJECT_DIR="${PROJECT_DIR:-/general/satvikpraveen/Optimal-Demo-Selection-ICL}"
cd "$PROJECT_DIR"
echo "queue:"; squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.6D %R" | head -30
echo
echo "results: $(ls results/raw/*.json 2>/dev/null | wc -l) files"
ls results/raw/*.json 2>/dev/null | sed 's#results/raw/##; s#__seed.*##' | sort | uniq -c | awk '{printf "  %-45s %s seeds\n", $2, $1}'
echo
echo "errors in logs:"; grep -l -E "Traceback|CUDA out of memory|failed" logs/*.err logs/*.log 2>/dev/null | head
