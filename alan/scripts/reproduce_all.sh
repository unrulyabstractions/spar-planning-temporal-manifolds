#!/bin/bash
# Reproduce every Qwen3-14B dataset, capture, and analysis from committed code, in order.
# Captures run sequentially on the GPUs; each run's analyses start in the background as soon as its
# capture finishes. Progress markers go to runs/reproduce_all.log. Usage: nohup scripts/reproduce_all.sh &
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; MODEL=Qwen/Qwen3-14B; CAP="--model $MODEL --batch-size 8 --split 19"
LOG=runs/reproduce_all.log; mkdir -p runs figures data/prompts
stamp() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
stamp "START commit $(git rev-parse --short HEAD)"

stamp "datasets"
$PY scripts/gen_prompts.py data/prompts/investment_n2000_s0.parquet --n 2000 --seed 0 > /dev/null
$PY scripts/gen_prompts.py data/prompts/investment_n16_s1.parquet --n 16 --seed 1 > /dev/null
$PY scripts/gen_prompts.py data/prompts/investment_phrasings_n2000_s0.parquet --n 2000 --seed 0 --phrasings 0,1,2,3 > /dev/null
$PY scripts/gen_prompts.py data/prompts/investment_units_paired_n2000_s1.parquet --n 2000 --seed 1 --unit-rewrite --paired > /dev/null
$PY scripts/gen_prompts.py data/prompts/investment_extended_n2000_s0.parquet --n 2000 --seed 0 --grid extended > /dev/null
$PY scripts/gen_matrix.py data/prompts/matrix --seed 0 > /dev/null
$PY - <<'PYEOF'
import pandas as pd
m = pd.read_parquet("data/prompts/matrix/matrix_s0.parquet"); k = pd.read_parquet("data/prompts/matrix/controls_s0.parquet")
pd.concat([m, k], ignore_index=True).to_parquet("data/prompts/matrix/capture_s0.parquet", index=False)
PYEOF
stamp "datasets done"

capture() {  # name prompts
  stamp "capture $1 start"
  $PY -u scripts/capture.py "$2" "runs/$1" $CAP > "runs/$1.log" 2>&1
  stamp "capture $1 $(tail -1 runs/$1.log)"
}
bg() {  # name command...   (analysis in background, its own log)
  local name=$1; shift
  ( "$@" > "runs/$name.analysis.log" 2>&1; echo "$(date '+%F %T') analysis $name done" >> $LOG ) &
}

capture qwen3-14b_investment_n2000_s0 data/prompts/investment_n2000_s0.parquet
bg qwen3-14b_investment_n2000_s0 bash -c "scripts/analyze_run.sh runs/qwen3-14b_investment_n2000_s0 && $PY scripts/within_bin_control.py runs/qwen3-14b_investment_n2000_s0"

capture qwen3-14b_matrix_s0 data/prompts/matrix/capture_s0.parquet
bg qwen3-14b_matrix_s0 bash -c "scripts/analyze_run.sh runs/qwen3-14b_matrix_s0 && \
  $PY scripts/transfer_eval.py runs/qwen3-14b_matrix_s0 figures/qwen3-14b_matrix_s0/transfer --train-renderings structured && \
  $PY scripts/transfer_eval.py runs/qwen3-14b_matrix_s0 figures/qwen3-14b_matrix_s0/transfer --train-renderings structured,plain,varied --distractor-sweep && \
  $PY scripts/transfer_eval.py runs/qwen3-14b_matrix_s0 figures/qwen3-14b_matrix_s0/transfer --train-renderings structured,plain,varied --train-domains investment"

capture qwen3-14b_extended_n2000_s0 data/prompts/investment_extended_n2000_s0.parquet
bg qwen3-14b_extended_n2000_s0 bash -c "scripts/analyze_run.sh runs/qwen3-14b_extended_n2000_s0 && \
  $PY scripts/hypothesis_checks.py runs/qwen3-14b_extended_n2000_s0"

capture qwen3-14b_phrasings_n2000_s0 data/prompts/investment_phrasings_n2000_s0.parquet
bg qwen3-14b_phrasings_n2000_s0 bash -c "$PY scripts/analyze_variants.py runs/qwen3-14b_phrasings_n2000_s0 figures/qwen3-14b_phrasings_n2000_s0 --factor phrasing --cells 14:R0,22:T3,22:R0,29:R0,37:T3 && \
  $PY scripts/hypothesis_checks.py runs/qwen3-14b_phrasings_n2000_s0 --ushape-cells 22:T3"

capture qwen3-14b_units_paired_n2000_s1 data/prompts/investment_units_paired_n2000_s1.parquet
bg qwen3-14b_units_paired_n2000_s1 bash -c "$PY scripts/analyze_paired.py runs/qwen3-14b_units_paired_n2000_s1 figures/qwen3-14b_units_paired_n2000_s1"

stamp "all captures done; waiting for analyses"
wait
stamp "checksums"
$PY scripts/run_checksums.py runs/qwen3-14b_investment_n2000_s0 runs/qwen3-14b_matrix_s0 runs/qwen3-14b_extended_n2000_s0 runs/qwen3-14b_phrasings_n2000_s0 runs/qwen3-14b_units_paired_n2000_s1 --out snapshot/checksums_post_20260925.json 2>&1 | grep -v Warning | tee -a $LOG
stamp "ALL DONE"
