#!/bin/bash
# Reproduce every dataset, capture, and analysis from committed code, in order (defaults: Qwen3-14B on the
# local 2-GPU split). Override with env: MODEL=Qwen/Qwen3.5-27B TAG=qwen3.5-27b CAP_ARGS="--batch-size 16".
# Captures run sequentially on the GPUs; each run's analyses start in the background as soon as its
# capture finishes. Progress markers go to runs/reproduce_all.log. Usage: nohup scripts/reproduce_all.sh &
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; MODEL=${MODEL:-Qwen/Qwen3-14B}; TAG=${TAG:-qwen3-14b}; CAP="--model $MODEL ${CAP_ARGS:---batch-size 8 --split 19}"
CHECKSUMS=${CHECKSUMS:-snapshot/checksums_post_20260925.json}
LOG=${REPRO_LOG:-runs/reproduce_all.log}; mkdir -p runs figures data/prompts   # REPRO_LOG: per-model log when several models share runs/
stamp() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
stamp "START commit $(git rev-parse --short HEAD) model $MODEL tag $TAG cap [$CAP]"

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

capture ${TAG}_investment_n2000_s0 data/prompts/investment_n2000_s0.parquet
bg ${TAG}_investment_n2000_s0 bash -c "scripts/analyze_run.sh runs/${TAG}_investment_n2000_s0 && $PY scripts/within_bin_control.py runs/${TAG}_investment_n2000_s0"

capture ${TAG}_matrix_s0 data/prompts/matrix/capture_s0.parquet
bg ${TAG}_matrix_s0 bash -c "scripts/analyze_run.sh runs/${TAG}_matrix_s0 && \
  $PY scripts/transfer_eval.py runs/${TAG}_matrix_s0 figures/${TAG}_matrix_s0/transfer --train-renderings structured && \
  $PY scripts/transfer_eval.py runs/${TAG}_matrix_s0 figures/${TAG}_matrix_s0/transfer --train-renderings structured,plain,varied --distractor-sweep && \
  $PY scripts/transfer_eval.py runs/${TAG}_matrix_s0 figures/${TAG}_matrix_s0/transfer --train-renderings structured,plain,varied --train-domains investment"

capture ${TAG}_extended_n2000_s0 data/prompts/investment_extended_n2000_s0.parquet
bg ${TAG}_extended_n2000_s0 bash -c "scripts/analyze_run.sh runs/${TAG}_extended_n2000_s0 && \
  $PY scripts/hypothesis_checks.py runs/${TAG}_extended_n2000_s0"

capture ${TAG}_phrasings_n2000_s0 data/prompts/investment_phrasings_n2000_s0.parquet
bg ${TAG}_phrasings_n2000_s0 bash -c "$PY scripts/analyze_variants.py runs/${TAG}_phrasings_n2000_s0 figures/${TAG}_phrasings_n2000_s0 --factor phrasing --cells 0.35L:R0,0.55L:T3,0.55L:R0,0.72L:R0,0.92L:T3 && \
  $PY scripts/hypothesis_checks.py runs/${TAG}_phrasings_n2000_s0 --ushape-cells 0.55L:T3"

capture ${TAG}_units_paired_n2000_s1 data/prompts/investment_units_paired_n2000_s1.parquet
bg ${TAG}_units_paired_n2000_s1 bash -c "$PY scripts/analyze_paired.py runs/${TAG}_units_paired_n2000_s1 figures/${TAG}_units_paired_n2000_s1"

stamp "all captures done; waiting for analyses"
wait
stamp "checksums"
$PY scripts/run_checksums.py runs/${TAG}_investment_n2000_s0 runs/${TAG}_matrix_s0 runs/${TAG}_extended_n2000_s0 runs/${TAG}_phrasings_n2000_s0 runs/${TAG}_units_paired_n2000_s1 --out "$CHECKSUMS" 2>&1 | grep -v Warning | tee -a $LOG
stamp "ALL DONE"
