#!/bin/bash
# Full experiment on ONE GPU with >= 40 GB (Qwen3-14B in bf16 is ~30 GB). Written for a rented vast.ai
# instance, but runs anywhere with the Python deps installed. Every step logs to runs/pipeline.log.
#
#   bash relevance_aware_decoder/scripts/run_pipeline.sh            # from the repo root
#   env: PY (python), MODEL, BATCH, SEED, LAYERS (evaluation sweep), SUBSET_LAYERS, SMOKE=1 (small prompt slice),
#        EXTRA_CAPTURE_ARGS (e.g. "--split 19" on Alan's 2-GPU machine)
#
# Steps: 1 prompts · 2 unit tests · 3 capture check (hooks vs library) · 4 capture · 5 evaluate · 6 shareable subset
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ALAN="$HERE/../alan"
cd "$HERE"
PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen3-14B}
BATCH=${BATCH:-16}
SEED=${SEED:-0}
PROMPTS=data/prompts_s${SEED}.parquet
RUN=runs/qwen3-14b_relevance_s${SEED}
[[ "${SMOKE:-0}" == 1 ]] && PROMPTS=data/prompts_s${SEED}_smoke.parquet && RUN=runs/smoke_$(basename "$MODEL")
export PYTHONPATH="$ALAN${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs data
LOG=runs/pipeline.log
stamp() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
stamp "START model $MODEL batch $BATCH prompts $PROMPTS run $RUN commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"

stamp "1/6 prompts";      $PY scripts/gen_prompts.py data --seed "$SEED" --smoke > runs/gen_prompts.log
stamp "2/6 unit tests";   $PY -m pytest -q tests >> "$LOG" 2>&1
stamp "3/6 capture check"; $PY "$ALAN/scripts/verify_capture.py" --model "$MODEL" --n 4 2>&1 | tee -a "$LOG" | tail -1 | grep -q "VERIFY OK" \
                            || { stamp "capture check FAILED (see $LOG)"; exit 1; }
stamp "4/6 capture ($($PY -c "import pandas as pd; print(len(pd.read_parquet('$PROMPTS')))") prompts)"
$PY -u "$ALAN/scripts/capture.py" "$PROMPTS" "$RUN" --model "$MODEL" --batch-size "$BATCH" ${EXTRA_CAPTURE_ARGS:-} > "$RUN.capture.log" 2>&1
stamp "   $(tail -1 "$RUN.capture.log")"
stamp "5/6 evaluate";     $PY -u scripts/evaluate.py "$RUN" "$PROMPTS" "results/$(basename "$RUN")" \
                            --layers "${LAYERS:-14,18,22,26,29,33,37}" > "$RUN.evaluate.log" 2>&1
stamp "6/6 subset";       $PY "$ALAN/scripts/export_subset.py" "$RUN" --layers "${SUBSET_LAYERS:-22,26,29}" \
                            --positions T0,T1,T2,T3,T4,T5,T6,T7,T8,R0 >> "$LOG" 2>&1
cp "$RUN/index.parquet" "$RUN/meta.json" "results/$(basename "$RUN")/"
stamp "DONE. Copy back: results/$(basename "$RUN")/ and $RUN/{subset.json,acts_subset_*} (full shards stay on the instance)"
