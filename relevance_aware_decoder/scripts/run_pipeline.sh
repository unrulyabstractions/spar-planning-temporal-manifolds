#!/bin/bash
# Full experiment on ONE GPU with >= 40 GB (Qwen3-14B in bf16 is ~30 GB). Written for a rented vast.ai
# instance, but runs anywhere with the Python deps installed. Every step logs to runs/pipeline.log.
#
#   bash relevance_aware_decoder/scripts/run_pipeline.sh            # from the repo root
#   env: PY (python), MODEL, BATCH, SEED (re-draws distractors only), LAYERS (evaluation sweep; also the exported
#        subset unless SUBSET_LAYERS is set), SMOKE=1 (small prompt slice), OVERWRITE=1 (replace an existing run),
#        EXTRA_CAPTURE_ARGS (e.g. "--split 19" on Alan's 2-GPU machine)
#
# Steps: 1 prompts · 2 unit tests · 3 capture check (hooks vs library) · 4 capture · 5 activation subset
#        (every evaluated layer × position, so evaluate.py can be rerun without the full shards) · 6 evaluate ·
#        7 checksums: RUN/SHA256SUMS over the copy-back set ("tier 1": everything except the full acts_NNNN shards;
#        verify with `sha256sum -c` before destroying anything) and RUN/SHA256SUMS.shards (the full shards, which stay
#        on the instance; kept as a record so a future recapture can be checked for byte-identity)
# RUN/manifest.json records code, prompts, model revision, software, hardware and knobs (scripts/manifest.py).
# Any failure stamps "FAILED at step ..." into runs/pipeline.log; if the capture had finished, checksums are still written.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ALAN="$HERE/../alan"
cd "$HERE"
PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen3-14B}
BATCH=${BATCH:-16}
SEED=${SEED:-0}
LAYERS=${LAYERS:-14,18,22,26,29,33,37}
SUBSET_LAYERS=${SUBSET_LAYERS:-$LAYERS}
POSITIONS=T0,T1,T2,T3,T4,T5,T6,T7,T8,R0
PROMPTS=data/prompts_s${SEED}.parquet
RUN=runs/qwen3-14b_relevance_s${SEED}
[[ "${SMOKE:-0}" == 1 ]] && PROMPTS=data/prompts_s${SEED}_smoke.parquet && RUN=runs/smoke_$(basename "$MODEL")
NAME=$(basename "$RUN"); RES=results/$NAME
export PY MODEL BATCH SEED LAYERS SUBSET_LAYERS                  # recorded in the manifest
export PYTHONPATH="$ALAN${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs data
LOG=runs/pipeline.log
stamp() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
STEP=setup; STEP_LOG=$LOG; CAPTURED=0

checksums() {   # paths relative to relevance_aware_decoder/; the log is not touched afterwards
    local p=() f
    cp "$LOG" "$RUN/pipeline.log"; cp runs/gen_prompts.log "$RUN/gen_prompts.log"   # snapshots: the shared logs change with later runs
    find "$RUN" -maxdepth 1 -name 'acts_[0-9]*.safetensors' -print0 | sort -z | xargs -0 -r sha256sum > "$RUN/SHA256SUMS.shards.tmp"
    mv "$RUN/SHA256SUMS.shards.tmp" "$RUN/SHA256SUMS.shards"
    for f in "$RUN" "$RES" "$RUN.capture.log" "$RUN.evaluate.log" data/prompts_s"${SEED}".* data/prompts_s"${SEED}"_smoke.*; do
        if [[ -e "$f" ]]; then p+=("$f"); fi
    done
    find "${p[@]}" -type f ! -name SHA256SUMS ! -name '*.tmp' ! -name 'acts_[0-9]*.safetensors' -print0 | sort -z | xargs -0 sha256sum > "$RUN/SHA256SUMS.tmp"
    mv "$RUN/SHA256SUMS.tmp" "$RUN/SHA256SUMS"
    echo "wrote $RUN/SHA256SUMS ($(wc -l < "$RUN/SHA256SUMS") files to copy back) and $RUN/SHA256SUMS.shards ($(wc -l < "$RUN/SHA256SUMS.shards") full shards, not copied)."
    echo "Verify the copy with:  cd relevance_aware_decoder && sha256sum -c $RUN/SHA256SUMS"
}
on_error() {
    local rc=$1 line=$2; trap - ERR
    stamp "FAILED at step $STEP / line $line, exit $rc (see $STEP_LOG)"
    if [[ -d "$RUN" ]]; then $PY scripts/manifest.py "$RUN/manifest.json" --stage done --status "failed at step $STEP (exit $rc)" >> "$LOG" 2>&1 || true; fi
    if [[ $CAPTURED == 1 ]]; then stamp "capture had finished: writing checksums anyway"; checksums || true; fi
    exit "$rc"
}
trap 'on_error $? $LINENO' ERR
step() { STEP=$1; STEP_LOG=$2; stamp "$STEP"; }

stamp "START model $MODEL batch $BATCH prompts $PROMPTS run $RUN commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
if [[ -e "$RUN/index.parquet" ]]; then                          # never overwrite a capture by accident
    if [[ "${OVERWRITE:-0}" == 1 ]]; then stamp "OVERWRITE=1: removing previous $RUN, $RES and their logs"; rm -rf "$RUN" "$RES" "$RUN".*.log
    else stamp "FAILED at step setup: $RUN already holds a capture; move it away or set OVERWRITE=1"; exit 1; fi
fi

step "1/7 prompts" runs/gen_prompts.log;  $PY scripts/gen_prompts.py data --seed "$SEED" --smoke > runs/gen_prompts.log 2>&1
mkdir -p "$RUN"; $PY scripts/manifest.py "$RUN/manifest.json" --stage start --prompts "$PROMPTS" --model "$MODEL" >> "$LOG" 2>&1
step "2/7 unit tests" "$LOG";              $PY -m pytest -q tests >> "$LOG" 2>&1
# our copy of alan/scripts/verify_capture.py (fixes the unspaced-label case, e.g. `I choose: **a)`)
step "3/7 capture check" "$LOG"
$PY scripts/verify_capture.py --model "$MODEL" --n 4 2>&1 | tee -a "$LOG" | tail -1 | grep -q "VERIFY OK" \
    || on_error 1 "$LINENO"                                    # no "VERIFY OK" on the last line
step "4/7 capture ($($PY -c "import pandas as pd; print(len(pd.read_parquet('$PROMPTS')))") prompts)" "$RUN.capture.log"
$PY -u "$ALAN/scripts/capture.py" "$PROMPTS" "$RUN" --model "$MODEL" --batch-size "$BATCH" ${EXTRA_CAPTURE_ARGS:-} > "$RUN.capture.log" 2>&1
CAPTURED=1; stamp "   $(tail -1 "$RUN.capture.log")"
$PY scripts/manifest.py "$RUN/manifest.json" --stage captured --model "$MODEL" >> "$LOG" 2>&1
step "5/7 activation subset (layers $SUBSET_LAYERS)" "$LOG"
$PY "$ALAN/scripts/export_subset.py" "$RUN" --layers "$SUBSET_LAYERS" --positions "$POSITIONS" >> "$LOG" 2>&1
step "6/7 evaluate" "$RUN.evaluate.log"
$PY -u scripts/evaluate.py "$RUN" "$PROMPTS" "$RES" --layers "$LAYERS" --positions "$POSITIONS" > "$RUN.evaluate.log" 2>&1
cp "$RUN/index.parquet" "$RUN/meta.json" "$RES/"
$PY scripts/manifest.py "$RUN/manifest.json" --stage done --status ok >> "$LOG" 2>&1
cp "$RUN/manifest.json" "$RES/"
step "7/7 checksums" "$LOG"
stamp "DONE. Copy back (tier 1): $RUN/ except acts_NNNN.safetensors, $RES/, runs/*.log, data/prompts_s${SEED}.*; verify with sha256sum -c $RUN/SHA256SUMS (COST.md step 6)"
checksums
