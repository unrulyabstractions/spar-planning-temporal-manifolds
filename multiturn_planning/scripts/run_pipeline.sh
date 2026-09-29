#!/bin/bash
# Full run on ONE GPU (Qwen3-14B bf16 ~ 30 GB: a 48 GB card). Every step logs to runs/pipeline.log.
#
#   bash multiturn_planning/scripts/run_pipeline.sh          # from the repo root
#   env: PY, MODEL, BATCH, NAME (run name), SMOKE=1 (15-conversation slice), OVERWRITE=1,
#        QUICK_EVAL=0 (full analysis here; the default 1 runs a few-minute sanity pass so a rented GPU isn't
#        kept idle during the CPU-bound analysis; run the full `scripts/evaluate.py` locally afterwards)
#
# Steps: 1 unit tests · 2 capture · 3 adherence report · 4 evaluate · 5 checksums (RUN/SHA256SUMS over
# everything to copy back; verify with `sha256sum -c` before destroying the instance).
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen3-14B}
BATCH=${BATCH:-16}
NAME=${NAME:-$(basename "$MODEL" | tr 'A-Z' 'a-z')_mtp_s0}
[[ "${SMOKE:-0}" == 1 ]] && NAME=smoke_$(basename "$MODEL") && SMOKE_ARG=--smoke
RUN=runs/$NAME; RES=results/$NAME; LOG=runs/pipeline.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs results
stamp() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
STEP=setup; CAPTURED=0
checksums() {
    cp "$LOG" "$RUN/pipeline.log"
    find "$RUN" "$RES" "$RUN".*.log -type f ! -name SHA256SUMS -print0 2>/dev/null | sort -z | xargs -0 sha256sum > "$RUN/SHA256SUMS"
}
on_error() {
    local rc=$1 line=$2; trap - ERR
    stamp "FAILED at step $STEP / line $line, exit $rc (see $LOG and $RUN.*.log)"
    if [[ $CAPTURED == 1 ]]; then stamp "capture had finished: writing checksums anyway"; checksums || true; fi
    exit "$rc"
}
trap 'on_error $? $LINENO' ERR
step() { STEP=$1; stamp "$STEP"; }

stamp "START model $MODEL batch $BATCH run $RUN commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
if [[ -e "$RUN/index.parquet" ]]; then
    if [[ "${OVERWRITE:-0}" == 1 ]]; then stamp "OVERWRITE=1: removing $RUN $RES"; rm -rf "$RUN" "$RES" "$RUN".*.log
    else stamp "FAILED at step setup: $RUN already holds a capture; move it or set OVERWRITE=1"; exit 1; fi
fi
step "1/5 unit tests";  $PY -m pytest -q tests >> "$LOG" 2>&1
step "2/5 capture";     $PY -u scripts/capture.py "$RUN" --model "$MODEL" --batch "$BATCH" ${SMOKE_ARG:-} > "$RUN.capture.log" 2>&1
CAPTURED=1; stamp "   $(tail -1 "$RUN.capture.log")"
step "3/5 adherence";   $PY scripts/adherence.py "$RUN" 2>/dev/null | tee "$RUN/adherence.txt" | tee -a "$LOG"
EVAL_ARGS=$([[ "${QUICK_EVAL:-1}" == 1 ]] && echo --quick || true)
step "4/5 evaluate ${EVAL_ARGS:-(full)}"; $PY -u scripts/evaluate.py "$RUN" $EVAL_ARGS > "$RUN.evaluate.log" 2>&1
step "5/5 checksums"; checksums
stamp "DONE: $(wc -l < "$RUN/SHA256SUMS") files in $RUN/SHA256SUMS ($(du -sh "$RUN" | cut -f1)). Verify the copy with: cd multiturn_planning && sha256sum -c $RUN/SHA256SUMS"
