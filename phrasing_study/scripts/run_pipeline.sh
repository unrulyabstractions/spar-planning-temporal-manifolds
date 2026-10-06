#!/bin/bash
# Full phrasing-study run on ONE GPU (Qwen3-14B bf16 ~ 30 GB: a 48 GB card). Every step logs to runs/pipeline.log.
#
#   bash phrasing_study/scripts/run_pipeline.sh          # from the repo root
#   env: PY, MODEL, BATCH (choice forward batch, 16), GEN_BATCH (generation batch, 16), NAME (run name),
#        SMOKE_FIRST=1 (default: a small slice of every protocol first, with sanity gates; 0 skips it),
#        OVERWRITE=1, GCS_DEST=<bucket>/<prefix> + GCS_KEY=<service-account key file> (upload + verify at the end)
#
# Steps: 1 unit tests · 2 smoke slice + gates · 3 capture (choice, state, multiturn) · 4 evaluate (behaviour, a few min)
#        · 5 checksums (RUN/SHA256SUMS = small files + results; RUN/SHA256SUMS.acts = activation shards)
#        · 6 upload RUN + results to gs://GCS_DEST/NAME/ with rclone, verified (md5 check + file count).
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen3-14B}
BATCH=${BATCH:-16}
GEN_BATCH=${GEN_BATCH:-16}
NAME=${NAME:-$(basename "$MODEL" | tr 'A-Z' 'a-z')_phrasing_s0}
RUN=runs/$NAME; RES=results/$NAME; LOG=runs/pipeline.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cores() {   # the container's CPU quota (cgroup v2 cpu.max), else nproc; rented boxes often see more cores than they get
    local q p; read -r q p 2>/dev/null < /sys/fs/cgroup/cpu.max || { nproc; return; }
    if [[ "$q" == max || -z "$p" ]]; then nproc; else echo $(( (q + p - 1) / p )); fi
}
NCORES=$(cores); export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$NCORES} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$NCORES}
mkdir -p runs results
stamp() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
STEP=setup; CAPTURED=0
checksums() {
    cp "$LOG" "$RUN/pipeline.log"
    find "$RUN" -maxdepth 1 -name '*_acts_*.safetensors' -print0 | sort -z | xargs -0 -r sha256sum > "$RUN/SHA256SUMS.acts"
    find "$RUN" "$RES" "$RUN".*.log -type f ! -name 'SHA256SUMS' ! -name '*_acts_*.safetensors' -print0 2>/dev/null \
        | sort -z | xargs -0 sha256sum > "$RUN/SHA256SUMS"
}
on_error() {
    local rc=$1 line=$2; trap - ERR
    stamp "FAILED at step $STEP / line $line, exit $rc (see $LOG and $RUN.*.log)"
    if [[ $CAPTURED == 1 ]]; then stamp "capture had finished: writing checksums anyway"; checksums || true; fi
    exit "$rc"
}
trap 'on_error $? $LINENO' ERR
step() { STEP=$1; stamp "$STEP"; }

stamp "START model $MODEL batch $BATCH/$GEN_BATCH threads $NCORES run $RUN commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
if [[ -e "$RUN/capture_meta.json" ]]; then
    if [[ "${OVERWRITE:-0}" == 1 ]]; then stamp "OVERWRITE=1: removing $RUN $RES"; rm -rf "$RUN" "$RES" "$RUN".*.log
    else stamp "FAILED at step setup: $RUN already exists; move it or set OVERWRITE=1"; exit 1; fi
fi
if [[ -n "${GCS_DEST:-}" ]]; then   # fail now, not after the run, if the bucket is unreachable
    : "${GCS_KEY:?GCS_DEST is set: also set GCS_KEY=<service-account key file>}"
    command -v rclone > /dev/null || { stamp "FAILED: rclone not installed (apt-get install -y rclone)"; exit 1; }
    rclone --gcs-service-account-file "$GCS_KEY" --gcs-bucket-policy-only lsf ":gcs:${GCS_DEST%%/*}" --max-depth 1 > /dev/null \
        || { stamp "FAILED: cannot list gs://${GCS_DEST%%/*} with $GCS_KEY"; exit 1; }
    stamp "bucket gs://${GCS_DEST%%/*} reachable"
fi

step "1/6 unit tests"; $PY -m pytest -q tests >> "$LOG" 2>&1
if [[ "${SMOKE_FIRST:-1}" == 1 ]]; then
    step "2/6 smoke slice"
    SM=runs/smoke_$NAME; rm -rf "$SM" "results/smoke_$NAME"
    $PY -u scripts/capture.py "$SM" --model "$MODEL" --batch "$BATCH" --gen-batch "$GEN_BATCH" --smoke > "$SM.capture.log" 2>&1
    $PY scripts/evaluate.py "$SM" > "$SM.evaluate.log" 2>&1
    $PY scripts/gates.py "$SM" 2>&1 | tee -a "$LOG"          # exits non-zero if a gate fails
else
    stamp "2/6 smoke slice skipped (SMOKE_FIRST=0)"
fi
step "3/6 capture"
$PY -u scripts/capture.py "$RUN" --model "$MODEL" --batch "$BATCH" --gen-batch "$GEN_BATCH" > "$RUN.capture.log" 2>&1
CAPTURED=1; stamp "   $(tail -1 "$RUN.capture.log")"
$PY -m pip freeze > "$RUN/pip_freeze.txt" 2>/dev/null || true
step "4/6 evaluate"; $PY -u scripts/evaluate.py "$RUN" > "$RUN.evaluate.log" 2>&1
$PY scripts/gates.py "$RUN" 2>&1 | tee -a "$LOG" || stamp "   (gates failed on the full run: look before trusting results)"
step "5/6 checksums"; checksums
stamp "   $(wc -l < "$RUN/SHA256SUMS") small files, $(wc -l < "$RUN/SHA256SUMS.acts") activation shards, $(du -sh "$RUN" | cut -f1) total"
if [[ -n "${GCS_DEST:-}" ]]; then
    step "6/6 upload to gs://$GCS_DEST/$NAME"
    RC=(rclone --gcs-service-account-file "$GCS_KEY" --gcs-bucket-policy-only --transfers 8 --checkers 16)
    D=":gcs:$GCS_DEST/$NAME"
    "${RC[@]}" copy "$RUN" "$D/run" >> "$LOG" 2>&1
    "${RC[@]}" copy "$RES" "$D/results" >> "$LOG" 2>&1
    for f in "$RUN".*.log; do "${RC[@]}" copyto "$f" "$D/logs/$(basename "$f")" >> "$LOG" 2>&1; done
    "${RC[@]}" check "$RUN" "$D/run" --one-way >> "$LOG" 2>&1
    "${RC[@]}" check "$RES" "$D/results" --one-way >> "$LOG" 2>&1
    n_local=$(find "$RUN" -type f | wc -l); n_remote=$("${RC[@]}" lsf -R --files-only "$D/run" | wc -l)
    [[ "$n_local" == "$n_remote" ]] || { stamp "FAILED: file count local $n_local vs bucket $n_remote"; exit 1; }
    stamp "   uploaded and verified: $n_local run files (md5 equal) -> gs://$GCS_DEST/$NAME/"
else
    stamp "6/6 upload skipped (GCS_DEST not set)"
fi
stamp "DONE $NAME. Verify a copy with: cd phrasing_study && sha256sum -c $RUN/SHA256SUMS"
