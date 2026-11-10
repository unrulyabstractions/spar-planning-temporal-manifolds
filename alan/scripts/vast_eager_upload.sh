#!/bin/bash
# Upload finished capture runs to GCS while the chain is still running, so the end-of-chain upload in
# scripts/vast_models_sequence.sh only tops up (rclone copy skips files already present and identical) and the GPU idles
# for minutes instead of ~45 min per model. Never deletes anything: the sequence script's rclone check (md5) over every
# file still gates the deletion of shards.
#
# A run folder runs/<TAG>_<name>/ is uploaded once its capture log runs/<TAG>_<name>.log has a "done:" line (the shards
# no longer change); it is re-copied each pass (cheap: unchanged files are skipped), which picks up later additions
# such as subset exports. Stops for a tag when runs/chain_<TAG>.log says CHAIN DONE or FAILED; low CPU/IO priority.
#
#   GCS_DEST=<bucket>/<prefix> GCS_KEY=<key> TAGS="gemma-4-31b" nohup scripts/vast_eager_upload.sh > runs/eager_upload.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
: "${GCS_DEST:?}" "${GCS_KEY:?}" "${TAGS:?set TAGS=<tag> [<tag> ...]}"
PASS_SECONDS=${PASS_SECONDS:-300}
RC=(nice -n 19 ionice -c 3 env GOMAXPROCS=4 rclone --gcs-service-account-file "$GCS_KEY" --gcs-bucket-policy-only --transfers 8 --checkers 8)
for TAG in $TAGS; do
  LOG=runs/eager_upload_${TAG}.log
  stamp() { echo "$(date '+%F %T') $*" >> "$LOG"; }
  until [ -f runs/chain_${TAG}.log ]; do sleep 60; done
  stamp "eager upload for $TAG: watching runs/${TAG}_*.log (pass every ${PASS_SECONDS}s)"
  while ! grep -qE "CHAIN DONE|FAILED" runs/chain_${TAG}.log; do
    for L in runs/${TAG}_*.log; do
      R=${L%.log}
      [ -d "$R" ] && grep -q "^done:" "$L" || continue
      if "${RC[@]}" copy "$R" ":gcs:$GCS_DEST/$TAG/runs/$(basename "$R")" >> "$LOG" 2>&1; then
        stamp "pass: $(basename "$R") in bucket"
      else
        stamp "pass: $(basename "$R") copy returned an error (retried next pass)"
      fi
      grep -qE "CHAIN DONE|FAILED" runs/chain_${TAG}.log && break
    done
    sleep "$PASS_SECONDS"
  done
  stamp "eager upload for $TAG stopped: chain finished ($(grep -oE 'CHAIN DONE|FAILED' runs/chain_${TAG}.log | head -1))"
done
