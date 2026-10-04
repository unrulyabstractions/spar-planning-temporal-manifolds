#!/bin/bash
# Offload capture runs to a GCS bucket and free their full shards locally. Per run folder: upload the folder and its
# runs/<name>.log to gs://GCS_DEST/<name>/, verify with rclone check (md5, one-way: every local file present and equal
# in the bucket) and a file count, and only then delete the run's full shards (acts_[0-9]*.safetensors). index.parquet,
# meta.json, subset files and logs stay; nothing outside the listed run folders is touched. Any failure stops the loop
# with that run's shards kept. Rerunning resumes (rclone copy skips files already uploaded).
#
#   GCS_DEST=<bucket>/<prefix> GCS_KEY=<service-account key> nohup scripts/gcs_offload_runs.sh runs/A runs/B ... > runs/offload.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
: "${GCS_DEST:?set GCS_DEST=<bucket>/<prefix>}" "${GCS_KEY:?set GCS_KEY=<service-account key file>}"
LOG=${OFFLOAD_LOG:-runs/offload.log}
stamp() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
die() { stamp "FAILED: $*"; exit 1; }
RC=(rclone --gcs-service-account-file "$GCS_KEY" --gcs-bucket-policy-only --transfers 8 --checkers 16)
stamp "OFFLOAD START commit $(git rev-parse --short HEAD) dest gs://$GCS_DEST, $# runs"
for R in "$@"; do
  R=${R%/}; name=$(basename "$R"); D=":gcs:$GCS_DEST/$name"
  [ -f "$R/meta.json" ] || die "$R is not a run folder (no meta.json)"
  stamp "$name: upload $(du -sh "$R" | cut -f1)"
  "${RC[@]}" copy "$R" "$D" >> "$LOG" 2>&1 || die "$name: upload"
  [ -f "runs/$name.log" ] && { "${RC[@]}" copyto "runs/$name.log" "$D/$name.log" >> "$LOG" 2>&1 || die "$name: upload log"; }
  "${RC[@]}" check "$R" "$D" --one-way >> "$LOG" 2>&1 || die "$name: rclone check failed"
  n_local=$(find "$R" -type f | wc -l); n_remote=$("${RC[@]}" lsf -R --files-only "$D" | grep -v "^$name.log$" | wc -l)
  [ "$n_local" = "$n_remote" ] || die "$name: file count local $n_local vs bucket $n_remote"
  rm -f "$R"/acts_[0-9]*.safetensors
  stamp "$name: verified ($n_local files, md5 equal); full shards deleted, folder now $(du -sh "$R" | cut -f1)"
done
stamp "OFFLOAD DONE; disk $(df -h . | tail -1 | awk '{print $4" free"}')"
