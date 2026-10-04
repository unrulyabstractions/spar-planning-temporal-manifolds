#!/bin/bash
# Several models in sequence on ONE rented instance, each model's outputs uploaded to a GCS bucket and verified before
# its full shards and weights are deleted: the instance disk holds one model's weights + shards (~55-63 GB + up to
# ~250 GB), not two. Run INSIDE tmux on the instance, from the repo root:
#
#   GCS_DEST=<bucket>/<prefix> GCS_KEY=/root/<service-account>.json \
#   MODELS="Qwen/Qwen3.8-27B:qwen3.8-27b google/gemma-4-31B-it:gemma-4-31b" \
#   nohup scripts/vast_models_sequence.sh > runs/sequence.out 2>&1 &
#
# Per model: delete the OTHER models' cached weights; scripts/vast_full_chain.sh (skipped if runs/chain_${TAG}.log already
# says CHAIN DONE, so a rerun resumes); upload runs/${TAG}_*, all runs/*.log, figures, checksums and prompts to
# gs://${GCS_DEST}/${TAG}/ with rclone; rclone check (md5, one-way: every local file present and equal in the bucket);
# only then delete runs/${TAG}_*/acts_[0-9]*.safetensors and the model's weights. Any failure stops the sequence with
# nothing deleted for that model.
set -u
cd "$(dirname "$0")/.."
: "${GCS_DEST:?set GCS_DEST=<bucket>/<prefix>}" "${GCS_KEY:?set GCS_KEY=<service-account key file>}"
MODELS=${MODELS:-"Qwen/Qwen3.8-27B:qwen3.8-27b google/gemma-4-31B-it:gemma-4-31b"}
LOG=runs/sequence.log; mkdir -p runs
stamp() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
die() { stamp "FAILED: $*"; exit 1; }
RC=(rclone --gcs-service-account-file "$GCS_KEY" --gcs-bucket-policy-only --transfers 16 --checkers 16)
HUB=${HF_HOME:-$HOME/.cache/huggingface}/hub
hfdir() { echo "$HUB/models--${1//\//--}"; }
# huggingface_hub >= 1.x keeps file contents in a shared content-addressed store (hub/blobs/xx/<sha256>) that the
# per-model folders only link into, so removing a model folder frees nothing by itself. free_weights removes the folder,
# then every file in the shared store that no remaining snapshot resolves to (older per-model layouts: the rm suffices).
free_weights() {
  local d; d=$(hfdir "$1"); [ -d "$d" ] || return 0
  rm -rf "$d"
  if [ -d "$HUB/blobs" ]; then
    find "$HUB"/models--*/snapshots -type l -exec readlink -f {} \; 2>/dev/null | sort -u > /tmp/hf_referenced.txt
    find "$HUB/blobs" -mindepth 2 -type f | sort -u | comm -23 - /tmp/hf_referenced.txt > /tmp/hf_orphans.txt
    xargs -r -a /tmp/hf_orphans.txt rm -f
  fi
  stamp "freed weights of $1 ($(wc -l < /tmp/hf_orphans.txt 2>/dev/null || echo 0) shared blobs); disk $(df -h . | tail -1 | awk '{print $4" free"}')"
}

stamp "SEQUENCE START commit $(git rev-parse --short HEAD) models [$MODELS] dest gs://$GCS_DEST"
"${RC[@]}" lsf ":gcs:$GCS_DEST" --max-depth 1 > /dev/null 2>> $LOG || die "cannot list gs://$GCS_DEST with $GCS_KEY"
echo "probe $(date)" > runs/.gcs_probe && "${RC[@]}" copyto runs/.gcs_probe ":gcs:$GCS_DEST/.probe" 2>> $LOG \
  && "${RC[@]}" deletefile ":gcs:$GCS_DEST/.probe" 2>> $LOG || die "cannot write to gs://$GCS_DEST"
stamp "bucket reachable and writable"

for spec in $MODELS; do
  MODEL=${spec%%:*}; TAG=${spec##*:}; D=":gcs:$GCS_DEST/$TAG"
  for other in $MODELS; do o=${other%%:*}; [ "$o" = "$MODEL" ] || free_weights "$o"; done
  if grep -qs "CHAIN DONE" runs/chain_${TAG}.log; then
    stamp "$TAG: chain already done, resuming at upload"
  else
    stamp "$TAG: chain start ($MODEL); disk $(df -h . | tail -1 | awk '{print $4" free"}')"
    MODEL=$MODEL TAG=$TAG scripts/vast_full_chain.sh > runs/chain_${TAG}.out 2>&1
    grep -qs "CHAIN DONE" runs/chain_${TAG}.log || die "$TAG: chain did not finish (runs/chain_${TAG}.log)"
  fi
  stamp "$TAG: upload start ($(du -shc runs/${TAG}_* | tail -1 | cut -f1) of runs)"
  UL=runs/upload_${TAG}.log
  "${RC[@]}" copy runs "$D/runs" --include "/${TAG}_*/**" --include "/*.log" --include "/*.out" >> $UL 2>&1 || die "$TAG: upload runs"
  "${RC[@]}" copy figures "$D/figures" --include "/${TAG}_*/**" --include "/stakes_pilot_${TAG}/**" >> $UL 2>&1 || die "$TAG: upload figures"
  "${RC[@]}" copy snapshot "$D/snapshot" --include "/checksums_${TAG}*" >> $UL 2>&1 || die "$TAG: upload snapshot"
  "${RC[@]}" copy data/prompts "$D/data/prompts" >> $UL 2>&1 || die "$TAG: upload prompts"
  "${RC[@]}" check runs "$D/runs" --one-way --include "/${TAG}_*/**" >> $UL 2>&1 || die "$TAG: rclone check of runs failed (see $UL)"
  "${RC[@]}" check figures "$D/figures" --one-way --include "/${TAG}_*/**" --include "/stakes_pilot_${TAG}/**" >> $UL 2>&1 || die "$TAG: rclone check of figures failed"
  # trailing slash: count files inside run folders only (runs/${TAG}_*.log are files too, and are not in the remote count)
  n_local=$(find runs/${TAG}_*/ -type f | wc -l)
  n_remote=$("${RC[@]}" lsf -R --files-only "$D/runs" --include "/${TAG}_*/**" | wc -l)
  [ "$n_local" = "$n_remote" ] || die "$TAG: file count local $n_local vs bucket $n_remote"
  stamp "$TAG: uploaded and verified ($n_local files under runs/, md5 equal)"
  rm -f runs/${TAG}_*/acts_[0-9]*.safetensors; stamp "$TAG: full shards deleted on the instance"
  free_weights "$MODEL"
done
stamp "SEQUENCE DONE"
