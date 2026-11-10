#!/bin/bash
# Fetch a model's results from the GCS bucket written by scripts/vast_models_sequence.sh: figures, per-run small files
# (index.parquet, meta.json, subset.json, SHA256SUMS.*), per-run logs, instance logs, checksums and prompts — not the
# activations. --subsets also fetches the tier-1 activation subsets (acts_subset_*; ~20 GB per model). Full shards
# (acts_[0-9]*) are never fetched. Verifies everything fetched with rclone check (md5) against the bucket.
#
#   GCS_KEY=<key> scripts/gcs_fetch_results.sh <bucket>/<tag> [--subsets]
# Layout: runs/<tag>_*/ and runs/<tag>_*.log, figures/<tag>_*/ and figures/stakes_pilot_<tag>/, snapshot/checksums_<tag>*,
# data/prompts_<tag>/, and the instance's untagged logs (tests.log, sequence.log, ...) in runs/vast_logs_<tag>/ so they
# cannot overwrite local logs of the same name.
set -eu
cd "$(dirname "$0")/.."
: "${GCS_KEY:?set GCS_KEY=<service-account key file>}"
SRC=${1:?<bucket>/<tag>}; TAG=$(basename "$SRC"); SUBSETS=${2:-}
RC=(rclone --gcs-service-account-file "$GCS_KEY" --gcs-bucket-policy-only --transfers 8 --checkers 16)
R=":gcs:$SRC"
RUNF=(--filter "- **/acts_[0-9]*.safetensors")
[ "$SUBSETS" = "--subsets" ] || RUNF+=(--filter "- **/acts_subset_*.safetensors")
RUNF+=(--filter "+ /${TAG}_*/**" --filter "+ /${TAG}_*.log" --filter "- *")
mkdir -p runs figures snapshot "data/prompts_${TAG}" "runs/vast_logs_${TAG}"
"${RC[@]}" copy "$R/runs" runs "${RUNF[@]}"
"${RC[@]}" copy "$R/runs" "runs/vast_logs_${TAG}" --filter "- /${TAG}_*" --filter "+ /*.log" --filter "+ /*.out" --filter "- *"
"${RC[@]}" copy "$R/figures" figures
"${RC[@]}" copy "$R/snapshot" snapshot
"${RC[@]}" copy "$R/data/prompts" "data/prompts_${TAG}"
echo "verifying (md5, every fetched file equal to the bucket)"
"${RC[@]}" check "$R/runs" runs "${RUNF[@]}" --one-way
"${RC[@]}" check "$R/figures" figures --one-way
"${RC[@]}" check "$R/snapshot" snapshot --one-way
"${RC[@]}" check "$R/data/prompts" "data/prompts_${TAG}" --one-way
echo "FETCH OK $TAG: $(find runs/${TAG}_*/ -type f | wc -l) run files, $(find figures/${TAG}_* figures/stakes_pilot_${TAG} -type f 2>/dev/null | wc -l) figure files"
