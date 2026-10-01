#!/bin/bash
# Copy tier 1 of a Vast.ai run back to this machine and verify it. Tier 1 = everything except the full
# activation shards: index/meta, analysis subsets, figures, logs, prompts, checksums. Rerun to resume.
# Usage: scripts/vast_fetch.sh HOST PORT [REMOTE_REPO=/workspace/SPAR-2026] [TAG=qwen3.5-27b]
set -eu
HOST=$1; PORT=$2; REMOTE=${3:-/workspace/SPAR-2026}; TAG=${4:-qwen3.5-27b}
cd "$(dirname "$0")/.."
rsync -avP --exclude 'acts_[0-9]*.safetensors' -e "ssh -p $PORT" "root@$HOST:$REMOTE/runs/${TAG}_*" runs/
rsync -avP -e "ssh -p $PORT" --include '*.log' --exclude '*' "root@$HOST:$REMOTE/runs/" runs/ || true   # chain/reproduce/tests/verify/capture logs (brace lists are not expanded by the remote shell)
rsync -avP -e "ssh -p $PORT" "root@$HOST:$REMOTE/figures/${TAG}_*" "root@$HOST:$REMOTE/figures/stakes_pilot_${TAG}" figures/ || true
rsync -avP -e "ssh -p $PORT" "root@$HOST:$REMOTE/snapshot/checksums_${TAG}*.json" snapshot/
rsync -avP -e "ssh -p $PORT" "root@$HOST:$REMOTE/data/prompts/" data/prompts_${TAG}/
echo "verifying tier-1 checksums (every line must say OK; the full shards stay on the instance, hashed in SHA256SUMS.shards)"
fail=0; for R in runs/${TAG}_*/; do ( cd $R && sha256sum -c SHA256SUMS.tier1 ) || fail=1; done
[ $fail = 0 ] && echo "ALL TIER-1 FILES OK — safe to destroy the instance" || echo "CHECKSUM FAILURES — rerun with rsync --checksum before destroying the instance"
