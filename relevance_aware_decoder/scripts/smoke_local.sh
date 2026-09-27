#!/bin/bash
# Plumbing test on the local 8 GB GPU before renting anything: a small Qwen3 on the smoke subset
# (1 configuration per split, every family and template group, ~600 prompts). Checks that prompts,
# capture, twins, evaluation and the summary all work end to end. The numbers mean nothing.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# Qwen3-1.7B has 28 decoder layers, so the 14B layer lists are replaced.
SMOKE=1 MODEL=${MODEL:-Qwen/Qwen3-1.7B} BATCH=${BATCH:-8} LAYERS=${LAYERS:-6,10,14,18,22,26} SUBSET_LAYERS=${SUBSET_LAYERS:-14,22} \
  bash "$HERE/run_pipeline.sh"
