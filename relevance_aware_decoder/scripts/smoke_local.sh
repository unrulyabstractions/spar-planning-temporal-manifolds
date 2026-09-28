#!/bin/bash
# Plumbing test on the local 8 GB GPU before renting anything: a small Qwen3 on the smoke subset
# (1 configuration per split, every family and template group, ~600 prompts). Checks that prompts,
# capture, twins, subset export, evaluation, manifest and checksums all work end to end. The numbers mean nothing.
# Replaces the previous smoke run (OVERWRITE=1 by default here; the full pipeline refuses to overwrite).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# Qwen3-1.7B has 28 decoder layers, so the 14B layer list is replaced (the subset exports the same layers).
SMOKE=1 OVERWRITE=${OVERWRITE:-1} MODEL=${MODEL:-Qwen/Qwen3-1.7B} BATCH=${BATCH:-8} LAYERS=${LAYERS:-6,10,14,18,22,26} \
  bash "$HERE/run_pipeline.sh"
