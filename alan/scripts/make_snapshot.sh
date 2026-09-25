#!/bin/bash
# Assemble a self-consistent snapshot of this repo's code, prompt datasets, run indices, figures and
# result tables into DEST (e.g. ~/spar-planning-temporal-manifolds/alan). Copies code from the
# committed tree (git archive HEAD), never from the working tree, and refuses to run on a dirty tree.
# Full activations (runs/*/acts_????.safetensors, ~14 GB per run) are NOT copied unless --with-acts RUN is
# given; compact subsets written by scripts/export_subset.py (subset.json + acts_subset_*) ARE copied.
# Usage: scripts/make_snapshot.sh DEST [--with-acts RUN_NAME ...]
set -eu
cd "$(dirname "$0")/.."
DEST=${1:?DEST}; shift || true
if [ -n "$(git status --porcelain)" ]; then echo "working tree is dirty; commit first" >&2; exit 1; fi
HASH=$(git rev-parse --short HEAD)
mkdir -p "$DEST"
echo "snapshot of CodeReclaimers/SPAR-2026 @ $HASH -> $DEST"
# 1. committed code and docs
git archive HEAD | tar -x -C "$DEST"
# 2. regenerable prompt datasets (small) and run indices / metadata (small), figures and tables
mkdir -p "$DEST/data/prompts" "$DEST/runs" "$DEST/figures"
cp -r data/prompts/. "$DEST/data/prompts/"
for r in runs/*/; do
  n=$(basename "$r"); mkdir -p "$DEST/runs/$n"
  cp "$r/index.parquet" "$r/meta.json" "$DEST/runs/$n/"
  # shareable activation subsets (scripts/export_subset.py), if present
  [ -f "$r/subset.json" ] && cp "$r/subset.json" "$r"/acts_subset_*.safetensors "$DEST/runs/$n/"
  [ -f "runs/$n.log" ] && cp "runs/$n.log" "$DEST/runs/"
  [ -f "runs/$n.analysis.log" ] && cp "runs/$n.analysis.log" "$DEST/runs/"
done
cp runs/reproduce_all.log "$DEST/runs/" 2>/dev/null || true
cp -r figures/. "$DEST/figures/"
# share one plotly.min.js instead of one per figure directory
first=$(ls "$DEST"/figures/*/plotly.min.js 2>/dev/null | head -1)
if [ -n "$first" ]; then
  cp "$first" "$DEST/figures/plotly.min.js"
  for f in "$DEST"/figures/*/plotly.min.js; do rm "$f"; done
  for h in "$DEST"/figures/*/*.html; do sed -i 's|src="plotly.min.js"|src="../plotly.min.js"|' "$h"; done
fi
# 3. optional activations
while [ $# -gt 0 ]; do
  case "$1" in --with-acts) shift; cp runs/"$1"/acts_*.safetensors "$DEST/runs/$1/"; echo "copied activations for $1";; esac; shift
done
# 4. manifest
{
  echo "snapshot_commit: $HASH"; echo "snapshot_date: $(date -u +%FT%TZ)"; echo "source: https://github.com/CodeReclaimers/SPAR-2026"
  echo "runs:"; for r in runs/*/; do n=$(basename "$r"); echo "  - $n: $(python3 -c "import json;m=json.load(open('$r/meta.json'));print(m['model_name'], m.get('n_prompts'), 'prompts;', m.get('started','?'), '->', m.get('finished','?'))")"; done
  echo "activation_subsets: $(ls "$DEST"/runs/*/acts_subset_*.safetensors 2>/dev/null | wc -l) files ($(du -ch "$DEST"/runs/*/acts_subset_*.safetensors 2>/dev/null | tail -1 | cut -f1))"
  for j in "$DEST"/runs/*/subset.json; do [ -f "$j" ] && echo "  - $(basename "$(dirname "$j")"): $(python3 -c "import json;m=json.load(open('$j'));print('layers', m['layers'], 'positions', m['positions'])")"; done
  echo "full_activations_included: $(ls "$DEST"/runs/*/acts_????.safetensors 2>/dev/null | wc -l) shard files"
} > "$DEST/SNAPSHOT.yaml"
du -sh "$DEST" | cut -f1 | xargs -I{} echo "snapshot size {}"
