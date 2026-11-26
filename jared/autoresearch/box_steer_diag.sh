#!/usr/bin/env bash
# Box driver for jared/SteeringQwen8B.md v2 (2026-11-26): float32 replay of three configurations on the saved
# 2026-11-25 Qwen3-8B thinking-on baseline (no generation). The CPU-side inputs were made locally and ship with
# `just push` (gitignored, but not excluded from the push):
#   steering/activations/<RUN>, prompt_bank/runs/<RUN>        copies of ~/SPAR/results/qwen8b_quick_20261125/
#   steering/methods/Qwen3-8B/<bundle>                         fit.py + bundle.py at prelabel / prompt_end
#   evaluation/draws/<RUN>/Qwen3-8B/L24_{prelabel,prompt_end}  shared_draws.py make
# Configurations, cheapest question first (stop after the first if `choice` does not beat the nulls):
#   prelabel_default   caa + choice (positive control) at the token before the answer letter
#   prelabel_regime    regime direction (both minus near_only, `none` left out) at prelabel
#   prompt_end_regime  the same direction fitted at the last prompt token, before the reasoning
# Run from the checkout root, under `bash -l`, inside tmux. Outputs go to /workspace/results (what `just sync` pulls).
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
PY=/venv/main/bin/python
RUN=Qwen3-8B__t0_s0__think
RESULTS=/workspace/results/steer_diag
export PYTHONPATH=$ROOT/prompt_bank:$ROOT/steering
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stamp() { echo "== $(date '+%H:%M:%S') $*"; }

replay() {   # name, site, arms, bundle directory names...
    local name=$1 site=$2 arms=$3
    shift 3
    local out=$RESULTS/$name
    local methods=$out/methods
    mkdir -p "$methods"
    for d in "$@"; do ln -sfn "$ROOT/steering/methods/Qwen3-8B/$d" "$methods/$d"; done
    stamp "$name: replay at $site, arms: $arms"
    (cd "$ROOT/evaluation" && $PY -u replay_eval.py items calibrate replay --bank "$ROOT/prompt_bank/data/t0_s0" \
        --run "$ROOT/prompt_bank/runs/$RUN" --activations "$ROOT/steering/activations/$RUN" --methods "$methods" \
        --draws "$ROOT/evaluation/draws/$RUN/Qwen3-8B/L24_$site" --out "$out" \
        --extraction-index "$ROOT/steering/activations/$RUN/index.jsonl" --k 19 --replay-batch "${BATCH:-8}" \
        --replay-dtype float32 --site "$site" --arms $arms --calib-rule peak) || { stamp "$name FAILED"; return 1; }
    (cd "$ROOT/evaluation" && $PY analyze.py --out "$out")
    $PY "$ROOT/evaluation/diagnose.py" "$out" --activations "$ROOT/steering/activations/$RUN" --methods "$methods" \
        --bank "$ROOT/prompt_bank/data/t0_s0" --baseline "$ROOT/prompt_bank/runs/$RUN" || echo "diagnostics failed"
    stamp "$name done"
}

replay prelabel_default prelabel "caa choice covariance" \
    "caa__${RUN}__default__L24_prelabel" "choice__${RUN}__default__L24_prelabel" || exit 1
replay prelabel_regime prelabel "caa covariance" "caa__${RUN}__regime__L24_prelabel" || exit 1
replay prompt_end_regime prompt_end "caa covariance" "caa__${RUN}__regime__L24_prompt_end" || exit 1
stamp "EXIT=0"
