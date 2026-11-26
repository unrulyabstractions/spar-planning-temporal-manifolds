#!/usr/bin/env bash
# Follow-up to box_steer_diag.sh (2026-11-26). One replay arm takes ~10 min in float32 on the A100 (178 items x 2
# signs, ~2.5k tokens), so 19 nulls per configuration (~3.3 h) fit the budget only once. This script waits for the
# first configuration (caa + choice at prelabel, 19 nulls) to finish, stops the original driver, then runs:
#   prelabel_regime    calibrate; if its strength equals the first configuration's (same site, same draws), reuse
#                      that configuration's 19 covariance arms, else replay a 5-null screen
#   prompt_end_regime  a 5-null screen (smallest possible p = 1/6: a screen, not a test)
# Cache reuse is safe only for the covariance arms: arm files are keyed by arm name alone (replay_eval.py, the
# stale-cache issue in jared/01_plan.md §5), and both configurations also have an arm named "caa".
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
PY=/venv/main/bin/python
RUN=Qwen3-8B__t0_s0__think
RESULTS=/workspace/results/steer_diag
LOG=$RESULTS/run.log
export PYTHONPATH=$ROOT/prompt_bank:$ROOT/steering
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stamp() { echo "== $(date '+%H:%M:%S') $*"; }

replay() {   # name, site, k, stages, bundle directory names...
    local name=$1 site=$2 k=$3 stages=$4
    shift 4
    local out=$RESULTS/$name
    mkdir -p "$out/methods"
    for d in "$@"; do ln -sfn "$ROOT/steering/methods/Qwen3-8B/$d" "$out/methods/$d"; done
    stamp "$name: $stages at $site, K = $k"
    (cd "$ROOT/evaluation" && $PY -u replay_eval.py $stages --bank "$ROOT/prompt_bank/data/t0_s0" \
        --run "$ROOT/prompt_bank/runs/$RUN" --activations "$ROOT/steering/activations/$RUN" --methods "$out/methods" \
        --draws "$ROOT/evaluation/draws/$RUN/Qwen3-8B/L24_$site" --out "$out" \
        --extraction-index "$ROOT/steering/activations/$RUN/index.jsonl" --k "$k" --replay-batch "${BATCH:-8}" \
        --replay-dtype float32 --site "$site" --arms caa covariance --calib-rule peak)
}

finish() {   # name
    local out=$RESULTS/$1
    (cd "$ROOT/evaluation" && $PY analyze.py --out "$out")
    $PY "$ROOT/evaluation/diagnose.py" "$out" --activations "$ROOT/steering/activations/$RUN" --methods "$out/methods" \
        --bank "$ROOT/prompt_bank/data/t0_s0" --baseline "$ROOT/prompt_bank/runs/$RUN" || echo "diagnostics failed"
    stamp "$1 done"
}

stamp "waiting for prelabel_default to finish"
until grep -q "prelabel_default done" "$LOG"; do sleep 20; done
tmux kill-session -t steer 2>/dev/null && stamp "stopped the original driver"
rm -rf "$RESULTS/prelabel_regime"   # it may have started there; nothing in it is reused

REGIME_PRE="caa__${RUN}__regime__L24_prelabel"
replay prelabel_regime prelabel 19 "items calibrate" "$REGIME_PRE" || { stamp "prelabel_regime calibrate FAILED"; exit 1; }
x_first=$($PY -c "import json; print(json.load(open('$RESULTS/prelabel_default/calibration.json'))['x_push'])")
x_regime=$($PY -c "import json; print(json.load(open('$RESULTS/prelabel_regime/calibration.json'))['x_push'])")
if [ "$x_first" = "$x_regime" ]; then
    stamp "same strength ($x_regime): reusing the 19 covariance arms and the unsteered pass"
    mkdir -p "$RESULTS/prelabel_regime/arms"
    for i in $(seq 0 18); do
        f=$($PY -c "import hashlib; print(hashlib.sha1(b'covariance[$i]').hexdigest()[:12])").npz
        cp "$RESULTS/prelabel_default/arms/$f" "$RESULTS/prelabel_regime/arms/$f"
    done
    cp "$RESULTS/prelabel_default/arms/_none.npz" "$RESULTS/prelabel_regime/arms/_none.npz"
    replay prelabel_regime prelabel 19 replay "$REGIME_PRE" || { stamp "prelabel_regime FAILED"; exit 1; }
else
    stamp "strength differs ($x_first vs $x_regime): 5-null screen"
    rm -rf "$RESULTS/prelabel_regime"
    replay prelabel_regime prelabel 5 "items calibrate replay" "$REGIME_PRE" || { stamp "prelabel_regime FAILED"; exit 1; }
fi
finish prelabel_regime

replay prompt_end_regime prompt_end 5 "items calibrate replay" "caa__${RUN}__regime__L24_prompt_end" \
    || { stamp "prompt_end_regime FAILED"; exit 1; }
finish prompt_end_regime
stamp "EXIT=0"
