#!/usr/bin/env bash
# Box driver for GOALS_2026-11-28.md: horizon-token steering and the activation-patching baseline on Qwen3-8B,
# thinking on, reusing the saved 2026-11-25 baseline (no new T0 generation). Same items, readout and split as the
# 2026-11-27 search (box_search.sh), so the numbers sit beside its screen and confirm tables. Stages, each skipped when
# its output exists (resumable), in priority order:
#   1 extract       layers 12 16 20 24, every site incl. horizon_last / horizon_mean (steering/extract.py)
#   2 fit, bundle   on the reserved-horizons `fit` prompts (CPU): caa, trend, probe at both horizon sites;
#                   curve-displace (k = 4) at horizon_last, as a patch subspace
#   3 items         readable items (as 2026-11-27)
#   4 screen        TF32 (own check), the 30 unseen-horizon `test` prompts: patch baselines first (text; full patch
#                   per layer; subspace patch per basis bundle; spans horizon and from_horizon), then the steering
#                   bundles x spans horizon / from_horizon, until SCREEN_END
#   5 select        steering winner (screen.py select) and the patch to confirm (screen.py patch-select)
#   6 confirm patch fp32 on the T0 verification grid: text, full and the chosen subspace patch, K_NULL null bases
#   7 confirm steer fp32: the steering winner and K_NULL covariance nulls
# Usage (inside tmux, under bash -l): box_horizon.sh <deadline unix time>. GPU work ends by the deadline; the caller
# also wraps the job in `timeout`.
set -uo pipefail
DEADLINE=${1:?deadline (unix time)}
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
PY=/venv/main/bin/python
MODEL=Qwen/Qwen3-8B
RUN=Qwen3-8B__t0_s0__think
BANK=$ROOT/prompt_bank/data/t0_s0
BASE=$ROOT/prompt_bank/runs/$RUN
SPLIT=$BANK/splits/reserved_horizons.jsonl
ACT=$ROOT/steering/activations/${RUN}__horizon
OUT=/workspace/results/horizon
FITS=$ROOT/steering/fits/${RUN}__horizon
METHODS=$ROOT/steering/methods/Qwen3-8B
OLD_DRAWS=$ROOT/evaluation/draws/$RUN/Qwen3-8B/L24_prelabel   # only to satisfy --draws in the items stage
LAYERS="16 12 20 24"                         # priority order: the 2026-11-27 screen's best trend layers first
SITES="horizon_last horizon_mean"
K_NULL=${K_NULL:-9}                          # nulls per confirm: rows [:9] of the shared draws (top up to 19 later)
CONFIRM_MIN=${CONFIRM_MIN:-85}               # minutes kept for both confirms (~22 fp32 arms + items/N0)
export PYTHONPATH=$ROOT/prompt_bank:$ROOT/steering
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
stamp() { echo "== $(date '+%H:%M:%S') $*"; }
minutes_left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
dir_of() { echo "$METHODS/${1}__${RUN}__horizon__nonone__L${2}_$3"; }   # method layer site
winner_field() { $PY -c "import json, sys; print(json.load(open(sys.argv[1]))$2)" "$1"; }

stamp "start; deadline $(date -d @"$DEADLINE" '+%H:%M'), $(minutes_left) min left"

# 1 extract ---------------------------------------------------------------------------------------------------------
if [ ! -f "$ACT/extract.json" ]; then
    stamp "download $MODEL"
    $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))" \
        || { stamp "download FAILED"; exit 1; }
    stamp "extract"
    (cd "$ROOT/steering" && $PY -u extract.py --bank "$BANK" --run "$BASE" --layers $LAYERS --split all \
        --out "$ACT") || { stamp "extract FAILED"; exit 1; }
fi

# 2 fit and bundle (CPU) --------------------------------------------------------------------------------------------
fit_one() {   # layer site
    [ -f "$FITS/nonone/L${1}_$2/fit.json" ] && return 0
    (cd "$ROOT/steering" && $PY fit.py --activations "$ACT" --bank "$BANK" --layer "$1" --site "$2" \
        --uids "$SPLIT" --uids-role fit --no-include-none --name nonone > "$OUT/fit_L${1}_$2.log" 2>&1) \
        || echo "fit L$1 $2 FAILED"
}
export -f fit_one
export ROOT PY ACT BANK SPLIT FITS OUT PYTHONPATH
stamp "fit"
for L in $LAYERS; do for S in $SITES; do echo "$L $S"; done; done | xargs -P 8 -L 1 bash -c 'fit_one "$@"' _
stamp "bundle"
for L in $LAYERS; do
    for S in $SITES; do
        methods="trend caa probe"; [ "$S" = horizon_last ] && methods="$methods curve-displace"
        [ -d "$(dir_of trend "$L" "$S")" ] && continue
        (cd "$ROOT/steering" && $PY bundle.py --fit "$FITS/nonone/L${L}_$S" --activations "$ACT" --methods $methods \
            >> "$OUT/bundle.log" 2>&1) || echo "bundle L$L $S FAILED"
    done
done
# steering candidates in priority order (trend first: the horizon direction with the regime part removed), and the
# patch subspaces (every basis at horizon_last)
STEER=(); PATCH=()
for m in trend caa probe; do for L in $LAYERS; do for S in $SITES; do STEER+=("$(dir_of "$m" "$L" "$S")"); done; done; done
for L in $LAYERS; do for m in trend caa probe curve-displace; do PATCH+=("$(dir_of "$m" "$L" horizon_last)"); done; done
# a failed fit or bundle leaves its configs out (listed), rather than stopping a whole screen invocation
keep_existing() { for d in "$@"; do if [ -d "$d" ]; then echo "$d"; else echo "missing bundle $d" >&2; fi; done; }
mapfile -t STEER < <(keep_existing "${STEER[@]}")
mapfile -t PATCH < <(keep_existing "${PATCH[@]}")

# 3 items -----------------------------------------------------------------------------------------------------------
mkdir -p "$OUT/items/methods"
ln -sfn "$(dir_of caa 16 horizon_last)" "$OUT/items/methods/"
if [ ! -f "$OUT/items/items.jsonl" ]; then
    (cd "$ROOT/evaluation" && $PY -u replay_eval.py items --bank "$BANK" --run "$BASE" --activations "$ACT" \
        --methods "$OUT/items/methods" --draws "$OLD_DRAWS" --out "$OUT/items" \
        --extraction-index "$ACT/index.jsonl") || { stamp "items FAILED"; exit 1; }
fi

# 4 screen ----------------------------------------------------------------------------------------------------------
SCREEN_END=$(( DEADLINE - CONFIRM_MIN * 60 ))
screen() { (cd "$ROOT/evaluation" && $PY -u screen.py screen --bank "$BANK" --run "$BASE" --activations "$ACT" \
    --items-from "$OUT/items" --uids "$SPLIT" --uids-role test --write-span horizon from_horizon --tf32 \
    --replay-batch 8 --deadline "$SCREEN_END" --out "$OUT/screen" "$@"); }
stamp "screen: patch baselines, then steering, until $(date -d @"$SCREEN_END" '+%H:%M')"
screen --patch-layers $LAYERS --patch-bundles "${PATCH[@]}" || stamp "screen (patch) FAILED"
screen --bundles "${STEER[@]}" --push-grid 0.2 0.4 0.8 1.6 || stamp "screen (steering) FAILED"

# 5 select ----------------------------------------------------------------------------------------------------------
(cd "$ROOT/evaluation" && $PY screen.py patch-select --out "$OUT/screen" --confirm-out "$OUT/confirm_patch") \
    || stamp "patch-select FAILED"
(cd "$ROOT/evaluation" && $PY screen.py select --out "$OUT/screen" --confirm-out "$OUT/confirm_steer") \
    || stamp "select FAILED"

draws_for() {   # layer site -> draws dir (made once)
    local d=$ROOT/evaluation/draws/${RUN}__horizon/Qwen3-8B/L${1}_$2
    [ -f "$d/SHA256SUMS" ] || (cd "$ROOT/evaluation" && $PY shared_draws.py make --activations "$ACT" \
        --bank "$BANK/prompts.jsonl" --model "$MODEL" --layer "$1" --site "$2" --out "draws/${RUN}__horizon/" >&2)
    echo "$d"
}
confirm() {   # out methods-bundle site span arms...
    local out=$1 bundle=$2 site=$3 span=$4; shift 4
    mkdir -p "$out/methods"; ln -sfn "$bundle" "$out/methods/$(basename "$bundle")"
    local layer; layer=$($PY -c "import json, sys; print(json.load(open(sys.argv[1] + '/method.json'))['layer'])" "$bundle")
    local draws; draws=$(draws_for "$layer" "$site") || { stamp "draws FAILED"; return 1; }
    (cd "$ROOT/evaluation" && timeout $(( $(minutes_left) * 60 )) $PY -u replay_eval.py items calibrate replay \
        --bank "$BANK" --run "$BASE" --activations "$ACT" --methods "$out/methods" --draws "$draws" --out "$out" \
        --extraction-index "$ACT/index.jsonl" --k "$K_NULL" --patch-k "$K_NULL" --replay-batch 8 \
        --replay-dtype float32 --site "$site" --write-span "$span" "$@") || stamp "confirm $out FAILED or timed out"
    (cd "$ROOT/evaluation" && $PY analyze.py --out "$out") || stamp "analyze $out FAILED"
}

# 6 confirm patch (fp32) --------------------------------------------------------------------------------------------
if [ -f "$OUT/screen/patch_choice.json" ] && [ "$(minutes_left)" -gt 20 ]; then
    P_BUNDLE=$(winner_field "$OUT/screen/patch_choice.json" "['basis']['path']")
    P_SPAN=$(winner_field "$OUT/screen/patch_choice.json" "['span']")
    P_K=$($PY -c "import sys; sys.path.insert(0, '$ROOT/evaluation'); import patching; from steer.methods import load
print(len(patching.method_basis(load(sys.argv[1]))))" "$P_BUNDLE")
    stamp "confirm patch: $(basename "$P_BUNDLE") (k = $P_K), span $P_SPAN; $(minutes_left) min left"
    confirm "$OUT/confirm_patch" "$P_BUNDLE" horizon_last "$P_SPAN" --patch --arms patch "patch-null:k$P_K"
fi

# 7 confirm steering winner (fp32) ----------------------------------------------------------------------------------
if [ -f "$OUT/screen/winner.json" ] && [ "$(minutes_left)" -gt 20 ]; then
    W_DIR=$(winner_field "$OUT/screen/winner.json" "['winner']['path']")
    W_S=$(winner_field "$OUT/screen/winner.json" "['winner']['site']")
    W_SPAN=$(winner_field "$OUT/screen/winner.json" "['winner']['span']")
    W_METHOD=$(winner_field "$OUT/screen/winner.json" "['winner']['method']")
    stamp "confirm steering: $W_METHOD $(basename "$W_DIR") span $W_SPAN; $(minutes_left) min left"
    confirm "$OUT/confirm_steer" "$W_DIR" "$W_S" "$W_SPAN" --arms "$W_METHOD" covariance
fi
stamp "EXIT=0; $(minutes_left) min left before the deadline"
