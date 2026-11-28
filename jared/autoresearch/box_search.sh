#!/usr/bin/env bash
# Box driver for GOALS_2026-11-27.md: the pre-registered steering search on Qwen3-8B, thinking on, reusing the saved
# 2026-11-25 baseline (no new T0 generation). Stages, each skipped when its output exists (resumable):
#   1 extract   activations at layers 4..34 step 2, sites label/prelabel/prompt_end/think_end, all splits
#   2 fit       per layer x site x fit set, on the reserved-horizons `fit` prompts only (CPU, parallel)
#   3 bundle    the screen's candidates, in priority order (bundles_<span>.txt)
#   4 screen    on the reserved-horizons `test` prompts (unseen horizons), TF32 if its own check passes
#   5 select    winner.json, and the confirm stage's calibration.json (strength fixed by the screen)
#   6 confirm   fp32 replay of the winner and K = 19 nulls at its layer/site/span on the T0 verification grid
#   7 gencheck  steered generation on a subset (if time)
#   (transfer of the frozen winner is not in this script: transfer_eval.py still needs a caa bundle and has no
#   --write-span; done after the run if the winner allows it)
# Usage (inside tmux, under bash -l): box_search.sh <deadline unix time>. GPU work ends by the deadline, so the
# watchdog stops the box even if nobody is watching; the whole job also runs under `timeout` (see the caller).
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
ACT=$ROOT/steering/activations/${RUN}__layers
OUT=/workspace/results/search
FITS=$ROOT/steering/fits/${RUN}__layers
METHODS=$ROOT/steering/methods/Qwen3-8B
OLD_DRAWS=$ROOT/evaluation/draws/$RUN/Qwen3-8B/L24_prelabel   # only to satisfy --draws in the items stage
SCREEN_LAYERS="20 16 24 28 12"              # priority order (GOALS: Design)
CONFIRM_MIN=${CONFIRM_MIN:-215}             # minutes kept for confirm: 20 fp32 arms x ~10 min, + items/N0
export PYTHONPATH=$ROOT/prompt_bank:$ROOT/steering
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
stamp() { echo "== $(date '+%H:%M:%S') $*"; }
minutes_left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }

stamp "start; deadline $(date -d @"$DEADLINE" '+%H:%M'), $(minutes_left) min left"

# 1 extract ---------------------------------------------------------------------------------------------------------
if [ ! -f "$ACT/extract.json" ]; then
    # extract.py refuses weights other than the baseline run's revision (read from the cached `main`), and a fresh box
    # has none cached: fetch `main` and say which commit it is
    stamp "download $MODEL"
    $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))" \
        || { stamp "download FAILED"; exit 1; }
    stamp "extract"
    (cd "$ROOT/steering" && $PY -u extract.py --bank "$BANK" --run "$BASE" --layers $(seq 4 2 34) --split all \
        --out "$ACT") || { stamp "extract FAILED"; exit 1; }
fi

# 2 fit (CPU) -------------------------------------------------------------------------------------------------------
fit_one() {   # layer site name extra-args...
    local L=$1 S=$2 name=$3; shift 3
    [ -f "$FITS/$name/L${L}_$S/fit.json" ] && return 0
    (cd "$ROOT/steering" && $PY fit.py --activations "$ACT" --bank "$BANK" --layer "$L" --site "$S" \
        --uids "$SPLIT" --uids-role fit --no-include-none --name "$name" "$@" \
        > "$OUT/fit_${name}_L${L}_$S.log" 2>&1) || echo "fit $name L$L $S FAILED"
}
export -f fit_one
export ROOT PY ACT BANK SPLIT FITS OUT PYTHONPATH
stamp "fit"
{
    for L in $SCREEN_LAYERS; do for S in prompt_end prelabel; do
        echo "$L $S nonone"
        echo "$L $S regime --caa-long-min 1.0 --caa-short-max 0.999"
        [ "$S" = prompt_end ] && echo "$L $S nonone-k8 --components 8"
    done; done
} | xargs -P 8 -L 1 bash -c 'fit_one "$@"' _

# 3 bundle: candidates in priority order ----------------------------------------------------------------------------
bundle() {   # name layer site methods...
    local name=$1 L=$2 S=$3; shift 3
    (cd "$ROOT/steering" && $PY bundle.py --fit "$FITS/$name/L${L}_$S" --activations "$ACT" --methods "$@" \
        >> "$OUT/bundle.log" 2>&1) || echo "bundle $name L$L $S FAILED"
}
dir_of() { echo "$METHODS/${1}__${RUN}__layers__${2}__L${3}_$4"; }   # method fitname layer site (checked below)
: > "$OUT/bundles_prompt.txt"; : > "$OUT/bundles_all.txt"
stamp "bundle"
for L in $SCREEN_LAYERS; do
    bundle nonone "$L" prompt_end caa trend probe curve-displace
    bundle regime "$L" prompt_end caa
    bundle nonone-k8 "$L" prompt_end curve-displace
    bundle nonone "$L" prelabel caa trend probe
    bundle regime "$L" prelabel caa
    [ "$L" = 24 ] && { bundle nonone 24 prompt_end choice; bundle nonone 24 prelabel choice; }
    for spec in "trend nonone prompt_end" "caa regime prompt_end" "caa nonone prompt_end" "curve-displace nonone prompt_end" \
                "curve-displace nonone-k8 prompt_end" "probe nonone prompt_end"; do
        set -- $spec
        d=$(dir_of "$1" "$2" "$L" "$3"); echo "$d" >> "$OUT/bundles_prompt.txt"; echo "$d" >> "$OUT/bundles_all.txt"
    done
    for spec in "trend nonone prelabel" "caa regime prelabel" "caa nonone prelabel" "probe nonone prelabel"; do
        set -- $spec; dir_of "$1" "$2" "$L" "$3" >> "$OUT/bundles_all.txt"
    done
    [ "$L" = 24 ] && { dir_of choice nonone 24 prompt_end >> "$OUT/bundles_all.txt"
                       dir_of choice nonone 24 prelabel >> "$OUT/bundles_all.txt"; }
done
missing=$(while read -r d; do [ -d "$d" ] || echo "$d"; done < "$OUT/bundles_all.txt")
[ -n "$missing" ] && { echo "missing bundles:"; echo "$missing"; }

# 4 screen ----------------------------------------------------------------------------------------------------------
SCREEN_END=$(( DEADLINE - CONFIRM_MIN * 60 ))
stamp "screen until $(date -d @"$SCREEN_END" '+%H:%M')"
# readable items (the items stage loads the model and wants a one-layer methods dir: give it one bundle)
mkdir -p "$OUT/items/methods"
ln -sfn "$(dir_of caa nonone 24 prompt_end)" "$OUT/items/methods/"
if [ ! -f "$OUT/items/items.jsonl" ]; then
    (cd "$ROOT/evaluation" && $PY -u replay_eval.py items --bank "$BANK" --run "$BASE" --activations "$ACT" \
        --methods "$OUT/items/methods" --site prompt_end --draws "$OLD_DRAWS" --out "$OUT/items" \
        --extraction-index "$ACT/index.jsonl") || { stamp "items FAILED"; exit 1; }
fi
# span `prompt` first for prompt_end methods (the patch reading), then every candidate with span `all`
for span in prompt all; do
    [ -f "$OUT/screen/winner.json" ] && break
    (cd "$ROOT/evaluation" && $PY -u screen.py screen --bank "$BANK" --run "$BASE" --activations "$ACT" \
        --items-from "$OUT/items" --uids "$SPLIT" --uids-role test --bundles $(grep -v '^$' "$OUT/bundles_$span.txt") \
        --write-span "$span" --tf32 --replay-batch 8 --deadline "$SCREEN_END" --out "$OUT/screen") \
        || stamp "screen ($span) FAILED"
done

# 5 select ----------------------------------------------------------------------------------------------------------
[ -f "$OUT/screen/winner.json" ] || (cd "$ROOT/evaluation" && $PY screen.py select --out "$OUT/screen" --confirm-out "$OUT/confirm") \
    || { stamp "select FAILED"; exit 1; }
W_DIR=$($PY -c "import json; print(json.load(open('$OUT/screen/winner.json'))['winner']['path'])")
W_L=$($PY -c "import json; print(json.load(open('$OUT/screen/winner.json'))['winner']['layer'])")
W_S=$($PY -c "import json; print(json.load(open('$OUT/screen/winner.json'))['winner']['site'])")
W_SPAN=$($PY -c "import json; print(json.load(open('$OUT/screen/winner.json'))['winner']['span'])")
W_METHOD=$($PY -c "import json; print(json.load(open('$OUT/screen/winner.json'))['winner']['method'])")
stamp "winner: $W_METHOD L$W_L $W_S span $W_SPAN ($W_DIR); $(minutes_left) min left"

# 6 confirm (fp32) --------------------------------------------------------------------------------------------------
DRAWS=$ROOT/evaluation/draws/${RUN}__layers/Qwen3-8B/L${W_L}_$W_S
if [ ! -f "$DRAWS/SHA256SUMS" ]; then
    (cd "$ROOT/evaluation" && $PY shared_draws.py make --activations "$ACT" --bank "$BANK/prompts.jsonl" --model "$MODEL" \
        --layer "$W_L" --site "$W_S" --out "draws/${RUN}__layers/") || { stamp "draws FAILED"; exit 1; }
fi
mkdir -p "$OUT/confirm/methods"
ln -sfn "$W_DIR" "$OUT/confirm/methods/$(basename "$W_DIR")"
if [ "$(minutes_left)" -gt 30 ]; then
    stamp "confirm"
    # pre-registered nulls: covariance-matched push draws for a push winner; for a curve winner (no push strength) its
    # label-shuffled refits, the curve null family replay_eval already has
    NULLS=covariance; case $W_METHOD in curve-*) NULLS="shuffled:$W_METHOD";; esac
    (cd "$ROOT/evaluation" && timeout $(( $(minutes_left) * 60 )) $PY -u replay_eval.py items calibrate replay \
        --bank "$BANK" --run "$BASE" --activations "$ACT" --methods "$OUT/confirm/methods" --draws "$DRAWS" \
        --out "$OUT/confirm" --extraction-index "$ACT/index.jsonl" --k 19 --replay-batch 8 --replay-dtype float32 \
        --site "$W_S" --write-span "$W_SPAN" --arms "$W_METHOD" $NULLS) || stamp "confirm FAILED or timed out"
    (cd "$ROOT/evaluation" && $PY analyze.py --out "$OUT/confirm")
    $PY "$ROOT/evaluation/diagnose.py" "$OUT/confirm" --activations "$ACT" --methods "$OUT/confirm/methods" \
        --bank "$BANK" --baseline "$BASE" || echo "diagnostics failed"
fi

# 7 gencheck and 8 transfer: only with time left --------------------------------------------------------------------
if [ "$(minutes_left)" -gt 25 ]; then
    stamp "gencheck"
    (cd "$ROOT/evaluation" && timeout $(( ($(minutes_left) - 5) * 60 )) $PY -u replay_eval.py gencheck \
        --bank "$BANK" --run "$BASE" --activations "$ACT" --methods "$OUT/confirm/methods" --draws "$DRAWS" \
        --out "$OUT/confirm" --extraction-index "$ACT/index.jsonl" --site "$W_S" --write-span "$W_SPAN" \
        --gen-arms "$W_METHOD" --gen-pairs 8) || stamp "gencheck FAILED or timed out"
fi
stamp "EXIT=0; $(minutes_left) min left before the deadline"
