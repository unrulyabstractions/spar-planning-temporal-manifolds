#!/usr/bin/env bash
# The whole loop on one local GPU: baseline -> extract -> fit -> bundle -> hook gate -> shared draws -> replay
# evaluation -> generation check -> report card. Every stage skips what already exists, so a rerun resumes.
#
#   evaluation/run_local.sh                         # defaults below
#   THINK=1 CAP=4096 BATCH=4 TAG=think evaluation/run_local.sh
#
# Settings (environment): MODEL, LAYER, THINK (0/1), CAP (max new tokens), BATCH, TAG (run name suffix), K (draws per
# null family), PY (python), STAGES (which runner stages), GEN_PAIRS, GEN_SAMPLES (generation-check passes on a
# sampled baseline), DECODING (baseline: auto | greedy | sample; auto samples when thinking is on; an older greedy run
# resumes only with DECODING=greedy).
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-1.7B}
LAYER=${LAYER:-18}
THINK=${THINK:-0}
CAP=${CAP:-512}
BATCH=${BATCH:-16}
TAG=${TAG:-nothink}
K=${K:-19}
GEN_PAIRS=${GEN_PAIRS:-16}
GEN_SAMPLES=${GEN_SAMPLES:-1}
DECODING=${DECODING:-auto}
STAGES=${STAGES:-"items calibrate replay gencheck"}
PY=${PY:-$(cd "$(dirname "$0")/.." && pwd)/../PTM_workspace/ptm_env/bin/python}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SHORT=${MODEL##*/}
RUN=${SHORT}__t0_s0__${TAG}
BANK=$ROOT/prompt_bank/data/t0_s0
RUN_DIR=$ROOT/prompt_bank/runs/$RUN
ACT=$ROOT/steering/activations/$RUN
FIT=$ROOT/steering/fits/$RUN/default/L${LAYER}_label
METHODS=$ROOT/steering/methods/$SHORT
DRAWS=$ROOT/evaluation/draws/$RUN/$SHORT/L${LAYER}_label   # per run: covariance/real_diff depend on the activations
OUT=$ROOT/evaluation/results/$RUN
THINKFLAG=$([ "$THINK" = 1 ] && echo "" || echo "--no-thinking")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stamp() { echo "== $(date '+%H:%M:%S') $*"; }

# A stage is done only when its completion file exists. A directory left by an interrupted stage is moved aside (never
# deleted) and the stage reruns: extraction's tools refuse to write into an existing directory.
set_aside() { [ -e "$1" ] && { echo "   incomplete: moving $1 aside"; mv "$1" "$1.partial-$(date +%Y%m%d-%H%M%S)"; }; true; }
baseline_done() {   # every prompt of the bank has a row (the baseline itself resumes batch by batch)
    [ -f "$RUN_DIR/responses.jsonl" ] && $PY -c "
import json,sys
bank={json.loads(l)['uid'] for l in open('$BANK/prompts.jsonl')}
got={json.loads(l)['uid'] for l in open('$RUN_DIR/responses.jsonl')}
sys.exit(0 if bank<=got else 1)"
}

stamp "baseline $RUN (thinking=$THINK, cap=$CAP, batch=$BATCH)"
baseline_done || (cd "$ROOT/prompt_bank" && $PY -u baseline.py --bank "$BANK" --model "$MODEL" --out "$RUN_DIR" \
    --batch-size "$BATCH" --max-new-tokens "$CAP" --decoding "$DECODING" $THINKFLAG)
baseline_done || { echo "baseline incomplete"; exit 1; }

stamp "extract (all splits: the runner reads the verification grid's label positions from the baseline)"
if [ ! -f "$ACT/extract.json" ] || [ ! -f "$ACT/L${LAYER}_label.npy" ]; then
    set_aside "$ACT"
    (cd "$ROOT/steering" && $PY extract.py --bank "$BANK" --run "$RUN_DIR" --layers "$LAYER" --split all --out "$ACT")
fi

stamp "fit"
if [ ! -f "$FIT/fit.json" ] || [ ! -f "$FIT/manifold.npz" ] || [ ! -f "$FIT/directions.npz" ] || [ ! -f "$FIT/fit_uids.jsonl" ]; then
    set_aside "$FIT"
    (cd "$ROOT/steering" && $PY fit.py --activations "$ACT" --bank "$BANK" --layer "$LAYER")
fi

stamp "bundle"
MISSING=()
for m in caa reward choice curve-snap curve-displace; do
    d=$METHODS/${m}__${RUN}__default__L${LAYER}_label
    [ -f "$d/method.json" ] || { set_aside "$d"; MISSING+=("$m"); }
done
[ ${#MISSING[@]} -eq 0 ] || (cd "$ROOT/steering" && $PY bundle.py --fit "$FIT" --activations "$ACT" --methods "${MISSING[@]}")
METHODS_RUN=$ROOT/evaluation/results/$RUN/methods
mkdir -p "$METHODS_RUN"
for d in "$METHODS"/*__"${RUN}"__default__L"${LAYER}"_label; do ln -sfn "$d" "$METHODS_RUN/$(basename "$d")"; done

stamp "hook gate with the CAA bundle"
CAA=$(ls -d "$METHODS"/caa__"${RUN}"__default__L"${LAYER}"_label)
[ -f "$OUT/hook_gate.ok" ] || { (cd "$ROOT/steering" && $PY tools/check_hook.py --method "$CAA" --bank "$BANK" --n 4 \
    --max-new-tokens 64) && touch "$OUT/hook_gate.ok"; }

stamp "shared draws"
[ -f "$DRAWS/SHA256SUMS" ] || { set_aside "$DRAWS"; (cd "$ROOT/evaluation" && $PY shared_draws.py make --activations "$ACT" --bank "$BANK/prompts.jsonl" \
    --model "$MODEL" --layer "$LAYER" --site label --out "draws/$RUN/"); }

stamp "replay evaluation: $STAGES"
(cd "$ROOT/evaluation" && $PY -u replay_eval.py $STAGES --bank "$BANK" --run "$RUN_DIR" --activations "$ACT" \
    --methods "$METHODS_RUN" --draws "$DRAWS" --out "$OUT" --extraction-index "$ACT/index.jsonl" --k "$K" --gen-pairs "$GEN_PAIRS" --gen-batch "$BATCH" --gen-samples "$GEN_SAMPLES")

stamp "report card"
(cd "$ROOT/evaluation" && $PY analyze.py --out "$OUT")
stamp "done: $OUT/report.md"
