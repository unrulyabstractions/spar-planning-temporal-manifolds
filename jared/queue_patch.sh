#!/usr/bin/env bash
# Box queue: the phrase number patch on Qwen3-8B with thinking on (27_spec_phrase_number_patch.md).
# 24 same-unit crossing pairs, 13 units (baseline, rerun, conditions 1-9 at L16, condition 10 at L12 and
# L22): 312 generations, about 3 h on one RTX 4090 (a 12-pair batch took ~400 s in the scan; each unit
# is two batches). Units are saved as they finish, in the order of the spec, so a stopped run is usable;
# to drop conditions 9 and 10 (~2 h) stop it after "subspace number L16".
#   1. inputs      patch_prepare.py ran locally and `just push` shipped steering/exp16_Qwen3-8B_patch
#                  (inputs.json, inputs.npz); they are copied to $DIR. Without them the queue stops
#   2. run         exp16_phrase_patch.py: baseline, rerun, then the conditions in spec order
#   3. summary     patch_summary.py: patch_summary.txt (a screen, no significance claim)
# Logs in $OUT/logs/; a rerun skips finished work. Does not stop the box.
#
#   cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
#   tmux new -s patch 'bash queue_patch.sh; bash'
#
# Watch: tail -f /workspace/results/horizon/logs/queue_patch.log

set -u
PY=${PY:-/venv/main/bin/python}
OUT=${OUT:-/workspace/results/horizon}
MODEL=Qwen/Qwen3-8B
BATCH=${BATCH:-12}   # generation batch; 24 pairs fill two batches
LOGS=$OUT/logs
SRC=steering/exp16_Qwen3-8B_patch
DIR=$OUT/exp16_Qwen3-8B_patch
mkdir -p "$LOGS" "$DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOGS/queue_patch.log"; }

step() {   # step <name> <command...>: run once, log to <name>.log, mark <name>.done on success
    local name=$1
    shift
    if [ -e "$LOGS/$name.done" ]; then
        log "$name: done earlier, skipped"
        return
    fi
    log "$name: start"
    if "$@" > "$LOGS/$name.log" 2>&1; then
        touch "$LOGS/$name.done"
        log "$name: ok"
    else
        log "$name: FAILED, last lines of $LOGS/$name.log:"
        tail -n 20 "$LOGS/$name.log" | tee -a "$LOGS/queue_patch.log"
        return 1
    fi
}

log "queue on $(nvidia-smi -L | wc -l) GPU(s), out $OUT"
log "$($PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])' 2>&1)"
step patch_download $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" || exit 1
if [ ! -e "$DIR/inputs.json" ] || [ ! -e "$DIR/inputs.npz" ]; then
    if [ -e "$SRC/inputs.json" ] && [ -e "$SRC/inputs.npz" ]; then
        cp "$SRC/inputs.json" "$SRC/inputs.npz" "$SRC/fit_report.txt" "$DIR/"
        log "inputs copied from $SRC"
    else
        log "no inputs: run analysis/patch_prepare.py locally and push $SRC"
        exit 1
    fi
fi
rm -rf "$DIR"/conditions/*.lock   # claims left by a stopped run; finished units are kept

step patch_run env CUDA_VISIBLE_DEVICES=0 $PY experiments/exp16_phrase_patch.py --dir "$DIR" --batch-size "$BATCH" || exit 1
# not a step: a rerun after more units finish should redo the summary
$PY analysis/patch_summary.py "$DIR" > "$LOGS/patch_summary.log" 2>&1 || log "summary FAILED (see patch_summary.log)"
log "queue finished; $DIR/patch_summary.txt:"
tee -a "$LOGS/queue_patch.log" < "$DIR/patch_summary.txt"
