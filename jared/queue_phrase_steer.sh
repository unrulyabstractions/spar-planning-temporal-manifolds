#!/usr/bin/env bash
# Box queue: E14, steering at the horizon phrase with thinking on (21_spec_phrase_steering.md), Qwen3-8B.
# One session, no pause (Jared, 2026-10-31):
#   1. extraction  exp14_phrase_states.py: one prefill pass over the pooled prompts, phrase states at L12-L18
#   2. fit         steering_prepare.py --phrase (CPU): the per-layer, per-role fit, targets, donors, steps;
#                  fit_report.txt lands before the run starts, so it can be synced and read while it goes
#   3. run         exp14_phrase_steer.py: 1,216 generations, one transition at a time (about 4.4 h on one 5090)
#   4. analysis    exp14_analysis.py: summary.txt
# The prompts file (steering/phrase_prompts_Qwen3-8B.json) was prepared locally and ships with `just push`,
# so this needs only the model. With several GPUs, GPU 0 runs steps 1-2 while the others wait for the
# inputs, then every GPU runs the driver; workers claim (transition, condition) units one at a time.
# Logs in $OUT/logs/; a rerun skips finished work. Does not stop the box.
#
#   cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
#   tmux new -s phrase 'bash queue_phrase_steer.sh; bash'
#
# Watch: tail -f /workspace/results/horizon/logs/queue_phrase_steer.log

set -u
PY=${PY:-/venv/main/bin/python}
OUT=${OUT:-/workspace/results/horizon}
MODEL=Qwen/Qwen3-8B
PROMPTS=steering/phrase_prompts_Qwen3-8B.json
BATCH=${BATCH:-12}   # generation batch, as stage B on the 5090
LOGS=$OUT/logs
DIR=$OUT/exp14_Qwen3-8B_phrase
N=$(nvidia-smi -L | wc -l)
mkdir -p "$LOGS"

log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOGS/queue_phrase_steer.log"; }

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
        tail -n 20 "$LOGS/$name.log" | tee -a "$LOGS/queue_phrase_steer.log"
        return 1
    fi
}

prepare() {   # extraction on GPU $1, then the fit on CPU
    step e14_states env CUDA_VISIBLE_DEVICES="$1" $PY experiments/exp14_phrase_states.py \
        --prompts "$PROMPTS" --out "$OUT" || return 1
    step e14_fit $PY analysis/steering_prepare.py --phrase "$DIR" || return 1
    log "fit report:"
    tee -a "$LOGS/queue_phrase_steer.log" < "$DIR/fit_report.txt"
}

worker() {   # one driver copy on GPU $1; it exits when no unit is left to claim
    log "driver on GPU $1: start"
    env CUDA_VISIBLE_DEVICES="$1" $PY experiments/exp14_phrase_steer.py --dir "$DIR" --batch-size "$BATCH" \
        > "$LOGS/e14_run_gpu$1.log" 2>&1 \
        && log "driver on GPU $1: finished" || log "driver on GPU $1: FAILED (see e14_run_gpu$1.log)"
}

wait_for_inputs() {   # the other GPUs start once GPU 0 has written the inputs, or give up if it failed
    until [ -e "$DIR/inputs.json" ]; do
        [ -e "$LOGS/prepare.failed" ] && return 1
        sleep 30
    done
}

log "queue on $N GPU(s), out $OUT"
log "$($PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])' 2>&1)"
step e14_download $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" || exit 1
rm -f "$LOGS/prepare.failed"
rm -rf "$DIR"/conditions/*.lock   # claims left by a stopped run; finished units are kept

(prepare 0 || touch "$LOGS/prepare.failed"; [ -e "$LOGS/prepare.failed" ] || worker 0) &
for ((gpu = 1; gpu < N; gpu++)); do
    (wait_for_inputs && worker "$gpu") &
done
wait
if [ -e "$LOGS/prepare.failed" ]; then
    log "queue stopped: extraction or fit failed"
    exit 1
fi
# not a step: a rerun after more units finish should redo the summary
$PY analysis/exp14_analysis.py "$DIR" > "$LOGS/e14_analysis.log" 2>&1 || log "analysis FAILED (see e14_analysis.log)"
log "queue finished; $DIR/summary.txt:"
tee -a "$LOGS/queue_phrase_steer.log" < "$DIR/summary.txt"
