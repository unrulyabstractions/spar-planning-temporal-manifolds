#!/usr/bin/env bash
# Box queue: E10 (07 section 12) split over every GPU of the instance, plus the
# E8 decision split at L22 (07 section 10). Each GPU runs one copy of
# exp10_patch_reasoning; the copies claim conditions one at a time, so they
# share the work without a fixed split. With several GPUs the last one runs the
# L22 job first, then joins E10. Logs in $OUT/logs/; a rerun skips finished work.
# Does not stop the box.
#
#   cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
#   tmux new -s queue 'bash pipeline/s1_data_vectors/queues/queue.sh; bash'
#
# Watch: tail -f /workspace/results/horizon/logs/queue.log

set -u
PY=/venv/main/bin/python
OUT=/workspace/results/horizon
LOGS=$OUT/logs
E10=$OUT/exp10_Qwen3-8B_reasoning
N=$(nvidia-smi -L | wc -l)
mkdir -p "$LOGS"

log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOGS/queue.log"; }

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
        tail -n 20 "$LOGS/$name.log" | tee -a "$LOGS/queue.log"
    fi
}

e10_worker() {   # one E10 copy on GPU $1; it exits when no condition is left to claim
    log "E10 worker on GPU $1: start"
    env CUDA_VISIBLE_DEVICES="$1" $PY -m pipeline.s2_steering.experiments.patch.exp10_patch_reasoning --out "$OUT" > "$LOGS/exp10_gpu$1.log" 2>&1 \
        && log "E10 worker on GPU $1: finished" || log "E10 worker on GPU $1: FAILED (see exp10_gpu$1.log)"
}

l22() {
    step exp8_decision_L22 env CUDA_VISIBLE_DEVICES="$1" $PY -m pipeline.s2_steering.experiments.ablate_steer.exp8_steering \
        --inputs steering/inputs_Qwen3-8B_decision_L22.npz --tests post --out "$OUT"
}

log "queue on $N GPU(s)"
rm -rf "$E10"/conditions/*.lock   # claims left by a stopped run; finished conditions are kept
step exp10_subspaces env CUDA_VISIBLE_DEVICES=0 $PY -m pipeline.s2_steering.experiments.patch.exp10_patch_reasoning --fit-only --out "$OUT"

if [ "$N" -eq 1 ]; then
    e10_worker 0
    l22 0
else
    for ((gpu = 0; gpu < N - 1; gpu++)); do
        e10_worker "$gpu" &
    done
    (l22 $((N - 1)); e10_worker $((N - 1))) &
    wait
fi
log "queue finished; E10 summary: $E10/summary.json"
