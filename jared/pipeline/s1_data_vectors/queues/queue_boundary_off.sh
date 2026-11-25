#!/usr/bin/env bash
# Box queue: EvalsBoundarySpec.md S1, thinking-off forward passes on every two-label eval type
# (exp21_boundary_off.py) on Qwen3-8B. Steps, each run once and marked <name>.done:
#   boundary_model   fetch the weights with parallel downloads
#   boundary_smoke   2 prompts per type into $OUT/smoke: plumbing on the real model
#   boundary_off     every type; resumes at the first type without rows.json
# Touches $LOGS/boundary.done at the end. Does not stop the box (the watchdog does once the GPU is idle).
#
#   cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
#   tmux new -d -s boundary 'bash -l pipeline/s1_data_vectors/queues/queue_boundary_off.sh; bash'

set -u
cd "$(dirname "$0")/../../.."
PY=${PY:-/venv/main/bin/python}
OUT=${OUT:-/workspace/results/horizon}
BATCH=${BATCH:-8}
TYPES=${TYPES:-all}
MODEL=Qwen/Qwen3-8B
RUN=$OUT/boundary_Qwen3-8B_think-off
LOGS=$OUT/logs
mkdir -p "$LOGS"

log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOGS/queue_boundary_off.log"; }

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
        tail -n 20 "$LOGS/$name.log" | tee -a "$LOGS/queue_boundary_off.log"
        return 1
    fi
}

log "queue on $(nvidia-smi -L | wc -l) GPU(s), out $RUN, types: $TYPES"
step boundary_model $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL', max_workers=8))" || exit 1
step boundary_smoke $PY -m pipeline.s1_data_vectors.experiments.exp21_boundary_off --model $MODEL --types "$TYPES" \
    --limit 2 --batch-size $BATCH --out "$OUT/smoke/boundary_Qwen3-8B_think-off" || exit 1
step boundary_off $PY -m pipeline.s1_data_vectors.experiments.exp21_boundary_off --model $MODEL --types "$TYPES" \
    --batch-size $BATCH --out "$RUN" || exit 1
touch "$LOGS/boundary.done"
log "all done"
