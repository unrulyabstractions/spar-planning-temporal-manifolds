#!/usr/bin/env bash
# Box queue: E13 steering with the pooled effects curve (16_spec_pristine_manifold.md §5), Qwen3-8B.
#   stage A: exp8_steering.py on steering/inputs_Qwen3-8B_pristine.npz, steer + post
#            (off suffix L16, off/on pre-label L22/L24; about 1.3 h on one GPU)
#   stage B: exp13_suffix_steer.py on steering/inputs_Qwen3-8B_pristine_suffix.npz, 8 conditions
#            on 96 prompts (about 27 min each, 3.6 h on one GPU)
# The inputs were prepared locally (pipeline/s1_data_vectors/analysis/steering_prepare.py --pooled [--stage-b]) and
# ship with `just push`, so this needs only the model, not the E0 runs.
# One GPU: stage A, then stage B. Several: the last GPU runs stage A and then joins
# stage B; the others start on stage B at once. Stage B copies claim conditions one at a
# time, so they share the work without a fixed split. Logs in $OUT/logs/; a rerun skips
# finished work. Does not stop the box.
#
#   cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
#   tmux new -s steer 'bash pipeline/s2_steering/queues/queue_pristine_steer.sh; bash'
#
# Watch: tail -f /workspace/results/horizon/logs/queue_pristine_steer.log

set -u
PY=${PY:-/venv/main/bin/python}
OUT=${OUT:-/workspace/results/horizon}
MODEL=Qwen/Qwen3-8B
STAGE_B_BATCH=${STAGE_B_BATCH:-12}   # generation batch; E0 used 20 on the 32 GB RTX 5090
LOGS=$OUT/logs
E13=$OUT/exp13_Qwen3-8B_pristine_suffix
N=$(nvidia-smi -L | wc -l)
mkdir -p "$LOGS"

log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOGS/queue_pristine_steer.log"; }

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
        tail -n 20 "$LOGS/$name.log" | tee -a "$LOGS/queue_pristine_steer.log"
        return 1
    fi
}

stage_a() {
    step e13_stage_a env CUDA_VISIBLE_DEVICES="$1" $PY -m pipeline.s2_steering.experiments.ablate_steer.exp8_steering \
        --inputs steering/inputs_Qwen3-8B_pristine.npz --tests steer,post --out "$OUT"
}

stage_b_worker() {   # one stage B copy on GPU $1; it exits when no condition is left to claim
    log "stage B worker on GPU $1: start"
    env CUDA_VISIBLE_DEVICES="$1" $PY -m pipeline.s2_steering.experiments.manifold_steer.exp13_suffix_steer \
        --inputs steering/inputs_Qwen3-8B_pristine_suffix.npz --batch-size "$STAGE_B_BATCH" --out "$OUT" \
        > "$LOGS/e13_stage_b_gpu$1.log" 2>&1 \
        && log "stage B worker on GPU $1: finished" || log "stage B worker on GPU $1: FAILED (see e13_stage_b_gpu$1.log)"
}

log "queue on $N GPU(s), out $OUT"
log "$($PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])' 2>&1)"
step e13_download $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" || exit 1
rm -rf "$E13"/conditions/*.lock   # claims left by a stopped run; finished conditions are kept

if [ "$N" -eq 1 ]; then
    stage_a 0
    stage_b_worker 0
else
    for ((gpu = 0; gpu < N - 1; gpu++)); do
        stage_b_worker "$gpu" &
    done
    (stage_a $((N - 1)); stage_b_worker $((N - 1))) &
    wait
fi
log "queue finished; stage A: $OUT/exp8_Qwen3-8B_pristine/results.json, stage B: $E13/summary.json"
