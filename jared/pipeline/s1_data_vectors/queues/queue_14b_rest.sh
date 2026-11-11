#!/usr/bin/env bash
# Box queue: the rest of E11 (10_spec_scale_14b.md section 7b), with E8's
# shared-prefix scoring. Inputs were prepared locally into steering/ and ship
# with `just push`, so this needs only the model, not the E0 runs.
#   GPU 0: on_L27_check (the thinking-on L27 cell again: the fast path must
#          reproduce the slow run's exp8_Qwen3-14B_answer_slot_on), then on_L30_L33
#   GPU 1: line_off and line_on (probe line, curve, mean difference, random at
#          L27/30/33; the line was dropped from the decision-split runs)
# With one GPU the four run in order. Logs in $OUT/logs/; a rerun skips
# finished steps. Does not stop the box.
#
#   cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
#   tmux new -s rest 'bash pipeline/s1_data_vectors/queues/queue_14b_rest.sh; bash'

set -u
PY=${PY:-/venv/main/bin/python}
OUT=${OUT:-/workspace/results/horizon}
LOGS=$OUT/logs
N=$(nvidia-smi -L | wc -l)
mkdir -p "$LOGS"

log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOGS/queue_14b_rest.log"; }

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
        tail -n 20 "$LOGS/$name.log" | tee -a "$LOGS/queue_14b_rest.log"
        return 1
    fi
}

exp8() {   # exp8 <gpu> <inputs name> <tests>
    step "exp8_14B_$2" env CUDA_VISIBLE_DEVICES="$1" $PY -m pipeline.s2_steering.experiments.ablate_steer.exp8_steering \
        --inputs "steering/inputs_Qwen3-14B_$2.npz" --tests "$3" --out "$OUT"
}

lane_0() { exp8 "$1" on_L27_check post; exp8 "$1" on_L30_L33 post; }
lane_1() { exp8 "$1" line_off steer; exp8 "$1" line_on post; }

log "queue on $N GPU(s), out $OUT"
# one download before the lanes start, so two processes do not fetch the same 30 GB
step download_14b $PY -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-14B')" || exit 1
if [ "$N" -eq 1 ]; then
    lane_0 0
    lane_1 0
else
    lane_0 0 &
    lane_1 1 &
    wait
fi
log "queue finished; results in $OUT/exp8_Qwen3-14B_{on_L27_check,on_L30_L33,line_off,line_on}/"
