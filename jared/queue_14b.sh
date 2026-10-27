#!/usr/bin/env bash
# Box queue: E11, the 8B map on Qwen3-14B (10_spec_scale_14b.md section 3).
#   A  GPU 0: E0 thinking on
#   B  GPU 1: E0 thinking off, its thinking-off inputs, then T1 patching
#   C/D after both E0 runs: answer-slot inputs at L27/30/33, then E8 split by
#      mode: GPU 0 runs the thinking-off cells and T5 in both modes, GPU 1 the
#      thinking-on cells.
# With one GPU the same steps run in order. Logs in $OUT/logs/; a rerun skips
# finished steps. Does not stop the box.
#
#   cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
#   SMOKE=1 bash queue_14b.sh          # 6 prompts into $OUT/smoke_14b/, to time it and check shapes
#   tmux new -s queue 'bash queue_14b.sh; bash'
#
# Watch: tail -f /workspace/results/horizon/logs/queue_14b.log

set -u
PY=${PY:-/venv/main/bin/python}
MODEL=Qwen/Qwen3-14B
NAME=Qwen3-14B
OUT=${OUT:-/workspace/results/horizon}
LIMIT=()
if [ "${SMOKE:-}" = 1 ]; then
    OUT=$OUT/smoke_14b
    LIMIT=(--limit 6)
fi
LOGS=$OUT/logs
INPUTS=$OUT/steering_inputs   # next to the runs, so a smoke never overwrites real inputs
N=$(nvidia-smi -L | wc -l)
mkdir -p "$LOGS" "$INPUTS"

# 8B relative depths of L24 / L27 / L30 (0.67 / 0.75 / 0.83) on 40 layers, fixed before the run
OFF_CELLS=off:pre-label:27,off:pre-label:30,off:pre-label:33
ON_CELLS=on:pre-label:27,on:pre-label:30,on:pre-label:33

log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOGS/queue_14b.log"; }

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
        tail -n 20 "$LOGS/$name.log" | tee -a "$LOGS/queue_14b.log"
        return 1
    fi
}

on_gpu() {   # on_gpu <gpu> <command...>
    local gpu=$1
    shift
    env CUDA_VISIBLE_DEVICES="$gpu" "$@"
}

e0() {   # e0 <gpu> <mode>; thinking on skips the order/label stability tests, as in run_all.sh
    local skip=()
    [ "$2" = on ] && skip=(--skip-behavior)
    step "e0_$2" on_gpu "$1" $PY experiments/exp0_baseline.py --model $MODEL --thinking "$2" --bank full \
        --out "$OUT" "${LIMIT[@]}" "${skip[@]}"
}

prepare() {   # prepare <name> <cells> [extra flags]
    step "prepare_$1" $PY analysis/steering_prepare.py "$OUT" --model $MODEL --out "$INPUTS" --name "$1" \
        --cells "$2" "${@:3}"
}

exp8() {   # exp8 <gpu> <inputs name> <tests>
    step "exp8_$2" on_gpu "$1" $PY experiments/exp8_steering.py --inputs "$INPUTS/inputs_${NAME}_$2.npz" \
        --tests "$3" --out "$OUT" "${LIMIT[@]}"
}

lane_b() {   # thinking off: the patching map needs no thinking-on run
    e0 "$1" off &&
        prepare patch off:pre-label:27 --no-reasoning &&
        exp8 "$1" patch patch
}

log "queue on $N GPU(s), out $OUT"
# one download before the lanes start, so two processes do not fetch the same 30 GB
step download $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" || exit 1

if [ "$N" -eq 1 ]; then
    lane_b 0
    e0 0 on
else
    e0 0 on &
    lane_b 1 &
    wait
fi

# thinking-on directions use natural closes only (08 section 9); thinking off has none forced
prepare answer_slot_off "$OFF_CELLS" --natural-only &&
    prepare answer_slot_on "$ON_CELLS" --natural-only || { log "inputs failed; E8 cells not run"; exit 1; }

if [ "$N" -eq 1 ]; then
    exp8 0 answer_slot_off ablate,steer,answer
    exp8 0 answer_slot_on post
else
    exp8 0 answer_slot_off ablate,steer,answer &
    exp8 1 answer_slot_on post &
    wait
fi
log "queue finished; results in $OUT/exp8_${NAME}_*/ and $OUT/exp0_${NAME}_*_full/"
