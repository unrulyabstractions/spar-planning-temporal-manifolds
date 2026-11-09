#!/usr/bin/env bash
# Box queue: Part B of the overnight scan, the steering screen on Qwen3-8B with thinking on
# (26_spec_overnight_choice_scan.md §2). 40 cells (layers 10/14/18/22/26 x number, unit, suffix, answer
# x manifold, random) on the same 10 prompts, plus the baseline and two no-op reruns: about 330
# generations (30 regenerating cells x 10, baseline, 2 reruns); the 10 answer-slot cells only re-score.
#   1. states      exp14_phrase_states.py: phrase states at L10/22/26 (states_L10-22-26.npz); the default
#                  states (L12-18, states.npz) are extracted too if they are not on the box
#   2. prepare     scan_prepare.py (CPU): the pooled fit per (layer, position), the 10 prompts, steps and
#                  random twins; the suffix and answer-slot fits read the saved E0 activations under $E0.
#                  If those are not on the box: sync $PDIR/states_L10-22-26.npz down, run scan_prepare.py
#                  locally, `just push` the folder $DIR (inputs.json, inputs.npz), and rerun this queue
#   3. run         exp15_scan.py: baseline, no-op reruns, then the cells pair by pair in layer order 18, 14,
#                  22, 10, 26; with several GPUs the workers claim units, and one worker passes again at the
#                  end for answer-slot cells that found the baseline unfinished
#   4. summary     scan_summary.py: scan_summary.txt (a screen, no significance claim)
# The prompts file (steering/phrase_prompts_Qwen3-8B.json) ships with `just push`. Logs in $OUT/logs/; a
# rerun skips finished work. Does not stop the box.
#
#   cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
#   tmux new -s scan 'bash pipeline/s2_steering/queues/queue_scan.sh; bash'
#
# Watch: tail -f /workspace/results/horizon/logs/queue_scan.log

set -u
PY=${PY:-/venv/main/bin/python}
OUT=${OUT:-/workspace/results/horizon}
E0=${E0:-$OUT}
MODEL=Qwen/Qwen3-8B
PROMPTS=steering/phrase_prompts_Qwen3-8B.json
BATCH=${BATCH:-12}   # generation batch; a cell's 10 prompts fit one batch
LOGS=$OUT/logs
PDIR=$OUT/exp14_Qwen3-8B_phrase
DIR=$OUT/exp15_Qwen3-8B_scan
N=$(nvidia-smi -L | wc -l)
mkdir -p "$LOGS"

log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOGS/queue_scan.log"; }

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
        tail -n 20 "$LOGS/$name.log" | tee -a "$LOGS/queue_scan.log"
        return 1
    fi
}

prepare() {   # extraction on GPU $1, then the fit on CPU
    step scan_states_default env CUDA_VISIBLE_DEVICES="$1" $PY -m pipeline.s1_data_vectors.experiments.exp14_phrase_states \
        --prompts "$PROMPTS" --out "$OUT" || return 1
    step scan_states_new env CUDA_VISIBLE_DEVICES="$1" $PY -m pipeline.s1_data_vectors.experiments.exp14_phrase_states \
        --prompts "$PROMPTS" --out "$OUT" --layers 10,22,26 || return 1
    if [ -e "$DIR/inputs.json" ] && [ -e "$DIR/inputs.npz" ]; then
        log "scan inputs found in $DIR (prepared elsewhere or earlier), skipped the fit"
    elif ls "$E0"/exp0_Qwen3-8B_think-on_*/activations.npz > /dev/null 2>&1; then
        step scan_prepare $PY -m pipeline.s1_data_vectors.analysis.scan_prepare --phrase-dir "$PDIR" --e0-root "$E0" || return 1
    else
        log "no E0 activations under $E0: fit locally with the synced $PDIR/states_L10-22-26.npz, push $DIR, rerun"
        return 1
    fi
    log "fit report:"
    tee -a "$LOGS/queue_scan.log" < "$DIR/fit_report.txt"
}

worker() {   # one driver copy on GPU $1; it exits when no unit is left to claim
    log "driver on GPU $1: start"
    env CUDA_VISIBLE_DEVICES="$1" $PY -m pipeline.s2_steering.experiments.manifold_steer.exp15_scan --dir "$DIR" --batch-size "$BATCH" \
        > "$LOGS/scan_run_gpu$1.log" 2>&1 \
        && log "driver on GPU $1: finished" || log "driver on GPU $1: FAILED (see scan_run_gpu$1.log)"
}

wait_for_inputs() {   # the other GPUs start once GPU 0 has written the inputs, or give up if it failed
    until [ -e "$DIR/inputs.json" ]; do
        [ -e "$LOGS/scan_prepare.failed" ] && return 1
        sleep 30
    done
}

log "queue on $N GPU(s), out $OUT"
log "$($PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])' 2>&1)"
step scan_download $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" || exit 1
rm -f "$LOGS/scan_prepare.failed"
rm -rf "$DIR"/conditions/*.lock   # claims left by a stopped run; finished units are kept

(prepare 0 || touch "$LOGS/scan_prepare.failed"; [ -e "$LOGS/scan_prepare.failed" ] || worker 0) &
for ((gpu = 1; gpu < N; gpu++)); do
    (wait_for_inputs && worker "$gpu") &
done
wait
if [ -e "$LOGS/scan_prepare.failed" ]; then
    log "queue stopped: extraction or fit failed"
    exit 1
fi
if [ "$N" -gt 1 ]; then
    worker 0   # answer-slot cells that a worker skipped because the baseline was still running
fi
# not a step: a rerun after more units finish should redo the summary
$PY -m pipeline.s2_steering.analysis.scan_summary "$DIR" > "$LOGS/scan_summary.log" 2>&1 || log "summary FAILED (see scan_summary.log)"
log "queue finished; $DIR/scan_summary.txt:"
tee -a "$LOGS/queue_scan.log" < "$DIR/scan_summary.txt"
