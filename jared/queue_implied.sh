#!/usr/bin/env bash
# Box queue: E18, implied horizons on Qwen3-8B with thinking on (29_spec_implied_horizon_side.md), 288
# generations in four modules, each usable on its own (decisions: Jared, 2026-11-07, in the spec):
#   1. M1  exp18_implied.py --modules M1: families A and C, near/both, with explicit twins and null twins (144)
#   2. M2  family B, near/both, twins, null (72)
#   3. M3  the none regime of A, B, C, with twins (48)
#   4. M4  vague quantifiers (24)
#   5. summary  implied_summary.py after every module (and so after a stop): implied_summary.txt/.json and the
#               audit sample. The readout (b) trains on the explicit E0 runs under $OUT (exp0_*_full, *_fill)
#               when they are on the box; otherwise rerun the summary where they are, with --probe-runs.
# A unit (up to 12 prompts, one batch) is saved as it finishes and skipped on a rerun, so a stopped queue
# resumes inside a module. Logs in $OUT/logs/. Does not stop the box. The whole run is meant to fit ~4 h of
# one RTX 4090; M1 alone is ~1.8 h. To stop early, kill the queue after any module: the summary is current.
#
#   cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
#   tmux new -s implied 'bash queue_implied.sh; bash'
#
# Watch: tail -f /workspace/results/horizon/logs/queue_implied.log

set -u
PY=${PY:-/venv/main/bin/python}
OUT=${OUT:-/workspace/results/horizon}
MODEL=Qwen/Qwen3-8B
BATCH=${BATCH:-12}   # generation batch; generate_batch steps down on OOM
LOGS=$OUT/logs
RUN=$OUT/exp18_Qwen3-8B_think-on_implied
mkdir -p "$LOGS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOGS/queue_implied.log"; }

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
        tail -n 20 "$LOGS/$name.log" | tee -a "$LOGS/queue_implied.log"
        return 1
    fi
}

summary() {   # not a step: a rerun after more units finish should redo it
    local probes=()
    for d in "$OUT"/exp0_Qwen3-8B_think-on_full "$OUT"/exp0_Qwen3-8B_think-on_fill; do
        [ -e "$d/activations.npz" ] && probes+=("$d")
    done
    if [ "${#probes[@]}" -eq 0 ]; then
        log "no explicit E0 activations under $OUT: the readout (b) stays pending"
        $PY analysis/implied_summary.py "$RUN" > "$LOGS/implied_summary.log" 2>&1 || log "summary FAILED (see implied_summary.log)"
    else
        $PY analysis/implied_summary.py "$RUN" --probe-runs "${probes[@]}" > "$LOGS/implied_summary.log" 2>&1 \
            || log "summary FAILED (see implied_summary.log)"
    fi
}

log "queue, out $OUT"
log "$($PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])' 2>&1)"
step implied_download $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" || exit 1

for module in M1 M2 M3 M4; do
    step "implied_$module" $PY experiments/exp18_implied.py --modules "$module" --batch-size "$BATCH" --out "$OUT" \
        || { summary; exit 1; }
    summary
done
log "queue finished; $RUN/implied_summary.txt:"
tee -a "$LOGS/queue_implied.log" < "$RUN/implied_summary.txt"
