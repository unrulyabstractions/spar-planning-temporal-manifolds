#!/usr/bin/env bash
# Box queue: the eval-type banks (spar_horizon/eval_types, EvalsPromptTypes.md) on Qwen3-8B, thinking on,
# reduced activations (layers 10/14/16/18/22/24/26/30). Steps, each run once and marked <name>.done:
#   evaltypes_smoke   2 prompts per type, 64 think tokens, into $OUT/smoke: plumbing on the real model
#   evaltypes_score_smoke
#   evaltypes_e0      every type, 3,072 think tokens; resumes at the first type without rows.json
#   evaltypes_score   summary.json / summary.md in the run dir
# Touches $LOGS/evaltypes.done at the end. Does not stop the box (the watchdog does once the GPU is idle).
#
#   cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
#   tmux new -d -s evaltypes 'bash pipeline/s1_data_vectors/queues/queue_evaltypes.sh; bash'

set -u
PY=${PY:-/venv/main/bin/python}
OUT=${OUT:-/workspace/results/horizon}
BATCH=${BATCH:-12}
TYPES=${TYPES:-all}
MODEL=Qwen/Qwen3-8B
LOGS=$OUT/logs
mkdir -p "$LOGS"

log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOGS/queue_evaltypes.log"; }

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
        tail -n 20 "$LOGS/$name.log" | tee -a "$LOGS/queue_evaltypes.log"
        return 1
    fi
}

log "queue on $(nvidia-smi -L | wc -l) GPU(s), out $OUT, types: $TYPES"
# Fetch the weights with parallel downloads first: from_pretrained fetches one file at a time, and on a
# slow host (1.7 MB/s per connection on 2026-11-17) that alone took hours.
step evaltypes_model $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL', max_workers=8))" || exit 1
step evaltypes_smoke $PY -m pipeline.s1_data_vectors.experiments.exp0_evaltypes --model $MODEL --types "$TYPES" \
    --limit 2 --max-think-tokens 64 --batch-size $BATCH --out "$OUT/smoke" || exit 1
step evaltypes_score_smoke $PY -m pipeline.s3_analysis.eval_types_score "$OUT/smoke/evaltypes_Qwen3-8B_think-on" || exit 1
step evaltypes_e0 $PY -m pipeline.s1_data_vectors.experiments.exp0_evaltypes --model $MODEL --types "$TYPES" \
    --batch-size $BATCH --out "$OUT" || exit 1
step evaltypes_score $PY -m pipeline.s3_analysis.eval_types_score "$OUT/evaltypes_Qwen3-8B_think-on" || exit 1
touch "$LOGS/evaltypes.done"
log "all done"
