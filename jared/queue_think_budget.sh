#!/usr/bin/env bash
# Box queue: is the force-closed share of the k-option banks a think-budget artefact?
# (28_spec_choice4_side.md and its decisions section). Qwen3-8B, thinking on, no activations. Inputs are the
# stage 0 files in steering/exp17_Qwen3-8B_think_budget/ (they ship with `just push`; the box has no results/).
# Steps, in this order (BANKS="choice4 choice8"):
#   <bank>_u1        unit 1: the original forced answer step again in new batches (the noise floor)
#   <bank>_u2        unit 2: each trace resumed from its 3,072 think tokens for up to 3,072 more
#   then, optional and last:
#   <bank>_u3_c<N>   unit 3: traces still open after unit 2 continue to 8,192, batch $BATCH_U3, the traces of the
#                    first $UNIT3_CHUNKS unit-2 chunks (raise UNIT3_CHUNKS to do more; the step name changes)
# Each step writes <bank>/unit<N>/chunk_NNN.{json,npz} of $CHUNK traces as it goes, so a stopped step resumes at
# its first missing chunk and any prefix of chunks can be summarised. At the end analysis/think_budget_summary.py
# writes think_budget_summary.{txt,json} (P0, C1, C2, C3 with their n) from whatever exists; it is not a step.
# Estimates on one RTX 4090: choice4 105 traces ~1.5 h (unit 2 ~1.4 h), choice8 74 traces ~1 h, unit 3 ~0.5 h.
# SMOKE=1 runs only the plumbing check on the real model: 4 traces of choice4, units 1 and 2 with 128 extra
# tokens, into $OUT/smoke. Logs in $OUT/logs/. Does not stop the box.
#
#   cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
#   tmux new -s budget 'SMOKE=1 bash queue_think_budget.sh; bash'     # first
#   tmux new -s budget 'bash queue_think_budget.sh; bash'
#
# Watch: tail -f /workspace/results/horizon/logs/queue_think_budget.log

set -u
PY=${PY:-/venv/main/bin/python}
OUT=${OUT:-/workspace/results/horizon}
BATCH=${BATCH:-8}
BATCH_U3=${BATCH_U3:-4}
CHUNK=${CHUNK:-24}
UNIT3_CHUNKS=${UNIT3_CHUNKS:-3}
BANKS=${BANKS:-"choice4 choice8"}
SMOKE=${SMOKE:-0}
MODEL=Qwen/Qwen3-8B
RUN=$OUT/exp17_${MODEL##*/}_think_budget
LOGS=$OUT/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"

log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOGS/queue_think_budget.log"; }

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
        tail -n 20 "$LOGS/$name.log" | tee -a "$LOGS/queue_think_budget.log"
        return 1
    fi
}

exp() {   # exp <bank> <unit> <args...>
    local bank=$1 unit=$2
    shift 2
    $PY experiments/exp17_think_budget.py --bank "$bank" --unit "$unit" "$@"
}

log "queue on $(nvidia-smi -L | wc -l) GPU(s), out $OUT, banks: $BANKS, smoke: $SMOKE"
log "$($PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])' 2>&1)"
step download $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" || exit 1

if [ "$SMOKE" = 1 ]; then
    step smoke_u1 exp choice4 1 --batch-size 4 --limit 4 --out "$OUT/smoke" || exit 1
    step smoke_u2 exp choice4 2 --batch-size 4 --limit 4 --extra-tokens 128 --out "$OUT/smoke" || exit 1
    $PY analysis/think_budget_summary.py "$OUT/smoke/exp17_${MODEL##*/}_think_budget" > "$LOGS/smoke_summary.log" 2>&1 \
        || log "smoke summary FAILED (see smoke_summary.log)"
    log "smoke finished; plumbing only: $OUT/smoke/exp17_${MODEL##*/}_think_budget/think_budget_summary.txt"
    exit 0
fi

for bank in $BANKS; do
    step ${bank}_u1 exp "$bank" 1 --batch-size "$BATCH" --chunk-size "$CHUNK" --out "$OUT" || exit 1
    step ${bank}_u2 exp "$bank" 2 --batch-size "$BATCH" --chunk-size "$CHUNK" --out "$OUT" || exit 1
done
for bank in $BANKS; do
    step ${bank}_u3_c$UNIT3_CHUNKS exp "$bank" 3 --batch-size "$BATCH_U3" --chunk-size "$CHUNK" \
        --max-chunks "$UNIT3_CHUNKS" --out "$OUT" || exit 1
done

# not a step: a rerun after more chunks finish should redo the summary
$PY analysis/think_budget_summary.py "$RUN" > "$LOGS/think_budget_summary.log" 2>&1 \
    || log "summary FAILED (see think_budget_summary.log)"
log "queue finished; $RUN/think_budget_summary.txt:"
tee -a "$LOGS/queue_think_budget.log" < "$RUN/think_budget_summary.txt"
