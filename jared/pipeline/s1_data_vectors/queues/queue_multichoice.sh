#!/usr/bin/env bash
# Box queue: Part A of the overnight scan (26_spec_overnight_choice_scan.md §1, §6), E0 on the k-option
# choice banks, Qwen3-8B, thinking on, with the reduced activation set (layers 10/14/16/18/22/24/26/30
# plus the horizon-phrase and per-option prompt positions), written in chunks of $CHUNK prompts.
# Per bank, two steps:
#   <bank>_e0         experiments/exp0_choice.py: chunks/chunk_NNN.{npz,json}, then answers.json and manifest.json
#   <bank>_structure  analysis/choice_structure.py on the finished chunks: choice_structure/results.{json,txt}, png
# BANKS picks the banks (default: the implicit-ROI pair); the explicit-ROI twins are not run by default:
#   BANKS="choice4_roi choice8_roi" bash queue_multichoice.sh
# SMOKE=1 runs only the plumbing check: Qwen3.5-0.8B (it never closes its think block) on 8 prompts of each
# bank in chunks of 4, 32 think tokens, into $OUT/smoke, then choice_structure on it. Nothing else runs.
# Logs in $OUT/logs/; a rerun skips finished steps, and a stopped e0 step resumes at its first missing chunk.
# To analyse a stopped run's chunks, run analysis/choice_structure.py on its run dir by hand. Does not stop the box.
#
#   cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
#   tmux new -s choice 'bash queue_multichoice.sh; bash'
#   tmux new -s smoke 'SMOKE=1 bash queue_multichoice.sh; bash'

set -u
PY=${PY:-/venv/main/bin/python}
OUT=${OUT:-/workspace/results/horizon}
BATCH=${BATCH:-12}
CHUNK=${CHUNK:-48}
BANKS=${BANKS:-"choice4 choice8"}
SMOKE=${SMOKE:-0}
MODEL=Qwen/Qwen3-8B
SMOKE_MODEL=Qwen/Qwen3.5-0.8B
LOGS=$OUT/logs
mkdir -p "$LOGS"

log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOGS/queue_multichoice.log"; }

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
        tail -n 20 "$LOGS/$name.log" | tee -a "$LOGS/queue_multichoice.log"
        return 1
    fi
}

log "queue on $(nvidia-smi -L | wc -l) GPU(s), out $OUT, banks: $BANKS, smoke: $SMOKE"
log "$($PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])' 2>&1)"

if [ "$SMOKE" = 1 ]; then
    step smoke_download $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$SMOKE_MODEL')" || exit 1
    for bank in $BANKS; do
        run=$OUT/smoke/exp0_${SMOKE_MODEL##*/}_think-on_$bank
        step ${bank}_smoke_e0 $PY experiments/exp0_choice.py --model $SMOKE_MODEL --thinking on --bank $bank \
            --limit 8 --chunk-size 4 --max-think-tokens 32 --batch-size 4 --out "$OUT/smoke" || exit 1
        step ${bank}_smoke_structure $PY analysis/choice_structure.py "$run" || exit 1
    done
    log "smoke finished; plumbing only, see $OUT/smoke/*/choice_structure/results.txt"
    exit 0
fi

step download $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" || exit 1
for bank in $BANKS; do
    run=$OUT/exp0_${MODEL##*/}_think-on_$bank
    step ${bank}_e0 $PY experiments/exp0_choice.py --model $MODEL --thinking on --bank $bank \
        --batch-size "$BATCH" --chunk-size "$CHUNK" --out "$OUT" || exit 1
    step ${bank}_structure $PY analysis/choice_structure.py "$run" || exit 1
done
log "queue finished; results in $OUT/exp0_${MODEL##*/}_think-on_<bank>/{answers.json,manifest.json,chunks/,choice_structure/}"
