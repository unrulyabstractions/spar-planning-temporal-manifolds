#!/usr/bin/env bash
# Box queue: E0 on a paired bank, Qwen3-8B: BANK=units (14_spec_unit_overlap.md
# section 3, the default) or BANK=fill (16_spec_pristine_manifold.md section 4).
#   GPU 0: thinking on, batch 20 (2x RTX 5090: 32 GB each), no stability tests
#   GPU 1: thinking off, with the order/label stability tests
# With one GPU the two run in order. Logs in $OUT/logs/; a rerun skips finished
# steps. Does not stop the box.
#
#   cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
#   tmux new -s units 'bash queue_units.sh; bash'
#   tmux new -s fill 'BANK=fill bash queue_units.sh; bash'

set -u
PY=${PY:-/venv/main/bin/python}
OUT=${OUT:-/workspace/results/horizon}
MODEL=Qwen/Qwen3-8B
BANK=${BANK:-units}
ON_BATCH=${ON_BATCH:-20}
LOGS=$OUT/logs
N=$(nvidia-smi -L | wc -l)
mkdir -p "$LOGS"

log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOGS/queue_$BANK.log"; }

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
        tail -n 20 "$LOGS/$name.log" | tee -a "$LOGS/queue_$BANK.log"
        return 1
    fi
}

e0_on() { step ${BANK}_e0_on env CUDA_VISIBLE_DEVICES="$1" $PY experiments/exp0_baseline.py --model $MODEL \
    --thinking on --bank $BANK --batch-size "$ON_BATCH" --skip-behavior --out "$OUT"; }
e0_off() { step ${BANK}_e0_off env CUDA_VISIBLE_DEVICES="$1" $PY experiments/exp0_baseline.py --model $MODEL \
    --thinking off --bank $BANK --out "$OUT"; }

log "queue on $N GPU(s), out $OUT"
# Blackwell cards (RTX 5090) need torch >= 2.7 built for CUDA 12.8; this line shows what the box has
log "$($PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])' 2>&1)"
step ${BANK}_download $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" || exit 1
if [ "$N" -eq 1 ]; then
    e0_off 0
    e0_on 0
else
    e0_on 0 &
    e0_off 1 &
    wait
fi
log "queue finished; results in $OUT/exp0_Qwen3-8B_think-{on,off}_$BANK/"
