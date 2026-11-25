#!/usr/bin/env bash
# E23 related qualities on the box (EvalsRelatedQualities.md §3-5), Qwen3-8B thinking on.
#
#   tmux new -d -s related 'bash -l pipeline/s2_steering/queues/queue_related.sh full'
#
#   pilot   10 items per set into $WORKSPACE/results/horizon/related_pilot (throughput, parsing, plumbing)
#   full    the pilot first; if any pilot step fails the queue stops there (set -e), so an unattended
#           night is not spent on a broken step. Then every step on all items into $WORKSPACE/results/horizon.
#
# Needs steering/related_box/cells_Qwen3-8B_on_L24.npz (related_cells.py, run locally; shipped by
# `just push`). The probe's Q2 scores on T0 and the eval types need their saved activations, which are
# local only: after `just sync`, rerun eval_probe_fit with --score-runs ~/SPAR/results/horizon.
set -euo pipefail
MODE=${1:?usage: queue_related.sh pilot|full}
cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
PY=/venv/main/bin/python
MODEL=Qwen/Qwen3-8B
CELL=steering/related_box/cells_Qwen3-8B_on_L24.npz
LOG="$WORKSPACE/results/horizon/related_run.log"
mkdir -p "$(dirname "$LOG")"
# scoring 8 traces of ~3k tokens ran out of memory on a 4090 (E20, E22): exp23 scores 4 at a time
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "$CELL" ] || { echo "missing $CELL: run related_cells.py locally and just push" | tee -a "$LOG"; exit 1; }

step() {   # step NAME COMMAND...: run, log, and print the elapsed time (the pilot's times give the cost estimate)
    local name=$1 start=$SECONDS
    shift
    echo "=== $(date -u +%FT%TZ) $name: $*" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
    echo "=== $name done in $((SECONDS - start)) s" | tee -a "$LOG"
}

run_all() {   # run_all OUT EXP0_LIMIT EXP24_LIMIT EXP23_LIMIT (empty limit = all)
    local out=$1 inputs=$1/related_inputs/inputs_Qwen3-8B_related.npz probe=$1/eval_awareness/eval_probe_Qwen3-8B.npz
    step "E0 related" $PY -u -m pipeline.s1_data_vectors.experiments.exp0_evaltypes --suite related --model $MODEL \
        --types all --batch-size 12 --max-think-tokens 3072 --layers 16,22,24,26,28,30,32 --out "$out" ${2:+--limit $2}
    step "E24 probe states" $PY -u -m pipeline.s1_data_vectors.experiments.exp24_eval_probe_states --model $MODEL \
        --layers 16,22,24,26,28,30,32 --out "$out" ${3:+--limit $3}
    step "probe fit" $PY -u -m pipeline.s1_data_vectors.analysis.eval_probe_fit "$out/eval_probe_states_Qwen3-8B" \
        --out "$out/eval_awareness" --score-runs "$out"
    step "prepare" $PY -u -m pipeline.s1_data_vectors.analysis.related_steering_prepare "$out" --cell $CELL \
        --out "$out/related_inputs" --model $MODEL
    step "E23 steer" $PY -u -m pipeline.s2_steering.experiments.ablate_steer.exp23_related_steer --inputs "$inputs" \
        --eval-probe "$probe" --probe-layers 28,32 --out "$out" ${4:+--limit $4}
    step "report" $PY -u -m pipeline.s2_steering.analysis.related_report "$out/exp23_Qwen3-8B_related/results.json" \
        --inputs "$inputs" --eval-probe "$probe"
}

case $MODE in
    pilot) run_all "$WORKSPACE/results/horizon/related_pilot" 20 20 10 ;;
    full)  run_all "$WORKSPACE/results/horizon/related_pilot" 20 20 10
           run_all "$WORKSPACE/results/horizon" "" "" "" ;;
    *)     echo "usage: queue_related.sh pilot|full" >&2; exit 2 ;;
esac
echo "RELATED $MODE DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
