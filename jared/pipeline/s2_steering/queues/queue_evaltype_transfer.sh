#!/usr/bin/env bash
# Inputs are staged by hand into steering/exp20_box/ (gitignored): the eval-type rows.json + settings.json,
# exp14 states.npz/.json and steering/phrase_prompts_Qwen3-8B.json; `just push` ships them (EvalsSteeringResults.md).
# E20 on the box (EvalsSteeringSpec.md §7): the 4 domain types on every arm, then the extras on
# donor and noop. Run in tmux under bash -l; the log goes to $WORKSPACE/results/horizon/exp20_run.log.
set -euo pipefail
cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
B=steering/exp20_box
LOG="$WORKSPACE/results/horizon/exp20_run.log"
mkdir -p "$(dirname "$LOG")"
# Scoring 8 traces of up to ~3,300 tokens ran out of memory on the 4090 (first run, 22:08), so 4,
# and the allocator setting the OOM message suggests against fragmentation after generation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON=(--evaltypes-dir $B/evaltypes --exp14-dir $B/exp14 --bank-prompts $B/phrase_prompts_Qwen3-8B.json
        --score-batch-size 4)
E20=pipeline.s2_steering.experiments.manifold_steer.exp20_evaltype_transfer
/venv/main/bin/python -m $E20 "${COMMON[@]}" \
  --types city_no_values,city_values,public_health,coding_agent 2>&1 | tee -a "$LOG"
/venv/main/bin/python -m $E20 "${COMMON[@]}" --arms donor,noop --rows-per-type 16 \
  --types distractor_temporal,multiturn_analyst,role_assistant,role_fund_manager,sycophancy_lean,system_prompt,system_prompt_clear,t0_w0,t0_w1_cash,t0_w3_clear \
  2>&1 | tee -a "$LOG"
echo "E20 DONE" | tee -a "$LOG"
