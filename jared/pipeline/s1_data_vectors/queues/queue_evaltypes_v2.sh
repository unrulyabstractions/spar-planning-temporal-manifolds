#!/usr/bin/env bash
# Box queue: the v2 eval types (EvalsPromptsV2.md), most important first, through queue_evaltypes.sh.
# Batch 5: larger batches ran out of memory on a 24 GB RTX 4090 at 3,072 think tokens.
#
#   tmux new -d -s evaltypes2 'bash /workspace/SPAR/spar-planning-temporal-manifolds/jared/pipeline/s1_data_vectors/queues/queue_evaltypes_v2.sh; bash'

cd "$(dirname "$0")/../../.." || exit 1
export BATCH=${BATCH:-5}
export TYPES=${TYPES:-t0_w0,t0_w1_cash,t0_w3_clear,roi_unequal_clear,system_prompt_clear,sycophancy_lean_clear,horizon_revised_clear,free_form_anchor}
exec bash pipeline/s1_data_vectors/queues/queue_evaltypes.sh
