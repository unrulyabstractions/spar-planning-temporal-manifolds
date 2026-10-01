#!/bin/bash
# Re-capture the matrix and the held-out distractor families with anchored response positions and a
# 160-token cap (so reasonings finish), for the late-readout comparison. Usage: nohup scripts/run_long_captures.sh &
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CAP="--model Qwen/Qwen3-14B --batch-size 8 --split 19 --max-new-tokens 160 --anchors"
.venv/bin/python -u scripts/capture.py data/prompts/matrix/capture_s0.parquet runs/qwen3-14b_matrix_s0_long $CAP > runs/qwen3-14b_matrix_s0_long.log 2>&1
.venv/bin/python -u scripts/capture.py data/prompts/matrix/families_s0.parquet runs/qwen3-14b_families_s0_long $CAP > runs/qwen3-14b_families_s0_long.log 2>&1
echo "LONG CAPTURES DONE" >> runs/qwen3-14b_families_s0_long.log
