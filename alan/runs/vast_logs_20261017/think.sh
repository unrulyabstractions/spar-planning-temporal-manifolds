#!/bin/bash
cd /workspace/SPAR-2026
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1
while [ ! -f runs/nothink.done ]; do sleep 5; done
P=data/prompts/unit_boundary_think_pilot.parquet
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g .venv/bin/python -u scripts/capture.py $P runs/qwen3-14b_think_pilot_g$g --model Qwen/Qwen3-14B \
    --thinking --max-new-tokens 8192 --batch-size 24 --no-acts --offset $((g*192)) --limit 192 > runs/qwen3-14b_think_pilot_g$g.log 2>&1 &
done
wait; echo THINK DONE > runs/think.done
