#!/bin/bash
# Sequential GPU chain for 2026-09-22: phrasings (14B) -> canonical (8B) -> paired units (14B)
set -u
cd /home/alan/SPAR-2026
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
.venv/bin/python -u scripts/capture.py data/prompts/investment_phrasings_n2000_s0.parquet runs/qwen3-14b_phrasings_n2000_s0 --model Qwen/Qwen3-14B --batch-size 8 --split 19 > runs/qwen3-14b_phrasings_n2000_s0.log 2>&1
.venv/bin/python -u scripts/capture.py data/prompts/investment_n2000_s0.parquet runs/qwen3-8b_investment_n2000_s0 --model Qwen/Qwen3-8B --batch-size 8 > runs/qwen3-8b_investment_n2000_s0.log 2>&1
.venv/bin/python -u scripts/capture.py data/prompts/investment_units_paired_n2000_s1.parquet runs/qwen3-14b_units_paired_n2000_s1 --model Qwen/Qwen3-14B --batch-size 8 --split 19 > runs/qwen3-14b_units_paired_n2000_s1.log 2>&1
echo "CHAIN DONE"
