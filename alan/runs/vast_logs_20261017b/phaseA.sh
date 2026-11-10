#!/bin/bash
# per model: download, tests, verify, 16-prompt thinking probe (cap 8192, peak memory sampled), no-thinking 384 pilot
set -u; cd /workspace/SPAR-2026
MODEL=$1; TAG=$2; PB=${3:-8}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/.hf_home
L=runs/phaseA_$TAG.log; st() { echo "$(date '+%F %T') $*" | tee -a $L; }
P=data/prompts/unit_boundary_think_pilot.parquet
st "PHASE A START $MODEL commit $(cat COMMIT)"
.venv/bin/hf download "$MODEL" > runs/hf_download_$TAG.log 2>&1 || { st "FAILED download"; exit 1; }
st "download done; disk $(df -h . | tail -1 | awk '{print $4}') free"
export HF_HUB_OFFLINE=1
.venv/bin/python -m pytest -q tests > runs/tests_$TAG.log 2>&1 || { st "FAILED tests"; exit 1; }
st "tests: $(tail -1 runs/tests_$TAG.log)"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/verify_capture.py --model "$MODEL" --n 4 > runs/verify_$TAG.log 2>&1
grep -q "VERIFY OK" runs/verify_$TAG.log && st "VERIFY OK" || { st "FAILED verify"; exit 1; }
( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> runs/mem_probe_$TAG.txt; sleep 10; done ) &
MP=$!
PIDS=""
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g .venv/bin/python -u scripts/capture.py $P runs/${TAG}_think_probe_g$g --model "$MODEL" --thinking --max-new-tokens 8192 --batch-size $PB --no-acts --offset $((g*8)) --limit 8 > runs/${TAG}_think_probe_g$g.log 2>&1 &
  PIDS="$PIDS $!"
done
wait $PIDS
kill $MP
st "probe: $(for g in 0 1; do tail -1 runs/${TAG}_think_probe_g$g.log; done | tr '\n' ' ') peak MiB $(sort -t, -k2 -n runs/mem_probe_$TAG.txt | tail -1)"
PIDS=""
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g .venv/bin/python -u scripts/capture.py $P runs/${TAG}_nothink_pilot_g$g --model "$MODEL" --batch-size 32 --no-acts --offset $((g*192)) --limit 192 > runs/${TAG}_nothink_pilot_g$g.log 2>&1 &
  PIDS="$PIDS $!"
done
wait $PIDS
st "nothink: $(for g in 0 1; do tail -1 runs/${TAG}_nothink_pilot_g$g.log; done | tr '\n' ' ')"
st "PHASE A DONE"
