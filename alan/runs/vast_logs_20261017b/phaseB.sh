#!/bin/bash
# per model: wait for phase A, thinking 384 pilot (cap 8192), merge, analyses, free weights; optionally start next phase A
set -u; cd /workspace/SPAR-2026
MODEL=$1; TAG=$2; BS=$3; NEXT_MODEL=${4:-}; NEXT_TAG=${5:-}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1
L=runs/phaseB_$TAG.log; st() { echo "$(date '+%F %T') $*" | tee -a $L; }
until grep -q "PHASE A DONE\|FAILED" runs/phaseA_$TAG.log 2>/dev/null; do sleep 15; done
grep -q FAILED runs/phaseA_$TAG.log && { st "FAILED phase A"; exit 1; }
P=data/prompts/unit_boundary_think_pilot.parquet
st "PHASE B START $MODEL batch $BS"
PIDS=""
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g .venv/bin/python -u scripts/capture.py $P runs/${TAG}_think_pilot_g$g --model "$MODEL" --thinking --max-new-tokens 8192 --batch-size $BS --no-acts --offset $((g*192)) --limit 192 > runs/${TAG}_think_pilot_g$g.log 2>&1 &
  PIDS="$PIDS $!"
done
wait $PIDS
st "think: $(for g in 0 1; do tail -1 runs/${TAG}_think_pilot_g$g.log; done | tr '\n' ' ')"
for k in think nothink; do
  .venv/bin/python runs/merge.py runs/${TAG}_${k}_pilot runs/${TAG}_${k}_pilot_g0 runs/${TAG}_${k}_pilot_g1 >> $L 2>&1 || st "FAILED merge $k"
  OMP_NUM_THREADS=8 .venv/bin/python scripts/unit_boundary.py runs/${TAG}_${k}_pilot figures/${TAG}_${k}_pilot --label "$MODEL $k (pilot, A100)" --n-boot 1000 > runs/ub_${TAG}_$k.log 2>&1 || st "FAILED unit_boundary $k"
done
sed -e "s/qwen3-14b_think_pilot/${TAG}_think_pilot/g" -e "s/qwen3-14b_nothink_pilot/${TAG}_nothink_pilot/g" -e "s/Qwen3-14B/$(basename $MODEL)/g" runs/paired.py > runs/paired_$TAG.py
.venv/bin/python runs/paired_$TAG.py > runs/paired_$TAG.out 2>&1 || st "FAILED paired"
st "analysis done"
rm -rf /workspace/.hf_home/hub/models--$(echo $MODEL | sed 's#/#--#') /workspace/.hf_home/hub/blobs
st "weights freed; disk $(df -h . | tail -1 | awk '{print $4}') free"
st "PHASE B DONE"
[ -n "$NEXT_MODEL" ] && runs/phaseA.sh "$NEXT_MODEL" "$NEXT_TAG" 4
