#!/bin/bash
# Full capture + analysis chain for one model on a rented GPU (Vast.ai). Run INSIDE tmux on the instance
# from the repo root:   MODEL=Qwen/Qwen3.5-27B TAG=qwen3.5-27b nohup scripts/vast_full_chain.sh > runs/chain.log 2>&1 &
# Steps (each gated on the previous): environment, model download, unit tests, hook verification (must print
# VERIFY OK), the standard reproduction (scripts/reproduce_all.sh: 5 datasets/captures + per-run analyses +
# checksums), the distractor-family and long anchored captures, the later analyses (curve model, relevance
# decoder, scaling test, late readout, depth profile, Guttman, LEACE, PLS/RRR, stakes), tier-1 subset export
# and checksums. Cells are fractional-depth specs, so nothing here assumes a 40-layer model.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Containers advertise nproc = host cores but run under a cgroup CPU quota (cat /sys/fs/cgroup/cpu.max; 7.68 cores on the
# 2026-09-29 A100 instance). Default BLAS/OpenMP pools sized to nproc oversubscribe the quota and starve the thread that
# launches GPU kernels (capture fell from 1.2 to 4–11 s/prompt). Cap every process to a few threads.
QUOTA=$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TOKENIZERS_PARALLELISM=false
MODEL=${MODEL:-Qwen/Qwen3.5-27B}; TAG=${TAG:-qwen3.5-27b}; export MODEL TAG
export CAP_ARGS=${CAP_ARGS:---batch-size 16}            # single 80 GB card: no --split
export CHECKSUMS=snapshot/checksums_${TAG}.json
PY=.venv/bin/python; LOG=runs/chain.log; mkdir -p runs figures snapshot
stamp() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
die() { stamp "FAILED: $*"; exit 1; }
stamp "CHAIN START commit $(git rev-parse --short HEAD) model $MODEL tag $TAG; nproc $(nproc), cgroup CPU quota ${QUOTA:-?} cores, OMP_NUM_THREADS $OMP_NUM_THREADS"

# 0. environment + model + tests + hook verification
[ -x .venv/bin/python ] || { command -v uv >/dev/null || pip install -q uv; uv sync --extra dev || die "uv sync"; }
$PY -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); import fla" || die "torch/fla import"
.venv/bin/hf download "$MODEL" > runs/hf_download.log 2>&1 || die "model download"
export HF_HUB_OFFLINE=1
$PY -m pytest -q tests > runs/tests.log 2>&1 || die "unit tests (see runs/tests.log)"
stamp "tests: $(tail -1 runs/tests.log)"
mkdir -p data/prompts; [ -f data/prompts/investment_n16_s1.parquet ] || $PY scripts/gen_prompts.py data/prompts/investment_n16_s1.parquet --n 16 --seed 1 > /dev/null || die "gen_prompts"
$PY -u scripts/verify_capture.py --model "$MODEL" --n 4 --anchors > runs/verify_${TAG}.log 2>&1
grep -q "VERIFY OK" runs/verify_${TAG}.log || die "verify_capture did not print VERIFY OK (see runs/verify_${TAG}.log)"
stamp "verify: $(grep -E 'mem per gpu|VERIFY' runs/verify_${TAG}.log | tr '\n' ' ')"

# 1. standard reproduction (datasets, 5 captures, per-run analyses, checksums)
scripts/reproduce_all.sh || die "reproduce_all"
grep -q "ALL DONE" runs/reproduce_all.log || die "reproduce_all did not finish"

# 2. distractor families (short) + long anchored captures (matrix, families)
CAP="--model $MODEL $CAP_ARGS"
capture() { stamp "capture $1 start"; $PY -u scripts/capture.py "$2" "runs/$1" $CAP $3 > "runs/$1.log" 2>&1 || die "capture $1"; stamp "capture $1 $(tail -1 runs/$1.log)"; }
capture ${TAG}_families_s0 data/prompts/matrix/families_s0.parquet ""
capture ${TAG}_matrix_s0_long data/prompts/matrix/capture_s0.parquet "--max-new-tokens 160 --anchors"
capture ${TAG}_families_s0_long data/prompts/matrix/families_s0.parquet "--max-new-tokens 160 --anchors"

# 3. later analyses (fractional-depth defaults inside each script)
M=runs/${TAG}_matrix_s0; F=runs/${TAG}_families_s0; ML=runs/${TAG}_matrix_s0_long; FL=runs/${TAG}_families_s0_long
C=runs/${TAG}_investment_n2000_s0; E=runs/${TAG}_extended_n2000_s0; FIG=figures/${TAG}_matrix_s0
run() { local name=$1; shift; stamp "analysis $name start"; "$@" > "$FIG/$name.log" 2>&1 || stamp "analysis $name FAILED (continuing)"; stamp "analysis $name done"; }
mkdir -p $FIG figures/${TAG}_matrix_s0_long figures/stakes_pilot_${TAG}
run curve_model        $PY scripts/curve_model.py $M $FIG/curve --extended-run $E
run relevance_decoder  $PY scripts/relevance_decoder.py $M $FIG/relevance --sweep
run relevance_families $PY scripts/relevance_decoder.py $M $FIG/relevance --families-run $F
run scaling_test       $PY scripts/scaling_test.py $M $F $FIG/scaling
run guttman_canonical  $PY scripts/guttman_check.py $C
run guttman_extended   $PY scripts/guttman_check.py $E --cells 0.55L:R0,0.92L:T3 --window 11
run leace_check        $PY scripts/leace_check.py $M $FIG/leace --sweep --canonical-run $C
run pls_check          $PY scripts/pls_check.py $M $FIG/pls --canonical-run $C --extended-run $E
run stakes_pilot       $PY scripts/stakes_pilot.py figures/stakes_pilot_${TAG} --canonical-run $C --matrix-run $M
run late_readout       $PY scripts/late_readout.py $ML $FL figures/${TAG}_matrix_s0_long
NL=$($PY -c "import json; print(json.load(open('$ML/meta.json'))['n_layers'])")
run depth_profile      $PY scripts/late_readout.py $ML $FL figures/${TAG}_matrix_s0_long/depth --layers "$(seq -s, 2 2 $NL)" --sites T3,R0,MEAN
run analyze_long       scripts/analyze_run.sh $ML

# 4. tier-1 subsets (7 standard depths × all positions per run) + checksums of everything
for R in $C $M $E runs/${TAG}_phrasings_n2000_s0 runs/${TAG}_units_paired_n2000_s1 $F $ML $FL; do
  stamp "export subset $R"; $PY scripts/export_subset.py $R --layers 0.35L,0.45L,0.55L,0.65L,0.72L,0.825L,0.92L --positions "$($PY -c "import json; print(','.join(json.load(open('$R/meta.json'))['position_labels']))")" > /dev/null 2>&1 || stamp "export $R FAILED"
done
$PY scripts/run_checksums.py runs/${TAG}_*/ --out snapshot/checksums_${TAG}_all.json 2>&1 | grep -v Warning | tail -3 | tee -a $LOG
for R in runs/${TAG}_*/; do ( cd $R && sha256sum index.parquet meta.json subset.json acts_subset_*.safetensors > SHA256SUMS.tier1 && sha256sum acts_[0-9]*.safetensors > SHA256SUMS.shards ); done
stamp "tier-1 size: $(du -shc runs/${TAG}_*/acts_subset_* figures | tail -1)"
stamp "CHAIN DONE"
