#!/bin/bash
# Full capture + analysis chain for one model on a rented GPU (Vast.ai). Run INSIDE tmux on the instance
# from the repo root:   MODEL=Qwen/Qwen3.5-27B TAG=qwen3.5-27b nohup scripts/vast_full_chain.sh > runs/chain_qwen3.5-27b.out 2>&1 &
# Logs are per tag (runs/chain_${TAG}.log, runs/reproduce_all_${TAG}.log), so several models can run in sequence on one
# instance without a later gate reading an earlier model's "ALL DONE" (scripts/vast_models_sequence.sh).
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
QUOTA=${CPU_QUOTA:-$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)}   # CPU_QUOTA overrides (tests)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TOKENIZERS_PARALLELISM=false
MODEL=${MODEL:-Qwen/Qwen3.5-27B}; TAG=${TAG:-qwen3.5-27b}; export MODEL TAG
export CAP_ARGS=${CAP_ARGS:---batch-size 16}            # single 80 GB card: no --split
export CHECKSUMS=snapshot/checksums_${TAG}.json
PY=.venv/bin/python; LOG=runs/chain_${TAG}.log; mkdir -p runs figures snapshot
export REPRO_LOG=runs/reproduce_all_${TAG}.log
stamp() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
die() { stamp "FAILED: $*"; exit 1; }
stamp "CHAIN START commit $(git rev-parse --short HEAD) model $MODEL tag $TAG; nproc $(nproc), cgroup CPU quota ${QUOTA:-?} cores, OMP_NUM_THREADS $OMP_NUM_THREADS"

DRY=${DRY_RUN:-0}   # 1: test the schedule only — steps 0, 1, 4 skipped, captures/analyses replaced by short sleeps
# 0. environment + model + tests + hook verification
if [ "$DRY" != 1 ]; then
[ -x .venv/bin/python ] || { command -v uv >/dev/null || pip install -q uv; uv sync --extra dev || die "uv sync"; }
$PY -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); import fla" || die "torch/fla import"
.venv/bin/hf download "$MODEL" > runs/hf_download_${TAG}.log 2>&1 || die "model download"
export HF_HUB_OFFLINE=1
$PY -m pytest -q tests > runs/tests.log 2>&1 || die "unit tests (see runs/tests.log)"
stamp "tests: $(tail -1 runs/tests.log)"
mkdir -p data/prompts; [ -f data/prompts/investment_n16_s1.parquet ] || $PY scripts/gen_prompts.py data/prompts/investment_n16_s1.parquet --n 16 --seed 1 > /dev/null || die "gen_prompts"
$PY -u scripts/verify_capture.py --model "$MODEL" --n 4 --anchors > runs/verify_${TAG}.log 2>&1
grep -q "VERIFY OK" runs/verify_${TAG}.log || die "verify_capture did not print VERIFY OK (see runs/verify_${TAG}.log)"
stamp "verify: $(grep -E 'mem per gpu|VERIFY' runs/verify_${TAG}.log | tr '\n' ' ')"

# 1. standard reproduction (datasets, 5 captures, per-run analyses, checksums)
scripts/reproduce_all.sh || die "reproduce_all"
grep -q "ALL DONE" $REPRO_LOG || die "reproduce_all did not finish"
fi

# 2-3. distractor families + long anchored captures, and the later analyses (fractional-depth defaults inside each
# script). To keep the GPU busy (Alan, 2026-10-03), each analysis starts as soon as its inputs exist: group A needs only
# the standard runs + families_s0 and runs in ONE background worker (sequential, $OMP_NUM_THREADS threads each) during the
# long captures; analyze_long starts once matrix_s0_long is captured; late_readout and depth_profile need both long runs.
# Overlap only when the cgroup CPU quota is >= OVERLAP_MIN_CORES (default 12; 0 = no quota): on the 7.68-core 2026-09-29
# instance an analysis beside a capture starved the kernel-launching thread, so there everything runs after the captures.
CAP="--model $MODEL $CAP_ARGS"
M=runs/${TAG}_matrix_s0; F=runs/${TAG}_families_s0; ML=runs/${TAG}_matrix_s0_long; FL=runs/${TAG}_families_s0_long
C=runs/${TAG}_investment_n2000_s0; E=runs/${TAG}_extended_n2000_s0; FIG=figures/${TAG}_matrix_s0
capture() { stamp "capture $1 start"
  if [ "$DRY" = 1 ]; then sleep "${DRY_CAPTURE_S:-2}"; echo "done: dry run" > "runs/$1.log"
  else $PY -u scripts/capture.py "$2" "runs/$1" $CAP $3 > "runs/$1.log" 2>&1 || die "capture $1"; fi
  stamp "capture $1 $(tail -1 runs/$1.log)"; }
run() { local name=$1; shift; stamp "analysis $name start"
  if [ "$DRY" = 1 ]; then sleep "${DRY_ANALYSIS_S:-1}"
  else "$@" > "$FIG/$name.log" 2>&1 || stamp "analysis $name FAILED (continuing)"; fi
  stamp "analysis $name done"; }
group_a() {
  run curve_model        $PY scripts/curve_model.py $M $FIG/curve --extended-run $E
  run relevance_decoder  $PY scripts/relevance_decoder.py $M $FIG/relevance --sweep
  run relevance_families $PY scripts/relevance_decoder.py $M $FIG/relevance --families-run $F
  run scaling_test       $PY scripts/scaling_test.py $M $F $FIG/scaling
  run guttman_canonical  $PY scripts/guttman_check.py $C
  run guttman_extended   $PY scripts/guttman_check.py $E --cells 0.55L:R0,0.92L:T3 --window 11
  run leace_check        $PY scripts/leace_check.py $M $FIG/leace --sweep --canonical-run $C
  run pls_check          $PY scripts/pls_check.py $M $FIG/pls --canonical-run $C --extended-run $E
  run stakes_pilot       $PY scripts/stakes_pilot.py figures/stakes_pilot_${TAG} --canonical-run $C --matrix-run $M
}
late_b() {
  run late_readout       $PY scripts/late_readout.py $ML $FL figures/${TAG}_matrix_s0_long
  local NL; if [ "$DRY" = 1 ]; then NL=64; else NL=$($PY -c "import json; print(json.load(open('$ML/meta.json'))['n_layers'])"); fi
  run depth_profile      $PY scripts/late_readout.py $ML $FL figures/${TAG}_matrix_s0_long/depth --layers "$(seq -s, 2 2 $NL)" --sites T3,R0,MEAN
}
mkdir -p $FIG figures/${TAG}_matrix_s0_long figures/stakes_pilot_${TAG}
OVERLAP_MIN_CORES=${OVERLAP_MIN_CORES:-12}
if [ "${QUOTA:-0}" = 0 ] || [ "${QUOTA:-0}" -ge "$OVERLAP_MIN_CORES" ]; then
  stamp "schedule: overlap (CPU quota ${QUOTA:-0} cores; 0 = none) — later analyses beside the remaining captures"
  capture ${TAG}_families_s0 data/prompts/matrix/families_s0.parquet ""
  group_a & PA=$!
  capture ${TAG}_matrix_s0_long data/prompts/matrix/capture_s0.parquet "--max-new-tokens 160 --anchors"
  run analyze_long scripts/analyze_run.sh $ML & PB=$!
  capture ${TAG}_families_s0_long data/prompts/matrix/families_s0.parquet "--max-new-tokens 160 --anchors"
  stamp "captures done"
  late_b
  wait $PA $PB
else
  stamp "schedule: serial (CPU quota ${QUOTA} cores < $OVERLAP_MIN_CORES) — analyses after the captures"
  capture ${TAG}_families_s0 data/prompts/matrix/families_s0.parquet ""
  capture ${TAG}_matrix_s0_long data/prompts/matrix/capture_s0.parquet "--max-new-tokens 160 --anchors"
  capture ${TAG}_families_s0_long data/prompts/matrix/families_s0.parquet "--max-new-tokens 160 --anchors"
  stamp "captures done"
  group_a; late_b; run analyze_long scripts/analyze_run.sh $ML
fi
stamp "later analyses done"

# 4. tier-1 subsets (7 standard depths × all positions per run) + checksums of everything
if [ "$DRY" != 1 ]; then
for R in $C $M $E runs/${TAG}_phrasings_n2000_s0 runs/${TAG}_units_paired_n2000_s1 $F $ML $FL; do
  stamp "export subset $R"; $PY scripts/export_subset.py $R --layers 0.35L,0.45L,0.55L,0.65L,0.72L,0.825L,0.92L --positions "$($PY -c "import json; print(','.join(json.load(open('$R/meta.json'))['position_labels']))")" > /dev/null 2>&1 || stamp "export $R FAILED"
done
$PY scripts/run_checksums.py runs/${TAG}_*/ --out snapshot/checksums_${TAG}_all.json 2>&1 | grep -v Warning | tail -3 | tee -a $LOG
for R in runs/${TAG}_*/; do ( cd $R && sha256sum index.parquet meta.json subset.json acts_subset_*.safetensors > SHA256SUMS.tier1 && sha256sum acts_[0-9]*.safetensors > SHA256SUMS.shards ); done
stamp "tier-1 size: $(du -shc runs/${TAG}_*/acts_subset_* figures | tail -1)"
fi
stamp "CHAIN DONE"
