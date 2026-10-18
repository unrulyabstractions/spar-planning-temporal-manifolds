#!/bin/bash
set -u; cd /workspace/SPAR-2026
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/workspace/.hf_home
st() { echo "$(date '+%F %T') $*" | tee -a runs/setup.log; }
st "SETUP START commit $(cat COMMIT)"
command -v uv >/dev/null || pip install -q uv
uv sync --extra dev > runs/uv_sync.log 2>&1 || { st "FAILED uv sync"; exit 1; }
.venv/bin/python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.device_count(), torch.cuda.get_device_name(0))" | tee -a runs/setup.log
.venv/bin/hf download Qwen/Qwen3-14B > runs/hf_download.log 2>&1 || { st "FAILED download"; exit 1; }
st "download done"
export HF_HUB_OFFLINE=1
.venv/bin/python -m pytest -q tests > runs/tests.log 2>&1 || { st "FAILED tests"; exit 1; }
st "tests: $(tail -1 runs/tests.log)"
mkdir -p data/prompts
.venv/bin/python scripts/gen_prompts.py data/prompts/investment_n16_s1.parquet --n 16 --seed 1 > /dev/null || { st "FAILED gen_prompts"; exit 1; }
.venv/bin/python scripts/gen_unit_boundary.py data/prompts/unit_boundary_n24.parquet > runs/gen_ub.log 2>&1 || { st "FAILED gen_ub"; exit 1; }
.venv/bin/python scripts/select_prompts.py data/prompts/unit_boundary_n24.parquet data/prompts/unit_boundary_think_pilot.parquet \
  --query "D == 10 and ratio == 5 and phrasing_id == 0 and gi % 2 == 1" > runs/select.log 2>&1 || { st "FAILED select"; exit 1; }
st "prompts: $(tail -1 runs/gen_ub.log) | $(tail -1 runs/select.log)"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/verify_capture.py --model Qwen/Qwen3-14B --n 4 > runs/verify.log 2>&1
grep -q "VERIFY OK" runs/verify.log && st "VERIFY OK" || { st "FAILED verify"; exit 1; }
st "SETUP DONE"
