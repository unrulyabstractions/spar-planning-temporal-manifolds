#!/bin/bash
# Option-delay-unit control (2x2: delays in years/months x horizon in months/decimal years, matched durations, n = 1200).
# The years-delay cells are runs/qwen3-14b_matched_months_dwm_n1200 (months rows) and runs/qwen3-14b_matched_months_years_dec2_n1200.
cd "$(dirname "$0")/.."
.venv/bin/python -u scripts/intrinsic_dim.py runs/qwen3-14b_matched_months_months_dm_n1200 figures/qwen3-14b_matched_months_months_dm_n1200 \
  --label "Qwen3-14B, month delays, horizon months" > runs/intrinsic_dim_delay_months_H_months.log 2>&1 &
.venv/bin/python -u scripts/intrinsic_dim.py runs/qwen3-14b_matched_months_years_dec2_dm_n1200 figures/qwen3-14b_matched_months_years_dec2_dm_n1200 \
  --label "Qwen3-14B, month delays, horizon decimal years" > runs/intrinsic_dim_delay_months_H_years.log 2>&1 &
wait
echo DELAY UNIT DONE
