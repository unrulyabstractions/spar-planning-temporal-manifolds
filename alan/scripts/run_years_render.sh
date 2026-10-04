#!/bin/bash
# Digit-count control: years 1..1200 rendered plain / zero-padded to 4 digits / with 2 decimals, each analysed on its own,
# on all values and on values >= 10 (padded single digits are sometimes read as thousands: progress-20261002.md).
cd "$(dirname "$0")/.."
R=runs/qwen3-14b_years_render_n1200; F=figures/qwen3-14b_years_render_n1200
declare -A Q=([plain]="rendering.isnull()" [pad4]="rendering == 'pad4'" [dec2]="rendering == 'dec2'")
for r in plain pad4 dec2; do
  .venv/bin/python -u scripts/intrinsic_dim.py $R $F/$r --query "${Q[$r]}" --label "Qwen3-14B, years 1-1200, $r" > runs/intrinsic_dim_years_$r.log 2>&1 &
  .venv/bin/python -u scripts/intrinsic_dim.py $R $F/${r}_ge10 --query "${Q[$r]} and horizon_value >= 10" --label "Qwen3-14B, years 10-1200, $r" > runs/intrinsic_dim_years_${r}_ge10.log 2>&1 &
done
wait
.venv/bin/python -u scripts/intrinsic_dim.py runs/qwen3-14b_matched_months_dwm_n1200 figures/qwen3-14b_matched_months_dwm_n1200/months_ge10 \
  --query "horizon_unit == 'months' and horizon_value >= 10" --label "Qwen3-14B, matched months, months >= 10" > runs/intrinsic_dim_matched_A_months_ge10.log 2>&1
echo YEARS RENDER DONE
