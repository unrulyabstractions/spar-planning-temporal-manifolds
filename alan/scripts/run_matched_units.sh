#!/bin/bash
# per-unit and pooled intrinsic-dimension analyses of the matched-duration sets
cd /home/alan/SPAR-2026
A=runs/qwen3-14b_matched_months_dwm_n1200; B=runs/qwen3-14b_matched_years_dwmy_n100
FA=figures/qwen3-14b_matched_months_dwm_n1200; FB=figures/qwen3-14b_matched_years_dwmy_n100
for u in days weeks months; do
  .venv/bin/python -u scripts/intrinsic_dim.py $A $FA/$u --query "horizon_unit == '$u'" --label "Qwen3-14B, matched months, $u" > runs/intrinsic_dim_matched_A_$u.log 2>&1 &
done
.venv/bin/python -u scripts/intrinsic_dim.py $A $FA/pooled --factors horizon_unit --group horizon_unit --label "Qwen3-14B, matched months, pooled" > runs/intrinsic_dim_matched_A_pooled.log 2>&1 &
wait
for u in days weeks months years; do
  .venv/bin/python -u scripts/intrinsic_dim.py $B $FB/$u --query "horizon_unit == '$u'" --label "Qwen3-14B, matched years, $u" > runs/intrinsic_dim_matched_B_$u.log 2>&1 &
done
.venv/bin/python -u scripts/intrinsic_dim.py $B $FB/pooled --factors horizon_unit --group horizon_unit --label "Qwen3-14B, matched years, pooled" > runs/intrinsic_dim_matched_B_pooled.log 2>&1 &
wait
echo MATCHED ANALYSES DONE
