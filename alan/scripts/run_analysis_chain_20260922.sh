#!/bin/bash
# Analysis chain for 2026-09-22: 8B sweep + path geometry now; paired-unit analysis when its capture lands.
cd /home/alan/SPAR-2026
export HF_HUB_OFFLINE=1
echo "=== 8B sweep ==="
.venv/bin/python scripts/analyze.py runs/qwen3-8b_investment_n2000_s0 figures/qwen3-8b_investment_n2000_s0 --top 3 2>&1 | grep -v -E "Warning|^layer"
echo "=== 8B path geometry ==="
.venv/bin/python scripts/geometry_path.py runs/qwen3-8b_investment_n2000_s0 figures/qwen3-8b_investment_n2000_s0 --cells 20:R0,33:T3,25:R0 --sweep-positions T3,R0 2>&1 | grep -E "^===|planarity|tortuosity|split-half|spacing per|^wrote"
echo "=== 8B hypothesis checks ==="
.venv/bin/python scripts/hypothesis_checks.py runs/qwen3-8b_investment_n2000_s0 --sink-layers 13,20,26,33 --ushape-cells 20:R0,20:T3,26:R0,33:T3 2>&1 | grep -v Warning
echo "=== waiting for paired capture ==="
until grep -q -E "^done:|Traceback|Error|Killed" runs/qwen3-14b_units_paired_n2000_s1.log 2>/dev/null; do sleep 20; done
tail -1 runs/qwen3-14b_units_paired_n2000_s1.log
echo "=== paired analysis ==="
.venv/bin/python scripts/analyze_paired.py runs/qwen3-14b_units_paired_n2000_s1 figures/qwen3-14b_units_paired_n2000_s1 2>&1 | grep -v Warning
echo "ANALYSIS CHAIN DONE"
