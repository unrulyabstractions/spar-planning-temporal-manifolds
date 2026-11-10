#!/bin/bash
# Data-driven cells: pick the layers where the canonical run's sweep peaks at T3 and R0 (|rho(PC1, log H)|), and
# rerun the cell-based analyses there (in addition to, not instead of, the fixed fractional-depth cells). Also picks
# the overall peak cell. Output goes to figures/<TAG>_matrix_s0/peak/ and figures/<TAG>_investment_n2000_s0/peak/.
# Usage: scripts/peak_cells.sh TAG        (e.g. qwen3.5-27b; needs figures/<TAG>_investment_n2000_s0/sweep.csv)
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
TAG=${1:?TAG}; PY=.venv/bin/python
C=runs/${TAG}_investment_n2000_s0; M=runs/${TAG}_matrix_s0; E=runs/${TAG}_extended_n2000_s0; F=runs/${TAG}_families_s0
ML=runs/${TAG}_matrix_s0_long; FL=runs/${TAG}_families_s0_long; OUT=figures/${TAG}_matrix_s0/peak; mkdir -p $OUT figures/${TAG}_investment_n2000_s0/peak
read LT3 LR0 LTOP PTOP <<< $($PY - <<PYEOF
import pandas as pd
s = pd.read_csv("figures/${TAG}_investment_n2000_s0/sweep.csv"); s["abs"] = s.rho_pc1_horizon.abs(); s = s.dropna(subset=["abs"])   # layer 0 (embeddings) is NaN and would sort last
t3 = s[s.position == "T3"].sort_values("abs").iloc[-1]; r0 = s[s.position == "R0"].sort_values("abs").iloc[-1]; top = s.sort_values("abs").iloc[-1]
print(int(t3.layer), int(r0.layer), int(top.layer), top.position)
PYEOF
)
NL=$($PY -c "import json; print(json.load(open('$C/meta.json'))['n_layers'])")
CELLS="$LT3:T3,$LR0:R0,$LTOP:$PTOP"
echo "peak cells for $TAG (n_layers $NL): T3 → L$LT3, R0 → L$LR0, overall → L$LTOP $PTOP  ($CELLS)" | tee $OUT/peak_cells.txt
run() { local name=$1; shift; echo "=== $name"; "$@" > "$OUT/$name.log" 2>&1 || echo "   FAILED (see $OUT/$name.log)"; }
run geometry_path     $PY scripts/geometry_path.py $C figures/${TAG}_investment_n2000_s0/peak --cells "$CELLS" --no-sweep
run guttman_canonical $PY scripts/guttman_check.py $C --cells "$CELLS"
run guttman_extended  $PY scripts/guttman_check.py $E --cells "$CELLS" --window 11
run curve_model       $PY scripts/curve_model.py $M $OUT/curve --extended-run $E --cells "$LT3:T3,$LR0:R0"
run relevance_decoder $PY scripts/relevance_decoder.py $M $OUT/relevance --cells "$LT3:T3,$LR0:R0"
run relevance_families $PY scripts/relevance_decoder.py $M $OUT/relevance --cells "$LT3:T3,$LR0:R0" --families-run $F
run scaling_test      $PY scripts/scaling_test.py $M $F $OUT/scaling --cells "$LT3:T3,$LR0:R0"
run leace_check       $PY scripts/leace_check.py $M $OUT/leace --cells "$CELLS" --canonical-run $C
run pls_check         $PY scripts/pls_check.py $M $OUT/pls --cells "$CELLS" --canonical-run $C --extended-run $E
run within_bin        $PY scripts/within_bin_control.py $C --cells "$CELLS"
[ -d $ML ] && run late_readout $PY scripts/late_readout.py $ML $FL $OUT/late --layers "$LT3,$LR0,$LTOP"
echo "PEAK CELLS DONE"
