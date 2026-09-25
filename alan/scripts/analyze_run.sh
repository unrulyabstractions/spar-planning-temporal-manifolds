#!/bin/bash
# Standard post-capture analysis for one run: sweep + figures, path geometry, hypothesis checks,
# and interactive 3-D views. Cells are chosen at fractional depths 0.55L / 0.72L / 0.92L
# (14B: L22/L29/L37; 8B: L20/L26/L33) at positions R0, R0, T3 plus T3 at 0.55L.
# Usage: scripts/analyze_run.sh RUN_DIR [FIGURES_DIR]
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
RUN=${1:?RUN_DIR}; OUT=${2:-figures/$(basename "$RUN")}
L=$(.venv/bin/python -c "import json,sys; print(json.load(open('$RUN/meta.json'))['n_layers'])")
A=$(python3 -c "print(round(0.55*$L))"); B=$(python3 -c "print(round(0.72*$L))"); C=$(python3 -c "print(round(0.92*$L))")
CELLS="$A:R0,$B:R0,$C:T3,$A:T3"
echo "=== analyze_run: $RUN -> $OUT (L=$L, cells $CELLS) ==="
echo "=== sweep ==="
.venv/bin/python scripts/analyze.py "$RUN" "$OUT" --top 4 2>&1 | grep -v -E "Warning|^layer"
echo "=== path geometry ==="
.venv/bin/python scripts/geometry_path.py "$RUN" "$OUT" --cells "$CELLS" --sweep-positions T3,R0 2>&1 | grep -E "^===|planarity|scree|cumulative|tortuosity|split-half|spacing per|per-layer|^  (T3|R0):|^wrote"
echo "=== hypothesis checks ==="
.venv/bin/python scripts/hypothesis_checks.py "$RUN" --sink-layers "$A,$B,$C" --ushape-cells "$A:R0,$A:T3,$B:R0,$C:T3" --ushape-range 0.0027,100 2>&1 | grep -v Warning
echo "=== 3-D views ==="
.venv/bin/python scripts/plot3d.py "$RUN" "$OUT" --cells "$CELLS" --basis centroid 2>&1 | grep -v Warning
.venv/bin/python scripts/plot3d.py "$RUN" "$OUT" --cells "$A:R0" --basis sample 2>&1 | grep -v Warning
echo "ANALYZE_RUN DONE"
