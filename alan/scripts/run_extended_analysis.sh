#!/bin/bash
# Waits for a capture to finish, then runs the standard analysis (scripts/analyze_run.sh).
# Usage: scripts/run_extended_analysis.sh RUN_DIR   (kept under its original name; works for any run)
cd "$(dirname "$0")/.."
RUN=${1:-runs/qwen3-14b_extended_n2000_s0}
until grep -q -E "^done:|Traceback|Error|Killed" "$RUN.log" 2>/dev/null; do sleep 20; done
tail -1 "$RUN.log"
scripts/analyze_run.sh "$RUN"
