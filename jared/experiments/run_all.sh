#!/usr/bin/env bash
# Run every experiment for both thinking modes. E0 first (others read its
# manifest), E5 thinking off only. Stops on the first failure.
#
#   bash experiments/run_all.sh                       # Qwen3-8B, full bank
#   MODEL=Qwen/Qwen3.5-0.8B LIMIT=6 bash experiments/run_all.sh   # local smoke
#   MODES="off" bash experiments/run_all.sh           # one mode
#   MAX_THINK=4096 MODES=on bash experiments/run_all.sh  # thinking budget
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-8B}"
MODES="${MODES:-off on}"
LIMIT="${LIMIT:-}"
PY="${PY:-python}"
EXTRA=()
[ -n "$LIMIT" ] && EXTRA+=(--limit "$LIMIT")
[ -n "${OUT:-}" ] && EXTRA+=(--out "$OUT")
[ -n "${MAX_THINK:-}" ] && EXTRA+=(--max-think-tokens "$MAX_THINK")

run() {
  local t0=$SECONDS
  echo; echo "=================== $* ==================="
  "$PY" "$@" "${EXTRA[@]}"
  echo ">> done in $((SECONDS - t0))s"
}

for mode in $MODES; do
  for exp in exp0_baseline exp1_nuisance exp2_paraphrase exp3_units exp4_perturb; do
    skip=(); [ "$exp" = exp0_baseline ] && [ "$mode" = on ] && skip=(--skip-behavior)
    run "experiments/$exp.py" --model "$MODEL" --thinking "$mode" "${skip[@]}"
  done
  if [ "$mode" = off ]; then
    run experiments/exp5_attribution.py --model "$MODEL" --thinking off
  fi
done
echo; echo "all experiments finished"
