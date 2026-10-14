#!/usr/bin/env bash
# Run every experiment for both thinking modes. E0 first (others read its
# manifest), E5 thinking off only. Stops on the first failure.
#
#   bash experiments/run_all.sh                       # Qwen3-8B, full bank
#   MODEL=Qwen/Qwen3.5-0.8B LIMIT=6 bash experiments/run_all.sh   # local smoke
#   MODES="off" bash experiments/run_all.sh           # one mode
#   MAX_THINK=4096 MODES=on bash experiments/run_all.sh  # thinking budget
#   BANK=full MODES=on bash experiments/run_all.sh    # the full bank (E0 only until E1-E4 are ported)
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
[ -n "${BANK:-}" ] && EXTRA+=(--bank "$BANK")
[ -n "${N_PROMPTS:-}" ] && EXTRA+=(--n-prompts "$N_PROMPTS")
EXPS="${EXPS:-exp0_baseline exp1_nuisance exp2_paraphrase exp3_units exp4_perturb}"
[ "${BANK:-starter}" = full ] && EXPS="exp0_baseline"

run() {
  local t0=$SECONDS
  echo; echo "=================== $* ==================="
  "$PY" "$@" "${EXTRA[@]}"
  echo ">> done in $((SECONDS - t0))s"
}

for mode in $MODES; do
  for exp in $EXPS; do
    skip=(); [ "$exp" = exp0_baseline ] && [ "$mode" = on ] && skip=(--skip-behavior)
    run "experiments/$exp.py" --model "$MODEL" --thinking "$mode" "${skip[@]}"
  done
  if [ "$mode" = off ] && [ "${BANK:-starter}" = starter ]; then
    run experiments/exp5_attribution.py --model "$MODEL" --thinking off
  fi
done
echo; echo "all experiments finished"
