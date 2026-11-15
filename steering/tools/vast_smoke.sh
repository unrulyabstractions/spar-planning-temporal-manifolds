#!/usr/bin/env bash
# End-to-end smoke of the steering pipeline on a rented GPU box (vast.ai), one stage at a time:
#
#   bash steering/tools/vast_smoke.sh STAGE [STAGE ...]     # from the repository root, on the box
#
# Stages, in order (each stops the script on failure):
#   crlf      refuse CRLF line endings in *.sh / *.py (a Windows checkout broke an earlier vast setup)
#   setup     install uv if missing; `uv sync --frozen` with each README's extras
#   weights   download the model, and what the tests load, into the Hugging Face cache (needs HF_TOKEN; never printed)
#   tests     both pytest suites
#   hook      tools/check_hook.py on the model at $LAYER (zero edit, roles, effect)
#   baseline  baseline.py on the first $N extraction prompts, thinking on
#   extract   extract.py at $LAYER
#   audit     prints every label site with the decoded tokens around it, for a hand audit
#   fit       fit.py at $LAYER (refuses, and logs the refusal, if the slice is too small)
#   bundle    bundle.py: every method
#   method    tools/check_hook.py --method on caa, curve-displace and curve-snap
#   sums      SHA256SUMS of the outputs, for the copy-back
#
# Settings (environment): MODEL (google/gemma-4-31B-it), LAYER (38), N (64), BATCH (8), MAX_NEW (4096),
# RUN (smoke). Outputs: prompt_bank/runs/<model>__<run>, steering/activations/<model>__<run>,
# steering/fits/<model>__<run>, steering/methods/<model>. Every line is flushed; run under nohup or tmux.
set -euo pipefail

MODEL=${MODEL:-google/gemma-4-31B-it}
LAYER=${LAYER:-38}
N=${N:-64}
BATCH=${BATCH:-8}     # 16 runs out of memory on an 80 GB GPU with Gemma 4 31B at 4096 new tokens
MAX_NEW=${MAX_NEW:-4096}
RUN=${RUN:-smoke}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SHORT=${MODEL##*/}
NAME=${SHORT}__${RUN}
BANK=$ROOT/prompt_bank/data/t0_s0
RUNDIR=$ROOT/prompt_bank/runs/$NAME
ACTS=$ROOT/steering/activations/$NAME
FITS=$ROOT/steering/fits/$NAME
export PYTHONUNBUFFERED=1
export PATH=$HOME/.local/bin:$PATH

say() { printf '\n=== %s  [%s UTC]\n' "$*" "$(date -u +%H:%M:%S)"; }
pb() { (cd "$ROOT/prompt_bank" && .venv/bin/python -u "$@"); }
st() { (cd "$ROOT/steering" && .venv/bin/python -u "$@"); }

stage_crlf() {
    say "crlf: refuse CRLF line endings"
    local bad
    bad=$(cd "$ROOT" && grep -rlI $'\r' --include='*.sh' --include='*.py' --include='justfile' \
          prompt_bank steering 2>/dev/null || true)
    if [ -n "$bad" ]; then echo "CRLF line endings in:"; echo "$bad"; exit 1; fi
    echo "no CRLF in prompt_bank/ or steering/ scripts"
}

stage_setup() {
    say "setup"
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv --version
    (cd "$ROOT/prompt_bank" && uv sync --frozen --all-extras)   # as its README: torch, pytest, ...
    (cd "$ROOT/steering" && uv sync --frozen --extra dev)            # as its README
    st -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'available', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('transformers', transformers.__version__)"
}

stage_weights() {
    say "weights: $MODEL"
    [ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is not set (the model is gated)"; exit 1; }
    local t0=$SECONDS
    # Independent of `setup`, so the two can run in parallel. Pass the token without writing it to disk, e.g.
    #   printf '%s\n' "$TOKEN" | ssh BOX 'read -r HF_TOKEN; export HF_TOKEN; nohup bash .../vast_smoke.sh weights ... &'
    local hf=(uvx --from "huggingface_hub[hf_xet]==2.4.0" hf download)
    HF_XET_HIGH_PERFORMANCE=1 "${hf[@]}" "$MODEL"
    # What the test suites load; without these the real-model tests SKIP (they need the cache, not the network).
    "${hf[@]}" Qwen/Qwen3.5-0.8B
    "${hf[@]}" Qwen/Qwen3.5-2B --include '*.json' '*.jinja' '*.txt'
    "${hf[@]}" Qwen/Qwen3.5-27B --include '*.json' '*.jinja' '*.txt'
    echo "downloaded in $((SECONDS - t0)) s"
}

stage_tests() {
    say "tests"
    (cd "$ROOT/prompt_bank" && .venv/bin/python -m pytest -q 2>&1 | tail -3)
    (cd "$ROOT/steering" && .venv/bin/python -m pytest -q 2>&1 | tail -3)
}

stage_hook() {
    say "hook: check_hook at layer $LAYER"
    st tools/check_hook.py --model "$MODEL" --layer "$LAYER" --bank "$BANK" --n 8 --max-new-tokens 64
}

stage_baseline() {
    say "baseline: first $N extraction prompts, batch $BATCH, max_new_tokens $MAX_NEW"
    pb baseline.py --bank data/t0_s0 --model "$MODEL" --split extraction --limit "$N" --batch-size "$BATCH" \
        --max-new-tokens "$MAX_NEW" --out "$RUNDIR"
}

stage_extract() {
    say "extract: layer $LAYER"
    st extract.py --bank "$BANK" --run "$RUNDIR" --layers "$LAYER" --out "$ACTS"
}

stage_audit() {
    say "audit: every label site, decoded in context (>>label<<)"
    st - "$RUNDIR" "$ACTS" "$MODEL" <<'EOF'
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
run, acts, model = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
tok = AutoTokenizer.from_pretrained(model)
gen = {}
for line in (run / "responses.jsonl").open():
    r = json.loads(line)
    gen[r["uid"]] = r["generated_ids"]
rows = [json.loads(l) for l in (acts / "index.jsonl").open()]
bad = 0
for r in rows:
    g = gen[r["uid"]]
    off = r["label_index"]                     # an index into the generated ids (steer/activations.py)
    at = tok.decode(g[off:off + 1])
    ctx = tok.decode(g[max(0, off - 8):off]) + ">>" + at + "<<" + tok.decode(g[off + 1:off + 3])
    # Checked against the DECODED token, not only extract's own label_token field.
    ok = (r["replay_predicts_label"] and at == r["label_token"]
          and at.strip() == r["choice_label"].rstrip(")").strip())
    bad += not ok
    print(f"{r['uid']} {r['choice_label']:4s} {r['chosen_option']:5s} margin {r['replay_label_margin']:+7.3f} "
          f"{'ok ' if ok else 'BAD'} {ctx!r}")
skipped = acts / "skipped.jsonl"
n_skip = sum(1 for _ in skipped.open()) if skipped.exists() else 0
print(f"{len(rows)} label sites, {bad} not matching the parsed choice; {n_skip} skipped (see skipped.jsonl)")
sys.exit(1 if bad else 0)
EOF
}

stage_fit() {
    say "fit: layer $LAYER"
    st fit.py --activations "$ACTS" --bank "$BANK" --layer "$LAYER"
}

stage_bundle() {
    say "bundle"
    st bundle.py --fit "$FITS/default/L${LAYER}_label" --activations "$ACTS"
}

stage_method() {
    local m d
    for m in caa curve-displace curve-snap; do
        d=$(ls -d "$ROOT/steering/methods/$SHORT/${m}__${NAME}__default__L${LAYER}_label" 2>/dev/null || true)
        [ -n "$d" ] || { echo "no $m bundle"; exit 1; }
        say "method: check_hook --method $m"
        st tools/check_hook.py --method "$d" --bank "$BANK" --n 4 --max-new-tokens 64
    done
}

stage_sums() {
    say "sums"
    (cd "$ROOT" && find "prompt_bank/runs/$NAME" "steering/activations/$NAME" "steering/fits/$NAME" \
        "steering/methods/$SHORT" -type f 2>/dev/null | sort | xargs sha256sum > "$ROOT/vast_smoke.SHA256SUMS")
    wc -l < "$ROOT/vast_smoke.SHA256SUMS"
}

[ $# -gt 0 ] || { sed -n 2,24p "$0"; exit 2; }
for s in "$@"; do
    declare -F "stage_$s" >/dev/null || { echo "unknown stage: $s"; exit 2; }
    "stage_$s"
done
say "done: $*"
