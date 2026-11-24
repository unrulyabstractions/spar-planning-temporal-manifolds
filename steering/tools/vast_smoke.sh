#!/usr/bin/env bash
# End-to-end smoke of the steering pipeline on a rented GPU box (vast.ai), one stage at a time:
#
#   bash steering/tools/vast_smoke.sh STAGE [STAGE ...]     # from the repository root, on the box
#
# Before copying the code to the box, in the git checkout:  (cd prompt_bank && python -m pbank.provenance stamp)
# That writes CODE_STAMP.json, which lets every output record its code commit on a box without .git.
#
# Stages, in order (each stops the script on failure):
#   crlf      refuse CRLF line endings in *.sh / *.py (a Windows checkout broke an earlier vast setup)
#   version   the code commit the outputs will record; fails if it is "unknown" (no git and no CODE_STAMP.json)
#   setup     install uv if missing; `uv sync --frozen` with each README's extras
#   weights   download the model, and what the tests load, into the Hugging Face cache (needs HF_TOKEN; never printed)
#   tests     both pytest suites
#   hook      tools/check_hook.py on the model at $LAYER (zero edit, roles, effect), with the run's decoding
#   horizon   extract_horizon.py: activations at the horizon's tokens, all layers, ALL 700 prompts (needs no baseline)
#   baseline  baseline.py on the first $N prompts of $SPLIT (all of them with N=all); resumable
#   extract   extract.py for the same split: layers $LAYERS (default all), every site (label, prelabel, prompt_end,
#             think_end), and the top next-token logits at the prelabel site
#   audit     prints every label site with the decoded tokens around it, for a hand audit
#   draws     ../evaluation/shared_draws.py: the shared random directions for $LAYER, at each site of $SITES
#   fit       fit.py at $LAYER, at each site of $SITES, with the shared draws if the `draws` stage wrote them
#             (refuses, and logs the refusal, if the slice is too small)
#   bundle    bundle.py: every method each fit has
#   method    tools/check_hook.py --method on this run's bundles: every label-site bundle on short replies, and the
#             $PRELABEL_CHECK prelabel-site bundles with the edit at the prelabel token, on whole replies. It writes
#             nothing, so `sums` can come before it.
#   sums      SHA256SUMS of the outputs, for the copy-back
#
# A smoke run is the default (the first 64 extraction prompts). The full run is N=all SPLIT=all: all 700 prompts,
# and both splits extracted, which the evaluation runner needs (it reads the verification prompts' label positions).
#
# Fits at other layers or sites need no GPU: extraction keeps every layer, and fit.py / bundle.py run anywhere.
#
# Settings (environment): MODEL (google/gemma-4-31B-it), LAYER (38), LAYERS (all, or layer numbers including
# LAYER), SITES ("label prelabel"), PRELABEL_CHECK ("caa curve-snap", or all), N (64, or all), SPLIT (extraction |
# verification | all; default extraction), BATCH (8), MAX_NEW (4096),
# RUN (smoke), THINKING (on | off; default on), DECODING (auto | greedy | sample; default auto, as baseline.py),
# SAMPLING (extra sampling options for a model without a generation config, e.g.
# "--temperature 0.6 --top-p 0.95 --top-k 20"; default none), DRAWS_MIN_ROWS (100; the covariance draws refuse fewer
# fitted prompts, so set it lower for a small smoke run and treat those draws as plumbing only).
# Outputs: prompt_bank/runs/<model>__<run>, steering/activations/<model>__<run>,
# steering/activations/<model>[-nothink]__t0_s0__horizon, steering/draws/<model>, steering/fits/<model>__<run>,
# steering/methods/<model>. Every line is flushed; run under nohup or tmux.
set -euo pipefail

MODEL=${MODEL:-google/gemma-4-31B-it}
LAYER=${LAYER:-38}
LAYERS=${LAYERS:-all}                 # what extract.py keeps: `all`, or layer numbers that include $LAYER
SITES=${SITES:-label prelabel}        # the sites the draws, fits, bundles and method checks are made at
PRELABEL_CHECK=${PRELABEL_CHECK:-caa curve-snap}   # prelabel-site bundles checked on the model (or: all)
case " $LAYERS " in *" all "*|*" $LAYER "*) ;; *) echo "LAYERS must be all or include LAYER=$LAYER"; exit 2;; esac
for _s in $SITES; do case $_s in label|prelabel) ;; *) echo "SITES: label and/or prelabel"; exit 2;; esac; done
N=${N:-64}
SPLIT=${SPLIT:-extraction}
case $SPLIT in extraction|verification|all) ;; *) echo "SPLIT must be extraction, verification or all"; exit 2;; esac
case $N in all) LIMIT=();; ''|*[!0-9]*) echo "N must be a number or all"; exit 2;; *) LIMIT=(--limit "$N");; esac
BATCH=${BATCH:-8}     # 16 runs out of memory on an 80 GB GPU with Gemma 4 31B at 4096 new tokens
MAX_NEW=${MAX_NEW:-4096}
RUN=${RUN:-smoke}
THINKING=${THINKING:-on}
DECODING=${DECODING:-auto}
SAMPLING=${SAMPLING:-}
DRAWS_MIN_ROWS=${DRAWS_MIN_ROWS:-100}
case $THINKING in on) NOTHINK=(); NOTHINK_HOOK=(--thinking); SUFFIX="";;
                  off) NOTHINK=(--no-thinking); NOTHINK_HOOK=(--no-thinking); SUFFIX="-nothink";;
                  *) echo "THINKING must be on or off"; exit 2;; esac
# shellcheck disable=SC2206
HOW=(--decoding "$DECODING" $SAMPLING)       # how replies are generated: the same for baseline.py and check_hook.py

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SHORT=${MODEL##*/}
NAME=${SHORT}__${RUN}
BANK=$ROOT/prompt_bank/data/t0_s0
RUNDIR=$ROOT/prompt_bank/runs/$NAME
ACTS=$ROOT/steering/activations/$NAME
FITS=$ROOT/steering/fits/$NAME
HORIZON=$ROOT/steering/activations/${SHORT}${SUFFIX}__t0_s0__horizon
DRAWS=$ROOT/steering/draws/$SHORT       # one subdirectory per layer and site: L<layer>_<site>
export PYTHONUNBUFFERED=1
# A container's CPU quota can be far below the machine's core count (15.36 of 128 on the A100 host of 2026-11-22).
# torch starts one thread per core it sees, and the tiny-model tests then take many minutes instead of seconds.
if [ -z "${OMP_NUM_THREADS:-}" ] && [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r _quota _period < /sys/fs/cgroup/cpu.max
    if [ "$_quota" != max ] && [ "$((_quota / _period))" -ge 1 ] && [ "$((_quota / _period))" -lt "$(nproc)" ]; then
        export OMP_NUM_THREADS=$((_quota / _period)) MKL_NUM_THREADS=$((_quota / _period))
    fi
fi
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

stage_version() {
    say "version: the code commit outputs will record"
    local v
    v=$(st -c "from steer.provenance import code_commit; print(code_commit())")
    echo "code commit: $v"
    case $v in unknown*) echo "no git and no CODE_STAMP.json: run 'python -m pbank.provenance stamp' in the git" \
                              "checkout and copy the code again"; exit 1;;
               *-modified) echo "code differs from the stamped commit: outputs will say so";; esac
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
    say "tests (OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}, $(nproc) cores visible)"
    (cd "$ROOT/prompt_bank" && .venv/bin/python -m pytest -q 2>&1 | tail -3)
    (cd "$ROOT/steering" && .venv/bin/python -m pytest -q 2>&1 | tail -3)
}

stage_hook() {
    say "hook: check_hook at layer $LAYER"
    st tools/check_hook.py --model "$MODEL" --layer "$LAYER" --bank "$BANK" --n 8 --max-new-tokens 64 \
        "${NOTHINK_HOOK[@]}" "${HOW[@]}"
}

stage_horizon() {
    say "horizon: activations at the horizon's tokens, all layers, all prompts (thinking $THINKING)"
    st extract_horizon.py --bank "$BANK" --model "$MODEL" "${NOTHINK[@]}" --out "$HORIZON"
}

stage_baseline() {
    say "baseline: N=$N of split $SPLIT, batch $BATCH, max_new_tokens $MAX_NEW, thinking $THINKING, ${HOW[*]}"
    pb baseline.py --bank data/t0_s0 --model "$MODEL" --split "$SPLIT" "${LIMIT[@]}" --batch-size "$BATCH" \
        --max-new-tokens "$MAX_NEW" --out "$RUNDIR" "${NOTHINK[@]}" "${HOW[@]}"
}

stage_extract() {
    say "extract: layers $LAYERS, every site, top prelabel logits"
    # shellcheck disable=SC2086
    st extract.py --bank "$BANK" --run "$RUNDIR" --layers $LAYERS --split "$SPLIT" --out "$ACTS"
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
    ok = (r.get("replay_consistent", r["replay_predicts_label"]) and at == r["label_token"]
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

stage_draws() {
    local site
    for site in $SITES; do
        say "draws: shared random directions, layer $LAYER, $site site (at least $DRAWS_MIN_ROWS fitted prompts)"
        st ../evaluation/shared_draws.py make --activations "$ACTS" --bank "$BANK/prompts.jsonl" --model "$MODEL" \
            --layer "$LAYER" --site "$site" --min-rows "$DRAWS_MIN_ROWS" --out "$ROOT/steering/draws"
        st ../evaluation/shared_draws.py verify --draws "$DRAWS/L${LAYER}_$site" --activations "$ACTS" \
            --bank "$BANK/prompts.jsonl"
    done
}

stage_fit() {
    local site draws
    for site in $SITES; do
        draws=()
        [ -d "$DRAWS/L${LAYER}_$site" ] && draws=(--draws "$DRAWS/L${LAYER}_$site")
        say "fit: layer $LAYER, $site site ${draws[*]:+with the shared draws}"
        st fit.py --activations "$ACTS" --bank "$BANK" --layer "$LAYER" --site "$site" "${draws[@]}"
    done
}

stage_bundle() {
    local site
    for site in $SITES; do
        say "bundle: $site site"
        st bundle.py --fit "$FITS/default/L${LAYER}_$site" --activations "$ACTS"
    done
}

stage_method() {
    # A label-site bundle is checked as it declares itself (every token), on short replies. A prelabel-site bundle is
    # checked with the edit at the prelabel token only, the one token whose edit can change the choice. That needs
    # whole replies (the label must be reached), which with thinking on is minutes per bundle, so only the methods in
    # $PRELABEL_CHECK are checked there: one linear method and one curve method cover both kinds of edit.
    # Every bundle is run even after a failure; the stage fails at the end if any did.
    local site d m found failed=() args
    for site in $SITES; do
        found=0
        for d in "$ROOT/steering/methods/$SHORT"/*__"${NAME}"__default__L"${LAYER}"_"$site"; do
            [ -d "$d" ] || continue
            found=$((found + 1))
            m=$(basename "$d" | sed 's/__.*//')
            args=(--max-new-tokens 64)
            if [ "$site" = prelabel ]; then
                case " $PRELABEL_CHECK " in *" all "*|*" $m "*) ;; *) echo "not checked on the model: $m ($site site)"; continue;; esac
                args=(--max-new-tokens "$MAX_NEW" --positions prelabel)
            fi
            say "method: check_hook --method $m ($site site) ${args[*]}"
            st tools/check_hook.py --method "$d" --bank "$BANK" --n 4 "${HOW[@]}" "${args[@]}" || failed+=("$m@$site")
        done
        [ "$found" -ge 3 ] || failed+=("only $found $site-site bundles of this run found")
    done
    [ ${#failed[@]} -eq 0 ] || { echo "method checks FAILED: ${failed[*]}"; exit 1; }
    echo "method checks: all passed"
}

stage_sums() {
    say "sums"
    # Only the output directories that exist: a run need not include every stage (e.g. no `horizon`).
    local d dirs=()
    for d in "prompt_bank/runs/$NAME" "steering/activations/$NAME" "steering/fits/$NAME" "steering/methods/$SHORT" \
             "steering/activations/$(basename "$HORIZON")" "steering/draws/$SHORT"; do
        if [ -d "$ROOT/$d" ]; then dirs+=("$d"); else echo "not present, so not listed: $d"; fi
    done
    [ ${#dirs[@]} -gt 0 ] || { echo "no outputs to list"; exit 1; }
    (cd "$ROOT" && find "${dirs[@]}" -type f | sort | xargs sha256sum > "$ROOT/vast_smoke.SHA256SUMS")
    wc -l < "$ROOT/vast_smoke.SHA256SUMS"
}

[ $# -gt 0 ] || { sed -n 2,40p "$0"; exit 2; }
for s in "$@"; do
    declare -F "stage_$s" >/dev/null || { echo "unknown stage: $s"; exit 2; }
    "stage_$s"
done
say "done: $*"
