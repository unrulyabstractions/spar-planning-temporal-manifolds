"""Model-backed local checks for the full bank, to run with the xpu env:

    ~/miniconda3/envs/intel_ai/bin/python local_checks.py            # everything
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py tests      # only the 0.8B generation tests
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py smoke      # only the two E0 smokes
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py steer      # E8 prepare + all five tests (after smoke)
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py decision   # E8 horizon/decision split, thinking on (after smoke)
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py units      # E9 unit rewrites, one context
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py reason     # E10 patch before reasoning, 4 pairs
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py prefix     # E8 prefix-cache scoring vs full passes
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py unitbank   # E0 on the units bank, both modes, 8 prompts
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py fillbank   # E0 on the fill bank, both modes, ~8 prompts
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py pooledbank # E0 for the E13 steering smokes, full + units
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py stagea     # E13 stage A: pooled prepare + E8 (after pooledbank)
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py stageb     # E13 stage B: prepare + suffix steer, 2 prompts (after pooledbank)
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py phrase     # E14: phrase states, fit, every steering unit, analysis

Runs, with whichever python launched this file:
  1. tests/test_generation.py (batched-vs-single parity, force-close, layouts, cache; starter bank)
  2. E0 on the 0.8B, full bank, thinking off, 6 prompts
  3. E0 on the 0.8B, full bank, thinking on, 6 prompts, 32-token think budget
     (its generations are cached in out/gen_cache for step 4)
  4. E8 on the 0.8B smoke outputs: prepare inputs at layer 16 (the 0.8B has 24
     layers, so the 8B's answer-slot layers 24-30 do not exist), then all five
     tests on 4 prompts (plumbing only: 6 prompts give no meaningful numbers)
  5. E8's horizon/decision split (07 section 10) at the same cell, thinking on only;
     all smoke think blocks are forced at 32 tokens, so this one is fitted on all prompts
  6. E9 (07 section 11): the unit renderings in one context, forward pass only
  7. E10 (07 section 12): every condition on 4 pairs with a 32-token think budget
     (plumbing only; the self-patch control should leave the labels unchanged)
  8. E8's shared-prefix scoring against one full pass per edit, on Qwen3-0.6B in
     bfloat16 (the 0.8B has linear-attention layers, where E8 falls back to full
     passes, so steps 4-5 do not exercise the cache)
  9. E0 on the 0.8B on the units bank (14_spec_unit_overlap.md), thinking off and
     on, 4 pairs; then every answers.json row must carry its pair labels (and, thinking
     off, a located label)
 10. the same on the fill bank (16_spec_pristine_manifold.md §4), about 8 prompts with
     one anchor, whose rows must also carry the realism labels and the anchor flag
 11. E0 on the 0.8B into out/pristine_smoke for the E13 steering smokes
     (16_spec_pristine_manifold.md §5): full bank 6 prompts and units bank 60 (fewer
     leaves a unit level collinear with the durations, and the fit refuses), both modes,
     32-token think budget, generations cached for step 12. No fill run: the anchor
     check reads the 8B's runs and needs 200 units-bank prompts, so the smoke pools the
     old banks only.
 12. stage A: steering_prepare.py --pooled at off suffix L12, off pre-label L16 and on
     pre-label L16 (the 8B's L22/L24 are past the 0.8B's last editable layer), then E8
     steer + post on 8 prompts: the effects curve, its invariant and in-span parts, U, random
 13. stage B: steering_prepare.py --pooled --stage-b at on suffix L12, then
     exp13_suffix_steer.py on 2 prompts, all 8 conditions, E0's 32-token budget
 14. E14 (21_spec_phrase_steering.md), the box queue end to end at small size: phrase
     states for the first 300 prompts of the 8B's prompts file (full bank only, so the
     fit leaves out the unit factor and says so), the fit with 2 prompts per transition,
     every unit of the driver with a 32-token think budget, then the analysis. Starts
     from an empty out/exp14_Qwen3.5-0.8B_phrase, because finished work is skipped.
and prints one line per step plus the answer-token line of each smoke.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3.5-0.8B"
CHECK_PAIRED = """
import json
bank, keys = "{bank}", {keys}
for mode in ("off", "on"):
    rows = json.load(open(f"out/exp0_Qwen3.5-0.8B_think-{{mode}}_{{bank}}/answers.json"))
    # a 32-token think budget can leave the 0.8B without a label, so labels are required thinking off only
    bad = [r["prompt"][-80:] for r in rows
           if not (all(k in r for k in keys) and r["pair_id"] and r["duration_id"] and (r.get("label") or mode == "on"))]
    print(f"{{bank}} bank {{mode}}: {{len(rows)}} rows, {{len({{r['pair_id'] for r in rows}})}} pairs, "
          f"{{len(bad)}} without pair labels or a located label")
    assert not bad, bad
"""
PAIR_KEYS = ["pair_id", "duration_id", "rendered_unit", "natural"]


def paired_bank_smoke(bank, keys):
    """E0 on a paired bank, both modes, about 8 prompts, then the answers.json label check."""
    return [
        [sys.executable, "experiments/exp0_baseline.py", "--model", MODEL, "--thinking", "off",
         "--bank", bank, "--limit", "8", "--no-cache", "--out", "out"],
        [sys.executable, "experiments/exp0_baseline.py", "--model", MODEL, "--thinking", "on",
         "--bank", bank, "--limit", "8", "--max-think-tokens", "32", "--skip-behavior", "--no-cache", "--out", "out"],
        [sys.executable, "-c", CHECK_PAIRED.format(bank=bank, keys=keys)],
    ]
SMOKE_ROOT = "out/pristine_smoke"
PHRASE_SMOKE = "out/exp14_Qwen3.5-0.8B_phrase"


def pooled_bank_smoke():
    """E0 on the full and units banks into SMOKE_ROOT, both modes, generations cached."""
    return [[sys.executable, "experiments/exp0_baseline.py", "--model", MODEL, "--thinking", mode, "--bank", bank,
             "--limit", limit, "--max-think-tokens", "32", "--skip-behavior", "--out", SMOKE_ROOT]
            for bank, limit in (("full", "6"), ("units", "60")) for mode in ("off", "on")]


STEPS = {
    "tests": [[sys.executable, "-m", "pytest", "-q", "tests/test_generation.py"]],
    "smoke": [
        [sys.executable, "experiments/exp0_baseline.py", "--model", MODEL, "--thinking", "off",
         "--bank", "full", "--limit", "6", "--no-cache"],
        [sys.executable, "experiments/exp0_baseline.py", "--model", MODEL, "--thinking", "on",
         "--bank", "full", "--limit", "6", "--max-think-tokens", "32", "--skip-behavior", "--out", "out"],
    ],
    "steer": [
        [sys.executable, "analysis/steering_prepare.py", "out", "--name", "smoke",
         "--cells", "off:pre-label:16,on:pre-label:16"],
        [sys.executable, "experiments/exp8_steering.py", "--inputs", "steering/inputs_Qwen3.5-0.8B_smoke.npz",
         "--limit", "4", "--batch-size", "4", "--long-batch-size", "2", "--out", "out"],
    ],
    "decision": [
        [sys.executable, "analysis/steering_prepare.py", "out", "--name", "decision_smoke",
         "--cells", "on:pre-label:16"],
        [sys.executable, "experiments/exp8_steering.py", "--inputs", "steering/inputs_Qwen3.5-0.8B_decision_smoke.npz",
         "--tests", "post", "--limit", "4", "--long-batch-size", "2", "--out", "out"],
    ],
    "units": [
        [sys.executable, "experiments/exp9_units.py", "--model", MODEL, "--contexts", "1", "--batch-size", "8",
         "--out", "out"],
    ],
    "reason": [
        [sys.executable, "experiments/exp10_patch_reasoning.py", "--model", MODEL, "--pairs", "4", "--self-pairs", "2",
         "--max-think-tokens", "32", "--batch-size", "4", "--score-batch-size", "4", "--out", "out"],
    ],
    "prefix": [[sys.executable, "experiments/check_prefix_cache.py"]],
    "unitbank": paired_bank_smoke("units", PAIR_KEYS),
    "fillbank": paired_bank_smoke("fill", PAIR_KEYS + ["delay_scale", "alan_range", "anchor"]),
    "pooledbank": pooled_bank_smoke(),
    "stagea": [
        [sys.executable, "analysis/steering_prepare.py", SMOKE_ROOT, "--pooled", "--model", MODEL,
         "--name", "pristine_smoke", "--cells", "off:suffix newline:12,off:pre-label:16,on:pre-label:16"],
        [sys.executable, "experiments/exp8_steering.py", "--inputs", "steering/inputs_Qwen3.5-0.8B_pristine_smoke.npz",
         "--tests", "steer,post", "--limit", "8", "--batch-size", "4", "--long-batch-size", "2", "--out", "out"],
    ],
    "stageb": [
        [sys.executable, "analysis/steering_prepare.py", SMOKE_ROOT, "--pooled", "--stage-b", "--model", MODEL,
         "--name", "pristine_suffix_smoke", "--cells", "on:suffix newline:12"],
        [sys.executable, "experiments/exp13_suffix_steer.py", "--inputs",
         "steering/inputs_Qwen3.5-0.8B_pristine_suffix_smoke.npz", "--limit", "2", "--batch-size", "2",
         "--score-batch-size", "2", "--out", "out"],
    ],
    "phrase": [
        [sys.executable, "-c", f"import shutil; shutil.rmtree({PHRASE_SMOKE!r}, ignore_errors=True)"],
        [sys.executable, "experiments/exp14_phrase_states.py", "--prompts", "steering/phrase_prompts_Qwen3-8B.json",
         "--model", MODEL, "--limit", "300", "--batch-size", "8", "--out", "out"],
        [sys.executable, "analysis/steering_prepare.py", "--phrase", PHRASE_SMOKE, "--per-transition", "2",
         "--allow-short"],
        [sys.executable, "experiments/exp14_phrase_steer.py", "--dir", PHRASE_SMOKE, "--limit", "2",
         "--batch-size", "2", "--score-batch-size", "2", "--max-think-tokens", "32"],
        [sys.executable, "analysis/exp14_analysis.py", PHRASE_SMOKE],
    ],
}


def run(cmd):
    t0 = time.time()
    print(f"\n$ {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=HERE, text=True, capture_output=True)
    out = p.stdout + p.stderr
    tail = [l for l in out.splitlines() if l.lstrip().startswith(("suffix tokens", "WARNING", "PEAK", "probe: best",
                                                                     "cached reasoning", "T1:", "T5", "necessary",
                                                                     "sufficient_line", "curve_beats_line", "horizon_beyond_decision",
                                                                     "ablate_beyond_decision", "prompts:", "wrote", "transfer",
                                                                     "subspace read", "plumbing", "4 pairs", "prefix cache",
                                                                     "units bank", "fill bank", "split ", "stage A",
                                                                     "stage B", "pooled runs", "toward",
                                                                     "fit L", "eligible", "  both-near", "  none-near",
                                                                     "  near-", "H1", "H2", "H3"))
            or " passed" in l or " failed" in l or "Error" in l]
    print("\n".join(tail[-14:]))
    status = "ok" if p.returncode == 0 else f"FAILED (exit {p.returncode})"
    print(f">> {status} in {time.time() - t0:.0f} s")
    if p.returncode != 0:
        print(out[-3000:])
    return p.returncode == 0


if __name__ == "__main__":
    which = sys.argv[1:] or list(STEPS)
    results = {name: all(run(c) for c in STEPS[name]) for name in which}
    print("\nsummary:", {k: ("ok" if v else "FAILED") for k, v in results.items()})
    sys.exit(0 if all(results.values()) else 1)
