"""Model-backed local checks for the full bank, to run with the xpu env:

    ~/miniconda3/envs/intel_ai/bin/python local_checks.py            # everything
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py tests      # only the 0.8B generation tests
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py smoke      # only the two E0 smokes
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py steer      # E8 prepare + all five tests (after smoke)
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py decision   # E8 horizon/decision split, thinking on (after smoke)
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py units      # E9 unit rewrites, one context
    ~/miniconda3/envs/intel_ai/bin/python local_checks.py reason     # E10 patch before reasoning, 4 pairs

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
and prints one line per step plus the answer-token line of each smoke.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3.5-0.8B"
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
                                                                     "subspace read", "plumbing", "4 pairs"))
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
