"""Plumbing smoke for E21 (pipeline/s1_data_vectors/experiments/exp21_boundary_off.py), to run with the xpu env:

    ~/miniconda3/envs/intel_ai/bin/python local_smoke_exp21.py

Runs the thinking-off runner on the 0.8B with --limit 2 on four types (t0_w0: one string prompt;
mt_first: multi-turn; persona_model: persona in a system message; no_horizon: no cue), into
out/exp21_smoke, then checks the saved files: shapes, finite values (NaN only in the phrase_end slot
of rows without a cue), token names resolved by the boundary analysis, the prelabel token is ':',
phrase_end decodes to the last token of the cue, and persona rows carry a stated horizon.
Plumbing only: a 0.8B gives no meaningful numbers.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline.s3_analysis.eval_types_boundary import load_run, position_index  # noqa: E402

MODEL = "Qwen/Qwen3.5-0.8B"
TYPES = ("t0_w0", "mt_first", "persona_model", "no_horizon")
HAS_CUE = {"t0_w0": False, "mt_first": True, "persona_model": True, "no_horizon": False}
OUT = HERE / "out" / "exp21_smoke"


def check(run_dir):
    problems = []
    types, tokens, mode = load_run(run_dir)
    if mode != "off":
        problems.append(f"settings.json thinking is {mode!r}, expected 'off'")
    positions = {name: position_index(tokens, name) for name in ("suffix_end", "prelabel", "phrase_end")}
    if positions != {"suffix_end": 4, "prelabel": 5, "phrase_end": 6}:
        problems.append(f"positions resolve to {positions}, expected 4, 5, 6 (tokens {list(tokens)})")
    for type_id in TYPES:
        rows = types.get(type_id, [])
        if len(rows) != 2:
            problems.append(f"{type_id}: {len(rows)} rows, expected 2")
            continue
        z = np.load(run_dir / type_id / "acts_0.npz")
        for layer in z["layers"]:
            acts = z[f"L{layer}"]
            if acts.shape[:2] != (2, 7) or acts.dtype != np.float16:
                problems.append(f"{type_id} L{layer}: shape {acts.shape} dtype {acts.dtype}")
            if not np.isfinite(acts[:, :6]).all():
                problems.append(f"{type_id} L{layer}: non-finite values before phrase_end")
            if np.isfinite(acts[:, 6]).all() != HAS_CUE[type_id]:
                problems.append(f"{type_id} L{layer}: phrase_end finite is {not HAS_CUE[type_id]}, expected {HAS_CUE[type_id]}")
        if not np.isfinite(z["label_logits"]).all() or z["label_logits"].shape != (2, 2):
            problems.append(f"{type_id}: label_logits {z['label_logits']}")
        for row in rows:
            if HAS_CUE[type_id]:
                token = row["phrase_end_token"]
                if token is None or not row["cue_text"].rstrip().endswith(token.strip()):
                    problems.append(f"{row['sample_uid']}: phrase_end token {token!r} is not the end of {row['cue_text']!r}")
            elif row["phrase_end_token"] is not None:
                problems.append(f"{row['sample_uid']}: phrase_end token {row['phrase_end_token']!r} without a cue")
    if list(tokens[:5]) != ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"] or tokens[5] != ":":
        problems.append(f"token names {list(tokens)}")
    persona = types.get("persona_model", [])
    for row in persona:
        print(f"{row['sample_uid']}: stated {row.get('stated_horizon_text')!r} -> {row.get('stated_horizon_years')}")
    if not any(r.get("stated_horizon_years") for r in persona):
        problems.append("persona_model: no stated horizon parsed in either row")
    if any("stated_horizon_text" in r for r in types.get("t0_w0", [])):
        problems.append("t0_w0 rows have a stated horizon; only no-horizon types should")
    return problems


def main():
    shutil.rmtree(OUT, ignore_errors=True)   # finished types are skipped on a rerun
    command = [sys.executable, "-m", "pipeline.s1_data_vectors.experiments.exp21_boundary_off",
               "--model", MODEL, "--types", ",".join(TYPES), "--limit", "2", "--batch-size", "2", "--out", str(OUT)]
    print("$", " ".join(command), flush=True)
    run = subprocess.run(command, cwd=HERE)
    if run.returncode != 0:
        print(f"FAIL: exp21 exited with {run.returncode}")
        sys.exit(1)
    settings = json.loads((OUT / "settings.json").read_text())
    print("layers:", settings["layers"])
    problems = check(OUT)
    print("PASS" if not problems else "FAIL:\n  " + "\n  ".join(problems))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
