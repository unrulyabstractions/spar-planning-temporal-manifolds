"""Plumbing smoke for E20 on eval types (exp20_evaltype_transfer.py), to run with the xpu env:

    ~/miniconda3/envs/intel_ai/bin/python local_smoke_exp20.py [EVALTYPES_DIR]

Runs the donor and noop arms on 2 city_values rows of the eval-type run (default
~/SPAR/results/horizon/evaltypes_Qwen3-8B_think-on) with the 0.8B and a 256-token think budget, into
out/exp20_smoke, then checks that both arms saved their rows with finite near log-odds and that the
type's summary.json exists. Plumbing only: a 0.8B and 256 think tokens give no meaningful numbers.
"""

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3.5-0.8B"
TYPE_ID = "city_values"
ARMS = ("donor", "noop")
OUT = HERE / "out" / "exp20_smoke"


def check(type_dir):
    problems = []
    for arm in ARMS:
        file = type_dir / "conditions" / f"{arm}.json"
        if not file.exists() or not (file.with_suffix(".npz")).exists():
            problems.append(f"{arm}: no saved conditions/{arm}.json and .npz")
            continue
        rows = json.loads(file.read_text())["rows"]
        if len(rows) != 2:
            problems.append(f"{arm}: {len(rows)} rows, expected 2")
        for row in rows:
            for key in ("near_lo", "log_h", "target_log_h", "think_len"):
                if not math.isfinite(row[key]):
                    problems.append(f"{arm} {row['sample_uid']}: {key} = {row[key]}")
            if not row["think_text"]:
                problems.append(f"{arm} {row['sample_uid']}: empty think_text")
    if not (type_dir / "summary.json").exists():
        problems.append("no summary.json")
    return problems


def main():
    evaltypes_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "~/SPAR/results/horizon/evaltypes_Qwen3-8B_think-on").expanduser()
    shutil.rmtree(OUT, ignore_errors=True)   # finished units are skipped on a rerun
    command = [sys.executable, "-m", "pipeline.s2_steering.experiments.manifold_steer.exp20_evaltype_transfer",
               "--types", TYPE_ID, "--arms", ",".join(ARMS), "--limit", "2", "--model", MODEL,
               "--max-think-tokens", "256", "--batch-size", "2", "--score-batch-size", "2",
               "--evaltypes-dir", str(evaltypes_dir), "--out", str(OUT)]
    print("$", " ".join(command), flush=True)
    run = subprocess.run(command, cwd=HERE)
    if run.returncode != 0:
        print(f"FAIL: exp20 exited with {run.returncode}")
        sys.exit(1)
    problems = check(OUT / f"exp20_{MODEL.split('/')[-1]}_evaltype_transfer" / TYPE_ID)
    print("PASS" if not problems else "FAIL:\n  " + "\n  ".join(problems))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
