"""Plumbing smoke for the related-qualities chain (EvalsRelatedQualities.md, queue_related.sh), to run
with the xpu env from jared/:

    ~/miniconda3/envs/intel_ai/bin/python local_smoke_related.py

Runs every step of the box queue on Qwen/Qwen3.5-0.8B with tiny limits into out/smoke_related (wiped
first): E0 on 2 items per set (256 think tokens), the eval-probe states and fit, a PLUMBING cell (fake
horizons on the smoke's own activations), the prepare step, E23 with 2 random directions, the report.
Prints PASS/FAIL per step and stops at the first failure. Plumbing only: a 0.8B, 256 think tokens and a
fake horizon curve give no meaningful numbers. The 0.8B has linear-attention layers, so E23 exercises
score_last_edits' full-pass fallback rather than the prefix cache the 8B uses.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3.5-0.8B"
SHORT = MODEL.split("/")[-1]
OUT = HERE / "out" / "smoke_related"
LAYERS = "8,12,16,20"          # the 0.8B has 24 layers
CELL_LAYER = "12"
PROBE_LAYERS = "16,20"         # after the edited layer, as on the 8B (edit 24, probes 28 and 32)
INPUTS = OUT / "related_inputs" / f"inputs_{SHORT}_related.npz"
PROBE = OUT / "eval_awareness" / f"eval_probe_{SHORT}.npz"

STEPS = [
    ("E0 related", ["pipeline.s1_data_vectors.experiments.exp0_evaltypes", "--suite", "related", "--model", MODEL,
                    "--types", "all", "--limit", "4", "--max-think-tokens", "256", "--batch-size", "4",
                    "--layers", LAYERS, "--out", str(OUT)]),
    ("E24 probe states", ["pipeline.s1_data_vectors.experiments.exp24_eval_probe_states", "--model", MODEL,
                          "--layers", LAYERS, "--limit", "20", "--batch-size", "4", "--out", str(OUT)]),
    ("probe fit", ["pipeline.s1_data_vectors.analysis.eval_probe_fit", str(OUT / f"eval_probe_states_{SHORT}"),
                   "--out", str(OUT / "eval_awareness"), "--score-runs", str(OUT)]),
    ("plumbing cell", ["pipeline.s1_data_vectors.analysis.related_cells", str(OUT), "--plumbing", "--model", MODEL,
                       "--layer", CELL_LAYER, "--out", str(OUT)]),
    ("prepare", ["pipeline.s1_data_vectors.analysis.related_steering_prepare", str(OUT),
                 "--cell", str(OUT / f"cells_{SHORT}_plumbing_L{CELL_LAYER}.npz"),
                 "--out", str(OUT / "related_inputs"), "--model", MODEL]),
    ("E23 steer", ["pipeline.s2_steering.experiments.ablate_steer.exp23_related_steer", "--inputs", str(INPUTS),
                   "--eval-probe", str(PROBE), "--probe-layers", PROBE_LAYERS, "--layer", CELL_LAYER, "--k", "2",
                   "--batch-size", "2", "--out", str(OUT)]),
    ("report", ["pipeline.s2_steering.analysis.related_report", str(OUT / f"exp23_{SHORT}_related" / "results.json"),
                "--inputs", str(INPUTS), "--eval-probe", str(PROBE)]),
]


def check_outputs():
    """Problems with what the chain wrote, beyond the steps' exit codes."""
    problems = []
    report = json.loads((OUT / f"exp23_{SHORT}_related" / "related_report.json").read_text())
    if not report["sets"]:
        problems.append("the report has no sets")
    results = json.loads((OUT / f"exp23_{SHORT}_related" / "results.json").read_text())
    for set_id, result in results["sets"].items():
        for row in result["rows"]:
            if "probe" not in row:
                problems.append(f"{set_id} {row['sample_uid']}: no probe readouts")
            if len(row) - len(("sample_uid", "set_id", "variant", "twin_id", "split", "log_h", "probe")) < 5:
                problems.append(f"{set_id} {row['sample_uid']}: too few conditions")
    unreadable = json.loads(INPUTS.with_suffix(".json").read_text())["unreadable"]
    print(f"rows unreadable at the prepare step: {len(unreadable)} "
          f"({', '.join(sorted({u['reason'] for u in unreadable})) or 'none'})")
    return problems


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    for name, args in STEPS:
        start = time.time()
        done = subprocess.run([sys.executable, "-m", *args], cwd=HERE)
        print(f"{'PASS' if done.returncode == 0 else 'FAIL'} {name} ({time.time() - start:.0f} s)", flush=True)
        if done.returncode != 0:
            sys.exit(f"smoke stopped at {name}; outputs so far in {OUT}")
    problems = check_outputs()
    print("\n".join(f"FAIL {p}" for p in problems) or f"PASS all steps; outputs in {OUT}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
