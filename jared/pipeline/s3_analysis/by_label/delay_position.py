"""Which delay pulls the horizon code: the near one, or the one listed nearer the horizon?
(overnight, 2026-10-29; exploratory, post hoc)

confound_robustness.py found, on regime-none prompts, that a longer near delay reads the
horizon shorter, and an option-order effect whose sign in 14B depends on the regime. In
the template the options are listed before the CONSTRAINT line that holds the horizon,
so the second-listed delay is the one textually nearer the horizon. Two
reparametrizations of the same three numbers, each with option order:
  role       log near delay, log far delay, near listed first
  position   log first-listed delay, log second-listed delay, near listed first
They are not linear rewrites of each other (first = near or far depending on the order),
so the within-duration R² says which describes the readout better. A code pulled by
textual proximity fits "position" and loses its order effect there; a code comparing the
horizon with the near delay fits "role".

Regime-none prompts at durations with enough of them, everything fitted on them
(confound_robustness.none_only); readouts: the top-4 probe and the horizon axis.

    python -m pipeline.s3_analysis.by_label.delay_position /home/Jared/SPAR/results/horizon
    -> <root>/analysis/delay_position.{json,txt}
"""

import json
import sys
from pathlib import Path

import numpy as np

from pipeline.s3_analysis.by_label.confound_robustness import none_only
from pipeline.s3_analysis.by_label.confound_sweep import (base_folds, calibration, cell_matrix, crossfit_readout, dataset,
                            fe_regression, horizon_axis, within)

CELLS = [("Qwen3-8B", ("full", "units"), "suffix newline", 16), ("Qwen3-8B", ("full", "units"), "suffix newline", 24),
         ("Qwen3-14B", ("full",), "suffix newline", 20), ("Qwen3-14B", ("full",), "suffix newline", 27)]


def parametrizations(near, far, near_first):
    """name -> (feature matrix, feature names) for the role and position descriptions."""
    first, second = np.where(near_first, near, far), np.where(near_first, far, near)
    order = near_first.astype(float)
    return {"role": (np.column_stack([near, far, order]), ["near", "far", "order"]),
            "position": (np.column_stack([first, second, order]), ["first", "second", "order"])}


def within_r2(r, V, groups):
    """R² of the within-group regression of r on the columns of V."""
    r_w, V_w = within(r, groups), within(V, groups)
    residual = r_w - V_w @ np.linalg.lstsq(V_w, r_w, rcond=None)[0]
    return float(1 - residual.var() / r_w.var())


def main(root):
    root = Path(root)
    rng = np.random.default_rng(0)
    results, lines = {}, []
    for model, banks, position, layer in CELLS:
        parts, data = dataset(root, model, "on", banks)
        X, d = none_only(cell_matrix(parts, position, layer), data)
        folds = base_folds(d["pairs"])
        axis = horizon_axis(X, d["y"], d["duration_id"])
        readouts = {"probe k=4": crossfit_readout(X, d["y"], d["duration_id"], folds, 4),
                    "axis": (X - X.mean(0)) @ axis / (axis @ axis)}
        near_first = d["features"][:, 4] > 0.5
        for readout_name, readout in readouts.items():
            readout = readout / calibration(readout, d["y"], d["duration_id"])
            name = f"{model} on {position} L{layer} none-only {readout_name}"
            results[name] = {}
            parts_text = []
            for label, (V, names) in parametrizations(d["features"][:, 0], d["features"][:, 1], near_first).items():
                fit = fe_regression(readout, V, names, d["durations"], d["pairs"], rng)
                r2 = within_r2(readout, V, d["durations"])
                results[name][label] = {"r2": r2, "effects": fit}
                parts_text.append(f"{label}: " + " ".join(f"{n} {v[0]:+.2f}{'*' if v[1] > 0 or v[2] < 0 else ''}"
                                                          for n, v in fit.items()) + f" (R² {r2:.3f})")
            lines.append(f"{name}: " + " | ".join(parts_text))
            print(lines[-1], flush=True)
    out = root / "analysis"
    (out / "delay_position.json").write_text(json.dumps(results, indent=1))
    (out / "delay_position.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
