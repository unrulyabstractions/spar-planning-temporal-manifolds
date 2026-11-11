"""How much does each confound depend on the readout? (overnight, 2026-10-29)

confound_sweep.py reads the horizon with one probe (top-8 duration-average PCs) and
found effects of option order and regime that the projection on the horizon axis did
not show. Here the same within-duration regression is run on several readouts of the
same cells:
  probe k   the cross-fitted ridge on the top-k PCs, k = 1, 2, 4, 8, 16
  axis      the projection on the horizon axis (probe-free)
An effect that holds across readouts is in the code; one that appears only for some k
is a direction those probes happen to weigh.

Every readout above, the axis included, is fitted to duration averages over all
prompts, and on the old banks the regime mix changes with the duration, so each can
read some of the regime as horizon. The second pass ("none only") repeats the table on
the regime-none prompts at durations with at least MIN_NONE of them (about 30 s to
1 year), with everything fitted on those prompts: no regime to absorb, and the delay
effects measured where no delay falls inside the horizon.

    python -m pipeline.s3_analysis.by_label.confound_robustness /home/Jared/SPAR/results/horizon
    -> <root>/analysis/confound_robustness.{json,txt}
"""

import json
import sys
from pathlib import Path

import numpy as np

from pipeline.s3_analysis.by_label.confound_sweep import (FEATURES, base_folds, calibration, cell_matrix, crossfit_readout, dataset,
                            fe_regression, horizon_axis)
from pipeline.s3_analysis.by_label.delay_margin import MIN_NONE

KS = (1, 2, 4, 8, 16)
SHOWN = ("near first", "regime near", "regime both", "log unit size", "log near delay", "log reward")
CELLS = [("Qwen3-8B", ("full", "units"), "on", "suffix newline", 16),
         ("Qwen3-8B", ("full", "units"), "on", "suffix newline", 24),
         ("Qwen3-8B", ("full", "units"), "on", "pre-label", 22),
         ("Qwen3-14B", ("full",), "on", "suffix newline", 20),
         ("Qwen3-14B", ("full",), "on", "suffix newline", 27)]


def readouts(X, data):
    """name -> (readout, calibration) for each probe size and the axis projection."""
    y, duration = data["y"], data["duration_id"]
    folds = base_folds(data["pairs"])
    out = {}
    for k in KS:
        readout = crossfit_readout(X, y, duration, folds, k)
        out[f"probe k={k}"] = (readout, calibration(readout, y, duration))
    axis = horizon_axis(X, y, duration)
    out["axis"] = ((X - X.mean(0)) @ axis / (axis @ axis), 1.0)
    return out


def none_only(X, data):
    """The regime-none rows at durations with at least MIN_NONE of them."""
    is_none = data["regime"] == "none"
    counts = {d: (is_none & (data["duration_id"] == d)).sum() for d in np.unique(data["duration_id"])}
    keep = is_none & np.array([counts[d] >= MIN_NONE for d in data["duration_id"]])
    return X[keep], {k: v[keep] for k, v in data.items()}


def main(root):
    root = Path(root)
    rng = np.random.default_rng(0)
    results, loaded, lines = {}, {}, []
    for model, banks, mode, position, layer in CELLS:
        key = (model, banks, mode)
        if key not in loaded:
            loaded.clear()
            loaded[key] = dataset(root, model, mode, banks)
        parts, data = loaded[key]
        X_all = cell_matrix(parts, position, layer)
        for subset in ("all", "none only"):
            X, data = (X_all, loaded[key][1]) if subset == "all" else none_only(X_all, loaded[key][1])
            name = f"{model} {'+'.join(banks)} {mode} {position} L{layer} {subset} ({len(X)} prompts)"
            results[name] = {}
            lines.append(f"{name} (equivalent decades; * = 95% CI excludes 0)")
            lines.append(f"  {'readout':12s} {'R²':>5s} " + " ".join(f"{f:>14s}" for f in SHOWN))
            for readout_name, (readout, slope) in readouts(X, data).items():
                fit = fe_regression(readout, data["features"], list(FEATURES), data["durations"], data["pairs"], rng)
                results[name][readout_name] = {"calibration": slope, "effects": fit}
                r2 = 1 - ((readout / slope - data["y"]) ** 2).sum() / ((data["y"] - data["y"].mean()) ** 2).sum() \
                    if readout_name != "axis" else float("nan")
                cells = []
                for f in SHOWN:
                    if f in fit:
                        c, lo, hi = (v / slope for v in fit[f])
                        cells.append(f"{c:+.2f}{'*' if lo > 0 or hi < 0 else ' '}".rjust(14))
                    else:
                        cells.append("–".rjust(14))
                lines.append(f"  {readout_name:12s} {r2:5.2f} " + " ".join(cells))
            print("\n".join(lines[-len(KS) - 3:]), flush=True)
    out = root / "analysis"
    (out / "confound_robustness.json").write_text(json.dumps(results, indent=1))
    (out / "confound_robustness.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
