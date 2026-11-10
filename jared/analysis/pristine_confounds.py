"""Does the pristine fit remove what the confound sweep found? (overnight, 2026-10-29)

The confound sweep measured, within a duration, how far each prompt feature moves a
horizon readout. Here the readout is the manifold itself, cross-fitted by base on the
pooled runs (8B full + units, and the fill run when it exists, without its anchors), as
pristine_manifold.py does. A cell whose fill anchors do not match the old runs
(`gated_jobs`) is fitted on the old runs only:
  old curve    per-duration averages, nearest-point readout (E8's curve)
  effects ⟂    duration effects with unit, regime, forced and order as factors and the
               log near and far delays as covariates, their span projected out (16 §4)
and the same within-duration regression on all features, in equivalent decades. A
feature the fit includes should read about 0; one it does not include (rewards, think
length) shows what is left.

    python analysis/pristine_confounds.py /home/Jared/SPAR/results/horizon
    -> <root>/analysis/pristine_confounds{,_fill}.{json,txt}
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from confound_sweep import FEATURES, base_folds, calibration, cell_matrix, dataset, fe_regression  # noqa: E402
from pristine_manifold import POOLED_BANKS, anchor_check, cross_fit, gated_jobs, load_parts  # noqa: E402
from spar_horizon.horizons import UNIT_YEARS, Horizon  # noqa: E402

K = 4
CELLS = [("on", "suffix newline", 16), ("on", "suffix newline", 24), ("on", "pre-label", 22),
         ("off", "suffix newline", 16)]
SHOWN = ("log near delay", "log far delay", "near first", "log unit size", "regime near", "regime both",
         "log reward", "forced", "log think length")


def unit_names(log_unit_size):
    """The rendered unit for each row, from its log10 unit size."""
    sizes = {u: np.log10(y) for u, y in UNIT_YEARS.items()}
    return np.array([min(sizes, key=lambda u: abs(sizes[u] - s)) for s in log_unit_size])


def main(root):
    root = Path(root)
    rng = np.random.default_rng(0)
    found = [b for b in POOLED_BANKS if all((root / f"exp0_Qwen3-8B_think-{m}_{b}").exists() for m in ("off", "on"))]
    anchors = {mode: anchor_check(root, mode, load_parts(root, mode, "pooled"),
                                  [(p, l) for m, p, l in CELLS if m == mode]) for mode in ("off", "on")}
    pooled_cells = gated_jobs(CELLS, anchors)
    results, lines, loaded = {}, [], {}
    for mode, position, layer in CELLS:
        banks = tuple(found) if (mode, position, layer) in pooled_cells else ("full", "units")
        if (mode, banks) not in loaded:
            loaded.clear()
            loaded[mode, banks] = dataset(root, "Qwen3-8B", mode, banks)
        parts, data = loaded[mode, banks]
        X = cell_matrix(parts, position, layer)
        V = data["features"]
        duration = np.array([Horizon.parse(d).log10_years for d in data["duration_id"]])
        factors = {"unit": unit_names(V[:, 5]), "regime": data["regime"], "forced": V[:, 8] > 0.5,
                   "order": V[:, 4] > 0.5}
        covariates = {"log near delay": V[:, 0], "log far delay": V[:, 1]}
        folds = base_folds(data["pairs"])
        name = f"8B {'+'.join(banks)} {mode} {position} L{layer}"
        results[name] = {}
        lines.append(f"{name} (equivalent decades, k = {K}; * = 95% CI excludes 0)")
        lines.append(f"  {'readout':10s} {'cal':>5s} " + " ".join(f"{f[:14]:>14s}" for f in SHOWN))
        for method in ("old curve", "effects ⟂"):
            readout = cross_fit(method, X, duration, factors, covariates if method != "old curve" else {}, folds, K)[0]
            slope = calibration(readout, data["y"], data["duration_id"])
            fit = fe_regression(readout, V, list(FEATURES), data["durations"], data["pairs"], rng)
            results[name][method] = {"calibration": slope, "effects": fit}
            cells = []
            for f in SHOWN:
                if f in fit:
                    c, lo, hi = (v / slope for v in fit[f])
                    cells.append(f"{c:+.2f}{'*' if lo > 0 or hi < 0 else ' '}".rjust(14))
                else:
                    cells.append("–".rjust(14))
            lines.append(f"  {method:10s} {slope:5.2f} " + " ".join(cells))
        print("\n".join(lines[-4:]), flush=True)
    out = root / "analysis"
    # the old-runs-only version (18 §9) keeps its files
    stem = "pristine_confounds_fill" if "fill" in found else "pristine_confounds"
    (out / f"{stem}.json").write_text(json.dumps({**results, "anchor check": anchors}, indent=1))
    (out / f"{stem}.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
