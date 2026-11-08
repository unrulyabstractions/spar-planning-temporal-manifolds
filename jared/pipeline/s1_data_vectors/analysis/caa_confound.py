"""What is the CAA direction made of? (post hoc, 2026-11-03)

`08` §4: the mean difference long-minus-short horizons (CAA) at the L24 answer slot
steers toward near, the wrong way. Two suspected causes: long horizons are force-closed
more often (the direction partly reads "was cut off"), and the short half of the
horizons is all none-regime (half of those prompts pick near). Saved activations only.

At each cell, directions from the 8B thinking-on full bank (prompts with a horizon):
  caa       mean of the long half of the durations minus the short half (log horizon median split)
  forced    forced minus natural closes, both demeaned within duration (so not the horizon)
  near      chose near minus chose far, demeaned within duration and regime
  none-vs-both  none minus both, demeaned within duration, on the durations that hold
            prompts of both regimes (delays vary per prompt, so some do)
  caa within <regime>  long minus short half inside one regime (no composition change)
For each, the cosine with caa, and how much of caa's projection on the near direction
remains after projecting out the forced direction (the forced part of the wrong-way
push). Null: |cosine| of random unit vectors in 4096 dims, ~0.012 SD.

    python analysis/caa_confound.py /home/Jared/SPAR/results/horizon OUT_DIR
    -> OUT_DIR/caa_confound.{json,txt}
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_dimensionality import positions  # noqa: E402
from unit_overlap import load_run  # noqa: E402
from spar_horizon.runs import REGIONS  # noqa: E402

CELLS = (("suffix newline", 16), ("think close", 24), ("pre-label", 24), ("pre-label", 22))


def demean_within(X, groups):
    out = X.copy()
    for g in np.unique(groups):
        rows = groups == g
        out[rows] -= X[rows].mean(0)
    return out


def unit(v):
    return v / np.linalg.norm(v)


def directions(X, labels):
    y, durations = labels["y"], labels["durations"]
    long = y > np.median(np.unique(y))
    caa = X[long].mean(0) - X[~long].mean(0)
    within = demean_within(X, durations)
    forced = labels["forced"]
    forced_dir = within[forced].mean(0) - within[~forced].mean(0)
    groups = np.array([f"{d}|{r}" for d, r in zip(durations, labels["regime"])])
    within_regime = demean_within(X, groups)
    near = labels["chose_near"]
    near_dir = within_regime[near].mean(0) - within_regime[~near].mean(0)
    # durations holding both none and both prompts: their within-duration regime contrast
    mixed = [d for d in np.unique(durations)
             if {"none", "both"} <= set(labels["regime"][durations == d])]
    rows = np.isin(durations, mixed)
    regime_dir = (within[rows & (labels["regime"] == "none")].mean(0)
                  - within[rows & (labels["regime"] == "both")].mean(0)) if mixed else np.zeros(X.shape[1])
    out = {"caa": caa, "forced": forced_dir, "near": near_dir, "none-vs-both": regime_dir}
    # long minus short within one regime: no change of regime composition between the halves
    for regime in ("none", "near", "both"):
        rows = labels["regime"] == regime
        cut = np.median(np.unique(y[rows]))
        out[f"caa within {regime}"] = X[rows & (y > cut)].mean(0) - X[rows & (y <= cut)].mean(0)
    return out, len(mixed)


def cell_result(X, labels):
    dirs, n_mixed = directions(X, labels)
    caa = dirs["caa"]
    out = {"n mixed durations": n_mixed, "|caa|": float(np.linalg.norm(caa)),
           "cos with caa": {k: float(unit(caa) @ unit(v)) for k, v in dirs.items() if k != "caa" and np.linalg.norm(v) > 0}}
    # the wrong-way push: caa's component along the near direction, before and after removing the forced direction
    near, forced = unit(dirs["near"]), unit(dirs["forced"])
    caa_no_forced = caa - (caa @ forced) * forced
    out["caa · near (per |caa|)"] = float(unit(caa) @ near)
    out["caa ⟂ forced · near (per |caa|)"] = float(caa_no_forced @ near / np.linalg.norm(caa))
    out["cos(forced, near)"] = float(forced @ near)
    # spread of the cell along the near direction, for scale: SD of projections over all prompts
    out["spread along near (SD, all prompts)"] = float((X @ near).std())
    out["caa step along near / spread"] = float((caa @ near) / (X @ near).std())
    out["within-regime caa · near (per |caa|)"] = {r: float(unit(dirs[f"caa within {r}"]) @ near) for r in ("none", "near", "both")}
    return out


def main(root, out_dir):
    root, out_dir = Path(root), Path(out_dir)
    z, labels = load_run(root, "on", "full", "Qwen3-8B")
    meta = {"regions": {r: tuple(int(x) for x in z[f"region_{r}"]) for r in REGIONS},
            "tokens": [str(t) for t in z["tokens"]]}
    rng = np.random.default_rng(0)
    null = np.abs([unit(a) @ unit(b) for a, b in rng.normal(size=(200, 2, z["L16"].shape[-1]))])
    results = {"null |cos| 95th pct": float(np.percentile(null, 95)), "cells": {}}
    for position, layer in CELLS:
        X = z[f"L{layer}"][labels["rows"], positions(meta)[position]].astype(np.float64)
        results["cells"][f"{position} L{layer}"] = cell_result(X, labels)
    (out_dir / "caa_confound.json").write_text(json.dumps(results, indent=1))
    lines = [f"CAA composition, 8B thinking on, full bank. Null |cos| 95th percentile {results['null |cos| 95th pct']:.3f}", ""]
    for name, r in results["cells"].items():
        lines.append(f"{name}: " + ", ".join(f"cos(caa, {k}) {v:+.2f}" for k, v in r["cos with caa"].items())
                     + f"; cos(forced, near) {r['cos(forced, near)']:+.2f}")
        lines.append(f"   caa·near {r['caa · near (per |caa|)']:+.3f} -> after removing forced {r['caa ⟂ forced · near (per |caa|)']:+.3f}"
                     f" (per |caa|); caa step along near = {r['caa step along near / spread']:+.2f} SD of prompts"
                     f" ({r['n mixed durations']} durations hold both none and both)")
        lines.append("   within-regime caa · near: " + ", ".join(f"{k} {v:+.3f}" for k, v in r["within-regime caa · near (per |caa|)"].items()))
    text = "\n".join(lines)
    (out_dir / "caa_confound.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:3])
