"""Are the steering directions stable enough to read a steering result? (post hoc, 2026-10-30)

Stage A/B steer along the pooled fit's effects curve (`fit_duration_effects`, no ⟂
step, k = 8, as `steering_prepare.pristine_cell`) and along its split (16 §5):
  effects    the curve's step per decade at a horizon: (Δ(+1) - Δ(-1)) / 2, the mean of
             the α = ±1 steps, so the curve's ends keep a step
  invariant  the step without its component in the span of the factor and delay effects
  in-span    that component (the two add up to the step)
  U          the first PC of the unit effects (`unit_direction`)
If two reasonable fits of the same cell disagree on these directions, a steering result
along one of them says little about "the" direction. So each is fitted on
  the two base folds (`base_folds`) of the pooled runs,
  the old runs (full + units) against the pooled runs (+ fill),
  thinking off against thinking on, at the cells both modes have,
and compared by cosine, per duration of the pooled grid, averaged (U: one vector). The
fold cosine is between half-data fits; "folds → full data" projects it to two fits on all
the data (`split_half_projection`). U is oriented by its sign on the curve's chord, so
its cosine with the chord is reported too: near 0, a refit can flip it. Also
the in-span share of the effects step (mean |in-span|² / |step|²), and the share in each
factor's own span. Two random directions
in 4,096 dimensions have |cosine| about 0.016.

    python analysis/direction_stability.py /home/Jared/SPAR/results/horizon
    -> <root>/analysis/direction_stability.{json,txt}

Reads saved activations only; runs no model.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from confound_sweep import FEATURES, base_folds, cell_matrix, dataset  # noqa: E402  (same folder)
from pristine_confounds import unit_names  # noqa: E402  (same folder)
from pristine_manifold import subset  # noqa: E402  (same folder)
from spar_horizon.horizons import Horizon  # noqa: E402
from spar_horizon.steering import curve_delta, fit_duration_effects, span_part, unit_direction  # noqa: E402

MODEL = "Qwen3-8B"
POOLED_BANKS = ("full", "units", "fill")
OLD_BANKS = ("full", "units")
# the steering cells of 16 §5: stage A (off suffix, both modes' answer slot) and stage B (on suffix)
STEERING_CELLS = [("off", "suffix newline", 16), ("off", "pre-label", 22), ("off", "pre-label", 24),
                  ("on", "suffix newline", 16), ("on", "pre-label", 22), ("on", "pre-label", 24)]
PARTS = ("effects", "invariant", "in-span", "U")


def banks_of(data):
    """Each row's bank, from confound_sweep's bank-prefixed durations."""
    return np.array([d.split("|")[0] for d in data["durations"]])


def fit_inputs(data):
    """Log duration, factors and delay covariates of the pooled fit (16 §4), from
    confound_sweep's feature matrix, as pristine_confounds builds them."""
    V = data["features"]
    column = {name: V[:, j] for j, name in enumerate(FEATURES)}
    duration = np.array([Horizon.parse(d).log10_years for d in data["duration_id"]])
    factors = {"unit": unit_names(column["log unit size"]), "regime": data["regime"],
               "forced": column["forced"] > 0.5, "order": column["near first"] > 0.5}
    covariates = {"log near delay": column["log near delay"], "log far delay": column["log far delay"]}
    return duration, factors, covariates


def fit_effects(X, duration, factors, covariates, rows):
    """The steering fit (effects curve, no ⟂ step, default k as steering_prepare) on `rows`."""
    return fit_duration_effects(X[rows], duration[rows], subset(factors, rows), orthogonal=False,
                                covariates=subset(covariates, rows))


def steering_directions(cell, log_h):
    """name -> [n, d] per row of `log_h`: the effects step per decade, its invariant and
    in-span parts, and U at the step's length (as stage A steers it). A row whose step is
    zero (its horizon beyond the curve's ends) is all zeros in every part."""
    step = (curve_delta(cell, log_h, 1) - curve_delta(cell, log_h, -1)) / 2
    in_span = span_part(step, cell["factor_span"])
    length = np.linalg.norm(step, axis=1, keepdims=True)
    return {"effects": step, "invariant": step - in_span, "in-span": in_span,
            "U": length * unit_direction(cell)[None, :]}


def mean_cosine(a, b):
    """Mean cosine between matching rows of a and b, over rows nonzero in both."""
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    keep = norms > 1e-12
    return float(np.mean((a[keep] * b[keep]).sum(1) / norms[keep]))


def in_span_share(directions):
    """Mean share of the effects step's squared length that lies in the factor span."""
    step, in_span = directions["effects"], directions["in-span"]
    total = (step ** 2).sum(1)
    keep = total > 1e-24
    return float(np.mean((in_span[keep] ** 2).sum(1) / total[keep]))


def factor_shares(cell, steps):
    """name -> mean share of the steps' squared length in the span of that factor's (or
    covariate's) effects alone: which factor the in-span part comes from. A small, noisy
    effect still spans a full direction, so a share here is not an effect size."""
    vectors = {name: effects for name, (_, effects) in cell["factors"].items() if len(effects)}
    vectors.update({name: slope[None, :] for name, slope in cell["covariates"].items()})
    total = (steps ** 2).sum(1)
    keep = total > 1e-24
    out = {}
    for name, effects in vectors.items():
        span = np.linalg.qr(effects.T)[0].T
        out[name] = float(np.mean((span_part(steps[keep], span) ** 2).sum(1) / total[keep]))
    return out


def compare(first, second):
    """part -> cosine between two fits' directions on the same grid."""
    return {part: mean_cosine(first[part], second[part]) for part in PARTS}


def u_orientation_margin(cell):
    """Cosine of U with the curve's chord (shortest to longest duration), whose sign
    orients U: near 0, a refit can flip U's sign while keeping its axis."""
    chord = (cell["curve_coords"][-1] - cell["curve_coords"][0]) @ cell["basis"]
    return float(unit_direction(cell) @ chord / np.linalg.norm(chord))


def split_half_projection(cosine):
    """The cosine expected between two fits on all the data, from that between two
    half-data fits, if each direction is a fixed signal plus independent fit noise
    (noise variance halves with twice the data): 2c / (1 + c)."""
    return 2 * cosine / (1 + cosine)


def cell_fits(X, data):
    """Four fits of one cell (pooled, each base fold, the old runs only): their steering
    directions on the pooled duration grid, U's orientation margin, and the pooled
    step's share in each factor's span."""
    duration, factors, covariates = fit_inputs(data)
    grid = np.unique(duration)
    folds = base_folds(data["pairs"])
    rows = {"pooled": np.ones(len(X), bool), "fold 0": folds == 0, "fold 1": folds == 1,
            "old": np.isin(banks_of(data), OLD_BANKS)}
    cells = {name: fit_effects(X, duration, factors, covariates, r) for name, r in rows.items()}
    directions = {name: steering_directions(cell, grid) for name, cell in cells.items()}
    return (directions, {name: u_orientation_margin(cell) for name, cell in cells.items()},
            factor_shares(cells["pooled"], directions["pooled"]["effects"]))


def summary_lines(results):
    lines = ["Steering-direction stability (post hoc). Mean cosine per duration; U is one vector. "
             "in-span share = mean |in-span|²/|step|² of the effects step (pooled fit).",
             f"{'cell':26s} {'comparison':16s} " + " ".join(f"{p:>9s}" for p in PARTS)]
    for cell, m in results.items():
        shares = ", ".join(f"{name} {share:.2f}" for name, share in m["in-span share"].items())
        for comparison, cosines in m["cosines"].items():
            lines.append(f"{cell:26s} {comparison:16s} " + " ".join(f"{cosines[p]:+9.2f}" for p in PARTS))
        lines.append(f"{'':26s} in-span share: {shares}")
        margins = ", ".join(f"{name} {v:+.2f}" for name, v in m["U orientation margin"].items())
        lines.append(f"{'':26s} cos(U, curve chord): {margins}")
        by_factor = ", ".join(f"{name} {v:.2f}" for name, v in m["pooled share by factor"].items())
        lines.append(f"{'':26s} pooled step's share by factor span alone: {by_factor}")
    return lines


def main(root):
    root = Path(root)
    results, pooled = {}, {}
    for mode in ("off", "on"):
        parts, data = dataset(root, MODEL, mode, POOLED_BANKS)
        for m, position, layer in STEERING_CELLS:
            if m != mode:
                continue
            fits, margins, shares = cell_fits(cell_matrix(parts, position, layer), data)
            pooled[mode, position, layer] = fits["pooled"]
            folds = compare(fits["fold 0"], fits["fold 1"])
            results[f"{mode} {position} L{layer}"] = {
                "cosines": {"fold 0 vs fold 1": folds,
                            "folds → full data": {p: split_half_projection(c) for p, c in folds.items()},
                            "old vs pooled": compare(fits["old"], fits["pooled"])},
                "in-span share": {name: in_span_share(d) for name, d in fits.items()},
                "U orientation margin": margins, "pooled share by factor": shares}
            print(f"done {mode} {position} L{layer}", flush=True)
        del parts, data
    for _, position, layer in STEERING_CELLS:
        if ("off", position, layer) in pooled and ("on", position, layer) in pooled:
            cosines = compare(pooled["off", position, layer], pooled["on", position, layer])
            results[f"on {position} L{layer}"]["cosines"]["off vs on"] = cosines
    out = root / "analysis"
    (out / "direction_stability.json").write_text(json.dumps(results, indent=1))
    text = "\n".join(summary_lines(results))
    (out / "direction_stability.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
