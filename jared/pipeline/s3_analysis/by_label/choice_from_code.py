"""Does the encoded horizon predict the choice beyond the true horizon? (post hoc, 2026-10-30)

The correlational precondition for steering (16 §5). Saved activations and answers of the
8B runs, pooled as the steering fit is (full + units + fill, anchors left out).

Q1. Readout residual -> choice. A horizon readout cross-fitted by base (`base_folds`):
      effects ⟂   the pooled fit with its factor span projected out (k = 4, as judged in 17/19)
      effects     the same without the ⟂ step (the steering curve's shape)
      probe       ridge on the top-8 duration-average PCs (confound_sweep's probe)
    Its residual, readout - true log10 horizon, enters a linear probability model of
    chose-near within duration × regime × bank, so the true horizon, the regime and the
    bank are held fixed. Coefficients are ΔP(near) per decade of readout residual.

Q2. Which part of the code? Out-of-fold coordinates of each prompt along the steering
    directions of a fit on the other base fold (`direction_stability.steering_directions`):
    invariant, in-span, U, and N_RANDOM random directions. Coordinates are in steering
    units: adding α × that direction's step (as stage A/B do) moves the coordinate by α.
    So a coefficient per α is the correlational prediction of steering's ΔP(near) at
    α = 1; per SD (of the coordinate within groups) compares directions fairly.

Controls in every model: log near and far delay, log near reward, log reward ratio,
forced close, and option order (near first), interacted with the fill bank and with
regime, because the order effect reverses on the fill (19 §7) and differs by regime
(18 §1). CIs resample bases, 200 times (`fe_regression`).

Expected signs, written down before any fit was looked at (positive = more near):
  near   −  a longer-read horizon brings the far delay toward fitting
  both   −  a longer read puts the far delay further inside, a shorter one can push it out
  none   ?  regime logic: a longer read makes the near delay fit first (+); a generic
            "longer horizon, more patient" reading says −
Rows: thinking off on all banks (the fill gives it the most choice variance); thinking
on on the old banks (near ceiling on the fill, 19 §7). The other row sets are reported
alongside.

    python -m pipeline.s3_analysis.by_label.choice_from_code /home/Jared/SPAR/results/horizon
    -> <root>/analysis/choice_from_code.{json,txt,png}

Reads saved activations only; runs no model.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pipeline.s3_analysis.by_label.confound_sweep import (FEATURES, base_folds, calibration, cell_matrix, crossfit_readout, dataset,
                            fe_regression, within)
from pipeline.s1_data_vectors.analysis.direction_stability import (MODEL, POOLED_BANKS, banks_of, fit_effects, fit_inputs,
                                 steering_directions)
from pipeline.s1_data_vectors.analysis.pristine_manifold import JUDGED_K, cross_fit

CELLS = {"off": [("suffix newline", 16), ("pre-label", 22), ("pre-label", 24)],
         "on": [("suffix newline", 16), ("think close", 22), ("think close", 24), ("pre-label", 22),
                ("pre-label", 24)]}
PRIMARY_ROWS = {"off": "all", "on": "old"}
ROW_SETS = ("all", "old", "fill")
SCOPES = ("all regimes", "none", "near", "both")
EXPECTED = {"all regimes": "", "none": "?", "near": "−", "both": "−"}
READOUTS = ("effects ⟂", "effects", "probe")
DIRECTIONS = ("invariant", "in-span", "U")
N_RANDOM = 50
# categorical slots 1-4 of the dataviz reference palette, in fixed order (as pristine_manifold)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
NULL_GRAY = "#c3c2b7"


# --- the regression ---------------------------------------------------------------------

def choice_frame(data):
    """Everything a choice regression needs besides the signal: outcome, controls,
    groups (duration × regime × bank), bases, and the row selectors."""
    column = {name: data["features"][:, j] for j, name in enumerate(FEATURES)}
    regime, bank = data["regime"], banks_of(data)
    near_first = column["near first"]
    controls = {name: column[name] for name in ("log near delay", "log far delay", "log reward", "log ratio",
                                                "forced")}
    controls.update({"near first": near_first, "near first × fill": near_first * (bank == "fill"),
                     "near first × near": near_first * (regime == "near"),
                     "near first × both": near_first * (regime == "both")})
    return {"chose_near": data["chose_near"], "controls": np.column_stack(list(controls.values())),
            "control_names": list(controls), "groups": np.array([f"{d}|{r}" for d, r in zip(data["durations"], regime)]),
            "bases": data["pairs"], "regime": regime, "bank": bank}


def row_mask(frame, row_set, scope):
    """Rows of a row set (all / old / fill banks) and regime scope."""
    rows = {"all": np.ones(len(frame["bank"]), bool), "old": frame["bank"] != "fill",
            "fill": frame["bank"] == "fill"}[row_set]
    return rows if scope == "all regimes" else rows & (frame["regime"] == scope)


def choice_effect(frame, signals, rows, rng):
    """Linear probability model of chose-near on the `signals` (name -> per-row values)
    and the controls, within groups, on `rows` (rows with a NaN signal left out).
    name -> {"effect": [ΔP(near) per unit, 2.5%, 97.5%] or None if it does not vary,
    "sd": its within-group SD}, plus n and the near rate."""
    rows = rows & np.all([np.isfinite(v) for v in signals.values()], axis=0)
    names = list(signals)
    V = np.column_stack([*(signals[n][rows] for n in names), frame["controls"][rows]])
    fit = fe_regression(frame["chose_near"][rows], V, names + frame["control_names"], frame["groups"][rows],
                        frame["bases"][rows], rng)
    out = {"n": int(rows.sum()), "near rate": float(frame["chose_near"][rows].mean())}
    for n in names:
        out[n] = {"effect": fit.get(n), "sd": float(within(signals[n][rows], frame["groups"][rows]).std())}
    return out


# --- Q2: coordinates along the steering directions ---------------------------------------

def alpha_coordinate(centered, direction):
    """Per row: the coordinate along that row's direction in steering units, x·P / P·P,
    so that adding α·P moves it by α. NaN where the direction is zero."""
    norm2 = (direction ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm2 > 1e-24, (centered * direction).sum(1) / norm2, np.nan)


def direction_coordinates(X, data, seed=0):
    """Out-of-fold coordinates (steering units) along invariant, in-span and U of a fit on
    the other base fold, and along N_RANDOM random directions at the effects step's length
    ([n, N_RANDOM])."""
    duration, factors, covariates = fit_inputs(data)
    folds = base_folds(data["pairs"])
    random = np.random.default_rng(seed).normal(size=(N_RANDOM, X.shape[1]))
    random /= np.linalg.norm(random, axis=1, keepdims=True)
    coords = {name: np.full(len(X), np.nan) for name in DIRECTIONS}
    coords["random"] = np.full((len(X), N_RANDOM), np.nan)
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        cell = fit_effects(X, duration, factors, covariates, train)
        directions = steering_directions(cell, duration[test])
        centered = X[test] - cell["mean"]
        for name in DIRECTIONS:
            coords[name][test] = alpha_coordinate(centered, directions[name])
        length = np.linalg.norm(directions["effects"], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            coords["random"][test] = np.where(length[:, None] > 1e-12, centered @ random.T / length[:, None], np.nan)
    return coords


def per_sd(entry):
    """[effect, low, high] per within-group SD of the signal."""
    return None if entry["effect"] is None else [v * entry["sd"] for v in entry["effect"]]


def direction_results(frame, coords, rows, rng):
    """Each direction alone (every regime scope), the three jointly, and the random
    directions' per-SD effects as a null; `random share above` is, per direction, the
    share of random directions with a larger |effect per SD| (all regimes)."""
    alone = {scope: {name: choice_effect(frame, {name: coords[name]}, row_mask(frame, rows, scope), rng)
                     for name in DIRECTIONS} for scope in SCOPES}
    joint = choice_effect(frame, {name: coords[name] for name in DIRECTIONS}, row_mask(frame, rows, "all regimes"),
                          rng)
    null = np.array([per_sd(choice_effect(frame, {"random": coords["random"][:, j]},
                                          row_mask(frame, rows, "all regimes"), rng)["random"])[0]
                     for j in range(N_RANDOM)])
    above = {name: float(np.mean(np.abs(null) >= abs(per_sd(alone["all regimes"][name][name])[0])))
             for name in DIRECTIONS}
    finite = np.isfinite(coords["invariant"]) & np.isfinite(coords["in-span"]) & np.isfinite(coords["U"])
    return {"alone": alone, "joint": joint, "random per SD": null.tolist(), "random share above": above,
            "correlation": np.corrcoef(np.vstack([coords[n][finite] for n in DIRECTIONS])).tolist()}


# --- per cell --------------------------------------------------------------------------

def cell_result(X, data, frame, mode, rng):
    duration, factors, covariates = fit_inputs(data)
    folds = base_folds(data["pairs"])
    y = data["y"]
    readouts = {"effects ⟂": cross_fit("effects ⟂", X, duration, factors, covariates, folds, JUDGED_K)[0],
                "effects": cross_fit("effects", X, duration, factors, covariates, folds, JUDGED_K)[0],
                "probe": crossfit_readout(X, y, data["duration_id"], folds)}
    by_readout = {name: {rows: {scope: choice_effect(frame, {"residual": readout - y}, row_mask(frame, rows, scope),
                                                     rng) for scope in SCOPES} for rows in ROW_SETS}
                  for name, readout in readouts.items()}
    # robustness: the unit word moves the readout (18 §7) and could move the choice by another path
    unit_size = data["features"][:, FEATURES.index("log unit size")]
    with_unit = choice_effect(frame, {"residual": readouts["effects ⟂"] - y, "log unit size": unit_size},
                              row_mask(frame, PRIMARY_ROWS[mode], "all regimes"), rng)
    q1 = {"calibration": {name: calibration(readout, y, data["duration_id"]) for name, readout in readouts.items()},
          "by readout": by_readout, "effects ⟂ + unit size control": with_unit}
    coords = direction_coordinates(X, data)
    q2 = {rows: direction_results(frame, coords, rows, rng) for rows in ROW_SETS}
    return {"Q1": q1, "Q2": q2}


# --- text and figure -------------------------------------------------------------------

def fmt(entry):
    if entry is None:
        return "–"
    c, lo, hi = entry
    return f"{c:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ' '}"


def sign_mark(scope, entry):
    """✓ / ✗: whether the point estimate has the expected sign (blank where none is predicted)."""
    if entry is None or EXPECTED[scope] in ("", "?"):
        return " "
    return "✓" if (entry[0] < 0) == (EXPECTED[scope] == "−") else "✗"


def summary_lines(results):
    lines = ["Choice from the code (post hoc). Linear probability model of chose-near within duration × regime × "
             "bank, controls: log near/far delay, log reward, log reward ratio, forced, near first (× fill, × regime).",
             "[95% CI] resampling bases; * = CI excludes 0. Q1: ΔP(near) per decade of readout residual (readout − "
             "true log10 horizon); expected sign near −, both −, none ? (✓/✗ = point estimate's sign).",
             "Q2: ΔP(near) per α of steering along the direction (coordinate in steering units, fit on the other "
             "base fold) and per within-group SD; 'rand≥' = share of 50 random directions with a larger |per SD|.",
             ""]
    for cell, m in results.items():
        rows = PRIMARY_ROWS[cell.split()[0]]
        q1, q2 = m["Q1"], m["Q2"][rows]
        counts = q1["by readout"]["effects ⟂"][rows]
        lines.append(f"== {cell}, {rows} banks   n / P(near): " + ", ".join(
            f"{scope} {counts[scope]['n']}/{counts[scope]['near rate']:.2f}" for scope in SCOPES))
        lines.append(f"  Q1 {'readout (calibration, residual sd)':38s}" + "".join(f"{s:>28s}" for s in SCOPES))
        for name, by_rows in q1["by readout"].items():
            label = f"{name} ({q1['calibration'][name]:.2f}, {by_rows[rows]['all regimes']['residual']['sd']:.2f})"
            lines.append(f"     {label:38s}" + "".join(
                f"{fmt(r['residual']['effect'])}{sign_mark(scope, r['residual']['effect'])}".rjust(28)
                for scope, r in by_rows[rows].items()))
        unit = q1["effects ⟂ + unit size control"]
        lines.append(f"     effects ⟂ with a unit-size control: residual {fmt(unit['residual']['effect'])}, "
                     f"log unit size {fmt(unit['log unit size']['effect'])}")
        for other in ROW_SETS:
            if other != rows:
                lines.append(f"     effects ⟂ on {other} banks: " + " | ".join(
                    f"{scope} {fmt(r['residual']['effect'])}{sign_mark(scope, r['residual']['effect'])} n={r['n']}"
                    for scope, r in q1["by readout"]["effects ⟂"][other].items()))
        null = np.array(q2["random per SD"])
        lines.append(f"  Q2 {'direction':10s} {'per α, all regimes':>26s} {'per SD':>8s} {'sd (α)':>7s} {'rand≥':>6s}"
                     + "".join(f"{'per α, ' + s:>28s}" for s in SCOPES[1:]))
        for name in DIRECTIONS:
            r = q2["alone"]["all regimes"][name][name]
            lines.append(f"     {name:10s} {fmt(r['effect']):>26s} {per_sd(r)[0]:+8.3f} {r['sd']:7.2f} "
                         f"{q2['random share above'][name]:6.2f}" + "".join(
                             f"{fmt(q2['alone'][s][name][name]['effect'])}{sign_mark(s, q2['alone'][s][name][name]['effect'])}"
                             .rjust(28) for s in SCOPES[1:]))
        lo, hi = np.percentile(null, [2.5, 97.5])
        corr = q2["correlation"]
        lines.append(f"     random per SD: 95% range [{lo:+.3f}, {hi:+.3f}]; joint per α: " + ", ".join(
            f"{n} {fmt(q2['joint'][n]['effect'])}" for n in DIRECTIONS)
            + f"; corr inv/in-span {corr[0][1]:+.2f}, inv/U {corr[0][2]:+.2f}, in-span/U {corr[1][2]:+.2f}")
        for other in ROW_SETS:
            if other != rows:
                lines.append(f"     per α on {other} banks, all regimes: " + ", ".join(
                    f"{n} {fmt(m['Q2'][other]['alone']['all regimes'][n][n]['effect'])}" for n in DIRECTIONS))
        lines.append("")
    return lines


def plot(results, path):
    """Left: Q1, the effects ⟂ residual's effect per decade, by regime scope. Right: Q2,
    each direction's effect per SD of its coordinate, with the random directions' 95%
    range behind. Primary rows of each mode."""
    cells = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(cells) + 2.4), sharey=True)
    ypos = np.arange(len(cells))[::-1]
    for i, cell in enumerate(cells):
        rows = PRIMARY_ROWS[cell.split()[0]]
        for j, scope in enumerate(SCOPES):
            e = results[cell]["Q1"]["by readout"]["effects ⟂"][rows][scope]["residual"]["effect"]
            if e is not None:
                y = ypos[i] + 0.27 - 0.18 * j
                axes[0].errorbar(e[0], y, xerr=[[e[0] - e[1]], [e[2] - e[0]]], fmt="o", ms=5, lw=1.5,
                                 color=COLORS[j], label=scope if i == 0 else None)
        q = results[cell]["Q2"][rows]
        lo, hi = np.percentile(q["random per SD"], [2.5, 97.5])
        axes[1].fill_betweenx([ypos[i] - 0.35, ypos[i] + 0.35], lo, hi, color=NULL_GRAY, alpha=0.6, lw=0,
                              label="random, 95% range" if i == 0 else None)
        for j, name in enumerate(DIRECTIONS):
            e = per_sd(q["alone"]["all regimes"][name][name])
            if e is not None:
                y = ypos[i] + 0.18 - 0.18 * j
                axes[1].errorbar(e[0], y, xerr=[[e[0] - e[1]], [e[2] - e[0]]], fmt="s", ms=5, lw=1.5,
                                 color=COLORS[j], label=name if i == 0 else None)
    labels = [f"{c}\n({PRIMARY_ROWS[c.split()[0]]} banks)" for c in cells]
    axes[0].set_yticks(ypos, labels, fontsize=8)
    axes[0].set_xlabel("ΔP(near) per decade of readout residual\n(effects ⊥ readout; positive = longer read, more near)",
                       fontsize=9)
    axes[1].set_xlabel("ΔP(near) per within-group SD of the coordinate\n(positive = toward a longer horizon, more near)",
                       fontsize=9)
    axes[0].set_title("Q1: readout residual → choice", fontsize=10)
    axes[1].set_title("Q2: which part of the code (all regimes)", fontsize=10)
    for ax in axes:
        ax.axvline(0, color="#7a7974", lw=0.8, ls="--", zorder=0)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(root):
    root = Path(root)
    rng = np.random.default_rng(0)
    results = {}
    jobs = [(mode, position, layer) for mode in ("off", "on") for position, layer in CELLS[mode]]
    loaded = {}
    out = root / "analysis"
    for mode, position, layer in tqdm(jobs, desc="cells"):
        if mode not in loaded:
            loaded.clear()   # one mode in memory at a time
            parts, data = dataset(root, MODEL, mode, POOLED_BANKS)
            loaded[mode] = (parts, data, choice_frame(data))
        parts, data, frame = loaded[mode]
        X = cell_matrix(parts, position, layer)
        results[f"{mode} {position} L{layer}"] = cell_result(X, data, frame, mode, rng)
        (out / "choice_from_code.json").write_text(json.dumps(results, indent=1))   # partial results survive
    text = "\n".join(summary_lines(results))
    (out / "choice_from_code.txt").write_text(text)
    plot(results, out / "choice_from_code.png")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
