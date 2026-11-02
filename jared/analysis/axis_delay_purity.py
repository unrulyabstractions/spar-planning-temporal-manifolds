"""Which steering axis moves the horizon without moving the delays? (overnight, 2026-10-30;
post hoc, not pre-registered)

19 §5: curves fitted on the old runs (delays in Alan's 1 week - 50 years) misread fill
prompts (delays of seconds to centuries) by up to 2 decades; curves fitted on the fill
read the old prompts. So the steering axis matters: a step along it should be a horizon
step and not also a delay step. Four candidate curves, each fitted with
pristine_manifold.fit (k = 4; unit, regime, forced and order as factors, log near and far
delay as covariates):
  (a) old effects      old runs only (8B full + units)
  (b) pooled effects   old + fill
  (c) pooled effects ⟂ old + fill, factor and delay effects projected out of the basis
  (d) fill effects ⟂   fill only, projected as (c)
Each curve's main direction is its points' slope on log horizon (delay_step.curve_direction):
the activation change per decade along the curve, the direction a 1-D step moves along.

Two measures, everything cross-fitted by base (fit on one fold, measured on the other, the
two folds averaged), so that (c) is not clean merely because it was fitted to the
same rows it is measured on:
  pull      the curve's nearest-point readout regressed on the true log horizon and the
            log near and far delays, across durations and within them (bank x duration
            fixed effects), on in-range (alan_range; every old-bank prompt) and
            out-of-range prompts separately. Readout decades per decade.
  purity    confound_sweep.axis_effects on the held-out rows: within a duration, each
            feature's effect on the activations, projected on the main direction, in
            decades of horizon per unit of the feature (all of confound_sweep.FEATURES
            that vary, so a delay's effect holds regime fixed). "In span": the length of
            the delay effect inside the curve's 4-D basis over the length of a horizon
            decade, which is what a k-D steering step could pick up. CIs resample bases
            (200 draws), paired across candidates, so differences between axes get CIs.
  step      the steering view of the same overlap: decades of delay contained in a
            one-decade horizon step along the main direction, (a·B) / |B|², with |B|²
            from the product of the two folds' independent estimates of the delay
            effect B (unbiased by noise). Point estimate only. Purity and step differ by
            |a|² / |B|², so an axis shortened by projection (c, d) does better on step
            than on purity.
Cosines between the four main directions come from fits on all rows.

    python analysis/axis_delay_purity.py /home/Jared/SPAR/results/horizon [--pre-label]
    -> <root>/analysis/axis_delay_purity.{json,txt,png}
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from confound_sweep import (FEATURES, axis_effects, base_folds, cell_matrix, dataset,  # noqa: E402
                            fe_coefficients, fe_regression, varying, within)
from delay_step import (BANKS, CELLS, K_CURVE, MODEL, PRE_LABEL_CELLS, curve_direction, fit_inputs,  # noqa: E402
                        project, row_records)
from pristine_manifold import fit, subset  # noqa: E402  (same folder)
from spar_horizon.steering import curve_readout  # noqa: E402  (read-only; another agent edits steering.py)

BOOT = 200
CANDIDATES = {"(a) old effects": ("effects", "old"), "(b) pooled effects": ("effects", "pooled"),
              "(c) pooled effects ⟂": ("effects ⟂", "pooled"), "(d) fill effects ⟂": ("effects ⟂", "fill")}
DELAYS = ("log near delay", "log far delay")
SIDE = ("regime near", "regime both", "near first", "log unit size")
SUBSETS = ("pooled", "old", "fill")
# categorical slots 1-2 of the dataviz reference palette, fixed order
COLORS = {"log near delay": "#2a78d6", "log far delay": "#eb6834"}


def source_rows(source, is_fill):
    return {"old": ~is_fill, "pooled": np.ones(len(is_fill), bool), "fill": is_fill}[source]


def fit_candidate(name, X, data, rows):
    """The candidate's curve fitted on `rows`."""
    method, _ = CANDIDATES[name]
    duration, factors, covariates = fit_inputs(data)
    return fit(method, X[rows], duration[rows], subset(factors, rows), subset(covariates, rows), K_CURVE)


# --- pull: what the readout answers ------------------------------------------------------

def pull(readout, data, rows, rng):
    """Across durations (one group, so an intercept) and within them: readout decades per
    decade of true horizon and of each delay."""
    y, V = data["y"][rows], data["features"][rows]
    across = np.column_stack([y, V[:, 0], V[:, 1]])
    names = ["log horizon", *DELAYS]
    return {"across": fe_regression(readout[rows], across, names, np.zeros(len(y)), data["pairs"][rows], rng),
            "within": fe_regression(readout[rows], V[:, :2], list(DELAYS), data["durations"][rows],
                                    data["pairs"][rows], rng)}


# --- purity: what a step along the axis also does ---------------------------------------

def common_columns(V, groups_by_fold):
    """FEATURES that vary within groups in every fold's rows."""
    kept = [set(varying(within(V[rows], groups), list(FEATURES))[1]) for rows, groups in groups_by_fold]
    return [n for n in FEATURES if all(n in k for k in kept)]


def purity(X, data, folds, cells_by_fold, rows, rng):
    """Per candidate: feature -> [estimate, 2.5%, 97.5%] in equivalent decades along its
    main direction, the delays' cosines with it and their in-span ratios; plus the paired
    bootstrap draws of the delay pull sqrt(near² + far²), for differences between axes."""
    names = list(CANDIDATES)
    V = data["features"]
    folds_rows = [(rows & (folds == f), data["durations"][rows & (folds == f)]) for f in np.unique(folds)]
    columns = common_columns(V, folds_rows)
    index = [list(FEATURES).index(c) for c in columns]
    estimates, draws, cosines, spans, effects, directions = [], [], [], [], [], []
    for (test, groups), cells in zip(folds_rows, cells_by_fold):
        bases = data["pairs"][test]
        V_w = within(V[test][:, index], groups)
        # per candidate: decades along the direction (1 column) and the 4 basis coordinates
        along = np.column_stack([project(cells[n], X[test]) for n in names])
        coords = [within((X[test] - cells[n]["mean"]) @ cells[n]["basis"].T, groups) for n in names]
        along_w = within(along, groups)
        estimates.append(fe_coefficients(along_w, V_w))                       # [features, candidates]
        lengths = [np.linalg.norm(curve_direction(cells[n])) for n in names]
        spans.append(np.array([np.linalg.norm(fe_coefficients(c, V_w), axis=1) / length
                               for c, length in zip(coords, lengths)]).T)  # [features, candidates]
        effects.append(fe_coefficients(within(X[test], groups), V_w))      # [features, d]
        directions.append({n: curve_direction(cells[n]) for n in names})
        cosines.append({n: axis_effects(X[test], V[test][:, index], columns, groups, curve_direction(cells[n]))
                        for n in names})
        keys = np.unique(bases)
        rows_of = {b: np.flatnonzero(bases == b) for b in keys}
        fold_draws = []
        for _ in range(BOOT):
            take = np.concatenate([rows_of[b] for b in rng.choice(keys, size=len(keys))])
            fold_draws.append(fe_coefficients(along_w[take], V_w[take]))
        draws.append(np.array(fold_draws))                                    # [boot, features, candidates]
    estimate, boot = np.mean(estimates, 0), np.mean(draws, 0)
    span = np.mean(spans, 0)
    # |effect|² from the two folds' independent estimates, free of the noise in either
    effect_size = (effects[0] * effects[1]).sum(1)
    delay_rows = [columns.index(d) for d in DELAYS]
    pull_size = np.sqrt((boot[:, delay_rows] ** 2).sum(1))                      # [boot, candidates]
    out = {}
    for j, n in enumerate(names):
        out[n] = {c: [float(estimate[i, j]), *np.percentile(boot[:, i, j], [2.5, 97.5]).tolist()]
                  for i, c in enumerate(columns)}
        out[n]["direction length"] = float(np.mean([np.linalg.norm(curve_direction(c[n])) for c in cells_by_fold]))
        out[n]["delay pull"] = [float(np.sqrt((estimate[delay_rows, j] ** 2).sum())),
                                *np.percentile(pull_size[:, j], [2.5, 97.5]).tolist()]
        for d in DELAYS:
            out[n][f"{d} cosine"] = float(np.mean([fold[n][d][1] for fold in cosines]))
            out[n][f"{d} in span"] = float(span[columns.index(d), j])
            i = columns.index(d)
            out[n][f"{d} per horizon step"] = float(np.mean([e[i] @ a[n] for e, a in zip(effects, directions)])
                                                    / effect_size[i])
    differences = {f"{a} - {b}": [float(out[a]["delay pull"][0] - out[b]["delay pull"][0]),
                                  *np.percentile(pull_size[:, i] - pull_size[:, k], [2.5, 97.5]).tolist()]
                   for i, a in enumerate(names) for k, b in enumerate(names) if i < k}
    return out, differences


def direction_cosines(X, data, is_fill):
    names = list(CANDIDATES)
    directions = [curve_direction(fit_candidate(n, X, data, source_rows(CANDIDATES[n][1], is_fill))) for n in names]
    unit = np.array([d / np.linalg.norm(d) for d in directions])
    return {a: {b: float(unit[i] @ unit[j]) for j, b in enumerate(names)} for i, a in enumerate(names)}


def cell_result(X, data, is_fill, alan, rng):
    folds = base_folds(data["pairs"])
    cells_by_fold, readouts = [], {n: np.empty(len(folds)) for n in CANDIDATES}
    for f in np.unique(folds):
        train, test = folds != f, folds == f
        cells = {n: fit_candidate(n, X, data, train & source_rows(source, is_fill))
                 for n, (_, source) in CANDIDATES.items()}
        for n, cell in cells.items():
            readouts[n][test] = curve_readout(cell, X[test])
        cells_by_fold.append(cells)
    out = {"pull": {n: {"in range": pull(r, data, alan, rng), "out of range": pull(r, data, ~alan, rng)}
                    for n, r in readouts.items()},
           "purity": {}, "differences": {}, "cosines": direction_cosines(X, data, is_fill),
           "n": {"in range": int(alan.sum()), "out of range": int((~alan).sum())}}
    for which in SUBSETS:
        out["purity"][which], out["differences"][which] = purity(X, data, folds, cells_by_fold,
                                                                 source_rows(which, is_fill), rng)
    return out


# --- output ------------------------------------------------------------------------------

def fmt(c):
    star = "*" if c[1] > 0 or c[2] < 0 else " "
    return f"{c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}]{star}"


def summary_lines(results):
    lines = ["* = 95% CI (bases resampled) excludes 0. Everything cross-fitted by base.", ""]
    for cell, m in results.items():
        lines.append(f"== {cell}   (in range {m['n']['in range']}, out of range {m['n']['out of range']})")
        lines.append("  pull: curve readout (decades) per decade; across durations: horizon, near, far | "
                     "within duration: near, far")
        for n, by_range in m["pull"].items():
            for rng_name, fits in by_range.items():
                across = " ".join(fmt(fits["across"][k]) for k in ("log horizon", *DELAYS))
                inside = " ".join(fmt(fits["within"][k]) for k in DELAYS if k in fits["within"])
                lines.append(f"    {n:22s} {rng_name:12s} across {across} | within {inside}")
        for which in SUBSETS:
            lines.append(f"  purity on {which} prompts: equivalent decades along the main direction per decade "
                         f"of delay; delay pull = sqrt(near² + far²); in span = |effect in the 4-D basis| / "
                         f"|horizon decade|; step = decades of delay in a one-decade horizon step along the "
                         f"axis (steering view)")
            for n, p in m["purity"][which].items():
                side = " ".join(f"{k} {p[k][0]:+.2f}" for k in SIDE if k in p)
                lines.append(f"    {n:22s} |dir| {p['direction length']:5.2f} near {fmt(p['log near delay'])} "
                             f"far {fmt(p['log far delay'])} "
                             f"pull {fmt(p['delay pull'])} | cos {p['log near delay cosine']:+.2f}/"
                             f"{p['log far delay cosine']:+.2f} step {p['log near delay per horizon step']:+.2f}/"
                             f"{p['log far delay per horizon step']:+.2f} in span {p['log near delay in span']:.2f}/"
                             f"{p['log far delay in span']:.2f} | {side}")
            lines.append("    pull differences: " + "; ".join(f"{k} {fmt(v)}"
                                                              for k, v in m["differences"][which].items()))
        names = list(m["cosines"])
        lines.append("  cosines between main directions: " + "; ".join(
            f"{a[:3]}·{b[:3]} {m['cosines'][a][b]:+.2f}" for i, a in enumerate(names) for b in names[i + 1:]))
        lines.append("")
    return lines


def plot(results, path):
    """Rows: cells. Per candidate axis, the near and far delay's pull along its main
    direction (pooled prompts), with 95% CIs."""
    cells = list(results)
    names = list(CANDIDATES)
    columns = min(3, len(cells))
    rows = -(-len(cells) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 2.9 * rows + 0.6), squeeze=False)
    for i, cell in enumerate(cells):
        ax = axes.flat[i]
        for k, delay in enumerate(DELAYS):
            for j, n in enumerate(names):
                est, lo, hi = results[cell]["purity"]["pooled"][n][delay]
                yy = j + (k - 0.5) * 0.25
                ax.errorbar(est, yy, xerr=[[est - lo], [hi - est]], fmt="o", color=COLORS[delay], ms=5, lw=1.5,
                            label=delay.replace("log ", "") if j == 0 else None)
        ax.axvline(0, color="#7a7974", lw=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace("⟂", "⊥") for n in names], fontsize=8)   # DejaVu Sans has no ⟂
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(cell, fontsize=9)
        ax.set_xlabel("decades of horizon along the axis per decade of delay", fontsize=8)
    for ax in axes.flat[len(cells):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Within a duration, how far a 10x longer delay moves activations along each candidate\n"
                 "steering axis (pooled prompts, cross-fitted; 0 = clean; error bars 95% CI)", fontsize=9, x=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(root, cells):
    root = Path(root)
    rng = np.random.default_rng(0)
    out = root / "analysis"
    results, loaded = {}, {}
    for mode, position, layer in tqdm(cells, desc="cells"):
        if mode not in loaded:
            loaded.clear()
            loaded[mode] = (*dataset(root, MODEL, mode, BANKS), row_records(root, mode))
        parts, data, records = loaded[mode]
        is_fill = np.array([r["bank"] == "fill" for r in records])
        alan = np.array([bool(r["alan_range"]) for r in records])
        results[f"{mode} {position} L{layer}"] = cell_result(cell_matrix(parts, position, layer), data, is_fill,
                                                             alan, rng)
        (out / "axis_delay_purity.json").write_text(json.dumps(results, indent=1))   # partial results survive
    text = "\n".join(summary_lines(results))
    (out / "axis_delay_purity.txt").write_text(text)
    plot(results, out / "axis_delay_purity.png")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1], CELLS + (PRE_LABEL_CELLS if "--pre-label" in sys.argv[2:] else []))
