"""Does the horizon code step where a delay crosses the horizon, or slope with the delay?
(overnight, 2026-10-30; post hoc, not pre-registered)

19 §4: in the fill bank, regime is almost a function of the delays at a fixed duration,
so "regime both" cannot be told apart from delay level by a regime coefficient. Here the
two are modelled together. The model, written down before fitting: within a duration
(bank and duration fixed effects), with margins m = log10(delay / horizon) for the near
and the far delay,

    r_i = a_d + Σ_{near, far} [ b·m + c·max(m, 0) + s·1(m < 0) ] + controls + e_i

  b, b + c  the smooth slope below and above the crossing (continuous at 0, so a kink at
            the crossing counts as smooth)
  s         the step: the jump at m = 0 between the two sides' lines
  controls  option order and log unit size (they vary within a duration and move the code,
            18 §1; forced is left out, it happens after the suffix)

A code that compares a delay with the horizon ("fits / doesn't fit") has s ≠ 0; one that
reads the delays' magnitudes has b ≠ 0 and s = 0. Robustness: the smooth part as a
restricted cubic spline (4 knots) in the absolute log delay, the magnitude model proper,
instead of the hinge; and steps alone, which is the regime coefficient of 19 §4. The fill
keeps every delay at least 0.2 decades from every horizon rendering, so on fill prompts
the step is the gap between the two lines across [-0.2, 0.2]; the old banks' delays cross
continuously.

Readouts, each cross-fitted by base (confound_sweep.base_folds), in equivalent decades:
each is divided by its calibration (readout decades per true decade across duration
means) on the old prompts, whose delays do not move with the horizon. On the fill they
do, so a fill calibration would mix the delays into the unit.
  fill probe     confound_sweep's ridge probe (8 duration-average PCs), fitted on the fill
                 prompts of the other fold only. Fill delays vary by ±2 decades within a
                 duration, so the ridge is pushed to ignore them; across durations they
                 track the horizon, so it may also use them.
  effects ⟂      pristine_manifold's pooled curve (k = 4): unit, regime, forced and order
                 factors and the log delays as covariates, their span projected out.
                 Regime is one of those factors, so the step direction is projected out by
                 construction: expect this readout to under-read the step.
  horizon axis   the main direction of the pooled effects curve (k = 4, nothing projected
                 out): its points' slope on log horizon. The duration effects hold the
                 delays fixed, so the axis has not learned them, and a projection cannot
                 subtract a nuisance (18 §1).
CIs resample base prompts (fe_regression, 200 draws). Data: 8B full + units + fill, every
cell pooled (Jared, 2026-10-30), anchors left out (confound_sweep.dataset).

Q3 (19 §5): the old runs' fit reads the fill's shortest third ~2 decades too long. Is
that the delays falling inside the horizon (regime) or their absolute scale? The
old-runs effects ⟂ curve (k = 4) reads every prompt, cross-fitted (the non-orthogonal
effects curve is not used: its nearest-point readout is compressed to ~0.55 per decade
even on in-range prompts). Its error on fill prompts is tabulated by regime and the
fill's delay_scale label, and regressed on the log horizon, the absolute log delays and
the two inside-the-horizon indicators, across fill prompts and across all prompts.

    python -m pipeline.s3_analysis.by_label.delay_step /home/Jared/SPAR/results/horizon [--pre-label]
    -> <root>/analysis/delay_step.{json,txt}, delay_step_{fill,pooled}.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pipeline.s3_analysis.by_label.confound_sweep import (base_folds, calibration, cell_matrix, dataset, fe_regression,
                            fit_probe, within)
from pipeline.s3_analysis.by_label.delay_margin import binned
from pipeline.s1_data_vectors.analysis.pristine_confounds import unit_names
from pipeline.s1_data_vectors.analysis.pristine_manifold import cross_fit, fit, subset
from pipeline.s3_analysis.by_datatype.unit_overlap import load_run
from pipeline.s3_analysis.by_datatype.written_unit import run_records
from spar_horizon.horizons import Horizon
from spar_horizon.steering import curve_readout

MODEL = "Qwen3-8B"
BANKS = ("full", "units", "fill")
K_CURVE = 4
CELLS = [("on", "suffix newline", 16), ("on", "suffix newline", 24),
         ("off", "suffix newline", 16), ("off", "suffix newline", 24)]
PRE_LABEL_CELLS = [("on", "pre-label", 22), ("on", "pre-label", 24)]
READOUTS = ("fill probe", "effects ⟂", "horizon axis")
SPECS = ("hinge", "spline", "step only")
MARGIN_BINS = np.arange(-3.5, 3.01, 0.5)
# categorical slots 1-3 of the dataviz reference palette, fixed order (as pristine_manifold)
COLORS = dict(zip(READOUTS, ("#2a78d6", "#eb6834", "#1baf7a")))


# --- inputs ------------------------------------------------------------------------------

def row_records(root, mode, banks=BANKS, model=MODEL):
    """The bank records aligned with confound_sweep.dataset's rows (it drops anchors and
    null prompts the same way), for labels dataset() does not carry: alan_range (old-bank
    records have none; their delays are Alan's by construction) and delay_scale."""
    out = []
    for bank in banks:
        _, labels = load_run(root, mode, bank, model)
        records = run_records(root, f"exp0_{model}_think-{mode}_{bank}")
        uids = labels["sample_uids"][~labels["anchor"]]
        out += [{"bank": bank, "alan_range": records[u].get("alan_range", True),
                 "delay_scale": records[u].get("delay_scale", "old bank")} for u in uids]
    return out


def fit_inputs(data):
    """The pooled fit's log-horizon targets, factors and delay covariates (as
    pristine_confounds builds them from the dataset's features)."""
    V = data["features"]
    duration = np.array([Horizon.parse(d).log10_years for d in data["duration_id"]])
    factors = {"unit": unit_names(V[:, 5]), "regime": data["regime"], "forced": V[:, 8] > 0.5,
               "order": V[:, 4] > 0.5}
    covariates = {"log near delay": V[:, 0], "log far delay": V[:, 1]}
    return duration, factors, covariates


def curve_direction(cell):
    """The curve's main direction [d]: the least-squares slope of its points on log
    horizon, the activation change per decade along the curve's trend."""
    points = cell["curve_coords"] @ cell["basis"]
    logh = cell["curve_logh"] - cell["curve_logh"].mean()
    return logh @ (points - points.mean(0)) / (logh @ logh)


def project(cell, X):
    """Decades along the curve's main direction, per row of X."""
    axis = curve_direction(cell)
    return (X - cell["mean"]) @ axis / (axis @ axis)


# --- readouts ----------------------------------------------------------------------------

def readouts(X, data, is_fill, folds):
    """name -> out-of-fold readout per row (log10 years, before calibration)."""
    duration, factors, covariates = fit_inputs(data)
    y, duration_id = data["y"], data["duration_id"]
    probe, axis = np.empty(len(y)), np.empty(len(y))
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        fill_train = train & is_fill
        probe[test] = fit_probe(X[fill_train], y[fill_train], duration_id[fill_train])(X[test])
        cell = fit("effects", X[train], duration[train], subset(factors, train), subset(covariates, train), K_CURVE)
        axis[test] = project(cell, X[test])
    orthogonal = cross_fit("effects ⟂", X, duration, factors, covariates, folds, K_CURVE)[0]
    return {"fill probe": probe, "effects ⟂": orthogonal, "horizon axis": axis}


# --- the step model ----------------------------------------------------------------------

def spline_terms(x, knots):
    """The nonlinear columns of a restricted cubic spline (Harrell): cubic between the
    knots, linear beyond the outer ones. len(knots) - 2 columns."""
    t = np.asarray(knots, dtype=float)
    k = len(t)

    def cube(v):
        return np.maximum(v, 0.0) ** 3

    columns = [(cube(x - t[j]) - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])) / (t[k - 1] - t[0]) ** 2
               for j in range(k - 2)]
    return np.column_stack(columns)


def step_design(margins, log_delays, spec, controls=None):
    """(V, names) for one spec. `margins` and `log_delays` map "near" / "far" to per-row
    log10(delay / horizon) and log10 delay. Within a duration the linear terms in the two
    are the same column up to a constant, so `spec` only changes the nonlinear part."""
    columns, names = [], []
    for role in ("near", "far"):
        m = margins[role]
        if spec in ("hinge", "spline"):
            columns.append(m)
            names.append(f"{role} slope")
        if spec == "hinge":
            columns.append(np.maximum(m, 0.0))
            names.append(f"{role} slope change above 0")
        if spec == "spline":
            knots = np.quantile(log_delays[role], [0.05, 0.35, 0.65, 0.95])
            spline = spline_terms(log_delays[role], knots)
            columns += list(spline.T)
            names += [f"{role} spline {j + 1}" for j in range(spline.shape[1])]
        columns.append((m < 0).astype(float))
        names.append(f"{role} step")
    for name, values in (controls or {}).items():
        columns.append(values)
        names.append(name)
    return np.column_stack(columns), names


def step_fits(readout, slope, margins, log_delays, controls, groups, bases, rng):
    """spec -> name -> [coefficient, 2.5%, 97.5%] in equivalent decades (readout / slope)."""
    out = {}
    for spec in SPECS:
        V, names = step_design(margins, log_delays, spec, controls)
        fitted = fe_regression(readout, V, names, groups, bases, rng)
        out[spec] = {n: [v / slope for v in c] for n, c in fitted.items()}
    return out


def subset_result(raw, slopes, rows, data, rng):
    """Every readout's step fits and margin bins on the rows in `rows`, in equivalent
    decades (each readout divided by its calibration in `slopes`)."""
    y, V = data["y"][rows], data["features"][rows]
    groups, bases, duration_id = data["durations"][rows], data["pairs"][rows], data["duration_id"][rows]
    margins = {"near": V[:, 0] - y, "far": V[:, 1] - y}
    log_delays = {"near": V[:, 0], "far": V[:, 1]}
    controls = {"near first": V[:, 4], "log unit size": V[:, 5]}
    out = {"n": int(rows.sum()), "bases": int(len(np.unique(bases))), "durations": int(len(np.unique(groups)))}
    for name, readout in raw.items():
        r = readout[rows]
        residual = within(r, groups) / slopes[name]
        out[name] = {"calibration here": calibration(r, y, duration_id),
                     "fits": step_fits(r, slopes[name], margins, log_delays, controls, groups, bases, rng),
                     "bins": {role: binned(residual, margins[role], bases, rng, MARGIN_BINS) for role in margins}}
    return out


# --- Q3: the old fit's misread of the fill -----------------------------------------------

def inside_design(y, V):
    """Log horizon, absolute log delays and the two inside-the-horizon indicators."""
    design = np.column_stack([y, V[:, 0], V[:, 1], V[:, 0] < y, V[:, 1] < y]).astype(float)
    return design, ["log horizon", "log near delay", "log far delay", "near inside", "far inside"]


def fill_misread(X, data, is_fill, records, folds, rng):
    """The old-runs effects ⟂ curve's error (readout - truth, decades), cross-fitted by base:
    each fold read by a fit on the other fold's old prompts. On fill prompts, by regime x
    delay_scale; regressed on inside_design across the fill prompts and across all
    prompts (old prompts span Alan's delays at every horizon, so there the absolute delay
    scale varies at a fixed horizon by up to 6 decades)."""
    duration, factors, covariates = fit_inputs(data)
    readout = np.empty(len(duration))
    grid = None
    for fold in np.unique(folds):
        train, test = (folds != fold) & ~is_fill, folds == fold
        cell = fit("effects ⟂", X[train], duration[train], subset(factors, train), subset(covariates, train),
                   K_CURVE)
        readout[test] = curve_readout(cell, X[test])
        grid = cell["curve_logh"]
    error = readout - data["y"]
    at_end = np.isclose(readout, grid[0]) | np.isclose(readout, grid[-1])
    f = is_fill
    y, V, bases, regime = data["y"][f], data["features"][f], data["pairs"][f], data["regime"][f]
    scale = np.array([r["delay_scale"] for r, keep in zip(records, f) if keep])
    table = {}
    for g in ("none", "near", "both"):
        for s in ("minutes-hours", "days-weeks", "months-years"):
            rows = (regime == g) & (scale == s)
            if rows.sum() >= 4:
                keys = np.unique(bases[rows])
                by_base = {b: error[f][rows][bases[rows] == b] for b in keys}
                boots = [np.concatenate([by_base[b] for b in rng.choice(keys, size=len(keys))]).mean()
                         for _ in range(200)]
                table[f"{g} | {s}"] = [float(error[f][rows].mean()), *np.percentile(boots, [2.5, 97.5]).tolist(),
                                       int(rows.sum()), float(np.median(10 ** V[rows, 0])),
                                       float(np.median(10 ** y[rows]))]
    one_group = np.zeros(len(error))   # no fixed effects: an intercept
    design, names = inside_design(data["y"], data["features"])
    return {"table": table, "mean error fill": float(error[f].mean()),
            "share at curve ends": {"fill": float(at_end[f].mean()), "old": float(at_end[~f].mean())},
            "regression fill": fe_regression(error[f], design[f], names, one_group[f], bases, rng),
            "regression pooled": fe_regression(error, design, names, one_group, data["pairs"], rng)}


# --- output ------------------------------------------------------------------------------

def plot(results, which, path):
    """Rows: cells; columns: near and far margin. The within-duration residual of each
    readout, binned by margin, on `which` prompts."""
    cells = list(results)
    fig, axes = plt.subplots(len(cells), 2, figsize=(9, 2.6 * len(cells)), squeeze=False, sharex="col")
    for row, cell in enumerate(cells):
        for col, role in enumerate(("near", "far")):
            ax = axes[row, col]
            for name in READOUTS:
                pts = np.array(results[cell][which][name]["bins"][role])
                if len(pts):
                    ax.plot(pts[:, 0], pts[:, 1], color=COLORS[name], lw=2, marker="o", ms=4,
                            label=name.replace("⟂", "⊥"))   # DejaVu Sans has no ⟂
                    ax.fill_between(pts[:, 0], pts[:, 2], pts[:, 3], color=COLORS[name], alpha=0.15, lw=0)
            ax.axhline(0, color="#7a7974", lw=0.8)
            ax.axvline(0, color="#3d3d3a", lw=1.0, ls="--")
            ax.tick_params(labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{cell}: {role} delay", fontsize=9)
            if row == len(cells) - 1:
                ax.set_xlabel(f"log10({role} delay / horizon)   (left of 0: inside the horizon)", fontsize=8)
            if col == 0:
                ax.set_ylabel("readout − duration mean\n(equivalent decades)", fontsize=8)
    axes[0, 1].legend(fontsize=8, frameon=False)
    title = {"fill": "fill prompts only (no delay within 0.2 decades of the horizon)",
             "pooled": "old + fill prompts"}[which]
    fig.suptitle(f"Within-duration readout residual against the delay margin, {title}.\n"
                 "A step at the dashed line = 'fits / doesn't fit'; a slope = delay magnitude.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fmt(c):
    star = "*" if c[1] > 0 or c[2] < 0 else " "
    return f"{c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}]{star}"


def summary_lines(results):
    lines = ["Equivalent decades; * = 95% CI (bases resampled) excludes 0. Step = jump at the crossing "
             "(inside minus outside); slope per decade of margin.", ""]
    for cell, by_subset in results.items():
        for which in ("fill", "pooled"):
            m = by_subset[which]
            lines.append(f"{cell}, {which} prompts: n {m['n']}, {m['bases']} bases, {m['durations']} durations")
            for name in READOUTS:
                fits = m[name]["fits"]
                lines.append(f"  {name} (calibration: old prompts {by_subset['calibration on old prompts'][name]:.2f}, "
                             f"these prompts {m[name]['calibration here']:.2f})")
                for spec in SPECS:
                    shown = [k for k in fits[spec] if "spline" not in k and k not in ("near first", "log unit size")]
                    lines.append(f"    {spec:9s} " + " | ".join(f"{k} {fmt(fits[spec][k])}" for k in shown))
        q3 = by_subset.get("fill misread")
        if q3:
            lines.append(f"  Q3, old-runs effects ⟂ curve (cross-fitted) on fill prompts: mean error "
                         f"{q3['mean error fill']:+.2f} decades; readouts at a curve end: "
                         f"fill {q3['share at curve ends']['fill']:.0%}, old {q3['share at curve ends']['old']:.0%}")
            for key, (mean, lo, hi, n, near, h) in q3["table"].items():
                lines.append(f"    {key:28s} {mean:+.2f} [{lo:+.2f}, {hi:+.2f}] n {n:3d}  "
                             f"median near delay {near:.2g} y, horizon {h:.2g} y")
            for which in ("fill", "pooled"):
                lines.append(f"    error regression, {which}: "
                             + " | ".join(f"{k} {fmt(c)}" for k, c in q3[f"regression {which}"].items()))
        lines.append("")
    return lines


def main(root, cells):
    root = Path(root)
    rng = np.random.default_rng(0)
    out = root / "analysis"
    results, loaded = {}, {}
    for mode, position, layer in tqdm(cells, desc="cells"):
        if mode not in loaded:
            loaded.clear()
            parts, data = dataset(root, MODEL, mode, BANKS)
            loaded[mode] = parts, data, row_records(root, mode)
        parts, data, records = loaded[mode]
        is_fill = np.array([r["bank"] == "fill" for r in records])
        X = cell_matrix(parts, position, layer)
        folds = base_folds(data["pairs"])
        raw = readouts(X, data, is_fill, folds)
        # equivalent decades: per true decade on the old prompts, whose delays do not move
        # with the horizon (on the fill they do, so a fill calibration mixes in the delays)
        old = ~is_fill
        slopes = {k: calibration(r[old], data["y"][old], data["duration_id"][old]) for k, r in raw.items()}
        name = f"{mode} {position} L{layer}"
        results[name] = {"calibration on old prompts": slopes,
                         "fill": subset_result(raw, slopes, is_fill, data, rng),
                         "pooled": subset_result(raw, slopes, np.ones(len(is_fill), bool), data, rng),
                         "fill misread": fill_misread(X, data, is_fill, records, folds, rng)}
        (out / "delay_step.json").write_text(json.dumps(results, indent=1))   # partial results survive
    text = "\n".join(summary_lines(results))
    (out / "delay_step.txt").write_text(text)
    for which in ("fill", "pooled"):
        plot(results, which, out / f"delay_step_{which}.png")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1], CELLS + (PRE_LABEL_CELLS if "--pre-label" in sys.argv[2:] else []))
