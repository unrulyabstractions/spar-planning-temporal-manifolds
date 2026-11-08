"""Can the horizon manifold be fitted free of the unit word? (16_spec_pristine_manifold.md §3)

On the 8B units bank (every base prompt in 2-3 renderings of one duration), three
curves per cell, all through `steering.fit_duration_effects`:
  old curve       per-duration averages (no factors), as E8's curve
  effects         duration effects with unit, regime, forced and option order as factors
  effects ⟂       the same, with the factor effects' span projected out of the basis
Readout: nearest point on the curve (`steering.curve_readout`). Error of a readout r
against the true log duration t, in percent: 10^|r - t| - 1 (2x or 1/2x is 100%).

  P1 no unit left   cross-fitted by base prompt (both renderings on one side): per pair,
                    readout gap minus true gap. Judged on the systematic gap (16 §3, after
                    the prototype): per rendering the mean over its bases, debiased for
                    trace noise, root mean square over renderings; <= 20% and <= half the
                    old curve's. The per-pair median alongside, split by whether the two
                    traces reason in the same main unit (written_unit.json, side view).
  P2 magnitude      leave-one-duration-out readout against the true duration, interior
                    durations (nearest-point readout cannot extrapolate past the ends).
                    Judged on rendering averages: within-unit R² >= 0.5. Per prompt
                    alongside: median error and a unit lookup's (training mean of the
                    same unit).
  P3 forced         AUC of forced vs natural from the k coordinates (thinking on); <= 0.60
  P4 regime         AUC of each regime pair from the coordinates centered within duration,
                    the largest of the three; <= 0.60
k = 2, 4, 8; judged at k = 4. CIs resample base prompts (P1 systematic: durations).

    python -m pipeline.s1_data_vectors.analysis.pristine_manifold /home/Jared/SPAR/results/horizon [pooled]
    -> <root>/analysis/pristine_manifold{,_pooled,_pooled_fill}.{json,txt,png}
The bank defaults to the units bank (the prototype). "pooled" (16 §4) fits the old full
and units runs with the fill run, adding the log near and far delays as covariates;
the fill run's anchors (re-runs of old prompts) are left out of the fit and checked
against the old runs first, and only the cells whose anchors match are fitted
(`gated_jobs`). Without a fill run it pools the old runs only (_pooled).

Reads saved activations only; runs no model.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm

from pipeline.s1_data_vectors.analysis.off_curve_content import center_within
from pipeline.s1_data_vectors.analysis.probe_dimensionality import positions
from pipeline.s3_analysis.by_datatype.unit_overlap import MODEL, load_run, pair_shifts
from pipeline.s3_analysis.by_datatype.written_unit import run_records
from spar_horizon.full_bank import NATURAL_UNITS
from spar_horizon.horizons import Horizon
from spar_horizon.probes import within_unit_r2
from spar_horizon.runs import REGIONS
from spar_horizon.steering import curve_readout, fit_duration_effects

LAYERS = (16, 22, 24, 27)
POSITIONS = {"off": ("suffix newline", "pre-label"), "on": ("suffix newline", "think close", "pre-label")}
KS = (2, 4, 8)
JUDGED_K = 4
METHODS = ("old curve", "effects", "effects ⟂")
BOOT = 1000
UNIT_ORDER = tuple(reversed(NATURAL_UNITS))   # seconds .. years
# categorical slots 1-7 of the dataviz reference palette, in fixed order (validated, light)
UNIT_COLORS = dict(zip(UNIT_ORDER, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")))


def percent(decades):
    """Symmetric percent error of a log10 error: 10^|e| - 1."""
    return 100 * (10 ** np.abs(decades) - 1)


def fit(method, X, duration, factors, covariates, k):
    if method == "old curve":
        return fit_duration_effects(X, duration, {}, k_max=k, orthogonal=False)
    return fit_duration_effects(X, duration, factors, k_max=k, orthogonal=method == "effects ⟂",
                                covariates=covariates)


def subset(columns, rows):
    return {name: values[rows] for name, values in columns.items()}


def cross_fit(method, X, duration, factors, covariates, folds, k):
    """Out-of-fold readout and coordinates for every row, fitting on the other fold."""
    readout, coords = np.empty(len(X)), np.empty((len(X), k))
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        cell = fit(method, X[train], duration[train], subset(factors, train), subset(covariates, train), k)
        readout[test] = curve_readout(cell, X[test])
        coords[test] = (X[test] - cell["mean"]) @ cell["basis"].T
    return readout, coords


def leave_one_duration_out(method, X, duration, factors, covariates, k):
    """Readout of each interior duration's rows from a fit on all other durations."""
    readout = np.full(len(X), np.nan)
    grid = np.unique(duration)
    for d in grid[1:-1]:
        train, test = duration != d, duration == d
        cell = fit(method, X[train], duration[train], subset(factors, train), subset(covariates, train), k)
        readout[test] = curve_readout(cell, X[test])
    return readout


def unit_lookup(y, duration, units):
    """Leave-one-duration-out: each row's log duration predicted by the mean of the
    other durations' rows that share its unit."""
    pred = np.full(len(y), np.nan)
    for d in np.unique(duration):
        test = duration == d
        for unit in np.unique(units[test]):
            train = ~test & (units == unit)
            if train.any():
                pred[test & (units == unit)] = y[train].mean()
    return pred


def auc(coords, target):
    """Cross-validated AUC of a logistic regression from the coordinates, folded to >= 0.5."""
    if len(np.unique(target)) < 2 or min(np.bincount(target.astype(int))) < 10:
        return float("nan")
    score = cross_val_predict(LogisticRegression(max_iter=2000), coords, target, cv=5, method="decision_function")
    value = roc_auc_score(target, score)
    return float(max(value, 1 - value))


def regime_auc(coords, regime, duration):
    """Largest AUC over regime pairs, from coordinates centered within each duration."""
    centered = center_within(coords, duration)
    values = []
    for a, b in (("none", "near"), ("near", "both"), ("none", "both")):
        rows = np.isin(regime, (a, b))
        values.append(auc(centered[rows], regime[rows] == b))
    return float(np.nanmax(values)) if not np.all(np.isnan(values)) else float("nan")


def median_ci(values, groups, rng):
    """Median and 95% CI resampling groups (base prompts)."""
    keys = np.unique(groups)
    by_group = [values[groups == g] for g in keys]
    boots = [np.median(np.concatenate([by_group[i] for i in rng.integers(len(keys), size=len(keys))]))
             for _ in range(BOOT)]
    return [float(np.median(values)), *np.percentile(boots, [2.5, 97.5]).tolist()]


def systematic_shift(shift, rows, labels, rng):
    """The unit's systematic shift: the root mean square, over renderings (duration,
    unit) with at least two bases, of each rendering's mean pair shift, as a percent.

    Each rendering's squared mean is reduced by its sampling variance, var / n, which
    makes it unbiased for the squared systematic shift. The raw |mean| overstates the
    shift by the reasoning noise between traces: with 4 bases and no systematic shift,
    it is half the per-pair spread. Debiasing per rendering and then taking a median
    does not work, because most renderings clip at zero.

    The CI resamples durations, whose renderings share base prompts. Resampling bases
    instead repeats a base within a rendering, which shrinks its variance and leaves
    the correction short."""
    durations, units = labels["durations"][rows], labels["units"][rows]
    squared, cluster = [], []
    for duration, unit in sorted(set(zip(durations, units))):
        s = shift[(durations == duration) & (units == unit)]
        if len(s) >= 2:
            squared.append(s.mean() ** 2 - s.var(ddof=1) / len(s))
            cluster.append(duration)
    squared, cluster = np.array(squared), np.array(cluster)

    def size(mean_square):
        return float(percent(np.sqrt(max(mean_square, 0.0))))

    members = [squared[cluster == c] for c in np.unique(cluster)]
    boots = [np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))]).mean()
             for _ in range(BOOT)]
    return [size(squared.mean()), *(size(b) for b in np.percentile(boots, [2.5, 97.5]))]


def rendering_average_r2(y, readout, durations, units):
    """Within-unit R² of readouts averaged per rendering (duration, unit): whether the
    manifold orders durations within a unit once the per-prompt reasoning noise
    averages out. A readout that only knows the unit scores about -0.25 on the units
    bank."""
    keys = sorted(set(zip(durations, units)))
    rows = [(durations == d) & (units == u) for d, u in keys]
    return within_unit_r2(np.array([y[r].mean() for r in rows]), np.array([readout[r].mean() for r in rows]),
                          np.array([u for _, u in keys]))


def fit_columns(labels, pooled):
    """The fit's factors and covariates (16 §2, §4) from a run's labels."""
    factors = {"unit": labels["units"], "regime": labels["regime"], "forced": labels["forced"],
               "order": labels["near_first"]}
    # pooled banks: the fill bank's delays track the horizon, the old banks' do not
    covariates = {"log near delay": labels["log_near"], "log far delay": labels["log_far"]} if pooled else {}
    return factors, covariates


def cell_measures(X, labels, folds, same_main, rng, pooled=False):
    y, duration, units = labels["y"], labels["duration"], labels["units"]
    factors, covariates = fit_columns(labels, pooled)
    lookup = unit_lookup(y, duration, units)
    interior = (duration > duration.min()) & (duration < duration.max())
    # a unit seen at no other duration has no lookup; those rows are left out of its median
    out = {"unit lookup P2": float(np.nanmedian(percent(lookup[interior] - y[interior])))}
    for k in KS:
        for method in METHODS:
            readout, coords = cross_fit(method, X, duration, factors, covariates, folds, k)
            others, shift = pair_shifts(readout, y, labels["pairs"], labels["natural"])
            p1 = percent(shift)
            lodo = leave_one_duration_out(method, X, duration, factors, covariates, k)
            p2 = percent(lodo[interior] - y[interior])
            row = {"P1": median_ci(p1, labels["pairs"][others], rng),
                   "P1 systematic": systematic_shift(shift, others, labels, rng),
                   "P1 p90": float(np.percentile(p1, 90)),
                   "P1 signed": float(100 * (10 ** np.mean(shift) - 1)),
                   "P2": float(np.median(p2)), "P2 p90": float(np.percentile(p2, 90)),
                   "within-unit R²": within_unit_r2(y[interior], lodo[interior], units[interior]),
                   "within-unit R² avg": rendering_average_r2(y[interior], lodo[interior],
                                                              labels["durations"][interior], units[interior]),
                   "P3": auc(coords, labels["forced"]),
                   "P4": regime_auc(coords, labels["regime"], duration)}
            if same_main is not None:
                split = same_main[others]
                row["P1 same written unit"] = float(np.median(p1[split])) if split.any() else float("nan")
                row["P1 different written unit"] = float(np.median(p1[~split])) if (~split).any() else float("nan")
            if k == JUDGED_K:
                row["readout"], row["lodo"] = readout.tolist(), lodo.tolist()
            out[f"{method}|{k}"] = row
    return out


def verdicts(measures):
    old = measures[f"old curve|{JUDGED_K}"]
    out = {}
    for method in METHODS[1:]:
        m = measures[f"{method}|{JUDGED_K}"]
        out[method] = {"P1": m["P1 systematic"][0] <= 20 and m["P1 systematic"][0] <= 0.5 * old["P1 systematic"][0],
                       "P2": m["within-unit R² avg"] >= 0.5,
                       "P3": None if np.isnan(m["P3"]) else m["P3"] <= 0.60,
                       "P4": m["P4"] <= 0.60}
    return out


POOLED_BANKS = ("full", "units", "fill")   # the old 8B runs and the fill bank (16 §4)
# largest median anchor gap, as a fraction of the distance between two distinct old
# prompts, at which a cell is pooled; set after the first fill run (19 §2), where the
# gaps were 0 (exact) or above 0.4, nothing in between
ANCHOR_TOLERANCE = 0.1


def mode_labels(root, mode, bank, model=MODEL):
    """One run's activation file and labels, with the fit's columns added."""
    z, labels = load_run(root, mode, bank, model)
    records = run_records(root, f"exp0_{model}_think-{mode}_{bank}")
    labels["duration"] = np.array([Horizon.parse(d).log10_years for d in labels["durations"]])
    labels["near_first"] = np.array([records[uid]["short_first"] for uid in labels["sample_uids"]])
    labels["log_near"] = np.log10([records[uid]["near_delay_years"] for uid in labels["sample_uids"]])
    labels["log_far"] = np.log10([records[uid]["far_delay_years"] for uid in labels["sample_uids"]])
    labels["bank"] = np.full(len(labels["y"]), bank)
    return z, labels


def load_parts(root, mode, bank, model=MODEL):
    """[(z, labels)] for `bank`, or for each pooled run that exists when bank is "pooled".
    `model` is the run-folder name ("Qwen3-8B")."""
    if bank != "pooled":
        return [mode_labels(root, mode, bank, model)]
    return [mode_labels(root, mode, b, model) for b in POOLED_BANKS
            if (root / f"exp0_{model}_think-{mode}_{b}").exists()]


def pooled_labels(parts):
    """The parts' labels concatenated, without the anchors (re-runs of old prompts)."""
    keys = [k for k in parts[0][1] if k != "rows"]
    labels = {k: np.concatenate([lab[k] for _, lab in parts]) for k in keys}
    return {k: v[~labels["anchor"]] for k, v in labels.items()}


def cell_matrix(parts, position, layer):
    """The cell's activations, stacked over the parts in pooled_labels' order."""
    blocks = []
    for z, labels in parts:
        meta = {"regions": {r: tuple(int(x) for x in z[f"region_{r}"]) for r in REGIONS},
                "tokens": [str(t) for t in z["tokens"]]}
        X = z[f"L{layer}"][labels["rows"], positions(meta)[position]].astype(np.float64)
        blocks.append(X[~labels["anchor"]])
    return np.concatenate(blocks)


def anchor_check(root, mode, parts, cells):
    """Whether the fill run reproduces the old runs on the re-run anchors: the share of
    anchors with the same reasoning (thinking on) and label, and per cell the median and
    largest distance between an anchor's two activations, as a fraction of the median
    distance between distinct old prompts (16 §4: pool only if negligible)."""
    by_bank = {labels["bank"][0]: (z, labels) for z, labels in parts}
    if "fill" not in by_bank:
        return None
    answers = {}
    for bank in POOLED_BANKS:
        path = root / f"exp0_{MODEL}_think-{mode}_{bank}" / "answers.json"
        for r in json.loads(path.read_text()):
            answers.setdefault(r["sample_uid"], {})[bank] = r
    fill_labels = by_bank["fill"][1]
    anchors = fill_labels["sample_uids"][fill_labels["anchor"]]
    old_bank = {uid: next(b for b in ("full", "units") if b in answers[uid]) for uid in anchors}
    same = [answers[uid]["fill"].get("think_text") == answers[uid][old_bank[uid]].get("think_text")
            and answers[uid]["fill"].get("label") == answers[uid][old_bank[uid]].get("label") for uid in anchors]
    out = {"n": int(len(anchors)), "same reasoning and label": float(np.mean(same)), "cells": {}}
    for position, layer in cells:
        X = {}
        for bank, (z, labels) in by_bank.items():
            meta = {"regions": {r: tuple(int(x) for x in z[f"region_{r}"]) for r in REGIONS},
                    "tokens": [str(t) for t in z["tokens"]]}
            X[bank] = (dict(zip(labels["sample_uids"], labels["rows"])), z[f"L{layer}"], positions(meta)[position])

        def vector(bank, uid):
            index, array, column = X[bank]
            return array[index[uid], column].astype(np.float64)

        gaps = np.array([np.linalg.norm(vector("fill", u) - vector(old_bank[u], u)) for u in anchors])
        index, array, column = X["units"]
        old = array[np.random.default_rng(0).choice(list(index.values()), 200, replace=False), column].astype(np.float64)
        spread = np.median(np.linalg.norm(old[:100] - old[100:], axis=1))
        out["cells"][f"{position} L{layer}"] = {"median": float(np.median(gaps) / spread),
                                               "max": float(gaps.max() / spread)}
    return out


def gated_jobs(jobs, anchors):
    """The (mode, position, layer) cells that may be pooled: all of them without a fill
    run, else every cell of a mode whose anchors match the old runs before the trace
    (median gap <= ANCHOR_TOLERANCE at each suffix cell). On the first fill run (19 §2)
    thinking-on traces did not reproduce under a different batch, so the cells after the
    trace differ by as much as two distinct prompts and cannot test the run setup; the
    suffix, before the trace, differs only by bf16 rounding. Jared pooled the trace
    cells on that basis (2026-10-30)."""
    if anchors is None:
        return list(jobs)

    def setup_matches(mode):
        cells = anchors[mode] and anchors[mode]["cells"]
        return not cells or all(gap["median"] <= ANCHOR_TOLERANCE
                                for cell, gap in cells.items() if cell.startswith("suffix newline"))

    return [(mode, position, layer) for mode, position, layer in jobs if setup_matches(mode)]


def written_same_main(root, labels):
    """Per row: whether both renderings of its pair reason in the same main unit (or None)."""
    path = root / "analysis" / "written_unit.json"
    if not path.exists():
        return None
    main = {r["sample_uid"]: r["main"] for r in json.loads(path.read_text())["rows"]}
    by_pair = {}
    for uid, pair in zip(labels["sample_uids"], labels["pairs"]):
        by_pair.setdefault(pair, set()).add(main.get(uid))
    return np.array([len(by_pair[p]) == 1 for p in labels["pairs"]])


def summary_text(results, bank):
    lines = [f"Pristine manifold on the 8B {bank} bank (16 §3). Percent errors [95% CI]; judged at k = {JUDGED_K}.",
             "P1 judged on sys, the systematic pair gap (pass <= 20% and <= half the old curve's; pair = per-pair median); "
             "P2 judged on wuR² avg, leave-one-duration-out readouts averaged per rendering (pass >= 0.5; "
             "per-prompt median error and wuR² alongside); P3 forced AUC, P4 regime AUC (pass <= 0.60).", ""]
    for mode, cells in results.items():
        for key, m in cells.items():
            lines.append(f"thinking {mode}, {key}   unit lookup P2 {m['unit lookup P2']:.0f}%")
            for k in KS:
                for method in METHODS:
                    r = m[f"{method}|{k}"]
                    written = (f"  same/diff written unit {r['P1 same written unit']:.0f}%/"
                               f"{r['P1 different written unit']:.0f}%") if "P1 same written unit" in r else ""
                    lines.append(f"  k={k} {method:10s} P1 sys {r['P1 systematic'][0]:4.0f}% "
                                 f"[{r['P1 systematic'][1]:.0f}, {r['P1 systematic'][2]:.0f}] "
                                 f"pair {r['P1'][0]:5.0f}% [{r['P1'][1]:.0f}, {r['P1'][2]:.0f}] "
                                 f"p90 {r['P1 p90']:4.0f}% signed {r['P1 signed']:+4.0f}% | "
                                 f"P2 wuR² avg {r['within-unit R² avg']:+.2f} per prompt {r['P2']:4.0f}% "
                                 f"wuR² {r['within-unit R²']:+.2f} | P3 {r['P3']:.2f} P4 {r['P4']:.2f}{written}")
            lines.append(f"  verdicts k={JUDGED_K}: {m['verdicts']}")
            lines.append("")
    return "\n".join(lines)


def plot(results, labels_by_mode, path):
    """Per cell (rows) and method (columns): cross-fitted readout against the true log
    duration, colored by unit; the two renderings of a pair joined by a gray line."""
    cells = [(mode, key) for mode, by_key in results.items() for key in by_key]
    fig, axes = plt.subplots(len(cells), len(METHODS), figsize=(3.2 * len(METHODS), 2.8 * len(cells)),
                             sharex=True, sharey=True, squeeze=False)
    for row, (mode, key) in enumerate(cells):
        labels = labels_by_mode[mode]
        y, units, pairs = labels["y"], labels["units"], labels["pairs"]
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            m = results[mode][key][f"{method}|{JUDGED_K}"]
            readout = np.array(m["readout"])
            for p in np.unique(pairs):
                rows = np.flatnonzero(pairs == p)
                ax.plot(y[rows], readout[rows], color="#c3c2b7", lw=0.6, zorder=1)
            for unit in UNIT_ORDER:
                rows = units == unit
                ax.scatter(y[rows], readout[rows], s=10, color=UNIT_COLORS[unit], label=unit, zorder=2, linewidths=0)
            lim = [y.min() - 0.3, y.max() + 0.3]
            ax.plot(lim, lim, color="#7a7974", lw=0.8, ls="--", zorder=0)
            ax.set_title(f"{mode} {key}\n{method}: P1 sys {m['P1 systematic'][0]:.0f}%, "
                         f"wuR² avg {m['within-unit R² avg']:+.2f}", fontsize=8)
            ax.tick_params(labelsize=7)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == len(cells) - 1:
                ax.set_xlabel("true log10 years", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"readout, k={JUDGED_K}", fontsize=8)
    axes[0, -1].legend(fontsize=7, frameon=False, loc="upper left", markerscale=1.5)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(root, bank="units"):
    root = Path(root)
    rng = np.random.default_rng(0)
    results, labels_by_mode = {}, {}
    jobs = [(mode, position, layer) for mode in ("off", "on") for position in POSITIONS[mode] for layer in LAYERS]
    parts = {mode: load_parts(root, mode, bank) for mode in ("off", "on")}
    loaded = {mode: pooled_labels(parts[mode]) for mode in ("off", "on")}
    pooled, anchors, with_fill = bank == "pooled", None, False
    if pooled:
        found = {mode: sorted({lab["bank"][0] for _, lab in parts[mode]}) for mode in parts}
        with_fill = all("fill" in f for f in found.values())
        print(f"pooled runs: {found}" + ("" if with_fill else " (no fill run: the old runs only)"))
        anchors = {mode: anchor_check(root, mode, parts[mode], [(p, l) for m, p, l in jobs if m == mode])
                   for mode in ("off", "on")}
        print(f"anchor check: {json.dumps(anchors)}")
        skipped = [job for job in jobs if job not in gated_jobs(jobs, anchors)]
        if skipped:
            print(f"not pooled, anchors differ from the old runs: {skipped}")
        jobs = gated_jobs(jobs, anchors)
    for mode, position, layer in tqdm(jobs, desc="cells"):
        labels = loaded[mode]
        labels_by_mode[mode] = labels
        # two folds of base prompts: both renderings of a pair on one side
        pair_ids = np.unique(labels["pairs"])
        fold_of = dict(zip(pair_ids, np.random.default_rng(0).permutation(len(pair_ids)) % 2))
        folds = np.array([fold_of[p] for p in labels["pairs"]])
        X = cell_matrix(parts[mode], position, layer)
        # the written-unit side view exists for the units bank only (pipeline/s3_analysis/by_datatype/written_unit.py)
        same_main = written_same_main(root, labels) if mode == "on" and bank == "units" else None
        measures = cell_measures(X, labels, folds, same_main, rng, pooled)
        measures["verdicts"] = verdicts(measures)
        results.setdefault(mode, {})[f"{position} L{layer}"] = measures
    out = root / "analysis"
    # the old-runs-only pooled baseline (17 §8) keeps its files
    name = "pristine_manifold" if bank == "units" else f"pristine_manifold_{bank}" + ("_fill" if with_fill else "")
    (out / f"{name}.json").write_text(json.dumps({**results, "anchor check": anchors}, indent=1))
    text = summary_text(results, bank)
    if anchors is not None:
        text = f"anchor check (16 §4): {json.dumps(anchors)}\n\n" + text
    (out / f"{name}.txt").write_text(text)
    plot(results, labels_by_mode, out / f"{name}.png")
    print(text)
    print(f"wrote {out}/{name}.{{json,txt,png}}")


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:3])
