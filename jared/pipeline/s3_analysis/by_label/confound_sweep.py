"""What else moves the horizon code within a duration? (overnight sweep, 2026-10-29)

Saved activations only. For each model, bank and cell, two measures of every
prompt feature other than the horizon, both within a duration (duration fixed
effects, so nothing that only varies with the horizon can score):

  readout   a cross-fitted horizon probe (2 folds of base prompts; ridge on the
            top-8 PCs of the training fold's duration averages, as unit_overlap's
            probe). Its residual from the duration mean is regressed on all the
            features at once. A coefficient is in decades of readout per unit of the
            feature; divided by the probe's calibration slope (readout per true decade
            across durations) it is in decades of horizon ("equivalent decades").
  axis      the features' within-duration effects on the activations themselves
            (multi-output least squares), projected on the horizon axis (the slope of
            the duration averages on log horizon): equivalent decades per unit of the
            feature, and the cosine between the effect and the axis.
The two differ by design. A probe can subtract a feature that has a direction of its
own (the ridge learns to cancel it), so a clean readout does not mean the code is
clean; a feature whose effect lies along the horizon axis cannot be cancelled and
moves both.

Features (each kept only where it varies within a duration):
  log near / far delay   log10 years; the full and units banks draw them independently
                         of the horizon
  log reward             log10 of the near reward
  log ratio              log10 of far reward / near reward
  near first             the near option listed first
  log unit size          log10 years per rendered unit (units bank: at a fixed
                         duration a bigger unit is a smaller number)
  regime near / both     against none
  forced                 the think block was force-closed (thinking on)
  log think length       log10 tokens in the think block (thinking on)
The decision (chose near) is post-treatment, so it is reported in a separate fit that
adds it to the features.

Groupings: every fit is repeated within each regime and each third of the durations,
for the features that still vary there.

CIs resample base prompts (pairs) on the within-duration data, 200 times.

    python -m pipeline.s3_analysis.by_label.confound_sweep /home/Jared/SPAR/results/horizon
    -> <root>/analysis/confound_sweep.{json,txt}
"""

import json
import sys
import zlib
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from pipeline.s1_data_vectors.analysis.probe_dimensionality import positions
from pipeline.s3_analysis.by_datatype.unit_overlap import load_run
from spar_horizon.e0_runs import read_answers, run_folder, run_records
from spar_horizon.horizons import UNIT_YEARS, Horizon
from spar_horizon.probes import fit_ridge
from spar_horizon.runs import REGIONS

K = 8
BOOT = 200
CELLS = {
    "Qwen3-8B": {"off": [("suffix newline", 16), ("suffix newline", 24), ("pre-label", 24)],
                 "on": [("suffix newline", 16), ("suffix newline", 24), ("think close", 24),
                        ("pre-label", 22), ("pre-label", 24)]},
    # 14B hands off one or two layers later than 8B (11_results_14b.md); L20 is its best suffix cell
    "Qwen3-14B": {"off": [("suffix newline", 20), ("suffix newline", 27), ("pre-label", 33)],
                  "on": [("suffix newline", 20), ("suffix newline", 27), ("think close", 33), ("pre-label", 33)]},
}
DATASETS = [("Qwen3-8B", ("full",)), ("Qwen3-8B", ("units",)), ("Qwen3-8B", ("full", "units")),
            ("Qwen3-14B", ("full",))]
FEATURES = ("log near delay", "log far delay", "log reward", "log ratio", "near first", "log unit size",
            "regime near", "regime both", "forced", "log think length")


# --- the statistics --------------------------------------------------------------------

def base_folds(bases):
    """Two folds of base prompts, the same in every run (Python's own string hash is
    randomized per process)."""
    return np.array([zlib.crc32(str(b).encode()) % 2 for b in bases])


def within(values, groups):
    """`values` (rows, or rows x columns) minus the mean of their group."""
    values = np.asarray(values, dtype=float)
    out = values.copy()
    for g in np.unique(groups):
        rows = groups == g
        out[rows] = values[rows] - values[rows].mean(0)
    return out


def varying(V, names, min_sd=1e-6):
    """The columns of a within-group feature matrix that still vary."""
    keep = [j for j in range(V.shape[1]) if V[:, j].std() > min_sd]
    return V[:, keep], [names[j] for j in keep]


def fe_coefficients(r, V):
    """Least-squares coefficients of the within-group target on the within-group features."""
    return np.linalg.lstsq(V, r, rcond=None)[0]


def fe_regression(r, V, names, groups, bases, rng):
    """name -> [coefficient, 2.5%, 97.5%] for the within-group regression of `r` on the
    columns of V, resampling bases. Columns that do not vary within groups are dropped."""
    r_w = within(r, groups)
    V_w, names = varying(within(V, groups), names)
    if not names:
        return {}
    coef = fe_coefficients(r_w, V_w)
    keys = np.unique(bases)
    rows_of = {b: np.flatnonzero(bases == b) for b in keys}
    boots = []
    for _ in range(BOOT):
        take = np.concatenate([rows_of[b] for b in rng.choice(keys, size=len(keys))])
        boots.append(fe_coefficients(r_w[take], V_w[take]))
    low, high = np.percentile(boots, [2.5, 97.5], axis=0)
    return {n: [float(c), float(lo), float(hi)] for n, c, lo, hi in zip(names, coef, low, high)}


def horizon_axis(X, y, durations):
    """The slope of the duration averages on log horizon [d]: the activation change per
    decade of horizon."""
    keys = np.unique(durations)
    averages = np.stack([X[durations == g].mean(0) for g in keys])
    logh = np.array([y[durations == g].mean() for g in keys])
    logh = logh - logh.mean()
    return (logh @ (averages - averages.mean(0))) / (logh @ logh)


def axis_effects(X, V, names, groups, axis):
    """name -> (equivalent decades along the horizon axis per unit of the feature, cosine
    of the feature's within-group activation effect with the axis)."""
    V_w, names = varying(within(V, groups), names)
    if not names:
        return {}
    B = np.linalg.lstsq(V_w, within(X, groups), rcond=None)[0]   # [features, d]
    along = B @ axis / (axis @ axis)
    cosine = B @ axis / (np.linalg.norm(B, axis=1) * np.linalg.norm(axis))
    return {n: [float(a), float(c)] for n, a, c in zip(names, along, cosine)}


def fit_probe(X, y, durations, k=K):
    """A log-horizon readout X -> y: a ridge on the top-k PCs of the duration averages,
    its penalty chosen holding out whole durations."""
    mean = X.mean(0)
    averages = np.stack([X[durations == g].mean(0) for g in np.unique(durations)]) - mean
    basis = PCA(n_components=min(k, len(averages) - 1)).fit(averages).components_
    model = fit_ridge((X - mean) @ basis.T, y, durations)
    return lambda X_new: model.predict((X_new - mean) @ basis.T)


def crossfit_readout(X, y, durations, folds, k=K):
    """Out-of-fold log-horizon readout: each fold read by a probe fitted on the other."""
    pred = np.empty(len(y))
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        pred[test] = fit_probe(X[train], y[train], durations[train], k)(X[test])
    return pred


def calibration(readout, y, durations):
    """Readout decades per true decade, across duration means."""
    keys = np.unique(durations)
    means = np.array([readout[durations == g].mean() for g in keys])
    truth = np.array([y[durations == g].mean() for g in keys])
    truth = truth - truth.mean()
    return float(truth @ (means - means.mean()) / (truth @ truth))


# --- the data --------------------------------------------------------------------------

def dataset(root, model, mode, banks):
    """Per-row labels and features for the runs of `banks`, concatenated, with the
    activation files and their row indices to read cells from."""
    parts, rows = [], []
    for bank in banks:
        z, labels = load_run(root, mode, bank, model)
        labels = {k: v[~labels["anchor"]] for k, v in labels.items()}   # the fill run's re-runs of old prompts
        run = run_folder(root, model, mode, bank)
        records = run_records(root, run.name)
        answers = {a["sample_uid"]: a for a in read_answers(run)}
        recs = [records[uid] for uid in labels["sample_uids"]]
        think = np.array([answers[uid].get("think_len", 0) for uid in labels["sample_uids"]], dtype=float)
        regime = labels["regime"]
        features = np.column_stack([
            np.log10([r["near_delay_years"] for r in recs]), np.log10([r["far_delay_years"] for r in recs]),
            np.log10([r["rewards"][0] for r in recs]), np.log10([r["rewards"][1] / r["rewards"][0] for r in recs]),
            [r["short_first"] for r in recs], np.log10([UNIT_YEARS[u] for u in labels["units"]]),
            regime == "near", regime == "both", labels["forced"],
            np.log10(np.maximum(think, 1)) if mode == "on" else np.zeros(len(recs))]).astype(float)
        parts.append((z, labels["rows"]))
        rows.append({"y": labels["y"], "durations": np.array([f"{bank}|{d}" for d in labels["durations"]]),
                     "duration_id": labels["durations"], "pairs": np.array([f"{bank}|{p}" for p in labels["pairs"]]),
                     "regime": regime, "chose_near": labels["chose_near"].astype(float), "features": features})
    data = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    return parts, data


def cell_matrix(parts, position, layer):
    blocks = []
    for z, rows in parts:
        meta = {"regions": {r: tuple(int(x) for x in z[f"region_{r}"]) for r in REGIONS},
                "tokens": [str(t) for t in z["tokens"]]}
        blocks.append(z[f"L{layer}"][rows, positions(meta)[position]].astype(np.float64))
    return np.concatenate(blocks)


def thirds(y):
    """Which third of the duration range (in log horizon) each row falls in."""
    cuts = np.quantile(np.unique(y), [1 / 3, 2 / 3])
    return np.digitize(y, cuts)


def cell_result(X, data, rng):
    y, groups, bases = data["y"], data["durations"], data["pairs"]
    # "full|3 years" and "units|3 years" are the same duration for the probe and the axis
    duration = data["duration_id"]
    folds = base_folds(bases)
    readout = crossfit_readout(X, y, duration, folds)
    slope = calibration(readout, y, duration)
    axis = horizon_axis(X, y, duration)
    V, names = data["features"], list(FEATURES)
    out = {"calibration": slope, "readout r2": float(1 - ((readout - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
           "readout": fe_regression(readout, V, names, groups, bases, rng),
           "axis": axis_effects(X, V, names, groups, axis),
           "with choice": fe_regression(readout, np.column_stack([V, data["chose_near"]]), names + ["chose near"],
                                        groups, bases, rng).get("chose near"),
           "groups": {}}
    for label, mask in [*((f"regime {g}", data["regime"] == g) for g in ("none", "near", "both")),
                        *((f"third {t + 1}", thirds(y) == t) for t in range(3))]:
        if mask.sum() >= 40:
            out["groups"][label] = {"n": int(mask.sum()),
                                    "readout": fe_regression(readout[mask], V[mask], names, groups[mask],
                                                             bases[mask], rng)}
    return out


def summary_lines(results):
    lines = []
    for name, cells in results.items():
        lines.append(f"== {name}")
        for cell, m in cells.items():
            lines.append(f"  {cell}: probe R² {m['readout r2']:.2f}, calibration {m['calibration']:.2f} "
                         f"readout dec per true dec")
            for feature, (c, lo, hi) in m["readout"].items():
                eq = c / m["calibration"]
                along, cos = m["axis"].get(feature, (float("nan"), float("nan")))
                star = "*" if lo > 0 or hi < 0 else " "
                lines.append(f"    {feature:17s} readout {c:+.3f} [{lo:+.3f}, {hi:+.3f}]{star} = {eq:+.3f} eq dec | "
                             f"axis {along:+.3f} eq dec, cos {cos:+.2f}")
            if m["with choice"]:
                c, lo, hi = m["with choice"]
                lines.append(f"    {'chose near':17s} readout {c:+.3f} [{lo:+.3f}, {hi:+.3f}] (separate fit)")
    return lines


def main(root):
    root = Path(root)
    rng = np.random.default_rng(0)
    jobs = [(model, banks, mode, cell) for model, banks in DATASETS for mode in ("off", "on")
            for cell in CELLS[model][mode]]
    results, loaded = {}, {}
    for model, banks, mode, (position, layer) in tqdm(jobs, desc="cells"):
        key = (model, banks, mode)
        if key not in loaded:
            loaded.clear()   # one dataset in memory at a time
            loaded[key] = dataset(root, model, mode, banks)
        parts, data = loaded[key]
        X = cell_matrix(parts, position, layer)
        name = f"{model} {'+'.join(banks)} {mode}"
        results.setdefault(name, {})[f"{position} L{layer}"] = cell_result(X, data, rng)
        (root / "analysis" / "confound_sweep.json").write_text(json.dumps(results, indent=1))   # partial results survive
    text = "\n".join(summary_lines(results))
    (root / "analysis" / "confound_sweep.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
