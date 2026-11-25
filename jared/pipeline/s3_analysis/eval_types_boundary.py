"""S0 of EvalsBoundarySpec: where does the bank's horizon structure hold on the eval types, and where does it stop?

    python -m pipeline.s3_analysis.eval_types_boundary --bank-on $RESULTS/exp0_Qwen3-8B_think-on_full \
        [--bank-off $RESULTS/exp0_Qwen3-8B_think-off_full] --out OUT RUN_DIR [RUN_DIR ...]

Per (type, layer, position) cell, with a 95% bootstrap CI over base items (rows of one item, i.e. its
swap and twin, are resampled together):
  fixed probe   the full-bank ridge probe (as in eval_types_probe.py), applied unchanged
  refit probe   ridge within the type, GroupKFold(5) holding out whole horizons (as the bank CV does), penalty
                by inner CV over the same groups (dual form: ~48 rows, 4096 dims)
  unit lookup   mean log horizon of the row's unit word, from the bank ("unit_bank") or from the training items
                of the type's own folds ("unit_cv"); no activations
  PCA           the bank's horizon averages define the PCs; PC1 is oriented to rise with horizon
  null          targets shuffled across items 200 times. The refit null reuses the penalty chosen on the real
                labels (and the same folds) instead of re-selecting it on every shuffle.
  tiers         strong / weak / above_random / fail, separately for fixed and refit; "moved" = fixed < refit

Target: log10 horizon, else log10 of the geometric midpoint of `interval_years`, else none. Rows without a
target only get the fixed probe's readout mean and std, to compare with the bank's out-of-fold readout std.
The position names resolve against each run's own `tokens`, so thinking-off runs and `phrase_end` plug in;
a position the bank lacks (phrase_end) gets no fixed probe and no PCA.
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from pipeline.s3_analysis.eval_types_probe import LAYERS, bank_data
from spar_horizon.geometry import ALPHAS
from spar_horizon.probes import INNER_FOLDS, fit_ridge

POSITION_NAMES = ("suffix_end", "think_end", "prelabel", "phrase_end")
ANCHOR_TOKEN = {"think_end": "think@1*", "prelabel": ":", "phrase_end": "phrase_end"}
TIERS = ("fail", "above_random", "weak", "strong")
FIGURE_CELLS = {(18, "suffix_end"), (22, "suffix_end"), (18, "prelabel"), (22, "prelabel")}


# ---------- targets, units, positions ----------

def position_index(tokens, name):
    tokens = [str(t) for t in tokens]
    if name == "suffix_end":
        return tokens.index("assistant") + 1 if "assistant" in tokens else None
    anchor = ANCHOR_TOKEN[name]
    return tokens.index(anchor) if anchor in tokens else None


def row_target(row):
    horizon = row.get("horizon")
    if horizon and np.isfinite(horizon):
        return float(np.log10(horizon))
    if row.get("interval_years"):
        lo, hi = row["interval_years"]
        return float(0.5 * (np.log10(lo) + np.log10(hi)))
    return None


MIN_UNIT_ROWS = 8


def prompt_text(prompt):
    """A prompt is a string or a list of {role, content} messages."""
    return prompt if isinstance(prompt, str) else "\n".join(m["content"] for m in prompt)


def unit_word(text):
    word = text.split()[-1].lower()
    return word[:-1] if word.endswith("s") else word


def bank_unit_means(bank_dir):
    """Mean log10 horizon per unit word over the bank prompts ("... time horizon: 3 weeks.")."""
    answers = json.loads((Path(bank_dir) / "answers.json").read_text())
    horizons = np.load(Path(bank_dir) / "activations.npz", mmap_mode="r")["horizons"]
    by_unit = {}
    for a, h in zip(answers, horizons):
        m = re.search(r"time horizon: (.*?)\.\n", a["prompt"])
        if m and np.isfinite(h):
            by_unit.setdefault(unit_word(m.group(1)), []).append(np.log10(h))
    return {u: float(np.mean(v)) for u, v in by_unit.items()}


# ---------- metrics ----------

def spearman(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def metrics(pred, y):
    """(Spearman, MAE, bias) in log10 units (decades), over rows with a prediction."""
    ok = np.isfinite(pred)
    if ok.sum() < 3:
        return (np.nan,) * 3
    pred, y = pred[ok], y[ok]
    return spearman(pred, y), float(np.mean(np.abs(pred - y))), float(np.mean(pred - y))


def bootstrap_cis(preds, y, item_rows, n_boot, rng):
    """{method: 3x2 array of [lo, hi] for (Spearman, MAE, bias)}; items resampled with their rows."""
    draws = {m: np.empty((n_boot, 3)) for m in preds}
    for b in range(n_boot):
        idx = np.concatenate([item_rows[i] for i in rng.integers(len(item_rows), size=len(item_rows))])
        for m, p in preds.items():
            draws[m][b] = metrics(p[idx], y[idx])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns: a method with no usable rows
        return {m: np.nanpercentile(d, [2.5, 97.5], axis=0).T for m, d in draws.items()}


def permute_items(y, item_index, n_draws, rng):
    """(rows, n_draws) targets: each draw moves whole items' targets onto other items."""
    first_row = np.array([np.flatnonzero(item_index == i)[0] for i in range(item_index.max() + 1)])
    item_targets = y[first_row]
    return np.stack([item_targets[rng.permutation(len(item_targets))][item_index] for _ in range(n_draws)], axis=1)


def tier(rho, rho_lo, null_p95, bias, unit_rho, check_bias):
    """Highest tier met. `unit_rho` is NaN where there is no unit lookup (or it is constant): no bar."""
    if not np.isfinite(rho_lo):
        return "fail"
    beats_unit = not np.isfinite(unit_rho) or rho > unit_rho
    if rho_lo >= 0.8 and (not check_bias or abs(bias) <= 1) and beats_unit:
        return "strong"
    if rho_lo >= 0.5:
        return "weak"
    return "above_random" if rho_lo > null_p95 else "fail"


# ---------- ridge in dual form ----------

class Fold:
    """One train/test split of a Gram matrix, with the train kernel centered and eigendecomposed once."""

    def __init__(self, gram, train, test):
        k = gram[np.ix_(train, train)]
        col, total = k.mean(0), k.mean()
        k_test = gram[np.ix_(test, train)]
        self.train, self.test = train, test
        self.w, self.U = np.linalg.eigh(k - col[None, :] - col[:, None] + total)
        self.k_test = k_test - k_test.mean(1, keepdims=True) - col[None, :] + total

    def predict(self, Y, alpha):
        """Y: (rows, draws). Ridge with intercept; predictions for the test rows."""
        y_train = Y[self.train]
        mean = y_train.mean(0)
        coef = self.U @ ((self.U.T @ (y_train - mean)) / (self.w + alpha)[:, None])
        return mean + self.k_test @ coef


def select_alpha(gram, train, y, items):
    inner = GroupKFold(min(INNER_FOLDS, len(np.unique(items[train])))).split(train, groups=items[train])
    folds = [Fold(gram, train[a], train[b]) for a, b in inner]
    errors = [sum(((f.predict(y[:, None], a)[:, 0] - y[f.test]) ** 2).sum() for f in folds) for a in ALPHAS]
    return ALPHAS[int(np.argmin(errors))]


class RefitProbe:
    """Within-type ridge: held-out predictions from GroupKFold by `groups` (whole horizons), penalty
    chosen by inner CV over the same groups and fixed per fold."""

    def __init__(self, gram, y, groups):
        self.n = len(y)
        self.folds, self.alphas = [], []
        for train, test in GroupKFold(min(5, len(np.unique(groups)))).split(gram, groups=groups):
            self.alphas.append(select_alpha(gram, train, y, groups))
            self.folds.append(Fold(gram, train, test))

    def held_out(self, Y):
        out = np.empty(Y.shape)
        for fold, alpha in zip(self.folds, self.alphas):
            out[fold.test] = fold.predict(Y, alpha)
        return out


def unit_cv_predictions(folds, y, units):
    """Mean training log horizon of the row's unit word; the fold mean for a unit it has not seen."""
    pred = np.full(len(y), np.nan)
    for fold in folds:
        train_units = units[fold.train]
        for i in fold.test:
            if units[i] is None:
                continue
            same = np.array([u == units[i] for u in train_units])
            pred[i] = y[fold.train][same].mean() if same.any() else y[fold.train].mean()
    return pred


# ---------- bank side of a cell ----------

def pca_axes(averages, k):
    """Center and top-k principal axes of the rows of `averages`."""
    mean = averages.mean(0)
    return mean, np.linalg.svd(averages - mean, full_matrices=False)[2][:k]


class BankCell:
    """Everything a cell needs from the bank: the fixed probe, its CV reference, and the PCA readouts."""

    def __init__(self, X, y):
        self.probe = fit_ridge(X, y, y)
        held_out = np.empty_like(y)
        for train, test in GroupKFold(5).split(X, y, y):
            held_out[test] = fit_ridge(X[train], y[train], y[train]).predict(X[test])
        self.cv = {"spearman": spearman(held_out, y), "mae": float(np.mean(np.abs(held_out - y))),
                   "readout_std": float(held_out.std())}
        self.levels = np.unique(y)
        averages = np.stack([X[y == h].mean(0) for h in self.levels])
        self.mean, self.axes = pca_axes(averages, 4)
        coords = (averages - self.mean) @ self.axes.T
        if spearman(coords[:, 0], self.levels) < 0:
            self.axes[0] *= -1
            coords[:, 0] *= -1
        self.coords = coords
        self.readouts = {k: np.linalg.lstsq(np.c_[coords[:, :k], np.ones(len(coords))], self.levels, rcond=None)[0]
                         for k in (2, 4)}

    def project(self, X):
        return (X - self.mean) @ self.axes.T

    def pca_report(self, X, y):
        coords = self.project(X)
        out = {"pc1_spearman": spearman(coords[:, 0], y)}
        for k, coef in self.readouts.items():
            rho, mae, bias = metrics(np.c_[coords[:, :k], np.ones(len(coords))] @ coef, y)
            out[f"k{k}"] = {"spearman": rho, "mae": mae, "bias": bias}
        return out


def own_pc1(X, y):
    """|Spearman| of the type's own horizon-average PC1 with horizon; PC sign is arbitrary."""
    levels = np.unique(y)
    if len(levels) < 3:
        return np.nan
    averages = np.stack([X[y == h].mean(0) for h in levels])
    mean, axes = pca_axes(averages, 1)
    return abs(spearman((averages - mean) @ axes[0], levels))


# ---------- one (type, cell) ----------

def method_report(pred, y, ci, null_p95):
    rho, mae, bias = metrics(pred, y)
    report = {"spearman": rho, "mae": mae, "bias": bias, "n": int(np.isfinite(pred).sum()),
              "ci": {"spearman": ci[0], "mae": ci[1], "bias": ci[2]}}
    if null_p95 is not None:
        report["null_p95"] = null_p95
    return report


def analyse_rows(X, y, items, units, bank, unit_table, n_boot, n_null, rng):
    """X, y: rows with a target. `items`: item index per row. `units`: unit word or None per row.
    `bank`: BankCell or None. Returns (report, per-method held-out predictions)."""
    gram = X.astype(np.float64) @ X.astype(np.float64).T
    # Folds hold out whole horizons, as the bank CV does: with ~14 distinct horizons over 48 items,
    # item folds leave every test horizon in training, and the refit could score by memorising each
    # horizon's tokens instead of reading an ordered structure. An item's rows share its target, so
    # they stay in one fold.
    horizon_groups = np.unique(y, return_inverse=True)[1]
    refit = RefitProbe(gram, y, horizon_groups)
    shuffled = permute_items(y, items, n_null, rng)
    preds = {"refit": refit.held_out(y[:, None])[:, 0],
             "unit_cv": unit_cv_predictions(refit.folds, y, units),
             "unit_bank": np.array([unit_table.get(u, np.nan) if u else np.nan for u in units])}
    # Implied and persona types show a unit word only in their few explicit twins; a rank correlation on
    # ~4 rows is no baseline, so the lookup is dropped (no bar for "strong") below MIN_UNIT_ROWS.
    for m in ("unit_cv", "unit_bank"):
        if np.isfinite(preds[m]).sum() < MIN_UNIT_ROWS:
            preds[m] = np.full(len(y), np.nan)
    null = {"refit": [spearman(p, y_s) for p, y_s in zip(refit.held_out(shuffled).T, shuffled.T)]}
    if bank is not None:
        preds["fixed"] = bank.probe.predict(X)
        null["fixed"] = [spearman(preds["fixed"], y_s) for y_s in shuffled.T]
    item_rows = [np.flatnonzero(items == i) for i in range(items.max() + 1)]
    cis = bootstrap_cis(preds, y, item_rows, n_boot, rng)
    p95 = {m: float(np.nanpercentile(v, 95)) for m, v in null.items()}

    report = {m: method_report(preds[m], y, cis[m], p95.get(m)) for m in preds}
    unit_rho = {"fixed": report["unit_bank"]["spearman"], "refit": report["unit_cv"]["spearman"]}
    for m in ("fixed", "refit"):
        if m in report:
            report[m]["tier"] = tier(report[m]["spearman"], cis[m][0][0], p95[m], report[m]["bias"],
                                     unit_rho[m], m == "fixed")
    if bank is not None:
        report["pca"] = {**bank.pca_report(X, y), "own_pc1_abs_spearman": own_pc1(X, y)}
        report["moved"] = TIERS.index(report["fixed"]["tier"]) < TIERS.index(report["refit"]["tier"])
    return report, preds


# ---------- run directories ----------

def load_run(run_dir):
    """{type: rows with activations}, the run's token names, and its thinking mode."""
    run_dir = Path(run_dir)
    settings = json.loads((run_dir / "settings.json").read_text())
    types, tokens = {}, None
    for rows_path in sorted(run_dir.glob("*/rows.json")):
        rows = [r for r in json.loads(rows_path.read_text()) if r.get("act_file")]
        if rows:
            types[rows_path.parent.name] = rows
            if tokens is None:
                tokens = np.load(rows_path.parent / rows[0]["act_file"])["tokens"]
    return types, tokens, settings["thinking"]


def type_activations(type_dir, rows, layer):
    files = {}
    out = []
    for r in rows:
        if r["act_file"] not in files:
            files[r["act_file"]] = np.load(type_dir / r["act_file"])[f"L{layer}"]
        out.append(files[r["act_file"]][r["act_row"]])
    return np.asarray(out, dtype=np.float32)


def bank_for(banks, mode, position):
    """Thinking-on and -off banks agree at suffix_end, so either serves there."""
    if banks.get(mode):
        return banks[mode]
    return banks["on"] if position == "suffix_end" else None


def bank_cell_for(banks, mode, layer, position, cache):
    bank_dir = bank_for(banks, mode, position)
    if bank_dir is None:
        return None
    tokens = cache.setdefault(bank_dir, np.load(Path(bank_dir) / "activations.npz")["tokens"])
    idx = position_index(tokens, position)
    return None if idx is None else BankCell(*bank_data(bank_dir, layer, idx))


def analyse_type(rows, X, bank, unit_table, n_boot, n_null, rng):
    """Record for one (type, cell): analysis of the rows with a target, readout stats for those without."""
    targets = [row_target(r) for r in rows]
    has = np.array([v is not None for v in targets])
    record = {"n_rows": len(rows), "n_with_target": int(has.sum())}
    figure_points = None
    if has.sum() >= 8:
        keys = np.array([r.get("twin_id", r["sample_uid"]) for r in rows])[has]
        items = np.unique(keys, return_inverse=True)[1]
        # Implied and persona rows carry their explicit twin's horizon_text as the label but not in the
        # prompt, so a lookup on it would read the answer; only a unit word the model can see counts.
        units = np.array([unit_word(r["horizon_text"]) if r.get("horizon_text")
                          and r["horizon_text"] in prompt_text(r["prompt"]) else None for r in rows],
                         dtype=object)[has]
        y = np.array([v for v in targets if v is not None])
        report, preds = analyse_rows(X[has], y, items, units, bank, unit_table, n_boot, n_null, rng)
        record.update(report, n_items=int(items.max() + 1))
        record["predictions"] = {"sample_uid": [r["sample_uid"] for r, h in zip(rows, has) if h], "target": y, **preds}
        if bank is not None:
            levels = np.unique(y)
            averages = np.stack([X[has][y == h].mean(0) for h in levels])
            figure_points = (bank.project(averages)[:, :2], levels)
    if bank is not None and (~has).any():
        readout = bank.probe.predict(X[~has])
        record["no_target"] = {"n": int((~has).sum()), "fixed_mean": float(readout.mean()),
                               "fixed_std": float(readout.std()), "bank_readout_std": bank.cv["readout_std"]}
        record["no_target_fixed"] = {"sample_uid": [r["sample_uid"] for r, h in zip(rows, has) if not h],
                                     "fixed": readout}
    return record, figure_points


def analyse_run(run_dir, banks, n_boot, n_null, rng, tag):
    """(results {type: {cell: record}}, figure points {(position, layer): {type: ...}}, bank cells)."""
    types, tokens, mode = load_run(run_dir)
    positions = {p: position_index(tokens, p) for p in POSITION_NAMES if position_index(tokens, p) is not None}
    unit_table = bank_unit_means(banks[mode] or banks["on"])
    results, figures, bank_cells, cache = {}, {}, {}, {}
    for layer in LAYERS:
        acts = {t: type_activations(Path(run_dir) / t, rows, layer) for t, rows in types.items()}
        for position, idx in positions.items():
            bank = bank_cell_for(banks, mode, layer, position, cache)
            bank_cells[(layer, position)] = bank
            for t, rows in tqdm(types.items(), desc=f"{tag}L{layer}_{position}", leave=False):
                record, points = analyse_type(rows, acts[t][:, idx], bank, unit_table, n_boot, n_null, rng)
                results.setdefault(t, {})[f"L{layer}_{position}"] = {"layer": layer, "position": position, **record}
                if points is not None and (layer, position) in FIGURE_CELLS:
                    figures.setdefault((position, layer), {})[t] = points
    return results, figures, bank_cells


# ---------- output ----------

def clean(obj):
    """JSON-ready: numpy to python, rounded, NaN to None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), 3)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def plot_pca(path, bank_cell, per_type, title):
    names = list(per_type)
    ncols = 6
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False, sharex=True, sharey=True)
    vmin, vmax = bank_cell.levels.min(), bank_cell.levels.max()
    for ax, name in zip(axes.ravel(), names):
        ax.scatter(*bank_cell.coords[:, :2].T, c=bank_cell.levels, s=8, alpha=0.35, vmin=vmin, vmax=vmax)
        xy, levels = per_type[name]
        points = ax.scatter(*xy.T, c=levels, s=30, edgecolors="k", linewidths=0.6, vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=8)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.colorbar(points, ax=axes, label="log10 horizon (years)", shrink=0.6)
    fig.suptitle(f"{title}: bank horizon averages (faint) and the type's horizon averages (outlined), PC1-PC2")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fmt(v, spec=".2f"):
    return "-" if v is None or not np.isfinite(v) else format(v, spec)


def compact_cell(rec):
    if "refit" not in rec:
        return "-"
    f, r, u = rec.get("fixed"), rec["refit"], rec["unit_bank"]
    fixed = "-" if f is None else f"{fmt(f['spearman'])} [{fmt(f['ci']['spearman'][0])},{fmt(f['ci']['spearman'][1])}]"
    letters = "".join(rec[m]["tier"][0].upper() if m in rec else "-" for m in ("fixed", "refit"))
    return f"{fixed} / {fmt(r['spearman'])} / {fmt(u['spearman'])} / {letters}{'!' if rec.get('moved') else ''}"


def full_row(t, cell, rec):
    if "refit" not in rec:
        return None
    def part(m):
        d = rec.get(m)
        if d is None:
            return ["-"] * 5
        ci = d["ci"]["spearman"]
        return [f"{fmt(d['spearman'])} [{fmt(ci[0])},{fmt(ci[1])}]", fmt(d["mae"]), fmt(d["bias"], "+.2f"),
                fmt(d.get("null_p95")), d.get("tier", "-")]
    unit = [rec["unit_bank"], rec["unit_cv"]]
    pca = rec.get("pca")
    cols = ([t, cell, str(rec["n_items"])] + part("fixed") + part("refit")
            + [f"{fmt(u['spearman'])} / {fmt(u['mae'])} / {fmt(u['bias'], '+.2f')}" for u in unit]
            + (["-"] * 4 if pca is None else [fmt(pca["pc1_spearman"]), fmt(pca["k2"]["spearman"]),
                                              fmt(pca["k4"]["spearman"]), fmt(pca["own_pc1_abs_spearman"])])
            + ["moved" if rec.get("moved") else ""])
    return "| " + " | ".join(cols) + " |"


def markdown(results, bank_cv, layers, positions):
    out = ["# Boundary S0: fixed vs refit probes, unit lookup, PCA\n",
           "Cells: `fixed rho [95% CI] / refit rho / unit-lookup rho (bank) / tiers fixed,refit`. "
           "Tiers: S strong, W weak, A above random, F fail; `!` = moved (fixed tier below refit).\n"]
    for position in positions:
        out.append(f"\n## {position}\n")
        out.append("| type | " + " | ".join(f"L{layer}" for layer in layers) + " |\n|---|" + "---|" * len(layers))
        ref = [bank_cv.get(f"L{layer}_{position}") for layer in layers]
        out.append("| bank CV (rho / MAE) | " + " | ".join("-" if b is None else f"{fmt(b['spearman'])} / {fmt(b['mae'])}"
                                                           for b in ref) + " |")
        for t, cells in results.items():
            out.append(f"| {t} | " + " | ".join(compact_cell(cells.get(f"L{layer}_{position}", {})) for layer in layers) + " |")
    out.append("\n## Full table\n")
    out.append("fixed/refit columns: rho [CI], MAE, bias (decades), null p95, tier. Unit columns: rho / MAE / bias for "
               "bank-trained and within-type CV lookups. PCA: PC1 rho, k=2 and k=4 readout rho, own-type PC1 |rho| "
               "(its sign is arbitrary).\n")
    head = ["type", "cell", "items"] + [f"{m} {c}" for m in ("fixed", "refit") for c in ("rho", "MAE", "bias", "null p95", "tier")]
    head += ["unit bank", "unit cv", "PC1", "k2", "k4", "own PC1", "moved"]
    out.append("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for t, cells in results.items():
        out += [r for cell, rec in cells.items() if (r := full_row(t, cell, rec))]
    controls = [(t, cell, rec["no_target"]) for t, cells in results.items() for cell, rec in cells.items() if "no_target" in rec]
    if controls:
        out.append("\n## Rows with no target: fixed-probe readout\n\n| type | cell | n | readout mean | readout std | bank OOF readout std |\n|---|---|---|---|---|---|")
        out += [f"| {t} | {c} | {d['n']} | {fmt(d['fixed_mean'])} | {fmt(d['fixed_std'])} | {fmt(d['bank_readout_std'])} |"
                for t, c, d in controls]
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-on", required=True)
    parser.add_argument("--bank-off")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    banks = {"on": args.bank_on, "off": args.bank_off}
    rng = np.random.default_rng(0)

    results, bank_cv = {}, {}
    for run in args.runs:
        run_name = Path(run).name
        tag = f"{run_name} " if len(args.runs) > 1 else ""
        run_results, figures, bank_cells = analyse_run(run, banks, args.n_boot, args.n_null, rng, tag)
        for t, cells in run_results.items():
            results[f"{run_name}:{t}" if len(args.runs) > 1 else t] = cells
        for (layer, position), cell in bank_cells.items():
            if cell is not None:
                bank_cv.setdefault(f"L{layer}_{position}", cell.cv)
        for (position, layer), per_type in figures.items():
            plot_pca(out / f"pca_{tag.replace(' ', '_')}{position}_L{layer}.png", bank_cells[(layer, position)],
                     per_type, f"{tag}{position} L{layer}")
    positions = list(dict.fromkeys(rec["position"] for cells in results.values() for rec in cells.values()))
    layers = sorted({rec["layer"] for cells in results.values() for rec in cells.values()})
    (out / "boundary.json").write_text(json.dumps(clean({"bank_cv": bank_cv, "types": results}), indent=1))
    (out / "boundary.md").write_text(markdown(results, bank_cv, layers, positions))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
