#!/usr/bin/env python
"""Paired unit-rewrite analysis: same option parameters, horizon rendered in different units.

Behavior: within-pair Δp_short = p_short(rendering) − p_short(canonical rendering), by unit.
Geometry (per cell): the displacement x(rendering) − x(canonical) of each pair, projected on the
local tangent of the canonical-centroid path (central difference between neighbouring canonical
bins) and expressed in units of the local adjacent-bin spacing ("grid steps"; + = toward longer),
plus the orthogonal remainder. Then the pair-level link: Spearman(Δalong, Δp_short) in mixed bins.
Also ridge transfer (train canonical → test rewrites) with residual by unit, as in analyze_variants.

Usage: analyze_paired.py RUN_DIR OUT_DIR [--cells 22:T3,37:T3,22:R0,29:R0]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from ptm.analysis import cell_metrics, _import_plt, cell_name
from ptm.store import RunData

UNIT_RANK = {"hours": 0, "days": 1, "weeks": 2, "months": 3, "years": 4, "decades": 5, "centuries": 6}
ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("out_dir"); ap.add_argument("--cells", default="22:T3,37:T3,22:R0,29:R0")
a = ap.parse_args()
run = RunData(a.run_dir); df = run.index.reset_index(drop=True); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
has_h = df["horizon_years"].notna().to_numpy()
is_canon = (df["horizon_text"] == df["horizon_canonical"]).to_numpy() & has_h
canon_idx = df[is_canon].set_index("pair_id").index
pairs = df[has_h & ~is_canon].copy()
pairs = pairs[pairs["pair_id"].isin(canon_idx)]
canon_rows = df[is_canon].set_index("pair_id")
pairs["p_short_canon"] = pairs["pair_id"].map(canon_rows["p_short"])
pairs["dp_short"] = pairs["p_short"] - pairs["p_short_canon"]
pairs["canon_years"] = pairs["pair_id"].map(canon_rows["horizon_years"])
pairs["canon_row"] = pairs["pair_id"].map(pd.Series(np.flatnonzero(is_canon), index=canon_idx))
print(f"run {run.meta['model_name']}  n={len(df)}  canonical rows {int(is_canon.sum())}  rewritten rows with a canonical partner {len(pairs)}")
print("\n=== behavior: within-pair Δp_short by unit (+ = more short-option preference than the canonical rendering) ===")
g = pairs.groupby("horizon_unit")["dp_short"].agg(["mean", "sem", "count"]).round(3).sort_index(key=lambda s: s.map(UNIT_RANK))
print(g.to_string())
print("\nby canonical horizon (long end) × unit, mean Δp_short:")
long = pairs[pairs.canon_years >= 2]
print(long.pivot_table(index="horizon_canonical", columns="horizon_unit", values="dp_short", aggfunc="mean").round(2)
      .reindex(sorted(long.horizon_canonical.unique(), key=lambda h: canon_rows[canon_rows.horizon_text == h].horizon_years.iloc[0])).to_string())
sys.stdout.flush()

bins = df.loc[is_canon].groupby("horizon_text")["horizon_years"].first().sort_values()
labels, years = bins.index.tolist(), bins.to_numpy()
rows = []
for cell in a.cells.split(","):
    layer, pos = cell.split(":"); layer = int(layer)
    m, ex = cell_metrics(run, layer, pos)
    X = ex["X"]
    # canonical centroids, local tangents and spacings
    ht = df["horizon_text"].to_numpy()
    C = {h: X[is_canon & (ht == h)].mean(0) for h in labels}
    tang, spacing = {}, {}
    for i, h in enumerate(labels):
        lo, hi = labels[max(i - 1, 0)], labels[min(i + 1, len(labels) - 1)]
        v = C[hi] - C[lo]; tang[h] = v / np.linalg.norm(v)
        spacing[h] = np.linalg.norm(v) / (min(i + 1, len(labels) - 1) - max(i - 1, 0))
    d = X[pairs.index.to_numpy()] - X[pairs["canon_row"].to_numpy()]
    t = np.stack([tang[h] for h in pairs["horizon_canonical"]]); sp = np.array([spacing[h] for h in pairs["horizon_canonical"]])
    along = np.sum(d * t, 1) / sp
    normal = np.linalg.norm(d - t * np.sum(d * t, 1)[:, None], axis=1) / sp
    # normal component of a *within-canonical* pair of samples gives the noise floor for `normal`
    pairs[f"along_{cell}"] = along; pairs[f"normal_{cell}"] = normal
    gg = pairs.groupby("horizon_unit").agg(along_mean=(f"along_{cell}", "mean"), along_sem=(f"along_{cell}", "sem"),
                                           normal_mean=(f"normal_{cell}", "mean"), n=("pair_id", "size")).round(2).sort_index(key=lambda s: s.map(UNIT_RANK))
    # noise floor: displacement between two canonical samples of the same bin (random pairs)
    rng = np.random.default_rng(0); floor = []
    for h in labels:
        idx = np.flatnonzero(is_canon & (ht == h))
        if len(idx) >= 2:
            p = rng.permutation(idx); dd = X[p[: len(p) // 2]] - X[p[len(p) // 2: 2 * (len(p) // 2)]]
            floor += list(np.linalg.norm(dd - tang[h] * (dd @ tang[h])[:, None], axis=1) / spacing[h])
    # ridge transfer canonical -> rewrites
    y = np.log10(df["horizon_years"].to_numpy(dtype=float))
    ridge = RidgeCV(alphas=np.logspace(1, 6, 6)).fit(X[is_canon], y[is_canon])
    te = pairs.index.to_numpy(); pred = ridge.predict(X[te]); r2 = r2_score(y[te], pred)
    resid = pd.Series(pred - y[te]).groupby(pairs["horizon_unit"].to_numpy()).mean().round(2).to_dict()
    mixed = pairs[(pairs.canon_years >= 2)]
    rho_link = spearmanr(mixed[f"along_{cell}"], mixed["dp_short"]).statistic if len(mixed) > 10 else np.nan
    rho_link_all = spearmanr(pairs[f"along_{cell}"], pairs["dp_short"]).statistic
    print(f"\n=== L{layer} {pos} ===")
    print(f"ridge transfer canonical->rewrites R^2 vs magnitude {r2:.3f}; residual (pred-true, decades) by unit {resid}")
    print(f"displacement in grid steps (along tangent, + = toward longer; normal = orthogonal remainder); noise floor for normal = {np.median(floor):.2f}")
    print(gg.to_string())
    print(f"pair-level link Spearman(Δalong, Δp_short): mixed bins (>=2y) {rho_link:+.3f} (n={len(mixed)}), all {rho_link_all:+.3f}")
    sys.stdout.flush()
    rows.append(dict(layer=layer, position=pos, r2_transfer=r2, rho_link_mixed=rho_link, rho_link_all=rho_link_all, normal_floor=float(np.median(floor)), **{f"along_{u}": v for u, v in gg["along_mean"].items()}))
pd.DataFrame(rows).to_csv(out / "paired_cells.csv", index=False)
pairs.to_csv(out / "pairs.csv", index=False)
# figure: along-tangent displacement by unit for each cell
plt = _import_plt()
cells = a.cells.split(","); fig, axes = plt.subplots(1, len(cells), figsize=(3.6 * len(cells), 3.6), sharey=True)
for ax, cell in zip(np.atleast_1d(axes), cells):
    gg = pairs.groupby("horizon_unit")[f"along_{cell}"].agg(["mean", "sem"]).sort_index(key=lambda s: s.map(UNIT_RANK))
    ax.axhline(0, color="#9a9a9a", lw=1); ax.errorbar(range(len(gg)), gg["mean"], yerr=1.96 * gg["sem"], fmt="o", color="#1b7f79", capsize=3)
    ax.set_xticks(range(len(gg))); ax.set_xticklabels(gg.index, rotation=45, ha="right", fontsize=8); ax.set_title(cell_name(run, int(cell.split(":")[0]), cell.split(":")[1]), fontsize=9); ax.spines[["top", "right"]].set_visible(False)
np.atleast_1d(axes)[0].set_ylabel("along-path displacement (grid steps, + = longer)")
fig.suptitle(f"{run.meta['model_name']}: unit rewrite displacement along the horizon path (95% CI)", fontsize=10); fig.tight_layout()
fig.savefig(out / "paired_along_by_unit.png", dpi=150); plt.close(fig)
print("wrote", out)
