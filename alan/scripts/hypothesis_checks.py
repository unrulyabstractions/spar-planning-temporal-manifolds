#!/usr/bin/env python
"""Two falsifiable-hypothesis checks on a canonical run.

H(sink): at a position dominated by one massive-activation direction, ordinality (|Spearman| of a
PC with log horizon) appears in the PCs below the dominant one.
H(U-shape): in the plane of the 17 canonical-bin centroids, PC1 is linear in log horizon and PC2 is
quadratic with a vertex near one year.

Usage: hypothesis_checks.py RUN_DIR [--sink-position T4] [--sink-layers 14,22,29,37] [--ushape-cells 22:R0,22:T3,29:R0,29:T3,37:T3,37:R0]
"""
import argparse
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from ptm.store import RunData

ap = argparse.ArgumentParser(); ap.add_argument("run_dir")
ap.add_argument("--sink-position", default="T4"); ap.add_argument("--sink-layers", default="14,22,29,37")
ap.add_argument("--ushape-cells", default="22:R0,22:T3,29:R0,29:T3,37:T3,37:R0")
ap.add_argument("--variant-column", default=None, help="extra categorical column for the variant check (e.g. rendering, condition)")
ap.add_argument("--ushape-range", default=None, help="restrict the centroid fit to horizons within [min,max] years, e.g. 0.0027,100 (the standard grid)")
a = ap.parse_args()
run = RunData(a.run_dir); df = run.index
has_h = df.horizon_years.notna().to_numpy(); y = np.log10(df.horizon_years.to_numpy(dtype=float))
print(f"run {run.meta['model_name']} n={len(df)}")
print(f"=== H(sink) at {a.sink_position}: |rho| of the k-th PC with log horizon (same PCA fit); evr_top1; top/median PC std ===")
print("layer  evr_top1  k=0     k=1     k=2     k=5    top/median")
for layer in [int(x) for x in a.sink_layers.split(",")]:
    X = run.get(layer, a.sink_position)[has_h]; Xc = X - X.mean(0)
    p = PCA(8, random_state=0).fit(Xc); Z = p.transform(Xc)
    s = np.linalg.svd(Xc, compute_uv=False)
    print(f"L{layer:2d}    {p.explained_variance_ratio_[0]:.3f}    " + "  ".join(f"{abs(spearmanr(Z[:, k], y[has_h]).statistic):.3f}" for k in [0, 1, 2, 5]) + f"   {s[0]/np.median(s[:50]):.0f}x")
print("\n=== H(U-shape): 17 canonical centroids in their own PC plane; polynomial fits vs log10 horizon ===")
print("cell     PC2: R2 quad  vertex(years)  R2 linear | PC1: R2 linear  R2 quad")
ht = df.horizon_text.to_numpy()
labels = df[has_h].groupby("horizon_text").horizon_years.first().sort_values()
if a.ushape_range:
    lo, hi = (float(x) for x in a.ushape_range.split(","))
    labels = labels[(labels >= lo * 0.999) & (labels <= hi * 1.001)]
    print(f"(U-shape fit restricted to {len(labels)} bins in [{lo}, {hi}] years)")
ly = np.log10(labels.to_numpy())
def fit(z, deg):
    c = np.polyfit(ly, z, deg); r = z - np.polyval(c, ly); return 1 - r.var() / z.var(), c
for cell in a.ushape_cells.split(","):
    layer, pos = cell.split(":"); layer = int(layer)
    X = run.get(layer, pos)
    C = np.stack([X[has_h & (ht == h)].mean(0) for h in labels.index])
    Zc = PCA(2).fit_transform(C)
    r2q2, cq = fit(Zc[:, 1], 2); r2l2, _ = fit(Zc[:, 1], 1); r2l1, _ = fit(Zc[:, 0], 1); r2q1, _ = fit(Zc[:, 0], 2)
    vertex = 10 ** (-cq[1] / (2 * cq[0])) if cq[0] != 0 else float("nan")
    print(f"L{layer} {pos}   {r2q2:.3f}         {vertex:6.2f}         {r2l2:.3f}     | {r2l1:.3f}           {r2q1:.3f}")


# ---------------------------------------------------------------------------------------------
# H(variants): on a run with a categorical `variant` column (phrasing_id or horizon_unit),
# ordinality within each variant vs pooled, and how strongly the variant clusters.
def variant_check(run: RunData, column: str, cells: str):
    from sklearn.metrics import silhouette_score
    df = run.index
    has_h = df.horizon_years.notna().to_numpy(); y = np.log10(df.horizon_years.to_numpy(dtype=float))
    v = df[column].astype(str).to_numpy()
    keys = sorted(set(v[has_h]))
    print(f"\n=== H(variants) by {column} ({keys}) ===")
    print("cell     pooled|rho_pc1|  per-variant|rho_pc1|  silhouette(variant,PC1-3)  |rho(PC1,variant-mean)|")
    for cell in cells.split(","):
        layer, pos = cell.split(":"); layer = int(layer)
        X = run.get(layer, pos)[has_h]; yy = y[has_h]; vv = v[has_h]
        Z = PCA(3, random_state=0).fit_transform(X)
        pooled = abs(spearmanr(Z[:, 0], yy).statistic)
        per = "/".join(f"{abs(spearmanr(PCA(1, random_state=0).fit_transform(X[vv == k])[:, 0], yy[vv == k]).statistic):.2f}" for k in keys if (vv == k).sum() > 10)
        sil = silhouette_score(Z, vv)
        means = {k: Z[vv == k, 0].mean() for k in keys}
        rho_v = abs(spearmanr(Z[:, 0], [means[k] for k in vv]).statistic)
        print(f"L{layer:2d} {pos}   {pooled:.3f}            {per}       {sil:+.3f}                    {rho_v:.3f}")


if __name__ == "__main__" and "phrasing_id" in run.index.columns and run.index["phrasing_id"].nunique() > 1:
    variant_check(run, "phrasing_id", "14:R0,22:T3,22:R0,29:R0,37:T3")
if __name__ == "__main__" and "horizon_unit" in run.index.columns and run.index["horizon_unit"].nunique() > 1:
    variant_check(run, "horizon_unit", "14:R0,22:T3,22:R0,29:R0,37:T3")

if __name__ == "__main__" and a.variant_column and a.variant_column in run.index.columns:
    variant_check(run, a.variant_column, "14:R0,22:T3,22:R0,29:R0,37:T3")
