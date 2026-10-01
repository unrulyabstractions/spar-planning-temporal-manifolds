#!/usr/bin/env python
"""Supervised, rotation-free polynomial ladder and dimension count of the horizon response.

For each cell (layer:position) and population, fit two-block estimators between activations X and the
Legendre polynomial targets of t = log10 H (degree 1..K, orthogonalised in degree order on the training
rows): PLS2 (Shantanu's suggestion; components ordered by covariance with the target block) and
reduced-rank ridge regression (RRR; rank-k best linear predictor of the block). Report, as a function of
the number of components k, the held-out R² of each degree column and of the whole block, the share of
held-out activation variance (and of between-horizon centroid variance) along each response direction (orthonormalised in component order), and the degree each component tracks
(argmax |corr| with the degree columns). k95 = smallest k reaching 95% of the best total R² over k ≤ kmax
is the supervised count of horizon-responsive dimensions.

Populations:
  matrix pooled        matrix run, main rows, train configs → test configs (3 renderings × 2 domains mixed)
  matrix LEACE         same, after erasing rendering×domain cell + scenario parameters (fit on train rows)
  matrix sub-cell      structured|investment rows only (train configs → test configs)
  canonical            single-scenario run (17 horizons), stratified 80/20 row split
  extended             25-horizon run, same split (wider t range)

Usage: pls_check.py MATRIX_RUN OUT_DIR [--cells 0.55L:T3,0.92L:T3,0.55L:R0,0.72L:R0] [--degree 4] [--kmax 6]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, train_test_split
from ptm.erasure import LEACE, one_hot
from ptm.pls import HorizonPLS, ReducedRankRidge
from ptm.store import RunData
from ptm.depth import resolve_cell, parse_layers

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("out_dir"); ap.add_argument("--cells", default="0.55L:T3,0.92L:T3,0.55L:R0,0.72L:R0")
ap.add_argument("--canonical-run", default="runs/qwen3-14b_investment_n2000_s0"); ap.add_argument("--extended-run", default="runs/qwen3-14b_extended_n2000_s0")
ap.add_argument("--degree", type=int, default=4); ap.add_argument("--kmax", type=int, default=6)
a = ap.parse_args()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
K, KMAX = a.degree, a.kmax
runs = {"matrix": RunData(a.run_dir)}
for name, path in [("canonical", a.canonical_run), ("extended", a.extended_run)]:
    if path and Path(path).exists(): runs[name] = RunData(path)
mdf = runs["matrix"].index.reset_index(drop=True)
main = (mdf.condition == "main").to_numpy(); m_tr = main & (mdf.split == "train").to_numpy(); m_te = main & (mdf.split == "test").to_numpy()
cell_lab = (mdf.rendering.astype(str) + "|" + mdf.domain.astype(str)).to_numpy()
par = np.column_stack([np.log10(mdf.short_reward), np.log10(mdf.long_reward / mdf.short_reward), np.log10(mdf.short_delay_years), np.log10(mdf.long_delay_years), mdf.short_first.astype(float)])
Znuis = np.hstack([one_hot(cell_lab), par])


def choose_alpha(X, Y_t, t_min, t_max):
    """5-fold CV on the training rows over a log grid, scored by the full-rank (rank = K) total R²."""
    best = (None, -np.inf)
    for alpha in np.logspace(0, 6, 7):
        sc = []
        for tr_i, va_i in KFold(5, shuffle=True, random_state=0).split(X):
            m = ReducedRankRidge(K, K, t_min, t_max, alpha).fit(X[tr_i], Y_t[tr_i]); sc.append(m.r2_total(X[va_i], Y_t[va_i]))
        if np.mean(sc) > best[1]: best = (alpha, float(np.mean(sc)))
    return best


def populations(layer, pos):
    X = runs["matrix"].get(layer, pos).astype(np.float64); t = np.log10(mdf.horizon_years.to_numpy(dtype=float))
    yield "matrix pooled", X[m_tr], t[m_tr], X[m_te], t[m_te]
    er = LEACE().fit(X[m_tr], Znuis[m_tr]); yield "matrix LEACE(cell+par)", er.transform(X[m_tr]), t[m_tr], er.transform(X[m_te]), t[m_te]
    sc = cell_lab == "structured|investment"; yield "matrix structured|investment", X[m_tr & sc], t[m_tr & sc], X[m_te & sc], t[m_te & sc]
    for name in ["canonical", "extended"]:
        if name not in runs: continue
        r = runs[name]; df = r.index; ok = r.valid_mask(pos) & df.horizon_years.notna().to_numpy()
        Xr = r.get(layer, pos).astype(np.float64)[ok]; tr_ = np.log10(df.horizon_years.to_numpy(dtype=float)[ok])
        i_tr, i_te = train_test_split(np.arange(len(tr_)), test_size=0.2, random_state=0, stratify=np.round(tr_, 6))
        yield f"{name} ({len(tr_)} rows, {len(np.unique(tr_))} horizons)", Xr[i_tr], tr_[i_tr], Xr[i_te], tr_[i_te]


rows = []
for cell in a.cells.split(","):
    layer, pos = resolve_cell(cell, runs["matrix"])
    print(f"\n=== L{layer} {pos} ===")
    for pname, Xtr, ttr, Xte, tte in populations(layer, pos):
        t_min, t_max = float(ttr.min()) - 0.05, float(ttr.max()) + 0.05
        alpha, cv = choose_alpha(Xtr, ttr, t_min, t_max)
        print(f"\n--- {pname}: train {len(ttr)}, test {len(tte)}; RRR alpha {alpha:g} (CV total R² {cv:.3f}) ---")
        print(f"{'estimator':6s} {'k':>2s} | " + " ".join(f"{'R2 d'+str(d):>6s}" for d in range(1, K + 1)) + f" {'total':>6s} | {'test-variance share per response dir':>32s} | {'centroid-variance share':>28s} | degree per component")
        for est in ["RRR", "PLS2"]:
            tot = []
            for k in range(1, KMAX + 1):
                m = (ReducedRankRidge(k, K, t_min, t_max, alpha) if est == "RRR" else HorizonPLS(k, K, t_min, t_max)).fit(Xtr, ttr)
                r2 = m.r2_per_degree(Xte, tte); rt = m.r2_total(Xte, tte); tot.append(rt)
                xv = m.x_variance_share(Xte); cv_ = m.centroid_variance_share(Xte, tte); dg = m.degree_correlations(Xte, tte).argmax(1) + 1
                rows.append(dict(layer=layer, position=pos, population=pname, estimator=est, k=k, alpha=alpha if est == "RRR" else np.nan,
                                 **{f"r2_deg{d + 1}": r2[d] for d in range(K)}, r2_total=rt, xvar=" ".join(f"{v:.3f}" for v in xv), cvar=" ".join(f"{v:.3f}" for v in cv_), degrees=" ".join(map(str, dg))))
                print(f"{est:6s} {k:2d} | " + " ".join(f"{v:6.3f}" for v in r2) + f" {rt:6.3f} | {' '.join(f'{v:.3f}' for v in xv):>32s} | {' '.join(f'{v:.2f}' for v in cv_):>28s} | {' '.join(map(str, dg))}")
            k95 = int(np.argmax(np.array(tot) >= 0.95 * max(tot))) + 1
            print(f"{est:6s} k95 = {k95}  (95% of best total held-out R² {max(tot):.3f})")
            rows.append(dict(layer=layer, position=pos, population=pname, estimator=est, k=0, k95=k95, best_total=max(tot)))
        sys.stdout.flush()
pd.DataFrame(rows).to_csv(out / "pls_check.csv", index=False)
print("\nwrote", out)
