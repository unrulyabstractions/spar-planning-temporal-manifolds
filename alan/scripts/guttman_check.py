#!/usr/bin/env python
"""Guttman-effect (horseshoe) check on the horizon-bin centroids.

For each cell: (a) polynomial ladder: R^2 of each centroid PC_k against orthogonal polynomials of log H by
degree; (b) banded similarity: R^2 of the centroid distance matrix explained by |Δ log H| alone;
(c) vertex-tracks-midpoint: the PC2 parabola is refit on sliding windows of consecutive bins and the vertex
is regressed on the window midpoint (slope ≈ 1: Guttman; ≈ 0: a fixed landmark); (d) noise: PC2 eigenvalue
vs split-half noise; (e) matched null: banded-covariance centroids (exp(-|Δt|/ℓ) with ℓ fit to the data, plus
matched centroid noise) pushed through the same statistics, 200 draws.

Usage: guttman_check.py RUN_DIR [--cells 0.55L:R0,0.92L:T3,0.55L:T3,0.72L:R0] [--window 9] [--restrict lo,hi]
"""
import argparse, sys
import numpy as np, pandas as pd
from numpy.polynomial import legendre
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from ptm.store import RunData
from ptm.depth import resolve_cell, parse_layers

ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--cells", default="0.55L:R0,0.92L:T3,0.55L:T3,0.72L:R0")
ap.add_argument("--window", type=int, default=9); ap.add_argument("--restrict", default=None, help="lo,hi in years: use only bins in this range")
ap.add_argument("--min-bin", type=int, default=20); ap.add_argument("--draws", type=int, default=200)
a = ap.parse_args()
run = RunData(a.run_dir); df = run.index.reset_index(drop=True)
has_h = df.horizon_years.notna().to_numpy().copy()
if "condition" in df.columns and df.condition.notna().any():
    has_h &= (df.condition == "main").to_numpy()


def ortho_r2(z, t, deg):
    """R^2 of z regressed on Legendre polynomials of t (mapped to [-1,1]) up to `deg`."""
    x = 2 * (t - t.min()) / (t.max() - t.min()) - 1
    V = legendre.legvander(x, deg); beta, *_ = np.linalg.lstsq(V, z, rcond=None)
    r = z - V @ beta; return 1 - r.var() / z.var()


def quad_vertex(z, t):
    c = np.polyfit(t, z, 2); return -c[1] / (2 * c[0]) if c[0] != 0 else np.nan


def stats(C, t, halves=None):
    """Guttman statistics for centroids C [n_bins, d] at log-horizons t."""
    Cc = C - C.mean(0); pca = PCA(min(4, len(C) - 1)).fit(Cc); Z = pca.transform(Cc); ev = pca.explained_variance_ratio_
    ladder = {k: [ortho_r2(Z[:, k], t, d) for d in range(1, 5)] for k in range(Z.shape[1])}
    # banded similarity: distances vs |Δt|
    D = np.linalg.norm(Cc[:, None] - Cc[None], axis=-1); iu = np.triu_indices(len(C), 1)
    dt = np.abs(t[:, None] - t[None])[iu]; dd = D[iu]
    f = lambda x, s, l: s * (1 - np.exp(-x / l))
    try:
        (s_, l_), _ = curve_fit(f, dt, dd, p0=[dd.max(), 1.0], maxfev=5000); r2_band = 1 - np.var(dd - f(dt, s_, l_)) / np.var(dd)
    except Exception:
        s_, l_, r2_band = np.nan, np.nan, np.nan
    # vertex vs window midpoint
    w = min(a.window, len(t)); mids, verts = [], []
    for i in range(len(t) - w + 1):
        tt = t[i:i + w]; zz = PCA(2).fit_transform(Cc[i:i + w])[:, 1]
        v = quad_vertex(zz, tt)
        if np.isfinite(v) and tt.min() - 1 < v < tt.max() + 1:
            mids.append(tt.mean()); verts.append(v)
    slope = np.polyfit(mids, verts, 1)[0] if len(mids) > 2 else np.nan
    return dict(ev=ev, ladder=ladder, band_len=l_, band_r2=r2_band, vertex_full=quad_vertex(Z[:, 1], t), mid_full=float(t.mean()),
                vertex_slope=slope, n_windows=len(mids), pc2_quad=ladder[1][1] if 1 in ladder else np.nan)


bins_all = df.loc[has_h].groupby("horizon_text")["horizon_years"].first().sort_values()
for cell in a.cells.split(","):
    layer, pos = resolve_cell(cell, run)
    v = has_h & run.valid_mask(pos); X = run.get(layer, pos)[v]; ht = df.loc[v, "horizon_text"].to_numpy()
    labels = [h for h in bins_all.index if (ht == h).sum() >= a.min_bin]; years = np.array([bins_all[h] for h in labels])
    if a.restrict:
        lo, hi = (float(x) for x in a.restrict.split(",")); keep = (years >= lo * 0.999) & (years <= hi * 1.001); labels = [l for l, k in zip(labels, keep) if k]; years = years[keep]
    t = np.log10(years)
    C = np.stack([X[ht == h].mean(0) for h in labels])
    Co = np.stack([X[np.flatnonzero(ht == h)[0::2]].mean(0) for h in labels]); Ce = np.stack([X[np.flatnonzero(ht == h)[1::2]].mean(0) for h in labels])
    st = stats(C, t)
    noise = (Co - Ce) / 2; noise_var_per_dim = float(np.mean(noise.var(0)))
    print(f"\n=== L{layer} {pos}  ({len(labels)} bins, log10 range [{t.min():.2f}, {t.max():.2f}], midpoint {t.mean():.2f} = {10**t.mean():.2f} y) ===")
    print("(a) polynomial ladder: R^2 of centroid PC_k vs orthogonal polynomials of log H, by max degree 1..4   (Guttman: PC_k needs degree k)")
    for k, r in st["ladder"].items():
        best = int(np.argmax(np.diff([0] + r) > 0.15) + 1) if any(np.diff([0] + r) > 0.15) else 0
        print(f"    PC{k+1} (evr {st['ev'][k]:.3f}): " + "  ".join(f"deg{d}: {rr:.2f}" for d, rr in zip(range(1, 5), r)))
    print(f"(b) banded similarity: centroid distance = s(1-exp(-|Δt|/ℓ)) fits with R^2 {st['band_r2']:.3f}, ℓ = {st['band_len']:.2f} decades")
    print(f"(c) vertex vs window midpoint (windows of {a.window} bins, {st['n_windows']} windows): slope {st['vertex_slope']:+.2f}   [full-range vertex 10^{st['vertex_full']:.2f} = {10**st['vertex_full']:.2f} y vs midpoint {10**st['mid_full']:.2f} y]")
    print(f"(d) PC2 share {st['ev'][1]:.3f} vs split-half noise share ≈ {noise_var_per_dim * C.shape[1] / (C - C.mean(0)).var(0).sum() / 1:.4f} of centroid variance")
    # (e) matched null: GP centroids with the fitted decay length, unit-normal per dim, plus matched noise
    rng = np.random.default_rng(0); d_sim = 300
    K = np.exp(-np.abs(t[:, None] - t[None]) / max(st["band_len"], 0.05)); L = np.linalg.cholesky(K + 1e-8 * np.eye(len(t)))
    sig_ratio = np.sqrt(noise_var_per_dim / np.mean((C - C.mean(0)).var(0)))   # noise sd relative to centroid sd per dim
    null = []
    for _ in range(a.draws):
        G = L @ rng.normal(size=(len(t), d_sim)); G = G + sig_ratio * rng.normal(size=G.shape)
        s2 = stats(G, t); null.append(dict(ev1=s2["ev"][0], ev2=s2["ev"][1], ev3=s2["ev"][2] if len(s2["ev"]) > 2 else np.nan, pc2quad=s2["pc2_quad"],
                                          pc3cubic=s2["ladder"][2][2] if 2 in s2["ladder"] else np.nan, vslope=s2["vertex_slope"], vertex=s2["vertex_full"]))
    nd = pd.DataFrame(null); q = lambda c: f"{nd[c].quantile(.05):.2f}–{nd[c].quantile(.95):.2f}"
    obs_pc3cubic = st["ladder"][2][2] if 2 in st["ladder"] else np.nan
    print(f"(e) matched banded null ({a.draws} draws, ℓ={st['band_len']:.2f}, noise ratio {sig_ratio:.2f}):  evr1 {q('ev1')} (obs {st['ev'][0]:.2f})  evr2 {q('ev2')} (obs {st['ev'][1]:.2f})  evr3 {q('ev3')} (obs {st['ev'][2]:.2f})")
    print(f"    PC2 quadratic R^2 {q('pc2quad')} (obs {st['pc2_quad']:.2f});  PC3 cubic R^2 {q('pc3cubic')} (obs {obs_pc3cubic:.2f});  vertex slope {q('vslope')} (obs {st['vertex_slope']:+.2f});  vertex 10^{nd.vertex.quantile(.05):.2f}–10^{nd.vertex.quantile(.95):.2f} (obs 10^{st['vertex_full']:.2f})")
    sys.stdout.flush()
