#!/usr/bin/env python
"""Item 5: characterize the horizon path geometry beyond PCA ordinality.

Per (layer, position) cell, using the 17 canonical-horizon bin centroids in the full residual space:
  - path length vs end-to-end distance (tortuosity), turning angles between successive segments,
  - planarity: variance of the 17 centroids captured by their own top-1/2/3 PCs,
  - arc-length position of each centroid vs log10 horizon (uniformity / saturation),
  - within-bin spread (mean distance to centroid) vs adjacent-centroid spacing (SNR),
  - within-bin participation ratio (local dimensionality).
Also a per-layer sweep of these at chosen positions, and figures.

Usage: geometry_path.py RUN_DIR OUT_DIR [--cells 22:R0,37:T3] [--sweep-positions T3,R0]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from ptm.analysis import _import_plt, COLOR_NULL, HORIZON_CMAP, cell_name
from ptm.store import RunData

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("out_dir")
ap.add_argument("--cells", default="22:R0,37:T3,22:T3,29:R0")
ap.add_argument("--sweep-positions", default="T3,R0")
ap.add_argument("--no-sweep", action="store_true")
a = ap.parse_args()
run = RunData(a.run_dir); df = run.index; out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
has_h = df["horizon_years"].notna().to_numpy()
bins = df.loc[has_h].groupby("horizon_text")["horizon_years"].first().sort_values()
labels, years = bins.index.tolist(), bins.to_numpy()


def _path_geom(C):
    seg = np.diff(C, axis=0); seglen = np.linalg.norm(seg, axis=1)
    tort = seglen.sum() / np.linalg.norm(C[-1] - C[0])
    cosang = np.sum(seg[:-1] * seg[1:], 1) / (seglen[:-1] * seglen[1:])
    return seglen, float(tort), np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def path_stats(X):
    """X: [n, d] for horizon rows only (aligned with df[has_h]).

    Curvature quantities are computed in the top-3 PC space *of the 17 centroids* (where their
    variance lives), because in the full space centroid noise is comparable to segment length.
    Split-half reliability: centroids from odd vs even samples, projected on the same plane.
    """
    ht = df.loc[has_h, "horizon_text"].to_numpy()
    C = np.stack([X[ht == h].mean(0) for h in labels])                      # [17, d]
    pc = PCA(3).fit(C); ev = pc.explained_variance_ratio_
    # centroid scree (all 16 non-zero components) vs a split-half noise scree:
    # noise centroids = (C_odd - C_even)/2 have the same sampling variance as C's estimation error
    idx_all = np.arange(len(ht)); odd_m, even_m = idx_all % 2 == 1, idx_all % 2 == 0
    Co_full = np.stack([X[(ht == h) & odd_m].mean(0) for h in labels]); Ce_full = np.stack([X[(ht == h) & even_m].mean(0) for h in labels])
    full_eig = np.sort(np.linalg.svd(C - C.mean(0), compute_uv=False) ** 2)[::-1] / (len(C) - 1)
    noise_eig = np.sort(np.linalg.svd((Co_full - Ce_full) / 2 - ((Co_full - Ce_full) / 2).mean(0), compute_uv=False) ** 2)[::-1] / (len(C) - 1)
    n_above = int(np.sum(full_eig[:16] > 2 * noise_eig[:16]))   # component k is real if it beats 2x the noise at the same rank (max 16)
    scree = full_eig / full_eig.sum(); noise_scree = noise_eig / full_eig.sum()
    cum = np.cumsum(scree); n95 = int(np.searchsorted(cum, 0.95) + 1); n99 = int(np.searchsorted(cum, 0.99) + 1)
    C3 = pc.transform(C)
    seglen3, tort3, angles3 = _path_geom(C3)
    seglen_full, tort_full, _ = _path_geom(C)
    arclen = np.concatenate([[0], np.cumsum(seglen3)])
    decades = np.diff(np.log10(years))
    spacing_per_decade = seglen3 / decades
    idx = np.arange(len(ht)); odd, even = idx % 2 == 1, idx % 2 == 0
    Co = pc.transform(np.stack([X[(ht == h) & odd].mean(0) for h in labels]))
    Ce = pc.transform(np.stack([X[(ht == h) & even].mean(0) for h in labels]))
    half_r = float(np.corrcoef(Co[:, :2].ravel(), Ce[:, :2].ravel())[0, 1])
    half_seg_noise = float(np.median(np.linalg.norm(Co - Ce, axis=1)) / np.median(seglen3))
    within = np.array([np.linalg.norm(X[ht == h] - C[i], axis=1).mean() for i, h in enumerate(labels)])
    pr = []
    for h in labels:
        Xb = X[ht == h]; Xb = Xb - Xb.mean(0)
        lam = np.linalg.svd(Xb, compute_uv=False) ** 2 / max(len(Xb) - 1, 1)
        pr.append(float(lam.sum() ** 2 / (lam ** 2).sum()))
    return dict(C=C, scree=scree, noise_scree=noise_scree, n_dims_above_noise=n_above, n95=n95, n99=n99,
                arclen=arclen, seglen=seglen3, seglen_full=seglen_full, tortuosity=tort3, tortuosity_full=tort_full,
                angles=angles3, planarity=ev, spacing_per_decade=spacing_per_decade, half_r=half_r, half_seg_noise=half_seg_noise,
                within=within, snr=float(np.median(seglen3) / np.median(within)), pr=np.array(pr), pc=pc)


plt = _import_plt()
rows = []
for cell in a.cells.split(","):
    layer, pos = cell.split(":"); layer = int(layer)
    X = run.get(layer, pos)[has_h]
    st = path_stats(X)
    spd = st["spacing_per_decade"]; ends_vs_mid = float(np.mean([spd[0], spd[-1]]) / np.median(spd[3:-3]))
    print(f"=== L{layer} {pos} ===")
    print(f"  centroid planarity: top-1/2/3 PCs of the 17 centroids explain {st['planarity'].round(3).tolist()}")
    print(f"  centroid scree (fraction of centroid variance), ranks 1-8: {' '.join(f'{v:.3f}' for v in st['scree'][:8])}")
    print(f"  split-half noise scree at the same ranks:              {' '.join(f'{v:.3f}' for v in st['noise_scree'][:8])}")
    print(f"  cumulative: beyond PC2 {1 - st['scree'][:2].sum():.3f}, beyond PC3 {1 - st['scree'][:3].sum():.3f}; components for 95% / 99% of centroid variance: {st['n95']} / {st['n99']}; above 2x noise: {st['n_dims_above_noise']} of 16")
    print(f"  in centroid top-3 PC space: path length {st['arclen'][-1]:.1f}, tortuosity {st['tortuosity']:.2f} (full-space {st['tortuosity_full']:.2f}); turning angles median {np.median(st['angles']):.0f} deg, max {st['angles'].max():.0f}")
    print(f"  split-half: centroid-plane correlation r = {st['half_r']:.3f}; median odd/even centroid gap / median segment = {st['half_seg_noise']:.2f}")
    print(f"  spacing per decade of horizon: first {spd[0]:.1f}, last {spd[-1]:.1f}, middle median {np.median(spd[3:-3]):.1f}  (ends/middle {ends_vs_mid:.2f})")
    print(f"  median adjacent spacing / median within-bin spread (full space) = {st['snr']:.2f}")
    print("  bin: years, arclen(3D), seg-to-next, seg/decade, within-spread, participation-ratio")
    for i, h in enumerate(labels):
        nxt = f"{st['seglen'][i]:.1f}" if i < len(labels) - 1 else "  -"
        spdi = f"{spd[i]:.1f}" if i < len(labels) - 1 else "  -"
        print(f"    {h:>9}  {years[i]:9.4f}  {st['arclen'][i]:7.1f}  {nxt:>6}  {spdi:>6}  {st['within'][i]:6.1f}  {st['pr'][i]:6.1f}")
    sys.stdout.flush()
    rows.append(dict(layer=layer, position=pos, n95=st["n95"], n99=st["n99"], n_dims_above_noise=st["n_dims_above_noise"], scree=list(np.round(st["scree"][:8], 4)), noise_scree=list(np.round(st["noise_scree"][:8], 4)),
                     tortuosity3=st["tortuosity"], tortuosity_full=st["tortuosity_full"], planarity2=float(st["planarity"][:2].sum()),
                     split_half_r=st["half_r"], ends_over_middle_spacing=ends_vs_mid, snr=st["snr"], mean_pr=float(st["pr"].mean())))
    # figure: centroid path in centroid-PC space with all points projected, plus arc length vs log horizon
    pcs = st["pc"]; P = pcs.transform(X)[:, :2]; Cp = pcs.transform(st["C"])[:, :2]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[2]
    ranks = np.arange(1, 9)
    ax.plot(ranks, st["scree"][:8], "-o", color="#1b7f79", ms=5, label="centroid PCA")
    ax.plot(ranks, st["noise_scree"][:8], "-s", color="#9a9a9a", ms=4, label="split-half noise")
    ax.plot(ranks, 2 * st["noise_scree"][:8], "--", color="#9a9a9a", lw=1, label="2x noise")
    ax.set_yscale("log"); ax.set_xlabel("centroid PC rank"); ax.set_ylabel("fraction of centroid variance")
    ax.set_title(f"centroid scree   dims above 2x noise: {st['n_dims_above_noise']}", fontsize=9); ax.legend(frameon=False, fontsize=8)
    ax = axes[0]
    ax.scatter(P[:, 0], P[:, 1], s=6, c=np.log10(df.loc[has_h, "horizon_years"]), cmap=HORIZON_CMAP, alpha=0.35, linewidths=0)
    ax.plot(Cp[:, 0], Cp[:, 1], "-", color="black", lw=1.2)
    sc = ax.scatter(Cp[:, 0], Cp[:, 1], s=60, c=np.log10(years), cmap=HORIZON_CMAP, edgecolors="black", linewidths=0.8, zorder=3)
    for i in [0, len(labels) // 2, len(labels) - 1]:
        ax.annotate(labels[i], Cp[i], textcoords="offset points", xytext=(6, 4), fontsize=8)
    fig.colorbar(sc, ax=ax, pad=0.01).set_label("log10 horizon (years)")
    ax.set_xlabel("centroid PC1"); ax.set_ylabel("centroid PC2"); ax.set_title(f"bin-centroid path   tortuosity {st['tortuosity']:.2f}, planarity(2) {st['planarity'][:2].sum():.2f}", fontsize=9)
    ax = axes[1]
    mid = np.sqrt(years[:-1] * years[1:])
    ax.plot(mid, spd, "-o", color="#1b7f79", ms=5, lw=1.5)
    ax.set_xscale("log"); ax.set_xlabel("stated horizon (years), segment midpoint"); ax.set_ylabel("centroid spacing per decade of horizon")
    ax.set_title(f"spacing profile   ends/middle {ends_vs_mid:.2f}", fontsize=9)
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{run.meta['model_name']}  {cell_name(run, layer, pos)}", fontsize=10); fig.tight_layout()
    fig.savefig(out / f"path_L{layer:02d}_{pos}.png", dpi=150); plt.close(fig)
pd.DataFrame(rows).to_csv(out / "path_cells.csv", index=False)

# per-layer sweep at chosen positions
if a.no_sweep:
    raise SystemExit(0)
sw = []
sweep_pos = a.sweep_positions.split(",")
for layer in range(1, run.n_layers + 1):
    XL = run.get_layer(layer)
    for pos in sweep_pos:
        st = path_stats(XL[has_h, run.pos_index(pos)])
        sw.append(dict(position=pos, layer=layer, n95=st["n95"], n99=st["n99"], n_dims_above_noise=st["n_dims_above_noise"], scree3=float(st["scree"][2]), noise3=float(st["noise_scree"][2]),
                       tortuosity=st["tortuosity"], planarity2=float(st["planarity"][:2].sum()),
                       split_half_r=st["half_r"], snr=st["snr"], mean_pr=float(st["pr"].mean())))
    if layer % 10 == 0:
        print(f"sweep layer {layer} done"); sys.stdout.flush()
print("per-layer components for 95% of centroid variance (layers 1..L):")
for pos in sweep_pos:
    print(f"  {pos}: " + " ".join(str(r["n95"]) for r in sw if r["position"] == pos))
print("per-layer fraction of centroid variance beyond PC2:")
for pos in sweep_pos:
    print(f"  {pos}: " + " ".join(f"{1 - r['planarity2']:.2f}" for r in sw if r["position"] == pos))
sw = pd.DataFrame(sw); sw.to_csv(out / "path_sweep.csv", index=False)
fig, axes = plt.subplots(1, 5, figsize=(19, 3.6))
for pos, col in zip(a.sweep_positions.split(","), ["#1b7f79", "#d9730d", "#5b5bd6", "#9a9a9a"]):
    s = sw[sw.position == pos]
    axes[0].plot(s.layer, s.tortuosity, "-", color=col, label=pos); axes[1].plot(s.layer, s.planarity2, "-", color=col, label=pos)
    axes[2].plot(s.layer, s.split_half_r, "-", color=col, label=pos); axes[3].plot(s.layer, s.mean_pr, "-", color=col, label=pos)
    axes[4].plot(s.layer, s.n95, "-", color=col, label=pos)
for ax, t in zip(axes, ["tortuosity in centroid top-3 PCs", "planarity: var. of centroids in top-2 PCs", "split-half reliability of centroid plane", "within-bin participation ratio", "centroid PCs needed for 95% of variance"]):
    ax.set_title(t, fontsize=9); ax.set_xlabel("layer"); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
fig.suptitle(f"{run.meta['model_name']}: centroid-path geometry by layer", fontsize=10); fig.tight_layout()
fig.savefig(out / "path_sweep.png", dpi=150); plt.close(fig)
print("wrote", out)
