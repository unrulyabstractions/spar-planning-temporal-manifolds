"""How much of the horizon feature lives in the top-k principal components?

For every (mode, layer, position):
- full_r2: CV R^2 of the ridge probe on all d_model dims (the ceiling)
- r2[k]:   CV R^2 of the same probe fitted on the top-k PC coordinates
- ratio[k] = r2[k] / full_r2  (fraction of the readable horizon signal in k dims)
- cos[k]:  cosine between the full-space probe direction and its projection
           onto the top-k PC subspace (no refit; same question from the direction side)
- rho[k]:  Spearman |rho| between the k-dim probe's prediction and log horizon
- evr[k]:  cumulative explained variance of the top-k PCs (for contrast: this is
           about the activations, not the feature)

    python analysis/feature_dimensionality.py results/horizon_v1_noprefill
    -> <root>/analysis/feature_dimensionality.json and .png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spar_horizon.geometry import _fit, horizon_probe  # noqa: E402
from spar_horizon.runs import load_activations  # noqa: E402

KS = [1, 2, 3, 4, 5, 8, 12, 20, 40]


def cv_r2_and_rho(X, y, folds=5, seed=0):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.zeros_like(y)
    for tr, te in kf.split(X):
        pred[te] = _fit(X[tr], y[tr]).predict(X[te])
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(r2), float(abs(spearmanr(pred, y)[0]))


def analyze_cell(X, log_h):
    """X: [n, d] horizon prompts only. Returns dict of the metrics above."""
    Xc = X - X.mean(0)
    if np.allclose(Xc, 0):
        return None
    full_r2, d_full, _ = horizon_probe(Xc, log_h)
    kmax = min(max(KS), Xc.shape[0] - 1, Xc.shape[1])
    pca = PCA(n_components=kmax).fit(Xc)
    Z = pca.transform(Xc)
    comps = pca.components_                       # [kmax, d], orthonormal rows
    out = {"full_r2": full_r2, "k": [], "r2": [], "ratio": [], "cos": [], "rho": [], "evr": []}
    for k in KS:
        if k > kmax:
            break
        r2, rho = cv_r2_and_rho(Z[:, :k], log_h)
        r2 = max(r2, 0.0)   # a negative CV R^2 means no usable signal, not a negative share
        proj = comps[:k].T @ (comps[:k] @ d_full)
        out["k"].append(k)
        out["r2"].append(r2)
        out["ratio"].append(min(r2 / full_r2, 1.0) if full_r2 > 0.05 else np.nan)
        out["cos"].append(float(np.linalg.norm(proj)))   # d_full is unit length
        out["rho"].append(rho)
        out["evr"].append(float(pca.explained_variance_ratio_[:k].sum()))
    return out


def run(root, modes=("off", "on")):
    root = Path(root)
    results = {}
    for mode in modes:
        d = root / f"exp0_Qwen3-8B_think-{mode}"
        if not (d / "activations.npz").exists():
            continue
        z = load_activations(d / "activations.npz")
        has_h = ~np.isnan(z["horizons"])
        log_h = np.log10(z["horizons"][has_h])
        cells = []
        for l, A in enumerate(z["activations"]):
            row = []
            for p in range(A.shape[1]):
                row.append(analyze_cell(A[has_h, p], log_h))
            cells.append(row)
            got = [c for c in row if c]
            if got:
                r3 = np.nanmean([c["ratio"][c["k"].index(3)] for c in got if 3 in c["k"]])
                print(f"{mode} L{l:<3} full R2 {np.mean([c['full_r2'] for c in got]):.3f}  "
                      f"ratio@k=3 {r3:.3f}  (mean over positions)")
        results[mode] = {"tokens": z["tokens"], "n_suffix": z["n_suffix"], "cells": cells}
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    (out / "feature_dimensionality.json").write_text(json.dumps(results))
    plot(results, out / "feature_dimensionality.png")
    print(f"wrote {out / 'feature_dimensionality.json'}")
    return results


def plot(results, path):
    modes = list(results)
    fig, axes = plt.subplots(2, len(modes), figsize=(6.5 * len(modes), 8.5), squeeze=False,
                             layout="constrained")
    for j, mode in enumerate(modes):
        r = results[mode]
        n_layers, n_pos = len(r["cells"]), len(r["tokens"])
        M = np.full((n_layers, n_pos), np.nan)
        for l in range(n_layers):
            for p in range(n_pos):
                c = r["cells"][l][p]
                if c and 3 in c["k"]:
                    M[l, p] = c["ratio"][c["k"].index(3)]
        ax = axes[0, j]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(n_pos), [repr(t) for t in r["tokens"]], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("layer")
        ax.set_title(f"thinking {mode}: fraction of horizon signal in top-3 PCs", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax = axes[1, j]
        pos = min(1, n_pos - 1)
        for l in (1, 6, 11, 16, 21, 26, 31, 36):
            if l >= n_layers or not r["cells"][l][pos]:
                continue
            c = r["cells"][l][pos]
            ax.plot(c["k"], c["ratio"], marker="o", ms=3, label=f"L{l}")
        c = r["cells"][min(11, n_layers - 1)][pos]
        if c:
            ax.plot(c["k"], c["evr"], ls="--", color="gray", label="cum. explained var. (L11)")
        ax.set_xscale("log")
        ax.set_xlabel("k = number of top PCs")
        ax.set_ylabel("horizon R² in k dims / full R²")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"thinking {mode}, token {r['tokens'][pos]!r}: signal captured vs k", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "results/horizon")
