"""Figures: 2D and 3D PCA panels per position, heatmaps."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


def _region(pos, n_suffix):
    if pos == n_suffix:
        return "response, answer"
    return "turn" if pos < n_suffix else "response"

# Named durations for the horizon colorbar, in years (matches prompts.HORIZONS).
_TICKS = [("30 s", 30 / 31_557_600), ("1 hr", 3600 / 31_557_600), ("1 day", 1 / 365),
          ("1 mo", 1 / 12), ("1 yr", 1), ("10 yr", 10), ("100 yr", 100), ("500 yr", 500)]


class ColorSpec:
    """How to color the horizon points: `values` per horizon prompt, a colormap,
    a norm, and (tick position, label) pairs for the colorbar."""

    def __init__(self, values, cmap, norm, ticks, label):
        self.values, self.cmap, self.norm, self.ticks, self.label = values, cmap, norm, ticks, label

    def scatter_kw(self):
        return {"c": self.values, "cmap": self.cmap, "norm": self.norm}


def horizon_spec(log_horizons):
    """Sequential turbo over log10 horizon (years), ticks at named durations."""
    lo, hi = float(np.min(log_horizons)), float(np.max(log_horizons))
    ticks = [(np.log10(v), lab) for lab, v in _TICKS if lo - 0.05 <= np.log10(v) <= hi + 0.05]
    return ColorSpec(log_horizons, "turbo", plt.Normalize(lo, hi), ticks, "time horizon")


def ratio_spec(log_horizons, far_delay_years):
    """Diverging coolwarm over log10(horizon / far option's delay), centered on 0:
    blue = far payout arrives after the horizon, red = in time."""
    r = np.asarray(log_horizons) - np.log10(np.asarray(far_delay_years))
    lo, hi = min(float(r.min()), -1e-6), max(float(r.max()), 1e-6)
    ticks = [(k, f"x{10.0 ** k:g}") for k in range(int(np.ceil(lo)), int(np.floor(hi)) + 1)]
    return ColorSpec(r, "coolwarm", matplotlib.colors.TwoSlopeNorm(0, lo, hi), ticks,
                     "horizon / far delay")


def _colorbar(fig, axes, spec):
    sm = plt.cm.ScalarMappable(cmap=spec.cmap, norm=spec.norm)
    sm.set_array(np.asarray(spec.values))
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cb.set_ticks([t for t, _ in spec.ticks])
    cb.set_ticklabels([lab for _, lab in spec.ticks])
    cb.set_label(spec.label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return cb


def pca_panels(layer, rho, Z_all, tokens, has_horizon, log_horizons, n_suffix, path, title,
               spec=None):
    """One 2D scatter per position at `layer`, colored by `spec` (default: log horizon)."""
    spec = spec or horizon_spec(log_horizons)
    n_positions = len(tokens)
    cols = 3
    rows = -(-n_positions // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols + 0.8, 2.9 * rows),
                             layout="constrained")
    for pos in range(rows * cols):
        ax = axes.flat[pos]
        if pos >= n_positions or Z_all[layer][pos] is None:
            ax.axis("off")
            continue
        Z = Z_all[layer][pos]
        ax.scatter(Z[has_horizon, 0], Z[has_horizon, 1], s=14, **spec.scatter_kw())
        ax.scatter(Z[~has_horizon, 0], Z[~has_horizon, 1], c="#8c8c8c", s=14)
        ax.set_title(f"{tokens[pos]!r} ({_region(pos, n_suffix)})  |rho|={rho[layer, pos]:.2f}",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    _colorbar(fig, axes.ravel().tolist(), spec)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def pca3_stats(activations, layer, has_horizon, log_horizons):
    """Per position at `layer`: explained variance of PC1-3 and |rho| of each
    PC with log horizon. Returns (evr [pos, 3], rho [pos, 3], Z3 list)."""
    X = activations[layer]
    n_positions = X.shape[1]
    evr = np.full((n_positions, 3), np.nan)
    rho3 = np.full((n_positions, 3), np.nan)
    Z3 = [None] * n_positions
    for pos in range(n_positions):
        X_pos = X[:, pos] - X[:, pos].mean(0)
        if np.allclose(X_pos, 0):
            continue
        pca = PCA(n_components=3).fit(X_pos)
        Z = pca.transform(X_pos)
        evr[pos] = pca.explained_variance_ratio_
        for k in range(3):
            rho3[pos, k] = abs(spearmanr(Z[has_horizon, k], log_horizons)[0])
        Z3[pos] = Z
    return evr, rho3, Z3


def pca3_panels(layer, Z3, evr, rho3, tokens, has_horizon, log_horizons, n_suffix, path, title,
                spec=None):
    """One 3D scatter per position at `layer`. Titles carry explained variance
    and the three Spearman values so a curve vs a line can be read off."""
    spec = spec or horizon_spec(log_horizons)
    n_positions = len(tokens)
    cols = 3
    rows = -(-n_positions // cols)
    fig = plt.figure(figsize=(3.8 * cols + 0.8, 3.4 * rows), layout="constrained")
    axes = []
    for pos in range(n_positions):
        ax = fig.add_subplot(rows, cols, pos + 1, projection="3d")
        axes.append(ax)
        if Z3[pos] is None:
            ax.axis("off")
            continue
        Z = Z3[pos]
        ax.scatter(Z[has_horizon, 0], Z[has_horizon, 1], Z[has_horizon, 2], s=10,
                   **spec.scatter_kw())
        ax.scatter(Z[~has_horizon, 0], Z[~has_horizon, 1], Z[~has_horizon, 2], c="#8c8c8c", s=10)
        ax.set_title(f"{tokens[pos]!r} ({_region(pos, n_suffix)})\n"
                     f"evr {evr[pos, 0]:.2f}/{evr[pos, 1]:.2f}/{evr[pos, 2]:.2f}  "
                     f"rho {rho3[pos, 0]:.2f}/{rho3[pos, 1]:.2f}/{rho3[pos, 2]:.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    _colorbar(fig, axes, spec)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def heatmap(M, row_labels, col_labels, path, title, vmin=0, vmax=1, fmt="{:.2f}", annotate=True):
    fig, ax = plt.subplots(figsize=(0.55 * len(col_labels) + 3, 0.28 * len(row_labels) + 2))
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    for i in range(M.shape[0] if annotate else 0):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < (vmin + vmax) / 2 else "black")
    fig.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
