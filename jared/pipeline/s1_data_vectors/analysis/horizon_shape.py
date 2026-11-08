"""The shape of the horizon representation at a few key cells.

For each cell, the activations are averaged per horizon ("horizon averages")
and a PCA is fitted to those averages, as in the probes' horizon-mean
ordering. Three panels per cell:
- left: every prompt's position along the first component against its log
  horizon, with the 1-input MLP and a straight line fitted to it;
- middle: the plane of the first two components, the 2-input MLP's
  prediction as filled contours, and the path of horizon averages;
- right: the path of horizon averages in the first three components.
Thinking-on prompts whose reasoning was force-closed are drawn as crosses.
The probes drawn here are fitted on all prompts, for display only; the
held-out R² printed in each title comes from probe_dimensionality.json.

    python analysis/horizon_shape.py results/horizon
    -> <root>/analysis/horizon_shape.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from probe_dimensionality import positions, run_file  # noqa: E402  (same folder)
from spar_horizon.probes import fit_predict  # noqa: E402
from spar_horizon.runs import REGIONS  # noqa: E402

CELLS = (("on", "suffix newline", 16), ("off", "pre-label", 21), ("on", "pre-label", 21), ("on", "think close", 19))
LINEAR, MLP, PROMPT, PATH = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b"
CMAP = "Blues"
TICKS = {"1 min": 1 / 525_960, "1 h": 1 / 8_766, "1 day": 1 / 365.25, "1 mo": 1 / 12, "1 y": 1, "10 y": 10, "100 y": 100}


def cell_data(z, label, layer):
    """X [n, d] at the cell, log10 horizon y [n], forced [n] (all False with thinking off)."""
    meta = {"regions": {r: tuple(int(x) for x in z[f"region_{r}"]) for r in REGIONS},
            "tokens": [str(t) for t in z["tokens"]]}
    keep = ~np.isnan(z["horizons"])
    X = z[f"L{layer}"][keep][:, positions(meta)[label]].astype(np.float64)
    y = np.log10(z["horizons"][keep])
    forced = z["forced_mask"][keep].astype(bool) if z["forced_mask"].size else np.zeros(len(y), bool)
    return X, y, forced


def horizon_components(X, y, k=3):
    """Prompt coordinates Z [n, k] and horizon-average coordinates P [h, k] on the
    top-k PCs of the horizon averages, the sorted horizons, and the averages'
    explained variance. Component 1 is oriented to grow with the horizon."""
    groups = np.round(y, 9)
    horizons = np.unique(groups)
    mean = X.mean(0)
    averages = np.stack([X[groups == h].mean(0) for h in horizons]) - mean
    pca = PCA(n_components=k).fit(averages)
    Z, P = (X - mean) @ pca.components_.T, averages @ pca.components_.T
    sign = np.sign(spearmanr(P[:, 0], horizons)[0]) or 1.0
    Z[:, 0] *= sign
    P[:, 0] *= sign
    return Z, P, horizons, pca.explained_variance_ratio_


def held_out(results, mode, label, layer, k):
    """'linear a, MLP b' held-out interior R² on the horizon-mean ordering at k, or ''."""
    if results is None:
        return ""
    c = results[mode]["cells"][label][layer]
    i = results[mode]["ks"].index(k)
    return f"held-out R² linear {c['r2_interior']['linear']['cent'][i]:.2f}, MLP {c['r2_interior']['mlp']['cent'][i]:.2f}"


def horizon_colorbar(fig, mappable, ax):
    bar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    bar.set_ticks([np.log10(v) for v in TICKS.values()], labels=list(TICKS))
    bar.set_label("horizon", fontsize=8)
    bar.ax.tick_params(labelsize=7)


def draw_prompts(ax, x, y, forced):
    ax.scatter(x[~forced], y[~forced], s=6, color=PROMPT, alpha=0.5, linewidths=0, label="prompt")
    if forced.any():
        ax.scatter(x[forced], y[forced], s=10, color=PROMPT, alpha=0.8, marker="x", linewidths=0.8,
                   label="prompt, reasoning force-closed")


def plot_cell(fig, axes, mode, label, layer, X, y, forced, results):
    Z, P, horizons, evr = horizon_components(X, y)
    title = f"{mode}: {label}, L{layer}"
    norm = plt.Normalize(y.min(), y.max())

    ax = axes[0]   # one input: position along component 1 -> log horizon
    draw_prompts(ax, Z[:, 0], y, forced)
    grid = np.linspace(Z[:, 0].min(), Z[:, 0].max(), 300)[:, None]
    ax.plot(grid[:, 0], fit_predict("linear", Z[:, :1], y, grid), color=LINEAR, lw=2, label="linear probe")
    ax.plot(grid[:, 0], fit_predict("mlp", Z[:, :1], y, grid), color=MLP, lw=2, label="MLP probe")
    ax.plot(P[:, 0], horizons, "o", color=PATH, ms=3, label="horizon average")
    ax.set_yticks([np.log10(v) for v in TICKS.values()], labels=list(TICKS), fontsize=7)
    ax.set_xlabel("position along component 1", fontsize=8)
    ax.set_title(f"{title}\n1 component: {held_out(results, mode, label, layer, 1)}", fontsize=8)

    ax = axes[1]   # two inputs: the MLP's prediction over the plane, with the path of averages
    pad = 0.08 * (Z[:, :2].max(0) - Z[:, :2].min(0))
    g0 = np.linspace(Z[:, 0].min() - pad[0], Z[:, 0].max() + pad[0], 160)
    g1 = np.linspace(Z[:, 1].min() - pad[1], Z[:, 1].max() + pad[1], 160)
    G0, G1 = np.meshgrid(g0, g1)
    surface = fit_predict("mlp", Z[:, :2], y, np.c_[G0.ravel(), G1.ravel()]).reshape(G0.shape)
    fill = ax.contourf(G0, G1, np.clip(surface, y.min(), y.max()), levels=14, cmap=CMAP, norm=norm, alpha=0.55)
    ax.contour(G0, G1, surface, levels=14, colors="white", linewidths=0.5, alpha=0.7)
    draw_prompts(ax, Z[:, 0], Z[:, 1], forced)
    ax.plot(P[:, 0], P[:, 1], "-", color=PATH, lw=1)
    ax.scatter(P[:, 0], P[:, 1], c=horizons, cmap=CMAP, norm=norm, s=22, edgecolors=PATH, linewidths=0.6, zorder=3)
    ax.set_xlabel(f"component 1 ({evr[0]:.0%} of average variance)", fontsize=8)
    ax.set_ylabel(f"component 2 ({evr[1]:.0%})", fontsize=8)
    ax.set_title(f"MLP prediction over components 1-2\n2 components: {held_out(results, mode, label, layer, 2)}",
                 fontsize=8)
    horizon_colorbar(fig, fill, ax)

    ax = axes[2]   # the path of horizon averages in three components
    ax.plot(P[:, 0], P[:, 1], P[:, 2], "-", color=PATH, lw=1)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=horizons, cmap=CMAP, norm=norm, s=18, edgecolors=PATH,
               linewidths=0.5, depthshade=False)
    ax.set_xlabel("comp 1", fontsize=7)
    ax.set_ylabel("comp 2", fontsize=7)
    ax.set_zlabel(f"comp 3 ({evr[2]:.0%})", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("path of the 48 horizon averages", fontsize=8)


def main(root, model="Qwen/Qwen3-8B"):
    root = Path(root)
    summary = root / "analysis" / "probe_dimensionality.json"
    results = json.loads(summary.read_text()) if summary.exists() else None
    fig = plt.figure(figsize=(16, 4.6 * len(CELLS)))
    for row, (mode, label, layer) in enumerate(CELLS):
        z = np.load(run_file(root, mode, model))
        X, y, forced = cell_data(z, label, layer)
        axes = [fig.add_subplot(len(CELLS), 3, 3 * row + 1), fig.add_subplot(len(CELLS), 3, 3 * row + 2),
                fig.add_subplot(len(CELLS), 3, 3 * row + 3, projection="3d")]
        plot_cell(fig, axes, mode, label, layer, X, y, forced, results)
        print(f"{mode} {label} L{layer}: {len(y)} prompts, {int(forced.sum())} force-closed")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    for ax in fig.axes[3:]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Shape of the log-horizon representation, full bank (probes fitted on all prompts, for display)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    out = root / "analysis" / "horizon_shape.png"
    fig.savefig(out, dpi=115)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    args = parser.parse_args()
    main(args.root, args.model)
