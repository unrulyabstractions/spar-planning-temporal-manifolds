"""Figures of the phrase number-patch experiment (27_spec_phrase_number_patch.md, results in 30_*).

    python -m pipeline.s3_analysis.figures.patch_figures --patch-dir ../results/horizon/exp16_Qwen3-8B_patch \\
        --phrase-dir ../results/horizon/exp14_Qwen3-8B_phrase

Writes <patch-dir>/figures/{patch_where_and_layer,patch_choice_shift,manifold_curve,manifold_all_pcs}.png. CPU only.
The manifold figure refits the pooled L16 number fit exactly as patch_prepare.patch_main does.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.colors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INK, MUTED, GRID = "#1f2328", "#6e7781", "#d8dee4"
BLUE, ORANGE, GREY = "#1f6feb", "#d1570d", "#8c959f"      # patch, control, no effect
LAYERS = (12, 16, 22)
CMAP = matplotlib.colormaps["Blues"].resampled(256)
CMAP = matplotlib.colors.ListedColormap(CMAP(np.linspace(0.3, 1, 256)))   # no near-white end on a white page

# (row label, unit file stem by layer, group). Each file is a conditions/<stem>.json.
ROWS = [
    ("no patch (rerun, other batch order)", {16: "rerun"}, "controls"),
    ("own states put back (self patch)", {16: "self_phrase_L16"}, "controls"),
    ("whole phrase", {16: "donor_phrase_L16"}, "swap in the donor's states"),
    ("number tokens", {12: "donor_number_L12", 16: "donor_number_L16", 22: "donor_number_L22"}, "swap in the donor's states"),
    ("unit token", {16: "donor_unit_L16"}, "swap in the donor's states"),
    ("number tokens, top-8 subspace only", {16: "subspace_number_L16"}, "swap in the donor's states"),
    ("manifold step, number tokens", {16: "manifold_number_L16"}, "add a step"),
    ("manifold step, number + unit", {16: "manifold_phrase_L16"}, "add a step"),
    ("random step, any direction", {16: "random_number_L16"}, "add a step"),
    ("random step, inside the 8-d subspace", {16: "inspan_number_L16"}, "add a step"),
]


def load_rows(patch_dir, stem):
    return json.loads((Path(patch_dir) / "conditions" / f"{stem}.json").read_text())["rows"]


def summary_counts(patch_dir):
    """{unit name: unit summary} from patch_summary.json."""
    summary = json.loads((Path(patch_dir) / "patch_summary.json").read_text())
    return {u["name"]: u for u in summary["units"]}, summary["noise"]


def stem_to_name(stem):
    """patch_summary names units 'donor number L16', files are donor_number_L16."""
    return stem.replace("_", " ")


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def figure_where_and_layer(patch_dir, out):
    units, noise = summary_counts(patch_dir)
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    n_rows = len(ROWS)
    ax.set_xlim(-0.5, len(LAYERS) - 0.5)
    ax.set_ylim(n_rows - 0.4 + 0.4, -0.6)
    for r, (label, stems, group) in enumerate(ROWS):
        for c, layer in enumerate(LAYERS):
            stem = stems.get(layer)
            if stem is None:
                ax.add_patch(plt.Rectangle((c - 0.46, r - 0.42), 0.92, 0.84, fill=False, ec=GRID, lw=0.8, ls=(0, (2, 3))))
                continue
            if stem == "rerun":
                hit, n = noise["lands_target"], noise["n"]
            else:
                u = units[stem_to_name(stem)]
                hit, n = u["lands_target"], u["n"]
            share = hit / n
            colour = ORANGE if "random" in label else BLUE
            ax.add_patch(plt.Rectangle((c - 0.46, r - 0.42), 0.92, 0.84, fc=colour, ec="none", alpha=0.08 + 0.92 * share))
            ax.text(c, r, f"{hit}/{n}", ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if share > 0.55 else INK)
    ax.set_yticks(range(n_rows), [row[0] for row in ROWS], fontsize=10, color=INK)
    ax.set_xticks(range(len(LAYERS)), [f"layer {layer}" for layer in LAYERS], fontsize=10, color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    group_starts = {}
    for r, (_, _, group) in enumerate(ROWS):
        group_starts.setdefault(group, r)
    for group, r in group_starts.items():
        ax.axhline(r - 0.5, color=GRID, lw=1)
        ax.text(len(LAYERS) - 0.45, r - 0.42, group, ha="right", va="bottom", fontsize=8.5, color=MUTED, style="italic")
    fig.suptitle("Where the horizon number is carried: traces that land in the target regime (of 24 pairs)",
                 x=0.02, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.02, 0.925, "Dashed cell = not run. Shade = share of 24 traces; blue = patch, orange = random control.", fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def paired_shift(baseline, rows):
    """Per pair, near log-odds minus the baseline's, for pairs where neither trace was cut off."""
    keep = [i for i, (a, b) in enumerate(zip(baseline, rows)) if not a["forced"] and not b["forced"]]
    return np.array([rows[i]["near_lo"] - baseline[i]["near_lo"] for i in keep])


def figure_choice_shift(patch_dir, out):
    baseline = load_rows(patch_dir, "baseline")
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    n_rows = len(ROWS)
    rng = np.random.default_rng(0)
    for r, (label, stems, _) in enumerate(ROWS):
        stem = stems[16]
        shifts = paired_shift(baseline, load_rows(patch_dir, stem))
        colour = ORANGE if "random" in label else (GREY if stem in ("rerun", "self_phrase_L16") else BLUE)
        ax.scatter(shifts, r + rng.uniform(-0.18, 0.18, len(shifts)), s=22, color=colour, alpha=0.75, lw=0)
        ax.plot([np.median(shifts)] * 2, [r - 0.3, r + 0.3], color=INK, lw=2)
        ax.text(1.01, r, f"n={len(shifts)}", transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=MUTED)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.set_yticks(range(n_rows), [row[0] for row in ROWS], fontsize=10, color=INK)
    ax.set_ylim(n_rows - 0.4, -0.6)
    ax.set_xlabel("change in near log-odds against the unpatched baseline, per pair   (black bar = median)",
                  fontsize=9.5, color=MUTED)
    style(ax)
    ax.tick_params(axis="y", length=0)
    fig.suptitle("Change in the choice's near log-odds, per pair, by what was patched (layer 16)",
                 x=0.02, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.02, 0.925, "Pairs where neither trace was cut off. The rerun row is the noise floor; the spread is large, so read medians as a screen.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def load_manifold(patch_dir, phrase_dir):
    """(inputs meta, the pooled L16 number fit, every prompt's coordinates on its basis, log10 horizons):
    the fit patch_prepare.patch_main makes."""
    from pipeline.s1_data_vectors.analysis.scan_prepare import load_phrase_states
    from pipeline.s1_data_vectors.analysis.steering_prepare import OUT, fit_phrase_cell, phrase_columns, varying_factors
    from pipeline.s2_steering.experiments.ablate_steer.exp8_steering import selected_records

    meta = json.loads((Path(patch_dir) / "inputs.json").read_text())
    states, uids = load_phrase_states(Path(phrase_dir), (16,))
    prompts_path = Path(meta["prompts_file"])
    if not prompts_path.exists():
        prompts_path = OUT / prompts_path.name
    records = selected_records(json.loads(prompts_path.read_text()))[:len(uids)]
    columns = phrase_columns(records)
    fit = fit_phrase_cell(states["L16|number"], columns, varying_factors(columns))
    raw = (states["L16|number"] - fit["mean"]) @ fit["basis"].T
    return meta, fit, raw, columns["log_h"]


def figure_manifold(patch_dir, phrase_dir, out):
    meta, fit, raw, log_h = load_manifold(patch_dir, phrase_dir)
    grid, coords = fit["curve_logh"], fit["curve_coords"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    ax = axes[0]
    ax.scatter(raw[:, 0], raw[:, 1], c=log_h, cmap=CMAP, s=7, alpha=0.35, lw=0, vmin=grid[0], vmax=grid[-1])
    ax.plot(coords[:, 0], coords[:, 1], color=INK, lw=1.2, zorder=3)
    points = ax.scatter(coords[:, 0], coords[:, 1], c=grid, cmap=CMAP, s=46, ec=INK, lw=0.8, zorder=4,
                        vmin=grid[0], vmax=grid[-1])
    # the 24 donor steps are chords of this curve: own horizon -> donor horizon
    for p in meta["prompts"]:
        a = [np.interp(p["log_h"], grid, coords[:, j]) for j in (0, 1)]
        b = [np.interp(p["donor_log_h"], grid, coords[:, j]) for j in (0, 1)]
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1, alpha=0.8), zorder=5)
    for target in (-1.5, 0, 1.5, 3):
        k = int(np.argmin(abs(grid - target)))
        ax.annotate(f"{10 ** grid[k]:.3g} yr", coords[k, :2], textcoords="offset points", xytext=(7, 8), fontsize=8.5, color=INK,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1), zorder=6)
    ax.set_xlabel("PC 1 of the 48 horizon averages", fontsize=9.5, color=MUTED)
    ax.set_ylabel("PC 2", fontsize=9.5, color=MUTED)
    ax.set_title("Layer 16 number-token states in the top-2 manifold directions", fontsize=10.5, color=INK, loc="left")
    cbar = fig.colorbar(points, ax=ax, pad=0.01, fraction=0.04)
    cbar.set_label("log10 horizon (years)", fontsize=8.5, color=MUTED)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    ax.text(0.02, 0.02, "pale dots: individual prompts;  black line: horizon averages\n"
            "orange arrows: the 24 pair steps (own horizon -> donor), drawn in 2 of the 8 dimensions", transform=ax.transAxes, fontsize=8, color=MUTED)
    style(ax)

    for j, ax in zip((0, 1), axes[1:]):
        ax.scatter(log_h, raw[:, j], s=7, color=GREY, alpha=0.25, lw=0)
        ax.plot(grid, coords[:, j], color=BLUE, lw=2)
        ax.set_xlabel("log10 horizon (years)", fontsize=9.5, color=MUTED)
        ax.set_ylabel(f"coordinate on PC {j + 1}", fontsize=9.5, color=MUTED)
        ax.set_title(f"PC {j + 1} against the written horizon", fontsize=10.5, color=INK, loc="left")
        style(ax)
    fig.suptitle("The manifold being steered: a curve through the per-horizon average states (layer 16, number tokens)",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    explained = np.var(coords - coords.mean(0), axis=0)
    print(f"curve points {len(grid)}, spread on PC1..PC3: {np.round(explained[:3] / explained.sum(), 2)}")


def smooth_share(grid, values, degree=3):
    """Share of the variance of a curve's coordinate that a cubic in log horizon explains."""
    fitted = np.polyval(np.polyfit(grid, values, degree), grid)
    return 1 - np.var(values - fitted) / np.var(values)


def figure_manifold_pcs(patch_dir, phrase_dir, out):
    """All eight manifold directions against the written horizon, with how much of each is a smooth trend."""
    meta, fit, raw, log_h = load_manifold(patch_dir, phrase_dir)
    grid, coords = fit["curve_logh"], fit["curve_coords"]
    spread = np.var(coords, axis=0) / np.var(coords, axis=0).sum()
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharex=True)
    for j, ax in enumerate(axes.ravel()):
        ax.scatter(log_h, raw[:, j], s=5, color=GREY, alpha=0.22, lw=0)
        ax.plot(grid, coords[:, j], color=BLUE, lw=1.8)
        ax.set_title(f"PC {j + 1}: {100 * spread[j]:.0f}% of curve spread, "
                     f"cubic trend R² {smooth_share(grid, coords[:, j]):.2f}", fontsize=9.5, color=INK, loc="left")
        style(ax)
        if j >= 4:
            ax.set_xlabel("log10 horizon (years)", fontsize=9, color=MUTED)
    fig.suptitle("All 8 manifold directions against the written horizon (layer 16, number tokens)",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.01, 0.925, "blue line: fitted state at each of 48 horizons, unit / regime / delay effects held at their average;  "
             "grey dots: every prompt's raw state, those effects left in", fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("cubic-trend R² per PC:", [round(float(smooth_share(grid, coords[:, j])), 2) for j in range(coords.shape[1])])


def main(args):
    patch_dir = Path(args.patch_dir)
    figures = patch_dir / "figures"
    figures.mkdir(exist_ok=True)
    figure_where_and_layer(patch_dir, figures / "patch_where_and_layer.png")
    figure_choice_shift(patch_dir, figures / "patch_choice_shift.png")
    figure_manifold(patch_dir, args.phrase_dir, figures / "manifold_curve.png")
    figure_manifold_pcs(patch_dir, args.phrase_dir, figures / "manifold_all_pcs.png")
    print(f"wrote 4 figures to {figures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--phrase-dir", required=True)
    main(parser.parse_args())
