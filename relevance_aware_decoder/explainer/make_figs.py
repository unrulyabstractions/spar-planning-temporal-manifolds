"""Figures for explainer/index.html (relevance-aware decoder, Qwen3-14B run).

Reads only ../results/qwen3-14b_relevance_s0/ and ../data/prompts_s0.parquet (both in git or regenerable),
writes PNGs to figs/. figs/pull_metrics.png is a static illustration and is not regenerated here.
Run from anywhere:  python relevance_aware_decoder/explainer/make_figs.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results/qwen3-14b_relevance_s0"
PROMPTS = HERE.parent / "data/prompts_s0.parquet"
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)

# page tokens (light plate) + series colors (orange = baseline; ordinal blue ramp A -> B -> C)
INK, MUTED, RULE = "#1A242D", "#56636E", "#D6DDE2"
PRE = "#eb6834"
TIER = {"A": "#86b6ef", "B": "#2a78d6", "C": "#104281"}
BLUE = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
ORANGE = LinearSegmentedColormap.from_list("orange", ["#fdf5f0", "#fbd9c8", "#f3a27f", "#eb6834", "#b8461b", "#7a2c0e"])
FAMS = ["entity_age", "prep_time", "tenure", "background_event", "other_horizon"]
FAM_LABEL = {"entity_age": "entity age", "prep_time": "prep time", "tenure": "tenure",
             "background_event": "background\nevent", "other_horizon": "other\nhorizon"}
LAYERS = [14, 18, 22, 26, 29, 33, 37]
POS = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "R0"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": RULE, "axes.linewidth": 1, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 160, "savefig.bbox": "tight",
})


def vdot(ax, x, y, lo, hi, color):
    ax.plot([x, x], [lo, hi], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter([x], [y], s=46, color=color, edgecolor="white", linewidth=1.6, zorder=3)


# 1 · behavior: does the model's choice move toward D? -----------------------------------------------------
beh = pd.read_csv(RES / "behavior_gate.csv").set_index("family").loc[FAMS]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.axvline(0, color=MUTED, lw=1, zorder=1)
for i, f in enumerate(FAMS):
    r = beh.loc[f]
    ax.plot([r.behavior_pull_lo, r.behavior_pull_hi], [i, i], color="#2a78d6", lw=2, solid_capstyle="round", zorder=2)
    ax.scatter([r.behavior_pull], [i], s=46, color="#2a78d6", edgecolor="white", linewidth=1.6, zorder=3)
    ax.text(0.155, i, f"{r.behavior_pull:+.3f}", va="center", ha="right", color=INK, fontsize=10)
ax.set_yticks(range(len(FAMS)), [FAM_LABEL[f].replace("\n", " ") for f in FAMS])
ax.set_xlim(-0.16, 0.16); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("behavior pull (fraction of the way the model's choice moves toward D)")
fig.savefig(OUT / "behavior_pull.png"); plt.close(fig)

# 2 · pull by tier at the two headline cells --------------------------------------------------------------
h = pd.read_csv(RES / "headline_cells.csv")
tiers = [("pre-C", "baseline decoder (all four templates)", PRE), ("A", "A · trained sentences", TIER["A"]),
         ("B", "B · new wording", TIER["B"]), ("C", "C · family left out of training", TIER["C"])]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, (cell, title) in zip(axes, [("L26:T1", "Layer 26, token T1 (Alan's cell)"),
                                    ("L18:T1", "Layer 18, token T1 (lowest dev error)")]):
    t = h[h.cell == cell]
    ax.axhline(0, color=MUTED, lw=1)
    for i, f in enumerate(FAMS):
        for j, (tier, _, col) in enumerate(tiers):
            r = t[(t.tier == tier) & (t.family == f)].iloc[0]
            vdot(ax, i + (j - 1.5) * 0.17, r.pull, r.pull_lo, r.pull_hi, col)
    ax.set_xticks(range(len(FAMS)), [FAM_LABEL[f] for f in FAMS]); ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=10)
    ax.set_ylim(-0.06, 0.32)
axes[0].set_ylabel("pull toward D (0 = ignores it, 1 = follows it)")
handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=8, markeredgecolor="white") for _, _, c in tiers]
fig.legend(handles, [lab for _, lab, _ in tiers], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.savefig(OUT / "pull_by_tier.png"); plt.close(fig)

# 3 · sweep: pull and accuracy lost over layer x position -------------------------------------------------
s = pd.read_csv(RES / "sweep.csv")
s["drop"] = s.within2x_twin - s.within2x
fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), sharex=True, sharey=True)
titles = {"pre-C": "baseline", "A": "A · trained sentences", "B": "B · new wording", "C": "C · family left out"}
for row, (metric, cmap, vmax, label) in enumerate([("pull", BLUE, 0.35, "pull toward D"),
                                                   ("drop", ORANGE, 0.70, "accuracy lost (within-2×, twin − distractor)")]):
    for col, tier in enumerate(["pre-C", "A", "B", "C"]):
        ax = axes[row, col]
        g = s[s.tier == tier].groupby("cell")[metric].mean()
        M = np.array([[g.get(f"L{l}:{p}", np.nan) for p in POS] for l in LAYERS])
        im = ax.imshow(M, vmin=0, vmax=vmax, cmap=cmap, aspect="auto", origin="lower")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for (l, p) in [(26, "T1"), (18, "T1")]:
            ax.scatter([POS.index(p)], [LAYERS.index(l)], s=60, facecolor="none", edgecolor=INK, linewidth=1.4)
        if row == 0:
            ax.set_title(titles[tier], loc="left", fontsize=11, color=INK)
        ax.set_xticks(range(len(POS)), POS, fontsize=8.5); ax.set_yticks(range(len(LAYERS)), LAYERS)
        if col == 0:
            ax.set_ylabel("layer")
    cb = fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.9, pad=0.015)
    cb.set_label(label, color=INK); cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED)
fig.savefig(OUT / "sweep.png"); plt.close(fig)

# 4 · one family up close: pull gone, disturbance not ------------------------------------------------------
P = np.load(RES / "predictions.npz")
idx = pd.read_parquet(RES / "index.parquet")[["sample_uid", "condition", "split"]]
d = pd.read_parquet(PROMPTS)[["sample_uid", "family", "template_group", "twin_uid", "log_gap"]]
df = idx.merge(d, on="sample_uid")
assert (P["sample_uid"] == df.sample_uid.to_numpy()).all()
row_of = {u: i for i, u in enumerate(df.sample_uid)}
twin = np.array([row_of.get(u, -1) if isinstance(u, str) else -1 for u in df.twin_uid])
fam, cell = "background_event", "L26:T1"
base = ((df.condition == "distractor") & (df.split == "test") & (df.family == fam)).to_numpy()
panels = [("baseline decoder", "baseline", base, PRE),
          ("A · trained sentences", "rad_all", base & (df.template_group == "seen").to_numpy(), TIER["A"]),
          ("C · family left out", f"lofo_{fam}", base, TIER["C"])]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), sharey=True)
for ax, (title, dec, m, col) in zip(axes, panels):
    p = P[f"{cell}|{dec}"]
    delta, gap = p[m] - p[twin[m]], df.log_gap.to_numpy()[m]
    b1, b0 = np.polyfit(gap, delta, 1)
    ax.axhline(0, color=MUTED, lw=1)
    ax.scatter(gap, delta, s=16, color=col, alpha=0.75, edgecolor="white", linewidth=0.5)
    xs = np.array([gap.min(), gap.max()]); ax.plot(xs, b0 + b1 * xs, color=INK, lw=2, solid_capstyle="round")
    rms = float(np.sqrt((delta ** 2).mean()))
    ax.text(0.03, 0.96, f"pull {b1:+.2f}   shift {delta.mean():+.2f}\nspread (RMS) {rms:.2f} decades",
            transform=ax.transAxes, va="top", fontsize=9.5, color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("gap = log D − log H (decades)")
axes[0].set_ylabel("readout(with sentence) − readout(twin)\n(decades)")
axes[0].set_ylim(-1.6, 1.6)
fig.savefig(OUT / "background_event_up_close.png"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
