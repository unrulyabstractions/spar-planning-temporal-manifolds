"""Schematic figures for evaluation_design/explainer/index.html (written next to this file, in figs/).

All curves are ILLUSTRATIONS (made-up parameters), not data. They show what each confound test expects to see.
Run from the repository root: python evaluation_design/explainer/make_figs.py  (needs numpy, matplotlib)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
INK, MUTED, RULE = "#1A242D", "#56636E", "#D6DDE2"
BLUE, ORANGE, RED, GREY, GREEN, PURPLE = "#2a78d6", "#eb6834", "#e34948", "#9aa3ab", "#2a9d6f", "#6250d6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": RULE, "axes.linewidth": 1, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 160, "savefig.bbox": "tight",
})
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731

# ---------------------------------------------------------------- f1: three signatures on the choice curve
x = np.linspace(-0.6, 0.6, 300)          # log10(H/D): stated horizon over the far option's delay
base = sig(x / 0.07)
cases = [
    ("Horizon steering", "the switch point moves;\nthe curve keeps its shape", sig((x + 0.25) / 0.07), BLUE),
    ("Decision bias", "far is chosen more everywhere,\neven where it can't pay off", 0.35 + 0.65 * base, ORANGE),
    ("Disruption", "the curve flattens:\nchoices get noisy", sig(x / 0.35), RED),
]
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
for ax, (title, note, y, c) in zip(axes, cases):
    ax.plot(x, base, color=GREY, lw=2, label="unsteered")
    ax.plot(x, y, color=c, lw=2.6, label="steered")
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_title(title, fontsize=10.5)
    ax.text(0.06, 0.3, note, fontsize=8.5, color=c, va="top")
    ax.set_xlabel("log10(stated horizon / far delay)")
    ax.set_xticks([-0.4, 0, 0.4])
axes[0].annotate("", (-0.25, 0.5), (0, 0.5), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
axes[0].set_ylabel("P(choose the far option)")
axes[0].legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("Illustration: three ways a steer can change the choice curve on the verification grid", fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "f1_signatures.png")
plt.close(fig)

# ---------------------------------------------------------------- f2: rescale vs replace
t = np.linspace(-2.5, 5, 200)            # stated horizon, log10 days (30 s .. 500 y is about -3.5 .. 5.3)
fig, ax = plt.subplots(figsize=(7.8, 3.8))
ax.plot(t, np.full_like(t, 0.3), color=BLUE, lw=2.6, label='"rescale": every stated horizon read 2× longer')
tgt = np.log10(365.25)
ax.plot(t, 0.35 * (tgt - t), color=ORANGE, lw=2.6,
        label='"replace": horizons pulled toward one value (here 1 year)')
ax.axhline(0, color=MUTED, lw=0.8)
ax.axvline(tgt, color=ORANGE, lw=0.8, ls=":")
ax.text(tgt + 0.08, 1.4, "1 year", color=ORANGE, fontsize=8.5)
ax.set_xticks([-2.5, -1, 0, 1, 2, 3, 4, 5], ["5 min", "2.4 h", "1 d", "10 d", "3 mo", "3 yr", "27 yr", "270 yr"])
ax.set_xlabel("stated horizon in the prompt")
ax.set_ylabel("realised shift Δ log10 H*\n(decades)")
ax.legend(fontsize=8.5, frameon=False, loc="lower left")
ax.set_title("Illustration: the same average shift, two different mechanisms", fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "f2_rescale_replace.png")
plt.close(fig)

# ---------------------------------------------------------------- f3: meaning vs letter
fig, ax = plt.subplots(figsize=(7.8, 3.6))
groups = ["horizon steer", "letter steer\n(toward a)", "mixed"]
far_a = [0.8, 0.8, 0.9]                  # shift in P(far) when the far option is labelled a)
far_b = [0.8, -0.8, 0.3]                 # ... when it is labelled b)
xx = np.arange(3)
ax.bar(xx - 0.18, far_a, 0.34, color=BLUE, label="prompts where far = a)")
ax.bar(xx + 0.18, far_b, 0.34, color=PURPLE, label="prompts where far = b)")
for i in range(3):
    m, l = (far_a[i] + far_b[i]) / 2, (far_a[i] - far_b[i]) / 2
    ax.text(i, 1.12, f"meaning {m:+.1f}\nletter {l:+.1f}", ha="center", fontsize=8.5, color=INK)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(xx, groups)
ax.set_ylim(-1.1, 1.45)
ax.set_ylabel("shift toward the far option\n(arbitrary units)")
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.set_title("Illustration: splitting an effect by option order", fontsize=10.5)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(OUT / "f3_meaning_letter.png")
plt.close(fig)

# ---------------------------------------------------------------- f4: rank test against K draws
rng = np.random.default_rng(1)
fig, ax = plt.subplots(figsize=(7.8, 3.2))
fams = [("isotropic", 0.25, GREY), ("covariance-matched", 0.5, ORANGE), ("label-shuffled", 0.45, PURPLE)]
for i, (name, s, c) in enumerate(fams):
    v = np.abs(rng.normal(0, s, 99))
    ax.scatter(v, i + rng.uniform(-0.18, 0.18, 99), s=11, color=c, alpha=0.75)
    q = np.quantile(v, 0.95)
    ax.plot([q, q], [i - 0.3, i + 0.3], color=INK, lw=1.6)
m = 1.05
ax.axvline(m, color=BLUE, lw=2.2)
ax.text(m + 0.02, 2.45, "method", color=BLUE, fontsize=9)
ax.set_yticks(range(3), [f[0] for f in fams])
ax.set_xlabel("|odd part of the shift| per direction (one dot = one random direction, K = 99)")
ax.set_title("Illustration: p = (1 + number of dots at or beyond the line) / (K + 1)", fontsize=10.5)
ax.grid(axis="y", visible=False)
ax.set_ylim(-0.6, 2.7)
fig.tight_layout()
fig.savefig(OUT / "f4_rank_test.png")
plt.close(fig)
print("ok")
