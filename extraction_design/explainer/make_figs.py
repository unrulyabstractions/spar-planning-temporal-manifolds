"""Figures for extraction_design/explainer/index.html (written next to this file, in figs/).

f1 is DATA: it is computed from the committed prompt bank (prompt_bank/data/t0_s0/prompts.jsonl), and the counts it
prints are checked against the numbers quoted in steering/README.md (choice 1 and 2). f2 is an ILLUSTRATION
(a made-up curve), showing what the two curve edits do.

Run from the repository root:
    uv run --no-project --with numpy --with matplotlib python extraction_design/explainer/make_figs.py
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)
BANK = HERE.parents[1] / "prompt_bank" / "data" / "t0_s0" / "prompts.jsonl"
DAYS_PER_YEAR = 365.25

# Same palette and style as evaluation_design/explainer/make_figs.py, so the two pages read as one set.
INK, MUTED, RULE = "#1A242D", "#56636E", "#D6DDE2"
BLUE, ORANGE, RED, GREY, GREEN, PURPLE = "#2a78d6", "#eb6834", "#e34948", "#9aa3ab", "#2a9d6f", "#6250d6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": RULE, "axes.linewidth": 1, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 160, "savefig.bbox": "tight",
})
REGIMES = [("none", GREY, "none: neither pays off in time"), ("near_only", ORANGE, "near_only: near is correct"),
           ("both", BLUE, "both: far is correct")]

# ---------------------------------------------------------------- f1: the bank's regimes and the CAA classes (DATA)
records = [json.loads(line) for line in BANK.open()]
fit_pool = [r for r in records if r["split"] == "extraction" and r["horizon"] is not None]   # what fit.py may use
grid = sorted({(r["horizon_years"], r["horizon_grid_value"]) for r in fit_pool if not r["rewritten"]})
# log10 days of each grid value, from its un-rewritten prompts (horizon_years is the grid value's duration there)
t_of = {g: np.log10(y * DAYS_PER_YEAR) for y, g in grid}
assert len(t_of) == 48, len(t_of)
by_h = {g: Counter() for g in t_of}
for r in fit_pool:
    by_h[r["horizon_grid_value"]][r["regime"]] += 1

pure_none = [g for g, c in by_h.items() if set(c) == {"none"}]
below_10d = [g for g in t_of if t_of[g] < 1.0]
short = Counter(r["regime"] for r in fit_pool if r["h_over_far_delay"] <= 0.25)
long_ = Counter(r["regime"] for r in fit_pool if r["h_over_far_delay"] >= 4)
print(f"fit pool {len(fit_pool)} prompts {dict(Counter(r['regime'] for r in fit_pool))}; "
      f"{len(below_10d)} grid horizons below 10 days, all pure none: {set(below_10d) <= set(pure_none)}")
print(f"CAA short class (H/D <= 1/4) {dict(short)}; long class (H/D >= 4) {dict(long_)}")
# The numbers steering/README.md quotes (choices 1-2): the figure must agree with them.
assert dict(short) == {"none": 279, "near_only": 71} and dict(long_) == {"both": 57}
assert len(below_10d) == 24 and set(below_10d) <= set(pure_none)

fig, (a, b) = plt.subplots(1, 2, figsize=(12.2, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
xs = np.array(sorted(t_of.values()))
order = sorted(t_of, key=t_of.get)
bottom = np.zeros(len(order))
for name, color, label in REGIMES:
    h = np.array([by_h[g][name] for g in order])
    a.bar(xs, h, width=0.15, bottom=bottom, color=color, label=label)
    bottom += h
a.axvline(1.0, color=MUTED, lw=0.9, ls=":")
a.text(1.08, 11.0, "10 days", color=MUTED, fontsize=8.5, ha="left")
a.text(-1.4, 10.6, f"all {len(below_10d)} horizons below 10 days are pure 'none'", color=MUTED, fontsize=8.5,
       ha="center")
ticks = {"1 min": 1 / 1440, "1 h": 1 / 24, "1 d": 1, "1 mo": DAYS_PER_YEAR / 12, "1 y": DAYS_PER_YEAR,
         "10 y": 10 * DAYS_PER_YEAR, "100 y": 100 * DAYS_PER_YEAR}
a.set_xticks([np.log10(v) for v in ticks.values()], list(ticks))
a.set_ylim(0, 14.5)
a.set_xlabel("stated horizon (log scale)")
a.set_ylabel("prompts per stated horizon")
a.set_title("Extraction split: 471 horizon prompts over the 48-value grid", fontsize=10.5)
a.legend(fontsize=8, frameon=False, loc="upper center", ncol=3, handlelength=1.2, columnspacing=1.0)

lhd = {name: [np.log10(r["h_over_far_delay"]) for r in fit_pool if r["regime"] == name] for name, _, _ in REGIMES}
bins = np.linspace(-8, 2.5, 43)
b.hist([lhd[n] for n, _, _ in REGIMES], bins=bins, stacked=True, color=[c for _, c, _ in REGIMES])
for v in (np.log10(0.25), np.log10(4)):
    b.axvline(v, color=INK, lw=1.1, ls="--")
b.annotate(f"short class, H/D <= 1/4:\n{short['none']} none + {short['near_only']} near_only",
           (np.log10(0.25), 52), (-6.9, 64), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
b.annotate(f"long class, H/D >= 4:\n{long_['both']} both", (np.log10(4), 40), (0.9, 64), fontsize=8.5, color=INK,
           arrowprops=dict(arrowstyle="-", color=MUTED))
b.set_ylim(0, 80)
b.set_xlabel("log10(stated horizon / far delay)")
b.set_ylabel("prompts")
b.set_title("Who is in CAA's two classes (all prompts kept)", fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "f1_bank_classes.png")
plt.close(fig)

# ---------------------------------------------------------------- f2: snap vs displace (ILLUSTRATION)
u = np.linspace(0, 1, 801)                     # curve parameter, increasing toward longer horizons
curve = np.stack([np.cos(np.pi * (0.15 + 0.7 * u)) * 2.2, np.sin(np.pi * (0.15 + 0.7 * u)) * 1.4], 1)
at = lambda s: curve[int(round(np.clip(s, 0, 1) * (len(u) - 1)))]  # noqa: E731
t_hat, step = 0.36, 0.24
i = int(round(t_hat * (len(u) - 1)))
tangent = curve[i + 1] - curve[i - 1]
normal = np.array([tangent[1], -tangent[0]]) / np.linalg.norm(tangent)   # outward: h is off the curve along it
if normal[1] < 0:
    normal = -normal
h = at(t_hat) + 0.7 * normal                   # an activation whose nearest curve point is c(t_hat)
snap0, snap = at(t_hat), at(t_hat + step)
disp = h + at(t_hat + step) - at(t_hat)

fig, ax = plt.subplots(figsize=(7.8, 4.6))
ax.plot(curve[:, 0], curve[:, 1], color=GREY, lw=2.4, label="curve c(t) through the per-horizon means")
ax.text(*(curve[0] + [0.0, -0.32]), "shorter horizons", fontsize=8.5, color=MUTED, ha="center")
ax.text(*(curve[-1] + [0.0, -0.32]), "longer horizons", fontsize=8.5, color=MUTED, ha="center")
ax.plot(*zip(h, snap0), color=MUTED, lw=0.9, ls=":")
ax.scatter(*h, s=60, color=INK, zorder=5)
ax.text(h[0] + 0.12, h[1] + 0.02, "h, an activation\n(off the curve)", fontsize=9, va="center")
ax.scatter(*snap0, s=40, color=GREY, zorder=5)
ax.text(snap0[0] + 0.1, snap0[1] - 0.3, "c(t̂), the nearest curve point\n= snap at Δ = 0, which is not h", fontsize=8.5,
        color=MUTED, ha="left", va="top")
ax.scatter(*snap, s=60, color=ORANGE, zorder=5)
ax.text(snap[0] - 0.05, snap[1] - 0.3, "snap: c(t̂ + Δ)\n(lands on the curve)", fontsize=9, color=ORANGE, ha="center",
        va="top")
ax.annotate("", disp, h, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8, shrinkA=6, shrinkB=6))
ax.annotate("", snap, snap0, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0, ls="--", shrinkA=5, shrinkB=6))
ax.scatter(*disp, s=60, color=BLUE, zorder=5)
ax.text(disp[0] - 0.12, disp[1] + 0.02, "displace: h + c(t̂ + Δ) − c(t̂)\n(keeps h's offset from the curve)",
        fontsize=9, color=BLUE, ha="right", va="center")
ax.set_xlim(-2.6, 2.6)
ax.set_ylim(0.1, 2.75)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("two of the curve's PCA coordinates; the part of h outside the subspace is never changed")
ax.set_title("Illustration: the two curve edits, for a shift of Δ decades toward longer horizons", fontsize=10.5)
ax.legend(fontsize=8, frameon=False, loc="lower center")
fig.tight_layout()
fig.savefig(OUT / "f2_snap_displace.png")
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
