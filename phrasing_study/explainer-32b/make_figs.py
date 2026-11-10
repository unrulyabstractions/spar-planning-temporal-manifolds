"""Figures for the phrasing-study 32B explainer (index.html next to this file).

Reads three choice runs from the shared repo (small files only, no activations):
  qwen3-14b_phrasing_s0  Qwen3-14B, old readout (one separator, " "), 2026-10-07
  qwen3-14b_choice_r2    Qwen3-14B, readout v2 (summed over " " and " **"), 2026-10-12
  qwen3-32b_choice_r2    Qwen3-32B, readout v2, 2026-10-12
and the CSVs scripts/evaluate.py wrote to results/<run>; writes PNGs and a summary CSV to figs/.
Run from anywhere: python phrasing_study/explainer-32b/make_figs.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent               # phrasing_study/
RUNS = {"s0": "qwen3-14b_phrasing_s0", "r14": "qwen3-14b_choice_r2", "r32": "qwen3-32b_choice_r2"}
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
from phr.analysis import boot_ci, choice_tables, paired  # noqa: E402

INK, MUTED, RULE = "#1A242D", "#56636E", "#D6DDE2"
C14, C32, OLD = "#2a78d6", "#eb6834", "#9aa3ab"          # 14B, 32B (validated pair), old readout
DIV = LinearSegmentedColormap.from_list(   # blue (fewer short choices) <-> gray <-> red (more short choices)
    "div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#9c2121"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": RULE, "axes.linewidth": 1, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 160, "savefig.bbox": "tight",
})

cfg = yaml.safe_load((ROOT / "runs" / RUNS["r32"] / "variants.yaml").read_text())
ch = {k: pd.read_parquet(ROOT / "runs" / r / "choice.parquet") for k, r in RUNS.items()}
res = lambda k, name: pd.read_csv(ROOT / "results" / RUNS[k] / f"{name}.csv")   # noqa: E731
per = {k: res(k, "choice_variants") for k in RUNS}
con = {k: res(k, "choice_contrasts") for k in RUNS}
floor = {k: float(per[k][per[k].family == "nulls"].mean_delta.abs().max()) for k in RUNS}
LEVELS = [h["canonical"] for h in cfg["horizons"]]


def robustness(c: pd.DataFrame) -> pd.DataFrame:
    """Per variant: share of cell pairs where both prompts put >= 0.5 on the two labels, and the paired delta there."""
    mass_ref = c[["variant", "cell", "mass_ab"]].rename(columns={"variant": "ref_variant", "mass_ab": "mass_ref"})
    d = paired(c).merge(mass_ref, on=["ref_variant", "cell"])
    d["reliable"] = (d.mass_ab >= 0.5) & (d.mass_ref >= 0.5)
    out = {}
    for v, g in d.groupby("variant"):
        gr = g[g.reliable]
        m, lo, hi = boot_ci(gr, "delta", "scenario") if len(gr) >= 20 else (np.nan, np.nan, np.nan)
        out[v] = dict(frac_reliable=g.reliable.mean(), rel_mean=m, rel_lo=lo, rel_hi=hi)
    return pd.DataFrame(out).T


rob = {k: robustness(ch[k]) for k in ("r14", "r32")}


def nice(v: str) -> str:
    """Readable label for a variant id."""
    lab = {"canonical": "canonical", "null_no_period": "no final period", "null_largest": "largest / greatest",
           "null_gives": "gives / provides", "null_carefully": "carefully / deeply", "null_thanks": "+ \"Thank you.\"",
           "null_please": "+ \"Please\"", "reword_planning_horizon": "\"Your planning horizon is h…\"",
           "reword_time_frame": "\"Evaluate … over a time frame of h…\"", "reword_within_next": "\"… within the next h\"",
           "reword_only_within": "\"Only outcomes within the next h matter\" *",
           "unit_days": "days (\"180 days\")", "unit_weeks": "weeks (\"26 weeks\")", "unit_months": "months (\"60 months\")",
           "unit_years": "years (\"0.5 years\")", "unit_decades": "decades (\"0.5 decades\")",
           "words": "words (\"six months\")", "named": "named (\"half a year\")", "opt_months": "options in months",
           "opt_months_h_months": "options + horizon in months †", "labels_AB": "labels A) / B)",
           "labels_12": "labels 1) / 2)", "constraint_first": "constraint before options", "markdown": "markdown",
           "prose": "plain prose", "task_brief": "task brief", "user_request": "person asking for help"}
    if v in lab:
        return lab[v]
    cues = {f"{dd}_{i}": t for dd, ts in cfg["choice"]["cues"].items() for i, t in enumerate(ts, 1)}
    return f"+ \"{cues[v]}\"" if v in cues else v


# summary table used by the page: every variant, three runs side by side --------------------------------------------
tab = None
for k in RUNS:
    t = per[k][["family", "variant", "mean_delta", "lo", "hi", "matters"]].rename(
        columns={"mean_delta": f"d_{k}", "lo": f"lo_{k}", "hi": f"hi_{k}", "matters": f"matters_{k}"})
    tab = t if tab is None else tab.merge(t.drop(columns="family"), on="variant")
tab = tab.join(rob["r32"][["frac_reliable"]].rename(columns={"frac_reliable": "frac_reliable_r32"}), on="variant")
c = ch["r14"]
old_style = c.assign(logit_diff=c.ld_space, p_short=1 / (1 + np.exp(-c.ld_space)), mass_ab=c.mass_space)
tab = tab.merge(choice_tables(old_style)["choice_variants"][["variant", "mean_delta"]].rename(columns={"mean_delta": "d_hw"}),
                on="variant")      # 14B rerun, " " path only: the Oct 7 readout on the new hardware
tab.to_csv(OUT / "variants_three_runs.csv", index=False)

# 1 · readout mass by layout: old readout vs summed, 14B and 32B ---------------------------------------------------
order = [("formatted", "Alan's lines"), ("constraint_first", "constraint first"), ("markdown", "markdown"),
         ("prose", "plain prose"), ("task_brief", "task brief"), ("user_request", "person asking")]
series = [("s0", OLD, "14B, old readout (\" \" only)"), ("r14", C14, "14B, summed readout"), ("r32", C32, "32B, summed readout")]
fig, ax = plt.subplots(figsize=(8.4, 4.6))
k, yt, yl = 0, [], []
for L, lab in order:
    for j, (run, col, _) in enumerate(series):
        v = ch[run][ch[run].layout == L].mass_ab
        q = v.quantile([0.25, 0.5, 0.75]).values
        y = k + j * 0.27
        ax.plot([q[0], q[2]], [y, y], color=col, lw=5, alpha=0.35, solid_capstyle="butt")
        ax.plot(q[1], y, "o", ms=7, color=col, mec="white", mew=1)
    yt.append(k + 0.27)
    yl.append(lab)
    k += 1.2
ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
ax.set_yticks(yt, yl, fontsize=9.5)
ax.set_ylim(k - 0.3, -0.4)
ax.set_xlim(0, 1.03)
ax.grid(axis="y", visible=False)
ax.set_xlabel("probability on the two answer labels (dot = median, bar = middle 50% of prompts)")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=lab) for _, c, lab in series],
          frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
fig.savefig(OUT / "f1_readout_mass.png")
plt.close(fig)

# 2 · what the fix changed on the 14B: every variant's delta, old vs new readout -------------------------------------
x = per["s0"][["variant", "family", "mean_delta"]].merge(per["r14"][["variant", "mean_delta"]], on="variant",
                                                           suffixes=("_old", "_new"))
x["alan"] = ~x.variant.str.contains(r"\+") & ~x.variant.isin(["markdown", "prose", "task_brief", "user_request",
                                                              "constraint_first"]) & ~x.variant.str.contains("@")
fig, ax = plt.subplots(figsize=(6.6, 6.0))
lim = (-0.7, 0.75)
ax.plot(lim, lim, color=RULE, lw=1.2, ls="--", zorder=1)
o = x[~x.alan]
ax.plot(o.mean_delta_old, o.mean_delta_new, "o", ms=6, mfc="white", mec=MUTED, mew=1.2, zorder=2,
        label="other layouts (markdown, prose, task brief, person asking, constraint first)")
a = x[x.alan]
ax.plot(a.mean_delta_old, a.mean_delta_new, "o", ms=6.5, color=C14, mec="white", mew=0.8, zorder=3,
        label="Alan's layout")
notes = {"labels_12": ("labels 1) / 2)", (25, -28)), "later_1": ("leaning later", (40, -45)),
         "markdown+sooner_1": ("markdown + leaning sooner", (12, -16)), "markdown": ("markdown layout", (-120, 45)),
         "implicit:mortgage_30": ("mortgage (implicit)", (10, -4)),
         "implicit:baby_8wk": ("8 weeks pregnant (implicit)", (8, -30))}
for v, (lab, off) in notes.items():
    r = x[x.variant == v]
    if len(r):
        r = r.iloc[0]
        ax.annotate(lab, (r.mean_delta_old, r.mean_delta_new), xytext=off, textcoords="offset points", fontsize=8.5,
                    color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=0, shrinkB=4))
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ax.set_aspect("equal")
ax.set_xlabel("Δ P(short) with the old readout (Oct 7)")
ax.set_ylabel("Δ P(short) with the summed readout (Oct 12)")
ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
fig.savefig(OUT / "f2_fix_effects.png")
plt.close(fig)

# 3 · canonical curves, both orders, 14B vs 32B ------------------------------------------------------------------------
xs = np.arange(len(LEVELS))
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
for ax, (run, col, title) in zip(axes, (("r14", C14, "Qwen3-14B"), ("r32", C32, "Qwen3-32B"))):
    can = ch[run][ch[run].variant == "canonical"]
    for sf, ls, lab in ((True, "-", "short option listed first"), (False, "--", "short option listed second")):
        m = can[can.short_first == sf].groupby("level").p_short.mean().reindex(LEVELS)
        ax.plot(xs, m.values, color=col, lw=2, ls=ls, marker="o", ms=5.5, mfc=col if sf else "white", mew=1.4, label=lab)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xticks(xs, LEVELS, rotation=30, ha="right")
    ax.set_ylim(-0.02, 1.04)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
axes[0].set_ylabel("P(choose the short option)")
fig.supxlabel("stated horizon (canonical prompt, mean over 16 scenarios)", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(OUT / "f3_curves.png")
plt.close(fig)

# 4 · every variant on Alan's layout, 14B vs 32B ---------------------------------------------------------------------
groups = [("nulls", "Null rewordings (noise floor)"), ("reword", "Rewordings of the constraint"),
          ("unit", "Same horizon, other unit or form"), ("option_unit", "Option delays in months"),
          ("labels", "Option labels"), ("layout", "Whole prompt structure"), ("cue", "Added sentence (cue)")]
rows = []
oc = {k: con[k][con[k].contrast.str.startswith("option order")].iloc[0] for k in ("r14", "r32")}
rows.append(("Option order", "short option first vs second",
             {k: (oc[k].mean_diff, oc[k].lo, oc[k].hi, (oc[k].lo > 0) | (oc[k].hi < 0)) for k in oc}))
for fam, title in groups:
    t = per["r14"][per["r14"].family == fam].sort_values("mean_delta")
    for r in t.itertuples():
        q = per["r32"].set_index("variant").loc[r.variant]
        rows.append((title, nice(r.variant), {"r14": (r.mean_delta, r.lo, r.hi, r.matters),
                                              "r32": (q.mean_delta, q.lo, q.hi, q.matters)}))
fig, ax = plt.subplots(figsize=(8.8, 0.36 * len(rows) + 2.6))
y, last, labels = 0, None, []
for g, lab, vals in rows:
    if g != last:
        y += 0.9 if last is not None else 0
        ax.text(-0.42, y, g, fontsize=10, fontweight="bold", color=INK, va="center", ha="left")
        y += 0.9
        last = g
    for run, col, off in (("r14", C14, -0.17), ("r32", C32, 0.17)):
        m, lo, hi, mat = vals[run]
        ax.plot([lo, hi], [y + off, y + off], color=col, lw=1.6, solid_capstyle="round", zorder=2)
        ax.plot(m, y + off, "o", ms=6.5, mfc=col if mat else "white", mec=col, mew=1.5, zorder=3)
    labels.append((y, lab))
    y += 1
ax.axvspan(-floor["r32"], floor["r32"], color="#EEF1F3", zorder=0)
for v in (-0.05, 0.05):
    ax.axvline(v, color=MUTED, lw=0.8, ls=":", zorder=1)
ax.axvline(0, color=MUTED, lw=1, zorder=1)
ax.set_yticks([p for p, _ in labels], [lab for _, lab in labels], fontsize=9)
ax.set_ylim(y, -1)
ax.set_xlim(-0.42, 0.42)
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Δ P(choose the short option) vs canonical prompt (95% CI over 16 scenarios)\n"
              "← acts like a longer horizon          acts like a shorter horizon →")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=C14, label="Qwen3-14B"),
                   plt.Line2D([], [], marker="o", ls="", color=C32, label="Qwen3-32B"),
                   plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.5, label="hollow: below the bar"),
                   plt.Rectangle((0, 0), 1, 1, fc="#EEF1F3", label=f"32B null floor ±{floor['r32']:.3f}")],
          frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=8.5)
fig.savefig(OUT / "f4_compare.png")
plt.close(fig)

# 5 · layout cross, side by side --------------------------------------------------------------------------------------
lays = ["formatted", "constraint_first", "prose", "task_brief", "user_request", "markdown"]
base = ["sooner_1", "later_1", "hurry_1", "relax_1", "neutral_1", "unit_days", "unit_months", "named"]
lay_lab = {"formatted": "Alan's\nlines", "constraint_first": "constr.\nfirst", "prose": "plain\nprose",
           "task_brief": "task\nbrief", "user_request": "person\nasking", "markdown": "mark-\ndown"}
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
for ax, (run, title) in zip(axes, (("r14", "Qwen3-14B"), ("r32", "Qwen3-32B"))):
    lc = res(run, "choice_layout_cross")
    M = lc.pivot_table(index="base_variant", columns="layout", values="mean_delta").reindex(index=base, columns=lays)
    ax.grid(False)
    im = ax.imshow(M.values, cmap=DIV, norm=TwoSlopeNorm(0, -0.45, 0.45), aspect="auto")
    for i, b in enumerate(base):
        for j, L in enumerate(lays):
            v = b if L == "formatted" else f"{L}+{b}"
            f = rob[run].loc[v, "frac_reliable"] if v in rob[run].index else 1.0
            val, low = M.values[i, j], f < 0.5
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=9, fontweight="bold" if low else "normal",
                    color=INK if abs(val) < 0.25 else "white", style="italic" if low else "normal",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white" if abs(val) < 0.25 else "#104281", ec="none", alpha=0.85) if low else None)
            if low:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", ec="#b8c0c6", lw=0))
    ax.set_xticks(range(len(lays)), [lay_lab[L] for L in lays], fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=11)
    ax.tick_params(length=0)
axes[0].set_yticks(range(len(base)), [nice(b).replace("+ ", "") for b in base], fontsize=9)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("Δ P(short) vs that layout's own canonical prompt", fontsize=9)
cb.outline.set_visible(False)
fig.savefig(OUT / "f5_layout_cross.png")
plt.close(fig)

# 6 · implicit horizons, 14B vs 32B ------------------------------------------------------------------------------------
items = {i["id"]: i for i in cfg["choice"]["implicit"]}
imp = per["r14"][(per["r14"].family == "implicit") & ~per["r14"].variant.str.contains("@")][["variant", "mean_delta"]]
imp = imp.merge(per["r32"][["variant", "mean_delta"]], on="variant", suffixes=("_14", "_32"))
imp["iid"] = imp.variant.str.split(":").str[1]
imp["det"] = imp.iid.map(lambda i: items[i]["determinacy"])
imp = imp.sort_values(["det", "mean_delta_14"])


def short_when(i):
    w = items[i]["when"].replace("before ", "", 1)
    return (w[:44] + "…") if len(w) > 45 else w


fig, ax = plt.subplots(figsize=(8.8, 6.4))
ax.grid(axis="y", visible=False)
for k, r in enumerate(imp.itertuples()):
    ax.plot([r.mean_delta_14, r.mean_delta_32], [k, k], color=RULE, lw=2, zorder=1)
    ax.plot(r.mean_delta_14, k, "o", ms=7.5, color=C14, mec="white", mew=1, zorder=3)
    ax.plot(r.mean_delta_32, k, "o", ms=7.5, color=C32, mec="white", mew=1, zorder=3)
n_con = int((imp.det == "concrete").sum())
ax.axhline(n_con - 0.5, color=MUTED, lw=0.8, ls=":")
ax.text(0.62, n_con - 0.9, "concrete ↑", fontsize=8.5, color=MUTED, ha="right")
ax.text(0.62, n_con - 0.0, "vague ↓", fontsize=8.5, color=MUTED, ha="right", va="top")
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(range(len(imp)), [f"{short_when(i)}  ({items[i]['twin']})" for i in imp.iid], fontsize=8.5)
ax.set_ylim(len(imp) - 0.4, -0.6)
ax.set_xlim(-0.65, 0.65)
ax.set_xlabel("Δ P(short): implicit − explicit twin (Alan's layout, investment domain,\nmean over 8 option configurations × 2 orders)\n"
              "← more patient than the twin          less patient →", fontsize=9.5)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=C14, label="Qwen3-14B"),
                   plt.Line2D([], [], marker="o", ls="", color=C32, label="Qwen3-32B")],
          frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
fig.savefig(OUT / "f6_implicit.png")
plt.close(fig)
print("figures written to", OUT)
