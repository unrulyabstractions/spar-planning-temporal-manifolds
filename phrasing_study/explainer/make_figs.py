"""Figures for the phrasing-study explainer (index.html next to this file).

Reads the Qwen3-14B run (runs/qwen3-14b_phrasing_s0: the small files, no activations needed) and the CSVs that
scripts/evaluate.py wrote to results/qwen3-14b_phrasing_s0; writes PNGs to figs/.
Run from anywhere: python phrasing_study/explainer/make_figs.py
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
RUN, RES = ROOT / "runs/qwen3-14b_phrasing_s0", ROOT / "results/qwen3-14b_phrasing_s0"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
from phr.analysis import boot_ci, paired  # noqa: E402

INK, MUTED, RULE = "#1A242D", "#56636E", "#D6DDE2"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"                 # categorical slots 1-3 (validated all-pairs)
GRAY = "#9aa3ab"
DIV = LinearSegmentedColormap.from_list(   # blue (fewer short choices) <-> gray <-> red (more short choices)
    "div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#9c2121"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": RULE, "axes.linewidth": 1, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 160, "savefig.bbox": "tight",
})

cfg = yaml.safe_load((RUN / "variants.yaml").read_text())
ch = pd.read_parquet(RUN / "choice.parquet")
per = pd.read_csv(RES / "choice_variants.csv")
con = pd.read_csv(RES / "choice_contrasts.csv")
floor = float(per[per.family == "nulls"].mean_delta.abs().max())

# robustness: the same paired delta restricted to pairs where BOTH prompts put >= 0.5 of their probability on the two
# label tokens (the readout is on-distribution there). Biased towards easy cells, so it is a check, not the estimate.
mass_ref = ch[["variant", "cell", "mass_ab"]].rename(columns={"variant": "ref_variant", "mass_ab": "mass_ref"})
d = paired(ch).merge(mass_ref, on=["ref_variant", "cell"])
d["reliable"] = (d.mass_ab >= 0.5) & (d.mass_ref >= 0.5)
rob = {}
for v, g in d.groupby("variant"):
    gr = g[g.reliable]
    m, lo, hi = boot_ci(gr, "delta", "scenario") if len(gr) >= 20 else (np.nan, np.nan, np.nan)
    rob[v] = dict(frac_reliable=g.reliable.mean(), rel_mean=m, rel_lo=lo, rel_hi=hi)
rob = pd.DataFrame(rob).T
per = per.join(rob, on="variant")
per["robust"] = per.matters & (np.sign(per.rel_mean) == np.sign(per.mean_delta)) & ((per.rel_lo > 0) | (per.rel_hi < 0))
per.to_csv(OUT / "choice_variants_with_robustness.csv", index=False)

LEVELS = [h["canonical"] for h in cfg["horizons"]]


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


# 1 · canonical curve and option order --------------------------------------------------------------------------
can = ch[ch.variant == "canonical"]
x = np.arange(len(LEVELS))
fig, ax = plt.subplots(figsize=(8.4, 4.0))
for sf, col, lab in ((True, S2, "short option listed first (a)"), (False, S1, "short option listed second (b)")):
    m = can[can.short_first == sf].groupby("level").p_short.mean().reindex(LEVELS)
    ax.plot(x, m.values, color=col, lw=2, marker="o", ms=6, zorder=3, label=lab)
m = can.groupby("level").p_short.mean().reindex(LEVELS)
ax.plot(x, m.values, color=INK, lw=1.4, ls="--", zorder=2, label="average of both orders")
ax.set_xticks(x, LEVELS, rotation=30, ha="right")
ax.set_ylim(-0.02, 1.04)
ax.set_ylabel("P(choose the short option)")
ax.set_xlabel("stated horizon (canonical wording)")
ax.legend(frameon=False, loc="lower left", fontsize=9.5)
ax.annotate("both options pay out after\nthe horizon: order decides", xy=(0, 0.57), xytext=(0.6, 0.30),
            fontsize=9, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
fig.savefig(OUT / "f1_curve.png")
plt.close(fig)

# 2 · ranking of all variants on the canonical layout -------------------------------------------------------------
groups = [("nulls", "Null rewordings (noise floor)"), ("reword", "Rewordings of the constraint"),
          ("unit", "Same horizon, other unit or form"), ("option_unit", "Option delays in months"),
          ("labels", "Option labels"), ("layout", "Whole prompt structure"), ("cue", "Added sentence (cue)")]
rows = []
o = con[con.contrast.str.startswith("option order")].iloc[0]
rows.append(("Option order", "short option first vs second", o.mean_diff, o.lo, o.hi, "robust"))
for fam, title in groups:
    t = per[per.family == fam].sort_values("mean_delta")
    for r in t.itertuples():
        status = "robust" if r.robust else ("matters" if r.matters else "no")
        rows.append((title, nice(r.variant), r.mean_delta, r.lo, r.hi, status))
fig, ax = plt.subplots(figsize=(8.6, 0.27 * len(rows) + 2.6))
ypos, y, last = [], 0, None
labels = []
for g, lab, m, lo, hi, st in rows:
    if g != last:
        y += 0.9 if last is not None else 0
        ax.text(-0.42, y, g, fontsize=10, fontweight="bold", color=INK, va="center", ha="left")
        y += 0.9
        last = g
    col = {"robust": INK, "matters": INK, "no": GRAY}[st]
    ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2)
    ax.plot(m, y, "o", ms=7, mfc=col if st == "robust" else "white", mec=col, mew=1.6, zorder=3)
    labels.append((y, lab))
    y += 1
ax.axvspan(-floor, floor, color="#EEF1F3", zorder=0)
for v in (-0.05, 0.05):
    ax.axvline(v, color=MUTED, lw=0.8, ls=":", zorder=1)
ax.axvline(0, color=MUTED, lw=1, zorder=1)
ax.set_yticks([p for p, _ in labels], [lab for _, lab in labels], fontsize=9)
ax.set_ylim(y, -1)
ax.set_xlim(-0.42, 0.42)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Δ P(choose the short option) vs canonical prompt (95% CI over 16 scenarios)\n"
              "← acts like a longer horizon          acts like a shorter horizon →")
ax.tick_params(axis="y", length=0)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, label="matters, and survives the readout check"),
                   plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.6, label="matters on the full readout only"),
                   plt.Line2D([], [], marker="o", ls="", mfc="white", mec=GRAY, mew=1.6, label="does not pass the bar"),
                   plt.Rectangle((0, 0), 1, 1, fc="#EEF1F3", label=f"null floor ±{floor:.3f}")],
          frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8.5)
fig.savefig(OUT / "f2_ranking.png")
plt.close(fig)

# 3 · layout cross ---------------------------------------------------------------------------------------------------
lc = pd.read_csv(RES / "choice_layout_cross.csv")
lays = ["formatted", "constraint_first", "prose", "task_brief", "user_request", "markdown"]
base = ["sooner_1", "later_1", "hurry_1", "relax_1", "neutral_1", "unit_days", "unit_months", "named"]
M = lc.pivot_table(index="base_variant", columns="layout", values="mean_delta").reindex(index=base, columns=lays)
fr = {}
for b in base:
    for L in lays:
        v = b if L == "formatted" else f"{L}+{b}"
        fr[(b, L)] = rob.loc[v, "frac_reliable"] if v in rob.index else np.nan
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.grid(False)
im = ax.imshow(M.values, cmap=DIV, norm=TwoSlopeNorm(0, -0.4, 0.4), aspect="auto")
for i, b in enumerate(base):
    for j, L in enumerate(lays):
        val, f = M.values[i, j], fr[(b, L)]
        low = f < 0.3
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=9,
                color=MUTED if low else (INK if abs(val) < 0.25 else "white"), style="italic" if low else "normal")
        if low:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", ec="#b8c0c6", lw=0))
lay_lab = {"formatted": "Alan's lines\n(canonical)", "constraint_first": "constraint\nfirst", "prose": "plain\nprose",
           "task_brief": "task\nbrief", "user_request": "person\nasking", "markdown": "markdown"}
ax.set_xticks(range(len(lays)), [lay_lab[L] for L in lays], fontsize=9)
ax.set_yticks(range(len(base)), [nice(b).replace("+ ", "") for b in base], fontsize=9)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("Δ P(short) vs that layout's own canonical prompt", fontsize=9)
cb.outline.set_visible(False)
fig.savefig(OUT / "f3_layout_cross.png")
plt.close(fig)

# 4 · implicit horizons: choice effect and stated horizon ------------------------------------------------------------
st = pd.read_parquet(RUN / "state.parquet")
items = {i["id"]: i for i in cfg["choice"]["implicit"]}
imp = per[(per.family == "implicit") & ~per.variant.str.contains("@")].copy()
imp["iid"] = imp.variant.str.split(":").str[1]
sv = st[st.family.isin(["implicit", "twin"])].pivot_table(index="implicit_id", columns="family", values="stated_years",
                                                          aggfunc="first")
imp["stated_ratio"] = imp.iid.map(np.log10(sv.implicit / sv.twin))
imp["det"] = imp.iid.map(lambda i: items[i]["determinacy"])
imp["num"] = imp.iid.map(lambda i: items[i]["with_number"])
imp = imp.sort_values(["det", "mean_delta"])


def short_when(i):
    w = items[i]["when"].replace("before ", "", 1)
    return (w[:44] + "…") if len(w) > 45 else w


fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 6.2), sharey=True, gridspec_kw=dict(width_ratios=[1.15, 1]))
yy = np.arange(len(imp))
for ax_ in (a1, a2):
    ax_.grid(axis="y", visible=False)
for k, r in enumerate(imp.itertuples()):
    col = S1 if r.det == "concrete" else S2
    a1.plot([r.lo, r.hi], [k, k], color=col, lw=1.6, solid_capstyle="round")
    a1.plot(r.mean_delta, k, "o", ms=7, color=col, mfc=col if r.num else "white", mew=1.6)
    a2.barh(k, r.stated_ratio, color=col, height=0.6, alpha=0.85)
    ans = st[(st.family == "implicit") & (st.implicit_id == r.iid)].answer.iloc[0]
    a2.text(max(r.stated_ratio, 0) + 0.06, k, f"\"{ans}\"", va="center", fontsize=8, color=MUTED)
a1.axvline(0, color=MUTED, lw=1)
a2.axvline(0, color=MUTED, lw=1)
a1.set_yticks(yy, [f"{short_when(i)}  ({items[i]['twin']})" for i in imp.iid], fontsize=8.5)
a1.set_ylim(len(imp) - 0.4, -0.6)
a1.set_xlabel("Δ P(short): implicit − explicit twin\n(Alan's layout; 95% CI over 8 configs)", fontsize=9.5)
a2.set_xticks([-1, 0, 1, 2, 3], ["÷10", "same", "×10", "×100", "×1000"])
a2.set_xlim(-1, 3.6)
a2.set_xlabel("horizon the model STATES for the implicit text,\nrelative to the twin's duration", fontsize=9.5)
a1.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=S1, label="concrete"),
                   plt.Line2D([], [], marker="o", ls="", color=S2, label="vague"),
                   plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.4, label="no number in the text"),
                   plt.Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, label="holds a number (not the horizon)")],
          frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.35, -0.13), ncol=4)
fig.tight_layout()
fig.savefig(OUT / "f4_implicit.png")
plt.close(fig)

# 5 · multi-turn cues ------------------------------------------------------------------------------------------------
mc = pd.read_csv(RES / "multiturn_comparisons.csv")


def pick(cond, where, direc):
    if where == "first":
        name = f"{cond}: first-turn cue direction '{direc}' vs none (plan end)"
        pat = f"{cond}: first-turn cue '{direc}_"
    else:
        name = f"{cond}: cue direction '{direc}' before step 3 vs none (steps >= 3)"
        pat = f"{cond}: cue '{direc}_"
    r = mc[mc.comparison == name].iloc[0]
    ph = mc[mc.comparison.str.startswith(pat) & (mc.comparison.str.contains("before") == (where == "step"))]
    return r, ph.mean_log10.values


fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
DIRS = [("hurry", S2, "hurry (3 phrasings)"), ("relax", S1, "relax (3 phrasings)"), ("neutral", GRAY, "neutral")]
for ax_, (cond, title) in zip(axes, (("base", "Horizon stated (\"within 2 years\" …)"), ("free", "No horizon stated (model picks)"))):
    ylabels, k = [], 0
    for where, wl in (("first", "in the first message"), ("step", "with the 'Continue' before step 3")):
        for direc, col, dl in DIRS:
            r, ph = pick(cond, where, direc)
            ax_.plot([r.lo, r.hi], [k, k], color=col, lw=2, solid_capstyle="round")
            ax_.plot(r.mean_log10, k, "o", ms=8, color=col, zorder=3)
            ax_.plot(ph, [k] * len(ph), "|", ms=11, mew=1.6, color=col, alpha=0.7)
            ylabels.append(f"{dl} · {wl}")
            k += 1
        k += 0.6
    ax_.axvline(0, color=MUTED, lw=1)
    ax_.set_title(title, fontsize=10.5, loc="left")
    ax_.set_xticks(np.log10([0.33, 0.5, 0.75, 1, 1.5, 2]), ["×0.33", "×0.5", "×0.75", "×1", "×1.5", "×2"])
    ax_.grid(axis="y", visible=False)
fig.supxlabel("plan horizons relative to the same conversation without the cue "
              "(dot = all phrasings pooled, 95% CI over 12 scenarios; ticks = each phrasing)", fontsize=9.5, color=INK)
yt = [0, 1, 2, 3.6, 4.6, 5.6]
axes[0].set_yticks(yt, ylabels, fontsize=9)
axes[0].set_ylim(6.2, -0.6)
fig.tight_layout()
fig.savefig(OUT / "f5_multiturn_cues.png")
plt.close(fig)

# 6 · multi-turn unit forms ----------------------------------------------------------------------------------------
cv = pd.read_csv(RES / "multiturn_conversations.csv")
mt_lv = cfg["multiturn"]["horizons"]
xs = np.log10([pd.Series(cv[cv.level == lv].h_years).iloc[0] for lv in mt_lv])
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.plot([xs[0] - 0.2, xs[-1] + 0.2], [xs[0] - 0.2, xs[-1] + 0.2], color=RULE, lw=1.2, ls="--", zorder=1)
ax.text(xs[-1] + 0.05, xs[-1] + 0.12, "plan ends at\nthe target", fontsize=8.5, color=MUTED, ha="right")
series = [("canonical", INK, "canonical (\"20 years\")", (cv.condition == "base") & cv.cue.isna()),
          ("unit_days", S2, "days (\"7300 days\")", cv.variant == "unit_days"),
          ("unit_months", S1, "months (\"240 months\")", cv.variant == "unit_months"),
          ("named", S3, "named (\"two decades\")", cv.variant == "named")]
for v, col, lab, sel in series:
    s = cv[sel].groupby("level").log_end.median().reindex(mt_lv)
    ok = s.notna().values
    ax.plot(xs[ok], s.values[ok], color=col, lw=2.2 if v == "canonical" else 1.8, marker="o", ms=6, label=lab, zorder=3)
ticks = {np.log10(1 / 12): "1 month", np.log10(0.5): "6 months", 0: "1 year", np.log10(2): "2 years",
         np.log10(5): "5 years", 1: "10 years", np.log10(20): "20 years"}
ax.set_xticks(xs, mt_lv)
ax.set_yticks(list(ticks), list(ticks.values()))
ax.set_xlabel("stated horizon (first message)")
ax.set_ylabel("plan end: largest step horizon\n(median over 12 scenarios)")
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.savefig(OUT / "f6_multiturn_units.png")
plt.close(fig)

# 7 · readout reliability ------------------------------------------------------------------------------------------
grp = ch.assign(g=np.where(ch.family.isin(["implicit", "twin"]), "implicit + twin items", "all other items"))
order = [("formatted", "Alan's lines"), ("constraint_first", "constraint first"), ("prose", "plain prose"),
         ("task_brief", "task brief"), ("user_request", "person asking"), ("markdown", "markdown")]
fig, ax = plt.subplots(figsize=(8.2, 3.8))
k, yl = 0, []
for L, lab in order:
    for g_, col in (("all other items", S1), ("implicit + twin items", S2)):
        v = grp[(grp.layout == L) & (grp.g == g_)].mass_ab
        if len(v) == 0:
            continue
        q = v.quantile([0.25, 0.5, 0.75]).values
        ax.plot([q[0], q[2]], [k, k], color=col, lw=6, alpha=0.35, solid_capstyle="butt")
        ax.plot(q[1], k, "o", ms=7, color=col)
        yl.append(f"{lab} · {'implicit + twin' if g_.startswith('implicit') else 'all other items'}")
        k += 1
ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
ax.set_yticks(range(k), yl, fontsize=9)
ax.set_ylim(k - 0.4, -0.6)
ax.set_xlim(0, 1.02)
ax.grid(axis="y", visible=False)
ax.set_xlabel("probability on the two label tokens after the prefilled \"I choose: \" (dot = median, bar = middle 50%)")
fig.savefig(OUT / "f7_readout.png")
plt.close(fig)
print("figures written to", OUT)
