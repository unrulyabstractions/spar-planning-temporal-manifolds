"""Figures for the choice4 think-budget experiment (E17, spec 28) and the implied-horizon experiment (E18, spec 29).

    python -m pipeline.s3_analysis.figures.choice4_implied_figures [results/horizon dir] [output dir]

Reads think_budget_summary.json and implied_summary.json (no model run) and writes
choice4_1..3 and implied_1..3 PNGs.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "horizon"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[3] / "figures"

# Okabe-Ito: colourblind-safe. BLUE = the reference/explicit/truth, ORANGE = the condition of interest.
BLUE, ORANGE, GREY, GREEN = "#0072B2", "#D55E00", "#8a8a8a", "#009E73"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 130})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def pct(d):
    return 100 * d["exact"] / d["n"]


# ---------------------------------------------------------------- choice4
c4 = json.load(open(RESULTS / "exp17_Qwen3-8B_think_budget" / "think_budget_summary.json"))["choice4"]
ref, u2 = c4["reference"], c4["unit2"]

# 1. Where the errors are: natural vs forced close.
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
groups = [("natural close", ref["natural"]["accuracy"], BLUE), ("forced close\n(hit 3,072 cap)", ref["forced"]["accuracy"], ORANGE)]
for i, (label, acc, colour) in enumerate(groups):
    ax[0].bar(i, pct(acc), color=colour, width=0.55)
    ax[0].text(i, pct(acc) + 1.5, f"{acc['exact']}/{acc['n']}", ha="center")
ax[0].set_xticks(range(2), [g[0] for g in groups])
ax[0].set_ylim(0, 110)
ax[0].set_ylabel("exact accuracy (%)")
ax[0].set_title("choice4: errors sit in the forced closes")
errors = [acc["n"] - acc["exact"] for _, acc, _ in groups]
ax[1].bar([0, 1], errors, color=[BLUE, ORANGE], width=0.55)
for i, e in enumerate(errors):
    ax[1].text(i, e + 0.8, str(e), ha="center")
ax[1].set_xticks(range(2), [g[0] for g in groups])
ax[1].set_ylabel("number of errors")
ax[1].set_title("44 errors in total")
save(fig, "choice4_1_errors_by_close.png")

# 2. Position bias: pick rate vs truth rate per option slot.
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
panels = [("natural close (n=141)", ref["natural"]["position"]),
          ("forced close (n=105)", ref["forced"]["position"]),
          ("continued, closed (n=68)", u2["position_closed"])]
x = np.arange(4)
for a, (title, pos) in zip(axes, panels):
    a.bar(x - 0.2, pos["truth_rate"], 0.4, color=BLUE, label="truth rate")
    a.bar(x + 0.2, pos["pick_rate"], 0.4, color=ORANGE, label="model pick rate")
    a.set_xticks(x, list("abcd"))
    a.set_title(f"{title}\nexcess-late {pos['excess_late']:+.2f}", fontsize=10)
    a.set_xlabel("option slot")
axes[0].set_ylabel("share of prompts")
axes[0].legend(frameon=False, loc="upper left")
save(fig, "choice4_2_position_bias.png")

# 3. Effect of more thinking room.
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.1, 1, 1.1]})
loop, noloop = u2["closed_by_loop"]["loop"], u2["closed_by_loop"]["no loop"]
reg = u2["closed_by_regime"]
labels = ["all", "no loop", "loop", "regime both", "regime near"]
vals = [u2["closed"], noloop, loop, reg["both"], reg["near"]]
ax[0].barh(range(5), [100 * k / n for k, n in vals], color=[GREY, GREEN, ORANGE, BLUE, BLUE], height=0.6)
for i, (k, n) in enumerate(vals):
    ax[0].text(100 * k / n + 1.5, i, f"{k}/{n}", va="center")
ax[0].set_yticks(range(5), labels)
ax[0].invert_yaxis()
ax[0].set_xlim(0, 120)
ax[0].set_xlabel("closed within +3,072 tokens (%)")
ax[0].set_title("Do forced traces finish?")

before, after = u2["original_accuracy_closed"], u2["accuracy_closed"]
ax[1].bar([0, 1], [pct(before), pct(after)], color=[ORANGE, BLUE], width=0.55)
for i, d in enumerate([before, after]):
    ax[1].text(i, pct(d) + 1.5, f"{d['exact']}/{d['n']}", ha="center")
ax[1].set_xticks([0, 1], ["forced label", "after continuing"])
ax[1].set_ylim(0, 110)
ax[1].set_ylabel("exact accuracy (%)")
ax[1].set_title(f"same 68 prompts\nwrong→right {u2['wrong_to_right']}, right→wrong {u2['right_to_wrong']}")

still = u2["still_open_accuracy"]
ax[2].bar([0, 1, 2], [pct(after), pct(still), pct(ref["natural"]["accuracy"])], color=[BLUE, ORANGE, GREY], width=0.55)
for i, d in enumerate([after, still, ref["natural"]["accuracy"]]):
    ax[2].text(i, pct(d) + 1.5, f"{d['exact']}/{d['n']}", ha="center")
ax[2].set_xticks([0, 1, 2], ["closed\nlater", "still open\n(forced again)", "natural\nclose"])
ax[2].set_ylim(0, 110)
ax[2].set_title("Traces that never close stay bad\n(8,192 tokens closed 0 of 25)", fontsize=10)
save(fig, "choice4_3_more_room.png")

# ---------------------------------------------------------------- implied
im = json.load(open(RESULTS / "exp18_Qwen3-8B_think-on_implied" / "implied_summary.json"))
beh, conv = im["behaviour"], im["conversion"]
fams = ["A", "B", "C"]
fam_names = {"A": "A: from stated dates", "B": "B: world knowledge", "C": "C: event + duration elsewhere"}


def acc(fam, arm):
    near, both = beh[fam][f"{arm} near"], beh[fam][f"{arm} both"]
    return 100 * (near["correct"] + both["correct"]) / (near["n"] + both["n"])


# 1. Implied vs explicit twin accuracy, by family and regime.
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
for a, fam in zip(axes, fams):
    cells = [("near", "near"), ("both", "both")]
    x = np.arange(2)
    for offset, arm, colour in [(-0.2, "explicit", BLUE), (0.2, "implied", ORANGE)]:
        vals = [100 * beh[fam][f"{arm} {r}"]["correct"] / beh[fam][f"{arm} {r}"]["n"] for r, _ in cells]
        a.bar(x + offset, vals, 0.4, color=colour, label=arm + (" twin" if arm == "explicit" else ""))
        for xi, v in zip(x + offset, vals):
            a.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)
    mc = beh[fam]["mcnemar"]
    a.set_xticks(x, ["near", "both"])
    a.set_ylim(0, 112)
    a.set_xlabel("true regime")
    a.set_title(f"{fam_names[fam]}\nMcNemar p={mc['p']:.2f} (n.s.)", fontsize=10)
axes[0].set_ylabel("accuracy (%), 16 rows per bar")
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14))
save(fig, "implied_1_behaviour.png")

# 2. Conversion: does the trace state a duration, and does it flip the regime?
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
x = np.arange(3)
ax[0].bar(x, [100 * conv[f]["converted_share"] for f in fams], color=BLUE, width=0.55)
for xi, f in zip(x, fams):
    ax[0].text(xi, 100 * conv[f]["converted_share"] + 1.5, f"{conv[f]['converted']}/{conv[f]['n']}", ha="center")
ax[0].set_xticks(x, fams)
ax[0].set_ylim(0, 112)
ax[0].set_ylabel("% of implied traces")
ax[0].set_title("Trace states a horizon duration")
ax[1].bar(x, [conv[f]["flips_regime"] for f in fams], color=ORANGE, width=0.55)
for xi, f in zip(x, fams):
    ax[1].text(xi, conv[f]["flips_regime"] + 0.15, str(conv[f]["flips_regime"]), ha="center")
ax[1].set_xticks(x, fams)
ax[1].set_ylabel("count of 40")
ax[1].set_title(f"Stated horizon in the wrong regime\n(mediation test: choice followed it in {im['mediation']['follows_stated']} of {im['mediation']['flips']} analysed)", fontsize=10)
save(fig, "implied_2_conversion.png")

# 3. Probe readout: Spearman rho, implied vs explicit, per site and family.
sites = ["think@1 L16", "prelabel L16", "suffix_1 L16"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
for a, site in zip(axes, sites):
    x = np.arange(3)
    for offset, arm, colour in [(-0.2, "explicit", BLUE), (0.2, "implied", ORANGE)]:
        a.bar(x + offset, [im["readout"][f"{site} {f}"][arm]["rho"] for f in fams], 0.4, color=colour, label=arm)
    a.set_xticks(x, fams)
    a.set_ylim(0, 1.05)
    a.set_title(site)
axes[0].set_ylabel("Spearman ρ, probe vs true log-duration")
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Is the horizon readable from activations? (ridge probe trained on explicit E0)", y=1.04, fontsize=11)
save(fig, "implied_3_readout.png")
