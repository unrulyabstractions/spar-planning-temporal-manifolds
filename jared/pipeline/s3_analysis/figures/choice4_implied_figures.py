"""Figures for the choice4 think-budget experiment (E17, spec 28) and the implied-horizon experiment (E18, spec 29).

    python -m pipeline.s3_analysis.figures.choice4_implied_figures RESULTS_HORIZON_DIR [OUTPUT_DIR]

Reads think_budget_summary.json and implied_summary.json under RESULTS_HORIZON_DIR (no model run) and writes
choice4_1..3 and implied_1..3 PNGs. Every proportion bar carries a Wilson 95% interval; every n and every
title number comes from the loaded JSON.
"""

import json
import sys
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "figures"

# Okabe-Ito: colourblind-safe. BLUE = the reference/explicit/truth, ORANGE = the condition of interest.
BLUE, ORANGE, GREY, GREEN = "#0072B2", "#D55E00", "#8a8a8a", "#009E73"
PROVISIONAL = "C0 failed, provisional"
FAMILIES = ["A", "B", "C"]
FAMILY_NAMES = {"A": "A: from stated dates", "B": "B: world knowledge", "C": "C: event + duration elsewhere"}
FIRST_CAP_TOKENS = 3072
SECOND_CAP_TOKENS = 8192


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for k successes in n trials, as fractions in [0, 1]."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom)


def significance_text(p, alpha=0.05):
    return f"p={p:.2f} ({'n.s.' if p >= alpha else 'p<0.05'})"


def percent_bars(ax, positions, counts, colours, width=0.55, horizontal=False, label_fmt="{k}/{n}"):
    """Draw bars of 100*k/n with Wilson 95% error bars and a k/n label. counts = [(k, n), ...]."""
    values = np.array([100 * k / n for k, n in counts])
    intervals = [wilson_interval(k, n) for k, n in counts]
    err = np.clip([[values[i] - 100 * lo for i, (lo, _) in enumerate(intervals)],
                   [100 * hi - values[i] for i, (_, hi) in enumerate(intervals)]], 0, None)
    if horizontal:
        ax.barh(positions, values, color=colours, height=width, xerr=err, error_kw={"lw": 1, "capsize": 2})
        for pos, (k, n), (_, hi) in zip(positions, counts, intervals):
            ax.text(100 * hi + 1.5, pos, label_fmt.format(k=k, n=n), va="center", fontsize=9)
    else:
        ax.bar(positions, values, color=colours, width=width, yerr=err, error_kw={"lw": 1, "capsize": 2})
        for pos, (k, n), (_, hi) in zip(positions, counts, intervals):
            ax.text(pos, 100 * hi + 1.5, label_fmt.format(k=k, n=n), ha="center", fontsize=9)


def proportion_bars_with_rates(ax, rates, n, offset, colour, label):
    """Bars for per-slot rates (share of n prompts) with Wilson 95% error bars."""
    x = np.arange(len(rates))
    intervals = [wilson_interval(round(r * n), n) for r in rates]
    err = np.clip([[r - lo for r, (lo, _) in zip(rates, intervals)], [hi - r for r, (_, hi) in zip(rates, intervals)]],
                  0, None)
    ax.bar(x + offset, rates, 0.4, color=colour, label=label, yerr=err, error_kw={"lw": 0.8, "capsize": 1.5})


def apply_style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                         "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 130})


def save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out_dir / name)


def acc_counts(accuracy):
    return accuracy["exact"], accuracy["n"]


# ---------------------------------------------------------------- choice4
def figure_errors_by_close(c4, out_dir):
    ref = c4["reference"]
    groups = [("natural close", ref["natural"]["accuracy"], BLUE),
              (f"forced close\n(hit {FIRST_CAP_TOKENS:,} cap)", ref["forced"]["accuracy"], ORANGE)]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    percent_bars(ax[0], range(2), [acc_counts(g[1]) for g in groups], [g[2] for g in groups])
    ax[0].set_xticks(range(2), [g[0] for g in groups])
    ax[0].set_ylim(0, 115)
    ax[0].set_ylabel("exact accuracy (%), Wilson 95% interval")
    ax[0].set_title("choice4: errors concentrate in forced closes\n(confounded with horizon and looping)", fontsize=10)
    errors = [g[1]["n"] - g[1]["exact"] for g in groups]
    ax[1].bar([0, 1], errors, color=[BLUE, ORANGE], width=0.55)
    for i, e in enumerate(errors):
        ax[1].text(i, e + 0.8, str(e), ha="center")
    ax[1].set_xticks(range(2), [g[0] for g in groups])
    ax[1].set_ylabel("number of errors")
    ax[1].set_title(f"{sum(errors)} errors in total (n={sum(g[1]['n'] for g in groups)} traces)", fontsize=10)
    save(fig, out_dir, "choice4_1_errors_by_close.png")


def figure_position_bias(c4, out_dir):
    ref, u2 = c4["reference"], c4["unit2"]
    panels = [("natural close", ref["natural"]["position"]),
              ("forced close", ref["forced"]["position"]),
              ("closed on continuing", u2["position_closed"]),
              ("same traces, forced label", u2["position_original_same_prompts"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.8), sharey=True)
    for a, (title, pos) in zip(axes, panels):
        proportion_bars_with_rates(a, pos["truth_rate"], pos["n"], -0.2, BLUE, "truth rate")
        proportion_bars_with_rates(a, pos["pick_rate"], pos["n"], 0.2, ORANGE, "model pick rate")
        a.set_xticks(np.arange(4), list("abcd"))
        a.set_title(f"{title}\n(n={pos['n']}) excess-late {pos['excess_late']:+.2f}", fontsize=10)
        a.set_xlabel("option slot")
    axes[0].set_ylabel("share of prompts (Wilson 95%)")
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, out_dir, "choice4_2_position_bias.png")


def figure_more_room(c4, out_dir):
    ref, u2, u3 = c4["reference"], c4["unit2"], c4["unit3"]
    closed_k, closed_n = u2["closed"]
    loop, noloop = u2["closed_by_loop"]["loop"], u2["closed_by_loop"]["no loop"]
    regime = u2["closed_by_regime"]
    labels = ["all", "no loop", "loop", "regime both", "regime near"]
    counts = [u2["closed"], noloop, loop, regime["both"], regime["near"]]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.2, 1, 1.1]})
    percent_bars(ax[0], range(5), counts, [GREY, GREEN, ORANGE, BLUE, BLUE], width=0.6, horizontal=True)
    ax[0].set_yticks(range(5), labels)
    ax[0].invert_yaxis()
    ax[0].set_xlim(0, 135)
    ax[0].set_xticks([0, 20, 40, 60, 80, 100])
    ax[0].set_xlabel(f"closed within +{FIRST_CAP_TOKENS:,} tokens (%, Wilson 95%)")
    ax[0].set_title(f"Do forced traces finish? (n={u2['n']})", fontsize=10)

    before, after = u2["original_accuracy_closed"], u2["accuracy_closed"]
    percent_bars(ax[1], [0, 1], [acc_counts(before), acc_counts(after)], [ORANGE, BLUE])
    ax[1].set_xticks([0, 1], ["forced label", "after continuing"])
    ax[1].set_ylim(0, 115)
    ax[1].set_ylabel("exact accuracy (%, Wilson 95%)")
    ax[1].set_title(f"same {closed_k} prompts (of {closed_n})\nwrong→right {u2['wrong_to_right']}, "
                    f"right→wrong {u2['right_to_wrong']}", fontsize=10)

    still, natural = u2["still_open_accuracy"], ref["natural"]["accuracy"]
    percent_bars(ax[2], [0, 1, 2], [acc_counts(after), acc_counts(still), acc_counts(natural)], [BLUE, ORANGE, GREY])
    ax[2].set_xticks([0, 1, 2], ["closed\nlater", "still open\n(forced again)", "natural\nclose"])
    ax[2].set_ylim(0, 115)
    ax[2].set_title(f"Traces still open are less accurate\n(+{SECOND_CAP_TOKENS:,} tokens closed {u3['closed_in_unit3']} "
                    f"of {u3['attempted']} attempted)", fontsize=10)
    save(fig, out_dir, "choice4_3_more_room.png")


# ---------------------------------------------------------------- implied
def figure_implied_behaviour(im, out_dir):
    beh = im["behaviour"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
    x = np.arange(2)
    for a, family in zip(axes, FAMILIES):
        for offset, arm, colour in [(-0.2, "explicit", BLUE), (0.2, "implied", ORANGE)]:
            cells = [beh[family][f"{arm} {regime}"] for regime in ("near", "both")]
            percent_bars(a, x + offset, [(c["correct"], c["n"]) for c in cells], [colour, colour], width=0.4,
                         label_fmt="{k}/{n}")
        a.set_xticks(x, ["near", "both"])
        a.set_ylim(0, 118)
        a.set_xlabel("true regime")
        a.set_title(f"{FAMILY_NAMES[family]}\nMcNemar {significance_text(beh[family]['mcnemar']['p'])}, "
                    f"{beh[family]['mcnemar']['pairs']} pairs", fontsize=10)
    axes[0].set_ylabel("accuracy (%, Wilson 95%)")
    handles = [Patch(color=BLUE, label="explicit twin"), Patch(color=ORANGE, label="implied")]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"Implied vs explicit twin behaviour ({PROVISIONAL})", y=1.04, fontsize=11, fontweight="bold")
    save(fig, out_dir, "implied_1_behaviour.png")


def figure_implied_conversion(im, out_dir):
    conv, med = im["conversion"], im["mediation"]
    x = np.arange(len(FAMILIES))
    total_flips = sum(conv[f]["flips_regime"] for f in FAMILIES)
    follow_lo, follow_hi = wilson_interval(med["follows_stated"], med["flips"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    percent_bars(ax[0], x, [(conv[f]["converted"], conv[f]["n"]) for f in FAMILIES], [BLUE] * 3)
    ax[0].set_xticks(x, FAMILIES)
    ax[0].set_ylim(0, 118)
    ax[0].set_ylabel("% of implied traces (Wilson 95%)")
    ax[0].set_title("Trace states a horizon duration", fontsize=10)
    percent_bars(ax[1], x, [(conv[f]["flips_regime"], conv[f]["n"]) for f in FAMILIES], [ORANGE] * 3)
    ax[1].set_xticks(x, FAMILIES)
    ax[1].set_ylim(0, 118)
    ax[1].set_ylabel("% of implied traces (Wilson 95%)")
    ax[1].set_title(f"Stated horizon in the wrong regime ({total_flips} traces)\n"
                    f"mediation on {med['flips']}: choice followed it in {med['follows_stated']}\n"
                    f"(95% CI {100 * follow_lo:.0f}-{100 * follow_hi:.0f}%)", fontsize=9)
    fig.text(0.5, -0.04, "Families: " + "; ".join(FAMILY_NAMES.values()), ha="center", fontsize=8)
    fig.suptitle(f"Implied horizons: conversion and mediation ({PROVISIONAL})", y=1.03, fontsize=11, fontweight="bold")
    save(fig, out_dir, "implied_2_conversion.png")


def figure_implied_readout(im, out_dir):
    readout = im["readout"]
    sites = ["think@1 L16", "prelabel L16", "suffix_1 L16"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
    x = np.arange(len(FAMILIES))
    for a, site in zip(axes, sites):
        for offset, arm, colour in [(-0.2, "explicit", BLUE), (0.2, "implied", ORANGE)]:
            a.bar(x + offset, [readout[f"{site} {f}"][arm]["rho"] for f in FAMILIES], 0.4, color=colour, label=arm)
        n_labels = [f"{f}\nn={readout[f'{site} {f}']['explicit']['n']}/{readout[f'{site} {f}']['implied']['n']}"
                    for f in FAMILIES]
        a.set_xticks(x, n_labels)
        a.set_ylim(0, 1.05)
        a.set_title(site, fontsize=10)
    axes[0].set_ylabel("Spearman ρ, probe vs true log-duration")
    axes[0].legend(frameon=False, loc="upper right")
    fig.text(0.5, -0.09, "n = rows per bar (explicit/implied); no interval, rho is not a proportion. "
             "Families: " + "; ".join(FAMILY_NAMES.values()), ha="center", fontsize=8)
    fig.suptitle(f"Is the horizon readable from activations? Ridge probe trained on explicit E0 ({PROVISIONAL})",
                 y=1.04, fontsize=11, fontweight="bold")
    save(fig, out_dir, "implied_3_readout.png")


def make_all_figures(results_root, out_dir):
    apply_style()
    c4 = json.load(open(results_root / "exp17_Qwen3-8B_think_budget" / "think_budget_summary.json"))["choice4"]
    im = json.load(open(results_root / "exp18_Qwen3-8B_think-on_implied" / "implied_summary.json"))
    figure_errors_by_close(c4, out_dir)
    figure_position_bias(c4, out_dir)
    figure_more_room(c4, out_dir)
    figure_implied_behaviour(im, out_dir)
    figure_implied_conversion(im, out_dir)
    figure_implied_readout(im, out_dir)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.exit(__doc__)
    results_root = Path(argv[0])
    out_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT
    make_all_figures(results_root, out_dir)


if __name__ == "__main__":
    main()
