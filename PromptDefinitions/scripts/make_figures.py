"""Draw the figures for 01_prompt_types.md from figures/prompt_stats.json.

    ~/miniconda3/envs/intel_ai/bin/python PromptDefinitions/scripts/make_figures.py

Regenerate the JSON first with extract_prompt_stats.py if the prompt banks change.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
STATS = json.loads((FIG_DIR / "prompt_stats.json").read_text())

# Reference palette from the dataviz skill (light mode), validated as an adjacent set.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
ORANGE, BLUE, AQUA, GRAY = "#eb6834", "#2a78d6", "#1baf7a", "#a3a29c"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
})

SECOND = 1 / (365.25 * 86400)
TIME_TICKS = {"1 s": SECOND, "1 min": 60 * SECOND, "1 h": 3600 * SECOND, "1 d": 1 / 365.25,
              "1 wk": 7 / 365.25, "1 mo": 1 / 12, "1 y": 1, "10 y": 10, "100 y": 100,
              "1000 y": 1000, "10k y": 10_000}


def log_time_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(list(TIME_TICKS.values()), list(TIME_TICKS.keys()))
    ax.minorticks_off()
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG_DIR / name)


def horizon_grids():
    """Each bank's horizon grid on a log axis, with the option delay ranges behind it.
    A horizon left of the short-delay band means no option pays in time (regime `none`)."""
    rows = [  # (grid name, short-delay band, long-delay band)
        ("Jared starter, 16", "Starter near", "Starter far"),
        ("Jared full bank, 48", "Alan / full bank short", "Alan / full bank long"),
        ("Alan standard, 17", "Alan / full bank short", "Alan / full bank long"),
        ("Alan extended, 25", "Alan / full bank short", "Alan / full bank long"),
        ("Alan matrix, 12", "Alan / full bank short", "Alan / full bank long"),
        ("Prior paper geometry, 16", "Prior short", "Prior long"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    heldout = set(np.round(STATS["matrix_heldout"], 9))
    for i, (name, short, long) in enumerate(rows):
        y = len(rows) - 1 - i
        for band, color in ((short, ORANGE), (long, AQUA)):
            lo, hi = STATS["delays"][band]
            ax.fill_between([lo, hi], y - 0.32, y + 0.32, color=color, alpha=0.22, linewidth=0)
        xs = np.array(STATS["grids"][name])
        is_held = np.array([round(x, 9) in heldout for x in xs]) if "matrix" in name else np.zeros(len(xs), bool)
        ax.scatter(xs[~is_held], np.full((~is_held).sum(), y), s=26, color=BLUE, zorder=3,
                   edgecolor=SURFACE, linewidth=1)
        if is_held.any():
            ax.scatter(xs[is_held], np.full(is_held.sum(), y), s=26, facecolor=SURFACE, edgecolor=BLUE,
                       linewidth=1.5, zorder=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    log_time_axis(ax)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=BLUE, label="stated horizon"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=BLUE,
                   label="held-out horizon (matrix)"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, alpha=0.3, label="short-option delays"),
        plt.Rectangle((0, 0), 1, 1, color=AQUA, alpha=0.3, label="long-option delays"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    ax.set_title("Horizon grids against the option delays")
    save(fig, "horizon_grids.png")


def regimes():
    """Share of prompts per regime, per dataset."""
    order = [("null", GRAY, "null (no horizon)"), ("none", ORANGE, "none: nothing pays in time"),
             ("near", BLUE, "near: only the short option pays"), ("both", AQUA, "both: both pay")]
    names = list(STATS["datasets"])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for row, name in enumerate(names):
        d = STATS["datasets"][name]
        left = 0.0
        for key, color, _ in order:
            share = d["regimes"].get(key, 0) / d["n"]
            if share == 0:
                continue
            ax.barh(row, share, left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.7)
            if share >= 0.07:
                text_color = INK if key in ("null", "both") else SURFACE
                ax.text(left + share / 2, row, f"{share:.0%}", ha="center", va="center", fontsize=9,
                        color=text_color)
            left += share
        ax.text(1.01, row, f"n={d['n']:,}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=lab) for _, c, lab in order]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.08), ncol=4, frameon=False,
              fontsize=9)
    ax.set_title("Where the stated horizon falls relative to the two payouts")
    save(fig, "regimes.png")


def unit_confound():
    """Left: in the full bank the unit word is fixed by the magnitude. Right: Alan's paired
    rewrites break that link for the same canonical horizon."""
    panels = [
        ("full_bank_units", "Full bank (J2): one unit per magnitude",
         ["second", "minute", "hour", "day", "week", "month", "year"]),
        ("units_paired", "Alan unit rewrites (A4): several units per magnitude",
         ["hour", "day", "week", "month", "year", "decade", "century"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    for ax, (key, title, units) in zip(axes, panels):
        pairs = {(u, round(y, 9)) for u, y in STATS[key]}
        xs = [y for u, y in pairs]
        ys = [units.index(u) for u, y in pairs]
        ax.scatter(xs, ys, s=28, color=BLUE, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.set_yticks(range(len(units)), units)
        ax.set_ylim(-0.6, len(units) - 0.4)
        log_time_axis(ax)
        ax.set_xlim(SECOND * 0.5, 2000)
        ax.set_title(title, fontsize=11)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelrotation=45)
    axes[0].set_ylabel("unit word in the prompt")
    axes[1].set_xlabel("horizon (canonical magnitude)", x=-0.1)
    fig.tight_layout()
    save(fig, "unit_confound.png")


def order_balance():
    """Share of prompts with the short / immediate option written first (labelled a or A)."""
    caa = STATS["caa_immediate_is_A"]
    rows = [(name, d["near_first"]) for name, d in STATS["datasets"].items()]
    rows += [("Prior CAA implicit-1000 (P2)", caa["temporal_scope/temporal_scope_implicit_expanded_1000.json"]),
             ("Prior CAA implicit-1000, AB-randomized",
              caa["temporal_scope_AB_randomized/temporal_scope_implicit_expanded_1000.json"])]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ax.barh(range(len(rows)), vals, color=BLUE, height=0.6, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(v - 0.015, i, f"{v:.0%}", ha="right", va="center", fontsize=9, color=SURFACE)
    ax.axvline(0.5, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_yticks(range(len(rows)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Short / immediate option written first (dashed line: balanced)")
    save(fig, "order_balance.png")


def distractor_gaps():
    """Augusto's distractor families: log10(D / H). prep_time sits almost entirely below 0."""
    gaps = STATS["distractor_gaps"]
    families = ["entity_age", "prep_time", "tenure", "background_event", "other_horizon"]
    bins = np.arange(-5, 5.01, 0.5)
    fig, axes = plt.subplots(1, len(families), figsize=(12, 2.8), sharey=True, sharex=True)
    for ax, fam in zip(axes, families):
        g = np.array(gaps[fam])
        ax.hist(g, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
        ax.axvline(0, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_title(fam, fontsize=10)
        ax.text(0.03, 0.95, f"D < H: {np.mean(g < 0):.0%}", transform=ax.transAxes, va="top", fontsize=9,
                color=INK)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_ylim(0, 115)   # headroom for the D < H label
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("prompts")
    axes[2].set_xlabel("log10(distractor duration D / stated horizon H)")
    fig.suptitle("Irrelevant-duration distractors (R1): how far D sits from H", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, "distractor_gaps.png")


if __name__ == "__main__":
    horizon_grids()
    regimes()
    unit_confound()
    order_balance()
    distractor_gaps()
