"""Presentation figures, from saved results only (no model; the 8B tokenizer decodes
the E10 reasoning):

1. decodable is not causal: per layer, how well the horizon decodes at a position
   against how much of the choice patching that position carries;
2. where steering acts: ablation and steering at the answer slot by layer (thinking on);
3. horizon or decision: the three checks at the L24 answer slot;
4. E10: patching the horizon before reasoning, and what the reasoning then says.

    python analysis/presentation_figures.py results/horizon
    -> jared/figures/pres_*.png, <root>/analysis/presentation_figures.txt
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spar_horizon.steering import bootstrap_mean  # noqa: E402

JARED = Path(__file__).resolve().parent.parent
FIGURES = JARED / "figures"
BLUE, ORANGE, AQUA, PINK, PURPLE, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#d6437b", "#7b4fd6", "#9a9893", "#52514e"
# T1 group index of the suffix newline (the '\n' right after <|im_end|>) and of the pre-label ':'
SUFFIX_NEWLINE, PRE_LABEL = 3, 13


def load(root, name):
    return json.loads((root / name / "results.json").read_text())


def mean_of(values):
    return np.array([v[0] for v in values])


# --- 1. decodable vs causal ---------------------------------------------------------

def plot_decodable_vs_causal(root, path):
    probes = json.loads((root / "analysis" / "probe_dimensionality.json").read_text())
    run1, run2 = load(root, "exp8_Qwen3-8B_full"), load(root, "exp8_Qwen3-8B_answer_slot")
    t1 = run1["patch"]["transfer"]

    def decodability(mode, cell):
        return np.array([layer["ceiling_interior"] for layer in probes[mode]["cells"][cell]])

    panels = [
        ("Suffix newline, thinking off", decodability("off", "suffix newline"),
         [("patch this token (T1)", mean_of([row[SUFFIX_NEWLINE] for row in t1]), ORANGE)]),
        ("Answer slot ':', thinking off", decodability("off", "pre-label"),
         [("patch this token (T1)", mean_of([row[PRE_LABEL] for row in t1]), ORANGE),
          ("patch across prompts (T5)", mean_of(run2["answer"]["off"]["transfer"]), PINK)]),
        ("Answer slot ':', thinking on", decodability("on", "pre-label"),
         [("patch across prompts (T5)", mean_of(run2["answer"]["on"]["transfer"]), PINK)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (title, r2, causal) in zip(axes, panels):
        layers = np.arange(1, len(r2))
        ax.plot(layers, r2[1:], color=BLUE, lw=2.4, label="horizon decodable (probe R², held-out horizons)")
        for name, values, color in causal:
            ax.plot(np.arange(1, len(values)), values[1:], color=color, lw=2.4, label=f"choice carried: {name}")
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("layer")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, frameon=False, loc="center right")
    axes[0].set_ylabel("R²  /  share of the choice shift carried")
    fig.suptitle("Decodable is not causal: the horizon reads out near-perfectly long before (or without) "
                 "the position carrying the choice", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- 2. steering by layer -------------------------------------------------------------

def toward(cell, direction):
    """Effect toward the rational answer at 2 decades, both signs pooled, with a bootstrap CI."""
    steer = cell["steer"].get(direction)
    if steer is None:
        return None
    return bootstrap_mean(np.r_[steer["2"]["toward"], steer["-2"]["toward"]])


def answer_slot_cells(root):
    """Thinking-on answer-slot cells by layer, all fitted on every prompt (the natural-closes refits at
    L22 and L24, runs 3 and 4, are left out so the layers are comparable)."""
    cells = {21: load(root, "exp8_Qwen3-8B_full")["cells"]["on|pre-label|21"]}
    run2 = load(root, "exp8_Qwen3-8B_answer_slot")["cells"]
    for layer in (24, 27, 30):
        cells[layer] = run2[f"on|pre-label|{layer}"]
    return cells


def plot_steering_by_layer(root, path, lines):
    cells = answer_slot_cells(root)
    decision = json.loads((root / "analysis" / "steering_decision_check.json").read_text())
    layers = sorted(cells)
    x = np.arange(len(layers))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5.2))

    removed = [cells[layer]["ablate"]["4"]["horizon"] for layer in layers]
    ax0.bar(x, [r[0] for r in removed], 0.6, color=BLUE, label="horizon subspace (k = 4) ablated",
            yerr=[[r[0] - r[1] for r in removed], [r[2] - r[0] for r in removed]],
            error_kw={"elinewidth": 0.8, "ecolor": INK})
    random_worst = [max(v[0] for n, v in row.items() if n.startswith("random"))
                    for layer in layers for row in [next(r for r in cells[layer]["ablate"].values()
                                                         if any(n.startswith("random") for n in r))]]
    ax0.scatter(x, random_worst, marker="_", s=400, color=GRAY, linewidths=3, zorder=3,
                label="random subspace, same k (worst of 3)")
    ab = [decision.get(f"on|pre-label|{layer}", {}).get("ab_from_subspace", np.nan) for layer in layers]
    ax0.plot(x, ab, "o--", color=PINK, label="a-vs-b label decodable from the subspace (R²)")
    ax0.set_xticks(x, [f"L{layer}" for layer in layers])
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_ylabel("share of the regime contrast removed  /  R²")
    ax0.set_title("Ablation at the answer slot: it grows with depth, but so does the label it contains", fontsize=10)
    ax0.legend(fontsize=8, frameon=False, loc="upper left")
    ax0.grid(axis="y", alpha=0.2)

    directions = [("curve", "along the curve of horizon averages", ORANGE), ("line", "along the probe line", BLUE),
                  ("mean difference", "mean difference (CAA)", AQUA), ("random 0", "random direction", GRAY)]
    width = 0.2
    for j, (direction, name, color) in enumerate(directions):
        stats = [toward(cells[layer], direction) for layer in layers]
        present = [i for i, s in enumerate(stats) if s is not None]
        mean = np.array([stats[i][0] for i in present])
        err = np.array([[stats[i][0] - stats[i][1] for i in present], [stats[i][2] - stats[i][0] for i in present]])
        ax1.bar(x[present] + (j - 1.5) * width, mean, width * 0.92, color=color, label=name, yerr=err,
                error_kw={"elinewidth": 0.8, "ecolor": INK})
        lines.append(f"steer {direction:16s} " + "  ".join(
            f"L{layers[i]} {stats[i][0]:+.2f}" for i in present))
    ax1.axhline(0, color=INK, lw=0.6)
    ax1.set_xticks(x, [f"L{layer}" for layer in layers])
    ax1.set_ylabel("effect toward the rational answer\n(share of the regime contrast, 2 decades)")
    ax1.set_title("Steering at the answer slot peaks at L24, at 0.15 of a regime change",
                  fontsize=10)
    ax1.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    ax1.grid(axis="y", alpha=0.2)
    for layer, r, a in zip(layers, removed, ab):
        lines.append(f"ablate L{layer}: removes {r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}], a-vs-b from subspace {a:.2f}")
    fig.suptitle("Thinking on, answer slot ':' after reasoning, by layer", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- 3. horizon or decision -----------------------------------------------------------

def plot_horizon_vs_decision(root, path):
    check = json.loads((root / "analysis" / "steering_decision_check.json").read_text())
    walk = json.loads((root / "analysis" / "causal_walkthrough.json").read_text())
    run3 = load(root, "exp8_Qwen3-8B_decision_L24")["cells"]["on|pre-label|24"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    ax = axes[0]
    layers = (24, 27, 30)
    series = [("horizon_from_subspace", "log horizon, from the subspace", BLUE),
              ("horizon_from_decision", "log horizon, from the choice alone", GRAY),
              ("near_from_subspace", "near-vs-far preference, from the subspace", ORANGE),
              ("ab_from_subspace", "a-vs-b label, from the subspace", PINK)]
    x = np.arange(len(layers))
    for j, (key, name, color) in enumerate(series):
        ax.bar(x + (j - 1.5) * 0.2, [check[f"on|pre-label|{layer}"][key] for layer in layers], 0.18,
               color=color, label=name)
    ax.set_xticks(x, [f"L{layer}" for layer in layers])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("cross-validated R²")
    ax.set_title("1. What the horizon subspace contains\n(4 top horizon-average PCs, thinking on)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    row = run3["ablate"]["4"]
    parts = [("horizon", "subspace\n(4 dims)", BLUE, "curve"),
             ("horizon without decision", "without u\n(3 dims)", AQUA, "curve without decision"),
             ("decision", "u alone\n(1 dim)", PINK, "decision part of curve"),
             ("random 0", "random", GRAY, "random 0")]
    for j, (ablate_name, label, color, steer_name) in enumerate(parts):
        a = row[ablate_name]
        s = toward(run3, steer_name)
        ax.bar(j - 0.19, a[0], 0.36, color=color, yerr=[[a[0] - a[1]], [a[2] - a[0]]],
               error_kw={"elinewidth": 0.8, "ecolor": INK})
        ax.bar(j + 0.19, s[0], 0.36, color=color, alpha=0.45, hatch="//", edgecolor=color,
               yerr=[[s[0] - s[1]], [s[2] - s[0]]], error_kw={"elinewidth": 0.8, "ecolor": INK})
    ax.set_xticks(range(len(parts)), [p[1] for p in parts], fontsize=9)
    ax.set_ylabel("share of the regime contrast")
    left = walk["decision_left_without_u"]
    note = f"after removing u the subspace still predicts the\npreference at R² {left['near_lo_without_u']:.2f}"
    ax.set_title("2. Split along the decision direction u\n(L24, natural closes; pre-registered)", fontsize=10)
    ax.text(0.98, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
    ax.legend(handles=[Patch(color=INK, label="ablation removes"),
                       Patch(facecolor="white", hatch="//", edgecolor=INK, label="steering moves (2 decades)")],
              fontsize=8, frameon=False, loc="center right")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[2]
    slopes = walk["within_edit"]
    bars = [("curve", "curve,\nall prompts", "slope", ORANGE), ("curve", "curve,\nidentical edits", "within", ORANGE),
            ("decision", "pure decision\npush, identical edits", "within", PINK),
            ("random 0", "random,\nidentical edits", "within", GRAY)]
    for j, (name, label, kind, color) in enumerate(bars):
        m, (lo, hi) = slopes[name][kind], slopes[name][kind + "_ci"]
        ax.bar(j, m, 0.6, color=color, alpha=1 if kind == "slope" else 0.55, yerr=[[m - lo], [hi - m]],
               error_kw={"elinewidth": 0.8, "ecolor": INK})
        ax.text(j, hi + 0.004, f"{m:.2f}", ha="center", fontsize=9)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_ylim(-0.02, 0.18)
    ax.set_xticks(range(len(bars)), [b[1] for b in bars], fontsize=8.5)
    ax.set_ylabel("slope on the real regime-change effect\n(1 = follows this prompt's delays fully)")
    ax.set_title("3. Does the effect follow this prompt's delays?\n"
                 "Same edit, different delays: the curve acts like a decision push", fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Horizon or decision at the L24 answer slot: the subspace holds both, "
                 "and what moves the choice is mostly the average decision per horizon", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- 4. E10: before reasoning -------------------------------------------------------------

UNIT_WORDS = ("second", "minute", "hour", "day", "week", "month", "year", "decade", "century")


def split_horizon(text):
    """'59 years' -> ('59', 'year'); None when there is no unit word."""
    match = re.match(r"([\d.,]+)\s+([a-z]+)", text)
    if not match:
        return None
    unit = next((u for u in UNIT_WORDS if match.group(2).startswith(u[:-1] if u == "century" else u)), None)
    return (match.group(1), unit) if unit else None


def states_number_with_donor_unit(reasoning, recipient, donor):
    """True when the reasoning states the recipient's number with the donor's unit ('59 years' -> '59 days')."""
    r, d = split_horizon(recipient), split_horizon(donor)
    if r is None or d is None:
        return False
    stem = "centur" if d[1] == "century" else d[1]
    return re.search(rf"\b{re.escape(r[0])}[\s-]+{stem}", reasoning) is not None


def e10_conditions(root, tokenizer):
    folder = root / "exp10_Qwen3-8B_reasoning" / "conditions"
    out = {}
    for file in sorted(folder.glob("*.json")):
        rows = json.loads(file.read_text())["rows"]
        arrays = np.load(file.with_suffix(".npz"))
        ids, offsets, prompt_len = arrays["ids"], arrays["offsets"], arrays["prompt_len"]
        for i, row in enumerate(rows):
            reasoning = tokenizer.decode(ids[offsets[i] + prompt_len[i]:offsets[i + 1]])
            row["units_differ"] = split_horizon(row["recipient_horizon"])[1] != split_horizon(row["donor_horizon"])[1]
            row["donor_unit"] = states_number_with_donor_unit(reasoning, row["recipient_horizon"], row["donor_horizon"])
        out[file.stem.replace("_", " ")] = rows
    return out


def hybrid_rate(rows):
    """Share of traces stating the recipient's number with the donor's unit, over pairs whose units differ
    (with the same unit that horizon is just the recipient's)."""
    return np.mean([r["donor_unit"] for r in rows if r["units_differ"]])


def plot_e10(conditions, path, lines):
    base, donor = conditions["recipient"], conditions["donor text"]
    flips = [i for i, (b, d) in enumerate(zip(base, donor)) if b["label"] != d["label"]]
    order = [("donor text", "donor's\nhorizon text"),
             ("full phrase L16", "whole\nphrase\nL16"), ("full phrase L12", "whole\nphrase\nL12"),
             ("full last L16", "last\ntoken\nL16"), ("full last L12", "last\ntoken\nL12"),
             ("subspace last L16", "last token,\nhorizon\n8 dims, L16"),
             ("subspace last L12", "last token,\nhorizon\n8 dims, L12"),
             ("random last L16", "last token,\nrandom\n8 dims, L16"),
             ("random last L12", "last token,\nrandom\n8 dims, L12")]
    colors = [INK, BLUE, BLUE, AQUA, AQUA, ORANGE, ORANGE, GRAY, GRAY]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(order))
    for j, ((name, _), color) in enumerate(zip(order, colors)):
        rows = conditions[name]
        reproduced = sum(rows[i]["label"] == donor[i]["label"] for i in flips) / len(flips)
        stable = [i for i in range(len(rows)) if i not in flips]
        spurious = sum(rows[i]["label"] != base[i]["label"] for i in stable) / len(stable)
        ax0.bar(j - 0.19, reproduced, 0.36, color=color)
        ax0.bar(j + 0.19, spurious, 0.36, color=color, alpha=0.35, hatch="//", edgecolor=color)
        ax0.text(j - 0.19, reproduced + 0.01, f"{reproduced:.0%}", ha="center", fontsize=8)
        named = [np.mean([r["names_recipient"] for r in rows]), np.mean([r["names_donor"] for r in rows]),
                 hybrid_rate(rows)]
        lines.append(f"E10 {name:18s} reproduces {reproduced:.2f} of {len(flips)} donor flips, spurious flips "
                     f"{spurious:.2f} of {len(stable)}; reasoning names recipient {named[0]:.2f}, donor {named[1]:.2f}, "
                     f"recipient number + donor unit {named[2]:.2f}")
    ax0.set_xticks(x, [o[1] for o in order], fontsize=8)
    ax0.set_ylim(0, 1.1)
    ax0.set_ylabel("share of pairs")
    ax0.set_title("Patching the horizon before reasoning changes what the model decides after ~1.7k tokens "
                  "of reasoning", fontsize=10)
    ax0.legend(handles=[Patch(color=INK, label=f"choice flips reproduced (of the {len(flips)} pairs where "
                                                 "the donor text flips it)"),
                        Patch(facecolor="white", hatch="//", edgecolor=INK,
                              label=f"spurious flips (of the {len(base) - len(flips)} pairs where it does not)")],
               fontsize=8, frameon=False, loc="upper right")
    ax0.grid(axis="y", alpha=0.2)

    shown = [("recipient", "unpatched"), ("subspace last L16", "subspace,\nL16"), ("subspace last L12", "subspace,\nL12"),
             ("full last L16", "last token,\nL16"), ("random last L16", "random,\nL16"), ("donor text", "donor text")]
    kinds = [("names_recipient", "states its own horizon", BLUE), ("donor_unit", "states its own number with "
             "the donor's unit (pairs whose units differ)", ORANGE), ("names_donor", "states the donor's horizon", INK)]
    x = np.arange(len(shown))
    for j, (key, label, color) in enumerate(kinds):
        heights = [hybrid_rate(conditions[name]) if key == "donor_unit" else np.mean([r[key] for r in conditions[name]])
                   for name, _ in shown]
        ax1.bar(x + (j - 1) * 0.26, heights, 0.24, color=color, label=label)
    ax1.set_xticks(x, [s[1] for s in shown], fontsize=8)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("share of reasoning traces")
    ax1.set_title("What the reasoning says the horizon is\n"
                  "(e.g. '59 years' patched toward '10 days' -> 'The time horizon is 59 days')", fontsize=10)
    ax1.legend(fontsize=8, frameon=False, loc="upper center")
    ax1.grid(axis="y", alpha=0.2)
    fig.suptitle("E10, thinking on: swap the horizon at L12/L16 for one in the other regime, "
                 "then let the model reason (88-96 pairs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(root):
    root = Path(root)
    FIGURES.mkdir(exist_ok=True)
    lines = []
    plot_decodable_vs_causal(root, FIGURES / "pres_1_decodable_vs_causal.png")
    plot_steering_by_layer(root, FIGURES / "pres_2_steering_by_layer.png", lines)
    plot_horizon_vs_decision(root, FIGURES / "pres_3_horizon_vs_decision.png")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    plot_e10(e10_conditions(root, tokenizer), FIGURES / "pres_4_e10_before_reasoning.png", lines)
    (root / "analysis" / "presentation_figures.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
