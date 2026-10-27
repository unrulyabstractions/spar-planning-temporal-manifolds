"""Figures for the evidence index (13_evidence.md) that no other script draws, from saved
results only (no model runs):

- evidence_e11_steering.png: E8 at the 14B answer slot (11 §3), next to 8B thinking on L24;
- evidence_probe_line.png: S4 on 14B, the probe line against the curve, mean difference and random;
- evidence_within_unit.png: within-unit R² per prompt vs on horizon averages (12 §1a);
- evidence_regime.png: the threshold-shift control and regime vs choice AUC (12 §2b).

    python analysis/evidence_figures.py results/horizon
    -> <root>/analysis/evidence_*.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spar_horizon.steering import bootstrap_mean  # noqa: E402

# categorical slots 1-4 of the dataviz reference palette, in fixed order
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY = "#8a8a85"

STEER_CELLS = (("Qwen3-14B", "answer_slot_off", "off|pre-label|27", "14B off L27"),
               ("Qwen3-14B", "answer_slot_off", "off|pre-label|30", "14B off L30"),
               ("Qwen3-14B", "answer_slot_off", "off|pre-label|33", "14B off L33"),
               ("Qwen3-14B", "answer_slot_on", "on|pre-label|27", "14B on L27"),
               ("Qwen3-14B", "on_L30_L33", "on|pre-label|30", "14B on L30"),
               ("Qwen3-14B", "on_L30_L33", "on|pre-label|33", "14B on L33"),
               ("Qwen3-8B", "decision_L24", "on|pre-label|24", "8B on L24"))


def steering_bars(verdicts):
    """The seven columns of 11 §3's table, in its order, from one cell's verdicts."""
    beyond, ablate = verdicts["horizon_beyond_decision"], verdicts["ablate_beyond_decision"]
    return {"ablate horizon subspace": verdicts["necessary"]["removed"],
            "ablate without decision": ablate["removed"],
            "ablate decision alone": ablate["decision_alone"],
            "steer curve": beyond["curve"],
            "steer curve without decision": beyond["without_decision"],
            "steer decision part": beyond["decision_part"],
            "steer random": beyond["random"]}


def plot_steering(root, path):
    rows = []
    for model, run, cell, name in STEER_CELLS:
        results = json.loads((root / f"exp8_{model}_{run}" / "results.json").read_text())
        rows.append((name, steering_bars(results["cells"][cell]["verdicts"])))
    labels = list(rows[0][1])
    colors = [BLUE, BLUE, BLUE, ORANGE, ORANGE, ORANGE, GREY]
    fig, axes = plt.subplots(1, len(rows), figsize=(2.3 * len(rows) + 2, 3.6), sharey=True)
    for ax, (name, bars) in zip(axes, rows):
        y = np.arange(len(labels))
        ax.barh(y, [bars[l] for l in labels], color=colors, height=0.7)
        ax.axvline(0, color="#444", lw=0.8)
        ax.set_title(name, fontsize=9)
        ax.set_xlim(-0.05, 1.0)   # thinking-on L30/L33 ablation reaches 0.88
        ax.grid(axis="x", alpha=0.2)
        for yi, l in zip(y, labels):
            ax.text(max(bars[l], 0) + 0.01, yi, f"{bars[l]:.2f}", va="center", fontsize=7, color="#333")
    axes[0].set_yticks(np.arange(len(labels)), labels, fontsize=8)   # shared y: set once, after the bars
    axes[0].invert_yaxis()
    fig.supxlabel("effect on the choice, fraction of the near-vs-both contrast (blue: ablation, orange: steering |α| = 2)",
                  fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


LINE_CELLS = (("line_off", "off|pre-label|27", "off L27"), ("line_off", "off|pre-label|30", "off L30"),
              ("line_off", "off|pre-label|33", "off L33"), ("line_on", "on|pre-label|27", "on L27"),
              ("line_on", "on|pre-label|30", "on L30"), ("line_on", "on|pre-label|33", "on L33"))
LINE_DIRECTIONS = (("curve", "curve of horizon averages", ORANGE), ("line", "probe line", BLUE),
                   ("mean difference", "mean difference", AQUA), ("random 0", "random", GREY))


def plot_probe_line(root, path):
    """S4 on 14B: steering toward the rational answer at |α| = 2, both signs pooled, 95% bootstrap CI."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(LINE_CELLS))
    width = 0.2
    for j, (direction, name, color) in enumerate(LINE_DIRECTIONS):
        stats = []
        for run, cell, _ in LINE_CELLS:
            steer = json.loads((root / f"exp8_Qwen3-14B_{run}" / "results.json").read_text())["cells"][cell]["steer"]
            stats.append(bootstrap_mean(np.r_[steer[direction]["2"]["toward"], steer[direction]["-2"]["toward"]]))
        mean = np.array([s[0] for s in stats])
        err = np.array([[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]])
        ax.bar(x + (j - 1.5) * width, mean, width * 0.92, color=color, label=name, yerr=err,
               error_kw={"elinewidth": 0.8, "ecolor": "#444"})
    ax.axhline(0, color="#444", lw=0.8)
    ax.axhline(0.05, color=GREY, ls="--", lw=0.8)
    ax.text(len(LINE_CELLS) - 0.5, 0.053, "S4: line <= 0.05", fontsize=7, color="#333", ha="right", va="bottom")
    ax.set_xticks(x, [name for _, _, name in LINE_CELLS], fontsize=8)
    ax.set_ylabel("effect toward the rational answer\n(fraction of the contrast, |α| = 2)", fontsize=8)
    ax.set_title("Qwen3-14B answer slot: steering by direction", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=7, ncol=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


WITHIN_CELLS = (("Qwen3-8B", "on:suffix newline:L16", "8B suffix newline L16"),
                ("Qwen3-8B", "off:pre-label:L21", "8B pre-label off L21"),
                ("Qwen3-8B", "on:pre-label:L21", "8B pre-label on L21"),
                ("Qwen3-14B", "on:suffix newline:L20", "14B suffix newline L20"),
                ("Qwen3-14B", "off:pre-label:L22", "14B pre-label off L22"),
                ("Qwen3-14B", "on:pre-label:L22", "14B pre-label on L22"))
WITHIN_PROBES = (("linear", 0, "linear k=1"), ("mlp", 0, "MLP k=1"), ("linear", 4, "linear k=8"))   # index into k = 1,2,3,4,8,64
FLOOR = -1.0   # values below are drawn at the floor and labelled


def plot_within(root, path):
    levels = json.loads((root / "analysis" / "probe_within_levels.json").read_text())
    fig, axes = plt.subplots(1, len(WITHIN_PROBES), figsize=(4.2 * len(WITHIN_PROBES), 3.8), sharey=True)
    y = np.arange(len(WITHIN_CELLS))
    for ax, (probe, k_index, title) in zip(axes, WITHIN_PROBES):
        for offset, level, color, legend in ((-0.18, "prompt", ORANGE, "per prompt"), (0.18, "horizon", BLUE, "horizon averages")):
            values = [levels[f"Qwen/{model}"][cell]["all"][probe][level][k_index] for model, cell, _ in WITHIN_CELLS]
            ax.barh(y + offset, [max(v, FLOOR) for v in values], height=0.34, color=color, label=legend)
            for yi, v in zip(y, values):
                ax.text(max(v, FLOOR) + (0.02 if v >= 0 else -0.02), yi + offset, f"{v:.2f}", va="center",
                        ha="left" if v >= 0 else "right", fontsize=7, color="#333")
        ax.axvline(0.5, color=GREY, ls="--", lw=0.8)
        ax.axvline(0.3, color=GREY, ls=":", lw=0.8)
        ax.axvline(0, color="#444", lw=0.8)
        ax.set_xlim(FLOOR - 0.35, 1.05)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, [name for _, _, name in WITHIN_CELLS], fontsize=8)   # shared y: set once
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7, loc="lower right")
    fig.supxlabel("within-unit R² (all prompts, interior horizons; bars clipped at -1; dashed 0.5 = suffix criterion, "
                  "dotted 0.3 = staircase criterion)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


SERIES = (("off", "suffix newline", BLUE, "-"), ("off", "pre-label", BLUE, "--"),
          ("on", "suffix newline", ORANGE, "-"), ("on", "pre-label", ORANGE, "--"))
HANDOFF = {"Qwen3-8B": 22, "Qwen3-14B": 25}
FROM_LAYER = 15


def plot_regime(root, path):
    shift = json.loads((root / "analysis" / "regime_comparison_shift.json").read_text())["results"]
    within = json.loads((root / "analysis" / "regime_comparison_within.json").read_text())["results"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for col, model in enumerate(("Qwen3-8B", "Qwen3-14B")):
        ax = axes[0, col]
        for mode, position, color, style in SERIES:
            cells = shift[model][mode][position]
            shifts = sorted(cells[next(iter(cells))], key=float)
            table = np.array([[cells[l][s] for s in shifts] for l in cells if int(l) >= FROM_LAYER])
            ax.plot([float(s) for s in shifts], np.median(table, 0), style, color=color, marker="o", ms=4,
                    label=f"thinking {mode}, {position}")
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_ylim(0.45, 1.02)
        ax.set_xlabel("threshold shift s (decades): label = log h - log far >= s", fontsize=8)
        ax.set_ylabel(f"within-horizon AUC, median over L{FROM_LAYER}+", fontsize=8)
        ax.set_title(f"{model}: peak at the true threshold (s = 0)?", fontsize=9)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
        ax = axes[1, col]
        for mode, position, color, style in SERIES:
            cells = within[model][mode][position]
            layers = sorted(int(l) for l in cells)
            ax.plot(layers, [cells[str(l)]["auc"] for l in layers], style, color=color, lw=1.6,
                    label=f"{mode}, {position}: regime")
            ax.plot(layers, [cells[str(l)]["choice_auc"] for l in layers], style, color=color, lw=0.8, alpha=0.45,
                    label=f"{mode}, {position}: choice")
        ax.axvline(HANDOFF[model], color=GREY, lw=0.8)
        ax.text(HANDOFF[model] + 0.3, 1.0, "hand-off", fontsize=7, color="#333", va="top")
        ax.set_ylim(0.3, 1.02)
        ax.set_xlabel("layer", fontsize=8)
        ax.set_ylabel("within-horizon AUC (projected probe)", fontsize=8)
        ax.set_title(f"{model}: regime AUC (thick) vs choice AUC (thin)", fontsize=9)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(root):
    root = Path(root)
    out = root / "analysis"
    plot_steering(root, out / "evidence_e11_steering.png")
    plot_probe_line(root, out / "evidence_probe_line.png")
    plot_within(root, out / "evidence_within_unit.png")
    plot_regime(root, out / "evidence_regime.png")
    print(f"wrote {out}/evidence_{{e11_steering,probe_line,within_unit,regime}}.png")


if __name__ == "__main__":
    main(sys.argv[1])
