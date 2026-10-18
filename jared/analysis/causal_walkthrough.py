"""Figures and numbers for 09_causal_tests_explained.md: the behavior baseline, the
prompts the patching used, the patching map, the ablations, and the tests of
whether the L24 effect is the horizon or the decision (E8 run 3).

Reads only saved results: the thinking-on E0 run (behavior and activations), the
three E8 runs, the run-2 and run-3 inputs, and the 8B tokenizer (to rebuild the
patching pairs exactly as the box did, same seed). No model.

    python analysis/causal_walkthrough.py results/horizon
    -> jared/figures/causal_*.png, <root>/analysis/causal_walkthrough.json and .txt
"""

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from horizon_shape import cell_data  # noqa: E402  (same folder)
from steering_analysis import cv_r2, plot_patching, saved_decision  # noqa: E402  (same folder)
from spar_horizon import steering as S  # noqa: E402
from spar_horizon.banks import build_records  # noqa: E402
from spar_horizon.config import Settings  # noqa: E402
from spar_horizon.full_bank import HORIZONS, regime  # noqa: E402

JARED = Path(__file__).resolve().parent.parent
FIGURES = JARED / "figures"
RUN3_INPUTS = JARED / "steering" / "inputs_Qwen3-8B_decision_L24.npz"
RUN2_INPUTS = JARED / "steering" / "inputs_Qwen3-8B_answer_slot.npz"
RUNS = {"first": "exp8_Qwen3-8B_full", "second": "exp8_Qwen3-8B_answer_slot", "third": "exp8_Qwen3-8B_decision_L24"}
CELL = "on|pre-label|24"
ALPHAS = ("-2", "-1", "1", "2")
REGIME_COLORS = {"none": "#9a9893", "near": "#2a78d6", "both": "#eb6834"}
DIRECTION_COLORS = {"curve": "#eb6834", "curve without decision": "#1baf7a", "decision part of curve": "#7b4fd6",
                    "decision": "#d6437b", "random 0": "#9a9893"}
INK = "#52514e"
TICKS = {"1 min": 1 / 525_960, "1 h": 1 / 8_766, "1 day": 1 / 365.25, "1 mo": 1 / 12, "1 y": 1, "10 y": 10,
         "100 y": 100}


def horizon_ticks(ax, axis="x"):
    positions, labels = [np.log10(v) for v in TICKS.values()], list(TICKS)
    (ax.set_xticks if axis == "x" else ax.set_yticks)(positions, labels)


# --- data ------------------------------------------------------------------------

def bank_records(meta):
    """The full-bank records in the order E8 used (the near/both prompts in `meta`)."""
    cfg = Settings(model=meta["model"], thinking="off", prefill="I choose:", bank="full", n_prompts=meta["n_prompts"],
                   seed=meta["seed"], limit=meta["limit"])
    records = build_records(cfg)
    selected = [records[i] for i in meta["selected"]]
    if [r["sample_uid"] for r in selected] != meta["sample_uids"]:
        raise ValueError("the prompt bank differs from the one the inputs were prepared on")
    return records, selected, cfg


def behavior(root, records):
    """Unedited thinking-on behavior of the 471 horizon prompts (saved E0 label logits)."""
    run = root / "exp0_Qwen3-8B_think-on_full"
    z = np.load(run / "activations.npz")
    _, near_lo, _ = saved_decision(run, z)
    answers = json.loads((run / "answers.json").read_text())
    by_uid = {r["sample_uid"]: r for r in records}
    rows = [by_uid[a["sample_uid"]] for a in answers if a["horizon"] is not None]
    X, log_h, forced = cell_data(z, "pre-label", 24)
    return {"near_lo": near_lo, "log_h": log_h, "forced": forced, "X": X,
            "regime": np.array([r["regime"] for r in rows]),
            "near_delay": np.array([r["near_delay_years"] for r in rows]),
            "far_delay": np.array([r["far_delay_years"] for r in rows])}


# --- 1. baseline behavior -----------------------------------------------------------

def behavior_numbers(b):
    means = {g: float(b["near_lo"][b["regime"] == g].mean()) for g in ("none", "near", "both")}
    groups = np.round(b["log_h"], 6)
    mixed = [h for h in np.unique(groups) if len(set(b["regime"][groups == h])) > 1]
    return {"mean_near_lo": means, "n": {g: int((b["regime"] == g).sum()) for g in means},
            "contrast": means["near"] - means["both"],
            "horizons_with_several_regimes": len(mixed), "horizons": int(len(np.unique(groups)))}


def plot_behavior(b, numbers, path):
    """Left: near log-odds by regime. Right: against the horizon, to show that one
    horizon falls in different regimes depending on the prompt's delays."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.9]})
    rng = np.random.default_rng(0)
    for x, g in enumerate(("none", "near", "both")):
        m = b["regime"] == g
        jitter = rng.uniform(-0.18, 0.18, m.sum())
        ax0.scatter(x + jitter, b["near_lo"][m], s=9, color=REGIME_COLORS[g], alpha=0.55, linewidths=0)
        ax0.plot([x - 0.3, x + 0.3], [numbers["mean_near_lo"][g]] * 2, color=INK, lw=2)
        ax0.text(x + 0.33, numbers["mean_near_lo"][g], f"{numbers['mean_near_lo'][g]:+.1f}", va="center", fontsize=8)
    ax0.annotate("", xy=(1.5, numbers["mean_near_lo"]["near"]), xytext=(1.5, numbers["mean_near_lo"]["both"]),
                 arrowprops={"arrowstyle": "<->", "color": INK})
    ax0.text(1.55, np.mean(list(numbers["mean_near_lo"].values())[1:]), f"C = {numbers['contrast']:.1f}", fontsize=9)
    ax0.set_xticks(range(3), [f"{g}\n(n={numbers['n'][g]})" for g in ("none", "near", "both")])
    ax0.set_xlim(-0.5, 2.9)
    ax0.axhline(0, color=INK, lw=0.5)
    ax0.set_ylabel("near log-odds  logit(near label) - logit(far label)")
    ax0.set_title("Unedited choice by regime (thinking on, 471 prompts)", fontsize=10)
    for g in ("none", "near", "both"):
        m = b["regime"] == g
        ax1.scatter(b["log_h"][m], b["near_lo"][m], s=10, color=REGIME_COLORS[g], alpha=0.65, linewidths=0,
                    label={"none": "none: horizon before the near payout",
                           "near": "near: only the near payout arrives in time",
                           "both": "both: both payouts arrive in time"}[g])
    horizon_ticks(ax1)
    ax1.axhline(0, color=INK, lw=0.5)
    ax1.set_xlabel("stated horizon")
    ax1.set_title(f"The same horizon lands in different regimes: {numbers['horizons_with_several_regimes']} of "
                  f"{numbers['horizons']} horizons hold prompts of two or three regimes", fontsize=10)
    ax1.legend(fontsize=8, frameon=False, loc="center left")
    ax1.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- 2. the patched prompts -----------------------------------------------------------

def patching_prompts(selected, cfg, tokenizer, run2_post_record):
    """The pairs of both patching experiments, rebuilt with the box's seeds.
    T1 (thinking off): each prompt and a copy with a horizon from the other regime.
    T5 (answer slot): each prompt and a different prompt of the other regime."""
    pairs = S.crossing_pairs(selected, tokenizer, cfg, random.Random(0), 96)
    t1 = []
    for p in pairs:
        rec = selected[p["record"]]
        donor_years = next(h.years for h in HORIZONS if str(h) == p["donor_horizon"])
        start, end = p["span"]
        t1.append({"recipient_regime": rec["regime"], "recipient_horizon": rec["horizon_text"],
                   "donor_horizon": p["donor_horizon"], "shift_decades": float(np.log10(donor_years / rec["horizon"])),
                   "span_tokens": end - start,
                   "recipient_span": tokenizer.decode(p["recipient"][start:end]),
                   "donor_span": tokenizer.decode(p["donor"][start:end])})
    on_rows = [selected[j] for j in run2_post_record]
    t5 = {}
    for mode, rows, limit in (("off", selected, 96), ("on", on_rows, 64)):
        pairs5 = S.answer_pairs([r["regime"] for r in rows], [r["near_label"] for r in rows], random.Random(0), limit)
        t5[mode] = {"n": len(pairs5),
                    "recipient_near": sum(rows[r]["regime"] == "near" for r, _ in pairs5),
                    "same_horizon": sum(rows[r]["horizon"] == rows[d]["horizon"] for r, d in pairs5)}
    shifts = np.array([p["shift_decades"] for p in t1])
    return {"t1": {"n": len(t1), "recipient_near": sum(p["recipient_regime"] == "near" for p in t1),
                   "shift_decades": {"median_abs": float(np.median(np.abs(shifts))), "min": float(shifts.min()),
                                     "max": float(shifts.max())},
                   "span_tokens": sorted({p["span_tokens"] for p in t1}), "examples": t1[:4]},
            "t5": t5, "example_prompt": selected[pairs[0]["record"]]["prompt"],
            "example_donor_horizon": pairs[0]["donor_horizon"]}


# --- 3. ablation ---------------------------------------------------------------------

def plot_ablation(runs, path):
    """Left: share of the contrast removed by mean-ablating the horizon subspace, per cell
    and k, with the worst random subspace. Right: run 3's split of the k = 4 subspace."""
    cells = [(k, run, v) for run in ("first", "second") for k, v in runs[run]["cells"].items()]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={"width_ratios": [3.2, 1]})
    x = np.arange(len(cells))
    shades = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, 4))
    for j, (k, shade) in enumerate(zip(("1", "2", "4", "8"), shades)):
        values = [res["ablate"][k]["horizon"][0] for _, _, res in cells]
        ax0.bar(x + (j - 1.5) * 0.18, values, 0.17, color=shade, label=f"horizon subspace, k = {k}")
    random_max = [max(v[0] for row in res["ablate"].values() for n, v in row.items() if n.startswith("random"))
                  for _, _, res in cells]
    ax0.scatter(x + 0.45, random_max, marker="_", s=260, color="#9a9893", linewidths=2.5, zorder=3,
                label="random subspace, same k (worst of 3)")
    ax0.axhline(0.5, color=INK, lw=0.8, ls="--")
    names = []
    for key, run, _ in cells:
        mode, label, layer = key.split("|")
        names.append(f"think {mode}\n{label} L{layer}\n(run {1 if run == 'first' else 2})")
    ax0.set_xticks(x, names, fontsize=7.5)
    ax0.set_ylabel("share of the near-vs-both contrast removed")
    ax0.set_title("Mean-ablation of the horizon-average subspace (dashed: the 'necessary' bar, 0.5)", fontsize=10)
    ax0.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    ax0.set_ylim(-0.05, 1.0)
    ax0.grid(axis="y", alpha=0.2)
    row = runs["third"]["cells"][CELL]["ablate"]["4"]
    parts = [("horizon", "horizon subspace\n(4 dims)"), ("horizon without decision", "without u\n(3 dims)"),
             ("decision", "u alone\n(1 dim)"),
             (max((n for n in row if n.startswith("random")), key=lambda n: row[n][0]), "random\n(4 dims, worst)")]
    for j, (name, label) in enumerate(parts):
        mean, low, high = row[name]
        ax1.bar(j, mean, 0.7, color=["#2a78d6", "#1baf7a", "#d6437b", "#9a9893"][j],
                yerr=[[mean - low], [high - mean]], error_kw={"elinewidth": 0.9, "ecolor": INK})
        ax1.text(j, high + 0.01, f"{mean:.2f}", ha="center", fontsize=8)
    ax1.set_xticks(range(len(parts)), [p[1] for p in parts], fontsize=8)
    ax1.set_title("Run 3, think on, pre-label L24", fontsize=10)
    ax1.set_ylim(-0.05, 1.0)
    ax1.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- 4. geometry at L24 -----------------------------------------------------------------

def plot_geometry(b, cell, path):
    """The prompts on the top-2 horizon-average PCs at think-on pre-label L24 (natural
    closes, as run 3 fitted them): colored by horizon with the curve of averages, and
    colored by the near log-odds with the decision direction u."""
    nat = ~b["forced"]
    Z = (b["X"][nat] - cell["mean"]) @ cell["basis"][:2].T
    u2 = cell["basis"][:2] @ cell["decision"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    points = ax0.scatter(Z[:, 0], Z[:, 1], c=b["log_h"][nat], cmap="viridis", s=10, alpha=0.6, linewidths=0)
    ax0.plot(cell["curve_coords"][:, 0], cell["curve_coords"][:, 1], color="#d6437b", lw=1.6, marker="o", ms=3,
             label="curve: the 48 horizon averages")
    bar = fig.colorbar(points, ax=ax0, fraction=0.04)
    bar.set_ticks([np.log10(v) for v in TICKS.values()], labels=list(TICKS))
    ax0.legend(fontsize=8, frameon=False, loc="best")
    ax0.set_title("Colored by the stated horizon", fontsize=10)
    lo = np.clip(b["near_lo"][nat], -15, 15)
    points = ax1.scatter(Z[:, 0], Z[:, 1], c=lo, cmap="coolwarm_r", s=10, alpha=0.7, linewidths=0, vmin=-15, vmax=15)
    scale = 0.45 * np.abs(Z).max() / np.linalg.norm(u2)
    ax1.annotate("", xy=tuple(u2 * scale), xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2})
    ax1.text(*(u2 * scale * 1.08), "u (toward near)", fontsize=9, color=INK)
    fig.colorbar(points, ax=ax1, fraction=0.04, label="near log-odds (clipped at 15)")
    ax1.set_title("Colored by the choice (near log-odds), with the decision direction u", fontsize=10)
    for ax in (ax0, ax1):
        ax.set_xlabel("horizon-average PC1")
        ax.set_ylabel("horizon-average PC2")
        ax.grid(alpha=0.2)
    fig.suptitle("Think on, pre-label ':' at L24, 371 natural closes: horizon and choice live in the same plane",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- 5-6. horizon or decision (run 3) ---------------------------------------------------

def steering_rows(steer, rows, real_by_regime, contrast):
    """One row per (prompt, alpha): the steering effect of every direction, the real
    effect of the regime change that alpha would cause for this prompt (the saved
    behavior's mean difference between the two regimes, in units of C), the
    transition, the identical-edit group (horizon, alpha), and the prompt index."""
    out = {"effect": {d: [] for d in steer}, "real": [], "transition": [], "group": [], "prompt": []}
    for a in ALPHAS:
        for d in steer:
            out["effect"][d].extend(steer[d][a]["effect"])
        for i, r in enumerate(rows):
            target = regime(r["horizon"] * 10 ** int(a), r["near_delay_years"], r["far_delay_years"])
            out["real"].append((real_by_regime[target] - real_by_regime[r["regime"]]) / contrast)
            out["transition"].append(f"{r['regime']}->{target}")
            out["group"].append(f"{np.log10(r['horizon']):.6f}|{a}")
            out["prompt"].append(i)
    return {k: (np.array(v) if k != "effect" else {d: np.array(e) for d, e in v.items()}) for k, v in out.items()}


def slope(effect, real):
    """Least-squares slope of the steering effect on the real effect (with intercept)."""
    e, p = effect - effect.mean(), real - real.mean()
    return float(e @ p / (p @ p))


def within_edit_slope(effect, real, group):
    """The same slope after removing each identical-edit group's mean from both: only
    differences between prompts that received the same edit (same horizon, same
    alpha), which differ only in their delays, rewards and wording, are left."""
    e, p = effect.astype(float).copy(), real.astype(float).copy()
    for g in np.unique(group):
        m = group == g
        e[m] -= e[m].mean()
        p[m] -= p[m].mean()
    return float(e @ p / (p @ p))


def bootstrap_prompts(stat, rows, n=1000, seed=0):
    """95% interval of stat(index) resampling prompts, each with all its alphas."""
    rng = np.random.default_rng(seed)
    by_prompt = [np.flatnonzero(rows["prompt"] == p) for p in range(rows["prompt"].max() + 1)]
    draws = [stat(np.concatenate([by_prompt[p] for p in rng.integers(0, len(by_prompt), len(by_prompt))]))
             for _ in range(n)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def decision_tests(rows):
    out = {}
    for d, effect in rows["effect"].items():
        if d == "mean difference":
            continue
        overall = slope(effect, rows["real"])
        within = within_edit_slope(effect, rows["real"], rows["group"])
        out[d] = {"slope": overall, "slope_ci": bootstrap_prompts(lambda i: slope(effect[i], rows["real"][i]), rows),
                  "within": within, "within_ci": bootstrap_prompts(
                      lambda i: within_edit_slope(effect[i], rows["real"][i], rows["group"][i]), rows)}
    transitions = {}
    for t in np.unique(rows["transition"]):
        m = rows["transition"] == t
        transitions[t] = {"n": int(m.sum()), "real": float(rows["real"][m].mean()),
                          **{d: float(e[m].mean()) for d, e in rows["effect"].items()}}
    return out, transitions


def plot_transitions(transitions, path):
    order = [t for t in ("near->none", "near->near", "near->both", "both->none", "both->near", "both->both") if t in transitions]
    directions = [d for d in DIRECTION_COLORS]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(order))
    width = 0.8 / len(directions)
    for j, d in enumerate(directions):
        ax.bar(x + (j - (len(directions) - 1) / 2) * width, [transitions[t][d] for t in order], width * 0.92,
               color=DIRECTION_COLORS[d], label=d)
    ax.set_xticks(x, [f"{t.replace('->', ' -> ')}\nn = {transitions[t]['n']}\nreal change: {transitions[t]['real']:+.2f}"
                      for t in order], fontsize=8.5)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_ylabel("steering effect on the near log-odds (units of C)")
    ax.set_title("Run 3, L24: steering effect by the regime change the edit implies for each prompt "
                 "(all alphas pooled)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_within_edit(rows, tests, path):
    """Left: the curve's effect against the real effect, after removing each identical-edit
    group's mean (what is left can only come from the prompt's own delays). Right: the
    slope before and after that removal, per direction."""
    effect, real, group = rows["effect"]["curve"], rows["real"], rows["group"]
    e, p = effect.copy(), real.copy()
    for g in np.unique(group):
        m = group == g
        e[m] -= e[m].mean()
        p[m] -= p[m].mean()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.3]})
    informative = np.abs(p) > 1e-9
    ax0.scatter(p[informative], e[informative], s=10, alpha=0.5, color=DIRECTION_COLORS["curve"], linewidths=0)
    lims = np.array([p.min(), p.max()])
    ax0.plot(lims, lims, color=INK, ls="--", lw=0.8, label="slope 1: follows this prompt's delays fully")
    ax0.plot(lims, tests["curve"]["within"] * lims, color=DIRECTION_COLORS["curve"], lw=2,
             label=f"fitted slope {tests['curve']['within']:.2f}")
    ax0.set_xlabel("real effect of the regime change, minus its mean over the same edit")
    ax0.set_ylabel("curve steering effect, minus its mean over the same edit")
    ax0.set_title(f"Curve, within identical edits ({informative.sum()} informative prompt-alpha rows)", fontsize=10)
    ax0.legend(fontsize=8, frameon=False)
    ax0.grid(alpha=0.2)
    directions = list(tests)
    x = np.arange(len(directions))
    for j, (key, name) in enumerate((("slope", "all rows"), ("within", "within identical edits"))):
        mean = np.array([tests[d][key] for d in directions])
        ci = np.array([tests[d][f"{key}_ci"] for d in directions])
        ax1.bar(x + (j - 0.5) * 0.38, mean, 0.36, color=[DIRECTION_COLORS[d] for d in directions],
                alpha=1.0 if j == 0 else 0.45, hatch=None if j == 0 else "//", edgecolor=INK, linewidth=0.4,
                yerr=[mean - ci[:, 0], ci[:, 1] - mean], error_kw={"elinewidth": 0.8, "ecolor": INK}, label=name)
    ax1.set_xticks(x, [d.replace(" of curve", "\nof curve").replace(" without", "\nwithout") for d in directions],
                   fontsize=8)
    ax1.axhline(0, color=INK, lw=0.6)
    ax1.set_ylabel("slope of steering effect on the real effect")
    ax1.set_title("Solid: all rows. Hatched: within identical edits (only the delays differ)", fontsize=10)
    ax1.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- main ----------------------------------------------------------------------------------

def main(root):
    root = Path(root)
    FIGURES.mkdir(exist_ok=True)
    runs = {name: json.loads((root / run / "results.json").read_text()) for name, run in RUNS.items()}
    meta3 = json.loads(RUN3_INPUTS.with_suffix(".json").read_text())
    z3 = np.load(RUN3_INPUTS)
    cell = {name: z3[f"{CELL}|{name}"] for name in ("mean", "basis", "curve_logh", "curve_coords", "decision")}
    records, selected, cfg = bank_records(meta3)
    b = behavior(root, records)
    out = {}

    out["behavior"] = behavior_numbers(b)
    plot_behavior(b, out["behavior"], FIGURES / "causal_1_behavior.png")

    tokenizer = AutoTokenizer.from_pretrained(meta3["model"])
    out["patching_prompts"] = patching_prompts(selected, cfg, tokenizer, np.load(RUN2_INPUTS)["post|record"])
    plot_patching(runs["first"], runs["second"], FIGURES / "causal_2_patching.png")

    plot_ablation(runs, FIGURES / "causal_3_ablation.png")
    out["ablation_run3"] = runs["third"]["cells"][CELL]["ablate"]["4"]
    out["ablation_mean_over"] = {"runs 1-2": "all 471 horizon prompts", "run 3": f"{meta3['cells'][0]['n']} natural closes"}

    plot_geometry(b, cell, FIGURES / "causal_4_geometry_L24.png")
    nat = ~b["forced"]
    Zs = (b["X"][nat] - cell["mean"]) @ cell["basis"].T
    u_coords = cell["basis"] @ cell["decision"]
    Zs_without = Zs - np.outer(Zs @ u_coords, u_coords) / (u_coords @ u_coords)
    out["decision_left_without_u"] = {"near_lo_from_subspace": cv_r2(Zs, b["near_lo"][nat]),
                                      "near_lo_without_u": cv_r2(Zs_without, b["near_lo"][nat]),
                                      "horizon_from_subspace": cv_r2(Zs, b["log_h"][nat]),
                                      "horizon_without_u": cv_r2(Zs_without, b["log_h"][nat])}

    rows3 = [selected[j] for j in z3["post|record"]]
    real_by_regime = out["behavior"]["mean_near_lo"]
    rows = steering_rows(runs["third"]["cells"][CELL]["steer"], rows3, real_by_regime, runs["third"]["contrast"]["on"])
    out["within_edit"], out["transitions"] = decision_tests(rows)
    out["verdicts_run3"] = runs["third"]["cells"][CELL]["verdicts"]
    plot_transitions(out["transitions"], FIGURES / "causal_5_transitions.png")
    plot_within_edit(rows, out["within_edit"], FIGURES / "causal_6_within_edit.png")

    (root / "analysis" / "causal_walkthrough.json").write_text(json.dumps(out, indent=1, default=float))
    text = report(out)
    (root / "analysis" / "causal_walkthrough.txt").write_text(text)
    print(text)
    print(f"\nwrote {FIGURES}/causal_*.png and {root / 'analysis'}/causal_walkthrough.json, .txt")


def report(out):
    b, p = out["behavior"], out["patching_prompts"]
    lines = [f"behavior (thinking on): mean near log-odds " +
             ", ".join(f"{g} {v:+.2f} (n={b['n'][g]})" for g, v in b["mean_near_lo"].items()) +
             f"; C = {b['contrast']:.2f}; {b['horizons_with_several_regimes']} of {b['horizons']} horizons in 2+ regimes",
             f"T1 pairs: {p['t1']['n']}, recipient near {p['t1']['recipient_near']}, |shift| median "
             f"{p['t1']['shift_decades']['median_abs']:.2f} decades (range {p['t1']['shift_decades']['min']:+.2f} to "
             f"{p['t1']['shift_decades']['max']:+.2f}), span tokens {p['t1']['span_tokens']}",
             *(f"  {e['recipient_regime']}: {e['recipient_span']!r} -> {e['donor_span']!r}" for e in p["t1"]["examples"]),
             *(f"T5 {m}: {v['n']} pairs, recipient near {v['recipient_near']}, same horizon {v['same_horizon']}"
               for m, v in p["t5"].items()),
             "run 3 ablation k=4: " + ", ".join(f"{n} {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
                                                 for n, v in out["ablation_run3"].items()),
             "decision left without u: " + ", ".join(f"{k} {v:.2f}" for k, v in out["decision_left_without_u"].items()),
             "transitions (mean effect, units of C):"]
    for t, v in out["transitions"].items():
        lines.append(f"  {t:12} n={v['n']:4} real {v['real']:+.2f}  " +
                     "  ".join(f"{d} {v[d]:+.2f}" for d in DIRECTION_COLORS))
    lines.append("slope of effect on real effect: all rows / within identical edits")
    for d, v in out["within_edit"].items():
        lines.append(f"  {d:24} {v['slope']:+.3f} [{v['slope_ci'][0]:+.3f}, {v['slope_ci'][1]:+.3f}]   "
                     f"{v['within']:+.3f} [{v['within_ci'][0]:+.3f}, {v['within_ci'][1]:+.3f}]")
    return "\n".join(lines)


if __name__ == "__main__":
    main(sys.argv[1])
