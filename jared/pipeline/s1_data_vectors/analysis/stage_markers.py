"""Stage markers of the horizon pipeline, for Qwen3-8B and Qwen3-14B (10_spec_scale_14b.md
sections 2 and 5, S1 and S5), from saved results only.

- encode_end: last layer where the horizon tokens carry >= 0.5 of the T1 patching transfer (thinking off)
- spread_peak: layer of the largest T1 transfer for the "prompt after the horizon" group (thinking off)
- handoff: first layer where the answer-slot (T5) transfer is >= 0.5, per mode
- bind: first layer where the a-vs-b log-odds at pre-label decodes at 5-fold CV R² >= 0.9 from the
  top-4 horizon-average PCs (thinking on: natural closes only)

Layers are hidden_states indices; relative depth = layer / number of decoder layers. A fractional layer
is the linear interpolation of the 0.5 (or 0.9) crossing between neighbouring layers.

    python -m pipeline.s1_data_vectors.analysis.stage_markers results/horizon
    -> <root>/analysis/stage_markers.json and stage_markers.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pipeline.s1_data_vectors.analysis.horizon_shape import cell_data, horizon_components
from pipeline.s2_steering.analysis.steering_analysis import cv_r2, saved_decision

MODELS = ("Qwen3-8B", "Qwen3-14B")
PATCH_RUN = {"Qwen3-8B": "exp8_Qwen3-8B_full", "Qwen3-14B": "exp8_Qwen3-14B_patch"}
ANSWER_RUN = {"Qwen3-8B": "exp8_Qwen3-8B_answer_slot", "Qwen3-14B": "exp8_Qwen3-14B_answer_slot_off"}
HORIZON_GROUP, AFTER_GROUP = 0, 1   # groups of the T1 patch results: horizon tokens, prompt after the horizon
TRANSFER, BIND_R2, K = 0.5, 0.9, 4
MARKERS = ("encode_end", "spread_peak", "handoff", "bind")
ORDER = {"off": MARKERS, "on": ("handoff", "bind")}
SHIFTED, FAILED = 0.08, 0.15   # S1: within SHIFTED of the 8B depth passes, beyond FAILED fails
COLORS = {"encode_end": "#2a78d6", "spread_peak": "#1baf7a", "handoff": "#eb6834", "bind": "#52514e"}


def exp0_run(root, model, mode):
    return Path(root) / f"exp0_{model}_think-{mode}_full"


def n_layers(z):
    """Decoder layers: hidden_states L0..Ln are stored."""
    return sum(k[0] == "L" and k[1:].isdigit() for k in z.files) - 1


def first_crossing(values, threshold):
    """(first layer with value >= threshold, its fractional position), or None. NaN never crosses."""
    values = np.asarray(values, float)
    above = np.flatnonzero(values >= threshold)
    if not len(above):
        return None
    i = int(above[0])
    if i == 0 or np.isnan(values[i - 1]):
        return i, float(i)
    return i, i - 1 + (threshold - values[i - 1]) / (values[i] - values[i - 1])


def last_crossing(values, threshold):
    """(last layer with value >= threshold, the fractional layer where it drops below), or None."""
    values = np.asarray(values, float)
    above = np.flatnonzero(values >= threshold)
    if not len(above):
        return None
    i = int(above[-1])
    if i == len(values) - 1:
        return i, float(i)
    return i, i + (values[i] - threshold) / (values[i] - values[i + 1])


def patch_markers(root, model):
    """encode_end and spread_peak from the T1 (thinking-off) patching map."""
    transfer = json.loads((Path(root) / PATCH_RUN[model] / "results.json").read_text())["patch"]["transfer"]
    horizon = [row[HORIZON_GROUP][0] for row in transfer]
    after = np.array([row[AFTER_GROUP][0] for row in transfer])
    peak = int(np.nanargmax(after))
    return {"encode_end": last_crossing(horizon, TRANSFER), "spread_peak": (peak, float(peak))}


def handoff(root, model, mode):
    transfer = json.loads((Path(root) / ANSWER_RUN[model] / "results.json").read_text())["answer"][mode]["transfer"]
    return first_crossing([row[0] for row in transfer], TRANSFER)


def ab_decode_r2(root, model, mode):
    """CV R² of the a-vs-b log-odds from the top-4 horizon-average PCs at pre-label, per layer
    (index = layer, NaN at layer 0). The PCs and the prompts are those of steering_prepare's cells."""
    run = exp0_run(root, model, mode)
    z = np.load(run / "activations.npz")
    ab = saved_decision(run, z)[0]
    out = np.full(n_layers(z) + 1, np.nan)
    for layer in tqdm(range(1, len(out)), desc=f"bind {model} {mode}"):
        X, log_h, forced = cell_data(z, "pre-label", layer)
        keep = ~forced & np.isfinite(ab)   # ab is NaN where the saved label logits are the wrong token pair
        Z = horizon_components(X[keep], log_h[keep], K)[0]
        out[layer] = cv_r2(Z, ab[keep])
    return out


def verdict_s1(markers):
    """S1 from {model: {mode: {marker: (layer, fractional)}}}: stage order in both modes, and every
    14B marker within SHIFTED (pass) or beyond FAILED (fail) of the 8B relative depth. Fractional layers
    are used so that two markers in the same integer layer are still ordered."""
    depth = {m: {mode: {k: markers[m][mode][k][1] / markers[m]["n_layers"] for k in ORDER[mode]} for mode in ORDER}
             for m in MODELS}
    inverted = [mode for mode in ORDER
                if any(depth["Qwen3-14B"][mode][a] >= depth["Qwen3-14B"][mode][b] for a, b in zip(ORDER[mode], ORDER[mode][1:]))]
    moves = {f"{mode} {k}": float(depth["Qwen3-14B"][mode][k] - depth["Qwen3-8B"][mode][k]) for mode in ORDER for k in ORDER[mode]}
    worst = max(abs(v) for v in moves.values())
    if inverted or worst > FAILED:
        label = "FAIL"
    elif worst <= SHIFTED:
        label = "PASS"
    else:
        label = "SHIFTED"
    return {"verdict": label, "order_inverted_in": inverted, "relative_depth_move": moves, "largest_move": worst}


def verdict_s5(markers):
    gap = markers["Qwen3-14B"]["on"]["handoff"][0] - markers["Qwen3-14B"]["off"]["handoff"][0]
    return {"verdict": "PASS" if abs(gap) <= 2 else "FAIL", "layers_on_minus_off": gap}


def table(markers):
    lines = [f"{'model':10} {'mode':4} {'marker':12} {'layer':>6} {'frac':>6} {'depth':>6} {'frac depth':>10}"]
    for model in MODELS:
        n = markers[model]["n_layers"]
        for mode in ("off", "on"):
            for name in ORDER[mode]:
                layer, frac = markers[model][mode][name]
                lines.append(f"{model:10} {mode:4} {name:12} L{layer:<5d} {frac:6.2f} {layer / n:6.3f} {frac / n:10.3f}")
    return "\n".join(lines)


def plot(markers, path):
    rows = [(m, mode) for mode in ("off", "on") for m in MODELS]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for y, (model, mode) in enumerate(rows):
        n = markers[model]["n_layers"]
        for name in ORDER[mode]:
            layer = markers[model][mode][name][0]
            ax.scatter(layer / n, y, s=90, color=COLORS[name], label=name if y == 0 else None, zorder=3)
            ax.annotate(f"L{layer}", (layer / n, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7)
    ax.set_yticks(range(len(rows)), [f"{m}, thinking {mode}" for m, mode in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("relative depth (layer / decoder layers)")
    ax.set_title("Stage markers by relative depth", fontsize=10)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def compute(root):
    markers = {}
    for model in MODELS:
        patch = patch_markers(root, model)
        n = n_layers(np.load(exp0_run(root, model, "off") / "activations.npz"))
        markers[model] = {"n_layers": n, "off": dict(patch), "on": {}, "ab_r2": {}}
        for mode in ("off", "on"):
            markers[model][mode]["handoff"] = handoff(root, model, mode)
            r2 = ab_decode_r2(root, model, mode)
            markers[model][mode]["bind"] = first_crossing(r2, BIND_R2)
            markers[model]["ab_r2"][mode] = [None if np.isnan(v) else float(v) for v in r2]
    return markers


def main(root):
    out = Path(root) / "analysis"
    out.mkdir(exist_ok=True)
    markers = compute(root)
    s1, s5 = verdict_s1(markers), verdict_s5(markers)
    print(table(markers))
    print(f"\nS1 stage order: {s1['verdict']}; order inverted in {s1['order_inverted_in'] or 'no mode'}; "
          f"14B minus 8B relative depth {({k: round(v, 3) for k, v in s1['relative_depth_move'].items()})}")
    print(f"S5 reasoning does not move the hand-off: {s5['verdict']} "
          f"(14B on minus off = {s5['layers_on_minus_off']} layers; 8B: "
          f"{markers['Qwen3-8B']['on']['handoff'][0] - markers['Qwen3-8B']['off']['handoff'][0]})")
    (out / "stage_markers.json").write_text(json.dumps({"markers": markers, "S1": s1, "S5": s5}, indent=1))
    plot(markers, out / "stage_markers.png")
    print(f"wrote {out}/stage_markers.json and stage_markers.png")


if __name__ == "__main__":
    main(sys.argv[1])
