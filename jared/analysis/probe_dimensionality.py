"""E7: linear vs MLP probes for log horizon on the top-k principal components,
leave-one-horizon-out, for every layer at the positions in `positions()`
(spec: 04_spec_probes.md; the probes are in spar_horizon/probes.py).

    python analysis/probe_dimensionality.py results/horizon_v3_think
    -> <root>/analysis/probe_dimensionality.json, probe_curves.png, probe_k90.png

    python analysis/probe_dimensionality.py results/horizon --natural-only
    -> thinking on only, prompts whose think block closed on its own; files end in _natural

    python analysis/probe_dimensionality.py results/horizon --model Qwen/Qwen3-14B --tag 14b
    -> another model's runs (default Qwen/Qwen3-8B); the tag keeps the 8B files

    python analysis/probe_dimensionality.py results/horizon --modes on --layers 4,8,16 \
        --positions "suffix newline,pre-label" --tag within
    -> a subset of cells; files end in _within (and _natural_within with --natural-only)

On the full bank every cell also gets within-unit R² (horizons that share a
unit word, see probes.within_unit_r2). Cells run in parallel across cores,
one thread each; about 25 minutes for v3 (both modes, 37 layers) on 14
cores, about 3 minutes per cell on the full bank.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spar_horizon.full_bank import HORIZONS  # noqa: E402
from spar_horizon.probes import KS, analyze_cell  # noqa: E402
from spar_horizon.runs import load_activations  # noqa: E402

LINEAR, MLP, CONTROL, CEILING = "#2a78d6", "#eb6834", "#9a9893", "#52514e"
# log10 years of each full-bank horizon -> its unit word ("84 minutes" -> "minute")
UNIT_BY_LOG_H = {round(float(np.log10(h.years)), 6): str(h).split()[-1].rstrip("s") for h in HORIZONS}


def units_of(log_h):
    """Unit word per prompt, or None when a horizon is not on the full-bank grid (the starter bank)."""
    keys = [round(float(v), 6) for v in log_h]
    if not all(k in UNIT_BY_LOG_H for k in keys):
        return None
    return np.array([UNIT_BY_LOG_H[k] for k in keys])


def positions(meta):
    """Label -> kept-layout index: the suffix newline after <|im_end|> (the
    probe position), the last suffix token, each think fraction, the think
    close, and the pre-label token."""
    regions, tokens = meta["regions"], meta["tokens"]
    out = {"suffix newline": regions["suffix"][0] + 1, "suffix last": regions["suffix"][1] - 1}
    for k in range(*regions["think"]):
        out[tokens[k].rstrip("*")] = k
    if regions["pre"][1] > regions["pre"][0]:
        out["think close"] = regions["pre"][0]
    out["pre-label"] = regions["prelabel"][0]
    return out


def run_file(root, mode, model="Qwen/Qwen3-8B"):
    """The activations of `model`'s E0 run for `mode` under `root`, or None; a results folder can hold several models."""
    hits = sorted(Path(root).glob(f"exp0_{model.split('/')[-1]}_think-{mode}*/activations.npz"))
    return hits[0] if hits else None


def run_mode(path, natural_only=False, layers=None, labels=None):
    """Probe results for one run, at every layer and position unless `layers` /
    `labels` select some (skipped cells stay None). `natural_only` drops
    prompts whose think block was force-closed: the forced rate varies with the
    horizon, so they can stand in for it."""
    meta = load_activations(path)
    keep = ~np.isnan(meta["horizons"])
    if natural_only and meta["forced_mask"].size:
        keep &= ~meta["forced_mask"].astype(bool)
    y = np.log10(meta["horizons"][keep])
    units = units_of(y)
    pos = {label: index for label, index in positions(meta).items() if labels is None or label in labels}
    layers = range(len(meta["activations"])) if layers is None else layers
    jobs = [(layer, label) for layer in layers for label in pos]
    results = Parallel(n_jobs=-2, return_as="generator")(
        delayed(analyze_cell)(meta["activations"][layer][keep][:, pos[label]], y, units) for layer, label in jobs)
    cells = {label: [None] * len(meta["activations"]) for label in pos}
    for (layer, label), res in tqdm(zip(jobs, results), total=len(jobs), desc=path.parent.name, unit="cell"):
        cells[label][layer] = res
    return {"tokens": meta["tokens"], "positions": pos, "n": int(keep.sum()),
            "n_horizons": int(len(np.unique(np.round(y, 9)))), "ks": list(KS), "cells": cells}


def best_layer(per_layer):
    """Index of the computed layer with the highest best R²."""
    return max((layer for layer, c in enumerate(per_layer) if c is not None), key=lambda layer: per_layer[layer]["best"])


def best_within(cell, probe, k):
    """Within-unit R² of `probe` at `k` components, the better of the two orderings."""
    values = [cell["r2_within"][probe][order][KS.index(k)] for order in ("act", "cent")]
    return max(v for v in values if not np.isnan(v))


def summarize(results):
    lines = []
    for mode, r in results.items():
        lines.append(f"\n{mode}: {r['n']} prompts, {r['n_horizons']} horizons. Interior R² at the best layer:")
        lines.append(f"  {'position':16} {'layer':>5} {'ceiling':>7} {'best':>6}  k90 linear act/cent   k90 mlp act/cent"
                     f"   control max")
        for label, per_layer in r["cells"].items():
            layer = best_layer(per_layer)
            c = per_layer[layer]
            k90 = c["k90"]
            control = max(v for v in c["control"]["mlp"]["act"] + c["control"]["linear"]["act"] if not np.isnan(v))
            lines.append(f"  {label:16} {layer:>5} {c['ceiling_interior']:>7.2f} {c['best']:>6.2f}  "
                         f"{str(k90['linear']['act']):>8} / {str(k90['linear']['cent']):<6}  "
                         f"{str(k90['mlp']['act']):>8} / {str(k90['mlp']['cent']):<6}  {control:>8.2f}")
        if "r2_within" not in next(c for per_layer in r["cells"].values() for c in per_layer if c is not None):
            continue
        lines.append(f"\n  Within-unit R² (interior; a unit lookup scores 0), at the layer with the best within-unit ceiling;"
                     f"\n  per k the better of the two PC orderings:")
        lines.append(f"  {'position':16} {'layer':>5} {'ceiling':>7}   linear k=1/2/4/8          MLP k=1/2/4/8")
        for label, per_layer in r["cells"].items():
            layer = max((i for i, c in enumerate(per_layer) if c is not None),
                        key=lambda i: per_layer[i]["ceiling_within"])
            c = per_layer[layer]
            curves = {probe: " ".join(f"{best_within(c, probe, k):5.2f}" for k in (1, 2, 4, 8)) for probe in ("linear", "mlp")}
            lines.append(f"  {label:16} {layer:>5} {c['ceiling_within']:>7.2f}   {curves['linear']}   {curves['mlp']}")
    return "\n".join(lines)


def plot_curves(results, path):
    """R² (interior) against k at each position's best layer."""
    rows = [(mode, label) for mode, r in results.items() for label in r["cells"]]
    ncol = 4
    nrow = -(-len(rows) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False, sharey=True)
    for ax, (mode, label) in zip(axes.flat, rows):
        per_layer = results[mode]["cells"][label]
        layer = best_layer(per_layer)
        c = per_layer[layer]
        for probe, color, name in (("linear", LINEAR, "linear"), ("mlp", MLP, "MLP")):
            for order, style in (("act", "-"), ("cent", "--")):
                ax.plot(KS, c["r2_interior"][probe][order], style, color=color, lw=2, marker="o", ms=4,
                        label=f"{name}, {'variance' if order == 'act' else 'horizon-mean'} PCs")
            ax.plot(KS, c["control"][probe]["act"], ":", color=CONTROL, lw=1.5,
                    label="shuffled-horizon control" if probe == "linear" else None)
        ax.axhline(c["ceiling_interior"], color=CEILING, lw=1, ls="-.", label="linear, all dims")
        ax.set_xscale("log")
        ax.set_xticks(KS[::2], [str(k) for k in KS[::2]])
        ax.set_ylim(-0.2, 1.02)
        ax.set_title(f"{mode}: {label}, L{layer}", fontsize=9)
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(rows):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("held-out R² (interior)")
    for ax in axes[-1]:
        ax.set_xlabel("top-k components")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Log-horizon probes on the top-k principal components, leave-one-horizon-out", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_k90(results, path):
    """Per mode: best interior R², and k90 of each probe (variance PCs), layer × position."""
    fig, axes = plt.subplots(len(results), 3, figsize=(15, 4.2 * len(results)), squeeze=False)
    for row, (mode, r) in zip(axes, results.items()):
        labels = list(r["cells"])
        best = np.array([[np.nan if c is None else c["best"] for c in r["cells"][lab]] for lab in labels])
        panels = [("best held-out R² (interior)", best, "Blues", (0, 1))]
        for probe, name in (("linear", "linear"), ("mlp", "MLP")):
            k90 = np.array([[np.nan if c is None or c["k90"][probe]["act"] is None else c["k90"][probe]["act"]
                             for c in r["cells"][lab]] for lab in labels], float)
            panels.append((f"{name}: k to reach 90% of best", np.log2(k90), "Blues_r", (0, np.log2(max(KS)))))
        for ax, (title, data, cmap, (lo, hi)) in zip(row, panels):
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("#e8e6e1")
            im = ax.imshow(data, aspect="auto", cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_yticks(range(len(labels)), labels, fontsize=8)
            ax.set_xlabel("layer")
            ax.set_title(f"{mode}: {title}", fontsize=9)
            bar = fig.colorbar(im, ax=ax, fraction=0.04)
            if "k to reach" in title:
                ticks = [1, 2, 4, 8, 16, 32, 64]
                bar.set_ticks(np.log2(ticks), labels=[str(t) for t in ticks])
                bar.set_label("k (gray: never, or nothing readable)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(root, natural_only=False, modes=("off", "on"), layers=None, labels=None, tag="", model="Qwen/Qwen3-8B"):
    root = Path(root)
    results = {}
    # forced closes exist only with thinking on, so the natural-only run skips thinking off
    for mode in (("on",) if natural_only else modes):
        path = run_file(root, mode, model)
        if path:
            results[mode] = run_mode(path, natural_only, layers, labels)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    tag = ("_natural" if natural_only else "") + (f"_{tag}" if tag else "")
    (out / f"probe_dimensionality{tag}.json").write_text(json.dumps(results, default=float))
    plot_curves(results, out / f"probe_curves{tag}.png")
    plot_k90(results, out / f"probe_k90{tag}.png")
    text = ("natural think closes only\n" if natural_only else "") + summarize(results)
    (out / f"probe_dimensionality{tag}.txt").write_text(text)
    print(text)
    print(f"\nwrote {out}/probe_dimensionality{tag}.json, probe_curves{tag}.png, probe_k90{tag}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="results folder holding exp0_*_think-<mode>*/activations.npz")
    parser.add_argument("--natural-only", action="store_true",
                        help="thinking on only, without force-closed prompts; writes *_natural files")
    parser.add_argument("--modes", default="off,on", help="comma-separated thinking modes")
    parser.add_argument("--layers", default=None, help="comma-separated layer indices (default: all)")
    parser.add_argument("--positions", default=None, help="comma-separated position labels (default: all)")
    parser.add_argument("--tag", default="", help="suffix for the output files, so a subset run keeps the full one")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="which model's runs to read; add --tag to keep another model's files apart")
    args = parser.parse_args()
    main(args.root, args.natural_only, tuple(args.modes.split(",")),
         [int(x) for x in args.layers.split(",")] if args.layers else None,
         args.positions.split(",") if args.positions else None, args.tag, args.model)
