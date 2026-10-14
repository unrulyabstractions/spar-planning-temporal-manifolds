"""E7: linear vs MLP probes for log horizon on the top-k principal components,
leave-one-horizon-out, for every layer at the positions in `positions()`
(spec: 04_spec_probes.md; the probes are in spar_horizon/probes.py).

    python analysis/probe_dimensionality.py results/horizon_v3_think
    -> <root>/analysis/probe_dimensionality.json, probe_curves.png, probe_k90.png

    python analysis/probe_dimensionality.py results/horizon --natural-only
    -> thinking on only, prompts whose think block closed on its own; files end in _natural

Cells run in parallel across cores, one thread each; about 25 minutes for
v3 (both modes, 37 layers) on 14 cores.
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
from spar_horizon.probes import KS, analyze_cell  # noqa: E402
from spar_horizon.runs import load_activations  # noqa: E402

LINEAR, MLP, CONTROL, CEILING = "#2a78d6", "#eb6834", "#9a9893", "#52514e"


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


def run_mode(path, natural_only=False):
    """Probe results for one run. `natural_only` drops prompts whose think
    block was force-closed: the forced rate varies with the horizon, so they
    can stand in for it."""
    meta = load_activations(path)
    keep = ~np.isnan(meta["horizons"])
    if natural_only and meta["forced_mask"].size:
        keep &= ~meta["forced_mask"].astype(bool)
    y = np.log10(meta["horizons"][keep])
    pos = positions(meta)
    jobs = [(layer, label) for layer in range(len(meta["activations"])) for label in pos]
    results = Parallel(n_jobs=-2, return_as="generator")(
        delayed(analyze_cell)(meta["activations"][layer][keep][:, pos[label]], y) for layer, label in jobs)
    cells = {label: [None] * len(meta["activations"]) for label in pos}
    for (layer, label), res in tqdm(zip(jobs, results), total=len(jobs), desc=path.parent.name, unit="cell"):
        cells[label][layer] = res
    return {"tokens": meta["tokens"], "positions": pos, "n": int(keep.sum()),
            "n_horizons": int(len(np.unique(np.round(y, 9)))), "ks": list(KS), "cells": cells}


def best_layer(per_layer):
    return int(np.argmax([c["best"] for c in per_layer]))


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
        best = np.array([[c["best"] for c in r["cells"][lab]] for lab in labels])
        panels = [("best held-out R² (interior)", best, "Blues", (0, 1))]
        for probe, name in (("linear", "linear"), ("mlp", "MLP")):
            k90 = np.array([[np.nan if c["k90"][probe]["act"] is None else c["k90"][probe]["act"]
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


def main(root, natural_only=False):
    root = Path(root)
    results = {}
    # forced closes exist only with thinking on, so the natural-only run skips thinking off
    for mode in (("on",) if natural_only else ("off", "on")):
        hits = sorted(root.glob(f"exp0_*_think-{mode}*/activations.npz"))
        if hits:
            results[mode] = run_mode(hits[0], natural_only)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    tag = "_natural" if natural_only else ""
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
    args = parser.parse_args()
    main(args.root, args.natural_only)
