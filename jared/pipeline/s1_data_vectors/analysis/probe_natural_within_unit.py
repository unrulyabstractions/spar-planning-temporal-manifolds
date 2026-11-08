"""E7 follow-up (06 §4-5): are the thinking-on probes reading "was cut off", and how much
of each probe's R² is magnitude within a unit word?

For five cells (suffix newline, last suffix token, think close, pre-label thinking on,
pre-label thinking off) on Qwen3-8B and Qwen3-14B: the probe_dimensionality sweep
(linear vs MLP, top-k PCs, leave-one-horizon-out, penalty chosen by CV grouped by
horizon) with R² and within-unit R² (`probes.within_unit_r2`), on all prompts and on
natural think closes only. The 8B layers are 06's; for the 14B the layer is the one with the
highest best R² on all prompts, as `probe_dimensionality.best_layer` picks it, and the
natural-only run uses the same layer. The 14B pick scores each layer with the linear probe
alone, no full-dimension ceiling and a few k (`SWEEP_KS`): the full analysis takes about 8 min a
cell on d = 5120, which across 41 layers is hours.

    python -m pipeline.s1_data_vectors.analysis.probe_natural_within_unit results/horizon
    -> <root>/analysis/probe_natural_within_unit.json, .png, .txt

    python -m pipeline.s1_data_vectors.analysis.probe_natural_within_unit results/horizon --levels
    -> <root>/analysis/probe_within_levels.json, .txt: within-unit R² per prompt and on
       the horizon averages (`probes.horizon_within_unit_r2`) at the layers the run above
       picked (plus 14B suffix newline L20), in 06's ordering, without the ceiling
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from pipeline.s1_data_vectors.analysis.horizon_shape import cell_data
from pipeline.s1_data_vectors.analysis.probe_dimensionality import LINEAR, MLP, run_file, units_of
from spar_horizon.probes import KS, analyze_cell, probe_curves

MODELS = {"Qwen/Qwen3-8B": 36, "Qwen/Qwen3-14B": 40}   # decoder layers; hidden states are L0..L<n>
CELLS = (("on", "suffix newline"), ("on", "suffix last"), ("on", "think close"),
         ("on", "pre-label"), ("off", "pre-label"))
LAYERS_8B = {("on", "suffix newline"): 16, ("on", "suffix last"): 10, ("on", "think close"): 19,
             ("on", "pre-label"): 21, ("off", "pre-label"): 21}   # 06 §1
# the PC ordering 06 §2 quotes per cell: variance at the suffix, horizon-mean elsewhere
ORDER = {"suffix newline": "act", "suffix last": "act", "think close": "cent", "pre-label": "cent"}
TABLE_KS = (1, 2, 3, 4, 8, 64)
SWEEP_KS = (2, 8, 64)
WORKERS = 4   # another heavy job shares the machine


def load_cell(path, label, layer, natural_only):
    """X, y at one (position, layer) of a run, loading only that layer; natural_only drops force-closed prompts."""
    X, y, forced = cell_data(np.load(path), label, layer)
    keep = ~forced if natural_only else np.ones(len(y), bool)
    return X[keep], y[keep]


def sweep_score(path, label, layer):
    """Best interior R² of the linear probe over SWEEP_KS and both orderings, on all prompts."""
    X, y = load_cell(path, label, layer, False)
    curves = probe_curves(X, y, ks=SWEEP_KS, probes=("linear",), ceiling=False)["r2_interior"]["linear"]
    return max((v for order in curves.values() for v in order if not np.isnan(v)), default=-np.inf)


def analyze_job(path, label, layer, natural_only):
    X, y = load_cell(path, label, layer, natural_only)
    cell = analyze_cell(X, y, units_of(y))
    cell["n"], cell["n_horizons"] = len(y), int(len(np.unique(np.round(y, 9))))
    return cell


def in_parallel(function, jobs, desc):
    """[function(*job) for job in jobs], in parallel across cells."""
    results = Parallel(n_jobs=WORKERS, return_as="generator")(delayed(function)(*job) for job in jobs)
    return list(tqdm(results, total=len(jobs), desc=desc, unit="cell"))


def pick_layers(root):
    """{(model, mode, label): (layer, per-layer sweep scores or None)}: 06's layers for the 8B, a sweep for the 14B."""
    picks = {("Qwen/Qwen3-8B", *cell): (layer, None) for cell, layer in LAYERS_8B.items()}
    model = "Qwen/Qwen3-14B"
    jobs = [(model, mode, label, layer) for mode, label in CELLS for layer in range(MODELS[model] + 1)]
    scores = in_parallel(sweep_score, [(run_file(root, mode, model), label, layer) for model, mode, label, layer in jobs],
                         "14B layer pick")
    for mode, label in CELLS:
        per_layer = [score for (_, m, lab, _), score in zip(jobs, scores) if (m, lab) == (mode, label)]
        picks[(model, mode, label)] = (int(np.argmax(per_layer)), per_layer)
    return picks


def run_all(root):
    """{model: {"mode:label": {layer, sweep_best, all, natural}}} for every model and cell."""
    picks = pick_layers(root)
    # thinking off has no forced closes, so its natural-only run would repeat the all-prompt one
    jobs = [(key, natural) for key in picks for natural in (False, True) if not (natural and key[1] == "off")]
    cells = in_parallel(analyze_job, [(run_file(root, mode, model), label, picks[key][0], natural)
                                      for key, natural in jobs for model, mode, label in [key]], "probe cells")
    results = {model: {} for model in MODELS}
    for ((model, mode, label), natural), cell in zip(jobs, cells):
        layer, sweep_best = picks[(model, mode, label)]
        entry = results[model].setdefault(f"{mode}:{label}", {"layer": layer, "sweep_best": sweep_best, "natural": None})
        entry["natural" if natural else "all"] = cell
    return results


def curve(cell, label, probe, kind):
    """R² (interior) or within-unit R² of `probe` over TABLE_KS, in the ordering 06 quotes for this cell."""
    key = "r2_interior" if kind == "r2" else "r2_within"
    return [cell[key][probe][ORDER[label]][KS.index(k)] for k in TABLE_KS]


def summarize(results):
    lines = []
    for model, cells in results.items():
        lines.append(f"\n{model}  (interior R² / within-unit R², k = {', '.join(map(str, TABLE_KS))}; "
                     f"ordering as 06: variance at the suffix, horizon-mean elsewhere)")
        for name, entry in cells.items():
            label = name.split(":")[1]
            for subset in ("all", "natural"):
                cell = entry[subset]
                if cell is None:
                    continue
                lines.append(f"  {name:20} L{entry['layer']:<3} {subset:8} n={cell['n']:3} ceiling "
                             f"{cell['ceiling_interior']:.2f} / within {cell['ceiling_within']:.2f}")
                for probe in ("linear", "mlp"):
                    for kind, text in (("r2", "R²    "), ("within", "within")):
                        values = " ".join(f"{v:6.2f}" for v in curve(cell, label, probe, kind))
                        lines.append(f"      {probe:6} {text} {values}")
    return "\n".join(lines)


def plot(results, path):
    """Per cell and model: R² and within-unit R² against k, natural closes (solid) vs all prompts (dashed)."""
    fig, axes = plt.subplots(len(CELLS), 2 * len(results), figsize=(4 * 2 * len(results), 3 * len(CELLS)),
                             squeeze=False)
    for m, (model, cells) in enumerate(results.items()):
        for row, (name, entry) in zip(axes, cells.items()):
            label = name.split(":")[1]
            for ax, kind, title in zip(row[2 * m:], ("r2", "within"), ("R²", "within-unit R²")):
                for subset, style in (("natural", "-"), ("all", "--")):
                    if entry[subset] is None:
                        continue
                    for probe, color in (("linear", LINEAR), ("mlp", MLP)):
                        ax.plot(TABLE_KS, curve(entry[subset], label, probe, kind), style, color=color, marker="o",
                                ms=3, label=f"{probe}, {'natural closes' if subset == 'natural' else 'all prompts'}")
                ax.set_xscale("log")
                ax.set_xticks(TABLE_KS, [str(k) for k in TABLE_KS])
                # within-unit R² goes far below -1 at small k; the .txt has those values
                ax.set_ylim(-0.2 if kind == "r2" else -1.0, 1.02)
                ax.grid(alpha=0.2)
                ax.set_title(f"{model.split('/')[-1]} {name} L{entry['layer']}: {title}", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.supxlabel("top-k components", y=0.04)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def levels_job(path, label, layer, natural_only):
    X, y = load_cell(path, label, layer, natural_only)
    return probe_curves(X, y, units_of(y), ks=TABLE_KS, orders=(ORDER[label],), ceiling=False)


def levels(root):
    """{model: {"mode:label:L<layer>": {subset: {probe: {"r2", "prompt", "horizon": [per k]}}}}} at the picked layers."""
    root = Path(root)
    picked = json.loads((root / "analysis" / "probe_natural_within_unit.json").read_text())
    cells = [(model, name, entry["layer"]) for model, entries in picked.items() for name, entry in entries.items()]
    cells.append(("Qwen/Qwen3-14B", "on:suffix newline", 20))   # the layer 11 §2 aligns at
    jobs = [(model, name, layer, natural) for model, name, layer in cells for natural in (False, True)
            if not (natural and name.startswith("off"))]
    outputs = in_parallel(levels_job, [(run_file(root, name.split(":")[0], model), name.split(":")[1], layer, natural)
                                       for model, name, layer, natural in jobs], "within-unit levels")
    results = {model: {} for model in MODELS}
    for (model, name, layer, natural), out in zip(jobs, outputs):
        order = ORDER[name.split(":")[1]]
        results[model].setdefault(f"{name}:L{layer}", {})["natural" if natural else "all"] = {
            probe: {"r2": out["r2_interior"][probe][order], "prompt": out["r2_within"][probe][order],
                    "horizon": out["r2_within_horizon"][probe][order]} for probe in ("linear", "mlp")}
    lines = [f"within-unit R² per prompt / on horizon averages, interior, k = {', '.join(map(str, TABLE_KS))}"]
    for model, entries in results.items():
        for name, subsets in entries.items():
            for subset, probes in subsets.items():
                for probe, rows in probes.items():
                    pairs = " ".join(f"{a:6.2f}/{b:5.2f}" for a, b in zip(rows["prompt"], rows["horizon"]))
                    lines.append(f"{model.split('/')[-1]:9} {name:26} {subset:8} {probe:6} {pairs}")
    (root / "analysis" / "probe_within_levels.json").write_text(json.dumps(results, default=float))
    (root / "analysis" / "probe_within_levels.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def main(root):
    root = Path(root)
    results = run_all(root)
    out = root / "analysis"
    (out / "probe_natural_within_unit.json").write_text(json.dumps(results, default=float))
    text = summarize(results)
    (out / "probe_natural_within_unit.txt").write_text(text)
    plot(results, out / "probe_natural_within_unit.png")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="results folder holding the exp0_* runs of both models")
    parser.add_argument("--levels", action="store_true", help="within-unit R² per prompt vs per horizon, at the picked layers")
    args = parser.parse_args()
    if args.levels:
        levels(args.root)
    else:
        main(args.root)
