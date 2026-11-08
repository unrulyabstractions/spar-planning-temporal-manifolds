"""E9 analysis: does the horizon readout before reasoning track the duration or
the unit word? (spec: 07_spec_steering.md section 11)

At each layer, a ridge probe for log10 horizon and the curve of horizon averages
are fitted on the 471 full-bank prompts of the thinking-on E0 run, at the suffix
newline and the last suffix token, and applied to the E9 prompts: 22 durations,
each rendered in 2-4 units, in the same 24 contexts. Reported per layer:

- unit share: the variance of the readout between the units of one duration,
  over the variance between all renderings (cell means over contexts). R3
  passes below 0.2.
- the shift of each unit from the canonical rendering, in decades, against the
  true log difference (renderings differ from the canonical by up to 0.05 decades);
- the extraction check: the originals against the saved E0 activations.

    python -m pipeline.s3_analysis.by_datatype.units_analysis results/horizon
    -> <root>/analysis/units.json, units.txt, units.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from spar_horizon.geometry import _fit

POSITIONS = {"suffix newline": ("newline", 1), "suffix last": ("last", 4)}   # E9 array, E0 kept index
JUDGED = ("suffix newline", 16)   # the R3 criterion's cell
CURVE_K = 8
UNIT_COLORS = {"seconds": "#9a9893", "minutes": "#1baf7a", "hours": "#2a78d6", "days": "#7b4fd6", "weeks": "#d6437b",
               "months": "#eb6834", "years": "#52514e", "decades": "#c9a227", "centuries": "#8a5a2b"}


def fit_readouts(X_bank, log_h, X_new):
    """Readouts of X_new [m, d] in log10 years: the ridge probe, and the curve (the
    log horizon of the nearest horizon average in the top-k PCs of the averages)."""
    mean = X_bank.mean(0)
    probe = _fit(X_bank - mean, log_h)
    groups = np.round(log_h, 9)
    horizons = np.unique(groups)
    averages = np.stack([X_bank[groups == h].mean(0) for h in horizons]) - mean
    basis = PCA(n_components=CURVE_K).fit(averages).components_
    coords, new = averages @ basis.T, (X_new - mean) @ basis.T
    nearest = np.argmin(((new[:, None] - coords[None]) ** 2).sum(-1), 1)
    return {"probe": probe.predict(X_new - mean), "curve": horizons[nearest]}


def unit_stats(readout, rows):
    """Unit share of the readout variance, and per rendering its mean shift from the
    canonical rendering of its duration (decades) next to the true difference."""
    durations = [r["duration"] for r in rows]
    cells = {}   # (duration, text) -> mean readout over contexts
    for value, r, duration in zip(readout, rows, durations):
        cells.setdefault((duration, r["text"]), []).append(value)
    cells = {key: np.mean(v) for key, v in cells.items()}
    values = np.array(list(cells.values()))
    within = np.mean([(v - np.mean([cells[k] for k in cells if k[0] == key[0]])) ** 2 for key, v in cells.items()])
    shifts = []
    for (duration, text), value in cells.items():
        if text == duration:
            continue
        true = np.log10(next(r["years"] for r in rows if r["text"] == text) /
                        next(r["years"] for r in rows if r["text"] == duration))
        shifts.append({"duration": duration, "text": text, "unit": text.split()[-1],
                       "shift": float(value - cells[(duration, duration)]), "true": float(true)})
    return {"unit_share": float(within / values.var()), "shifts": shifts}


def main(root):
    root = Path(root)
    e9 = root / "exp9_Qwen3-8B_units"
    rows = json.loads((e9 / "prompts.json").read_text())
    acts = np.load(e9 / "activations.npz")
    e0 = np.load(root / "exp0_Qwen3-8B_think-on_full" / "activations.npz")
    keep = ~np.isnan(e0["horizons"])
    log_h = np.log10(e0["horizons"][keep])
    is_original = np.array([r["original"] for r in rows])
    renderings = [r for r in rows if not r["original"]]
    true = np.log10([r["years"] for r in renderings])
    originals = [r["record"] for r in rows if r["original"]]
    out, lines = {}, []
    for label, (array, kept) in POSITIONS.items():
        A = acts[array]
        out[label] = {}
        lines.append(f"\n{label}: unit share of the readout variance (R3 passes below 0.20), "
                     f"slope of the readout on the true log duration, extraction check")
        lines.append(f"  {'layer':>5} {'probe share':>11} {'curve share':>11} {'probe slope':>11} {'cos(E9, E0)':>11}")
        for layer in range(1, A.shape[1] - 1):
            X_bank = e0[f"L{layer}"][keep][:, kept].astype(np.float64)
            X_new = A[:, layer].astype(np.float64)
            read = fit_readouts(X_bank, log_h, X_new[~is_original])
            saved = e0[f"L{layer}"][originals][:, kept].astype(np.float64)
            fresh = X_new[is_original]
            cos = float(np.mean((saved * fresh).sum(1) / np.linalg.norm(saved, axis=1) / np.linalg.norm(fresh, axis=1)))
            stats = {name: unit_stats(values, renderings) for name, values in read.items()}
            slope = float(np.polyfit(true, read["probe"], 1)[0])
            out[label][layer] = {"cos_e0": cos, "probe_slope": slope,
                                 **{f"{name}_unit_share": s["unit_share"] for name, s in stats.items()},
                                 "shifts": stats["probe"]["shifts"], "readout": read["probe"].tolist()}
            if layer % 4 == 0 or (label, layer) == JUDGED:
                lines.append(f"  {layer:>5} {stats['probe']['unit_share']:>11.2f} {stats['curve']['unit_share']:>11.2f} "
                             f"{slope:>11.2f} {cos:>11.4f}")
    label, layer = JUDGED
    judged = out[label][layer]
    lines.append(f"\nR3 at {label} L{layer}: probe unit share {judged['probe_unit_share']:.2f} -> "
                 f"{'PASS' if judged['probe_unit_share'] < 0.2 else 'fail'} (curve: {judged['curve_unit_share']:.2f})")
    lines.append("Shift of each rendering from its canonical one at the judged cell (probe, decades; true difference):")
    by_unit = {}
    for s in judged["shifts"]:
        by_unit.setdefault(s["unit"].rstrip("s").replace("centurie", "century"), []).append(s)
    for unit, items in by_unit.items():
        lines.append(f"  {unit:8} n={len(items):2}  mean shift {np.mean([s['shift'] for s in items]):+.2f}  "
                     f"(true {np.mean([s['true'] for s in items]):+.3f}): "
                     + ", ".join(f"{s['text']} {s['shift']:+.2f}" for s in items[:6]))
    text = "\n".join(lines)
    (root / "analysis" / "units.txt").write_text(text)
    (root / "analysis" / "units.json").write_text(json.dumps(out))
    plot(judged, renderings, true, root / "analysis" / "units.png")
    print(text)
    print(f"\nwrote {root / 'analysis'}/units.txt, units.json, units.png")


def plot(judged, renderings, true, path):
    """Probe readout against the true duration at the judged cell, one point per
    rendering (mean over contexts), colored by unit."""
    readout = np.array(judged["readout"])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    texts = sorted({r["text"] for r in renderings}, key=lambda t: next(r["years"] for r in renderings if r["text"] == t))
    for text in texts:
        m = np.array([r["text"] == text for r in renderings])
        unit = next(r["unit"] for r in renderings if r["text"] == text)
        ax.scatter(true[m][0], readout[m].mean(), color=UNIT_COLORS[unit], s=36, edgecolors="#222", linewidths=0.3,
                   label=unit if text == next(t for t in texts if next(r["unit"] for r in renderings if r["text"] == t) == unit) else None)
    lims = [true.min() - 0.3, true.max() + 0.3]
    ax.plot(lims, lims, color="#52514e", lw=0.8, ls="--")
    ax.set_xlabel("true duration, log10 years")
    ax.set_ylabel("probe readout, log10 years (mean over 24 contexts)")
    ax.set_title(f"{JUDGED[0]}, L{JUDGED[1]}: same duration, different units", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
