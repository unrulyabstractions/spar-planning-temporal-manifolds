"""Does the horizon code step at unit boundaries? (overnight, 2026-10-29)

On the full bank every horizon is in its natural unit, which changes at thresholds:
"43 hours", then "3 days". If the readout is a * log10(number) + b * log10(unit size),
then within a unit it rises by a per decade of horizon, and across a boundary, where the
unit grows by Δu decades and the number shrinks by about as much, it jumps by about
(b - a) Δu. A pure duration code has a = b: no jumps, one slope. A number line (a > b)
runs steeper within units and drops at each boundary; a unit staircase (b > a) is flat
within units and jumps up.

Per cell: leave-one-duration-out readout (unit_overlap's probe, top-8 duration-average
PCs), averaged per duration; residuals from a straight line on log horizon. Then:
  within-unit slope   the slope of the duration means on log horizon inside each unit,
                      pooled (one slope, unit intercepts), against the overall slope
  boundary jumps      the step in residual between the last duration of a unit and the
                      first of the next, against the steps between neighbours in one unit
The TRIM shortest and longest durations are left out: a leave-one-duration-out
readout cannot extrapolate past the training range, so the end durations read pulled
inward, which would add negative steps at both ends. Each boundary's own jump is
reported as well as their mean. CIs resample prompts within durations.

    python -m pipeline.s3_analysis.by_datatype.unit_staircase /home/Jared/SPAR/results/horizon
    -> <root>/analysis/unit_staircase.{json,txt,png}
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.s3_analysis.by_label.confound_sweep import cell_matrix, dataset
from pipeline.s3_analysis.by_datatype.unit_overlap import lodo_predict
from spar_horizon.full_bank import NATURAL_UNITS
from spar_horizon.horizons import Horizon

BOOT = 500
PLOTTED = ("Qwen3-8B full on suffix newline L16", "Qwen3-8B full on suffix newline L24",
           "Qwen3-14B full on suffix newline L20", "Qwen3-14B full on suffix newline L27")
# dataviz reference palette, slots 1-7, in unit order seconds .. years
UNIT_COLORS = dict(zip(reversed(NATURAL_UNITS), ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                                                 "#6250d6")))
TRIM = 2
CELLS = [("Qwen3-8B", "off", "suffix newline", 16), ("Qwen3-8B", "on", "suffix newline", 16),
         ("Qwen3-8B", "on", "suffix newline", 24), ("Qwen3-8B", "off", "pre-label", 24),
         ("Qwen3-8B", "on", "pre-label", 24),
         ("Qwen3-14B", "off", "suffix newline", 20), ("Qwen3-14B", "on", "suffix newline", 20),
         ("Qwen3-14B", "on", "suffix newline", 27), ("Qwen3-14B", "off", "pre-label", 33)]


def staircase(means, logh, units):
    """(overall slope, pooled within-unit slope, mean boundary step, mean within-unit
    step) of per-duration means sorted by log horizon. Steps are in residuals from the
    overall line."""
    return staircase_steps(means, logh, units)[:4]


def staircase_steps(means, logh, units):
    """`staircase`'s four numbers, then each boundary's step in order of horizon."""
    order = np.argsort(logh)
    means, logh, units = means[order], logh[order], units[order]
    overall = np.polyfit(logh, means, 1)[0]
    centered_h = np.concatenate([logh[units == u] - logh[units == u].mean() for u in np.unique(units)])
    centered_m = np.concatenate([means[units == u] - means[units == u].mean() for u in np.unique(units)])
    within_slope = centered_h @ centered_m / (centered_h @ centered_h)
    residual = means - overall * logh
    steps = np.diff(residual)
    boundary = units[1:] != units[:-1]
    return (float(overall), float(within_slope), float(steps[boundary].mean()), float(steps[~boundary].mean()),
            *steps[boundary].tolist())


def cell_staircase(readout, data, rng):
    durations = data["duration_id"]
    keys = np.array(sorted(np.unique(durations), key=lambda d: Horizon.parse(d).log10_years))[TRIM:-TRIM]
    logh = np.array([Horizon.parse(d).log10_years for d in keys])
    units = np.array([Horizon.parse(d).unit for d in keys])
    rows = [np.flatnonzero(durations == d) for d in keys]
    point = staircase_steps(np.array([readout[r].mean() for r in rows]), logh, units)
    boots = [staircase_steps(np.array([readout[rng.choice(r, len(r))].mean() for r in rows]), logh, units)
             for _ in range(BOOT)]
    low, high = np.percentile(boots, [2.5, 97.5], axis=0)
    boundaries = [f"{a}->{b}" for a, b in zip(units[:-1], units[1:]) if a != b]
    names = ("overall slope", "within-unit slope", "boundary step", "within-unit step", *boundaries)
    out = {n: [p, float(lo), float(hi)] for n, p, lo, hi in zip(names, point, low, high)}
    means = np.array([readout[r].mean() for r in rows])
    out["per duration"] = {"logh": logh.tolist(), "unit": units.tolist(),
                           "residual": (means - np.polyval(np.polyfit(logh, means, 1), logh)).tolist()}
    return out, len(boundaries)


def plot(results, path):
    """Per-duration readout residual from the overall line, colored by the natural unit:
    a duration code is flat, a number line drops at each unit boundary."""
    fig, axes = plt.subplots(1, len(PLOTTED), figsize=(3.4 * len(PLOTTED), 3.0), sharey=True, squeeze=False)
    for ax, cell in zip(axes[0], PLOTTED):
        d = results[cell]["per duration"]
        logh, units, residual = np.array(d["logh"]), np.array(d["unit"]), np.array(d["residual"])
        ax.plot(logh, residual, color="#c3c2b7", lw=1, zorder=1)
        for unit in UNIT_COLORS:
            rows = units == unit
            if rows.any():
                ax.scatter(logh[rows], residual[rows], s=22, color=UNIT_COLORS[unit], label=unit, zorder=2, linewidths=0)
        ax.axhline(0, color="#7a7974", lw=0.8)
        m = results[cell]["boundary step"]
        ax.set_title(f"{cell.replace(' full', '').replace(' suffix newline', ', suffix')}\n"
                     f"mean boundary step {m[0]:+.2f} [{m[1]:+.2f}, {m[2]:+.2f}]", fontsize=8)
        ax.set_xlabel("log10 horizon (years)", fontsize=8)
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].set_ylabel("readout − overall line\n(decades, per duration)", fontsize=8)
    axes[0, -1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(root):
    root = Path(root)
    rng = np.random.default_rng(0)
    results, loaded, lines = {}, {}, []
    for model, mode, position, layer in CELLS:
        key = (model, mode)
        if key not in loaded:
            loaded.clear()
            loaded[key] = dataset(root, model, mode, ("full",))
        parts, data = loaded[key]
        X = cell_matrix(parts, position, layer)
        readout = lodo_predict(X, data["y"], data["duration_id"], np.ones(len(X), bool), 8)
        name = f"{model} full {mode} {position} L{layer}"
        results[name], n_boundaries = cell_staircase(readout, data, rng)
        m = results[name]
        lines.append(f"{name}: " + " | ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in m.items()
                                              if k != "per duration")
                     + f" ({n_boundaries} boundaries)")
        print(lines[-1], flush=True)
    out = root / "analysis"
    (out / "unit_staircase.json").write_text(json.dumps(results, indent=1))
    (out / "unit_staircase.txt").write_text("\n".join(lines))
    plot(results, out / "unit_staircase.png")


if __name__ == "__main__":
    main(sys.argv[1])
