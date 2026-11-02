"""Does the horizon code read the horizon relative to the delays? (overnight, 2026-10-29)

confound_sweep.py found that, within a duration, prompts whose delays fall inside the
horizon (regime near / both) read longer, and a longer near delay reads shorter. A
probe fitted on all prompts could produce that by itself: the duration averages it is
fitted to mix regime into the horizon (long durations are mostly near/both on the old
banks). So here the probe is fitted on regime-none prompts only, at durations with at
least MIN_NONE of them (30 s to about 1 year on the full bank), and every prompt at
those durations is read against the none prompts' mean readout there.

The residual is then binned by the margin log10(delay / horizon), for the near and the
far delay. A code that only reads the horizon is flat. A comparison of horizon and
delay bends or steps where the margin crosses 0 (the delay falls inside the horizon).

Two readouts, as in confound_sweep.py: the none-fitted probe (equivalent decades, via
its calibration) and the projection on the none prompts' horizon axis (probe-free,
equivalent decades). CIs resample base prompts.

    python analysis/delay_margin.py /home/Jared/SPAR/results/horizon
    -> <root>/analysis/delay_margin.{json,txt,png}   (--replot redraws from the json)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from confound_sweep import (base_folds, calibration, cell_matrix, crossfit_readout, dataset,  # noqa: E402
                            fit_probe, horizon_axis)

MIN_NONE = 4
BINS = np.arange(-2.0, 2.01, 0.5)          # margin bins, decades
BOOT = 200
CELLS = [("Qwen3-8B", ("full", "units"), "off", "suffix newline", 16),
         ("Qwen3-8B", ("full", "units"), "on", "suffix newline", 16),
         ("Qwen3-8B", ("full", "units"), "on", "suffix newline", 24),
         ("Qwen3-8B", ("full", "units"), "on", "pre-label", 22),
         ("Qwen3-14B", ("full",), "off", "suffix newline", 20),
         ("Qwen3-14B", ("full",), "on", "suffix newline", 20),
         ("Qwen3-14B", ("full",), "on", "suffix newline", 27),
         ("Qwen3-14B", ("full",), "on", "pre-label", 33)]
COLORS = {"probe": "#2a78d6", "axis": "#eb6834"}


def none_reference(values, durations, is_none):
    """`values` minus the mean of the regime-none rows of the same duration."""
    out = np.full(len(values), np.nan)
    for d in np.unique(durations):
        rows = durations == d
        out[rows] = values[rows] - values[rows & is_none].mean()
    return out


def binned(residual, margin, bases, rng, bins=BINS):
    """Per margin bin: [center, mean, 2.5%, 97.5%, n], CIs resampling bases within the bin."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        rows = np.flatnonzero((margin >= lo) & (margin < hi))
        if len(rows) < 10:
            continue
        keys = np.unique(bases[rows])
        by_base = {b: residual[rows][bases[rows] == b] for b in keys}
        boots = [np.concatenate([by_base[b] for b in rng.choice(keys, size=len(keys))]).mean() for _ in range(BOOT)]
        out.append([float((lo + hi) / 2), float(residual[rows].mean()), *np.percentile(boots, [2.5, 97.5]).tolist(),
                    int(len(rows))])
    return out


def cell_curves(X, data, rng):
    y, duration, bases = data["y"], data["duration_id"], data["pairs"]
    is_none = data["regime"] == "none"
    counts = {d: int((is_none & (duration == d)).sum()) for d in np.unique(duration)}
    keep = np.array([counts[d] >= MIN_NONE for d in duration])
    X, y, duration, bases, is_none = X[keep], y[keep], duration[keep], bases[keep], is_none[keep]
    near, far = data["features"][keep, 0], data["features"][keep, 1]
    # fit on none prompts only, cross-fitted by base
    folds = base_folds(bases)
    readout = np.zeros(len(y))
    none_rows = np.flatnonzero(is_none)
    readout[none_rows] = crossfit_readout(X[none_rows], y[none_rows], duration[none_rows], folds[none_rows])
    # the other regimes, read by each fold's none probe, averaged
    others = np.flatnonzero(~is_none)
    for fold in (0, 1):
        train = none_rows[folds[none_rows] != fold]
        readout[others] += 0.5 * fit_probe(X[train], y[train], duration[train])(X[others])
    slope = calibration(readout[none_rows], y[none_rows], duration[none_rows])
    axis = horizon_axis(X[none_rows], y[none_rows], duration[none_rows])
    along = (X - X[none_rows].mean(0)) @ axis / (axis @ axis)
    residuals = {"probe": none_reference(readout, duration, is_none) / slope,
                 "axis": none_reference(along, duration, is_none)}
    out = {"n": int(keep.sum()), "n_none": int(is_none.sum()), "calibration": slope, "durations": int(len(np.unique(duration)))}
    for name, residual in residuals.items():
        out[name] = {"near margin": binned(residual, near - y, bases, rng),
                     "far margin": binned(residual, far - y, bases, rng)}
    return out


def plot(results, path):
    """Near margin only: on the testable durations (up to about 1 year) the far delays
    (2 years or more) never fall inside the horizon. Each panel has its own scale: the
    answer-slot cells move by decades, the suffix cells by tenths."""
    cells = list(results)
    fig, axes = plt.subplots(1, len(cells), figsize=(2.9 * len(cells), 3.2), squeeze=False)
    for col, cell in enumerate(cells):
        ax = axes[0, col]
        for name, color in COLORS.items():
            pts = np.array(results[cell][name]["near margin"])
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2, marker="o", ms=4, label=name)
            ax.fill_between(pts[:, 0], pts[:, 2], pts[:, 3], color=color, alpha=0.15, lw=0)
        ax.axhline(0, color="#7a7974", lw=0.8)
        ax.axvline(0, color="#7a7974", lw=0.8, ls="--")
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(cell.replace(" suffix newline", "\nsuffix").replace(" pre-label", "\npre-label"), fontsize=8)
        ax.set_xlabel("log10(near delay / horizon)", fontsize=8)
        if col == 0:
            ax.set_ylabel("readout − none mean\n(eq. decades)", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Within a duration, against regime-none prompts (probe fitted on them): does the near delay move "
                 "the horizon code? Left of the dashed line the delay falls inside the horizon.", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summary_lines(results):
    lines = []
    for cell, m in results.items():
        lines.append(f"{cell}: {m['n']} prompts at {m['durations']} durations ({m['n_none']} none), "
                     f"calibration {m['calibration']:.2f}")
        for name in COLORS:
            for margin in ("near margin", "far margin"):
                pts = " ".join(f"{c:+.2f}:{v:+.2f}[{lo:+.2f},{hi:+.2f}]n{n}" for c, v, lo, hi, n in m[name][margin])
                lines.append(f"  {name:5s} {margin}: {pts}")
    return lines


def main(root, replot=False):
    root = Path(root)
    out = root / "analysis"
    if replot:
        plot(json.loads((out / "delay_margin.json").read_text()), out / "delay_margin.png")
        return
    rng = np.random.default_rng(0)
    results, loaded = {}, {}
    for model, banks, mode, position, layer in CELLS:
        key = (model, banks, mode)
        if key not in loaded:
            loaded.clear()
            loaded[key] = dataset(root, model, mode, banks)
        parts, data = loaded[key]
        name = f"{model} {'+'.join(banks)} {mode} {position} L{layer}"
        results[name] = cell_curves(cell_matrix(parts, position, layer), data, rng)
        print(name, "done", flush=True)
    (out / "delay_margin.json").write_text(json.dumps(results, indent=1))
    text = "\n".join(summary_lines(results))
    (out / "delay_margin.txt").write_text(text)
    plot(results, out / "delay_margin.png")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1], replot="--replot" in sys.argv[2:])
