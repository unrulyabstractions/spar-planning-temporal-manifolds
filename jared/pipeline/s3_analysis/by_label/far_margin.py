"""The far-delay version of delay_margin.py, for long durations (overnight, 2026-10-29).

delay_margin.py fits on regime-none prompts, which on the old banks exist only up to
about 1 year. Above that the reference is regime near (only the near option pays): fit
the probe and the axis on near prompts, at durations with at least MIN_REFERENCE of
them, and read every near and both prompt there against the near prompts' mean. Binned
by log10(far delay / horizon): a code that compares the horizon with the far delay
steps where that margin crosses 0 (the far delay falls inside the horizon, regime both).

    python -m pipeline.s3_analysis.by_label.far_margin /home/Jared/SPAR/results/horizon
    -> <root>/analysis/far_margin.{json,txt,png}
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.s3_analysis.by_label.confound_sweep import base_folds, calibration, cell_matrix, crossfit_readout, dataset, fit_probe, horizon_axis  # noqa: E501
from pipeline.s3_analysis.by_label.delay_margin import COLORS, binned, none_reference

MIN_REFERENCE = 4
CELLS = [("Qwen3-8B", ("full", "units"), "on", "suffix newline", 16),
         ("Qwen3-8B", ("full", "units"), "on", "suffix newline", 24),
         ("Qwen3-14B", ("full",), "on", "suffix newline", 20),
         ("Qwen3-14B", ("full",), "on", "suffix newline", 27)]


def cell_curves(X, data, rng):
    y, duration, bases = data["y"], data["duration_id"], data["pairs"]
    is_ref = data["regime"] == "near"
    counts = {d: int((is_ref & (duration == d)).sum()) for d in np.unique(duration)}
    keep = np.array([counts[d] >= MIN_REFERENCE for d in duration]) & np.isin(data["regime"], ("near", "both"))
    X, y, duration, bases, is_ref = X[keep], y[keep], duration[keep], bases[keep], is_ref[keep]
    far = data["features"][keep, 1]
    folds = base_folds(bases)
    readout = np.zeros(len(y))
    ref_rows = np.flatnonzero(is_ref)
    readout[ref_rows] = crossfit_readout(X[ref_rows], y[ref_rows], duration[ref_rows], folds[ref_rows])
    others = np.flatnonzero(~is_ref)
    for fold in (0, 1):
        train = ref_rows[folds[ref_rows] != fold]
        readout[others] += 0.5 * fit_probe(X[train], y[train], duration[train])(X[others])
    slope = calibration(readout[ref_rows], y[ref_rows], duration[ref_rows])
    axis = horizon_axis(X[ref_rows], y[ref_rows], duration[ref_rows])
    along = (X - X[ref_rows].mean(0)) @ axis / (axis @ axis)
    residuals = {"probe": none_reference(readout, duration, is_ref) / slope,
                 "axis": none_reference(along, duration, is_ref)}
    out = {"n": int(keep.sum()), "n_near": int(is_ref.sum()), "calibration": slope,
           "durations": int(len(np.unique(duration)))}
    for name, residual in residuals.items():
        out[name] = binned(residual, far - y, bases, rng)
    return out


def plot(results, path):
    cells = list(results)
    fig, axes = plt.subplots(1, len(cells), figsize=(3.2 * len(cells), 3.2), squeeze=False)
    for ax, cell in zip(axes[0], cells):
        for name, color in COLORS.items():
            pts = np.array(results[cell][name])
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2, marker="o", ms=4, label=name)
            ax.fill_between(pts[:, 0], pts[:, 2], pts[:, 3], color=color, alpha=0.15, lw=0)
        ax.axhline(0, color="#7a7974", lw=0.8)
        ax.axvline(0, color="#7a7974", lw=0.8, ls="--")
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(cell.replace(" suffix newline", "\nsuffix"), fontsize=8)
        ax.set_xlabel("log10(far delay / horizon)", fontsize=8)
    axes[0, 0].set_ylabel("readout − regime-near mean\n(eq. decades)", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Long durations, against regime-near prompts (probe fitted on them): does the far delay move the "
                 "horizon code? Left of the dashed line the far delay falls inside the horizon (regime both).",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(root):
    root = Path(root)
    rng = np.random.default_rng(0)
    results, loaded, lines = {}, {}, []
    for model, banks, mode, position, layer in CELLS:
        key = (model, banks, mode)
        if key not in loaded:
            loaded.clear()
            loaded[key] = dataset(root, model, mode, banks)
        parts, data = loaded[key]
        name = f"{model} {'+'.join(banks)} {mode} {position} L{layer}"
        m = results[name] = cell_curves(cell_matrix(parts, position, layer), data, rng)
        lines.append(f"{name}: {m['n']} prompts at {m['durations']} durations ({m['n_near']} near), "
                     f"calibration {m['calibration']:.2f}")
        for readout in COLORS:
            lines.append(f"  {readout:5s} far margin: " + " ".join(
                f"{c:+.2f}:{v:+.2f}[{lo:+.2f},{hi:+.2f}]n{n}" for c, v, lo, hi, n in m[readout]))
        print("\n".join(lines[-3:]), flush=True)
    out = root / "analysis"
    (out / "far_margin.json").write_text(json.dumps(results, indent=1))
    (out / "far_margin.txt").write_text("\n".join(lines))
    plot(results, out / "far_margin.png")


if __name__ == "__main__":
    main(sys.argv[1])
