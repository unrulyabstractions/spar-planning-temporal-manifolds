"""What the horizon probe reads on the units bank: the duration, the number, or the unit word?

log10(duration) = log10(number) + log10(unit size), and the units bank varies the two
apart ("15 months" and "65 weeks" are one duration). Fitting the readout as
a * log10(number) + b * log10(unit size) over all prompts gives the

    number index (a - b) / (a + b)   +1: the probe reads the number and ignores the unit
                                       0: it weighs both alike, i.e. reads the duration
                                      -1: it reads the unit word and ignores the number

It is a ratio, so a weak probe, which shrinks a and b together, keeps its index.
Also per pair: the pair slope, the readout shift (other rendering minus natural, net
of their true difference; unit_overlap.pair_shifts) per decade of change in the
number, in the probe's raw decades (a - b for a linear readout).

Probe as in unit_overlap.py: fitted on natural renderings only, one duration held out
at a time, top-8 duration-average PCs. CIs resample base prompts.

    python analysis/unit_readout.py results/horizon [--replot]
    -> <root>/analysis/unit_readout.{json,png}, unit_readout_cells.json (with readouts;
       --replot redraws from it without refitting)
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from probe_dimensionality import positions  # noqa: E402  (same folder)
from spar_horizon.full_bank import NATURAL_UNITS, UNIT_YEARS  # noqa: E402
from spar_horizon.probes import r2  # noqa: E402
from unit_overlap import MODEL, REGIONS, load_run, lodo_predict, pair_shifts  # noqa: E402

K = 8
WORKERS = 4
N_BOOT = 1000
# the cells drawn in detail: early (surface form), clean mid-layer, late, and after thinking
DETAIL = (("off", "suffix newline", 3), ("off", "suffix newline", 15), ("off", "suffix newline", 30),
          ("on", "think close", 19))
CURVES = (("off", "suffix newline", "#2a78d6", "-"), ("off", "pre-label", "#2a78d6", "--"),
          ("on", "think close", "#eb6834", "-"), ("on", "pre-label", "#eb6834", "--"))
UNIT_COLORS = dict(zip(NATURAL_UNITS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")))


def log_numbers(y, units):
    """log10 of the number written in each prompt ("65" in "65 weeks"); y is log10 years."""
    return y - np.log10([UNIT_YEARS[u] for u in units])


def bootstrap_by_pair(statistic, pairs, rng):
    """statistic(rows) on all rows, and its 95% CI from resampling base prompts."""
    by_pair = {}
    for i, p in enumerate(pairs):
        by_pair.setdefault(p, []).append(i)
    groups = list(by_pair.values())
    boots = [statistic(np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))]))
             for _ in range(N_BOOT)]
    return [statistic(np.arange(len(pairs))), *np.percentile(boots, [2.5, 97.5]).tolist()]


def number_index(pred, labels, rng):
    """(a - b) / (a + b) for pred ~ a * log10(number) + b * log10(unit size) + c, with CI."""
    y, units = labels["y"], labels["units"]
    design = np.column_stack([log_numbers(y, units), np.log10([UNIT_YEARS[u] for u in units]), np.ones(len(y))])

    def index(rows):
        (a, b, _), *_ = np.linalg.lstsq(design[rows], pred[rows], rcond=None)
        return float((a - b) / (a + b))

    return bootstrap_by_pair(index, labels["pairs"], rng)


def number_shifts(pred, labels):
    """Per non-natural row: its readout shift (pair_shifts) and the change in
    log10(number) from its pair's natural rendering."""
    y, natural, pairs = labels["y"], labels["natural"], labels["pairs"]
    rows, shift = pair_shifts(pred, y, pairs, natural)
    numbers = log_numbers(y, labels["units"])
    natural_row = {p: i for i, (p, n) in enumerate(zip(pairs, natural)) if n}
    d_log_number = numbers[rows] - numbers[[natural_row[pairs[i]] for i in rows]]
    return rows, shift, d_log_number


def pair_slope(pred, labels, rng):
    """Least-squares slope through 0 of the readout shift on the change in log10(number), with CI."""
    rows, shift, d_log_number = number_shifts(pred, labels)

    def slope(r):
        return float((shift[r] * d_log_number[r]).sum() / (d_log_number[r] ** 2).sum())

    return bootstrap_by_pair(slope, labels["pairs"][rows], rng)


def cell_job(args):
    root, mode, position, layer = args
    z, labels = load_run(root, mode)
    meta = {"regions": {r: tuple(int(x) for x in z[f"region_{r}"]) for r in REGIONS},
            "tokens": [str(t) for t in z["tokens"]]}
    X = z[f"L{layer}"][:, positions(meta)[position]].astype(np.float64)
    y, natural = labels["y"], labels["natural"]
    with threadpool_limits(1):
        pred = lodo_predict(X, y, labels["durations"], natural, K)
    rng = np.random.default_rng(layer)
    return mode, position, layer, {"pred": pred.tolist(), "index": number_index(pred, labels, rng),
                                   "pair_slope": pair_slope(pred, labels, rng),
                                   "r2_natural": r2(y, pred, natural), "r2_other": r2(y, pred, ~natural)}


def rendering_means(values, durations, units):
    """Mean of `values` per rendering (duration, unit), and the keys in that order."""
    keys = sorted(set(zip(durations, units)))
    return keys, np.array([values[(durations == d) & (units == u)].mean() for d, u in keys])


def plot_predicted_vs_actual(ax, labels, pred, title):
    durations, units = labels["durations"], labels["units"]
    keys, true_mean = rendering_means(labels["y"], durations, units)
    _, pred_mean = rendering_means(pred, durations, units)
    natural_keys = {(d, u) for d, u, n in zip(durations, units, labels["natural"]) if n}
    lo, hi = labels["y"].min() - 0.3, labels["y"].max() + 0.3
    ax.plot([lo, hi], [lo, hi], color="#8a8a85", lw=1, zorder=0)
    for d in sorted(set(durations)):   # join the renderings of one duration
        idx = [i for i, (dd, _) in enumerate(keys) if dd == d]
        ax.plot(true_mean[idx], pred_mean[idx], color="#c3c2b7", lw=1, zorder=1)
    for i, (d, u) in enumerate(keys):
        natural = (d, u) in natural_keys
        ax.scatter(true_mean[i], pred_mean[i], s=34, zorder=2, linewidths=1.5,
                   facecolor=UNIT_COLORS[u] if natural else "white", edgecolor=UNIT_COLORS[u])
    ax.set(xlim=(lo, hi), ylim=(lo, hi), title=title, xlabel="actual log10 horizon (years)",
           ylabel="probe readout, log10 years")


def plot_shift_vs_number(ax, labels, pred, slope, title):
    units = labels["units"]
    rows, shift, d_log_number = number_shifts(pred, labels)
    keys, mean_shift = rendering_means(shift, labels["durations"][rows], units[rows])
    _, mean_d = rendering_means(d_log_number, labels["durations"][rows], units[rows])
    span = np.array([-1.6, 1.6])
    for reference, text in ((1, "reads number"), (0, "reads duration"), (-1, "reads unit")):
        ax.plot(span, reference * span, color="#8a8a85", lw=1, ls=":" if reference else "-", zorder=0)
        ax.annotate(text, (-1.5, -1.5 * reference), fontsize=7, color="#5f5e5a", ha="left",
                    va="bottom" if reference <= 0 else "top")
    for (d, u), dx, dy in zip(keys, mean_d, mean_shift):
        ax.scatter(dx, dy, s=34, facecolor="white", edgecolor=UNIT_COLORS[u], linewidths=1.5, zorder=2)
    ax.plot(span, slope[0] * span, color="#1a1a19", lw=2, zorder=1)
    ax.set(xlim=span, ylim=span, xlabel="change in log10(number), other - natural",
           ylabel="readout shift, decades (rendering means)",
           title=f"{title}\npair slope {slope[0]:+.2f} [{slope[1]:+.2f}, {slope[2]:+.2f}]")


def plot(results, labels_by_mode, path):
    fig = plt.figure(figsize=(4.2 * len(DETAIL), 12.4))
    grid = fig.add_gridspec(3, len(DETAIL), height_ratios=(1, 1, 0.9))
    for j, (mode, position, layer) in enumerate(DETAIL):
        cell = results[mode][position][layer]
        pred = np.array(cell["pred"])
        title = f"thinking {mode}, {position}, L{layer}"
        plot_predicted_vs_actual(fig.add_subplot(grid[0, j]), labels_by_mode[mode], pred,
                                 f"{title}\nR² natural {cell['r2_natural']:.2f}, other {cell['r2_other']:.2f}")
        plot_shift_vs_number(fig.add_subplot(grid[1, j]), labels_by_mode[mode], pred, cell["pair_slope"], title)
    half = len(DETAIL) // 2
    for axis, key, ylabel, title in (
            (fig.add_subplot(grid[2, :half]), "index", "number index (95% CI)",
             "across all prompts: readout ~ a·log10(number) + b·log10(unit), (a - b) / (a + b)"),
            (fig.add_subplot(grid[2, half:]), "pair_slope", "pair slope, decades per decade (95% CI)",
             "within pairs: readout shift per decade of change in the number")):
        for mode, position, color, ls in CURVES:
            layers = sorted(results[mode][position])
            w = np.array([results[mode][position][l][key] for l in layers])
            axis.fill_between(layers, w[:, 1], w[:, 2], color=color, alpha=0.15, linewidth=0)
            axis.plot(layers, w[:, 0], color=color, ls=ls, lw=2, label=f"thinking {mode}, {position}")
        for level, text in ((1, "reads the number"), (0, "reads the duration"), (-1, "reads the unit word")):
            axis.axhline(level, color="#8a8a85", lw=1, ls="-" if level == 0 else ":")
            axis.annotate(text, (0.3, level), fontsize=8, color="#5f5e5a", va="bottom")
        axis.set(ylim=(-1.15, 1.15), xlabel="layer", ylabel=ylabel)
        axis.set_title(title, fontsize=9)
    fig.axes[-1].legend(fontsize=8, loc="lower right")
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor=c, markeredgecolor=c, label=u)
               for u, c in UNIT_COLORS.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor="white",
                           markeredgecolor="#5f5e5a", label="hollow: other rendering")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=len(handles), fontsize=8,
               frameon=False)
    for a in fig.axes:
        a.grid(alpha=0.2)
        a.tick_params(labelsize=7)
    fig.suptitle(f"{MODEL}, units bank: what the horizon probe reads (k = {K}, fitted on natural renderings)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(root, replot=False):
    root = Path(root)
    out = root / "analysis"
    cells_path = out / "unit_readout_cells.json"
    jobs, labels_by_mode = [], {}
    for mode, position, _, _ in CURVES:
        if mode not in labels_by_mode:
            z, labels_by_mode[mode] = load_run(root, mode)
            n_layers = sum(1 for key in z.files if key[0] == "L" and key[1:].isdigit())
        jobs += [(root, mode, position, layer) for layer in range(1, n_layers)]   # layer 0 is constant here
    if replot:
        results = {mode: {pos: {int(l): c for l, c in cells.items()} for pos, cells in by_pos.items()}
                   for mode, by_pos in json.loads(cells_path.read_text()).items()}
    else:
        results = {}
        with ProcessPoolExecutor(WORKERS) as pool:
            for n, (mode, position, layer, cell) in enumerate(pool.map(cell_job, jobs), 1):
                results.setdefault(mode, {}).setdefault(position, {})[layer] = cell
                if n % 10 == 0 or n == len(jobs):
                    print(f"{n}/{len(jobs)} cells", flush=True)
        out.mkdir(exist_ok=True)
        cells_path.write_text(json.dumps(results))
    summary = {mode: {pos: {l: {k: v for k, v in c.items() if k != "pred"} for l, c in cells.items()}
                      for pos, cells in by_pos.items()} for mode, by_pos in results.items()}
    (out / "unit_readout.json").write_text(json.dumps(summary, indent=1))
    plot(results, labels_by_mode, out / "unit_readout.png")
    print(f"wrote {out}/unit_readout.{{json,png}}")


if __name__ == "__main__":
    main(sys.argv[1], replot="--replot" in sys.argv[2:])
