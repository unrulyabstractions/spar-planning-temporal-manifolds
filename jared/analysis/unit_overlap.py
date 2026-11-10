"""Horizon or unit word? E0 on the units bank (spec: 14_spec_unit_overlap.md §4).

Every base prompt appears in two or three renderings of one duration ("15 months",
"65 weeks"), so unit and magnitude are no longer confounded. At each layer of the
suffix newline, think close (thinking on) and pre-label cells:

- transfer: a ridge on the top-k horizon-average PCs, fitted on natural renderings
  only and holding out whole durations, scores the held-out duration's natural and
  other renderings. A duration code reads both alike; a unit-word code misreads the
  others by their unit's size.
- shift: per pair, the readout of the other rendering minus the natural one's, net of
  their (<= 10%) difference in duration, in decades, averaged per rendering ("65 weeks")
  before taking its size, as in E9, so prompt scatter does not count. Split by whether
  the other rendering uses the smaller unit ("65 weeks" for "15 months") or the larger.
- within unit: Check 1's within-unit R² (per prompt and on horizon averages) from a
  probe fitted on all renderings, now with units spanning 2-3 decades each.
- share: on the per-(duration, unit) means of the top-k coordinates, the variance
  only the unit explains, only the duration explains (type II), and the part either
  explains. Units still track durations in this bank (weeks only cover weeks to
  months), so the shared part is large and the unique parts are small for any code;
  their contrast is the reading.
Behavior, per mode: how often a pair's renderings get the same choice, and the near
log-odds of the other rendering minus the natural one.

    python analysis/unit_overlap.py results/horizon
    -> <root>/analysis/unit_overlap.{json,txt,png}
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from probe_dimensionality import positions  # noqa: E402  (same folder)
from spar_horizon.full_bank import NATURAL_UNITS, natural_horizon  # noqa: E402
from spar_horizon.probes import fit_ridge, horizon_within_unit_r2, r2, within_unit_r2  # noqa: E402
from spar_horizon.runs import REGIONS  # noqa: E402
from spar_horizon.steering import bootstrap_mean  # noqa: E402

MODEL = "Qwen3-8B"
KS = (1, 2, 4, 8)
POSITIONS = ("suffix newline", "think close", "pre-label")
WORKERS = 4


# --- the measures, on one cell -------------------------------------------------------

def lodo_predict(X, y, durations, train, k):
    """Out-of-fold log-horizon predictions for every row, holding out one duration
    at a time and fitting only on the `train` rows of the others: a ridge (inner CV
    grouped by duration) on the top-k PCs of the training durations' averages."""
    pred = np.full(len(y), np.nan)
    for d in np.unique(durations):
        fit_rows = train & (durations != d)
        mean = X[fit_rows].mean(0)
        averages = np.stack([X[fit_rows & (durations == g)].mean(0) for g in np.unique(durations[fit_rows])]) - mean
        basis = PCA(n_components=k).fit(averages).components_
        model = fit_ridge((X[fit_rows] - mean) @ basis.T, y[fit_rows], durations[fit_rows])
        test = durations == d
        pred[test] = model.predict((X[test] - mean) @ basis.T)
    return pred


def pair_shifts(pred, y, pairs, natural):
    """Per non-natural row: its readout minus its pair's natural readout, minus the
    true difference in log duration (decades)."""
    natural_row = {p: i for i, (p, n) in enumerate(zip(pairs, natural)) if n}
    others = np.flatnonzero(~natural)
    reference = np.array([natural_row[pairs[i]] for i in others])
    return others, (pred[others] - pred[reference]) - (y[others] - y[reference])


def rendering_shift_size(shift, durations, units):
    """Mean over renderings (duration, unit) of |their mean shift|."""
    keys = list(zip(durations, units))
    return float(np.mean([abs(shift[[k == key for k in keys]].mean()) for key in set(keys)]))


def smaller_unit(units, natural_units):
    """True where the rendered unit is smaller than the natural one (a bigger number)."""
    return np.array([NATURAL_UNITS.index(u) > NATURAL_UNITS.index(n) for u, n in zip(units, natural_units)])


def unit_duration_shares(Z, durations, units):
    """Shares of the variance of the per-(duration, unit) means of Z [n, k]: what the
    unit explains beyond the duration, the duration beyond the unit (type II), and
    what either explains (the confounded part)."""
    cells = sorted(set(zip(durations, units)))
    means = np.stack([Z[(durations == d) & (units == u)].mean(0) for d, u in cells])
    means = means - means.mean(0)

    def dummies(values):
        levels = sorted(set(values))
        return np.array([[v == level for level in levels[1:]] for v in values], float)

    D, U = dummies([d for d, _ in cells]), dummies([u for _, u in cells])
    ones = np.ones((len(cells), 1))

    def residual(design):
        coef, *_ = np.linalg.lstsq(design, means, rcond=None)
        return ((means - design @ coef) ** 2).sum()

    total, full = (means ** 2).sum(), residual(np.hstack([ones, D, U]))
    unit = (residual(np.hstack([ones, D])) - full) / total
    duration = (residual(np.hstack([ones, U])) - full) / total
    return {"unit": float(unit), "duration": float(duration), "shared": float(1 - full / total - unit - duration)}


def cell_measures(X, y, durations, units, natural_units, pairs, natural, ks=KS):
    """Every per-cell measure, per k."""
    out = {}
    others = ~natural
    smaller = smaller_unit(units, natural_units)
    with threadpool_limits(1):
        for k in ks:
            transfer = lodo_predict(X, y, durations, natural, k)
            rows, shift = pair_shifts(transfer, y, pairs, natural)
            pooled = lodo_predict(X, y, durations, np.ones(len(y), bool), k)
            mean = X.mean(0)
            averages = np.stack([X[durations == d].mean(0) for d in np.unique(durations)]) - mean
            Z = (X - mean) @ PCA(n_components=k).fit(averages).components_.T
            out[k] = {"r2_natural": r2(y, transfer, natural), "r2_other": r2(y, transfer, others),
                      "shift_abs": rendering_shift_size(shift, durations[rows], units[rows]),
                      "shift_smaller_unit": float(shift[smaller[rows]].mean()) if smaller[rows].any() else None,
                      "shift_larger_unit": float(shift[~smaller[rows]].mean()) if (~smaller[rows]).any() else None,
                      "within_prompt": within_unit_r2(y, pooled, units),
                      "within_horizon": horizon_within_unit_r2(y, pooled, units),
                      "share": unit_duration_shares(Z, durations, units)}
    return out


# --- behavior --------------------------------------------------------------------------

def behavior(near_log_odds, chose_near, pairs, natural, units, natural_units):
    """Same choice across a pair's renderings, and the other rendering's near
    log-odds minus the natural one's (bootstrap CI), by unit direction."""
    by_pair = {}
    for p, c in zip(pairs, chose_near):
        by_pair.setdefault(p, set()).add(bool(c))
    rows, diff = pair_shifts(near_log_odds, np.zeros(len(pairs)), pairs, natural)
    smaller = smaller_unit(units, natural_units)[rows]
    return {"pairs": len(by_pair), "same_choice": float(np.mean([len(c) == 1 for c in by_pair.values()])),
            "diff_all": bootstrap_mean(diff), "abs_diff": float(np.abs(diff).mean()),
            "diff_smaller_unit": bootstrap_mean(diff[smaller]), "diff_larger_unit": bootstrap_mean(diff[~smaller])}


# --- reading the run --------------------------------------------------------------------

def single_rendering(row):
    """Pair labels for a full-bank row: one natural rendering, its own pair."""
    h = natural_horizon(row["horizon"])
    return {"pair_id": row["sample_uid"], "duration_id": str(h), "rendered_unit": h.unit, "natural": True}


def load_run(root, mode, bank="units", model=MODEL):
    """The run's activation file and per-kept-prompt labels, or None. `bank` is "units"
    or "fill" (paired banks) or "full", whose prompts are single natural renderings,
    each its own pair. Rows without a horizon (the full bank's null prompts) are
    dropped; labels["rows"] indexes the kept rows in the activation file."""
    run = root / f"exp0_{model}_think-{mode}_{bank}"
    if not (run / "activations.npz").exists():
        return None
    z = np.load(run / "activations.npz")
    rows = [r for r in json.loads((run / "answers.json").read_text()) if "think_len" in r]   # kept prompts
    if len(rows) != len(z["horizons"]):
        raise ValueError(f"{run}: {len(rows)} kept answers but {len(z['horizons'])} activation rows")
    keep = np.array([i for i, r in enumerate(rows) if r["horizon"] is not None])
    rows = [rows[i] if "pair_id" in rows[i] else {**rows[i], **single_rendering(rows[i])} for i in keep]
    natural_unit = {r["pair_id"]: r["rendered_unit"] for r in rows if r["natural"]}
    ab = z["label_logits"][:, 0] - z["label_logits"][:, 1]
    labels = {"y": np.log10(z["horizons"][keep]),
              "durations": np.array([r["duration_id"] for r in rows]),
              "units": np.array([r["rendered_unit"] for r in rows]),
              "natural_units": np.array([natural_unit[r["pair_id"]] for r in rows]),
              "pairs": np.array([r["pair_id"] for r in rows]),
              "natural": np.array([r["natural"] for r in rows]),
              "sample_uids": np.array([r["sample_uid"] for r in rows]),
              "regime": np.array([r["regime"] for r in rows]),
              "forced": np.array([bool(r.get("forced", False)) for r in rows]),
              "near_log_odds": np.where([r["near_label"] == "a" for r in rows], ab[keep], -ab[keep]),
              "chose_near": np.array([r["label"] == r["near_label"] for r in rows]),
              "anchor": np.array([bool(r.get("anchor", False)) for r in rows]),
              "rows": keep}
    return z, labels


def cell_job(args):
    root, mode, position, layer = args
    z, labels = load_run(root, mode)
    meta = {"regions": {r: tuple(int(x) for x in z[f"region_{r}"]) for r in REGIONS},
            "tokens": [str(t) for t in z["tokens"]]}
    X = z[f"L{layer}"][:, positions(meta)[position]].astype(np.float64)
    keys = ("y", "durations", "units", "natural_units", "pairs", "natural")
    return mode, position, layer, cell_measures(X, *(labels[key] for key in keys))


def summary_text(results, behaviors):
    lines = ["Units bank (14_spec_unit_overlap.md): duration or unit word?",
             "transfer: R² of a probe fitted on natural renderings, on held-out natural / other renderings;",
             "shift: other minus natural readout, decades (grid step 0.15); unit sh / dur sh: variance only",
             "the unit / only the duration explains (type II, per-(duration, unit) means; the rest is shared).", ""]
    for mode, b in behaviors.items():
        lines.append(f"behavior, thinking {mode}: {b['pairs']} pairs, same choice {b['same_choice']:.2f}; "
                     f"near log-odds other - natural {b['diff_all'][0]:+.2f} [{b['diff_all'][1]:+.2f}, {b['diff_all'][2]:+.2f}], "
                     f"mean |diff| {b['abs_diff']:.2f}")
    for mode, cells in results.items():
        for position, layers in cells.items():
            lines += ["", f"thinking {mode}, {position}  (k = 8)",
                      f"{'layer':>5} {'R² nat':>7} {'R² other':>8} {'|shift|':>7} {'smaller':>7} {'larger':>7} "
                      f"{'within/p':>8} {'within/h':>8} {'unit sh':>7} {'dur sh':>7}"]
            for layer in sorted(layers):
                m = layers[layer][KS[-1]]
                fmt = lambda v: f"{v:+.2f}" if v is not None else "  -  "  # noqa: E731
                lines.append(f"{int(layer):>5} {m['r2_natural']:7.2f} {m['r2_other']:8.2f} {m['shift_abs']:7.2f} "
                             f"{fmt(m['shift_smaller_unit']):>7} {fmt(m['shift_larger_unit']):>7} "
                             f"{m['within_prompt']:8.2f} {m['within_horizon']:8.2f} "
                             f"{m['share']['unit']:7.2f} {m['share']['duration']:7.2f}")
    return "\n".join(lines)


def plot(results, path):
    """Per mode (rows) and position (columns), by layer at k = 8: transfer R² on natural
    and other renderings, mean |shift| and the unit's variance share."""
    modes = list(results)
    fig, axes = plt.subplots(len(modes), len(POSITIONS), figsize=(4.4 * len(POSITIONS), 3.4 * len(modes)),
                             squeeze=False)
    for i, mode in enumerate(modes):
        for j, position in enumerate(POSITIONS):
            ax = axes[i, j]
            layers = results[mode].get(position)
            if not layers:
                ax.axis("off")
                continue
            xs = sorted(layers)
            m = [layers[l][KS[-1]] for l in xs]
            ax.plot([int(l) for l in xs], [v["r2_natural"] for v in m], color="#2a78d6", label="R², natural")
            ax.plot([int(l) for l in xs], [v["r2_other"] for v in m], color="#eb6834", label="R², other rendering")
            ax.plot([int(l) for l in xs], [v["shift_abs"] for v in m], color="#8a8a85", ls="--", label="mean |shift|, decades")
            ax.plot([int(l) for l in xs], [v["share"]["unit"] for v in m], color="#1baf7a", ls=":", label="unit variance share")
            ax.set_ylim(-0.2, 1.05)
            ax.set_title(f"thinking {mode}, {position}", fontsize=9)
            ax.set_xlabel("layer", fontsize=8)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{MODEL}, units bank: probe fitted on natural renderings (k = {KS[-1]})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(root):
    root = Path(root)
    jobs, behaviors = [], {}
    for mode in ("off", "on"):
        loaded = load_run(root, mode)
        if loaded is None:
            continue
        z, labels = loaded
        behaviors[mode] = behavior(labels["near_log_odds"], labels["chose_near"], labels["pairs"], labels["natural"],
                                   labels["units"], labels["natural_units"])
        n_layers = sum(1 for key in z.files if key[0] == "L" and key[1:].isdigit())
        for position in POSITIONS:
            if position == "think close" and mode == "off":
                continue
            jobs += [(root, mode, position, layer) for layer in range(n_layers)]
    results = {}
    with ProcessPoolExecutor(WORKERS) as pool:
        for n, (mode, position, layer, measures) in enumerate(pool.map(cell_job, jobs), 1):
            results.setdefault(mode, {}).setdefault(position, {})[layer] = measures
            if n % 10 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} cells", flush=True)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    (out / "unit_overlap.json").write_text(json.dumps({"ks": KS, "behavior": behaviors, "results": results}, indent=1))
    (out / "unit_overlap.txt").write_text(summary_text(results, behaviors))
    plot(results, out / "unit_overlap.png")
    print(f"wrote {out}/unit_overlap.{{json,txt,png}}")


if __name__ == "__main__":
    main(sys.argv[1])
