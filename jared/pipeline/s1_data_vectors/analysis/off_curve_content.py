"""What does the off-curve part of the steering subspace hold?

At the 14B thinking-on hand-off, mean-ablating the horizon subspace without the
decision direction removes 0.33 of the near-vs-both contrast while steering along
the curve without the decision moves 0.01, and ~64% of what the ablation removes
lies off the curve of horizon averages (both models). This asks what that
off-curve part carries.

Per (model, layer), thinking on, pre-label, natural closes, as steering_prepare
fits the E8 inputs: the top-8 horizon-average PCs, the decision direction inside
them, and B3 = the top 4 without the decision (the subspace E8's "horizon without
decision" ablation acts on). On the near/both prompts, everything is centered
within each horizon, so only the off-curve part can score. Within a horizon the
margin log h - log far is -log far + const, so "far delay" and "horizon relative
to the options" are the same target here. Targets:
  log far, log near, regime (both = 1), near log-odds, a-vs-b log-odds,
  near is a, log think length.
Scores: pooled within-horizon Pearson r of leave-one-horizon-out ridge predictions
  - B3 coordinates (3 features), against a null that shuffles the target within
    each horizon;
  - 3 random directions orthogonal to the top-8 basis (median of RANDOM seeds):
    a control for "any 3 directions of the residual stream";
  - the full residual stream (ridge, CV penalty): the ceiling at this cell.

    python -m pipeline.s1_data_vectors.analysis.off_curve_content /home/Jared/SPAR/results/horizon
    -> <root>/analysis/off_curve_content.json, and a table on stdout

Reads the saved activations only; runs no model. Post hoc (not pre-registered).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from tqdm import tqdm

from pipeline.s1_data_vectors.analysis.horizon_shape import cell_data
from spar_horizon.banks import build_records
from spar_horizon.e0_runs import run_dir, run_settings, saved_decision
from spar_horizon.probes import reduce_to_span
from spar_horizon.steering import basis_without, decision_direction, fit_cell, random_orthonormal

LAYERS = {"Qwen3-8B": list(range(14, 36, 2)) + [21, 23, 25], "Qwen3-14B": list(range(15, 40, 2)) + [26, 28, 30]}
TARGETS = ("log far", "log near", "regime", "near log-odds", "a-vs-b log-odds", "near is a", "log think length")
SHUFFLES = 200
RANDOM = 10
CUT_K = 4   # the k of E8's thinking-on ablation


def center_within(values, groups):
    """values [n] or [n, d] minus their mean in each group."""
    out = np.array(values, dtype=float)
    for g in np.unique(groups):
        rows = groups == g
        out[rows] -= out[rows].mean(0)
    return out


def within_r(F, target, groups, model=None):
    """Pooled within-group Pearson r of leave-one-group-out predictions of `target`
    from features F, both centered within each group (only within-group variation
    can score). `model` is a fresh sklearn regressor per fold (default: near-OLS ridge)."""
    Fc, tc = center_within(F, groups), center_within(target, groups)
    pred = np.empty(len(tc))
    for g in np.unique(groups):
        test = groups == g
        fit = (model() if model else Ridge(alpha=1e-6)).fit(Fc[~test], tc[~test])
        pred[test] = fit.predict(Fc[test])
    pred = center_within(pred, groups)
    denominator = np.sqrt((pred ** 2).sum() * (tc ** 2).sum())
    return float((pred * tc).sum() / denominator) if denominator > 0 else 0.0


def shuffle_within(target, groups, rng):
    out = target.copy()
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        out[rows] = target[rng.permutation(rows)]
    return out


def load_run(root, model):
    """Activations, the horizon rows' records and decisions, thinking on. (Not `e0_runs.load_run_activations`:
    this one needs the full records and the saved decision, and has no labels or unit pairs.)"""
    run = run_dir(root, model, "on")
    z = np.load(run / "activations.npz")
    keep = ~np.isnan(z["horizons"])
    records = [r for r, k in zip(build_records(run_settings(root, model, "on")), keep) if k]
    if not np.allclose([r["horizon"] for r in records], z["horizons"][keep]):
        raise ValueError(f"{run}: rebuilt records do not match the saved prompt order")
    ab, near_lo, _ = saved_decision(run, z)
    regime = np.array([r["regime"] for r in records])
    targets = {"log far": np.log10([r["far_delay_years"] for r in records]),
               "log near": np.log10([r["near_delay_years"] for r in records]),
               "regime": (regime == "both").astype(float),
               "near log-odds": near_lo, "a-vs-b log-odds": ab,
               "near is a": np.array([r["near_label"] == "a" for r in records], float),
               "log think length": np.log10(z["think_len"][keep])}
    return z, regime, near_lo, targets


def evaluate_layer(z, regime, near_lo, targets, layer, rng):
    X, log_h, forced = cell_data(z, "pre-label", layer)
    natural = ~forced
    # the same fit as steering_prepare.py --natural-only
    cell = fit_cell(X[natural], log_h[natural])
    decision, decision_r2 = decision_direction(X[natural], near_lo[natural], cell)
    B3 = basis_without(cell["basis"][:CUT_K], decision)
    rows = natural & np.isin(regime, ("near", "both"))
    Xs, groups = X[rows], np.round(log_h[rows], 9)
    Z3 = (Xs - cell["mean"]) @ B3.T
    # random directions orthogonal to the whole top-8 basis
    randoms = []
    for seed in range(RANDOM):
        R = random_orthonormal(X.shape[1], CUT_K - 1, seed)
        R = R - (R @ cell["basis"].T) @ cell["basis"]
        q, _ = np.linalg.qr(R.T)
        randoms.append((Xs - cell["mean"]) @ q)
    full = reduce_to_span(Xs)
    ceiling_model = lambda: RidgeCV(alphas=np.logspace(-3, 3, 13))
    out = {"decision_r2": decision_r2, "n": int(rows.sum()), "horizons": int(len(np.unique(groups))), "targets": {}}
    for name in TARGETS:
        t = targets[name][rows]
        null = [within_r(Z3, shuffle_within(t, groups, rng), groups) for _ in range(SHUFFLES)]
        out["targets"][name] = {"b3": within_r(Z3, t, groups), "null_p95": float(np.quantile(null, 0.95)),
                                "random": float(np.median([within_r(F, t, groups) for F in randoms])),
                                "full": within_r(full, t, groups, ceiling_model)}
    return out


def main(root, models):
    root = Path(root)
    results = {}
    for model in models:
        z, regime, near_lo, targets = load_run(root, model)
        rng = np.random.default_rng(0)
        results[model] = {layer: evaluate_layer(z, regime, near_lo, targets, layer, rng)
                          for layer in tqdm(sorted(LAYERS[model]), desc=model, unit="layer")}
    out = root / "analysis" / "off_curve_content.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    for model, layers in results.items():
        print(f"\n{model}  within-horizon r: B3 / null p95 / random / full   (n, horizons = "
              f"{next(iter(layers.values()))['n']}, {next(iter(layers.values()))['horizons']})")
        print("layer " + "".join(f"{name:>24}" for name in TARGETS))
        for layer, r in layers.items():
            cells = "".join(f"{v['b3']:>6.2f}/{v['null_p95']:.2f}/{v['random']:.2f}/{v['full']:.2f}"
                            .rjust(24) for v in r["targets"].values())
            print(f"L{layer:<4} {cells}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="folder with the full-bank E0 runs")
    parser.add_argument("--models", default="Qwen3-8B,Qwen3-14B")
    args = parser.parse_args()
    main(args.root, args.models.split(","))
