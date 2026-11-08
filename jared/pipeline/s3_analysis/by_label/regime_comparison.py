"""Is there an explicit horizon-vs-delay comparison before the hand-off?

The regime of a near/both prompt is sign(log h - log far delay): linear in log h
and in log far. A regime probe can therefore succeed by reading the horizon and
the far delay separately, without any comparison. This test removes both and
asks whether the regime can still be read.

Per cell (model, thinking mode, layer, position) on the near- and both-regime
prompts (thinking on: natural closes only), in each leave-one-horizon-out fold,
fitted on the training prompts only:
  1. project out the top 8 PCs of the per-horizon averages (steering.fit_cell),
  2. project out the top <= 5 PCs of the per-far-delay averages,
  3. fit a logistic regression for the regime on what remains,
and the pooled out-of-fold AUC is compared with a null that shuffles the regime
labels within each far delay. The full-X probe (no projection) is the reference.

    python analysis/regime_comparison.py results/horizon
    -> <root>/analysis/regime_comparison.json and .png
    python analysis/regime_comparison.py results/horizon --within --shuffles 200
    -> regime_comparison_within.json and .png: the same cells scored by the AUC inside each
       held-out horizon (one horizon has one score level, and a probe cannot learn the far delay
       once its per-delay means are projected out, so only a comparison can rank the regimes there)
    python analysis/regime_comparison.py results/horizon --far-control
    -> regime_comparison_far_control.json: that within-horizon AUC against a probe trained on a
       fixed far-delay threshold instead of the regime (exactly 0.5 by construction: a check)
    python analysis/regime_comparison.py results/horizon --shift-control --layers 16,18,22,25,28,30
    -> regime_comparison_shift.json: the within-horizon AUC for thresholds moved by SHIFT_GRID
       decades, sign(log h - log far - shift), each with its own probe. A comparison at h = far
       peaks at shift 0; a smooth horizon x delay interaction scores shifted thresholds alike
    python analysis/regime_comparison.py results/horizon --models Qwen3-8B --layers 20,22 --shuffles 50   (quick look)

Reads the saved activations only; runs no model.
"""

import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from horizon_shape import cell_data  # noqa: E402  (same folder)
from steering_analysis import saved_decision  # noqa: E402  (same folder)
from steering_prepare import run_dir, run_settings  # noqa: E402  (same folder)
from spar_horizon.banks import build_records  # noqa: E402
from spar_horizon.steering import fit_cell  # noqa: E402

MODELS = {"Qwen3-8B": 36, "Qwen3-14B": 40}
MODES = ("off", "on")
POSITIONS = ("suffix newline", "pre-label")
K_HORIZON, K_FAR = 8, 5
C = 100.0            # on features scaled to unit mean squared norm; see pilot note in the report
SHUFFLES, MAX_SHUFFLES = 200, 1000
SHIFTS = (-0.5, 0.5)  # decades; see offset_auc
SHIFT_GRID = (-0.5, -0.25, 0.0, 0.25, 0.5)  # decades; see shift_control
EXTEND_ABOVE = 0.90  # a cell whose AUC beats this null quantile at 200 shuffles gets 1000
HANDOFF_FIRST = {"Qwen3-8B": 22, "Qwen3-14B": 25}   # pre-registered: nothing above the null before these layers


def reduce_to_span(X):
    """Coordinates of the rows of X in the span of the centered data, scaled to unit
    mean squared norm so that one penalty C suits every layer. n < d, so this is a
    lossless rotation (no labels used): linear projections and L2-penalized probes
    give the same answers in it, at a fraction of the cost."""
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    rank = int((S > 1e-8 * S[0]).sum())
    R = U[:, :rank] * S[:rank]
    return R / np.sqrt((R ** 2).sum(1).mean()) if rank else R


def projected_folds(R, log_h, log_far, k_horizon=K_HORIZON, k_far=K_FAR):
    """Leave-one-horizon-out folds [(train, test, F_train, F_test)], where F is R with
    the horizon-average and far-delay-average subspaces of the training rows removed."""
    folds = []
    groups = np.round(log_h, 9)
    for horizon in np.unique(groups):
        test = groups == horizon
        train = ~test
        basis = np.vstack([fit_cell(R[train], log_h[train], k_max=k_horizon)["basis"],
                           fit_cell(R[train], log_far[train], k_max=k_far)["basis"]])
        # the two subspaces overlap (far delays come with their horizons), so orthonormalize the union
        _, singular, vt = np.linalg.svd(basis, full_matrices=False)
        basis = vt[singular > 1e-8 * singular[0]]
        mean = R[train].mean(0)
        F_train, F_test = (R[train] - mean), (R[test] - mean)
        F_train, F_test = F_train - F_train @ basis.T @ basis, F_test - F_test @ basis.T @ basis
        # what is left is small next to the horizon variance, so rescale it or C would over-regularize
        scale = np.sqrt((F_train ** 2).sum(1).mean())
        folds.append((train, test, F_train / scale, F_test / scale))
    return folds


def unprojected_folds(R, log_h):
    """The same folds without the projection (the full-X reference); R is already at unit scale."""
    groups = np.round(log_h, 9)
    return [(groups != h, groups == h, R[groups != h], R[groups == h]) for h in np.unique(groups)]


def out_of_fold_scores(folds, labels):
    """Pooled out-of-fold decision values of a logistic regime probe. Balanced class weights,
    because a held-out horizon is nearly all one regime, which shifts the training class
    balance and so the intercept against the held-out label, and the pooled AUC with it."""
    scores = np.empty(len(labels))
    for train, test, F_train, F_test in folds:
        scores[test] = LogisticRegression(C=C, max_iter=500, class_weight="balanced").fit(F_train, labels[train]).decision_function(F_test)
    return scores


def shuffle_within(labels, groups, rng):
    """Labels permuted among the rows of each group (here: each far delay)."""
    shuffled = labels.copy()
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        shuffled[rows] = labels[rng.permutation(rows)]
    return shuffled


def pooled_auc(scores, labels, log_h):
    return roc_auc_score(labels, scores)


def within_horizon_auc(scores, labels, log_h):
    """Mean AUC inside each held-out horizon that has both regimes. Inside one horizon the
    score cannot rank by horizon, and the regime is set by the far delay alone, so this
    asks for the far delay or the comparison beyond what the projection removed.
    NaN when no horizon has both regimes."""
    groups = np.round(log_h, 9)
    aucs = [roc_auc_score(labels[groups == g], scores[groups == g]) for g in np.unique(groups)
            if 0 < labels[groups == g].sum() < (groups == g).sum()]
    return float(np.mean(aucs)) if aucs else float("nan")


def null_aucs(folds, labels, far_groups, log_h, shuffles, rng, metric=pooled_auc):
    values = []
    for _ in range(shuffles):
        shuffled = shuffle_within(labels, far_groups, rng)
        values.append(metric(out_of_fold_scores(folds, shuffled), shuffled, log_h))
    return np.array(values)


def offset_auc(folds, log_h, log_far, shift):
    """Control, not part of the pre-registered test: the same projected folds reading the
    labels sign(log h - log far - shift). If the regime AUC is just what the horizon
    information left after the projection gives for a threshold on h, these score alike;
    an explicit comparison would favour the true threshold (shift 0)."""
    labels = log_h - log_far > shift
    if labels.all() or not labels.any():
        return float("nan")
    return float(roc_auc_score(labels, out_of_fold_scores(folds, labels)))   # pooled, whatever the main metric


def wilson(k, n, z=1.96):
    """95% Wilson interval for k of n."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def follow_counts(scores, regime_both, chose_near):
    """How the held-out probe reads the irrational prompts (chose far in `near`, chose
    near in `both`) and, as a control, the rational ones: calls that match the regime."""
    irrational = chose_near == regime_both   # regime both, chose near / regime near, chose far
    follows_regime = (scores > 0) == regime_both
    out = {}
    for name, rows in (("irrational", irrational), ("rational", ~irrational)):
        k, n = int(follows_regime[rows].sum()), int(rows.sum())
        out[name] = {"follows_regime": k, "n": n, "ci": wilson(k, n)}
    return out


def evaluate_cell(X, log_h, log_far, regime_both, chose_near, seed, metric=pooled_auc):
    """Reference and projected AUCs, the null, and the irrational-prompt counts for one cell,
    or None when the projection would leave nothing (embedding layer: one token per position)."""
    R = reduce_to_span(X)
    if R.shape[1] <= K_HORIZON + K_FAR:
        return None
    far_groups = np.round(log_far, 9)
    reference = out_of_fold_scores(unprojected_folds(R, log_h), regime_both)
    folds = projected_folds(R, log_h, log_far)
    scores = out_of_fold_scores(folds, regime_both)
    auc = metric(scores, regime_both, log_h)
    rng = np.random.default_rng(seed)
    null = null_aucs(folds, regime_both, far_groups, log_h, SHUFFLES, rng, metric)
    if auc > np.nanquantile(null, EXTEND_ABOVE):
        null = np.r_[null, null_aucs(folds, regime_both, far_groups, log_h, MAX_SHUFFLES - SHUFFLES, rng, metric)]
    return {"auc": float(auc), "reference_auc": float(metric(reference, regime_both, log_h)),
            "null_p95": float(np.nanquantile(null, 0.95)), "null_mean": float(np.nanmean(null)),
            "p_value": float((1 + (null >= auc).sum()) / (1 + len(null))), "shuffles": len(null),
            "offset_auc": {str(shift): offset_auc(folds, log_h, log_far, shift) for shift in SHIFTS},
            "choice_auc": float(metric(scores, ~chose_near, log_h)),
            "decision": follow_counts(scores, regime_both, chose_near),
            "reference_decision": follow_counts(reference, regime_both, chose_near)}


def far_threshold_control(X, log_h, log_far, regime_both, chose_near, seed):
    """Control for the within-horizon score (not pre-registered). Inside one horizon the regime
    is a threshold on the far delay, so a probe that merely kept some far-delay information would
    pass the test. Same projected folds, same horizons, same metric, but the probe is trained
    on a fixed far-delay threshold (above the median, whatever the horizon). The regime beats it
    only if the probe tracks where the horizon falls among the delays."""
    R = reduce_to_span(X)
    if R.shape[1] <= K_HORIZON + K_FAR:
        return None
    folds = projected_folds(R, log_h, log_far)
    far_high = log_far > np.median(log_far)
    groups = np.round(log_h, 9)
    both_classes = lambda labels: np.array([0 < labels[groups == g].sum() < (groups == g).sum() for g in groups])
    shared = both_classes(regime_both) & both_classes(far_high)   # rows of horizons where both labels vary
    return {"regime_auc": within_horizon_auc(out_of_fold_scores(folds, regime_both)[shared], regime_both[shared], log_h[shared]),
            "far_threshold_auc": within_horizon_auc(out_of_fold_scores(folds, far_high)[shared], far_high[shared], log_h[shared]),
            "horizons": int(len(np.unique(groups[shared])))}


def shift_labels(log_h, log_far, shift):
    """Regime labels for a threshold moved by `shift` decades; ties count as "both", as in the bank."""
    return log_h - log_far >= shift - 1e-9


def shift_control(X, log_h, log_far, regime_both, chose_near, seed):
    """Control for the within-horizon score (not pre-registered): {shift: within-horizon AUC}
    of a projected probe trained and scored on `shift_labels`; shift 0 is the regime itself."""
    R = reduce_to_span(X)
    if R.shape[1] <= K_HORIZON + K_FAR:
        return None
    folds = projected_folds(R, log_h, log_far)
    return {str(shift): within_horizon_auc(out_of_fold_scores(folds, shift_labels(log_h, log_far, shift)),
                                           shift_labels(log_h, log_far, shift), log_h) for shift in SHIFT_GRID}


def load_prompts(root, model, mode):
    """Per horizon prompt of the run (rows as in cell_data): the near/both mask (thinking
    on: natural closes only), log far delay, regime == both, chose_near, with the
    records rebuilt from the manifest checked against the saved activations."""
    run = run_dir(root, model, mode)
    z = np.load(run / "activations.npz")
    records = build_records(run_settings(root, model, mode))
    saved_far, saved_h = z["far_delay_years"], z["horizons"]
    if len(records) != len(saved_far) or not np.array_equal(
            [r["far_delay_years"] for r in records], saved_far) or not np.allclose(
            [np.nan if r["horizon"] is None else r["horizon"] for r in records], saved_h, equal_nan=True):
        raise ValueError(f"{run}: rebuilt records do not match the saved prompt order")
    keep = ~np.isnan(saved_h)
    rows = [r for r, k in zip(records, keep) if k]
    _, _, chose_near = saved_decision(run, z)
    regime = np.array([r["regime"] for r in rows])
    forced = z["forced_mask"][keep].astype(bool) if z["forced_mask"].size else np.zeros(len(rows), bool)
    selected = np.isin(regime, ("near", "both")) & ~forced
    return {"selected": selected, "log_far": np.log10([r["far_delay_years"] for r in rows])[selected],
            "both": (regime == "both")[selected], "chose_near": chose_near[selected]}


def run_layer(task):
    """All positions of one (model, mode, layer): cell results keyed by position."""
    root, model, mode, layer, seed, evaluate = task
    p = load_prompts(root, model, mode)
    z = np.load(run_dir(root, model, mode) / "activations.npz")
    out = {}
    for label in POSITIONS:
        X, log_h, _ = cell_data(z, label, layer)
        out[label] = evaluate(X[p["selected"]], log_h[p["selected"]], p["log_far"], p["both"], p["chose_near"], seed)
    return model, mode, layer, out


def first_above(layers, results, quantity="auc", threshold="null_p95", run_length=1):
    """First layer from which `run_length` consecutive layers have AUC above the null p95, or None."""
    above = [r[quantity] > r[threshold] for r in results]
    for i in range(len(layers) - run_length + 1):
        if all(above[i:i + run_length]):
            return layers[i]
    return None


def summarize(results):
    """Per model/mode/position: first layer above the null (one layer; two in a row)."""
    summary = {}
    for model, by_mode in results.items():
        for mode, by_position in by_mode.items():
            for label, by_layer in by_position.items():
                layers = sorted(by_layer, key=int)
                rows = [by_layer[layer] for layer in layers]
                ints = [int(layer) for layer in layers]
                summary[f"{model}|{mode}|{label}"] = {
                    "first_above_null": first_above(ints, rows),
                    "first_two_in_a_row": first_above(ints, rows, run_length=2),
                    "layers_above_null": [i for i, r in zip(ints, rows) if r["auc"] > r["null_p95"]]}
    return summary


def plot(results, path):
    fig, axes = plt.subplots(len(MODELS) * len(MODES), len(POSITIONS), figsize=(11, 12), sharey=True)
    for row, (model, mode) in enumerate((m, d) for m in MODELS for d in MODES):
        for col, label in enumerate(POSITIONS):
            ax = axes[row, col]
            by_layer = results.get(model, {}).get(mode, {}).get(label, {})
            layers = sorted(by_layer, key=int)
            x = [int(layer) for layer in layers]
            ax.plot(x, [by_layer[layer]["reference_auc"] for layer in layers], color="grey", label="full X")
            ax.plot(x, [by_layer[layer]["auc"] for layer in layers], color="C3", label="projected")
            ax.plot(x, [by_layer[layer]["null_p95"] for layer in layers], color="k", ls=":", label="null p95")
            ax.axvline(HANDOFF_FIRST[model], color="C0", lw=0.8)
            ax.set_title(f"{model}, thinking {mode}, {label}", fontsize=8)
            ax.set_ylim(0.3, 1.02)
            if row == len(axes) - 1:
                ax.set_xlabel("layer")
            if col == 0:
                ax.set_ylabel("regime AUC (leave one horizon out)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(root, models, layers, workers, evaluate, name, seed=0):
    root = Path(root)
    tasks = [(root, model, mode, layer, seed, evaluate) for model in models for mode in MODES
             for layer in (layers if layers is not None else range(MODELS[model] + 1))]
    results = {}
    out_path = root / "analysis" / f"{name}.json"
    with Pool(workers, initializer=threadpool_limits, initargs=(1,)) as pool:   # one BLAS thread per worker
        for model, mode, layer, cells in tqdm(pool.imap_unordered(run_layer, tasks), total=len(tasks)):
            for label, cell in cells.items():
                if cell is None:
                    continue
                results.setdefault(model, {}).setdefault(mode, {}).setdefault(label, {})[str(layer)] = cell
    out_path.parent.mkdir(exist_ok=True)
    report = {"results": results}
    if evaluate not in (far_threshold_control, shift_control):
        report = {"settings": {"k_horizon": K_HORIZON, "k_far": K_FAR, "C": C, "shuffles": SHUFFLES,
                               "max_shuffles": MAX_SHUFFLES, "extend_above": EXTEND_ABOVE, "offset_decades": SHIFTS},
                  "summary": summarize(results), "results": results}
        plot(results, out_path.with_suffix(".png"))
    out_path.write_text(json.dumps(report, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="folder with the full-bank E0 runs")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--layers", help="comma-separated layers (default: all)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--shuffles", type=int, help="override the 200 base shuffles (quick looks)")
    parser.add_argument("--within", action="store_true", help="score by the AUC inside each held-out horizon (see the docstring)")
    parser.add_argument("--far-control", action="store_true", help="the fixed far-delay-threshold control of the within-horizon score")
    parser.add_argument("--shift-control", action="store_true", help="the within-horizon score at shifted thresholds (see the docstring)")
    args = parser.parse_args()
    if args.shuffles:
        SHUFFLES = MAX_SHUFFLES = args.shuffles
    layers = None if args.layers is None else [int(x) for x in args.layers.split(",")]
    if args.far_control:
        main(args.root, args.models.split(","), layers, args.workers, far_threshold_control, "regime_comparison_far_control")
    elif args.shift_control:
        main(args.root, args.models.split(","), layers, args.workers, shift_control, "regime_comparison_shift")
    else:
        metric = within_horizon_auc if args.within else pooled_auc
        main(args.root, args.models.split(","), layers, args.workers, partial(evaluate_cell, metric=metric),
             "regime_comparison_within" if args.within else "regime_comparison")
