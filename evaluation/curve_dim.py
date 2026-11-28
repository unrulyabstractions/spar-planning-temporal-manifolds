"""How many dimensions does the horizon curve need? (CPU only: numpy; shared by diagnose.py and layer_sweep.py.)

Two questions about the per-horizon mean activations of one site:
  * which principal components of the means carry horizon structure beyond chance: each component's eigenvalue is
    compared with the same eigenvalue after the horizon labels are shuffled across prompts (95th percentile of
    N_PERMUTATIONS shuffles; absolute units, so a component is "above the null" when the ratio exceeds 1);
  * how well a curve of k components (the steering curve of steer.manifold) places a prompt whose horizon was never
    fitted, against a ridge probe of log10 horizon on the raw activations.
Folds hold out whole horizon values (names dealt to the folds in turn) (a prompt-level fold would score a curve on horizons it was fitted on).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "steering"), str(ROOT / "prompt_bank")]

from steer.fitting import cross_validated_t, log10_days   # noqa: E402,F401  (log10_days is re-exported)
from steer.stats import spearman                           # noqa: E402

N_PERMUTATIONS = 50
SEED = 0
TABLE_KS = tuple(range(1, 9))        # eigenvalue / null rows
CURVE_KS = (1, 2, 3, 4, 6, 8)        # curve fits scored on held-out horizons
N_FOLDS = 5
BANDWIDTH = 0.3                      # decades; the steering curve's default
RIDGE_ALPHAS = np.logspace(0, 6, 13)
MAX_COMPONENTS_TESTED = 10


def horizon_folds(groups, n_folds=N_FOLDS):
    """Test masks that hold out whole horizons: the sorted distinct horizon names are dealt to the folds in turn. The
    names sort as text, which scatters the horizons over the folds without regard to their length."""
    fold_of = {g: i % n_folds for i, g in enumerate(sorted(set(groups)))}
    fold = np.array([fold_of[g] for g in groups])
    return [fold == i for i in range(n_folds)]


def mean_eigenvalues(X, group_codes, n):
    """Squared singular values of the centred per-group mean activations (the means' PCA eigenvalues, unnormalised)."""
    sums = np.zeros((group_codes.max() + 1, X.shape[1]))
    np.add.at(sums, group_codes, X)
    means = sums / np.bincount(group_codes)[:, None]
    return np.linalg.svd(means - means.mean(0), compute_uv=False)[:n] ** 2


def eigenvalues_against_null(X, groups, n_perm=N_PERMUTATIONS, seed=SEED, n=MAX_COMPONENTS_TESTED):
    """(eigenvalues, null 95th percentile) of the first n components of the horizon means; the null shuffles the
    horizon labels across prompts."""
    _, codes = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    n = min(n, codes.max())
    lam = mean_eigenvalues(X, codes, n)
    null = np.array([mean_eigenvalues(X, rng.permutation(codes), n) for _ in range(n_perm)])
    return lam, np.percentile(null, 95, axis=0)


def leading_components_above(lam, null95):
    """How many components, counting from the first, exceed their null before one does not."""
    below = lam <= null95
    return int(np.argmax(below)) if below.any() else len(lam)


def ridge_predict(X_train, y_train, X_test, alphas=RIDGE_ALPHAS):
    """Ridge regression with an intercept; alpha by leave-one-out error on the training rows (closed form)."""
    x_mean, y_mean = X_train.mean(0), y_train.mean()
    U, s, Vt = np.linalg.svd(X_train - x_mean, full_matrices=False)
    uy = U.T @ (y_train - y_mean)
    loo_mse = []
    for a in alphas:
        shrink = s ** 2 / (s ** 2 + a)
        residual = (y_train - y_mean) - U @ (shrink * uy)
        leverage = 1 / len(y_train) + (U ** 2) @ shrink
        loo_mse.append(np.mean((residual / (1 - leverage)) ** 2))
    alpha = alphas[int(np.argmin(loo_mse))]
    coef = Vt.T @ (s / (s ** 2 + alpha) * uy)
    return (X_test - x_mean) @ coef + y_mean


def held_out_scores(t, t_hat):
    return {"spearman": spearman(t, t_hat), "median_abs_err": float(np.median(np.abs(t_hat - t)))}


def analyse(X, t, groups, ks=CURVE_KS, n_perm=N_PERMUTATIONS, seed=SEED):
    """Everything the curve-dimensionality table shows for activations X [n, d] of prompts at log10 horizons t with
    horizon names `groups`: eigenvalue / null ratios, components above the null, and held-out-horizon scores of the
    curve at each k in `ks` and of the ridge probe."""
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(t, dtype=float)
    groups = list(groups)
    lam, null95 = eigenvalues_against_null(X, groups, n_perm, seed)
    folds = horizon_folds(groups)
    curve = {k: held_out_scores(t, cross_validated_t(X, t, groups, folds, k, BANDWIDTH)) for k in ks}
    ridge_hat = np.empty(len(X))
    for test in folds:
        ridge_hat[test] = ridge_predict(X[~test], t[~test], X[test])
    return {"n_prompts": len(X), "n_horizons": len(set(groups)), "eigenvalue": lam.tolist(), "null95": null95.tolist(),
            "eig_over_null95": (lam / null95).tolist(), "n_above_null": leading_components_above(lam, null95),
            "curve": curve, "ridge": held_out_scores(t, ridge_hat)}


def verdict(result, k_used, k_more=8):
    """One plain line: the curve at the bundle's k against the ridge probe, and what k_more components would add."""
    curve = result["curve"]
    line = (f"curve at k={k_used}, held-out horizons: rho {curve[k_used]['spearman']:.2f} "
            f"(ridge {result['ridge']['spearman']:.2f})")
    if k_more in curve and k_more != k_used:
        line += f"; k={k_more} adds {curve[k_more]['spearman'] - curve[k_used]['spearman']:+.2f}"
    return line + f"; {result['n_above_null']} of {len(result['eigenvalue'])} components tested above the shuffled-horizon null"
