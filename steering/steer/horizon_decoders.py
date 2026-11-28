"""Horizon directions with the regime part removed: `trend` and `probe`.

Both work on activations and log-horizons demeaned within each regime, so a regime's offset (which answer is correct
there, and its prompt template) cannot enter them. `trend` is the covariance of the activation with the log-horizon,
the "activation pattern" of a horizon decoder (Haufe et al. 2014), which is what an additive edit needs. `probe` is the
ridge decoder's weight vector for the same target, kept for comparison: decoder weights can point away from the
pattern where the noise directions are correlated with the horizon.
"""

from __future__ import annotations

import numpy as np

from steer.directions import Direction
from steer.stats import spearman, unit

HORIZON_FOLDS = 5
RIDGE_ALPHAS = np.logspace(0, 6, 13)


def demean_by_regime(X: np.ndarray, t: np.ndarray, regime: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float64 copies of X and t with each regime's mean subtracted."""
    Xc, tc = X.astype(np.float64), np.array(t, dtype=np.float64)
    for g in set(regime):
        m = regime == g
        Xc[m] -= Xc[m].mean(axis=0)
        tc[m] -= tc[m].mean()
    return Xc, tc


def horizon_folds(t: np.ndarray, k: int = HORIZON_FOLDS) -> list[np.ndarray]:
    """`k` boolean test masks, each holding whole horizons: the sorted distinct horizons are dealt out in turn."""
    fold_of = {value: i % k for i, value in enumerate(np.unique(t))}
    fold = np.array([fold_of[value] for value in t])
    return [fold == i for i in range(k)]


def ridge_weights(Xc: np.ndarray, tc: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Ridge regression weights [len(alphas), d] of tc on Xc (no intercept), by the SVD closed form."""
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    coef = (U.T @ tc)[None, :] * s[None, :] / (s[None, :] ** 2 + alphas[:, None])
    return coef @ Vt


def horizon_decoders(X: np.ndarray, t: np.ndarray, regime: np.ndarray) -> tuple[Direction, Direction, dict]:
    """(trend, probe, diagnostics) for activations X [n, d], log10-days t [n] and regimes [n].

    Held-out means held-out horizons: HORIZON_FOLDS folds of whole horizons. Regime means are taken once over all
    prompts, and the ridge penalty is chosen on the same folds that score it, so the Spearman values are slightly
    optimistic for `probe`."""
    Xc, tc = demean_by_regime(X, t, regime)
    folds = horizon_folds(t)
    trend_score, probe_score = np.empty(len(t)), np.empty((len(RIDGE_ALPHAS), len(t)))
    for test in folds:
        trend_score[test] = Xc[test] @ unit(Xc[~test].T @ tc[~test])
        probe_score[:, test] = ridge_weights(Xc[~test], tc[~test], RIDGE_ALPHAS) @ Xc[test].T
    held_out_mse = ((probe_score - tc) ** 2).mean(axis=1)
    best = int(np.argmin(held_out_mse))
    raw_trend = Xc.T @ tc
    raw_probe = ridge_weights(Xc, tc, RIDGE_ALPHAS[best:best + 1])[0]
    trend = Direction(unit(raw_trend), float(np.linalg.norm(raw_trend)), len(t), 0)
    probe = Direction(unit(raw_probe), float(np.linalg.norm(raw_probe)), len(t), 0)
    diagnostics = {
        "n_prompts": len(t), "n_horizons": int(len(np.unique(t))), "folds": HORIZON_FOLDS,
        "trend": {"held_out_horizon_spearman": spearman(trend_score, tc)},
        "probe": {"alpha": float(RIDGE_ALPHAS[best]), "alpha_at_grid_edge": best in (0, len(RIDGE_ALPHAS) - 1),
                  "held_out_horizon_spearman": spearman(probe_score[best], tc),
                  "held_out_mse": float(held_out_mse[best])},
        "cos_trend_probe": float(trend.vector @ probe.vector),
    }
    return trend, probe, diagnostics
