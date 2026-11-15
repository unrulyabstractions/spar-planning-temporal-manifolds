"""The fit itself, as a library function: directions, curve and diagnostics from activations and prompt records.

`fit.py` is the command-line wrapper (loading, writing, logging). The same function serves refits on chosen prompts or
relabelled horizons, e.g. for a label-shuffled null. Definitions are in fit.py's docstring and the README.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from pbank.durations import Duration
from steer.directions import Direction, mean_difference, random_directions, threshold_masks, typical_norm
from steer.manifold import Manifold, fit_manifold
from steer.stats import grouped_folds, spearman

FIT_REGIMES = ("none", "near_only", "both")


@dataclass(frozen=True)
class FitSettings:
    """Every setting that changes the fit (fit.py's options of the same names)."""
    regimes: tuple = FIT_REGIMES
    caa_long_min: float = 4.0
    caa_short_max: float = 0.25
    reward_high_min: float = 25
    reward_low_max: float = 5
    n_random: int = 100
    components: int = 4
    bandwidth: float = 0.3
    folds: int = 5
    seed: int = 0


@dataclass
class FitResult:
    caa: Direction
    reward: Direction
    choice: Direction
    random: np.ndarray          # [n_random, d] unit vectors
    typical_norm: float
    manifold: Manifold
    per_horizon: list[dict]     # per_horizon.jsonl rows
    diagnostics: dict           # fit.json "diagnostics"
    regime_counts: dict         # fit.json "regimes"
    n_prompts: int


def log10_days(duration: str) -> float:
    return float(np.log10(float(Duration.parse(duration).days)))


def select(uids: list[str], records: dict, regimes) -> list[int]:
    """Indices into `uids` of the prompts a fit uses: a stated horizon and a regime in `regimes`."""
    return [i for i, uid in enumerate(uids)
            if records[uid].horizon is not None and records[uid].regime in regimes]


def cross_validated_t(X, t, groups, folds_of, k, bandwidth) -> np.ndarray:
    """t-hat of every prompt from a curve fitted without its fold."""
    t_hat = np.empty(len(X))
    for test in folds_of:
        m = fit_manifold(X[~test], t[~test], [g for g, te in zip(groups, test) if not te], k, bandwidth)
        t_hat[test] = m.project_t(X[test])
    return t_hat


def fit(X: np.ndarray, recs: list, chosen: np.ndarray, s: FitSettings) -> FitResult:
    """Fit on the rows of X; recs[i] is row i's prompt record and chosen[i] its baseline choice ("near"/"far")."""
    regime = np.array([r.regime for r in recs], dtype=object)   # object: elements stay plain str (clean JSON keys)
    t = np.array([log10_days(r.horizon_grid_value) for r in recs])
    groups = [r.horizon_grid_value for r in recs]
    hd = np.array([r.h_over_far_delay for r in recs])
    mult = np.array([r.reward_multiplier for r in recs])

    # Directions.
    long_, short = threshold_masks(hd, s.caa_long_min, s.caa_short_max)
    caa = mean_difference(X, long_, short)
    high, low = threshold_masks(mult, s.reward_high_min, s.reward_low_max)
    reward = mean_difference(X, high, low)
    choice = mean_difference(X, chosen == "far", chosen == "near")
    random = random_directions(X.shape[1], s.n_random, s.seed)
    norm = typical_norm(X)

    # Manifold curve, and its check on held-out prompts.
    manifold = fit_manifold(X, t, groups, s.components, s.bandwidth)
    folds = grouped_folds([r.pair_id or r.uid for r in recs], s.folds, s.seed)
    t_cv = cross_validated_t(X, t, groups, folds, s.components, s.bandwidth)
    err = np.abs(t_cv - t)
    t_lo, t_hi = manifold.t_range

    names = sorted(set(groups), key=lambda g: t[groups.index(g)])
    per_horizon = []
    for g, tm, zm in zip(names, manifold.t_means, manifold.z_means):
        mask = np.array([x == g for x in groups])
        mean = X[mask].mean(axis=0, dtype=np.float64)
        per_horizon.append({"horizon": g, "log10_days": round(float(tm), 4), "n": int(mask.sum()),
                            "regimes": dict(Counter(regime[mask])),
                            "chose_far": int((chosen[mask] == "far").sum()), "chose_near": int((chosen[mask] == "near").sum()),
                            "caa_projection": float(mean @ caa.vector),
                            "t_hat": float(manifold.project_t(mean[None])[0])})

    by_regime = lambda mask: dict(Counter(regime[mask]))
    diagnostics = {
        "caa": {"n_long": caa.n_plus, "n_short": caa.n_minus, "long_by_regime": by_regime(long_),
                "short_by_regime": by_regime(short), "raw_norm": caa.raw_norm,
                "raw_norm_over_typical_norm": caa.raw_norm / norm,
                "cos_reward": float(caa.vector @ reward.vector), "cos_choice": float(caa.vector @ choice.vector),
                "max_abs_cos_random": float(np.abs(random @ caa.vector).max()),
                "spearman_logH_vs_mean_projection": spearman([h["log10_days"] for h in per_horizon],
                                                             [h["caa_projection"] for h in per_horizon])},
        "reward": {"n_high": reward.n_plus, "n_low": reward.n_minus, "raw_norm": reward.raw_norm},
        "choice": {"n_far": choice.n_plus, "n_near": choice.n_minus, "raw_norm": choice.raw_norm},
        "typical_norm": norm,
        "manifold": {
            "components": s.components, "bandwidth_decades": s.bandwidth,
            "explained_variance_of_means": [round(float(v), 4) for v in manifold.explained],
            "t_range_log10_days": [t_lo, t_hi],
            "in_sample_median_abs_err_decades": float(np.median(np.abs(
                [h["t_hat"] - h["log10_days"] for h in per_horizon]))),
            "held_out": {"folds": s.folds, "spearman": spearman(t, t_cv),
                         "median_abs_err_decades": float(np.median(err)),
                         "median_abs_err_by_regime": {r: float(np.median(err[regime == r])) for r in sorted(set(regime))},
                         "share_at_range_ends": float(np.mean((t_cv <= t_lo) | (t_cv >= t_hi)))},
        },
    }
    return FitResult(caa=caa, reward=reward, choice=choice, random=random, typical_norm=norm, manifold=manifold,
                     per_horizon=per_horizon, diagnostics=diagnostics, regime_counts=dict(Counter(regime)),
                     n_prompts=len(recs))
