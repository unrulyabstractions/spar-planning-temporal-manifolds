"""The fit itself, as a library function: directions, curve and diagnostics from activations and prompt records.

`fit.py` is the command-line wrapper (loading, writing, logging). The same function serves refits on chosen prompts or
relabelled horizons, e.g. for a label-shuffled null. Definitions are in fit.py's docstring and the README.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pbank.durations import Duration
from steer.directions import Direction, mean_difference, random_directions, threshold_masks, typical_norm
from steer.manifold import Manifold, fit_manifold
from steer.stats import grouped_folds, spearman

FIT_REGIMES = ("none", "near_only", "both")


@dataclass(frozen=True)
class FitSettings:
    """Every setting that changes the fit (fit.py's options of the same names)."""
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
    random: np.ndarray          # [n, d] isotropic unit vectors: the fit's own, or the shared draws' if given
    typical_norm: float
    manifold: Manifold
    per_horizon: list[dict]     # per_horizon.jsonl rows
    diagnostics: dict           # fit.json "diagnostics"
    regime_counts: dict         # fit.json "regimes"
    n_prompts: int
    caa_class: np.ndarray       # per row: "long", "short" or "" (in neither CAA class)
    reward_class: np.ndarray    # per row: "high", "low" or ""
    fold: np.ndarray            # per row: the held-out fold it was predicted in (0 .. folds-1)


def log10_days(duration: str) -> float:
    return float(np.log10(float(Duration.parse(duration).days)))


def fit_regimes(include_none: bool) -> tuple:
    """The regimes a fit uses. "none" prompts (no option pays off within the horizon, so no answer is correct) are
    in by default, as the report's Methods section says they are "used for extraction"."""
    return FIT_REGIMES if include_none else tuple(r for r in FIT_REGIMES if r != "none")


@dataclass(frozen=True)
class Selection:
    rows: list[int]          # indices into the activation set's rows, in row order
    excluded: dict           # reason -> number of extracted prompts left out for it


def select(uids: list[str], records: dict, include_none: bool = True, exclude_horizons=()) -> Selection:
    """The prompts a fit uses, out of an activation set's `uids`. In order, a prompt is left out if it is:
      * not in the extraction split (the verification grid is never fitted on, whatever extract.py was run on);
      * a no-horizon (null control) prompt: it has no horizon to place on the curve;
      * in the "none" regime, when include_none is false;
      * at a stated horizon in `exclude_horizons` (grid values, e.g. "3 years"), held out for calibration.
    The first applicable reason is counted."""
    exclude = set(exclude_horizons)
    regimes = fit_regimes(include_none)
    rows, excluded = [], {"not_extraction_split": 0, "no_horizon": 0, "none_regime": 0, "excluded_horizon": 0}
    for i, uid in enumerate(uids):
        r = records[uid]
        reason = ("not_extraction_split" if r.split != "extraction" else
                  "no_horizon" if r.horizon is None else
                  "none_regime" if r.regime not in regimes else
                  "excluded_horizon" if r.horizon_grid_value in exclude else None)
        if reason:
            excluded[reason] += 1
        else:
            rows.append(i)
    return Selection(rows=rows, excluded=excluded)


def cross_validated_t(X, t, groups, folds_of, k, bandwidth) -> np.ndarray:
    """t-hat of every prompt from a curve fitted without its fold."""
    t_hat = np.empty(len(X))
    for test in folds_of:
        m = fit_manifold(X[~test], t[~test], [g for g, te in zip(groups, test) if not te], k, bandwidth)
        t_hat[test] = m.project_t(X[test])
    return t_hat


def fit(X: np.ndarray, recs: list, chosen: np.ndarray, s: FitSettings,
        random_sets: Optional[dict] = None) -> FitResult:
    """Fit on the rows of X; recs[i] is row i's prompt record and chosen[i] its baseline choice ("near"/"far").

    random_sets: the random unit vectors of the random-baseline diagnostic, {family: [n, d]}, with at least
    "isotropic" (e.g. steer.draws.load_shared(...).families). Default: s.n_random isotropic vectors from s.seed. They
    enter only the diagnostics and FitResult.random, never the methods' parameters."""
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
    if random_sets is None:
        random_sets = {"isotropic": random_directions(X.shape[1], s.n_random, s.seed)}
    random = random_sets["isotropic"]
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
                "abs_cos_random": {f: {"n": int(len(v)), "max": float(np.abs(v @ caa.vector).max()),
                                       "p95": float(np.percentile(np.abs(v @ caa.vector), 95))}
                                   for f, v in random_sets.items()},
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
    fold = np.full(len(recs), -1)
    for f, test in enumerate(folds):
        fold[test] = f
    return FitResult(caa=caa, reward=reward, choice=choice, random=random, typical_norm=norm, manifold=manifold,
                     per_horizon=per_horizon, diagnostics=diagnostics, regime_counts=dict(Counter(regime)),
                     n_prompts=len(recs), caa_class=np.where(long_, "long", np.where(short, "short", "")),
                     reward_class=np.where(high, "high", np.where(low, "low", "")), fold=fold)
