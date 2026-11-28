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
from steer.horizon_decoders import horizon_decoders
from steer.directions import Direction, mean_difference, random_directions, threshold_masks, typical_norm
from steer.manifold import Manifold, fit_manifold
from steer.stats import grouped_folds, spearman

FIT_REGIMES = ("none", "near_only", "both")
WITHIN_KEY = "caa_within"


def within_key(regime: str) -> str:
    return f"caa_{regime}"


def within_regime_classes(t: np.ndarray, regime: np.ndarray, quantile: float) -> tuple[dict, np.ndarray]:
    """Long and short classes inside each regime, by log-horizon t.

    In regime g, with lo and hi the `quantile` and 1 - `quantile` quantiles of its prompts' t: long = t >= hi, short =
    t <= lo. Whole grid horizons fall on one side, so a class holds about, not exactly, that share of the regime.
    Returns ({regime: (long mask, short mask, lo, hi)}, per-row class "long" / "short" / ""). A regime with fewer than
    two distinct horizons, or whose two quantiles coincide, gets no classes."""
    if not 0 < quantile < 0.5:
        raise ValueError(f"within_quantile must be between 0 and 0.5, not {quantile}")
    classes, label = {}, np.full(len(t), "", dtype=object)
    for g in sorted(set(regime)):
        m = regime == g
        lo, hi = np.quantile(t[m], [quantile, 1 - quantile])
        if not lo < hi:
            continue
        long_, short = m & (t >= hi), m & (t <= lo)
        classes[g] = (long_, short, float(lo), float(hi))
        label[long_], label[short] = "long", "short"
    return classes, label


def pooled_direction(parts: dict, sizes: dict) -> Optional[Direction]:
    """The unit vector along the average of the regimes' raw mean differences, weighted by regime size."""
    if not parts:
        return None
    total = sum(sizes[g] for g in parts)
    raw = sum(sizes[g] / total * d.vector * d.raw_norm for g, d in parts.items())
    norm = float(np.linalg.norm(raw))
    return Direction(raw / norm, norm, sum(d.n_plus for d in parts.values()), sum(d.n_minus for d in parts.values()))


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
    within_quantile: float = 1 / 3                 # within-regime CAA: the share of a regime's prompts in each class
    within_pool: tuple = ("near_only", "both")     # the regimes whose within-regime differences `caa-within` pools


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
    within: dict                # regime -> Direction of the within-regime CAA (absent if it could not be formed)
    within_pooled: Optional[Direction]   # caa-within: the pooled within-regime CAA, or None
    within_class: np.ndarray    # per row: "long", "short" or "", within the row's own regime
    trend: Direction            # horizon covariance pattern, regimes demeaned (steer/horizon_decoders.py)
    probe: Direction            # ridge decoder weights for the same target

    def direction(self, key: str) -> Direction:
        """The direction stored under `key` in directions.npz (steer.fitting.direction_keys)."""
        if key in ("caa", "reward", "choice", "trend", "probe"):
            return getattr(self, key)
        found = self.within_pooled if key == WITHIN_KEY else self.within.get(key.removeprefix("caa_"))
        if found is None:
            raise KeyError(f"this fit has no {key} direction (see fit.json, diagnostics.caa_within)")
        return found
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


def select(uids: list[str], records: dict, include_none: bool = True, exclude_horizons=(),
           allowed_uids: Optional[set] = None) -> Selection:
    """The prompts a fit uses, out of an activation set's `uids`. In order, a prompt is left out if it is:
      * not in the extraction split (the verification grid is never fitted on, whatever extract.py was run on);
      * a no-horizon (null control) prompt: it has no horizon to place on the curve;
      * in the "none" regime, when include_none is false;
      * at a stated horizon in `exclude_horizons` (grid values, e.g. "3 years"), held out for calibration;
      * not in `allowed_uids`, when that is given ("not_in_uids", counted only for prompts the rules above keep).
    The first applicable reason is counted."""
    exclude = set(exclude_horizons)
    regimes = fit_regimes(include_none)
    rows, excluded = [], {"not_extraction_split": 0, "no_horizon": 0, "none_regime": 0, "excluded_horizon": 0,
                        "not_in_uids": 0}
    for i, uid in enumerate(uids):
        r = records[uid]
        reason = ("not_extraction_split" if r.split != "extraction" else
                  "no_horizon" if r.horizon is None else
                  "none_regime" if r.regime not in regimes else
                  "excluded_horizon" if r.horizon_grid_value in exclude else
                  "not_in_uids" if allowed_uids is not None and uid not in allowed_uids else None)
        if reason:
            excluded[reason] += 1
        else:
            rows.append(i)
    if allowed_uids is None:
        del excluded["not_in_uids"]
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

    # Within-regime CAA: long minus short horizons among prompts of one regime, so the correct answer is the same
    # in both classes; and the pool of those differences over s.within_pool.
    classes, within_class = within_regime_classes(t, regime, s.within_quantile)
    within = {g: mean_difference(X, long_g, short_g) for g, (long_g, short_g, _, _) in classes.items()}
    sizes = {g: int((regime == g).sum()) for g in within}
    pooled = pooled_direction({g: within[g] for g in s.within_pool if g in within}, sizes)
    chose_far = chosen == "far"

    def composition(a: np.ndarray, b: np.ndarray) -> dict:
        return {"n_long": int(a.sum()), "n_short": int(b.sum()),
                "chose_far_share_long": float(chose_far[a].mean()), "chose_far_share_short": float(chose_far[b].mean())}

    def cosines(v: np.ndarray) -> dict:
        return {"cos_caa": float(v @ caa.vector), "cos_choice": float(v @ choice.vector),
                "cos_reward": float(v @ reward.vector)}

    within_diag = {
        "quantile": s.within_quantile, "pool": [g for g in s.within_pool if g in within],
        "not_formed": sorted(set(regime) - set(within)),
        "by_regime": {g: {"n_regime": sizes[g], **composition(classes[g][0], classes[g][1]),
                          "short_max_log10_days": classes[g][2], "long_min_log10_days": classes[g][3],
                          "raw_norm": within[g].raw_norm, **cosines(within[g].vector)} for g in within},
        "pooled": None if pooled is None else {"n_long": pooled.n_plus, "n_short": pooled.n_minus,
                                               "raw_norm": pooled.raw_norm, **cosines(pooled.vector)},
        "agreement": {f"{a}|{b}": float(within[a].vector @ within[b].vector)
                      for i, a in enumerate(sorted(within)) for b in sorted(within)[i + 1:]},
    }
    if random_sets is None:
        random_sets = {"isotropic": random_directions(X.shape[1], s.n_random, s.seed)}
    random = random_sets["isotropic"]
    norm = typical_norm(X)
    trend, probe, horizon_diag = horizon_decoders(X, t, regime)
    horizon_diag["classes"] = {"n_prompts": len(t), "by_regime": dict(Counter(regime))}
    for key, d in (("trend", trend), ("probe", probe)):
        horizon_diag[key].update(cosines(d.vector))

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
        "caa_within": within_diag,
        "horizon_decoders": horizon_diag,
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
                     reward_class=np.where(high, "high", np.where(low, "low", "")), fold=fold,
                     within=within, within_pooled=pooled, within_class=within_class, trend=trend, probe=probe)


def direction_keys(result: FitResult) -> list[str]:
    """The directions a fit has, as directions.npz names them: caa, reward, choice, trend, probe, then the within-regime CAAs."""
    return (["caa", "reward", "choice", "trend", "probe"] + [within_key(g) for g in FIT_REGIMES if g in result.within]
            + ([WITHIN_KEY] if result.within_pooled is not None else []))
