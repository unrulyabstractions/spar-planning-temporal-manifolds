"""Random-direction nulls and the comparison statistic for the evaluation pipeline (design.md sections 2-4).

Numpy reference implementation, written to match the extraction team's `steering/steer/directions.py` conventions:
directions are unit vectors (rows of an [n, d] array), edits scale them by `strength * typical_norm`, and every
function takes masks/labels rather than prompt records.

Null ladder:
  isotropic(d, n, seed)                       N1  v ~ N(0, I)
  covariance(X, n, seed)                      N2  v ~ N(0, Sigma_X)
  real_differences(X, n, seed)                N2' v = h_i - h_j, random pairs
  in_span(U, n, seed)                         N3  random unit vector in span(U)
  shuffled_mean_difference(X, y, ..., strata) N4  the CAA recipe with labels permuted within strata
  norm_profile_twin(delta, r)                 matching for per-token edits (manifold slide): ||delta_t|| * r

Statistic:
  odd_even(effect_plus, effect_minus)         odd = (E(+k) - E(-k)) / 2, even = (E(+k) + E(-k)) / 2
  rank_p(stat, null)                          (1 + #{|null| >= stat}) / (K + 1)
"""

from __future__ import annotations

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def isotropic(d: int, n: int, seed: int) -> np.ndarray:
    """N1: `n` isotropic random unit vectors in R^d."""
    return unit(np.random.default_rng(seed).standard_normal((n, d)))


def covariance(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    """N2: unit vectors along draws from N(0, Sigma), Sigma = covariance of the rows of X (activations at one layer and
    site). Drawn as g @ Xc / sqrt(m) with g ~ N(0, I_m), so Sigma is never formed (d can be 5,120)."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0)
    g = np.random.default_rng(seed).standard_normal((n, len(X)))
    return unit(g @ Xc / np.sqrt(len(X)))


def real_differences(X: np.ndarray, n: int, seed: int, groups: np.ndarray | None = None) -> np.ndarray:
    """N2': unit vectors along h_i - h_j for random pairs of rows. With `groups` (e.g. horizon labels), both rows of a
    pair come from the same group, so the difference carries no label information ("within-label" real difference)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    if groups is None:
        i, j = rng.integers(0, len(X), n), rng.integers(0, len(X), n)
        same = i == j
        while same.any():
            j[same] = rng.integers(0, len(X), same.sum())
            same = i == j
        return unit(X[i] - X[j])
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    idx = [ix for ix in idx if len(ix) >= 2]
    out = []
    for _ in range(n):
        ix = idx[rng.integers(len(idx))]
        a, b = rng.choice(ix, 2, replace=False)
        out.append(X[a] - X[b])
    return unit(np.stack(out))


def in_span(U: np.ndarray, n: int, seed: int) -> np.ndarray:
    """N3: random unit vectors in the row span of U ([k, d], orthonormal rows, e.g. the manifold's PCA basis)."""
    U = np.asarray(U, dtype=np.float64)
    return unit(np.random.default_rng(seed).standard_normal((n, U.shape[0])) @ U)


def shuffled_mean_difference(X: np.ndarray, y: np.ndarray, plus, minus, n: int, seed: int,
                             strata: np.ndarray | None = None) -> np.ndarray:
    """N4: the mean-difference recipe re-run with labels `y` permuted within `strata`.

    `plus(y)` / `minus(y)` turn a label vector into the two class masks (e.g. lambda y: y >= 4 for H/D >= 4), so the
    shuffled direction uses exactly the method's thresholds. Strata keep the composition the permutation must not
    break: e.g. (config, option order), or regime, which makes the null test "horizon beyond regime"."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = [np.arange(len(y))] if strata is None else [np.flatnonzero(strata == s) for s in np.unique(strata)]
    out = []
    for _ in range(n):
        yp = y.copy()
        for ix in groups:
            yp[ix] = y[rng.permutation(ix)]
        p, m = plus(yp), minus(yp)
        if p.sum() == 0 or m.sum() == 0:
            raise ValueError("a shuffled class is empty; widen the strata")
        out.append(X[p].mean(0) - X[m].mean(0))
    return unit(np.stack(out))


def norm_profile_twin(delta: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Random twin of a per-token edit: the same norm at every token, direction r. delta [..., T, d], r [d] unit."""
    return np.linalg.norm(delta, axis=-1, keepdims=True) * unit(r)


def odd_even(effect_plus, effect_minus) -> tuple[np.ndarray, np.ndarray]:
    """Directional (odd) and disruption (even) parts of a +/- pair of mean effects."""
    p, m = np.asarray(effect_plus, dtype=float), np.asarray(effect_minus, dtype=float)
    return (p - m) / 2, (p + m) / 2


def rank_p(stat: float, null: np.ndarray) -> float:
    """Exact rank p-value of a method's odd part against K random directions' |odd parts| (sign-free null)."""
    null = np.abs(np.asarray(null, dtype=float))
    return float((1 + np.sum(null >= stat)) / (len(null) + 1))


def span_share(V: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Fraction of each row's squared norm that lies in span(U): how much of a direction can touch the horizon
    subspace at all (isotropic ~ k/d)."""
    return np.linalg.norm(np.asarray(V) @ np.asarray(U).T, axis=-1) ** 2 / np.linalg.norm(V, axis=-1) ** 2
