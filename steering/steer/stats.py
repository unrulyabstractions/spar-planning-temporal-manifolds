"""Small statistics helpers, in numpy only (no scipy dependency)."""

from __future__ import annotations

import numpy as np


def ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n, ties given the mean of the ranks they span."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):           # average over ties
        tied = x == v
        if tied.sum() > 1:
            r[tied] = r[tied].mean()
    return r


def spearman(x, y) -> float:
    """Spearman rank correlation (Pearson correlation of the ranks)."""
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def grouped_folds(groups: list[str], k: int, seed: int) -> list[np.ndarray]:
    """`k` boolean test masks that partition the items, never splitting a group across folds.

    Groups are shuffled with `seed` and dealt round-robin, so fold sizes differ by at most one group.
    """
    unique = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    fold_of_group = {g: i % k for i, g in enumerate(unique)}
    fold = np.array([fold_of_group[g] for g in groups])
    return [fold == i for i in range(k)]


def unit(v: np.ndarray) -> np.ndarray:
    """`v` scaled to unit L2 norm."""
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("cannot normalize a zero vector")
    return v / n
