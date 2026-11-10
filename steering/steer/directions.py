"""Mean-difference directions, random control directions, and the linear edit.

Report, Appendix A.3: "CAA: mean of long-horizon activations (H/D >= 4) minus mean of short-horizon ones
(H/D <= 1/4), normalized. Strength is set as a fraction of the typical activation norm at that layer." D is the far
option's delay (Appendix A.1). Methods: "Controls are random directions and a reward-contrast direction, both
matched in norm."

Every direction here is stored as a unit vector, and every edit scales it by `strength * typical_norm`, so all
directions applied at the same strength are matched in norm by construction.

The functions take boolean masks rather than prompt records, so that *which* prompts form each class (the
thresholds, which regimes are included) stays a decision of the caller (fit.py), recorded in its output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import unit


@dataclass
class Direction:
    vector: np.ndarray    # unit vector, float64 [d]
    raw_norm: float       # norm of the difference of means before normalizing (in activation units)
    n_plus: int           # prompts in the class the direction points towards
    n_minus: int          # prompts in the class it points away from


def mean_difference(X: np.ndarray, plus: np.ndarray, minus: np.ndarray) -> Direction:
    """Unit vector along mean(X[plus]) - mean(X[minus]). Means are taken in float64."""
    if plus.sum() == 0 or minus.sum() == 0:
        raise ValueError(f"empty class: {plus.sum()} prompts in plus, {minus.sum()} in minus")
    if (plus & minus).any():
        raise ValueError("a prompt is in both classes")
    diff = X[plus].mean(axis=0, dtype=np.float64) - X[minus].mean(axis=0, dtype=np.float64)
    return Direction(unit(diff), float(np.linalg.norm(diff)), int(plus.sum()), int(minus.sum()))


def threshold_masks(values: np.ndarray, high_min: float, low_max: float) -> tuple[np.ndarray, np.ndarray]:
    """(values >= high_min, values <= low_max). Used for H/D (4 and 1/4) and for the reward multiplier."""
    if not low_max < high_min:
        raise ValueError(f"the two classes overlap: low_max {low_max} >= high_min {high_min}")
    values = np.asarray(values, dtype=float)
    return values >= high_min, values <= low_max


def random_directions(d: int, n: int, seed: int) -> np.ndarray:
    """`n` random unit vectors in R^d (isotropic: normalized standard Gaussians), as rows of an [n, d] array."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def typical_norm(X: np.ndarray) -> float:
    """The typical activation norm at a layer: the mean L2 norm of its activations."""
    return float(np.linalg.norm(X.astype(np.float64), axis=1).mean())


def steer_linear(h: np.ndarray, direction: np.ndarray, strength: float, norm: float) -> np.ndarray:
    """The linear edit: h + strength * norm * direction, for any leading shape of h (last axis is d).

    `strength` is a fraction of the typical activation norm `norm` (see typical_norm); its sign sets the sense.
    This is the numpy reference; an edit inside the model applies the same formula to a torch tensor
    (see steer.layers.edit).
    """
    return h + strength * norm * direction
