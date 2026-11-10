"""Mean-difference directions, controls and the linear edit, on synthetic activations with a known direction."""

import numpy as np
import pytest

from steer.directions import (mean_difference, random_directions, steer_linear, threshold_masks, typical_norm)
from steer.stats import grouped_folds, ranks, spearman


def synthetic(n=400, d=64, seed=0):
    """Activations = 3 * log10(H/D) along a known unit vector u, plus isotropic noise."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    log_hd = rng.uniform(-3, 3, n)
    X = (3 * log_hd[:, None] * u + rng.standard_normal((n, d))).astype(np.float32)
    return X, 10 ** log_hd, u


def test_caa_recovers_the_planted_direction():
    X, hd, u = synthetic()
    long_, short = threshold_masks(hd, 4, 0.25)
    caa = mean_difference(X, long_, short)
    cos = float(caa.vector @ u)
    print(f"CAA vs planted direction: cos {cos:.4f} ({caa.n_plus} long, {caa.n_minus} short)")
    assert cos > 0.98
    assert abs(np.linalg.norm(caa.vector) - 1) < 1e-12


def test_masks_and_empty_or_overlapping_classes():
    long_, short = threshold_masks(np.array([0.1, 0.25, 1, 4, 9]), 4, 0.25)
    assert long_.tolist() == [False, False, False, True, True]
    assert short.tolist() == [True, True, False, False, False]          # ties count in the class (<= and >=)
    with pytest.raises(ValueError):
        threshold_masks(np.array([1.0]), 1, 2)
    X = np.zeros((3, 2))
    with pytest.raises(ValueError):
        mean_difference(X, np.array([True, False, False]), np.array([False, False, False]))
    with pytest.raises(ValueError):
        mean_difference(X, np.array([True, True, False]), np.array([True, False, False]))


def test_random_controls_are_unit_seeded_and_spread():
    R = random_directions(256, 50, seed=3)
    assert np.allclose(np.linalg.norm(R, axis=1), 1)
    assert np.array_equal(R, random_directions(256, 50, seed=3))
    max_cos = np.abs(R @ R.T - np.eye(50)).max()
    print(f"largest |cos| between two of 50 random directions in R^256: {max_cos:.3f} (1/sqrt(d) = {1 / 16:.3f})")
    assert max_cos < 0.35


def test_linear_edit_moves_by_strength_times_typical_norm():
    X, _, u = synthetic(n=10)
    norm = typical_norm(X)
    moved = steer_linear(X, u, 0.25, norm)
    shift = np.linalg.norm(moved - X, axis=1)
    print(f"typical norm {norm:.3f}; edit at strength 0.25 moves every row by {shift[0]:.4f}")
    assert np.allclose(shift, 0.25 * norm)
    assert np.array_equal(steer_linear(X, u, 0.0, norm), X)


def test_stats_helpers():
    assert ranks(np.array([10, 20, 20, 5])).tolist() == [2, 3.5, 3.5, 1]
    assert spearman([1, 2, 3, 4], [10, 30, 20, 40]) == pytest.approx(0.8)
    groups = ["g1", "g1", "g2", "g3", "g3", "g4", "g5"]
    folds = grouped_folds(groups, 3, seed=0)
    assert sum(f.astype(int) for f in folds).tolist() == [1] * 7                 # a partition
    for f in folds:                                                              # groups never split
        assert len({g for g, x in zip(groups, f) if x} & {g for g, x in zip(groups, f) if not x}) == 0
