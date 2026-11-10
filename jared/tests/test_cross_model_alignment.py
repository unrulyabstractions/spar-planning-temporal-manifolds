"""Synthetic checks of analysis/cross_model_alignment.py: a rotated, rescaled, re-embedded copy of a
manifold aligns perfectly, independent noise does not."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from cross_model_alignment import (cka_null, first_aligned_coordinate, cross_model_readout,  # noqa: E402
                                   label_shuffle_null, linear_cka, remove_unit_means, s2_summary,
                                   shuffle_p_value, unit_lookup_r2)
from probe_dimensionality import UNIT_BY_LOG_H  # noqa: E402

H = 48


@pytest.fixture
def manifold():
    """48 horizon averages [H, 64] on a curved 3-D manifold, and a log horizon that is monotone along it."""
    rng = np.random.default_rng(0)
    log_h = np.linspace(-5, 2, H)
    t = (log_h - log_h.mean()) / log_h.std()
    latent = np.c_[t, 0.4 * t ** 2, 0.2 * np.sin(2 * t)]
    return latent @ rng.normal(size=(3, 64)) + 0.01 * rng.normal(size=(H, 64)), log_h


def rotated_copy(A, dim, scale=3.7, seed=1):
    """Isometric embedding into `dim` dimensions (orthonormal columns), then a rescale."""
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(dim, A.shape[1])))
    return scale * A @ Q.T


def test_rotation_and_rescale_give_cka_one_and_readout_one(manifold):
    A8, log_h = manifold
    A14 = rotated_copy(A8, 80)
    assert linear_cka(A8, A14) == pytest.approx(1.0, abs=1e-9)
    assert cross_model_readout(A14, A8, log_h) > 0.99
    rho8, rho14 = first_aligned_coordinate(A14, A8, log_h)
    assert abs(rho8) > 0.95 and abs(rho14) > 0.95


def test_independent_noise_falls_inside_the_null():
    rng = np.random.default_rng(2)
    A, B = rng.normal(size=(H, 64)), rng.normal(size=(H, 80))
    null = cka_null(A, B, shuffles=300)
    assert linear_cka(A, B) < np.percentile(null, 99)   # CKA of pure noise is ~0.6 at 48 rows: that is why the null exists


def test_a_shuffled_copy_is_not_aligned(manifold):
    A8, log_h = manifold
    A14 = rotated_copy(A8, 80)[np.random.default_rng(3).permutation(H)]
    null = cka_null(A8, rotated_copy(A8, 80), shuffles=300)
    assert linear_cka(A8, A14) < np.percentile(null, 99)
    assert cross_model_readout(A14, A8, log_h) < 0.5


def test_unit_means_removed_per_unit():
    horizons = np.array([-6.021983, -5.836346, 0.477121, 0.69897])   # two seconds, two years
    out = remove_unit_means(np.array([1.0, 3.0, 10.0, 20.0]), horizons)
    assert out == pytest.approx([-1.0, 1.0, -5.0, 5.0])


def test_s2_best_match_on_a_diagonal():
    n8, n14 = 6, 12
    C = np.exp(-((np.arange(1, n8 + 1)[:, None] * 2 - np.arange(1, n14 + 1)[None]) ** 2))
    s2 = s2_summary(C, n8, n14, anchor_handoff=None)
    assert s2["spearman"] == pytest.approx(1.0) and s2["mean_depth_gap"] < 1e-9 and s2["pass"]


def test_shared_prompt_noise_passes_the_row_null_but_not_the_label_shuffle_null():
    """Two models that share only prompt-level noise (no horizon signal): permuting horizon averages
    calls their CKA significant, shuffling horizon labels over the shared prompts does not."""
    rng = np.random.default_rng(4)
    log_h = np.repeat(np.array(sorted(UNIT_BY_LOG_H)), 6)
    shared = rng.normal(size=(len(log_h), 3))
    X8 = shared @ rng.normal(size=(3, 40)) + 0.1 * rng.normal(size=(len(log_h), 40))
    X14 = shared @ rng.normal(size=(3, 50)) + 0.1 * rng.normal(size=(len(log_h), 50))
    horizons = np.unique(log_h)
    A8, A14 = (np.stack([X[log_h == h].mean(0) for h in horizons]) for X in (X8, X14))
    observed = linear_cka(A8, A14)
    assert observed > np.percentile(cka_null(A8, A14, shuffles=300), 99)
    null = label_shuffle_null(X8, X14, log_h, within_unit=False, shuffles=300)
    assert shuffle_p_value(observed, null) > 0.05
    within = linear_cka(remove_unit_means(A8, horizons), remove_unit_means(A14, horizons))
    within_null = label_shuffle_null(X8, X14, log_h, within_unit=True, shuffles=300)
    assert shuffle_p_value(within, within_null) > 0.05


def test_unit_lookup_r2_is_high_across_units_and_negative_within():
    horizons = np.array(sorted(UNIT_BY_LOG_H))
    assert unit_lookup_r2(horizons) > 0.9
    assert unit_lookup_r2(horizons, within_unit=True) < 0
