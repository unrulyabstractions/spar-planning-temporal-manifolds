
import numpy as np
import pytest

from pipeline.s1_data_vectors.analysis import direction_stability as ds
from synthetic_pooled import synthetic_pooled


def pooled_directions(X, data):
    duration, factors, covariates = ds.fit_inputs(data)
    cell = ds.fit_effects(X, duration, factors, covariates, np.ones(len(X), bool))
    return cell, ds.steering_directions(cell, np.unique(duration))


def test_the_split_adds_up_and_the_invariant_part_leaves_the_factor_span():
    X, data, _ = synthetic_pooled()
    cell, directions = pooled_directions(X, data)
    assert np.allclose(directions["invariant"] + directions["in-span"], directions["effects"])
    assert np.abs(directions["invariant"] @ cell["factor_span"].T).max() < 1e-10
    assert np.allclose(np.linalg.norm(directions["U"], axis=1), np.linalg.norm(directions["effects"], axis=1))


def test_a_planted_horizon_direction_is_recovered_by_both_folds():
    X, data, planted = synthetic_pooled(latent_shift=0)
    fits, margins, _ = ds.cell_fits(X, data)
    cosines = ds.compare(fits["fold 0"], fits["fold 1"])
    assert cosines["effects"] > 0.95 and cosines["invariant"] > 0.95
    assert abs(ds.mean_cosine(fits["pooled"]["effects"], np.tile(planted["horizon"], (10, 1)))) > 0.95
    # the planted horizon is orthogonal to every factor effect
    assert ds.in_span_share(fits["pooled"]) < 0.05
    # U is the unit effect, oriented so that it does not point against the curve
    assert abs(fits["pooled"]["U"][0] @ planted["unit"]) / np.linalg.norm(fits["pooled"]["U"][0]) > 0.99
    assert all(m >= 0 for m in margins.values())


def test_the_in_span_share_recovers_a_planted_overlap_with_the_unit_effect():
    X, data, _ = synthetic_pooled(unit_share_of_horizon=0.5, latent_shift=0)
    cell, directions = pooled_directions(X, data)
    assert ds.in_span_share(directions) == pytest.approx(0.5, abs=0.05)
    shares = ds.factor_shares(cell, directions["effects"])
    assert shares["unit"] == pytest.approx(0.5, abs=0.05)
    assert shares["regime"] < 0.05 and shares["log near delay"] < 0.05


def test_horizon_variation_within_a_duration_leaks_into_the_span_of_small_factor_effects():
    """Why the in-span share can overstate the factors' real overlap with the horizon: the
    span counts every factor effect's direction whatever its size, and a small effect's
    estimate picks up chance correlation with within-duration horizon variation."""
    X, data, _ = synthetic_pooled(latent_shift=0.3)
    _, directions = pooled_directions(X, data)
    assert ds.in_span_share(directions) > 0.15


def test_split_half_projection():
    assert ds.split_half_projection(1.0) == 1.0
    assert ds.split_half_projection(0.0) == 0.0
    assert ds.split_half_projection(0.5) == pytest.approx(2 / 3)
