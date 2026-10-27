"""Projection and null logic of analysis/regime_comparison.py on synthetic prompts.

The horizon and the far delay are written into strong directions (8 and 5 cosines,
the sizes of the two removed subspaces), so the projection can remove them. The
comparison variable is a weaker 9th direction."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import regime_comparison as rc  # noqa: E402

D = 100
HORIZON_AMPLITUDE = 8


def synthetic_prompts(comparison_amplitude, seed=0, n=240):
    rng = np.random.default_rng(seed)
    horizons, fars = np.linspace(-1, 2, 24), np.linspace(0.3, 1.7, 6)
    log_h, log_far = rng.choice(horizons, n), rng.choice(fars, n)
    regime_both = log_h > log_far
    cosines = lambda t, count: HORIZON_AMPLITUDE * np.cos(np.outer((t - t.min()) / np.ptp(t) * np.pi, np.arange(1, count + 1)))
    X = rng.normal(size=(n, D))
    X[:, :8] += cosines(log_h, 8)
    X[:, 8:13] += cosines(log_far, 5)
    X[:, 13] += comparison_amplitude * np.where(regime_both, 1.0, -1.0)
    return X, log_h, log_far, regime_both, rng.random(n) < 0.5


@pytest.fixture(autouse=True)
def few_shuffles(monkeypatch):
    monkeypatch.setattr(rc, "SHUFFLES", 100)
    monkeypatch.setattr(rc, "MAX_SHUFFLES", 100)


def test_a_planted_comparison_survives_the_projection():
    cell = rc.evaluate_cell(*synthetic_prompts(comparison_amplitude=6), seed=0)
    assert cell["auc"] > cell["null_p95"] + 0.1   # attenuated (about 0.75): the comparison is correlated with h and far
    assert cell["p_value"] < 0.05


def test_a_regime_that_is_only_horizon_and_delay_is_removed():
    X, log_h, log_far, regime_both, chose_near = synthetic_prompts(comparison_amplitude=0.0)
    cell = rc.evaluate_cell(X, log_h, log_far, regime_both, chose_near, seed=0)
    assert cell["auc"] <= cell["null_p95"]
    assert cell["reference_auc"] > cell["auc"] + 0.1   # the unprojected probe does read it from h and far


def test_the_projection_removes_the_horizon_and_delay_directions():
    X, log_h, log_far, regime_both, _ = synthetic_prompts(comparison_amplitude=6)
    train, test, F_train, F_test = rc.projected_folds(X, log_h, log_far)[0]
    share_in_strong_dims = lambda M: (M[:, :13] ** 2).sum() / (M ** 2).sum()   # F is rescaled, so compare shares
    for F, rows in ((F_train, train), (F_test, test)):
        assert share_in_strong_dims(F) < 0.2 * share_in_strong_dims(X[rows] - X[train].mean(0))
    assert np.corrcoef(F_train[:, 13], regime_both[train])[0, 1] > 0.2


def test_shuffles_stay_within_far_delay():
    rng = np.random.default_rng(0)
    far = np.repeat([0, 1, 2], [10, 20, 30])
    labels = rng.random(60) < 0.4
    shuffled = rc.shuffle_within(labels, far, rng)
    assert not np.array_equal(shuffled, labels)
    for group in (0, 1, 2):
        assert shuffled[far == group].sum() == labels[far == group].sum()


def test_irrational_prompts_are_those_whose_choice_contradicts_the_regime():
    regime_both = np.array([True, True, False, False])
    chose_near = np.array([True, False, False, True])   # row 0: both, chose near; row 3: near regime, chose far? no: chose near
    scores = np.array([-1.0, 1.0, -1.0, -1.0])
    counts = rc.follow_counts(scores, regime_both, chose_near)
    assert counts["irrational"]["n"] == 2 and counts["rational"]["n"] == 2
    assert counts["irrational"]["follows_regime"] == 1   # row 3 is called near, as its regime


def test_within_horizon_auc_ignores_what_only_separates_horizons():
    log_h = np.repeat([0.0, 1.0], 6)
    labels = np.array([0, 0, 0, 1, 1, 1] * 2, bool)
    scores_by_horizon = log_h * 10 + np.random.default_rng(0).normal(size=12)   # ranks horizons, not regimes
    assert rc.within_horizon_auc(scores_by_horizon, labels, log_h) < 0.9
    assert rc.within_horizon_auc(np.where(labels, 1.0, -1.0), labels, log_h) == 1.0
    assert np.isnan(rc.within_horizon_auc(scores_by_horizon, np.zeros(12, bool), log_h))


def test_a_planted_comparison_beats_the_fixed_far_threshold_inside_a_horizon():
    X, log_h, log_far, regime_both, chose_near = synthetic_prompts(comparison_amplitude=6)
    control = rc.far_threshold_control(X, log_h, log_far, regime_both, chose_near, seed=0)
    assert control["regime_auc"] > control["far_threshold_auc"] + 0.1


def test_shift_labels_at_zero_are_the_regime_ties_included():
    log_h = np.array([0.0, 1.0, 1.0, 2.0])
    log_far = np.array([1.0, 1.0, 1.0, 1.0])   # the middle two are ties, regime "both" in the bank
    assert rc.shift_labels(log_h, log_far, 0.0).tolist() == [False, True, True, True]
    assert rc.shift_labels(log_h, log_far, 0.5).tolist() == [False, False, False, True]
