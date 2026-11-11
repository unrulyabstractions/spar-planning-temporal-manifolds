import numpy as np

import pipeline.s3_analysis.by_label.confound_sweep as cs


def planted(seed=0, n_durations=12, per=20, d=24):
    """Activations = horizon axis * log h + delay direction * log delay + noise, with the
    delay drawn independently of the horizon, and a feature that is constant within each
    duration (it must be dropped)."""
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.normal(size=(d, 3)))[0].T
    y = np.repeat(np.linspace(-3, 2, n_durations), per)
    durations = np.repeat([f"d{i}" for i in range(n_durations)], per)
    delay = rng.uniform(-1, 1, len(y))
    X = np.outer(y, q[0]) + np.outer(delay, 0.5 * q[0] + q[1]) + 0.05 * rng.normal(size=(len(y), d))
    V = np.column_stack([delay, y])          # the second column only varies across durations
    return X, y, durations, V, q


def test_within_removes_group_means():
    v = np.array([1.0, 3.0, 10.0, 14.0])
    assert np.allclose(cs.within(v, np.array(["a", "a", "b", "b"])), [-1, 1, -2, 2])


def test_fe_regression_recovers_a_planted_slope_and_drops_constant_features():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 10)
    x = rng.normal(size=300)
    r = 0.4 * x + groups + 0.05 * rng.normal(size=300)    # group offsets must not matter
    out = cs.fe_regression(r, np.column_stack([x, groups]), ["x", "group"], groups, np.arange(300), rng)
    assert list(out) == ["x"]
    c, lo, hi = out["x"]
    assert abs(c - 0.4) < 0.02 and lo < 0.4 < hi


def test_axis_effects_measure_the_feature_along_the_horizon_axis():
    X, y, durations, V, q = planted()
    axis = cs.horizon_axis(X, y, durations)
    assert np.allclose(axis, q[0], atol=0.05)
    out = cs.axis_effects(X, V, ["delay", "log h"], durations, axis)
    along, cosine = out["delay"]
    assert abs(along - 0.5) < 0.05                       # half a decade of horizon per decade of delay
    assert abs(cosine - 0.5 / np.hypot(0.5, 1)) < 0.05
    assert "log h" not in out


def test_crossfit_readout_carries_a_delay_only_when_it_has_no_direction_of_its_own():
    rng = np.random.default_rng(1)
    X, y, durations, V, q = planted()
    folds = np.arange(len(y)) % 2
    readout = cs.crossfit_readout(X, y, durations, folds, k=4)
    assert np.corrcoef(readout, y)[0, 1] > 0.95
    assert 0.8 < cs.calibration(readout, y, durations) < 1.2
    # the delay also moves its own direction q1, which the duration averages span (their
    # mean delays differ), so the ridge learns to subtract it: the readout stays clean
    assert abs(np.corrcoef(cs.within(readout, durations), V[:, 0])[0, 1]) < 0.2
    # purely along the horizon axis, it cannot be told apart, and the readout moves with it
    delay = V[:, 0]
    X_axis = np.outer(y + 0.5 * delay, q[0]) + 0.05 * rng.normal(size=X.shape)
    readout = cs.crossfit_readout(X_axis, y, durations, folds, k=4)
    assert np.corrcoef(cs.within(readout, durations), delay)[0, 1] > 0.9


def test_base_folds_are_stable_and_split_bases():
    bases = np.array([f"p{i}" for i in range(200)])
    folds = cs.base_folds(bases)
    assert list(folds[:6]) == list(cs.base_folds(bases[:6]))     # a pure function of the base id
    assert 80 < folds.sum() < 120 and set(folds) == {0, 1}
