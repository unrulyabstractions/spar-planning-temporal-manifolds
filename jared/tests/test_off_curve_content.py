
import numpy as np

from pipeline.s1_data_vectors.analysis.off_curve_content import center_within, shuffle_within, within_r


def test_center_within_removes_group_means():
    groups = np.array([0, 0, 1, 1, 1])
    values = np.array([1.0, 3.0, 10.0, 11.0, 15.0])
    centered = center_within(values, groups)
    assert np.allclose(centered, [-1, 1, -2, -1, 3])


def test_within_r_ignores_between_group_signal():
    # the target differs only between groups: nothing to find within them
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 4)
    target = groups.astype(float) + 0.0
    F = np.c_[groups + rng.normal(size=len(groups)) * 0.01, rng.normal(size=(len(groups), 2))]
    target = target + rng.normal(size=len(groups)) * 1e-9   # within-group variation is pure noise
    assert abs(within_r(F, target, groups)) < 0.3


def test_within_r_finds_within_group_signal_and_the_shuffle_null_does_not():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(30), 4)
    signal = rng.normal(size=len(groups))
    F = np.c_[signal + 0.1 * rng.normal(size=len(groups)), rng.normal(size=(len(groups), 2))]
    target = 5.0 * groups + signal
    assert within_r(F, target, groups) > 0.9
    shuffled = shuffle_within(target, groups, rng)
    assert abs(within_r(F, shuffled, groups)) < 0.4
