import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import delay_margin as dm  # noqa: E402


def test_none_reference_subtracts_the_none_mean_of_each_duration():
    values = np.array([1.0, 3.0, 10.0, 20.0, 22.0])
    durations = np.array(["a", "a", "a", "b", "b"])
    is_none = np.array([True, True, False, True, False])
    assert np.allclose(dm.none_reference(values, durations, is_none), [-1, 1, 8, 0, 2])


def test_binned_means_land_in_their_margin_bins():
    rng = np.random.default_rng(0)
    margin = rng.uniform(-2, 2, 2000)
    residual = np.where(margin < 0, 0.5, 0.0) + 0.01 * rng.normal(size=2000)   # a step where the delay enters
    out = dm.binned(residual, margin, np.arange(2000), rng)
    centers = [c for c, *_ in out]
    assert centers == [-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]
    assert all(abs(m - (0.5 if c < 0 else 0.0)) < 0.01 and lo <= m <= hi for c, m, lo, hi, n in out)
