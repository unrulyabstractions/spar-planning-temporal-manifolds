import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from delay_position import parametrizations, within_r2  # noqa: E402


def test_a_positional_pull_fits_position_better_than_role():
    rng = np.random.default_rng(0)
    n = 600
    groups = np.repeat(np.arange(30), 20)
    near, far = rng.uniform(-2, 0, n), rng.uniform(0, 2, n)
    near_first = rng.random(n) < 0.5
    second = np.where(near_first, far, near)
    readout = -0.3 * second + groups + 0.05 * rng.normal(size=n)     # pulled by the delay listed second
    p = parametrizations(near, far, near_first)
    r2 = {name: within_r2(readout, V, groups) for name, (V, _) in p.items()}
    assert r2["position"] > 0.95 > r2["role"]


def test_parametrizations_swap_delays_by_order():
    V, names = parametrizations(np.array([1.0, 1.0]), np.array([5.0, 5.0]), np.array([True, False]))["position"]
    assert names == ["first", "second", "order"]
    assert V.tolist() == [[1.0, 5.0, 1.0], [5.0, 1.0, 0.0]]
