import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import behaviour_step as bs  # noqa: E402


def planted(rng, jump, slope, durations=30, per_duration=20):
    """dataset()-shaped rows whose near log-odds jump by `jump` where the far delay falls
    inside the horizon and slope by `slope` per decade of far margin; the near delay is
    always inside."""
    y = np.repeat(np.linspace(-3, 2, durations), per_duration)
    far_margin = rng.uniform(-2, 2, len(y))
    near_margin = -rng.uniform(0.5, 2, len(y))
    features = np.zeros((len(y), 10))
    features[:, 0], features[:, 1] = y + near_margin, y + far_margin
    features[:, 2:6] = rng.normal(size=(len(y), 4))
    lo = jump * (far_margin < 0) + slope * far_margin + 0.5 * rng.normal(size=len(y))
    data = {"y": y, "features": features, "durations": np.repeat(np.arange(durations), per_duration).astype(str),
            "pairs": np.arange(len(y)).astype(str)}
    return lo, data


def test_a_planted_jump_and_a_planted_slope_are_told_apart():
    rng = np.random.default_rng(0)
    lo, data = planted(rng, jump=-10.0, slope=0.0)
    hinge = bs.fit_subset(lo, data, np.ones(len(lo), bool), rng)["fits"]["hinge"]
    assert abs(hinge["far step"][0] + 10) < 0.5 and hinge["far step"][2] < 0
    assert abs(hinge["far slope"][0]) < 0.5
    lo, data = planted(rng, jump=0.0, slope=3.0)
    hinge = bs.fit_subset(lo, data, np.ones(len(lo), bool), rng)["fits"]["hinge"]
    assert abs(hinge["far step"][0]) < 0.5 and hinge["far step"][1] < 0 < hinge["far step"][2]
    assert abs(hinge["far slope"][0] - 3) < 0.3


def test_far_bins_show_the_jump_at_zero():
    rng = np.random.default_rng(1)
    lo, data = planted(rng, jump=-10.0, slope=0.0)
    bins = bs.fit_subset(lo, data, np.ones(len(lo), bool), rng)["bins"]["far"]
    inside = [b[1] for b in bins if b[0] < 0]
    outside = [b[1] for b in bins if b[0] > 0]
    assert np.mean(outside) - np.mean(inside) > 9
