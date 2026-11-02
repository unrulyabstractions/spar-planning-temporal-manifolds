import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import crossing_split as cs  # noqa: E402

BEHAVIOUR = {"none": -0.6, "near": 13.0, "both": -0.6}


def prompts(rng, n=200):
    """Near-regime prompts: horizon 1 year, near delay 0.1 years, far delay 10^U(0.2, 2.8)
    years, so a +1 or +2 decade step crosses the far delay for some and not others."""
    far = 10 ** rng.uniform(0.2, 2.8, n)
    return [{"horizon": 1.0, "near_delay_years": 0.1, "far_delay_years": f, "regime": "near"} for f in far]


def test_target_regime_is_clamped_to_the_curve():
    rows = [{"horizon": 100.0, "near_delay_years": 1.0, "far_delay_years": 500.0, "regime": "near"}]
    grid = np.linspace(-6, 2.5, 48)   # up to ~316 years: +2 decades stops short of the far delay
    assert cs.target_regimes(rows, 2, grid)[0] == "near"
    assert cs.target_regimes(rows, 2, np.linspace(-6, 3, 48))[0] == "both"


def test_strata_skip_mixed_targets_and_small_groups_and_take_the_sign_from_behaviour():
    regimes = np.array(["both"] * 30 + ["near"] * 30)
    targets = {-2: np.array(["near"] * 10 + ["none"] * 10 + ["both"] * 10 + ["none"] * 5 + ["near"] * 25),
               1: np.array(["both"] * 30 + ["both"] * 20 + ["near"] * 10)}
    used, skipped = cs.strata(regimes, targets, BEHAVIOUR)
    names = {s["name"]: s for s in used}
    # both@-2 reaches near and none; near@-2 has only 5 crossers; both@+1 has no crossers
    assert set(names) == {"near@+1"}
    assert len(skipped) == 3
    # near -> both: the behaviour map falls from 13 to -0.6, so the predicted sign is -1
    assert names["near@+1"]["sign"] == -1 and names["near@+1"]["target"] == "both"


def test_a_planted_crossing_gap_is_recovered_and_a_decision_push_shows_none():
    rng = np.random.default_rng(0)
    rows = prompts(rng)
    regimes = np.array([r["regime"] for r in rows])
    grid = np.linspace(-6, 3, 48)
    targets = {a: cs.target_regimes(rows, a, grid) for a in (1, 2)}
    used, _ = cs.strata(regimes, targets, BEHAVIOUR)
    assert {s["name"] for s in used} == {"near@+1", "near@+2"}
    noise = {a: 0.02 * rng.normal(size=len(rows)) for a in (1, 2)}
    # horizon edit: crossers move 0.2 C toward far (negative near effect), stayers do not
    horizon = {a: -0.2 * (targets[a] == "both") + noise[a] for a in (1, 2)}
    # decision push: every prompt moves 0.2 C toward far
    push = {a: -0.2 + noise[a] for a in (1, 2)}
    point, boot = cs.gaps({"horizon": horizon, "push": push}, used, n_boot=500)
    assert abs(point["horizon"][2] - 0.2) < 0.01 and cs.interval(boot["horizon"])[0] > 0.15
    assert abs(point["push"][2]) < 0.01
    low, high = cs.interval(boot["push"])
    assert low < 0 < high
    # paired: both directions share the noise, so it cancels and every resample gives 0.2
    low, high = cs.interval(boot["horizon"] - boot["push"])
    assert np.isclose(low, 0.2) and np.isclose(high, 0.2)


def test_crossing_regression_separates_crossing_from_step_length():
    rng = np.random.default_rng(1)
    rows = prompts(rng)
    regimes = np.array([r["regime"] for r in rows])
    targets = {a: cs.target_regimes(rows, a, np.linspace(-6, 3, 48)) for a in (1, 2)}
    used, _ = cs.strata(regimes, targets, BEHAVIOUR)
    step_length = {a: rng.uniform(1, 3, len(rows)) for a in (1, 2)}
    # the effect follows step length only: crossing should get a coefficient near 0
    effects = {a: -0.05 * step_length[a] + 0.01 * rng.normal(size=len(rows)) for a in (1, 2)}
    base = rng.normal(size=len(rows))
    coef, low, high = cs.crossing_regression(effects, used, base, step_length, np.zeros(len(rows), bool), n_boot=300)
    assert abs(coef) < 0.01 and low < 0 < high


def test_none_minus_near_compares_the_two_targets_of_both_regime_prompts():
    regimes = np.array(["both"] * 6 + ["near"] * 2)
    target = np.array(["none"] * 3 + ["near"] * 3 + ["none"] * 2)
    effects = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 9.0, 9.0])   # near-regime rows must be ignored
    out = cs.none_minus_near(effects, regimes, target, n_boot=200)
    assert out["n_none"] == 3 and out["n_near"] == 3
    assert np.isclose(out["difference"][0], -0.2)
