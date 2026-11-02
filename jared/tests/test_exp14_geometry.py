import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import exp14_geometry as eg  # noqa: E402
from spar_horizon.steering import curve_delta, fit_duration_effects  # noqa: E402


def planted_states(seed=0, n=1200, d=24, noise=0.01):
    """States = duration effect + unit effect + regime effect + a delay slope + noise, with every unit
    rendered at every duration, so the fit can separate them."""
    rng = np.random.default_rng(seed)
    log_h = rng.choice(np.linspace(-2, 2, 6), n)
    unit = rng.choice(np.array(["months", "years"]), n)
    regime = rng.choice(np.array(["both", "near", "none"]), n)
    log_far = rng.normal(size=n)
    duration_effect = {h: rng.normal(size=d) for h in np.unique(log_h)}
    unit_effect = {"months": np.zeros(d), "years": 3 * rng.normal(size=d)}
    regime_effect = {r: rng.normal(size=d) for r in ("both", "near", "none")}
    slope = rng.normal(size=d)
    X = (np.array([duration_effect[h] for h in log_h]) + np.array([unit_effect[u] for u in unit])
         + np.array([regime_effect[r] for r in regime]) + np.outer(log_far, slope) + noise * rng.normal(size=(n, d)))
    return X, log_h, {"unit": unit, "regime": regime}, {"log far delay": log_far}


def test_decompose_scores_the_step_and_the_step_plus_unit():
    step, unit_change = np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])
    result = eg.decompose(step + unit_change, step, unit_change, np.zeros((0, 3)))
    assert np.isclose(result["r2_step_unit"], 1.0)
    assert np.isclose(result["r2_step"], 0.2)          # 1 - 4/5
    assert np.isclose(result["unit_share"], 0.8)
    assert np.isclose(result["cos_step"], 1 / np.sqrt(5))
    assert np.isclose(result["cos_step_unit"], 0.0) and np.isclose(result["invariant_norm_share"], 1.0)


def test_the_invariant_part_drops_the_steps_component_along_the_factor_span():
    # the step leans against the unit change; the factor span is that axis, so the invariant part is orthogonal to it
    step, unit_change = np.array([1.0, -1.0, 0.0]), np.array([0.0, 2.0, 0.0])
    result = eg.decompose(np.array([1.0, 1.0, 0.0]), step, unit_change, np.array([[0.0, 1.0, 0.0]]))
    assert np.isclose(result["cos_step_unit"], -1 / np.sqrt(2))
    assert np.isclose(result["cos_invariant_unit"], 0.0)
    assert np.isclose(result["invariant_norm_share"], 0.5)


def test_a_text_change_with_a_new_unit_is_the_duration_step_plus_the_unit_change():
    X, log_h, factors, covariates = planted_states()
    cell = fit_duration_effects(X, log_h, factors, orthogonal=False, covariates=covariates)
    adjusted = eg.residualized(X, cell, factors, covariates)
    own = (np.isclose(log_h, 2.0)) & (factors["unit"] == "years")
    donor = (np.isclose(log_h, -1.2)) & (factors["unit"] == "months")
    change = adjusted[donor].mean(0) - adjusted[own].mean(0)
    step = curve_delta(cell, np.array([2.0]), np.array([-3.2]))[0]
    effects = eg.level_effects(cell, "unit")
    result = eg.decompose(change, step, effects["months"] - effects["years"], cell["factor_span"])
    assert result["r2_step_unit"] > 0.99
    assert result["r2_step"] < 0.5          # the unit is most of this change
    # within one unit the duration step is the whole change
    same_unit = (np.isclose(log_h, -1.2)) & (factors["unit"] == "years")
    within = adjusted[same_unit].mean(0) - adjusted[own].mean(0)
    assert eg.decompose(within, step, np.zeros_like(step), cell["factor_span"])["r2_step"] > 0.99


def test_residualized_keeps_the_unit_and_removes_regime_and_delays():
    X, log_h, factors, covariates = planted_states(noise=0.0)
    cell = fit_duration_effects(X, log_h, factors, orthogonal=False, covariates=covariates)
    adjusted = eg.residualized(X, cell, factors, covariates)
    same_text = np.isclose(log_h, 0.4) & (factors["unit"] == "years")
    assert np.allclose(adjusted[same_text], adjusted[same_text][0], atol=1e-6)   # regime and delay gone
    other_unit = np.isclose(log_h, 0.4) & (factors["unit"] == "months")
    assert not np.allclose(adjusted[other_unit][0], adjusted[same_text][0], atol=1e-3)   # unit kept


def row(horizon_text, target_regime, near=1.0, far=10.0):
    return {"horizon_text": horizon_text, "target_regime": target_regime,
            "near_delay_years": near, "far_delay_years": far}


def test_reachable_in_own_unit():
    assert eg.reachable_in_own_unit(row("39 years", "near"))                    # 1-9 years
    assert not eg.reachable_in_own_unit(row("39 years", "none"))                # needs < 1 year
    assert eg.reachable_in_own_unit(row("5 months", "none"))                    # 1-11 months
    assert not eg.reachable_in_own_unit(row("30 seconds", "near", near=0.5, far=5.0))   # 999 s is 17 minutes
    assert eg.reachable_in_own_unit(row("30 seconds", "none", near=0.5, far=5.0))


def test_thirds_split_by_size_and_average_the_value():
    size = np.arange(9, dtype=float)
    value = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1], float)
    bins = eg.thirds(size, value)
    assert [b[2] for b in bins] == [3, 3, 3]
    assert np.allclose([b[3] for b in bins], [0, 1 / 3, 1])
