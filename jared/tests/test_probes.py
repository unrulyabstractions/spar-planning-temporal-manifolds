"""E7 probes on planted signals: a flat direction, a curved manifold, and the control."""

import numpy as np

from spar_horizon import probes as P


def horizons(n_groups=16, per_group=20):
    return np.repeat(np.linspace(-6, 2.7, n_groups), per_group)


def embed(signal, d=200, noise=0.1, seed=0):
    """Place the columns of `signal` [n, m] on random orthonormal directions in d dims, plus noise."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, signal.shape[1])))
    return signal @ q.T * 10 + rng.normal(scale=noise, size=(len(signal), d))


def test_loho_folds_hold_out_whole_groups():
    groups = np.array([1, 1, 2, 2, 3])
    folds = P.loho_folds(groups)
    assert len(folds) == 3
    for train, test in folds:
        assert len(set(groups[test])) == 1 and set(groups[test]).isdisjoint(groups[train])


def test_flat_signal_is_linear_at_k1():
    y = horizons()
    X = embed(y[:, None])
    out = P.probe_curves(X, y, ks=(1, 2), probes=("linear",))
    assert out["r2"]["linear"]["act"][0] > 0.95 and out["r2"]["linear"]["cent"][0] > 0.95
    assert out["ceiling"] > 0.95


def test_curved_signal_needs_mlp():
    # a nearly closed circle in 2 dims: decodable from 2 dims, but only nonlinearly.
    # Judged on interior horizons: the held-out extremes are extrapolation for any probe.
    y = horizons()
    theta = (y - y.min()) / (y.max() - y.min()) * 1.8 * np.pi
    X = embed(np.c_[np.cos(theta), np.sin(theta)])
    out = P.probe_curves(X, y, ks=(2,), orders=("act",), ceiling=False)
    mlp, linear = out["r2_interior"]["mlp"]["act"][0], out["r2_interior"]["linear"]["act"][0]
    assert mlp > 0.9 and mlp > linear + 0.15


def test_cent_basis_is_capped_by_training_horizons():
    y = horizons(n_groups=5, per_group=4)
    X = embed(y[:, None], d=50)
    _, bases = P.fold_bases(X[4:], np.round(y[4:], 9), kmax=64)
    assert bases["cent"].shape[0] == 3 and bases["act"].shape[0] == 15


def test_k_beyond_basis_is_nan():
    y = horizons(n_groups=5, per_group=4)
    out = P.probe_curves(embed(y[:, None], d=50), y, ks=(1, 8), probes=("linear",), ceiling=False)
    assert np.isnan(out["r2"]["linear"]["cent"][1]) and not np.isnan(out["r2"]["linear"]["act"][1])


def test_permuted_control_keeps_groups_and_loses_order():
    y = horizons()
    yp = P.permuted_horizons(y)
    assert sorted(set(np.round(yp, 9))) == sorted(set(np.round(y, 9)))
    for g in np.unique(y):
        assert len(set(yp[y == g])) == 1
    out = P.analyze_cell(embed(y[:, None]), y)
    assert out["best"] > 0.95 and max(out["control"]["linear"]["act"]) < 0.5
    assert out["ceiling_interior"] > 0.95
    assert out["k90"]["linear"]["act"] == 1


def test_k_to_reach():
    assert P.k_to_reach([0.1, 0.5, 0.95], 0.9, ks=(1, 2, 4)) == 4
    assert P.k_to_reach([0.1, float("nan")], 0.9, ks=(1, 2)) is None


def test_constant_cell_is_silent_and_unreadable():
    import warnings
    y = horizons(n_groups=6, per_group=5)
    X = np.ones((len(y), 20))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = P.probe_curves(X, y, ks=(1,), orders=("act",), probes=("linear",))
    assert out["ceiling_interior"] <= 0.0 and out["r2_interior"]["linear"]["act"][0] <= 0.0


def test_within_unit_r2_is_zero_for_a_unit_lookup_and_one_for_a_perfect_probe():
    y = np.repeat(np.arange(8.0), 3)
    units = np.repeat(np.array(["minute", "hour"]), 12)
    lookup = np.where(units == "minute", y[units == "minute"].mean(), y[units == "hour"].mean())
    assert abs(P.within_unit_r2(y, lookup, units)) < 1e-12
    assert P.within_unit_r2(y, y, units) == 1.0
    # a constant offset per unit is not penalized: only the ordering and scale within a unit count
    assert P.within_unit_r2(y, y + np.where(units == "hour", 5.0, -1.0), units) == 1.0


def test_within_unit_r2_separates_magnitude_from_unit():
    # planted: one direction carries the unit step, a second the magnitude within the unit;
    # a probe on the unit direction alone scores high overall but 0 within units
    y = horizons()
    units = np.where(y < np.median(y), "short", "long")
    unit_only = embed(np.c_[(units == "long").astype(float)], noise=0.01)
    both = embed(np.c_[(units == "long").astype(float), y / 5], noise=0.01)
    r_unit = P.probe_curves(unit_only, y, units, ks=(1,), orders=("act",), probes=("linear",), ceiling=False)
    r_both = P.probe_curves(both, y, units, ks=(2,), orders=("act",), probes=("linear",), ceiling=False)
    assert r_unit["r2_interior"]["linear"]["act"][0] > 0.6
    assert r_unit["r2_within"]["linear"]["act"][0] < 0.05
    assert r_both["r2_within"]["linear"]["act"][0] > 0.95



def test_horizon_within_unit_r2_ignores_prompt_scatter_around_the_horizon_means():
    # predictions whose horizon averages are exact but whose prompts scatter widely:
    # the prompt-level score is far below 0, the horizon-average score is 1
    y = np.repeat(np.arange(8.0), 20)
    units = np.repeat(np.array(["minute", "hour"]), 80)
    scatter = np.random.default_rng(0).normal(scale=3.0, size=(8, 20))
    pred = y + (scatter - scatter.mean(1, keepdims=True)).ravel()
    assert P.within_unit_r2(y, pred, units) < -1
    assert abs(P.horizon_within_unit_r2(y, pred, units) - 1.0) < 1e-12
    # a unit lookup still scores 0 at the horizon level
    lookup = np.where(units == "minute", 1.5, 5.5)
    assert abs(P.horizon_within_unit_r2(y, lookup, units)) < 1e-12

def test_horizon_within_unit_r2_keeps_renderings_of_one_value_apart():
    # "60 minutes" and "1 hour" are one value in two units: merging them would average the
    # two units' predictions and score a perfect probe below 1
    y = np.repeat(np.array([0.0, 1.0, 2.0, 2.0, 3.0, 4.0]), 5)
    units = np.repeat(np.array(["minute", "minute", "minute", "hour", "hour", "hour"]), 5)
    pred = y + np.where(units == "hour", 0.5, -0.5)
    assert abs(P.horizon_within_unit_r2(y, pred, units) - 1.0) < 1e-12


def test_ridge_inner_cv_holds_out_whole_groups():
    # with prompt-level LOO the near-duplicate prompts of a horizon favour a tiny penalty;
    # the grouped inner CV must still run and fit with few groups
    y = horizons(n_groups=3, per_group=4)
    X = embed(y[:, None], d=20)
    model = P.fit_ridge(X, y, np.round(y, 9))
    assert model.alpha_ in P.ALPHAS
    assert P.fit_ridge(X[:4], y[:4], np.round(y[:4], 9)).alpha_ in P.ALPHAS   # one group: falls back
