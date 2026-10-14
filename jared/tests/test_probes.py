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
