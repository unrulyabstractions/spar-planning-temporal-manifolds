import numpy as np

from spar_horizon.geometry import display_layer, horizon_probe, probe_sweep, sweep, transfer_matrix


def planted(n=120, d=32, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    log_h = rng.uniform(-6, 3, n)
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    X = np.outer(log_h, direction) * 5 + rng.normal(scale=noise, size=(n, d))
    return X, log_h, direction


def test_probe_recovers_planted_direction():
    X, log_h, direction = planted()
    r2, d, _ = horizon_probe(X, log_h)
    assert r2 > 0.95
    assert abs(np.dot(d, direction)) > 0.95


def test_probe_fails_on_noise():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 32))
    log_h = rng.uniform(-6, 3, 120)
    r2, _, _ = horizon_probe(X, log_h)
    assert r2 < 0.3


def test_sweep_finds_signal_and_skips_constant_columns():
    X, log_h, _ = planted()
    const = np.ones((120, 1, 32))
    acts = [np.concatenate([X[:, None, :], const], axis=1)]
    has_h = np.ones(120, dtype=bool)
    rho, Z = sweep(acts, ["signal", "const"], has_h, log_h, verbose=False)
    assert rho[0, 0] > 0.95
    assert np.isnan(rho[0, 1]) and Z[0][1] is None


def test_display_layer_prefers_mean():
    rho = np.array([[0.99, 0.1, 0.1], [0.8, 0.8, 0.8], [np.nan, np.nan, np.nan]])
    assert display_layer(rho) == 1


def test_probe_sweep_shape():
    X, log_h, _ = planted()
    acts = [np.stack([X, X * 0 + 1], axis=1), np.stack([X, X], axis=1)]
    r2 = probe_sweep(acts, np.ones(120, dtype=bool), log_h)
    assert r2.shape == (2, 2)
    assert r2[0, 0] > 0.9 and np.isnan(r2[0, 1]) and r2[1, 1] > 0.9


def test_transfer_matrix_diagonal_and_shared_axis():
    X1, h1, direction = planted(seed=0)
    rng = np.random.default_rng(5)
    h2 = rng.uniform(-6, 3, 120)
    X2 = np.outer(h2, direction) * 5 + rng.normal(scale=0.1, size=X1.shape) + 3.0
    M = transfer_matrix([X1, X2], [h1, h2])
    assert M.shape == (2, 2)
    assert np.all(M > 0.95)
