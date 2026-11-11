"""The manifold curve on synthetic activations that lie near a known curve.

Ground truth: a curve g(t) = (cos 0.6 t, sin 0.6 t, 0.3 t) in a random 3-dimensional subspace of R^64, sampled at
48 log-times spaced like the bank's horizon grid (30 s - 500 y), 10 prompts per horizon, plus isotropic noise.
"""

import numpy as np
import pytest

from steer.manifold import Manifold, fit_manifold, local_linear

D = 64


def true_curve(t):
    t = np.atleast_1d(t)
    return np.stack([np.cos(0.6 * t), np.sin(0.6 * t), 0.3 * t], axis=1)


def synthetic(noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((D, 3)))       # [D, 3] orthonormal
    offset = rng.standard_normal(D)
    t_grid = np.log10(30 / 86400) + np.arange(48) * np.log10(1.53)   # log10(days), 30 s upwards
    t = np.repeat(t_grid, 10)
    X = offset + true_curve(t) @ basis.T + noise * rng.standard_normal((len(t), D))
    groups = [f"h{i:02d}" for i in np.repeat(np.arange(48), 10)]
    return X.astype(np.float32), t, groups, basis, offset


def test_local_linear_is_exact_on_lines_including_the_ends():
    t = np.linspace(0, 5, 20)
    Y = np.stack([2 * t + 1, -t], axis=1)
    fit = local_linear(t, Y, np.ones(20), np.array([0.0, 2.5, 5.0]), bandwidth=0.5)
    print("local-linear on a line at t = 0, 2.5, 5:", fit.round(6).tolist())
    assert np.allclose(fit, [[1, 0], [6, -2.5], [11, -5]])


@pytest.mark.parametrize("noise,min_cos", [(0.05, 0.995), (0.005, 0.9999)])
def test_curve_recovers_subspace_and_horizon_of_each_prompt(noise, min_cos):
    # The subspace angle error is statistical (PCA of noisy means) and shrinks with the noise: measured smallest
    # principal cosine 0.9969 at noise 0.05, 0.99997 at 0.005.
    X, t, groups, basis, _ = synthetic(noise=noise)
    m = fit_manifold(X, t, groups, k=3, bandwidth=0.3)
    err = np.abs(m.project_t(X) - t)
    subspace_overlap = np.linalg.svd(m.U.T @ basis, compute_uv=False)
    print(f"noise {noise}: explained {m.explained.round(3)}; principal cosines with the true subspace "
          f"{subspace_overlap.round(5)}; median |t-hat - t| {np.median(err):.4f} decades, max {err.max():.3f}")
    assert subspace_overlap.min() > min_cos
    assert np.median(err) < 0.05


def test_displace_is_identity_at_zero_and_snap_is_not():
    X, t, groups, _, _ = synthetic()
    m = fit_manifold(X, t, groups, k=3, bandwidth=0.3)
    disp, snap = m.steer(X, 0.0, "displace"), m.steer(X, 0.0, "snap")
    print(f"delta 0: displace moves rows by up to {np.abs(disp - X).max():.2e}; snap by up to "
          f"{np.linalg.norm(snap - X, axis=1).max():.3f} (the off-curve offset it removes)")
    assert np.allclose(disp, X, atol=1e-6)
    assert np.linalg.norm(snap - X, axis=1).max() > 1e-3


def _inner_moves(m, X, t, delta, mode):
    lo, hi = m.t_range
    inner = (t > lo + 1.1) & (t < hi - 1.1)                 # room to move a decade either way
    moved = m.steer(X[inner], delta, mode)
    shift = m.project_t(moved) - m.project_t(X[inner])
    orth = (moved - X[inner]) - (moved - X[inner]) @ m.U @ m.U.T
    offset = np.linalg.norm(m.coords(X[inner]) - m.curve[m._nearest(X[inner])], axis=1)
    return shift, orth, offset


@pytest.mark.parametrize("delta", [-1.0, 0.5, 1.0])
def test_snap_moves_t_hat_by_exactly_delta(delta):
    # Snap puts the subspace part ON the curve at t-hat + delta, so the new t-hat is t-hat + delta up to the
    # polyline's resolution (grid spacing 0.0036 decade here).
    X, t, groups, _, _ = synthetic()
    m = fit_manifold(X, t, groups, k=3, bandwidth=0.3)
    shift, orth, _ = _inner_moves(m, X, t, delta, "snap")
    print(f"snap, delta {delta:+}: t-hat shift median error {np.median(np.abs(shift - delta)):.4f}, "
          f"max {np.abs(shift - delta).max():.4f}; largest change orthogonal to the subspace {np.abs(orth).max():.2e}")
    assert np.median(np.abs(shift - delta)) < 0.004
    assert np.abs(orth).max() < 1e-9


@pytest.mark.parametrize("noise", [0.05, 0.005])
def test_displace_shift_error_tracks_the_off_curve_offset(noise):
    # Displace keeps the activation's offset from the curve. That offset is normal to the curve at t-hat but, where
    # the curve turns, not at t-hat + delta, so the new nearest point is off by roughly offset * turning / speed.
    # This is a property of the mode, not a bug: measured median errors 0.028 decade at noise 0.05 and 0.010 at
    # 0.005 (where the offset is mostly the smoother's bias on this helix), against 0.0017 for snap.
    X, t, groups, _, _ = synthetic(noise=noise)
    m = fit_manifold(X, t, groups, k=3, bandwidth=0.3)
    shift, orth, offset = _inner_moves(m, X, t, 1.0, "displace")
    err = np.abs(shift - 1.0)
    speed = np.linalg.norm(np.gradient(m.curve, m.t_grid, axis=0), axis=1).min()
    print(f"displace, noise {noise}: median offset {np.median(offset):.4f}, median t-hat error {np.median(err):.4f}, "
          f"max {err.max():.4f}; bound offset/min speed, max {(offset / speed).max():.4f}")
    assert np.all(err <= offset / speed + 0.004)
    assert np.abs(orth).max() < 1e-9


def test_steer_clamps_to_the_observed_range():
    X, t, groups, _, _ = synthetic()
    m = fit_manifold(X, t, groups, k=3, bandwidth=0.3)
    top = m.project_t(m.steer(X, 10.0, "snap"))
    print(f"delta +10 decades lands at t-hat {np.unique(top.round(4))} (range end {m.t_range[1]:.4f})")
    assert np.allclose(top, m.t_range[1])


def test_save_and_load_round_trip(tmp_path):
    X, t, groups, _, _ = synthetic()
    m = fit_manifold(X, t, groups, k=3, bandwidth=0.3)
    m.save(tmp_path / "m.npz")
    m2 = Manifold.load(tmp_path / "m.npz")
    assert np.array_equal(m.project_t(X), m2.project_t(X))
    assert m2.bandwidth == 0.3
