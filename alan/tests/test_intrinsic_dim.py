"""Levina–Bickel MLE of intrinsic dimension (ptm/intrinsic_dim.py) on data of known dimension.

Reference values from the paper: 5-d Gaussian, n = 1000, k = 10..20 is "near the true value" (Fig. 1a);
Swiss roll, n = 1000: 2.1 (SD 0.02) (Table 1). Every manifold is checked both axis-aligned and after a random
rotation into a higher ambient space, since the estimate depends only on distances.
"""
import numpy as np
import pytest

from ptm.intrinsic_dim import (dedupe_rows, gaussian_matched, knn_distances, levina_bickel, local_mle,
                               mle_curve)


def embed(Y: np.ndarray, D: int, rng) -> np.ndarray:
    """Isometric embedding of [n, m] into R^D by a random orthonormal map."""
    Q, _ = np.linalg.qr(rng.standard_normal((D, Y.shape[1])))
    return Y @ Q.T


def test_eq8_by_hand():
    # one point, neighbours at 1, 2, 4: k=3 -> [ (log 4 + log 2) / 2 ]^-1 = 2 / (3 log 2)
    D = np.array([[1.0, 2.0, 4.0]])
    got = local_mle(D, 3)[0]
    want = 2.0 / (3.0 * np.log(2.0))
    print(f"\n  eq.(8) k=3 on T=(1,2,4): {got:.6f} (hand {want:.6f}); k-2 form {local_mle(D, 3, True)[0]:.6f}")
    assert got == pytest.approx(want, rel=1e-12)
    assert local_mle(D, 3, unbiased=True)[0] == pytest.approx(want / 2, rel=1e-12)


def test_knn_distances_exact_against_brute_force():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 40)) * 50 + 1e3                         # large offset: cancellation-prone
    D = knn_distances(X, 25)
    full = np.linalg.norm(X[:, None] - X[None], axis=2)
    np.fill_diagonal(full, np.inf)
    ref = np.sort(full, axis=1)[:, :25]
    print(f"\n  knn max abs err vs brute force: {np.abs(D - ref).max():.2e}")
    np.testing.assert_allclose(D, ref, rtol=1e-12)


@pytest.mark.parametrize("m", [1, 2, 5, 10])
def test_gaussian_recovers_dimension(m):
    rng = np.random.default_rng(m)
    Y = rng.standard_normal((1000, m))
    est_axis = levina_bickel(np.pad(Y, ((0, 0), (0, 64 - m))))
    est_rot = levina_bickel(embed(Y, 64, rng) * 37.0)                     # rotated and scaled
    print(f"\n  N_{m}(0,I) n=1000: axis-aligned {est_axis:.3f}, rotated+scaled {est_rot:.3f}")
    assert est_axis == pytest.approx(est_rot, rel=1e-8)
    tol = 0.12 if m <= 5 else 0.15                                        # paper: negative bias grows with m
    assert est_axis == pytest.approx(m, rel=tol)


def test_swiss_roll_matches_paper_table1():
    rng = np.random.default_rng(1)
    t = 1.5 * np.pi * (1 + 2 * rng.random(1000))
    h = 21 * rng.random(1000)
    Y = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
    est = levina_bickel(Y)
    est_rot = levina_bickel(embed(Y, 100, rng))
    print(f"\n  Swiss roll n=1000: {est:.3f} (paper Table 1: 2.1), rotated into R^100: {est_rot:.3f}")
    assert est == pytest.approx(est_rot, rel=1e-8)
    assert 1.9 <= est <= 2.3


def test_curved_one_dimensional_manifold():
    # a closed curve in R^6 with non-constant curvature, uniformly random parameter
    rng = np.random.default_rng(2)
    s = 2 * np.pi * rng.random(1000)
    Y = np.column_stack([np.cos(s), np.sin(s), 0.5 * np.cos(2 * s), 0.5 * np.sin(2 * s), 0.3 * np.cos(3 * s),
                         0.3 * np.sin(3 * s)])
    est = levina_bickel(embed(Y, 200, rng))
    print(f"\n  closed curve in R^6 -> R^200, n=1000: {est:.3f}")
    assert est == pytest.approx(1.0, rel=0.1)


def test_curve_over_k_decreases_with_k_for_gaussian():
    # Fig. 1: positive bias at very small k, then roughly flat with a slow negative drift
    rng = np.random.default_rng(3)
    D = knn_distances(rng.standard_normal((1000, 5)), 100)
    c = mle_curve(D, [3, 5, 10, 20, 50, 100])
    print("\n  N_5 m_k at k=3,5,10,20,50,100: " + " ".join(f"{v:.2f}" for v in c))
    assert c[0] > c[2] > c[-1]


def test_duplicates_raise_and_dedupe():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((50, 8)).astype(np.float32)
    Xd = np.vstack([X, X[:5]])
    with pytest.raises(ValueError):
        knn_distances(Xd, 10)
    keep = dedupe_rows(Xd)
    assert keep.tolist() == list(range(50))


def test_gaussian_matched_has_same_covariance_spectrum():
    rng = np.random.default_rng(5)
    X = embed(rng.standard_normal((400, 3)) * [5.0, 2.0, 0.5], 30, rng)
    G = gaussian_matched(X, np.random.default_rng(6))
    sx = np.linalg.svd(X - X.mean(0), compute_uv=False)[:3]
    sg = np.linalg.svd(G - G.mean(0), compute_uv=False)[:3]
    print(f"\n  singular values X {np.round(sx, 1)} vs matched Gaussian {np.round(sg, 1)}")
    np.testing.assert_allclose(sg, sx, rtol=0.15)
