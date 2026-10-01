"""HorizonPLS / ReducedRankRidge on synthetic data: recover the planted linear + quadratic response
directions and their degrees where PCA's leading component is a nuisance offset; report no response at
unplanted degrees; the component count tracks the planted dimension. PLS2 is shown to degrade when the
nuisance variance dominates, and RRR to hold."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from ptm.pls import HorizonPLS, LegendreTargets, ReducedRankRidge

D = 80


def make(seed, n=900, cubic=0.0, nuis=1.5):
    rng = np.random.default_rng(seed)
    t = rng.choice(np.linspace(-2.5, 2, 12), n)
    cls = rng.integers(0, 3, n); offs = rng.normal(size=(3, D)) * nuis
    W = np.linalg.qr(rng.normal(size=(D, 3)))[0]                                # orthonormal planted directions
    T = LegendreTargets(3, -2.5, 2).fit(t)(t)                                    # unit-variance degree 1..3 columns
    X = offs[cls] + 4.0 * np.outer(T[:, 0], W[:, 0]) + 3.0 * np.outer(T[:, 1], W[:, 1]) + cubic * np.outer(T[:, 2], W[:, 2]) + rng.normal(size=(n, D))
    return X, t, cls, W


def angles(A, B):
    sv = np.linalg.svd(A.T @ B, compute_uv=False)
    return np.degrees(np.arccos(np.clip(sv, 0, 1)))


def check(model, X, t, W, tr, te, label):
    m = model.fit(X[tr], t[tr])
    r2 = m.r2_per_degree(X[te], t[te]); ang = angles(m.response_directions, W[:, :2]); C = m.degree_correlations(X[te], t[te])
    print(f"{label:24s} held-out R² by degree {np.round(r2, 3)}; principal angles of response directions to planted span {np.round(ang, 1)}°; degree assignment {C.argmax(1) + 1}")
    return r2, ang, C


def test_recovers_planted_degrees_where_pca_fails():
    X, t, cls, W = make(0); tr, te = np.arange(600), np.arange(600, 900)
    pc1 = PCA(1).fit(X[tr]).transform(X[te])[:, 0]; rho_pca = abs(spearmanr(pc1, t[te]).statistic)
    print(f"\nnuisance 1.5: PCA |rho(PC1, t)| {rho_pca:.2f}")
    r2, ang, C = check(ReducedRankRidge(2, degree=4, t_min=-2.5, t_max=2, alpha=1.0), X, t, W, tr, te, "RRR k=2")
    assert r2[0] > 0.9 and r2[1] > 0.85 and r2[2] < 0.1 and r2[3] < 0.1 and ang.max() < 15
    assert C[0].argmax() == 0 and C[1].argmax() == 1                            # component 1 ↔ linear, 2 ↔ quadratic
    # PLS2 spends components on deflating nuisance: k=2 predicts only the linear column, k=4 both (documented)
    r2_2, _, _ = check(HorizonPLS(2, degree=4, t_min=-2.5, t_max=2), X, t, W, tr, te, "PLS2 k=2")
    r2_4, ang4, _ = check(HorizonPLS(4, degree=4, t_min=-2.5, t_max=2), X, t, W, tr, te, "PLS2 k=4")
    assert r2_4[0] > 0.9 and r2_4[1] > 0.85 and r2_4[2] < 0.1 and r2_4[3] < 0.1 and ang4.max() < 15
    assert r2_2[1] < r2_4[1] - 0.2
    assert rho_pca < 0.3


def test_rrr_holds_under_dominant_nuisance_where_pls_degrades():
    X, t, cls, W = make(3, nuis=6.0); tr, te = np.arange(600), np.arange(600, 900)
    print("\nnuisance 6.0 (class offsets ≫ signal):")
    r2p, angp, _ = check(HorizonPLS(2, degree=4, t_min=-2.5, t_max=2), X, t, W, tr, te, "PLS2 k=2")
    r2r, angr, _ = check(ReducedRankRidge(2, degree=4, t_min=-2.5, t_max=2, alpha=1.0), X, t, W, tr, te, "RRR k=2")
    assert r2r[0] > 0.9 and r2r[1] > 0.85          # (response-direction angles are not asserted here: their estimate has noise ∝ nuisance/√n)
    assert r2p[0] < 0.7                                                          # documents the failure mode


def test_component_count_tracks_planted_dimension():
    for cubic, expect in [(0.0, 2), (2.0, 3)]:
        X, t, cls, W = make(1, cubic=cubic, nuis=6.0); tr, te = np.arange(600), np.arange(600, 900)
        tot = [ReducedRankRidge(k, degree=4, t_min=-2.5, t_max=2, alpha=1.0).fit(X[tr], t[tr]).r2_total(X[te], t[te]) for k in range(1, 6)]
        k95 = int(np.argmax(np.array(tot) >= 0.95 * max(tot))) + 1
        print(f"\ncubic weight {cubic}: RRR total held-out R² by rank {np.round(tot, 3)} → k95 = {k95} (planted {expect})")
        assert k95 == expect


def test_targets_are_orthonormal_on_fit_rows():
    t = np.random.default_rng(2).uniform(-2.5, 2, 500); T = LegendreTargets(4, -2.5, 2).fit(t)(t)
    G = T.T @ T / (len(t) - 1)
    print(f"\nGram of orthogonalised targets, max |off-diag| {np.abs(G - np.eye(4)).max():.1e}")
    assert np.abs(G - np.eye(4)).max() < 1e-10
