"""LEACE properties on synthetic data: zero cross-covariance / equal class means after erasure,
idempotence, rank, minimal change versus the orthogonal projection, preservation of an independent
graded signal, and transfer of the erasure to held-out rows."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from ptm.erasure import LEACE, OrthogonalErasure, cross_covariance, one_hot

D, N = 60, 600


def make(seed=0, n=N):
    """x = class offset (3 classes) + t·w_h + anisotropic noise, class and t independent."""
    rng = np.random.default_rng(seed)
    cls = rng.integers(0, 3, n); t = rng.uniform(-2, 2, n)
    offs = rng.normal(size=(3, D)) * 3.0
    w_h = rng.normal(size=D); w_h /= np.linalg.norm(w_h)
    scales = np.exp(rng.uniform(-2, 2, D))                 # anisotropic noise
    X = offs[cls] + 4.0 * t[:, None] * w_h[None, :] + rng.normal(size=(n, D)) * scales
    return X, cls, t


def test_zero_cross_covariance_and_equal_class_means():
    X, cls, t = make()
    Z = one_hot(cls); er = LEACE().fit(X, Z); Xe = er.transform(X)
    before = np.abs(cross_covariance(X, Z)).max(); after = np.abs(cross_covariance(Xe, Z)).max()
    means = np.stack([Xe[cls == c].mean(0) for c in range(3)])
    spread = np.abs(means - means.mean(0)).max()
    print(f"\nmax |cross-cov| before {before:.3f} after {after:.2e}; class-mean spread after {spread:.2e}; rank {er.rank}")
    assert after < 1e-9 * max(before, 1.0) and spread < 1e-9 and er.rank == 2


def test_idempotent_and_rank_zero_for_constant_concept():
    X, cls, t = make(1)
    er = LEACE().fit(X, one_hot(cls)); X1 = er.transform(X); X2 = er.transform(X1)
    assert np.abs(X2 - X1).max() < 1e-9
    er0 = LEACE().fit(X, np.ones((len(X), 1)))
    assert er0.rank == 0 and np.abs(er0.transform(X) - X).max() < 1e-12
    print(f"\nidempotence max |r(r(x)) − r(x)| {np.abs(X2 - X1).max():.1e}; constant concept rank {er0.rank}")


def test_leace_changes_less_than_orthogonal_projection():
    X, cls, t = make(2); Z = one_hot(cls)
    Xl = LEACE().fit(X, Z).transform(X); Xo = OrthogonalErasure().fit(X, Z).transform(X)
    mse_l = ((Xl - X) ** 2).mean(); mse_o = ((Xo - X) ** 2).mean()
    assert np.abs(cross_covariance(Xo, Z)).max() < 1e-9      # both erase
    print(f"\nMSE change: LEACE {mse_l:.4f}  orthogonal {mse_o:.4f}  (ratio {mse_l / mse_o:.3f})")
    assert mse_l < mse_o


def test_graded_signal_preserved_and_erasure_transfers_to_heldout():
    X, cls, t = make(3, 1200); Z = one_hot(cls)
    tr, te = np.arange(800), np.arange(800, 1200)
    er = LEACE().fit(X[tr], Z[tr]); Xe = er.transform(X)
    r2_before = Ridge(1e-3).fit(X[tr], t[tr]).score(X[te], t[te]); r2_after = Ridge(1e-3).fit(Xe[tr], t[tr]).score(Xe[te], t[te])
    acc_before = LogisticRegression(max_iter=2000).fit(X[tr], cls[tr]).score(X[te], cls[te])
    acc_after = LogisticRegression(max_iter=2000).fit(Xe[tr], cls[tr]).score(Xe[te], cls[te])
    print(f"\nheld-out: ridge R² for t {r2_before:.3f} -> {r2_after:.3f}; class accuracy {acc_before:.3f} -> {acc_after:.3f} (chance 0.333)")
    assert r2_after > 0.9 * r2_before and acc_after < 0.45 and acc_before > 0.95


def test_continuous_concept_block():
    """Erasing a continuous covariate zeroes its cross-covariance and kills its linear decodability."""
    rng = np.random.default_rng(4); n = 500
    u = rng.normal(size=n); t = rng.normal(size=n); w_u = rng.normal(size=D); w_t = rng.normal(size=D)
    X = u[:, None] * w_u + t[:, None] * w_t + rng.normal(size=(n, D))
    Xe = LEACE().fit(X, u).transform(X)
    r2_u = Ridge(1e-3).fit(Xe, u).score(Xe, u); r2_t = Ridge(1e-3).fit(Xe, t).score(Xe, t)
    print(f"\ncontinuous concept: in-sample R² for u after erasure {r2_u:.3f} (target ≈ 0), for t {r2_t:.3f}")
    assert abs(r2_u) < 1e-6          # zero cross-covariance ⇒ the least-squares coefficient is exactly zero
    assert r2_t > 0.8
