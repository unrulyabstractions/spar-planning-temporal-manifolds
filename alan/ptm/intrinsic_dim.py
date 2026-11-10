"""Intrinsic dimension by the maximum-likelihood estimator of Levina & Bickel, "Maximum Likelihood
Estimation of Intrinsic Dimension", NIPS 17 (2004).

With T_1(x) <= ... <= T_k(x) the Euclidean distances from x to its k nearest neighbours (x itself
excluded), the local estimate is the paper's eq. (8)

    m_k(x) = [ 1/(k-1) * sum_{j=1}^{k-1} log( T_k(x) / T_j(x) ) ]^{-1}.

§3.1: under the Poisson approximation, U = m^{-1} sum_j log(T_k/T_j) ~ Gamma(k-1, 1) and E U^{-1} = 1/(k-2),
so dividing by k-2 instead of k-1 makes the estimate unbiased to first order (`unbiased=True`); the two differ
by the exact factor (k-2)/(k-1). The global estimate is eq. (9): average the local estimates over all points,
then average over k = k1..k2; the paper fixes k1 = 10, k2 = 20 throughout. The paper reports a negative bias
that grows with m and shrinks with n (Fig. 1, Fig. 2b), so a value is only interpretable next to a synthetic
calibration at the same n.
"""

from __future__ import annotations

import numpy as np


def knn_distances(X: np.ndarray, k_max: int, block: int = 256) -> np.ndarray:
    """[n, k_max] distances to the k_max nearest other points, ascending. Candidates come from the Gram
    expansion |x|^2 + |y|^2 - 2 x.y in float64 on centered data; the selected distances are then recomputed
    directly as |x - y| so near-duplicates are not lost to cancellation. Raises on zero distances
    (duplicate points make log(T_k / T_j) infinite: deduplicate first)."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0)
    n = X.shape[0]
    if not 1 <= k_max < n:
        raise ValueError(f"k_max={k_max} needs 1 <= k_max < n={n}")
    sq = np.einsum("ij,ij->i", X, X)
    out = np.empty((n, k_max))
    for s in range(0, n, block):
        e = min(s + block, n)
        d2 = sq[s:e, None] + sq[None, :] - 2.0 * (X[s:e] @ X.T)
        d2[np.arange(e - s), np.arange(s, e)] = np.inf                     # exclude self
        # a margin of extra candidates guards against ordering errors in the approximate distances
        m = min(n - 1, k_max + 8)
        cand = np.argpartition(d2, m - 1, axis=1)[:, :m]
        for r in range(s, e, 8):                                            # 8 rows at a time: [8, m, d] float64
            q = min(r + 8, e)
            exact = np.linalg.norm(X[r:q, None, :] - X[cand[r - s : q - s]], axis=2)
            out[r:q] = np.sort(exact, axis=1)[:, :k_max]
    if (out[:, 0] <= 0).any():
        raise ValueError(f"{int((out[:, 0] <= 0).sum())} points have a zero-distance neighbour: deduplicate first")
    return out


def local_mle(D: np.ndarray, k: int, unbiased: bool = False) -> np.ndarray:
    """Eq. (8) at every point from a sorted distance table D [n, >=k]. unbiased=True divides by k-2."""
    if k < 3 or k > D.shape[1]:
        raise ValueError(f"k={k} must be in [3, {D.shape[1]}]")
    s = np.log(D[:, k - 1 : k] / D[:, : k - 1]).sum(axis=1)               # sum_{j=1}^{k-1} log(T_k / T_j)
    return ((k - 2) if unbiased else (k - 1)) / s


def mle_curve(D: np.ndarray, ks, unbiased: bool = False) -> np.ndarray:
    """m_k = mean_i m_k(X_i) for each k (eq. 9, left)."""
    return np.array([local_mle(D, k, unbiased).mean() for k in ks])


def levina_bickel(X: np.ndarray, k1: int = 10, k2: int = 20, unbiased: bool = False,
                  D: np.ndarray | None = None) -> float:
    """Eq. (9): mean over k = k1..k2 of the point-averaged eq. (8) estimate."""
    if D is None:
        D = knn_distances(X, k2)
    return float(mle_curve(D, range(k1, k2 + 1), unbiased).mean())


def dedupe_rows(X: np.ndarray) -> np.ndarray:
    """Indices of the first occurrence of each distinct row (exact equality), in original order."""
    _, first = np.unique(np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))),
                         return_index=True)
    return np.sort(first)


def gaussian_matched(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A Gaussian sample with the same n and the same sample covariance as X (an ellipsoid with X's PCA
    spectrum). Its estimate is what X would give if it filled its covariance ellipsoid with no manifold
    structure."""
    Xc = np.asarray(X, dtype=np.float64) - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    n = Xc.shape[0]
    Z = rng.standard_normal((n, S.size))
    return (Z * (S / np.sqrt(n - 1))) @ Vt
