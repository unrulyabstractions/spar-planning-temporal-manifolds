"""Linear concept erasure.

LEACE (Belrose et al. 2023, "LEACE: Perfect linear concept erasure in closed form"): the affine map
r(x) = x − W⁺ P W (x − μ) that zeroes the cross-covariance between x and a concept block z while
changing x as little as possible in mean-squared error. W = Σ_xx^{-1/2} whitens x, P is the orthogonal
projector onto the column space of W Σ_xz, and W⁺ maps back. Zero cross-covariance means no linear
predictor of z from r(x) beats a constant; for a one-hot z it is equivalent to equal class-conditional
means (the paper's Theorem 2.3), so no linear classifier of the concept beats chance on the fitted
distribution.

Implementation works in the principal basis of the fitted data (n < d is the normal case here):
Xc = U S Vᵀ, per-axis standard deviations σ = S/√(n−1). In those coordinates W is diag(1/σ), so with
Q an orthonormal basis of diag(1/σ) Vᵀ Σ_xz the erasure is

    r(x) = x − V diag(σ) Q Qᵀ diag(1/σ) Vᵀ (x − μ) = x − A B (x − μ),    A = V diag(σ) Q,  B = Qᵀ diag(1/σ) Vᵀ.

B A = I, so r is idempotent; rank(A) = rank(Σ_xz) ≤ k (k − 1 for one-hot z). `OrthogonalErasure` is the
comparison: the ordinary orthogonal projection removing the span of Σ_xz (equivalently the span of the
class-mean differences), which also zeroes the cross-covariance but is not MSE-minimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def one_hot(labels) -> np.ndarray:
    labels = np.asarray(labels); levels = np.unique(labels)
    return (labels[:, None] == levels[None, :]).astype(np.float64)


def _orth(M: np.ndarray, rel_tol: float) -> np.ndarray:
    """Orthonormal basis of the column space of M, dropping singular values below rel_tol × the largest."""
    if M.shape[1] == 0:
        return np.zeros((M.shape[0], 0))
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    keep = s > rel_tol * s[0] if s.size and s[0] > 0 else np.zeros_like(s, dtype=bool)
    return U[:, keep]


@dataclass
class LEACE:
    rel_tol: float = 1e-6          # singular values of Xc below rel_tol × s_max are treated as zero
    shrinkage: float = 0.0         # adds shrinkage × mean variance to every axis variance before whitening
    mu: np.ndarray = field(default=None, repr=False)
    A: np.ndarray = field(default=None, repr=False)     # [d, q]
    B: np.ndarray = field(default=None, repr=False)     # [q, d]

    def fit(self, X: np.ndarray, Z: np.ndarray) -> "LEACE":
        X = np.asarray(X, dtype=np.float64); Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1: Z = Z[:, None]
        n = len(X); self.mu = X.mean(0); Xc = X - self.mu; Zc = Z - Z.mean(0)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        keep = s > self.rel_tol * s[0]
        U, s, V = U[:, keep], s[keep], Vt[keep].T
        var = s ** 2 / (n - 1) + self.shrinkage * float(np.mean(s ** 2 / (n - 1)))
        sig = np.sqrt(var)
        Sxz_red = (s[:, None] * (U.T @ Zc)) / (n - 1)          # Vᵀ Σ_xz
        Q = _orth(Sxz_red / sig[:, None], self.rel_tol)          # basis of W Σ_xz in the principal coordinates
        self.A = V @ (sig[:, None] * Q)
        self.B = (Q.T / sig[None, :]) @ V.T
        return self

    @property
    def rank(self) -> int:
        return self.A.shape[1]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X - ((X - self.mu) @ self.B.T) @ self.A.T


@dataclass
class OrthogonalErasure:
    """x − Q Qᵀ (x − μ) with Q an orthonormal basis of Σ_xz (the class-mean-difference span for one-hot z)."""
    rel_tol: float = 1e-6
    mu: np.ndarray = field(default=None, repr=False)
    Q: np.ndarray = field(default=None, repr=False)

    def fit(self, X: np.ndarray, Z: np.ndarray) -> "OrthogonalErasure":
        X = np.asarray(X, dtype=np.float64); Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1: Z = Z[:, None]
        self.mu = X.mean(0); Xc = X - self.mu; Zc = Z - Z.mean(0)
        self.Q = _orth(Xc.T @ Zc / (len(X) - 1), self.rel_tol)
        return self

    @property
    def rank(self) -> int:
        return self.Q.shape[1]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X - ((X - self.mu) @ self.Q) @ self.Q.T


def cross_covariance(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64); Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1: Z = Z[:, None]
    return (X - X.mean(0)).T @ (Z - Z.mean(0)) / (len(X) - 1)
