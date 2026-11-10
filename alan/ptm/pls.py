"""Two-block partial least squares (PLS2) between activations and a polynomial basis of log horizon.

PCA orders directions by variance, so on a mixed population the horizon response can sit below the
nuisance components and the PCA basis can rotate between runs (the extended-grid L37 T3 case, where PC1
and PC2 swapped the linear and quadratic terms). PLS2 orders directions by covariance with a target block
Y instead. With Y = Legendre polynomials of t = log10 H (degree 1..K), orthogonalised in degree order on
the fitted rows so column k is a degree-k polynomial uncorrelated with lower degrees, each PLS component
is a direction in activation space maximally covariant with the (residual) polynomial response. The
held-out R² of each Y column as a function of the number of components, and the activation-variance
share carried by each component, give a rotation-free polynomial ladder and a supervised count of
horizon-responsive dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.legendre import legvander
from sklearn.cross_decomposition import PLSRegression


def _variance_share(X, dirs) -> np.ndarray:
    Q = np.linalg.qr(dirs)[0]; Xc = X - X.mean(0)
    return ((Xc @ Q) ** 2).sum(0) / (Xc ** 2).sum()


@dataclass
class LegendreTargets:
    degree: int = 4
    t_min: float = -3.0
    t_max: float = 2.5
    R: np.ndarray = field(default=None, repr=False)     # from a QR on the fitted rows: columns orthogonalised in degree order

    def raw(self, t):
        u = (2 * (np.asarray(t, dtype=float) - self.t_min) / (self.t_max - self.t_min)) - 1
        return legvander(u, self.degree)[:, 1:]

    def fit(self, t):
        P = self.raw(t); Pc = P - P.mean(0)
        _, R = np.linalg.qr(Pc)
        self.R = R / np.sqrt(len(P) - 1); self.mean = P.mean(0)          # unit-variance columns on the fitted rows
        return self

    def __call__(self, t):
        return np.linalg.solve(self.R.T, (self.raw(t) - self.mean).T).T     # (P − mean) R^{-1}


@dataclass
class HorizonPLS:
    n_components: int = 2
    degree: int = 4
    t_min: float = -3.0
    t_max: float = 2.5
    targets: LegendreTargets = field(default=None, repr=False)
    pls: PLSRegression = field(default=None, repr=False)

    def fit(self, X, t):
        X = np.asarray(X, dtype=np.float64)
        self.targets = LegendreTargets(self.degree, self.t_min, self.t_max).fit(t)
        Y = self.targets(t)
        self.pls = PLSRegression(n_components=self.n_components, scale=False, max_iter=2000, tol=1e-8).fit(X, Y)
        self._XtY = (X - X.mean(0)).T @ (Y - Y.mean(0)) / (len(X) - 1)      # [d, K] cross-covariance with the targets
        return self

    @property
    def directions(self) -> np.ndarray:
        """[d, k] unit vectors: the rotations mapping centred X to component scores."""
        W = self.pls.x_rotations_
        return W / np.linalg.norm(W, axis=0, keepdims=True)

    @property
    def response_directions(self) -> np.ndarray:
        """[d, k] unit vectors: the forward (response) directions — regression of X on the target
        combinations each component predicts (Y is orthonormal on the fitted rows, so this is the
        cross-covariance with those combinations). Differs from `directions` (the decoder weights)
        whenever the nuisance covariance is anisotropic."""
        Vq = np.linalg.qr(self.pls.y_loadings_)[0]
        L = self._XtY @ Vq
        return L / np.linalg.norm(L, axis=0, keepdims=True)

    def scores(self, X):
        return self.pls.transform(np.asarray(X, dtype=np.float64))

    def r2_per_degree(self, X, t) -> np.ndarray:
        """Held-out R² of each orthogonalised Legendre column (degree 1..K) from this model."""
        Y = self.targets(t); Yh = self.pls.predict(np.asarray(X, dtype=np.float64))
        return 1 - ((Yh - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)

    def r2_total(self, X, t) -> float:
        Y = self.targets(t); Yh = self.pls.predict(np.asarray(X, dtype=np.float64))
        return float(1 - ((Yh - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())

    def x_variance_share(self, X) -> np.ndarray:
        """Share of the total variance of X along each response direction (orthonormalised in component
        order, so later shares exclude what earlier directions already carry)."""
        return _variance_share(np.asarray(X, dtype=np.float64), self.response_directions)

    def centroid_variance_share(self, X, t) -> np.ndarray:
        """Same, for the between-horizon centroid variance (the quantity the PCA ladder ranks)."""
        X = np.asarray(X, dtype=np.float64); tv = np.unique(t)
        return _variance_share(np.stack([X[t == v].mean(0) for v in tv]), self.response_directions)

    def degree_correlations(self, X, t) -> np.ndarray:
        """[k, K] |corr| between each component score and each orthogonalised degree column (degree assignment)."""
        S = self.scores(X); Y = self.targets(t)
        return np.abs(np.corrcoef(S.T, Y.T)[:S.shape[1], S.shape[1]:])


@dataclass
class ReducedRankRidge:
    """Rank-k ridge regression of the Legendre target block on X (reduced-rank regression, RRR).

    PLS2 picks directions by raw covariance with Y, so when the activations carry nuisance variance far
    larger than the horizon response (the residual stream), sampling noise in the cross-covariance pulls
    the components toward high-variance nuisance directions and the held-out fit degrades. RRR instead
    takes the full ridge predictor Ŷ = Xc B and keeps the best rank-k approximation of the fitted values
    (SVD of Ŷ on the training rows): the rank-k linear map that predicts the polynomial block best. Held-out
    R² as a function of k is then a direct count of horizon-responsive dimensions. Dual (kernel) form, so
    the cost is O(n² d) with n < d.
    """
    rank: int = 2
    degree: int = 4
    t_min: float = -3.0
    t_max: float = 2.5
    alpha: float = 1.0
    targets: LegendreTargets = field(default=None, repr=False)
    B: np.ndarray = field(default=None, repr=False)          # [d, K] full ridge coefficients
    V: np.ndarray = field(default=None, repr=False)          # [K, rank] right singular vectors of the fitted values
    x_mean: np.ndarray = field(default=None, repr=False)
    y_mean: np.ndarray = field(default=None, repr=False)
    L: np.ndarray = field(default=None, repr=False)

    def fit(self, X, t):
        X = np.asarray(X, dtype=np.float64); n = len(X)
        self.targets = LegendreTargets(self.degree, self.t_min, self.t_max).fit(t); Y = self.targets(t)
        self.x_mean = X.mean(0); self.y_mean = Y.mean(0); Xc = X - self.x_mean; Yc = Y - self.y_mean
        K = Xc @ Xc.T
        self.B = Xc.T @ np.linalg.solve(K + self.alpha * np.eye(n), Yc)
        Yhat = Xc @ self.B
        _, _, Vt = np.linalg.svd(Yhat, full_matrices=False)
        self.V = Vt[: self.rank].T
        self.L = Xc.T @ (Yc @ self.V) / (n - 1)                  # [d, rank] response directions: regression of X on the kept target combinations
        return self

    @property
    def response_directions(self) -> np.ndarray:
        return self.L / np.linalg.norm(self.L, axis=0, keepdims=True)

    @property
    def coef(self) -> np.ndarray:
        return self.B @ self.V @ self.V.T                      # rank-k coefficients [d, K]

    @property
    def directions(self) -> np.ndarray:
        W = self.B @ self.V
        return W / np.linalg.norm(W, axis=0, keepdims=True)

    def scores(self, X):
        return (np.asarray(X, dtype=np.float64) - self.x_mean) @ (self.B @ self.V)

    def predict(self, X):
        return (np.asarray(X, dtype=np.float64) - self.x_mean) @ self.coef + self.y_mean

    def r2_per_degree(self, X, t) -> np.ndarray:
        Y = self.targets(t); Yh = self.predict(X)
        return 1 - ((Yh - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)

    def r2_total(self, X, t) -> float:
        Y = self.targets(t); Yh = self.predict(X)
        return float(1 - ((Yh - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())

    def x_variance_share(self, X) -> np.ndarray:
        """Share of the total variance of X along each response direction (orthonormalised in component
        order, so later shares exclude what earlier directions already carry)."""
        return _variance_share(np.asarray(X, dtype=np.float64), self.response_directions)

    def centroid_variance_share(self, X, t) -> np.ndarray:
        """Same, for the between-horizon centroid variance (the quantity the PCA ladder ranks)."""
        X = np.asarray(X, dtype=np.float64); tv = np.unique(t)
        return _variance_share(np.stack([X[t == v].mean(0) for v in tv]), self.response_directions)

    def degree_correlations(self, X, t) -> np.ndarray:
        S = self.scores(X); Y = self.targets(t)
        return np.abs(np.corrcoef(S.T, Y.T)[:S.shape[1], S.shape[1]:])
