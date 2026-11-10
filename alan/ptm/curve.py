"""Forward geometric model of the horizon manifold at one cell, and its inverse.

    x = mu + f(t) + c(context) + eps,      t = log10 horizon (years), x in R^d

f is a curve given by a basis expansion f(t) = B^T phi(t): 'line' (phi = [t]), 'quad' ([t, t^2]),
'spline' (natural cubic B-spline basis with K interior knots). Context enters as additive offsets
('offset': one indicator per context level) or as per-context curves ('interaction': basis × context).
All coefficients are fit jointly by ridge least squares. The fitted curve gives basis-free geometry
(tangent, curvature, tortuosity, spectrum of the curve's own covariance) and a decoder that projects
an activation onto the curve (1-D search over t).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import BSpline


# ------------------------------------------------------------------ bases

def _knots(t_min: float, t_max: float, n_interior: int, k: int = 3) -> np.ndarray:
    inner = np.linspace(t_min, t_max, n_interior + 2)[1:-1]
    return np.concatenate([[t_min] * (k + 1), inner, [t_max] * (k + 1)])


@dataclass
class Basis:
    kind: str                      # line | quad | spline
    t_min: float = -3.0
    t_max: float = 2.5
    n_interior: int = 4            # spline interior knots
    knots: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.kind == "spline" and self.knots is None:
            self.knots = _knots(self.t_min, self.t_max, self.n_interior)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if self.kind == "line":
            return t[:, None]
        if self.kind == "quad":
            return np.column_stack([t, t ** 2])
        if self.kind == "spline":
            tc = np.clip(t, self.t_min, self.t_max)
            M = BSpline.design_matrix(tc, self.knots, 3).toarray()
            return M[:, 1:]        # drop one column: the B-splines sum to 1, which the intercept already carries
        raise ValueError(self.kind)

    @property
    def size(self) -> int:
        return self(np.array([0.0])).shape[1]


def quantile_basis(t, n_interior: int = 8) -> Basis:
    """Cubic spline basis with interior knots at equally spaced quantiles of t, so every basis function has data support
    when t is sampled unevenly (log10 of a dense integer grid has 1 point in [0, 0.3) and hundreds per 0.3 decades at
    the top). Equally spaced knots there leave the first basis functions resting on a handful of points, and a
    near-unregularised fit extrapolates wildly in held-out folds."""
    t = np.asarray(t, dtype=float); lo, hi = float(t.min()), float(t.max())
    inner = np.quantile(t, np.linspace(0, 1, n_interior + 2)[1:-1])
    return Basis("spline", t_min=lo, t_max=hi, n_interior=n_interior, knots=np.concatenate([[lo] * 4, inner, [hi] * 4]))


def cv_r2(X, t, make_basis, n_splits: int = 5, alpha: float = 1e-6, seed: int = 0, ctx=None, context: str = "none",
          interpolate_only: bool = False, make_level_basis=None):
    """K-fold held-out R² of the forward curve, as a share of variance about the training-fold mean.

    context: none (x = mu + f(t)) | offset (+ c(ctx)) | interaction (CurveModel per-level curves on make_basis's knots)
             | separate (each level fit on its own, basis make_level_basis(t of that level); the baseline stays the pooled
             training-fold mean, so the score is comparable across contexts).
    interpolate_only: score only test rows whose t lies inside the training rows' t range (of the same level when there
             is a context); excluded rows get NaN errors. The extreme value of each level otherwise lands in some test
             fold and is extrapolated, which a curve fit is not meant to do.
    Returns (R², per-row squared error, per-row squared error of the training-fold mean)."""
    from sklearn.model_selection import KFold
    X = np.asarray(X, dtype=np.float64); t = np.asarray(t, dtype=float)
    c = None if ctx is None else np.asarray(ctx)
    err = np.full(len(X), np.nan); err0 = np.full(len(X), np.nan)
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        if context == "none":
            pred = CurveModel(make_basis(), alpha=alpha).fit(X[tr], t[tr]).predict(t[te])
        elif context == "separate":
            pred = np.zeros((len(te), X.shape[1]))
            for lv in np.unique(c[te]):
                a_tr, a_te = tr[c[tr] == lv], c[te] == lv
                pred[a_te] = CurveModel(make_level_basis(t[c == lv]), alpha=alpha).fit(X[a_tr], t[a_tr]).predict(t[te][a_te])
        else:
            pred = CurveModel(make_basis(), context=context, alpha=alpha).fit(X[tr], t[tr], c[tr]).predict(t[te], c[te])
        keep = np.ones(len(te), bool)
        if interpolate_only:
            for lv in (np.unique(c[te]) if c is not None else [None]):
                m_te = np.ones(len(te), bool) if lv is None else c[te] == lv
                t_tr = t[tr] if lv is None else t[tr][c[tr] == lv]
                keep &= ~m_te | ((t[te] >= t_tr.min()) & (t[te] <= t_tr.max()))
        e = ((X[te] - pred) ** 2).sum(1); e0 = ((X[te] - X[tr].mean(0)) ** 2).sum(1)
        err[te] = np.where(keep, e, np.nan); err0[te] = np.where(keep, e0, np.nan)
    return float(1 - np.nansum(err) / np.nansum(err0)), err, err0


# ------------------------------------------------------------------ model

@dataclass
class CurveModel:
    basis: Basis
    context: str = "none"          # none | offset | interaction
    alpha: float = 1.0
    levels: list = field(default_factory=list)
    mu: np.ndarray = None
    W: np.ndarray = None           # [n_features, d]
    feat_names: list = field(default_factory=list)

    # design matrix without intercept
    def _design(self, t, ctx):
        Phi = self.basis(t)
        cols, names = [Phi], [f"phi{i}" for i in range(Phi.shape[1])]
        if self.context != "none":
            ctx = np.asarray(ctx)
            for lv in self.levels[1:]:                       # first level is the reference
                ind = (ctx == lv).astype(float)[:, None]
                cols.append(ind); names.append(f"off:{lv}")
                if self.context == "interaction":
                    cols.append(ind * Phi); names += [f"phi{i}:{lv}" for i in range(Phi.shape[1])]
        return np.hstack(cols), names

    def fit(self, X, t, ctx=None):
        X = np.asarray(X, dtype=np.float64); t = np.asarray(t, dtype=float)
        if self.context != "none":
            self.levels = sorted(set(map(str, ctx))); ctx = np.asarray(list(map(str, ctx)))
        D, self.feat_names = self._design(t, ctx)
        self.mu = X.mean(0); Dm = D.mean(0); Xc = X - self.mu; Dc = D - Dm
        A = Dc.T @ Dc + self.alpha * np.eye(D.shape[1])
        self.W = np.linalg.solve(A, Dc.T @ Xc)
        self.mu = self.mu - Dm @ self.W          # so that predict(D) = mu + D W exactly at the fitted centring
        return self

    def predict(self, t, ctx=None):
        if self.context != "none":
            ctx = np.asarray(list(map(str, ctx)))
        D, _ = self._design(np.asarray(t, dtype=float), ctx)
        return self.mu + D @ self.W

    def curve(self, t, level: Optional[str] = None):
        """Points on the curve f(t) for one context level (reference level if None)."""
        ctx = None if self.context == "none" else np.array([level or self.levels[0]] * len(np.atleast_1d(t)))
        return self.predict(np.atleast_1d(t), ctx)

    # -------------------------------------------------------------- inverse
    def decode(self, X, ctx=None, n_grid: int = 400, n_local: int = 41):
        """Project each row of X onto its context's curve: t* = argmin_t ||x - f(t) - c||.
        Coarse grid over [t_min, t_max], then a per-row local grid of ±1 coarse step. Returns (t*, residual)."""
        X = np.asarray(X, dtype=np.float64)
        grid = np.linspace(self.basis.t_min, self.basis.t_max, n_grid); step = grid[1] - grid[0]
        ctx_arr = None if self.context == "none" else np.asarray(list(map(str, ctx)))
        groups = {None: np.arange(len(X))} if ctx_arr is None else {lv: np.flatnonzero(ctx_arr == lv) for lv in set(ctx_arr)}
        t_best = np.zeros(len(X)); r_best = np.zeros(len(X))

        def nearest(Xr, tvals, lv):
            C = self.curve(tvals, lv)
            d2 = (Xr ** 2).sum(1)[:, None] - 2 * Xr @ C.T + (C ** 2).sum(1)[None, :]
            j = d2.argmin(1); return tvals[j], np.sqrt(np.maximum(d2[np.arange(len(Xr)), j], 0))

        for lv, rows in groups.items():
            Xr = X[rows]
            tb, rb = nearest(Xr, grid, lv)
            local = np.clip(tb[:, None] + np.linspace(-step, step, n_local)[None, :], self.basis.t_min, self.basis.t_max)
            flat = np.unique(local)                                   # evaluate the curve once on the union
            C = self.curve(flat, lv)
            d2 = (Xr ** 2).sum(1)[:, None] - 2 * Xr @ C.T + (C ** 2).sum(1)[None, :]
            # restrict each row to its own window
            lo = np.searchsorted(flat, local[:, 0]); hi = np.searchsorted(flat, local[:, -1], side="right")
            for i in range(len(rows)):
                seg = d2[i, lo[i]:hi[i]]; j = seg.argmin()
                t_best[rows[i]] = flat[lo[i] + j]; r_best[rows[i]] = np.sqrt(max(seg[j], 0))
        return t_best, r_best

    # -------------------------------------------------------------- basis-free geometry
    def geometry(self, t_lo: float, t_hi: float, n: int = 400, level: Optional[str] = None) -> dict:
        t = np.linspace(t_lo, t_hi, n); C = self.curve(t, level)
        dt = t[1] - t[0]
        v = np.gradient(C, dt, axis=0); a = np.gradient(v, dt, axis=0)
        speed = np.linalg.norm(v, axis=1)
        # curvature kappa = |a_perp| / |v|^2 with a_perp = a - (a.v̂) v̂
        vhat = v / speed[:, None]; a_perp = a - (np.sum(a * vhat, 1))[:, None] * vhat
        kappa = np.linalg.norm(a_perp, axis=1) / speed ** 2
        length = float(np.trapezoid(speed, t)); chord = float(np.linalg.norm(C[-1] - C[0]))
        # dimension of the curve: spectrum of the covariance of densely sampled curve points
        Cc = C - C.mean(0); sv = np.linalg.svd(Cc, compute_uv=False); var = sv ** 2 / (sv ** 2).sum(); cum = np.cumsum(var)
        n90, n95, n99 = (int(np.searchsorted(cum, q) + 1) for q in (0.90, 0.95, 0.99))
        # turning angle between tangents at the ends
        cosang = float(np.clip(vhat[0] @ vhat[-1], -1, 1))
        return dict(t=t, speed=speed, kappa=kappa, length=length, chord=chord, tortuosity=length / chord,
                    turn_deg=float(np.degrees(np.arccos(cosang))), curve_var_top=var[:5].tolist(), n90=n90, n95=n95, n99=n99,
                    t_kappa_max=float(t[np.argmax(kappa)]))
