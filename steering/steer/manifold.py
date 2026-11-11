"""The manifold curve: a smooth curve through the per-horizon mean activations, indexed by log-time, and the
edit that slides an activation along it.

Report, Appendix A.3: "Manifold curve: PCA on the per-horizon mean activations (top 3-8 components). We fit a
smooth curve through them, indexed by log-time t. To steer, each token is projected onto the curve (t-hat) and
moved to the curve point at t-hat + Delta (clamped to the observed range), leaving the rest of the activation
unchanged. Delta is measured in decades."

Construction, step by step (fit_manifold):

1. Per-horizon means. Prompts are grouped by the horizon they state (its grid value, so a rewritten "100 weeks"
   groups with "23 months"), and each group's activations are averaged: one point m_j per horizon, at log-time
   t_j = log10(horizon in days).
2. PCA. The means are centred on their average mu, and the top k right singular vectors of the centred means
   form an orthonormal basis U [d, k] of the subspace. A point's coordinates are z = U^T (h - mu).
3. Smooth curve. Each coordinate of the means is regressed on t with a Gaussian-kernel local-linear smoother
   (bandwidth in decades, each mean weighted by its prompt count): c(t) in R^k. Local-linear rather than a
   plain kernel average, so the curve is not pulled inwards at the two ends of the grid. The curve is
   evaluated on a dense grid of t over the observed range [t_min, t_max] and stored as that polyline.
4. Projection. t-hat of an activation is the t of the nearest polyline point to its coordinates z.

The edit (Manifold.steer) has two modes, because the report's sentence admits two readings:

* "snap" (the literal reading, "moved to the curve point"): the subspace part of h is replaced by
  c(t-hat + Delta); the part of h orthogonal to the subspace is unchanged. At Delta = 0 this is NOT the
  identity: it removes the activation's off-curve offset within the subspace.
* "displace": h is moved by c(t-hat + Delta) - c(t-hat), keeping its offset from the curve. Delta = 0 is the
  identity.

Both leave the component orthogonal to the subspace unchanged. Which one the experiments use is open; see the
README.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

STEER_MODES = ("snap", "displace")
GRID_POINTS = 2001          # polyline resolution: about 0.004 decade over the 30 s - 500 y range


def horizon_means(X: np.ndarray, t: np.ndarray, groups: list[str]):
    """(group names, t per group, mean activation per group [m, d] float64, count per group), sorted by t."""
    names = sorted(set(groups), key=lambda g: t[groups.index(g)])
    idx = {g: i for i, g in enumerate(names)}
    rows = np.array([idx[g] for g in groups])
    means = np.zeros((len(names), X.shape[1]))
    np.add.at(means, rows, X.astype(np.float64))
    counts = np.bincount(rows, minlength=len(names))
    tg = np.array([t[groups.index(g)] for g in names], dtype=float)
    return names, tg, means / counts[:, None], counts


def local_linear(t_obs: np.ndarray, Y: np.ndarray, w: np.ndarray, t_eval: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian-kernel local-linear regression of the rows of Y [m, k] on t_obs [m], evaluated at t_eval [e].

    At each t0 it fits Y ~ a + b (t - t0) by weighted least squares, with weights w_i K((t_i - t0) / bandwidth),
    and returns a. Closed form: a = (S2 T0 - S1 T1) / (S0 S2 - S1^2), with S_r = sum W dt^r and T_r = sum W dt^r Y.
    """
    dt = t_obs[None, :] - t_eval[:, None]                              # [e, m]
    W = w[None, :] * np.exp(-0.5 * (dt / bandwidth) ** 2)
    S0, S1, S2 = W.sum(1), (W * dt).sum(1), (W * dt * dt).sum(1)
    T0, T1 = W @ Y, (W * dt) @ Y
    det = S0 * S2 - S1 * S1
    if np.any(det <= 1e-12 * S0 * S0):
        raise ValueError("local-linear fit is degenerate somewhere; bandwidth too small for the horizon spacing?")
    return (S2[:, None] * T0 - S1[:, None] * T1) / det[:, None]


@dataclass
class Manifold:
    mu: np.ndarray            # [d] centre of the per-horizon means
    U: np.ndarray             # [d, k] orthonormal basis of the PCA subspace
    t_grid: np.ndarray        # [G] dense log10(days) grid over the observed range
    curve: np.ndarray         # [G, k] curve coordinates at t_grid
    t_means: np.ndarray       # [m] log-time of each per-horizon mean
    z_means: np.ndarray       # [m, k] coordinates of each per-horizon mean
    counts: np.ndarray        # [m] prompts behind each mean
    bandwidth: float          # smoother bandwidth, decades
    explained: np.ndarray     # [k] share of the means' variance along each component

    @property
    def t_range(self) -> tuple[float, float]:
        return float(self.t_grid[0]), float(self.t_grid[-1])

    def coords(self, H: np.ndarray) -> np.ndarray:
        """Subspace coordinates z = U^T (h - mu) of each row of H."""
        return (H.astype(np.float64) - self.mu) @ self.U

    def _nearest(self, H: np.ndarray) -> np.ndarray:
        Z = self.coords(np.atleast_2d(H))
        d2 = (Z * Z).sum(1)[:, None] - 2 * Z @ self.curve.T + (self.curve * self.curve).sum(1)[None, :]
        return d2.argmin(axis=1)

    def project_t(self, H: np.ndarray) -> np.ndarray:
        """t-hat: the log-time of the curve point nearest to each row of H (in subspace coordinates)."""
        return self.t_grid[self._nearest(H)]

    def point(self, t: np.ndarray) -> np.ndarray:
        """Curve coordinates at log-time(s) t, clamped to the observed range; linear between grid points."""
        t = np.clip(np.atleast_1d(np.asarray(t, dtype=float)), *self.t_range)
        return np.stack([np.interp(t, self.t_grid, self.curve[:, j]) for j in range(self.curve.shape[1])], axis=1)

    def steer(self, H: np.ndarray, delta: float, mode: str) -> np.ndarray:
        """Slide each row of H along the curve by `delta` decades (clamped to the observed range); see the module
        docstring for the two modes. Returns an array of H's shape, float64."""
        if mode not in STEER_MODES:
            raise ValueError(f"mode must be one of {STEER_MODES}")
        H2 = np.atleast_2d(H).astype(np.float64)
        idx = self._nearest(H2)
        target = self.point(self.t_grid[idx] + delta)
        start = self.coords(H2) if mode == "snap" else self.curve[idx]
        return (H2 + (target - start) @ self.U.T).reshape(np.shape(H))

    def save(self, path: Path) -> None:
        np.savez(path, mu=self.mu, U=self.U, t_grid=self.t_grid, curve=self.curve, t_means=self.t_means,
                 z_means=self.z_means, counts=self.counts, bandwidth=self.bandwidth, explained=self.explained)

    @classmethod
    def load(cls, path: Path) -> "Manifold":
        with np.load(path) as f:
            return cls(**{k: f[k] for k in f.files} | {"bandwidth": float(f["bandwidth"])})


def fit_manifold(X: np.ndarray, t: np.ndarray, groups: list[str], k: int, bandwidth: float) -> Manifold:
    """The curve through the per-horizon means of activations X [n, d] (see the module docstring).

    `t` is each prompt's log10(horizon in days); `groups` names its horizon (prompts with the same name are
    averaged together).
    """
    _, tm, means, counts = horizon_means(X, np.asarray(t, dtype=float), list(groups))
    if k >= len(tm):
        raise ValueError(f"{k} components from only {len(tm)} per-horizon means")
    mu = means.mean(axis=0)
    _, s, Vt = np.linalg.svd(means - mu, full_matrices=False)
    U = Vt[:k].T
    z = (means - mu) @ U
    t_grid = np.linspace(tm[0], tm[-1], GRID_POINTS)
    curve = local_linear(tm, z, counts.astype(float), t_grid, bandwidth)
    return Manifold(mu=mu, U=U, t_grid=t_grid, curve=curve, t_means=tm, z_means=z, counts=counts,
                    bandwidth=float(bandwidth), explained=(s[:k] ** 2) / (s ** 2).sum())
