"""Relevance-aware decoding: estimate the "mentioned duration" nuisance subspace from matched pairs
(prompt with an irrelevant duration minus the same prompt without it) and project it out before ridge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def matched_pairs(df: pd.DataFrame, condition: str, base_rendering: str = "plain") -> tuple[np.ndarray, np.ndarray]:
    """Row indices (control_rows, partner_rows) pairing each `condition` row with the main-matrix row of
    the same config, domain, horizon and `base_rendering` (the distractor-free counterpart)."""
    main = df[(df.condition == "main") & (df.rendering == base_rendering)]
    key = main.set_index(["config_id", "domain", "horizon_text"]).index
    lookup = dict(zip(key, main.index))
    ctl = df[df.condition == condition]
    keys = list(zip(ctl.config_id, ctl.domain, ctl.horizon_text))
    ok = [k in lookup for k in keys]
    return ctl.index.to_numpy()[ok], np.array([lookup[k] for k, o in zip(keys, ok) if o])


def nuisance_subspace(diffs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k right singular vectors of the (uncentred) difference matrix, and the fraction of the
    differences' energy they capture. Uncentred so a consistent mean displacement is the first direction."""
    U, s, Vt = np.linalg.svd(diffs, full_matrices=False)
    energy = s ** 2 / (s ** 2).sum()
    return Vt[:k].T, np.cumsum(energy)[:k]


def project_out(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Remove the components of X along the orthonormal columns of V."""
    return X - (X @ V) @ V.T


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Ridge:
    return Ridge(alpha=alpha).fit(X, y)


def principal_angle_to_subspace(v: np.ndarray, B: np.ndarray) -> float:
    """Angle in degrees between unit vector v and the subspace spanned by the orthonormal columns of B."""
    v = v / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(np.linalg.norm(B.T @ v), 0, 1))))


def pair_pull(pred_ctl: np.ndarray, pred_partner: np.ndarray, D: np.ndarray, H: np.ndarray) -> float:
    """Distractor pull free of regression-to-the-mean: Spearman correlation between the within-pair change
    in the prediction (distractor prompt minus its distractor-free partner) and log D − log H."""
    from scipy.stats import spearmanr
    return float(spearmanr(pred_ctl - pred_partner, D - H).statistic)


def pair_slope(pred_ctl: np.ndarray, pred_partner: np.ndarray, D: np.ndarray, H: np.ndarray) -> float:
    """Magnitude of the pull: least-squares slope of the within-pair prediction change on (log D − log H),
    in decades of estimate per decade of distractor offset (1 = the estimate follows the distractor fully)."""
    x = D - H; dy = pred_ctl - pred_partner; xc = x - x.mean()
    return float((xc @ (dy - dy.mean())) / (xc @ xc))
