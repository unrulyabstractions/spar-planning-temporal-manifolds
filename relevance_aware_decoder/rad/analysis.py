"""Metrics for the relevance-aware decoder experiment.

Primary metric: the PAIRED PULL FRACTION. For a distractor prompt and its clean twin (same
configuration, domain and horizon, no extra sentence):
    delta = prediction(distractor prompt) - prediction(twin)          [log10 years]
    gap   = log10(D) - log10(H)
    pull  = OLS slope of delta on gap (with intercept)
pull = 0: the readout ignores the distractor; pull = 1: it moves all the way to the distractor.
Pairing removes everything the two prompts share, including the decoder's ordinary horizon-dependent
error. That matters because gap = log D - log H is itself correlated with H, so Alan's unpaired metric
spearman(pred - H, gap) can be non-zero from regression-to-the-mean alone; we report it too, for
continuity with his numbers.

Behavioral counterpart, on the model's fp32 choice logits:
    lo            = log-odds of choosing the short option
    delta_lo      = lo(distractor prompt) - lo(twin)
    beta_H        = slope of lo on log10 H over clean prompts (how much a decade of horizon moves the choice)
    behavior pull = slope(delta_lo on gap) / beta_H
On the same scale as the readout pull: the fraction of the way the MODEL's choice moves toward the distractor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LOG2 = float(np.log10(2))


# ------------------------------------------------------------------ ridge with many alphas from one SVD

class RidgePath:
    """Ridge regression (sklearn's objective: ||y - Xw - b||^2 + alpha ||w||^2) for several alphas from one
    thin SVD of the centred design. device: "auto" = torch on CUDA if available, else numpy float64;
    "numpy" = numpy float64; anything else = that torch device (float32)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, device: str = "auto"):
        self.x_mean = X.mean(0)
        self.y_mean = float(y.mean())
        Xc = X - self.x_mean
        yc = y - self.y_mean
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "numpy"
            except ImportError:
                device = "numpy"
        if device == "numpy":
            U, self.s, self.Vt = np.linalg.svd(Xc.astype(np.float64), full_matrices=False)
            self.uty = U.T @ yc
        else:
            import torch
            U, s, Vt = torch.linalg.svd(torch.as_tensor(Xc, dtype=torch.float32, device=device), full_matrices=False)
            uty = U.T @ torch.as_tensor(yc, dtype=torch.float32, device=device)
            self.s, self.uty, self.Vt = (t.double().cpu().numpy() for t in (s, uty, Vt))

    def coef(self, alpha: float) -> tuple[np.ndarray, float]:
        w = self.Vt.T @ (self.s / (self.s ** 2 + alpha) * self.uty)
        return w, self.y_mean - float(self.x_mean @ w)

    def predict(self, X: np.ndarray, alpha: float) -> np.ndarray:
        w, b = self.coef(alpha)
        return X @ w + b


# ------------------------------------------------------------------ accuracy

def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def accuracy(pred: np.ndarray, y: np.ndarray) -> dict:
    err = pred - y
    return dict(n=int(len(y)), rmse=float(np.sqrt(np.mean(err ** 2))), bias=float(err.mean()),
                within2x=float(np.mean(np.abs(err) <= LOG2)),
                rho=float(spearmanr(pred, y).statistic) if len(y) > 2 and np.std(y) > 0 else np.nan)


# ------------------------------------------------------------------ pull

def _slope(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    denom = float(x @ x)
    return float(x @ (y - y.mean()) / denom) if denom > 0 else np.nan


def cluster_bootstrap(stat, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of stat(idx) when resampling whole clusters (scenarios) with replacement."""
    rng = np.random.default_rng(seed)
    ids = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in ids}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[c] for c in rng.choice(ids, size=len(ids), replace=True)])
        vals.append(stat(idx))
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (np.nan, np.nan)


def pull_stats(pred_d: np.ndarray, pred_twin: np.ndarray, log_h: np.ndarray, log_d: np.ndarray,
               clusters: np.ndarray, n_boot: int = 2000) -> dict:
    delta = pred_d - pred_twin
    gap = log_d - log_h
    stat = lambda idx: _slope(gap[idx], delta[idx])
    lo, hi = cluster_bootstrap(stat, clusters, n_boot) if n_boot else (np.nan, np.nan)
    return dict(n=int(len(delta)), pull=stat(np.arange(len(delta))), pull_lo=lo, pull_hi=hi,
                shift=float(delta.mean()),                                               # constant offset from the extra sentence
                rho_paired=float(spearmanr(delta, gap).statistic),
                rho_resid_unpaired=float(spearmanr(pred_d - log_h, gap).statistic),      # Alan's metric
                within2x=float(np.mean(np.abs(pred_d - log_h) <= LOG2)),
                within2x_twin=float(np.mean(np.abs(pred_twin - log_h) <= LOG2)))


# ------------------------------------------------------------------ behavior

def short_log_odds(df: pd.DataFrame) -> np.ndarray:
    """fp32 log-odds of the short option from the two label logits; NaN where the format was not followed."""
    la, lb = df.logit_a.to_numpy(float), df.logit_b.to_numpy(float)
    sf = df.short_first.to_numpy(bool)
    lo = np.where(sf, la - lb, lb - la)
    lo[df.choice.isna().to_numpy()] = np.nan
    return lo


def horizon_effect(clean: pd.DataFrame) -> float:
    """beta_H: slope of short log-odds on log10 H over clean prompts (negative: longer horizon, fewer short choices)."""
    lo = short_log_odds(clean); lh = np.log10(clean.horizon_years.to_numpy(float))
    ok = ~np.isnan(lo)
    return _slope(lh[ok], lo[ok])


def behavior_pull(dist: pd.DataFrame, twins: pd.DataFrame, beta_h: float, n_boot: int = 2000) -> dict:
    """dist and twins row-aligned (twins.iloc[i] is the clean twin of dist.iloc[i])."""
    d_lo, t_lo = short_log_odds(dist), short_log_odds(twins)
    gap = dist.log_gap.to_numpy(float)
    ok = ~(np.isnan(d_lo) | np.isnan(t_lo))
    delta, gap, cl = (d_lo - t_lo)[ok], gap[ok], dist.scenario_id.to_numpy()[ok]
    stat = lambda idx: _slope(gap[idx], delta[idx]) / beta_h
    lo, hi = cluster_bootstrap(stat, cl, n_boot) if n_boot else (np.nan, np.nan)
    flips = float(np.mean(dist.chose_short.to_numpy()[ok] != twins.chose_short.to_numpy()[ok]))
    return dict(n=int(ok.sum()), behavior_pull=stat(np.arange(ok.sum())), behavior_pull_lo=lo, behavior_pull_hi=hi,
                mean_delta_log_odds=float(delta.mean()), choice_flip_rate=flips,
                rho_delta_gap=float(spearmanr(delta, gap).statistic))


# ------------------------------------------------------------------ text baselines (same definitions as alan/scripts/transfer_eval.py)

import re  # noqa: E402

from ptm.horizons import Horizon  # noqa: E402

DUR = re.compile(r"(\d[\d,]*)\s+(minutes?|hours?|days?|weeks?|months?|years?|decades?|century|centuries)\b")


def regex_first(t: str) -> float:
    m = DUR.search(t)
    return np.log10(Horizon(float(m.group(1).replace(",", "")), m.group(2)).years) if m else np.nan


def regex_horizon(t: str) -> float:
    for sent in re.split(r"(?<=[.!?])\s+", t):
        if "horizon" in sent.lower():
            m = DUR.search(sent)
            if m:
                return np.log10(Horizon(float(m.group(1).replace(",", "")), m.group(2)).years)
    return regex_first(t)
