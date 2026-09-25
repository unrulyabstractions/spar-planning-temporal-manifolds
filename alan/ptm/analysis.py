"""Geometry and behavior metrics per (layer, position) cell, plus figures.

Ordinality follows the prior work: fit PCA separately in each cell and rank-correlate PC1 with
log10 horizon (null-horizon rows excluded from the fit). The reward ratio in the same prompts is
the specificity control. Behavior is summarised as the silhouette of chose_short in PC space and
as the model's p_short as a function of horizon.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score

from .store import RunData

# Fixed categorical colors for the two behaviors; horizon uses a sequential ramp; null is gray.
COLOR_SHORT = "#1b7f79"
COLOR_LONG = "#d9730d"
COLOR_NULL = "#9a9a9a"
HORIZON_CMAP = "viridis"


@dataclass
class CellMetrics:
    layer: int
    position: str
    n_horizon: int
    rho_pc1_horizon: float
    rho_pc2_horizon: float
    rho_pc3_horizon: float
    rho_pc1_reward_ratio: float
    rho_pc1_short_reward: float
    rho_pc1_short_delay: float
    rho_pc1_long_delay: float
    evr_pc1: float
    evr_pc2: float
    evr_pc3: float
    cum_evr_2: float = np.nan          # cumulative explained variance, top-2 / top-5 / top-10
    cum_evr_5: float = np.nan
    cum_evr_10: float = np.nan
    n_pc_horizon: int = 0              # number of top-10 PCs with |rho(PC_k, log horizon)| > 0.3
    evr_top10: str = ""                # JSON list of the top-10 explained-variance ratios
    rho_top10: str = ""                # JSON list of |rho(PC_k, log horizon)|, k = 1..10
    n_choice: int = 0
    silhouette_choice_pc3: float = np.nan
    rho_pc1_p_short: float = np.nan
    ridge_r2_horizon: float = np.nan


def _rho(a, b) -> float:
    if len(a) < 3 or np.nanstd(b) == 0:
        return np.nan
    return float(spearmanr(a, b, nan_policy="omit").statistic)


def cell_metrics(run: RunData, layer: int, position: str, supervised: bool = False, n_components: int = 10,
                 X: Optional[np.ndarray] = None) -> tuple[CellMetrics, dict]:
    df = run.index
    valid = run.valid_mask(position)
    if X is None:
        X = run.get(layer, position)
    has_h = valid & df["horizon_years"].notna().to_numpy()
    y = np.log10(df.loc[has_h, "horizon_years"].to_numpy(dtype=float))
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0).fit(X[has_h])
    Zh = pca.transform(X[has_h])
    sub = df.loc[has_h]
    rhos = [abs(_rho(Zh[:, k], y)) for k in range(Zh.shape[1])]
    evr = pca.explained_variance_ratio_
    m = dict(
        layer=layer, position=position, n_horizon=int(has_h.sum()),
        rho_pc1_horizon=_rho(Zh[:, 0], y), rho_pc2_horizon=_rho(Zh[:, 1], y), rho_pc3_horizon=_rho(Zh[:, 2], y),
        rho_pc1_reward_ratio=_rho(Zh[:, 0], np.log10(sub["long_reward"] / sub["short_reward"])),
        rho_pc1_short_reward=_rho(Zh[:, 0], np.log10(sub["short_reward"])),
        rho_pc1_short_delay=_rho(Zh[:, 0], np.log10(sub["short_delay_years"])),
        rho_pc1_long_delay=_rho(Zh[:, 0], np.log10(sub["long_delay_years"])),
        evr_pc1=float(pca.explained_variance_ratio_[0]), evr_pc2=float(pca.explained_variance_ratio_[1]),
        evr_pc3=float(pca.explained_variance_ratio_[2]),
        cum_evr_2=float(evr[:2].sum()), cum_evr_5=float(evr[:5].sum()), cum_evr_10=float(evr[:10].sum()),
        n_pc_horizon=int(sum(r > 0.3 for r in rhos if not np.isnan(r))),
        evr_top10=json.dumps([round(float(v), 4) for v in evr[:10]]),
        rho_top10=json.dumps([round(float(r), 3) if not np.isnan(r) else None for r in rhos[:10]]),
    )
    chose = df["chose_short"]
    has_c = valid & chose.notna().to_numpy()
    labels = chose[has_c].astype(bool).to_numpy()
    Zc = pca.transform(X[has_c]) if has_c.any() else np.zeros((0, n_components))
    m["n_choice"] = int(has_c.sum())
    m["silhouette_choice_pc3"] = float(silhouette_score(Zc, labels)) if (labels.size > 2 and 0 < labels.sum() < labels.size) else np.nan
    m["rho_pc1_p_short"] = _rho(Zc[:, 0], df.loc[has_c, "p_short"].to_numpy(dtype=float)) if has_c.any() else np.nan
    if supervised and has_h.sum() >= 20:
        ridge = RidgeCV(alphas=np.logspace(1, 6, 6))
        m["ridge_r2_horizon"] = float(cross_val_score(ridge, X[has_h], y, cv=5, scoring="r2").mean())
    extras = dict(pca=pca, X=X, valid=valid, has_h=has_h, has_c=has_c, y=y)
    return CellMetrics(**m), extras


def sweep(run: RunData, layers: Optional[Iterable[int]] = None, positions: Optional[Iterable[str]] = None,
          supervised: bool = False, log=print) -> pd.DataFrame:
    layers = list(layers) if layers is not None else list(range(run.n_layers + 1))
    positions = list(positions) if positions is not None else list(run.labels)
    rows = []
    for l in layers:                       # layer-outer: one sequential read per layer
        XL = run.get_layer(l)
        for p in positions:
            m, _ = cell_metrics(run, l, p, supervised=supervised, X=XL[:, run.pos_index(p)])
            rows.append(asdict(m))
        best = max((r for r in rows if r["layer"] == l), key=lambda r: abs(r["rho_pc1_horizon"]) if not np.isnan(r["rho_pc1_horizon"]) else -1)
        ev = json.loads(best["evr_top10"]); rh = json.loads(best["rho_top10"])
        log(f"layer {l:2d}: best |rho_pc1| = {abs(best['rho_pc1_horizon']):.3f} at {best['position']}  (reward-ratio rho {best['rho_pc1_reward_ratio']:+.2f}, choice silhouette {best['silhouette_choice_pc3']:.2f}) "
            f"| evr top-5 {' '.join(f'{v:.3f}' for v in ev[:5])} cum10 {best['cum_evr_10']:.2f} | |rho| by PC {' '.join(f'{r:.2f}' if r is not None else ' nan' for r in rh[:6])} | horizon PCs {best['n_pc_horizon']}")
    return pd.DataFrame(rows)


def behavior_by_horizon(run: RunData) -> pd.DataFrame:
    df = run.index.copy()
    df["horizon_bin"] = df["horizon_text"].fillna("none")
    g = df.groupby("horizon_bin", dropna=False).agg(
        horizon_years=("horizon_years", "first"), n=("sample_uid", "size"),
        n_format_ok=("choice", lambda s: s.notna().sum()),
        frac_short=("chose_short", lambda s: s.dropna().astype(float).mean()),
        p_short_mean=("p_short", "mean"),
    ).reset_index()
    return g.sort_values("horizon_years", na_position="last")


# ---------------------------------------------------------------- figures

def _show(tok: str) -> str:
    return tok.replace("\n", "\\n")


def position_token_labels(run: RunData, top: int = 2) -> dict[str, str]:
    """Human-readable token string per position label.

    Transition positions are fixed per template (taken from the first sample). Response positions
    vary, so the `top` most frequent tokens at that index are shown, joined by '|'.
    """
    import json as _json
    out = {}
    trans = _json.loads(run.index["transition_tokens"].iloc[0])
    resp = [_json.loads(v) for v in run.index["response_tokens"]]
    for lab in run.labels:
        k = int(lab[1:])
        if lab.startswith("T"):
            out[lab] = _show(trans[k]) if k < len(trans) else ""
        else:
            vc = pd.Series([r[k] for r in resp if k < len(r)]).value_counts()
            out[lab] = "|".join(_show(t) for t in vc.index[:top]) if len(vc) else ""
    return out


def cell_name(run: RunData, layer: int, position: str) -> str:
    return f"layer {layer}  {position} ({position_token_labels(run)[position]!r})"


def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_cell(run: RunData, layer: int, position: str, out: Path, title: Optional[str] = None) -> Path:
    plt = _import_plt()
    m, ex = cell_metrics(run, layer, position)
    X, valid, has_h = ex["X"], ex["valid"], ex["has_h"]
    Z = ex["pca"].transform(X[valid])
    df = run.index[valid]
    hy = df["horizon_years"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    null = np.isnan(hy)
    ax.scatter(Z[null, 0], Z[null, 1], s=9, c=COLOR_NULL, label="no horizon", linewidths=0)
    sc = ax.scatter(Z[~null, 0], Z[~null, 1], s=9, c=np.log10(hy[~null]), cmap=HORIZON_CMAP, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("log10 horizon (years)")
    ax.set_title(f"horizon   |rho(PC1)| = {abs(m.rho_pc1_horizon):.2f}")
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax = axes[1]
    ch = df["chose_short"]
    for val, col, lab in [(True, COLOR_SHORT, "chose short"), (False, COLOR_LONG, "chose long")]:
        mk = (ch == val).to_numpy()
        ax.scatter(Z[mk, 0], Z[mk, 1], s=9, c=col, label=lab, linewidths=0)
    unk = ch.isna().to_numpy()
    if unk.any():
        ax.scatter(Z[unk, 0], Z[unk, 1], s=9, c=COLOR_NULL, label="no parsed choice", linewidths=0)
    ax.set_title(f"behavior   silhouette = {m.silhouette_choice_pc3:.2f}")
    ax.legend(loc="best", frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title or f"{run.meta['model_name']}  {cell_name(run, layer, position)}", fontsize=10)
    fig.tight_layout()
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_sweep(table: pd.DataFrame, run: RunData, out: Path, metric: str = "rho_pc1_horizon", absolute: bool = True) -> Path:
    plt = _import_plt()
    piv = table.pivot(index="layer", columns="position", values=metric)
    piv = piv[[p for p in run.labels if p in piv.columns]]
    vals = piv.abs() if absolute else piv
    fig, ax = plt.subplots(figsize=(0.7 * len(piv.columns) + 2.5, 0.16 * len(piv.index) + 2.4))
    im = ax.imshow(vals.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1 if "rho" in metric else None, origin="lower")
    toklab = position_token_labels(run)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{c}\n{toklab[c]}" for c in piv.columns], fontsize=7)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_xlabel("position"); ax.set_ylabel("layer (residual stream after l layers)")
    fig.colorbar(im, ax=ax, label=("|" + metric + "|") if absolute else metric)
    ax.set_title(f"{run.meta['model_name']}   T = transition window, R = generated tokens (most frequent shown)", fontsize=8)
    fig.tight_layout()
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_behavior(run: RunData, out: Path) -> Path:
    plt = _import_plt()
    g = behavior_by_horizon(run)
    gh = g[g["horizon_years"].notna()]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(gh["horizon_years"], gh["frac_short"], "-o", color=COLOR_SHORT, ms=5, lw=2, label="fraction chose short")
    ax.plot(gh["horizon_years"], gh["p_short_mean"], "-s", color=COLOR_LONG, ms=4, lw=1.5, label="mean p(short)")
    gn = g[g["horizon_years"].isna()]
    if len(gn):
        ax.axhline(float(gn["frac_short"].iloc[0]), color=COLOR_NULL, ls="--", lw=1, label="no-horizon prompts")
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("stated time horizon (years)"); ax.set_ylabel("short-option preference")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{run.meta['model_name']}: behavior vs horizon", fontsize=10)
    fig.tight_layout()
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    return out
