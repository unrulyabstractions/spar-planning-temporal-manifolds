"""Geometry: PCA sweep against log horizon, the horizon probe, and transfer."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold


def sweep(activations, tokens, has_horizon, log_horizons, n_components=2, verbose=True):
    """|rho|(PC1, log horizon) per (layer, position). Returns rho[layer, pos]
    (nan where the column is constant) and PCA embeddings Z[layer][pos]."""
    n_positions = len(tokens)
    rho = np.full((len(activations), n_positions), np.nan)
    Z_all = [[None] * n_positions for _ in activations]
    if verbose:
        print("\nlayer  " + "  ".join(f"{t!r:>10}" for t in tokens))
    for layer, X in enumerate(activations):
        row = []
        for pos in range(n_positions):
            X_pos = X[:, pos] - X[:, pos].mean(0)
            if np.allclose(X_pos, 0):
                row.append("        --")
                continue
            Z = PCA(n_components=n_components).fit_transform(X_pos)
            rho[layer, pos] = abs(spearmanr(Z[has_horizon, 0], log_horizons)[0])
            Z_all[layer][pos] = Z
            row.append(f"{rho[layer, pos]:>10.3f}")
        if verbose:
            print(f"{layer:>5}  " + "  ".join(row))
    return rho, Z_all


def display_layer(rho):
    """Layer with the best mean |rho| across positions."""
    with np.errstate(invalid="ignore"):
        mean_rho = np.array([np.nanmean(r) if not np.all(np.isnan(r)) else -np.inf for r in rho])
    return int(np.argmax(mean_rho))


ALPHAS = np.logspace(-2, 6, 17)


def _fit(X, y):
    """Ridge with the penalty chosen by leave-one-out CV over ALPHAS. The
    penalty must scale with d_model (4096 on the 8B), so a fixed alpha overfits."""
    return RidgeCV(alphas=ALPHAS).fit(X, y)


def horizon_probe(X, log_h, folds=5, seed=0):
    """Ridge regression from activations [n, d] to log horizon [n].
    Returns (outer-CV R^2, direction unit vector, fitted model on all data)."""
    if len(X) < folds:
        folds = max(2, len(X))
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    preds = np.zeros_like(log_h, dtype=float)
    for tr, te in kf.split(X):
        preds[te] = _fit(X[tr], log_h[tr]).predict(X[te])
    ss_res = ((log_h - preds) ** 2).sum()
    ss_tot = ((log_h - log_h.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    model = _fit(X, log_h)
    d = model.coef_ / (np.linalg.norm(model.coef_) + 1e-12)
    return r2, d, model


def probe_sweep(activations, has_horizon, log_horizons, positions=None):
    """CV R^2 of the horizon probe per (layer, position)."""
    n_positions = activations[0].shape[1]
    positions = range(n_positions) if positions is None else positions
    r2 = np.full((len(activations), n_positions), np.nan)
    for layer, X in enumerate(activations):
        for pos in positions:
            X_pos = X[has_horizon, pos]
            if np.allclose(X_pos - X_pos.mean(0), 0):
                continue
            r2[layer, pos] = horizon_probe(X_pos, log_horizons)[0]
    return r2


def transfer_matrix(banks, log_hs):
    """banks[i] = activations [n_i, d] for template i, log_hs[i] its targets.
    Entry (i, j) = Spearman |rho| of probe trained on i, evaluated on j."""
    k = len(banks)
    M = np.full((k, k), np.nan)
    for i in range(k):
        _, _, model = horizon_probe(banks[i], log_hs[i])
        for j in range(k):
            pred = model.predict(banks[j])
            M[i, j] = abs(spearmanr(pred, log_hs[j])[0])
    return M
