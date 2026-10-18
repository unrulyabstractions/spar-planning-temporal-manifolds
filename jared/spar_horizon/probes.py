"""E7: linear and MLP probes for log horizon on the top-k principal components,
with leave-one-horizon-out cross-validation (spec: 04_spec_probes.md).

Every prompt at one horizon value is held out together, so a probe only scores
by placing an unseen horizon between the ones it was trained on. Dimensions
are ordered two ways, both fitted on the training fold only:
- "act":  PCA of the activations, ordered by total variance;
- "cent": PCA of the per-horizon mean activations, ordered by how much a
  direction separates horizons. At most (training horizons - 1) components.
"""

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

from .geometry import ALPHAS, _fit

KS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
ORDERS = ("act", "cent")
PROBES = ("linear", "mlp")
MLP_HIDDEN = 32
MLP_ALPHA = 1.0   # L2 penalty on standardized inputs; chosen by the sanity sweep recorded in 04_spec_probes.md
INNER_FOLDS = 5


def loho_folds(groups):
    """(train, test) index pairs, one per distinct value of `groups`."""
    return [(np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in np.unique(groups)]


def fold_bases(X_train, groups_train, kmax):
    """Training-fold mean and the two orthonormal bases [k, d] ("act", "cent")."""
    mean = X_train.mean(0)
    Xc = X_train - mean
    centroids = np.stack([Xc[groups_train == g].mean(0) for g in np.unique(groups_train)])
    # Template tokens at layer 0 (the embeddings) are identical in every prompt, so the total
    # variance is 0 and PCA's explained-variance ratio is 0/0. The cell is then correctly
    # unreadable; the warning is noise.
    with np.errstate(invalid="ignore", divide="ignore"):
        act = PCA(n_components=min(kmax, len(Xc) - 1)).fit(Xc).components_
        cent = PCA(n_components=min(kmax, len(centroids) - 1)).fit(centroids).components_
    return mean, {"act": act, "cent": cent}


def fit_ridge(Z_train, y_train, groups_train=None):
    """Ridge with the penalty chosen by inner CV. With `groups_train` the inner CV
    holds out whole horizons, as the outer one does; leave-one-prompt-out (the
    fallback) keeps same-horizon prompts in training and picks too small a penalty
    for a held-out horizon."""
    n_groups = 0 if groups_train is None else len(np.unique(groups_train))
    if n_groups < 2:
        return _fit(Z_train, y_train)
    splits = list(GroupKFold(n_splits=min(INNER_FOLDS, n_groups)).split(Z_train, y_train, groups_train))
    return RidgeCV(alphas=ALPHAS, cv=splits).fit(Z_train, y_train)


def fit_predict(probe, Z_train, y_train, Z_test, groups_train=None):
    """Predictions on Z_test. The ridge works on the raw components (its penalty
    is chosen by inner CV, see `fit_ridge`); the MLP on components and target
    standardized with training-fold statistics."""
    if probe == "linear":
        return fit_ridge(Z_train, y_train, groups_train).predict(Z_test)
    z_mean, z_std = Z_train.mean(0), Z_train.std(0) + 1e-8
    y_mean, y_std = y_train.mean(), y_train.std()
    mlp = MLPRegressor(hidden_layer_sizes=(MLP_HIDDEN,), activation="tanh", solver="lbfgs",
                       alpha=MLP_ALPHA, max_iter=2000, random_state=0)
    with warnings.catch_warnings():
        # L-BFGS hitting max_iter is common on noisy cells and harmless at alpha 1; it floods the log
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp.fit((Z_train - z_mean) / z_std, (y_train - y_mean) / y_std)
    return mlp.predict((Z_test - z_mean) / z_std) * y_std + y_mean


def r2(y, pred, mask=None):
    """Coefficient of determination of pooled out-of-fold predictions."""
    if mask is not None:
        y, pred = y[mask], pred[mask]
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def within_unit_r2(y, pred, units, mask=None):
    """R² after removing each unit word's mean from the target and from the
    predictions: how well a probe orders and scales horizons that share a unit
    ("3 hours" vs "12 hours"). A probe that only knows the unit word predicts a
    constant within each unit and scores 0."""
    if mask is not None:
        y, pred, units = y[mask], pred[mask], units[mask]
    y_dev, pred_dev = y.astype(float), pred.astype(float)
    for unit in np.unique(units):
        in_unit = units == unit
        y_dev[in_unit] = y[in_unit] - y[in_unit].mean()
        pred_dev[in_unit] = pred[in_unit] - pred[in_unit].mean()
    return float(1 - ((y_dev - pred_dev) ** 2).sum() / (y_dev ** 2).sum())


def probe_curves(X, y, units=None, ks=KS, orders=ORDERS, probes=PROBES, ceiling=True):
    """Leave-one-horizon-out R² for every (probe, order, k), on all prompts and
    on interior horizons only. Holding out one of the two extreme horizons is
    extrapolation, where a tanh MLP saturates by construction, so interior R²
    is the fair comparison between probes. X: [n, d], y: [n] log horizon,
    units: [n] unit word of each horizon, or None. k beyond a basis' size is
    NaN. Returns {"r2": {probe: {order: [per k]}}, "r2_interior": same,
    "ceiling", "ceiling_interior": full-dim ridge R², or None without
    `ceiling`}; with `units` also "r2_within" and "ceiling_within"
    (`within_unit_r2` on interior horizons). Single-threaded: the problems are
    tiny and parallelism belongs one level up, across cells."""
    with threadpool_limits(1):
        return _probe_curves(X, y, units, ks, orders, probes, ceiling)


def _probe_curves(X, y, units, ks, orders, probes, ceiling):
    groups = np.round(y, 9)
    interior = (groups != groups.min()) & (groups != groups.max())
    preds = {(p, o, k): np.full(len(y), np.nan) for p in probes for o in orders for k in ks}
    full = np.full(len(y), np.nan)
    for train, test in loho_folds(groups):
        mean, bases = fold_bases(X[train], groups[train], max(ks))
        if ceiling:
            full[test] = fit_ridge(X[train] - mean, y[train], groups[train]).predict(X[test] - mean)
        for order in orders:
            basis = bases[order]
            Z_train, Z_test = (X[train] - mean) @ basis.T, (X[test] - mean) @ basis.T
            for k in ks:
                if k > len(basis):
                    continue
                for probe in probes:
                    preds[(probe, order, k)][test] = fit_predict(probe, Z_train[:, :k], y[train], Z_test[:, :k],
                                                                 groups[train])

    def table(score):
        return {p: {o: [score(preds[(p, o, k)]) if not np.isnan(preds[(p, o, k)]).any() else float("nan")
                        for k in ks] for o in orders} for p in probes}
    out = {"r2": table(lambda pred: r2(y, pred)), "r2_interior": table(lambda pred: r2(y, pred, interior)),
           "ceiling": r2(y, full) if ceiling else None,
           "ceiling_interior": r2(y, full, interior) if ceiling else None}
    if units is not None:
        out["r2_within"] = table(lambda pred: within_unit_r2(y, pred, units, interior))
        out["ceiling_within"] = within_unit_r2(y, full, units, interior) if ceiling else None
    return out


def permuted_horizons(y, seed=0):
    """y with the horizon values shuffled across horizon groups: every prompt of
    one horizon gets the same wrong horizon. The control for selectivity."""
    groups = np.round(y, 9)
    values = np.unique(groups)
    mapping = dict(zip(values, np.random.default_rng(seed).permutation(values)))
    return np.array([mapping[g] for g in groups])


def k_to_reach(curve, target, ks=KS):
    """Smallest k whose R² reaches `target`, or None."""
    for k, value in zip(ks, curve):
        if not np.isnan(value) and value >= target:
            return k
    return None


def analyze_cell(X, y, units=None):
    """Curves, permuted-label control (act ordering, no ceiling), and k90 per
    probe and ordering: the smallest k whose interior R² reaches 90% of the
    best interior R² either probe (or the ceiling) reaches at this cell; None
    when nothing is readable (best below 0.1). With `units`, the curves also
    carry within-unit R² (see `probe_curves`)."""
    result = probe_curves(X, y, units)
    result["control"] = probe_curves(X, permuted_horizons(y), orders=("act",), ceiling=False)["r2_interior"]
    curves = [v for p in PROBES for o in ORDERS for v in result["r2_interior"][p][o] if not np.isnan(v)]
    best = max(curves + [result["ceiling_interior"]])
    result["best"] = best
    result["k90"] = {p: {o: (k_to_reach(result["r2_interior"][p][o], 0.9 * best) if best >= 0.1 else None)
                         for o in ORDERS} for p in PROBES}
    return result
