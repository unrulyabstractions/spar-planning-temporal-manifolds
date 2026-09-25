#!/usr/bin/env python
"""Item 4 control: does the PC coordinate predict the choice beyond the prompt's own parameters?

Features available to the model in the prompt: log short delay, log long delay, log short reward,
log reward ratio, option order, log stated horizon. We fit 5-fold logistic regressions for
chose_short with (a) parameters, (b) parameters + pairwise interactions, (c) each of those + the
top-3 PCA coordinates of the activations at a chosen (layer, position), and report held-out
log-loss and AUC, overall and per mixed bin.

PCA is fit INSIDE each training fold (a ColumnTransformer stage of the pipeline), so no test-fold
activation influences the axes used to score it. `--leaky` reproduces the earlier, incorrect
procedure (PCA fit on all horizon rows before cross-validation) for comparison only.
"""
import argparse, sys
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from ptm.analysis import cell_metrics
from ptm.store import RunData

ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--cells", default="22:T3,37:T3,22:R0,29:R0")
ap.add_argument("--leaky", action="store_true", help="fit PCA on all rows before CV (the earlier procedure; for comparison)")
a = ap.parse_args()
run = RunData(a.run_dir); df = run.index
cv = StratifiedKFold(5, shuffle=True, random_state=0)
N_PARAMS = 6


def fit(F, y, degree=1, use_params=True, use_acts=False):
    """F = [6 prompt parameters | activation columns]. PCA(3) on the activation block is fit per fold."""
    stages = []
    if use_params:
        stages.append(("params", "passthrough", slice(0, N_PARAMS)))
    if use_acts:
        stages.append(("pca", PCA(3, random_state=0), slice(N_PARAMS, None)))
    pipe = make_pipeline(ColumnTransformer(stages), StandardScaler(), PolynomialFeatures(degree, include_bias=False),
                         StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))
    p = cross_val_predict(pipe, F, y, cv=cv, method="predict_proba")[:, 1]
    return p, log_loss(y, p), roc_auc_score(y, p)


base = df[df["chose_short"].notna() & df["horizon_years"].notna()].copy()
y = base["chose_short"].astype(int).to_numpy()
P = np.column_stack([np.log10(base["short_delay_years"]), np.log10(base["long_delay_years"]),
                     np.log10(base["short_reward"]), np.log10(base["long_reward"] / base["short_reward"]),
                     base["short_first"].astype(float), np.log10(base["horizon_years"])]).astype(np.float32)
print(f"n={len(base)}  PCA fit {'on ALL rows before CV (leaky)' if a.leaky else 'inside each training fold'}  mixed bins = stated horizons with 10-90% short")
p_b1, ll1, auc1 = fit(P, y, 1); p_b2, ll2, auc2 = fit(P, y, 2)
print(f"prompt parameters, linear      : logloss {ll1:.3f}  AUC {auc1:.3f}")
print(f"prompt parameters + interactions: logloss {ll2:.3f}  AUC {auc2:.3f}")
mixed = base.groupby("horizon_text")["chose_short"].mean().pipe(lambda s: s[(s > 0.1) & (s < 0.9)]).index.tolist()
for cell in a.cells.split(","):
    layer, pos = cell.split(":"); layer = int(layer)
    if a.leaky:
        m, ex = cell_metrics(run, layer, pos)
        A = ex["pca"].transform(ex["X"])[:, :3][base.index.to_numpy()].astype(np.float32)
        F = np.hstack([P, A])           # the 3 precomputed PCs stand in for the activation block
    else:
        A = run.get(layer, pos)[base.index.to_numpy()]
        F = np.hstack([P, A])
    p_z, ll_z, auc_z = fit(F, y, 1, use_params=False, use_acts=True)
    p_p, ll_p, auc_p = fit(F, y, 1, use_acts=True)
    p_i, ll_i, auc_i = fit(F, y, 2, use_acts=True)
    print(f"\n=== L{layer} {pos} ===")
    print(f"PC1-3 alone                     : logloss {ll_z:.3f}  AUC {auc_z:.3f}")
    print(f"params + PC1-3, linear          : logloss {ll_p:.3f}  AUC {auc_p:.3f}   (Δ vs params-linear {ll1-ll_p:+.3f})")
    print(f"params + PC1-3 + interactions   : logloss {ll_i:.3f}  AUC {auc_i:.3f}   (Δ vs params-interactions {ll2-ll_i:+.3f})")
    print("per mixed bin: horizon, n, logloss params+inter -> params+inter+PC (Δ), AUC")
    for h in sorted(mixed, key=lambda h: base.loc[base.horizon_text == h, "horizon_years"].iloc[0]):
        mk = (base["horizon_text"] == h).to_numpy(); yy = y[mk]
        if 0 < yy.sum() < len(yy):
            l0, l1 = log_loss(yy, p_b2[mk], labels=[0, 1]), log_loss(yy, p_i[mk], labels=[0, 1])
            print(f"   {h:>9}  n={mk.sum():3d}  {l0:.3f} -> {l1:.3f} ({l0-l1:+.3f})  AUC {roc_auc_score(yy, p_b2[mk]):.2f} -> {roc_auc_score(yy, p_i[mk]):.2f}")
    sys.stdout.flush()
