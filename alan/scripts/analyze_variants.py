#!/usr/bin/env python
"""Item 1 analysis: is the horizon coordinate a magnitude representation or a lexical one?

For a run whose prompts vary the *rendering* of the same horizon (units) or the *phrasing* of the
constraint line, at each requested (layer, position) cell:

  1. Cross-variant generalization: ridge trained on the reference rendering (canonical units /
     phrasing 0) only, tested on the other renderings. Reports R^2 against log10 magnitude and,
     for units, against log10 of the written number. A magnitude representation gives high R^2
     on magnitude and transfers across units; a lexical one tracks the written number or fails
     to transfer.
  2. Separation index (units only): mean centroid distance between renderings of the SAME
     canonical horizon, divided by the mean centroid distance between ADJACENT canonical horizons
     in the SAME unit. << 1 means magnitude organizes the space; >> 1 means unit does.
  3. Spearman of PC1 with log magnitude, with the written number, and with unit rank.
  4. A scatter at the cell: PC1-PC2 colored by log magnitude, marker shape by unit / phrasing.

Usage: analyze_variants.py RUN_DIR OUT_DIR --factor unit|phrasing [--cells 22:T3,37:T3,22:R0,29:R0]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from ptm.analysis import cell_metrics, COLOR_NULL, HORIZON_CMAP, _import_plt, cell_name
from ptm.store import RunData

UNIT_RANK = {"hours": 0, "days": 1, "weeks": 2, "months": 3, "years": 4, "decades": 5, "centuries": 6}
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("out_dir"); ap.add_argument("--factor", choices=["unit", "phrasing"], required=True)
ap.add_argument("--cells", default="22:T3,37:T3,22:R0,29:R0")
a = ap.parse_args()
run = RunData(a.run_dir); df = run.index; out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
has_h = df["horizon_years"].notna().to_numpy()
if a.factor == "unit":
    variant = df["horizon_unit"].fillna("none")
    is_ref = (df["horizon_text"] == df["horizon_canonical"]).to_numpy()
    group = df["horizon_canonical"].fillna("none")
else:
    variant = df["phrasing_id"].map(lambda v: f"p{int(v)}" if pd.notna(v) else "none")
    is_ref = (df["phrasing_id"] == 0).to_numpy()
    group = df["horizon_text"].fillna("none")
logmag = np.log10(df["horizon_years"].to_numpy(dtype=float))
lognum = np.log10(df["horizon_value"].to_numpy(dtype=float)) if a.factor == "unit" else None
print(f"run {run.meta['model_name']} n={len(df)} factor={a.factor}; reference rows {int((is_ref & has_h).sum())}, variant rows {int((~is_ref & has_h).sum())}")
print("variants:", variant[has_h].value_counts().to_dict())
rows = []
plt = _import_plt()
for cell in a.cells.split(","):
    layer, pos = cell.split(":"); layer = int(layer)
    m, ex = cell_metrics(run, layer, pos)
    X = ex["X"]; valid = ex["valid"]
    tr = valid & has_h & is_ref; te = valid & has_h & ~is_ref
    ridge = RidgeCV(alphas=np.logspace(1, 6, 6)).fit(X[tr], logmag[tr])
    pred = ridge.predict(X[te])
    r2_mag = r2_score(logmag[te], pred)
    rec = dict(layer=layer, position=pos, n_train=int(tr.sum()), n_test=int(te.sum()), r2_transfer_magnitude=r2_mag,
               rho_pc1_magnitude=float(spearmanr(ex["pca"].transform(X[valid & has_h])[:, 0], logmag[valid & has_h]).statistic))
    line = f"L{layer:2d} {pos}: ridge(ref->variants) R^2 vs magnitude {r2_mag:.3f}"
    if a.factor == "unit":
        r2_num = r2_score(lognum[te], pred)
        # per-unit residual: does the prediction shift systematically with unit at fixed magnitude?
        resid = pd.DataFrame(dict(unit=variant[te].to_numpy(), resid=pred - logmag[te])).groupby("unit")["resid"].mean().round(3).to_dict()
        Zh = ex["pca"].transform(X[valid & has_h])
        rho_num = float(spearmanr(Zh[:, 0], lognum[valid & has_h]).statistic)
        rho_unit = float(spearmanr(Zh[:, 0], variant[valid & has_h].map(UNIT_RANK)).statistic)
        # separation index in the top-10 PC space (fit on all horizon rows)
        from sklearn.decomposition import PCA
        Z10 = PCA(10, random_state=0).fit_transform(X[valid & has_h])
        d = df[valid & has_h].assign(z=list(Z10))
        cent = d.groupby(["horizon_canonical", "horizon_text"])["z"].apply(lambda s: np.mean(np.stack(s), 0))
        within = []
        for canon, sub in cent.groupby(level=0):
            vs = list(sub.values)
            within += [np.linalg.norm(vs[i] - vs[j]) for i in range(len(vs)) for j in range(i + 1, len(vs))]
        ref_cent = d[d.horizon_text == d.horizon_canonical].groupby("horizon_canonical")["z"].apply(lambda s: np.mean(np.stack(s), 0))
        order = sorted(ref_cent.index, key=lambda h: d.loc[d.horizon_canonical == h, "horizon_years"].iloc[0])
        between = [np.linalg.norm(ref_cent[order[i]] - ref_cent[order[i + 1]]) for i in range(len(order) - 1)]
        sep = float(np.mean(within) / np.mean(between))
        rec.update(r2_transfer_written_number=r2_num, rho_pc1_written_number=rho_num, rho_pc1_unit_rank=rho_unit,
                   sep_index_same_horizon_between_units_over_adjacent_horizons=sep, resid_by_unit=str(resid))
        line += f" | vs written number {r2_num:.3f} | rho_pc1: magnitude {rec['rho_pc1_magnitude']:+.3f} number {rho_num:+.3f} unit-rank {rho_unit:+.3f} | sep index {sep:.2f} | pred-true by unit {resid}"
    else:
        per = pd.DataFrame(dict(v=variant[te].to_numpy(), t=logmag[te], p=pred)).groupby("v").apply(lambda g: r2_score(g.t, g.p)).round(3).to_dict()
        rec.update(r2_by_phrasing=str(per)); line += f" | R^2 by phrasing {per}"
    print(line); sys.stdout.flush(); rows.append(rec)
    # figure
    Z = ex["pca"].transform(X[valid])
    dv = df[valid]; vv = variant[valid].to_numpy(); hy = dv["horizon_years"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    null = np.isnan(hy)
    ax.scatter(Z[null, 0], Z[null, 1], s=10, c=COLOR_NULL, label="no horizon", linewidths=0)
    keys = sorted(set(vv[~null]), key=lambda u: UNIT_RANK.get(u, 0) if a.factor == "unit" else u)
    for k, mk in zip(keys, MARKERS):
        sel = (~null) & (vv == k)
        sc = ax.scatter(Z[sel, 0], Z[sel, 1], s=16, c=np.log10(hy[sel]), cmap=HORIZON_CMAP, marker=mk, vmin=np.nanmin(np.log10(hy)), vmax=np.nanmax(np.log10(hy)), linewidths=0.3, edgecolors="k", label=k)
    fig.colorbar(sc, ax=ax, pad=0.01).set_label("log10 horizon (years)")
    ax.legend(frameon=False, fontsize=8, title=a.factor); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{run.meta['model_name']}  {cell_name(run, layer, pos)}: color = magnitude, marker = {a.factor}", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"variants_{a.factor}_L{layer:02d}_{pos}.png", dpi=150); plt.close(fig)
pd.DataFrame(rows).to_csv(out / f"variants_{a.factor}.csv", index=False)
print("wrote", out)
