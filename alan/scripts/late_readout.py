#!/usr/bin/env python
"""Late-readout comparison: boundary and early-response sites vs anchored late sites (M0 = first reasoning
token, E0 = last generated token, MEAN = mean over the reasoning), on the long (anchored) matrix and
family runs.

Per (layer, site):
  transfer   ridge on train configs × 3 renderings × 8 horizons -> within-2x on new scenarios (A) and
             withheld horizons on new scenarios (G)
  relevance  m=0 (no distractor training) and m=5-style augmentation with mention+role: slope / pull /
             within-2x on each of the six families (mention, role from the matrix run; four held-out
             families from the families run), partners = plain main rows of the matrix run
  controls   restatement: does the generated reasoning contain the stated horizon text? accuracy split by
             restated / not; within-choice ordinality (Spearman of ridge prediction with log H inside each
             chose_short class, new scenarios); behavior: does the site's PC/ridge coordinate predict
             p_short beyond a 2nd-order model of the prompt parameters (within-bin, as in step 3)
Usage: late_readout.py MATRIX_LONG_RUN FAMILIES_LONG_RUN OUT_DIR [--layers 0.55L,0.65L,0.72L,0.92L] [--sites T3,T8,R0,R3,M0,E0,MEAN]
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from ptm.relevance import matched_pairs, pair_pull, pair_slope
from ptm.store import RunData
from ptm.depth import resolve_cell, parse_layers

ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("families_run"); ap.add_argument("out_dir")
ap.add_argument("--layers", default="0.55L,0.65L,0.72L,0.92L"); ap.add_argument("--sites", default="T3,T8,R0,R3,M0,E0,MEAN")
a = ap.parse_args()
run = RunData(a.run_dir); df = run.index.reset_index(drop=True); fr = RunData(a.families_run); fdf = fr.index.reset_index(drop=True)
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); LOG2 = np.log10(2)
layers = parse_layers(a.layers, run); sites = a.sites.split(",")
y = np.log10(df.horizon_years.to_numpy(dtype=float)); yf = np.log10(fdf.horizon_years.to_numpy(dtype=float))
main = (df.condition == "main").to_numpy(); heldout = df.horizon_heldout.fillna(False).astype(bool).to_numpy()
tr = main & (df.split == "train").to_numpy() & ~heldout
A = main & (df.split == "test").to_numpy() & ~heldout; G = main & (df.split == "test").to_numpy() & heldout
plain = df[(df.condition == "main") & (df.rendering == "plain")]; lookup = dict(zip(zip(plain.config_id, plain.domain, plain.horizon_text), plain.index))
fam = {}
for c in ["mention", "role"]:
    ci, pi = matched_pairs(df, c); fam[c] = ("m", ci, pi)
for c in sorted(fdf.condition.unique()):
    d = fdf[fdf.condition == c]; keys = list(zip(d.config_id, d.domain, d.horizon_text)); ok = np.array([k in lookup for k in keys])
    fam[c] = ("f", d.index.to_numpy()[ok], np.array([lookup[k] for k, o in zip(keys, ok) if o]))
families = sorted(fam)

# generation stats and restatement
gen_len = df.n_generated.to_numpy(); print(f"matrix long run: n={len(df)}, generated tokens median {np.median(gen_len):.0f}, at cap (160): {np.mean(gen_len >= 160):.3f}; anchors valid: " +
      ", ".join(f"{lab} {run.valid_mask(lab).mean():.3f}" for lab in ["M0", "E0", "MEAN"] if lab in run.labels))
def restates(row):
    if row.horizon_text is None or not isinstance(row.gen_text, str): return False
    return row.horizon_text.lower() in row.gen_text.lower()
df["restates"] = [restates(r) for r in df.itertuples()]
print(f"reasoning restates the stated horizon text: main {df.loc[main, 'restates'].mean():.3f} (structured {df.loc[main & (df.rendering=='structured').to_numpy(), 'restates'].mean():.2f}, plain {df.loc[main & (df.rendering=='plain').to_numpy(), 'restates'].mean():.2f}, varied {df.loc[main & (df.rendering=='varied').to_numpy(), 'restates'].mean():.2f})")
sys.stdout.flush()

rows = []
print(f"\n{'cell':10s} {'A w2x':>6s} {'G w2x':>6s} {'w2x A restate/not':>18s} {'rho|short rho|long':>18s} | m=0: mean slope / w2x over 6 families | +mention/role: slope / w2x on 4 held-out | behavior Δlogloss")
for layer in layers:
    XL = run.get_layer(layer); XF = fr.get_layer(layer)
    for site in sites:
        if site not in run.labels: continue
        k = run.pos_index(site); kf = fr.pos_index(site); X = XL[:, k].astype(np.float64); Xf = XF[:, kf].astype(np.float64)
        v = run.valid_mask(site); vf = fr.valid_mask(site)
        trm = tr & v; Am = A & v; Gm = G & v
        r0 = Ridge(alpha=1.0).fit(X[trm], y[trm])
        pA = r0.predict(X[Am]); w2A = float(np.mean(np.abs(pA - y[Am]) <= LOG2)); w2G = float(np.mean(np.abs(r0.predict(X[Gm]) - y[Gm]) <= LOG2))
        rs = df.restates.to_numpy()[Am]; w2_re = float(np.mean(np.abs(pA[rs] - y[Am][rs]) <= LOG2)) if rs.any() else np.nan; w2_no = float(np.mean(np.abs(pA[~rs] - y[Am][~rs]) <= LOG2)) if (~rs).any() else np.nan
        ch = df.chose_short.to_numpy()[Am]
        rho_s = spearmanr(pA[ch == True], y[Am][ch == True]).statistic if (ch == True).sum() > 10 else np.nan
        rho_l = spearmanr(pA[ch == False], y[Am][ch == False]).statistic if (ch == False).sum() > 10 else np.nan
        # relevance m=0 and augmented with mention+role (all configs; held-out families never in training)
        def fam_scores(pred_fn):
            res = {}
            for c in families:
                src, ci, pi = fam[c]; Xs, ys, sdf, vs = (X, y, df, v) if src == "m" else (Xf, yf, fdf, vf)
                ok = vs[ci] & v[pi]; ci2, pi2 = ci[ok], pi[ok]
                D = np.log10(sdf.distractor_years[ci2].to_numpy(dtype=float)); H = ys[ci2]
                pc, pp = pred_fn(Xs[ci2]), pred_fn(X[pi2])
                res[c] = (pair_slope(pc, pp, D, H), pair_pull(pc, pp, D, H), float(np.mean(np.abs(pc - H) <= LOG2)))
            return res
        s0 = fam_scores(r0.predict)
        aug_rows = np.concatenate([np.flatnonzero(trm)] + [fam[c][1][v[fam[c][1]]] for c in ["mention", "role"]])
        r1 = Ridge(alpha=1.0).fit(X[aug_rows], y[aug_rows]); s1 = fam_scores(r1.predict)
        held4 = [c for c in families if c not in ("mention", "role")]
        m0_slope, m0_w2 = np.mean([s0[c][0] for c in families]), np.mean([s0[c][2] for c in families])
        m1_slope, m1_w2 = np.mean([s1[c][0] for c in held4]), np.mean([s1[c][2] for c in held4])
        # behavior: within-bin control (params+interactions vs + top-3 PCs of this site), as in step 3, on rows with a choice
        base = main & v & df.chose_short.notna().to_numpy() & df.horizon_years.notna().to_numpy()
        d = df[base]; yb = d.chose_short.astype(int).to_numpy()
        P = np.column_stack([np.log10(d.short_delay_years), np.log10(d.long_delay_years), np.log10(d.short_reward), np.log10(d.long_reward / d.short_reward), d.short_first.astype(float), np.log10(d.horizon_years)])
        from sklearn.decomposition import PCA
        from sklearn.compose import ColumnTransformer
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        def ll(F, use_acts):
            stages = [("params", "passthrough", slice(0, 6))] + ([("pca", PCA(3, random_state=0), slice(6, None))] if use_acts else [])
            pipe = make_pipeline(ColumnTransformer(stages), StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))
            return log_loss(yb, cross_val_predict(pipe, F, yb, cv=cv, method="predict_proba")[:, 1])
        F = np.hstack([P, X[base]]); dll = ll(F, False) - ll(F, True)
        rec = dict(layer=layer, site=site, A_w2x=w2A, G_w2x=w2G, A_w2x_restated=w2_re, A_w2x_not_restated=w2_no, rho_within_short=rho_s, rho_within_long=rho_l,
                   m0_slope=m0_slope, m0_w2x=m0_w2, aug_slope_heldout4=m1_slope, aug_w2x_heldout4=m1_w2, behavior_dlogloss=dll,
                   **{f"m0_{c}_slope": s0[c][0] for c in families}, **{f"m0_{c}_w2x": s0[c][2] for c in families}, **{f"aug_{c}_slope": s1[c][0] for c in families})
        rows.append(rec)
        print(f"L{layer:2d} {site:5s} {w2A:6.2f} {w2G:6.2f} {w2_re:8.2f} / {w2_no:5.2f}   {rho_s:+6.2f} / {rho_l:+5.2f}    | {m0_slope:+.3f} / {m0_w2:.2f}                       | {m1_slope:+.3f} / {m1_w2:.2f}                   | {dll:+.3f}")
        sys.stdout.flush()
t = pd.DataFrame(rows); t.to_csv(out / "late_readout.csv", index=False)
print("\nper-family m=0 slope at each site (layer with best A w2x per site):")
best = t.sort_values("A_w2x", ascending=False).groupby("site").head(1).set_index("site")
print(best[[f"m0_{c}_slope" for c in families]].rename(columns=lambda c: c.replace("m0_", "").replace("_slope", "")).round(2).to_string())
print("wrote", out)
