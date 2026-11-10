#!/usr/bin/env python
"""Stakes pilot on existing captures: is the absolute reward magnitude ("stakes": the short option's
amount, 7 levels 1,000–100,000 chosen at random per prompt) (a) a graded quantity in the activations,
(b) a modulator of the horizon geometry, (c) a behavioural factor beyond the reward ratio?

Per cell (layer:position) and population (canonical run; matrix main rows per domain):
  (a) ridge R² (5-fold) for log stakes vs log reward ratio vs log H; |rho(PC_k, log stakes)| k=1..3 of the
      standard PCA; ordinality of the 7 stakes centroids after removing the horizon centroid (Spearman of
      centroid-PC1, split-half reliability of that direction), same for reward ratio and horizon; chord
      length of each centroid path in units of the mean within-(horizon) spread.
  (b) horizon geometry within stakes strata low {1k, 2.5k} / mid {5k, 10k} / high {25k, 50k, 100k}:
      |rho(PC1)| per stratum, angles between strata (centroid PC1, top-2 subspace), chord-length ratio,
      frozen ridge for log H trained on one stratum and tested on another (within-2×, RMSE) vs in-stratum
      CV, and the forward curve model with stratum as context: held-out R² offset vs interaction.
  (c) logistic regression of chose_short on log H, log ratio, log stakes, log delays, order: coefficient of
      log stakes with a bootstrap 90% interval; and the switch horizon (fitted p_short = 0.5) per stratum.

Usage: stakes_pilot.py OUT_DIR [--cells 0.55L:T3,0.92L:T3,0.55L:R0,0.72L:R0] [--canonical-run RUN] [--matrix-run RUN]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from ptm.curve import Basis, CurveModel
from ptm.store import RunData
from ptm.depth import resolve_cell, parse_layers

ap = argparse.ArgumentParser()
ap.add_argument("out_dir"); ap.add_argument("--cells", default="0.55L:T3,0.92L:T3,0.55L:R0,0.72L:R0")
ap.add_argument("--canonical-run", default="runs/qwen3-14b_investment_n2000_s0"); ap.add_argument("--matrix-run", default="runs/qwen3-14b_matrix_s0")
ap.add_argument("--boot", type=int, default=200)
a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
LOG2 = np.log10(2)
STRATA = {"low": (1000, 2500), "mid": (5000, 10000), "high": (25000, 50000, 100000)}


def rho(x, y): return abs(float(spearmanr(x, y).statistic))


def centroids(X, lab):
    lv = np.unique(lab); return lv, np.stack([X[lab == v].mean(0) for v in lv])


def centroid_ordinality(X, lab, rng):
    """Spearman of centroid-PC1 with the level value; split-half reliability = |cos| between the PC1
    directions fitted on two random halves of the rows."""
    lv, C = centroids(X, lab); Cc = C - C.mean(0); Vt = np.linalg.svd(Cc, full_matrices=False)[2]
    r = rho(Cc @ Vt[0], lv)
    h = rng.permutation(len(X)) < len(X) // 2
    d = []
    for m in (h, ~h):
        _, Ch = centroids(X[m], lab[m]); Ch = Ch - Ch.mean(0); d.append(np.linalg.svd(Ch, full_matrices=False)[2][0])
    return r, abs(float(d[0] @ d[1])), Vt[0], C


def chord(C): return float(np.linalg.norm(C[-1] - C[0]))


def populations():
    r = RunData(a.canonical_run); df = r.index.reset_index(drop=True)
    yield "canonical investment", r, df, np.ones(len(df), bool)
    m = RunData(a.matrix_run); mdf = m.index.reset_index(drop=True)
    for dom in ["investment", "climate"]:
        yield f"matrix main {dom}", m, mdf, ((mdf.condition == "main") & (mdf.domain == dom)).to_numpy()


rows = []
for pname, run, df, base in populations():
    t = np.log10(df.horizon_years.to_numpy(dtype=float)); ls = np.log10(df.short_reward.to_numpy(dtype=float))
    lr = np.log10((df.long_reward / df.short_reward).to_numpy(dtype=float))
    strat = np.full(len(df), "", dtype=object)
    for k, vals in STRATA.items(): strat[np.isin(df.short_reward.to_numpy(), vals)] = k
    print(f"\n################ {pname}: n={base.sum()}, stakes levels {sorted(set(df.short_reward[base]))}")
    # ---- (c) behaviour (cell-independent)
    ok = base & df.chose_short.notna().to_numpy() & ~np.isnan(t)
    P = np.column_stack([t, lr, ls, np.log10(df.short_delay_years), np.log10(df.long_delay_years), df.short_first.astype(float)])[ok]
    y = df.chose_short[ok].astype(int).to_numpy(); names = ["log H", "log ratio", "log stakes", "log short delay", "log long delay", "short first"]
    sc = StandardScaler().fit(P); lg = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(P), y)
    rng = np.random.default_rng(0); boots = []
    for _ in range(a.boot):
        i = rng.integers(0, len(y), len(y)); boots.append(LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(P[i]), y[i]).coef_[0])
    B = np.array(boots); lo, hi = np.percentile(B, 5, axis=0), np.percentile(B, 95, axis=0)
    print("behaviour: standardised logistic coefficients for chose_short (90% bootstrap interval):")
    for j, nm in enumerate(names): print(f"   {nm:16s} {lg.coef_[0][j]:+.3f}  [{lo[j]:+.3f}, {hi[j]:+.3f}]")
    for k in STRATA:
        m = ok & (strat == k); tt = t[m]; yy = df.chose_short[m].astype(int).to_numpy()
        l1 = LogisticRegression(C=10, max_iter=5000).fit(tt[:, None], yy); sw = -l1.intercept_[0] / l1.coef_[0][0]
        print(f"   stratum {k:4s} n={m.sum():4d}: P(short) {yy.mean():.2f}; fitted switch horizon 10^{sw:.2f} = {10**sw:.2f} y")
    sys.stdout.flush()
    for cell in a.cells.split(","):
        layer, pos = resolve_cell(cell, run)
        valid = base & run.valid_mask(pos) & ~np.isnan(t); X = run.get(layer, pos).astype(np.float64)[valid]
        tt, ss, rr, st = t[valid], ls[valid], lr[valid], strat[valid]; rng = np.random.default_rng(0)
        rec = dict(population=pname, layer=layer, position=pos, n=int(valid.sum()))
        # ---- (a) stakes as a graded quantity
        for nm, yv in [("H", tt), ("stakes", ss), ("ratio", rr)]:
            rec[f"ridge_r2_{nm}"] = float(cross_val_score(Ridge(1.0), X, yv, cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean())
        Z = PCA(10, svd_solver="randomized", random_state=0).fit_transform(X)
        rec["rho_pc_stakes"] = [rho(Z[:, k], ss) for k in range(3)]; rec["rho_pc_H"] = [rho(Z[:, k], tt) for k in range(3)]
        _, CH = centroids(X, tt); Xh = X - CH[np.searchsorted(np.unique(tt), tt)]           # remove the horizon centroid
        spread = float(np.linalg.norm(Xh, axis=1).mean())
        r_s, rel_s, d_s, Cs = centroid_ordinality(Xh, ss, rng); r_r, rel_r, d_r, Cr = centroid_ordinality(Xh, rr, rng); r_h, rel_h, d_h, Ch = centroid_ordinality(X, tt, rng)
        rec.update(cent_rho_stakes=r_s, rel_stakes=rel_s, cent_rho_ratio=r_r, rel_ratio=rel_r, cent_rho_H=r_h, rel_H=rel_h,
                   chord_stakes=chord(Cs) / spread, chord_ratio=chord(Cr) / spread, chord_H=chord(Ch) / spread, angle_stakes_H=float(np.degrees(np.arccos(abs(d_s @ d_h)))))
        print(f"\n=== L{layer} {pos} (n={valid.sum()}) ===")
        print(f"(a) ridge R² 5-fold: log H {rec['ridge_r2_H']:.3f}  log stakes {rec['ridge_r2_stakes']:.3f}  log ratio {rec['ridge_r2_ratio']:.3f};  |rho(PC1-3, stakes)| {np.round(rec['rho_pc_stakes'], 2).tolist()} vs H {np.round(rec['rho_pc_H'], 2).tolist()}")
        print(f"    centroid ordinality / split-half |cos| / chord (within-spread units): stakes {r_s:.2f} / {rel_s:.2f} / {rec['chord_stakes']:.2f};  ratio {r_r:.2f} / {rel_r:.2f} / {rec['chord_ratio']:.2f};  horizon {r_h:.2f} / {rel_h:.2f} / {rec['chord_H']:.2f};  angle(stakes dir, horizon dir) {rec['angle_stakes_H']:.0f}°")
        # ---- (b) horizon geometry within strata
        dirs, subs, chords, rhos = {}, {}, {}, {}
        for k in STRATA:
            m = st == k; _, C = centroids(X[m], tt[m]); Cc = C - C.mean(0); Vt = np.linalg.svd(Cc, full_matrices=False)[2]
            dirs[k] = Vt[0]; subs[k] = Vt[:2].T; chords[k] = chord(C) / spread; rhos[k] = rho(PCA(1, random_state=0).fit_transform(X[m])[:, 0], tt[m])
        pairs = [("low", "mid"), ("mid", "high"), ("low", "high")]
        ang1 = {f"{p}-{q}": float(np.degrees(np.arccos(abs(dirs[p] @ dirs[q])))) for p, q in pairs}
        ang2 = {f"{p}-{q}": float(np.degrees(np.arccos(np.clip(np.linalg.svd(subs[p].T @ subs[q], compute_uv=False), 0, 1))).max()) for p, q in pairs}
        print(f"(b) per stratum |rho(PC1, H)| " + " ".join(f"{k} {v:.2f}" for k, v in rhos.items()) + ";  chord/spread " + " ".join(f"{k} {v:.2f}" for k, v in chords.items()))
        print(f"    angle between strata horizon directions (PC1 / top-2 subspace): " + "  ".join(f"{k} {ang1[k]:.0f}°/{ang2[k]:.0f}°" for k in ang1))
        tr_line = []
        for p, q in [("low", "high"), ("high", "low")]:
            mp, mq = st == p, st == q; rd = Ridge(1.0).fit(X[mp], tt[mp]); e = rd.predict(X[mq]) - tt[mq]
            cv_in = cross_val_score(Ridge(1.0), X[mq], tt[mq], cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean()
            tr_line.append(f"{p}→{q}: within-2x {np.mean(np.abs(e) <= LOG2):.2f}, RMSE {np.sqrt(np.mean(e**2)):.3f} dec, R² {1 - (e**2).sum() / ((tt[mq] - tt[mq].mean())**2).sum():.3f} (in-{q} CV R² {cv_in:.3f})")
            rec[f"transfer_{p}_{q}_w2x"] = float(np.mean(np.abs(e) <= LOG2))
        print("    frozen ridge for log H across strata: " + "; ".join(tr_line))
        lo_, hi_ = tt.min() - 0.05, tt.max() + 0.05; r2s = {}
        for ctx in ["none", "offset", "interaction"]:
            sc_ = []
            for i_tr, i_te in KFold(5, shuffle=True, random_state=0).split(X):
                mdl = CurveModel(Basis("quad", lo_, hi_), context=ctx, alpha=1.0).fit(X[i_tr], tt[i_tr], st[i_tr]); Xh_ = mdl.predict(tt[i_te], st[i_te])
                sc_.append(1 - ((Xh_ - X[i_te]) ** 2).sum() / ((X[i_te] - X[i_te].mean(0)) ** 2).sum())
            r2s[ctx] = float(np.mean(sc_))
        rec.update({f"curve_{k}": v for k, v in r2s.items()}, **{f"angle1_{k}": v for k, v in ang1.items()}, **{f"rho_{k}": v for k, v in rhos.items()})
        print(f"    quad curve model, stratum as context, 5-fold R²: none {r2s['none']:.4f}  offset {r2s['offset']:.4f}  interaction {r2s['interaction']:.4f}")
        rows.append(rec); sys.stdout.flush()
pd.DataFrame(rows).to_csv(out / "stakes_pilot.csv", index=False)
print("\nwrote", out)
