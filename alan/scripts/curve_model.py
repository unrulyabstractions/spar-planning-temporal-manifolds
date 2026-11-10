#!/usr/bin/env python
"""Step 4: forward geometric model of the horizon manifold on the matched matrix run, validated on
held-out horizons and scenarios, inverted as a decoder against ridge, with basis-free geometry,
scenario bootstrap, an absolute-vs-relative parameterization test, and the distractor decomposition.

Per cell (layer:position):
  models  = {line, quad, spline} × context {none, offset, interaction}, context = rendering|domain
  train   = main, split=train configs, training horizons
  forward = R^2 in activation space on: A new scenarios (train horizons); F withheld horizons (train
            scenarios); G withheld horizons × new scenarios; X extrapolation (extended-grid run, horizons
            outside [1 d, 100 y], context structured|investment)
  inverse = decode t by projection onto the curve; within-2x and RMSE vs a ridge decoder on the same rows
  geometry= tortuosity, end-to-end turning angle, curvature peak location, curve-covariance n90/n95/n99
  bootstrap over training configurations for the selected model
  relative-time test: refit with t = log10(H / short option delay)
  distractor: residuals of mention/role rows decomposed into along-tangent and orthogonal components

Usage: curve_model.py MATRIX_RUN OUT_DIR [--extended-run RUN] [--cells 0.55L:T3,0.65L:T1,0.55L:R0] [--boot 50]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from ptm.curve import Basis, CurveModel
from ptm.store import RunData
from ptm.depth import resolve_cell, parse_layers

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("out_dir"); ap.add_argument("--extended-run", default="runs/qwen3-14b_extended_n2000_s0")
ap.add_argument("--cells", default="0.55L:T3,0.65L:T1,0.55L:R0"); ap.add_argument("--boot", type=int, default=50); ap.add_argument("--alpha", type=float, default=1.0)
a = ap.parse_args()
run = RunData(a.run_dir); df = run.index.reset_index(drop=True); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
ext = RunData(a.extended_run) if a.extended_run and Path(a.extended_run).exists() else None
LOG2 = np.log10(2)
t_all = np.log10(df.horizon_years.to_numpy(dtype=float))
ctx_all = (df.rendering.astype(str) + "|" + df.domain.astype(str)).to_numpy()
main = (df.condition == "main").to_numpy(); heldout = df.horizon_heldout.fillna(False).astype(bool).to_numpy()
tr = main & (df.split == "train").to_numpy() & ~heldout
suites = {
    "A new scenarios, train horizons": main & (df.split == "test").to_numpy() & ~heldout,
    "F withheld horizons, train scenarios": main & (df.split == "train").to_numpy() & heldout,
    "G withheld horizons, new scenarios": main & (df.split == "test").to_numpy() & heldout,
}
T_MIN, T_MAX = float(np.nanmin(t_all[main])) - 0.05, float(np.nanmax(t_all[main])) + 0.05


def r2(Xhat, X):
    return float(1 - ((Xhat - X) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())


def centroid_err(model, X, t, ctx):
    """mean over horizon bins of ||predicted centroid − actual centroid|| / mean within-bin spread."""
    vals = []
    for tv in np.unique(t):
        m = t == tv
        for lv in np.unique(ctx[m]):
            mm = m & (ctx == lv)
            if mm.sum() < 5: continue
            C = X[mm].mean(0); spread = np.linalg.norm(X[mm] - C, axis=1).mean()
            pred = model.predict(np.array([tv]), np.array([lv]))[0]
            vals.append(np.linalg.norm(pred - C) / spread)
    return float(np.mean(vals)) if vals else np.nan


rows_fw, rows_dec, geo_rows = [], [], []
for cell in a.cells.split(","):
    layer, pos = resolve_cell(cell, run); k = run.pos_index(pos)
    X = run.get_layer(layer)[:, k].astype(np.float64)
    if ext is not None:
        Xe = ext.get_layer(layer)[:, ext.pos_index(pos)].astype(np.float64); te = np.log10(ext.index.horizon_years.to_numpy(dtype=float))
        exm = ~np.isnan(te) & ((te < T_MIN + 0.05) | (te > T_MAX - 0.05)); Xe, te = Xe[exm], te[exm]; ce = np.array(["structured|investment"] * len(te))
    print(f"\n=== L{layer} {pos}  (train rows {tr.sum()}) ===")
    print(f"{'model':22s} {'ctx':12s} {'trainR2':>7s} " + " ".join(f"{s[:1]:>7s}" for s in suites) + f" {'X extrap':>8s} {'A cent':>7s} {'F cent':>7s}")
    fitted = {}
    for kind in ["line", "quad", "spline"]:
        for ctxmode in ["none", "offset", "interaction"]:
            m = CurveModel(Basis(kind, T_MIN, T_MAX), context=ctxmode, alpha=a.alpha).fit(X[tr], t_all[tr], ctx_all[tr])
            fitted[(kind, ctxmode)] = m
            rec = dict(layer=layer, position=pos, model=kind, ctx=ctxmode, train_r2=r2(m.predict(t_all[tr], ctx_all[tr]), X[tr]))
            for name, mask in suites.items():
                rec[name[:1]] = r2(m.predict(t_all[mask], ctx_all[mask]), X[mask])
            rec["X"] = r2(m.predict(te, ce), Xe) if ext is not None else np.nan
            rec["A_cent"] = centroid_err(m, X[suites["A new scenarios, train horizons"]], t_all[suites["A new scenarios, train horizons"]], ctx_all[suites["A new scenarios, train horizons"]])
            rec["F_cent"] = centroid_err(m, X[suites["F withheld horizons, train scenarios"]], t_all[suites["F withheld horizons, train scenarios"]], ctx_all[suites["F withheld horizons, train scenarios"]])
            rows_fw.append(rec)
            print(f"{kind:22s} {ctxmode:12s} {rec['train_r2']:7.3f} " + " ".join(f"{rec[s[:1]]:7.3f}" for s in suites) + f" {rec['X']:8.3f} {rec['A_cent']:7.2f} {rec['F_cent']:7.2f}")
    sys.stdout.flush()

    # ---- decoder vs ridge (selected: quad+offset and spline+offset), same train rows
    ridge = Ridge(alpha=1.0).fit(X[tr], t_all[tr])
    print(f"\n{'decoder':28s} " + " ".join(f"{s[:1]+' w2x':>7s} {s[:1]+' rmse':>8s}" for s in suites) + ("   X w2x  X rmse" if ext is not None else ""))
    decs = {"ridge": None, "quad+offset": fitted[("quad", "offset")], "spline+offset": fitted[("spline", "offset")], "quad+interaction": fitted[("quad", "interaction")]}
    for name, m in decs.items():
        line = f"{name:28s} "; rec = dict(layer=layer, position=pos, decoder=name)
        for sname, mask in list(suites.items()) + ([("X", None)] if ext is not None else []):
            if sname == "X": Xs, ts, cs = Xe, te, ce
            else: Xs, ts, cs = X[mask], t_all[mask], ctx_all[mask]
            th = ridge.predict(Xs) if m is None else m.decode(Xs, cs if m.context != "none" else None)[0]
            e = th - ts; w2 = float(np.mean(np.abs(e) <= LOG2)); rm = float(np.sqrt(np.mean(e ** 2)))
            rec[f"{sname[:1]}_w2x"] = w2; rec[f"{sname[:1]}_rmse"] = rm; line += f"{w2:7.2f} {rm:8.3f} "
        rows_dec.append(rec); print(line)
    sys.stdout.flush()

    # ---- basis-free geometry (quad+offset and spline+offset, reference level), with bootstrap over training configs
    for kind in ["quad", "spline"]:
        m = fitted[(kind, "offset")]; g = m.geometry(np.log10(1 / 365.25), 2.0)
        cfgs = df.config_id[tr].to_numpy(); uniq = np.unique(cfgs); rng = np.random.default_rng(0); boots = []
        for b in range(a.boot):
            pick = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(cfgs == c) for c in pick]); rows = np.flatnonzero(tr)[idx]
            mb = CurveModel(Basis(kind, T_MIN, T_MAX), context="offset", alpha=a.alpha).fit(X[rows], t_all[rows], ctx_all[rows])
            gb = mb.geometry(np.log10(1 / 365.25), 2.0); Fm = suites["F withheld horizons, train scenarios"]
            boots.append(dict(tort=gb["tortuosity"], turn=gb["turn_deg"], n95=gb["n95"], tk=gb["t_kappa_max"], F=r2(mb.predict(t_all[Fm], ctx_all[Fm]), X[Fm])))
        bd = pd.DataFrame(boots); q = lambda c: f"[{bd[c].quantile(.05):.2f}, {bd[c].quantile(.95):.2f}]"
        print(f"\ngeometry {kind}+offset (1 d–100 y, reference context {m.levels[0]}): tortuosity {g['tortuosity']:.2f} {q('tort')}, end-to-end turn {g['turn_deg']:.0f}° {q('turn')}, "
              f"curvature peak at 10^{g['t_kappa_max']:.2f} y = {10**g['t_kappa_max']:.2f} y {q('tk')}, curve n90/n95/n99 {g['n90']}/{g['n95']}/{g['n99']}, curve var top-3 {np.round(g['curve_var_top'][:3], 3).tolist()}; "
              f"bootstrap F R2 {q('F')}  ({a.boot} resamples of {len(uniq)} training configs)")
        geo_rows.append(dict(layer=layer, position=pos, model=kind, **{k2: v for k2, v in g.items() if k2 not in ("t", "speed", "kappa")}, boot=bd.describe().loc[["5%", "50%", "95%"]].to_dict() if False else None))
        pd.DataFrame(dict(t=g["t"], speed=g["speed"], kappa=g["kappa"])).to_csv(out / f"curvature_L{layer:02d}_{pos}_{kind}.csv", index=False)
    # offset vs interaction: are context effects constant offsets?
    fo, fi = [r for r in rows_fw if r["layer"] == layer and r["position"] == pos and r["model"] == "quad" and r["ctx"] == "offset"][0], [r for r in rows_fw if r["layer"] == layer and r["position"] == pos and r["model"] == "quad" and r["ctx"] == "interaction"][0]
    print(f"context effects (quad): offset vs interaction held-out R2  A {fo['A']:.3f} vs {fi['A']:.3f}   F {fo['F']:.3f} vs {fi['F']:.3f}   G {fo['G']:.3f} vs {fi['G']:.3f}")

    # ---- absolute vs relative parameterization
    trel = t_all - np.log10(df.short_delay_years.to_numpy(dtype=float))
    for label, tt in [("absolute log H", t_all), ("relative log(H/short delay)", trel)]:
        lo, hi = float(np.nanmin(tt[main])) - 0.05, float(np.nanmax(tt[main])) + 0.05
        m = CurveModel(Basis("quad", lo, hi), context="offset", alpha=a.alpha).fit(X[tr], tt[tr], ctx_all[tr])
        A = suites["A new scenarios, train horizons"]; F = suites["F withheld horizons, train scenarios"]
        print(f"parameterization {label:28s}: held-out R2  A {r2(m.predict(tt[A], ctx_all[A]), X[A]):.3f}   F {r2(m.predict(tt[F], ctx_all[F]), X[F]):.3f}")

    # ---- distractor decomposition (mention / role rows; plain|<domain> context)
    m = fitted[("quad", "offset")]
    for cond in ["mention", "role"]:
        mk = (df.condition == cond).to_numpy(); Xm, tm, cm = X[mk], t_all[mk], ctx_all[mk]
        D = np.log10(df.distractor_years[mk].to_numpy(dtype=float))
        res = Xm - m.predict(tm, cm)                                            # residual at the TRUE horizon
        eps = 1e-3; tang = (m.predict(tm + eps, cm) - m.predict(tm - eps, cm)) / (2 * eps)   # local tangent per row
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        along = np.sum(res * tang, 1); orth = np.linalg.norm(res - along[:, None] * tang, axis=1)
        speed = np.linalg.norm((m.predict(tm + eps, cm) - m.predict(tm - eps, cm)) / (2 * eps), axis=1)   # ||f'|| : activation units per decade
        base = main & (df.rendering == "plain").to_numpy() & ~heldout
        orth_base = np.linalg.norm(X[base] - m.predict(t_all[base], ctx_all[base]), axis=1)
        th, _ = m.decode(Xm, cm)
        print(f"distractor {cond:8s} n={mk.sum()}: along-tangent shift in decades (along/||f'||): rho with (D−H) {spearmanr(along / speed, D - tm).statistic:+.3f}, "
              f"mean |shift| {np.mean(np.abs(along / speed)):.2f} dec; orthogonal residual median {np.median(orth):.1f} vs plain-main {np.median(orth_base):.1f}, rho(orth, |D−H|) {spearmanr(orth, np.abs(D - tm)).statistic:+.3f}; "
              f"curve-decoder within-2x {np.mean(np.abs(th - tm) <= LOG2):.2f}, rho(decoded−H, D−H) {spearmanr(th - tm, D - tm).statistic:+.3f}")
    sys.stdout.flush()

pd.DataFrame(rows_fw).to_csv(out / "forward_models.csv", index=False); pd.DataFrame(rows_dec).to_csv(out / "decoders.csv", index=False)
print("\nwrote", out)
