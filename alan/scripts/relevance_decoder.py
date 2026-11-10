#!/usr/bin/env python
"""Relevance-aware decoding on the matched matrix run.

Per cell: (1) ridge baseline (mixed renderings, train configs, train horizons); (2) ridge after projecting
out a k-dimensional nuisance subspace estimated from matched pairs (control prompt minus its distractor-free
partner), k in --ks, with within-type config hold-out (N from `fit` configs, pull measured on `test`
configs) and cross-type transfer (N from mention, pull on role, and the reverse); (3) ridge trained with
the control prompts of the fit configs included at their true labels. Pull = pair_pull (within-pair change
in prediction vs log D − log H); accuracy = within-2x on new scenarios. Angles between the first nuisance
direction, the ridge weight, and the horizon curve's plane (quad+offset, step 4). Scenario bootstrap for
the k=3 numbers. --sweep repeats the k=3 projected decoder over all layers × positions.

Usage: relevance_decoder.py RUN_DIR OUT_DIR [--cells 0.55L:T3,0.65L:T1,0.55L:R0] [--ks 1,2,3,5] [--boot 30] [--sweep]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from ptm.curve import Basis, CurveModel
from ptm.relevance import fit_ridge, matched_pairs, nuisance_subspace, pair_pull, pair_slope, principal_angle_to_subspace, project_out
from ptm.store import RunData
from ptm.depth import resolve_cell, parse_layers

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("out_dir"); ap.add_argument("--cells", default="0.55L:T3,0.65L:T1,0.55L:R0")
ap.add_argument("--ks", default="1,2,3,5"); ap.add_argument("--boot", type=int, default=30); ap.add_argument("--sweep", action="store_true")
ap.add_argument("--n-fit-configs", type=int, default=6)
ap.add_argument("--families-run", default=None, help="run of held-out distractor families (generalization test); partners come from RUN_DIR")
a = ap.parse_args()
run = RunData(a.run_dir); df = run.index.reset_index(drop=True); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
ks = [int(x) for x in a.ks.split(",")]; LOG2 = np.log10(2)
y = np.log10(df.horizon_years.to_numpy(dtype=float))
main = (df.condition == "main").to_numpy(); heldout = df.horizon_heldout.fillna(False).astype(bool).to_numpy()
tr = main & (df.split == "train").to_numpy() & ~heldout
testA = main & (df.split == "test").to_numpy() & ~heldout
pairs = {c: matched_pairs(df, c) for c in ["mention", "role"]}
ctl_cfgs = sorted(set(df.config_id[pairs["mention"][0]]))
fit_cfgs, hold_cfgs = ctl_cfgs[: a.n_fit_configs], ctl_cfgs[a.n_fit_configs:]
print(f"control configs: {len(ctl_cfgs)} (nuisance fit on {len(fit_cfgs)}, pull measured on {len(hold_cfgs)}); pairs mention {len(pairs['mention'][0])}, role {len(pairs['role'][0])}")


def split_pairs(cond, cfgs):
    ci, pi = pairs[cond]; m = df.config_id[ci].isin(cfgs).to_numpy(); return ci[m], pi[m]


def evaluate(decoder_fn, X, label):
    """decoder_fn(Xrows) -> predictions. Returns accuracy on new scenarios and pulls on held-out-config pairs."""
    rec = dict(decoder=label)
    pred = decoder_fn(X[testA]); rec["A_w2x"] = float(np.mean(np.abs(pred - y[testA]) <= LOG2)); rec["A_rmse"] = float(np.sqrt(np.mean((pred - y[testA]) ** 2)))
    for cond in ["mention", "role"]:
        ci, pi = split_pairs(cond, hold_cfgs)
        D = np.log10(df.distractor_years[ci].to_numpy(dtype=float))
        rec[f"{cond}_pull"] = pair_pull(decoder_fn(X[ci]), decoder_fn(X[pi]), D, y[ci])
        rec[f"{cond}_slope"] = pair_slope(decoder_fn(X[ci]), decoder_fn(X[pi]), D, y[ci])
        rec[f"{cond}_w2x"] = float(np.mean(np.abs(decoder_fn(X[ci]) - y[ci]) <= LOG2))
    return rec


def run_cell(layer, pos, verbose=True):
    X = run.get_layer(layer)[:, run.pos_index(pos)].astype(np.float64)
    rows = []
    base = fit_ridge(X[tr], y[tr]); rows.append(dict(**evaluate(base.predict, X, "ridge"), k=0, N_from="-"))
    # nuisance subspaces from the fit configs, per type and pooled
    diffs = {c: X[split_pairs(c, fit_cfgs)[0]] - X[split_pairs(c, fit_cfgs)[1]] for c in ["mention", "role"]}
    diffs["both"] = np.vstack([diffs["mention"], diffs["role"]])
    energy = {}
    for src in ["mention", "role", "both"]:
        for k in ks:
            V, en = nuisance_subspace(diffs[src], k); energy[(src, k)] = en[-1]
            r = fit_ridge(project_out(X[tr], V), y[tr])
            rows.append(dict(**evaluate(lambda Z, r=r, V=V: r.predict(project_out(Z, V)), X, f"ridge − N({src},k={k})"), k=k, N_from=src))
    # trained with the control prompts of the fit configs
    ci = np.concatenate([split_pairs(c, fit_cfgs)[0] for c in ["mention", "role"]])
    aug = np.concatenate([np.flatnonzero(tr), ci]); r = fit_ridge(X[aug], y[aug])
    rows.append(dict(**evaluate(r.predict, X, "ridge + control prompts in training"), k=0, N_from="data"))
    t = pd.DataFrame(rows); t["layer"] = layer; t["position"] = pos
    # angles: first nuisance direction vs ridge weight, and vs the horizon curve plane (top-2 directions of the fitted curve)
    ctx = (df.rendering.astype(str) + "|" + df.domain.astype(str)).to_numpy()
    cm = CurveModel(Basis("quad", y[main].min() - 0.05, y[main].max() + 0.05), context="offset", alpha=1.0).fit(X[tr], y[tr], ctx[tr])
    C = cm.curve(np.linspace(np.log10(1 / 365.25), 2.0, 200)); Cc = C - C.mean(0); Bplane = np.linalg.svd(Cc, full_matrices=False)[2][:2].T
    angles = {}
    for src in ["mention", "role"]:
        v1 = nuisance_subspace(diffs[src], 1)[0][:, 0]
        angles[src] = dict(vs_ridge_weight=principal_angle_to_subspace(v1, (base.coef_ / np.linalg.norm(base.coef_))[:, None]),
                           vs_curve_plane=principal_angle_to_subspace(v1, Bplane), energy_k1=float(energy[(src, 1)]), energy_k3=float(energy[(src, 3)]) if 3 in ks else np.nan)
    if verbose:
        print(f"\n=== L{layer} {pos} ===")
        print(f"{'decoder':34s} {'A w2x':>6s} {'A rmse':>7s} {'mention pull':>13s} {'slope':>6s} {'w2x':>5s} {'role pull':>10s} {'slope':>6s} {'w2x':>5s}")
        for r_ in rows:
            print(f"{r_['decoder']:34s} {r_['A_w2x']:6.2f} {r_['A_rmse']:7.3f} {r_['mention_pull']:+13.3f} {r_['mention_slope']:+6.2f} {r_['mention_w2x']:5.2f} {r_['role_pull']:+10.3f} {r_['role_slope']:+6.2f} {r_['role_w2x']:5.2f}")
        for src in ["mention", "role"]:
            g = angles[src]
            print(f"nuisance({src}) v1: energy k=1 {g['energy_k1']:.2f}, k=3 {g['energy_k3']:.2f}; angle to ridge weight {g['vs_ridge_weight']:.0f}°, to curve plane {g['vs_curve_plane']:.0f}°")
        sys.stdout.flush()
    return t, angles, X


all_rows, all_angles = [], []
for cell in a.cells.split(","):
    layer, pos = resolve_cell(cell, run)
    t, angles, X = run_cell(layer, pos); all_rows.append(t)
    for src, g in angles.items(): all_angles.append(dict(layer=layer, position=pos, src=src, **g))
    # scenario bootstrap for ridge and ridge − N(both, k=3): resample train configs AND fit configs
    if a.boot and 3 in ks:
        rng = np.random.default_rng(0); cfg_tr = df.config_id[tr].to_numpy(); uniq = np.unique(cfg_tr); boots = []
        for b in range(a.boot):
            pick_tr = rng.choice(uniq, len(uniq), replace=True); rows_tr = np.concatenate([np.flatnonzero(tr)[cfg_tr == c] for c in pick_tr])
            pick_fit = rng.choice(fit_cfgs, len(fit_cfgs), replace=True)
            dsum = np.vstack([X[split_pairs(c, [cf])[0]] - X[split_pairs(c, [cf])[1]] for c in ["mention", "role"] for cf in pick_fit])
            V = nuisance_subspace(dsum, 3)[0]
            r0 = fit_ridge(X[rows_tr], y[rows_tr]); r3 = fit_ridge(project_out(X[rows_tr], V), y[rows_tr])
            e0 = evaluate(r0.predict, X, "ridge"); e3 = evaluate(lambda Z, r=r3, V=V: r.predict(project_out(Z, V)), X, "proj")
            boots.append(dict(A0=e0["A_w2x"], A3=e3["A_w2x"], m0=e0["mention_pull"], m3=e3["mention_pull"], r0_=e0["role_pull"], r3_=e3["role_pull"]))
        bd = pd.DataFrame(boots); q = lambda c: f"[{bd[c].quantile(.05):+.2f}, {bd[c].quantile(.95):+.2f}]"
        print(f"bootstrap ({a.boot}): ridge A w2x {q('A0')} mention pull {q('m0')} role pull {q('r0_')} | ridge − N(both,3): A w2x {q('A3')} mention pull {q('m3')} role pull {q('r3_')}")
        sys.stdout.flush()
pd.concat(all_rows).to_csv(out / "relevance_decoders.csv", index=False); pd.DataFrame(all_angles).to_csv(out / "relevance_angles.csv", index=False)

# ---------------------------------------------------------------- generalization to held-out distractor families
if a.families_run:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    fr = RunData(a.families_run); fdf = fr.index.reset_index(drop=True)
    fams = sorted(fdf.condition.unique())
    plain = df[(df.condition == "main") & (df.rendering == "plain")]
    lookup = dict(zip(zip(plain.config_id, plain.domain, plain.horizon_text), plain.index))
    print(f"\n=== generalization to held-out distractor families ({a.families_run}: {fams}) ===")
    # behavior: do choices follow D?  (5-fold logistic, standardized coefficients)
    print("behavior: standardized logistic coefficients for chose_short  logH / logD  (5-fold logloss params+H vs params+H+D)")
    for fam in fams:
        d = fdf[(fdf.condition == fam) & fdf.chose_short.notna()]
        yb = d.chose_short.astype(int).to_numpy()
        P = np.column_stack([np.log10(d.short_delay_years), np.log10(d.long_delay_years), np.log10(d.short_reward), np.log10(d.long_reward / d.short_reward), d.short_first.astype(float)])
        H = np.log10(d.horizon_years.to_numpy(dtype=float))[:, None]; Dd = np.log10(d.distractor_years.to_numpy(dtype=float))[:, None]
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(np.hstack([P, H, Dd]), yb); co = mdl[-1].coef_[0]
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        ll = lambda F: float(-np.mean(np.log(np.clip(cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)), F, yb, cv=cv, method="predict_proba")[np.arange(len(yb)), yb], 1e-9, 1))))
        print(f"  {fam:17s} n={len(d)}  logH {co[5]:+.2f}  logD {co[6]:+.2f}   logloss +H {ll(np.hstack([P, H])):.3f}  +H+D {ll(np.hstack([P, H, Dd])):.3f}")
    frows = []
    for cell in a.cells.split(","):
        layer, pos = resolve_cell(cell, run); k_ = run.pos_index(pos)
        X = run.get_layer(layer)[:, k_].astype(np.float64); Xf = fr.get_layer(layer)[:, fr.pos_index(pos)].astype(np.float64)
        base = fit_ridge(X[tr], y[tr])
        dboth = np.vstack([X[split_pairs(c, fit_cfgs)[0]] - X[split_pairs(c, fit_cfgs)[1]] for c in ["mention", "role"]]); V3 = nuisance_subspace(dboth, 3)[0]
        proj = fit_ridge(project_out(X[tr], V3), y[tr])
        ci_old = np.concatenate([split_pairs(c, fit_cfgs)[0] for c in ["mention", "role"]]); aug_rows = np.concatenate([np.flatnonzero(tr), ci_old]); aug = fit_ridge(X[aug_rows], y[aug_rows])
        decoders = {"ridge": (base.predict, base.predict), "ridge − N(mention+role,k=3)": (lambda Z: proj.predict(project_out(Z, V3)), lambda Z: proj.predict(project_out(Z, V3))), "ridge + mention/role in training": (aug.predict, aug.predict)}
        print(f"\n--- L{layer} {pos}: held-out families (pull, slope, within-2x on family rows; all decoders fit WITHOUT these families) ---")
        print(f"{'decoder':34s} " + " ".join(f"{f[:16]:>22s}" for f in fams))
        for name, (fn_f, fn_m) in decoders.items():
            line = f"{name:34s} "
            for fam in fams:
                d = fdf[fdf.condition == fam]; keys = list(zip(d.config_id, d.domain, d.horizon_text)); ok = np.array([k in lookup for k in keys])
                ci = d.index.to_numpy()[ok]; pi = np.array([lookup[k] for k, o in zip(keys, ok) if o])
                # generalization is measured on ALL configs of the family (none were used to fit anything), but also report on hold configs only
                Dv = np.log10(fdf.distractor_years[ci].to_numpy(dtype=float)); Hv = np.log10(fdf.horizon_years[ci].to_numpy(dtype=float))
                pc, pp = fn_f(Xf[ci]), fn_m(X[pi])
                pull, slope, w2 = pair_pull(pc, pp, Dv, Hv), pair_slope(pc, pp, Dv, Hv), float(np.mean(np.abs(pc - Hv) <= LOG2))
                frows.append(dict(layer=layer, position=pos, decoder=name, family=fam, pull=pull, slope=slope, w2x=w2, n=len(ci)))
                line += f" {pull:+6.2f} {slope:+5.2f} {w2:5.2f}   "
            print(line)
        # within-family projection (N from the family's own fit configs, scored on its hold configs) for comparison
        line = f"{'ridge − N(own family,k=3)':34s} "
        for fam in fams:
            d = fdf[fdf.condition == fam]; keys = list(zip(d.config_id, d.domain, d.horizon_text)); ok = np.array([k in lookup for k in keys])
            ci = d.index.to_numpy()[ok]; pi = np.array([lookup[k] for k, o in zip(keys, ok) if o]); cfg = fdf.config_id[ci].to_numpy()
            fit_m = np.isin(cfg, fit_cfgs); hold_m = np.isin(cfg, hold_cfgs)
            Vf = nuisance_subspace(Xf[ci[fit_m]] - X[pi[fit_m]], 3)[0]; rf = fit_ridge(project_out(X[tr], Vf), y[tr])
            Dv = np.log10(fdf.distractor_years[ci[hold_m]].to_numpy(dtype=float)); Hv = np.log10(fdf.horizon_years[ci[hold_m]].to_numpy(dtype=float))
            pc, pp = rf.predict(project_out(Xf[ci[hold_m]], Vf)), rf.predict(project_out(X[pi[hold_m]], Vf))
            pull, slope, w2 = pair_pull(pc, pp, Dv, Hv), pair_slope(pc, pp, Dv, Hv), float(np.mean(np.abs(pc - Hv) <= LOG2))
            frows.append(dict(layer=layer, position=pos, decoder="ridge − N(own family,k=3) [hold configs]", family=fam, pull=pull, slope=slope, w2x=w2, n=int(hold_m.sum())))
            line += f" {pull:+6.2f} {slope:+5.2f} {w2:5.2f}   "
        print(line); sys.stdout.flush()
    pd.DataFrame(frows).to_csv(out / "relevance_families.csv", index=False)

if a.sweep:
    print("\n=== sweep: ridge vs ridge − N(both, k=3): new-scenario w2x | mention pull | role pull ===")
    positions = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "R0"]; layers = parse_layers("0.35L,0.45L,0.55L,0.65L,0.72L,0.825L,0.92L", run)
    srows = []
    print("layer  " + "  ".join(f"{p:>22s}" for p in positions))
    for l in layers:
        XL = run.get_layer(l); line = f"L{l:2d}  "
        for p in positions:
            X = XL[:, run.pos_index(p)].astype(np.float64)
            d = np.vstack([X[split_pairs(c, fit_cfgs)[0]] - X[split_pairs(c, fit_cfgs)[1]] for c in ["mention", "role"]]); V = nuisance_subspace(d, 3)[0]
            e0 = evaluate(fit_ridge(X[tr], y[tr]).predict, X, "ridge"); r3 = fit_ridge(project_out(X[tr], V), y[tr]); e3 = evaluate(lambda Z, r=r3, V=V: r.predict(project_out(Z, V)), X, "proj")
            srows.append(dict(layer=l, position=p, **{f"ridge_{k}": v for k, v in e0.items() if k != "decoder"}, **{f"proj3_{k}": v for k, v in e3.items() if k != "decoder"}))
            line += f"  {e0['A_w2x']:.2f}→{e3['A_w2x']:.2f}|{e0['mention_pull']:+.2f}→{e3['mention_pull']:+.2f}|{e0['role_pull']:+.2f}→{e3['role_pull']:+.2f}"
        print(line); sys.stdout.flush()
    st = pd.DataFrame(srows); st.to_csv(out / "relevance_sweep.csv", index=False)
    ok = st[st.proj3_A_w2x >= 0.85]
    print(f"cells with projected w2x >= 0.85: {len(ok)}/{len(st)}; among them min |mention pull| {ok.proj3_mention_pull.abs().min():.2f}, min |role pull| {ok.proj3_role_pull.abs().min():.2f}")
    print(ok.assign(s=ok.proj3_mention_pull.abs() + ok.proj3_role_pull.abs()).sort_values("s").head(5)[["layer", "position", "ridge_A_w2x", "proj3_A_w2x", "ridge_mention_pull", "proj3_mention_pull", "ridge_role_pull", "proj3_role_pull"]].round(3).to_string(index=False))
print("wrote", out)
