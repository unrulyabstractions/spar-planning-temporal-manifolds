#!/usr/bin/env python
"""Leave-one-family-out scaling test: does a decoder become invariant to irrelevant durations as more
distractor families enter training?

Families: mention, role (matrix run controls) + the held-out families run (option_age, future_event,
frequency, elapsed_planning). For each held-out family F and each subset S of the other five (|S| = m,
m = 0..5): (a) augmentation: ridge on main train rows + all rows of the families in S; (b) projection:
nuisance subspace from the pooled pair differences of S at rank 3 and 3m, projected out before ridge.
Scored on F: pair pull, slope (decades/decade), within-2x on F rows (all configs, and hold configs only),
plus within-2x on new scenarios. Ceiling: F's own directions (fit configs -> hold configs). Also the
principal angles between the six families' rank-3 nuisance subspaces.

Usage: scaling_test.py MATRIX_RUN FAMILIES_RUN OUT_DIR [--cells 0.55L:T3,0.65L:T1,0.55L:R0]
"""
import argparse, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import subspace_angles
from ptm.relevance import fit_ridge, matched_pairs, nuisance_subspace, pair_pull, pair_slope, project_out
from ptm.store import RunData
from ptm.depth import resolve_cell, parse_layers

ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("families_run"); ap.add_argument("out_dir")
ap.add_argument("--cells", default="0.55L:T3,0.65L:T1,0.55L:R0"); ap.add_argument("--n-fit-configs", type=int, default=6)
a = ap.parse_args()
run = RunData(a.run_dir); df = run.index.reset_index(drop=True); fr = RunData(a.families_run); fdf = fr.index.reset_index(drop=True)
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); LOG2 = np.log10(2)
y = np.log10(df.horizon_years.to_numpy(dtype=float))
main = (df.condition == "main").to_numpy(); heldout = df.horizon_heldout.fillna(False).astype(bool).to_numpy()
tr = main & (df.split == "train").to_numpy() & ~heldout; testA = main & (df.split == "test").to_numpy() & ~heldout
plain = df[(df.condition == "main") & (df.rendering == "plain")]; lookup = dict(zip(zip(plain.config_id, plain.domain, plain.horizon_text), plain.index))

# family -> (source run 'm'|'f', control rows, partner rows in matrix, D, H, config ids)
fam_rows = {}
for c in ["mention", "role"]:
    ci, pi = matched_pairs(df, c); fam_rows[c] = ("m", ci, pi)
for c in sorted(fdf.condition.unique()):
    d = fdf[fdf.condition == c]; keys = list(zip(d.config_id, d.domain, d.horizon_text)); ok = np.array([k in lookup for k in keys])
    fam_rows[c] = ("f", d.index.to_numpy()[ok], np.array([lookup[k] for k, o in zip(keys, ok) if o]))
families = sorted(fam_rows); ctl_cfgs = sorted(set(df.config_id[fam_rows["mention"][1]])); fit_cfgs, hold_cfgs = ctl_cfgs[: a.n_fit_configs], ctl_cfgs[a.n_fit_configs:]
print(f"families {families}; {len(ctl_cfgs)} control configs (own-family ceiling: fit {len(fit_cfgs)} / hold {len(hold_cfgs)})")

rows = []
for cell in a.cells.split(","):
    layer, pos = resolve_cell(cell, run)
    X = run.get_layer(layer)[:, run.pos_index(pos)].astype(np.float64); Xf = fr.get_layer(layer)[:, fr.pos_index(pos)].astype(np.float64)
    F = {}   # per family: Xc, Xp, D, H, cfg, labels
    for c, (src, ci, pi) in fam_rows.items():
        src_df, src_X = (df, X) if src == "m" else (fdf, Xf)
        F[c] = dict(Xc=src_X[ci], Xp=X[pi], D=np.log10(src_df.distractor_years[ci].to_numpy(dtype=float)), H=np.log10(src_df.horizon_years[ci].to_numpy(dtype=float)), cfg=src_df.config_id[ci].to_numpy())

    def score(pred_fn, c, mask=None):
        f = F[c]; m = np.ones(len(f["H"]), bool) if mask is None else mask
        pc, pp = pred_fn(f["Xc"][m]), pred_fn(f["Xp"][m])
        return dict(pull=pair_pull(pc, pp, f["D"][m], f["H"][m]), slope=pair_slope(pc, pp, f["D"][m], f["H"][m]), w2x=float(np.mean(np.abs(pc - f["H"][m]) <= LOG2)))

    print(f"\n=== L{layer} {pos} ===")
    # principal angles between rank-3 nuisance subspaces of the families
    subs = {c: nuisance_subspace(F[c]["Xc"] - F[c]["Xp"], 3)[0] for c in families}
    ang = pd.DataFrame([[np.degrees(subspace_angles(subs[a_], subs[b_]).min()) for b_ in families] for a_ in families], index=families, columns=families)
    print("smallest principal angle (deg) between rank-3 family nuisance subspaces:"); print(ang.round(0).astype(int).to_string())
    for held in families:
        others = [c for c in families if c != held]; hold_m = np.isin(F[held]["cfg"], hold_cfgs)
        # ceiling: own family, fit configs -> hold configs
        fit_m = np.isin(F[held]["cfg"], fit_cfgs); Vo = nuisance_subspace(F[held]["Xc"][fit_m] - F[held]["Xp"][fit_m], 3)[0]
        ro = fit_ridge(project_out(X[tr], Vo), y[tr]); so = score(lambda Z: ro.predict(project_out(Z, Vo)), held, hold_m)
        rows.append(dict(layer=layer, position=pos, held=held, m=-1, decoder="own-family ceiling (hold configs)", subset="", **so, w2x_hold=so["w2x"], A_w2x=np.nan))
        for m in range(0, 6):
            for S in itertools.combinations(others, m):
                # augmentation
                extra_X = [F[c]["Xc"] for c in S]; extra_y = [F[c]["H"] for c in S]
                Xa = np.vstack([X[tr]] + extra_X) if S else X[tr]; ya = np.concatenate([y[tr]] + extra_y) if S else y[tr]
                r = fit_ridge(Xa, ya); s_all = score(r.predict, held); s_hold = score(r.predict, held, hold_m)
                rows.append(dict(layer=layer, position=pos, held=held, m=m, decoder="augmentation", subset="+".join(S), **s_all, w2x_hold=s_hold["w2x"], A_w2x=float(np.mean(np.abs(r.predict(X[testA]) - y[testA]) <= LOG2))))
                # projection
                if S:
                    diffs = np.vstack([F[c]["Xc"] - F[c]["Xp"] for c in S])
                    for k in sorted({3, 3 * m}):
                        V = nuisance_subspace(diffs, k)[0]; rp = fit_ridge(project_out(X[tr], V), y[tr])
                        s_all = score(lambda Z: rp.predict(project_out(Z, V)), held); s_hold = score(lambda Z: rp.predict(project_out(Z, V)), held, hold_m)
                        rows.append(dict(layer=layer, position=pos, held=held, m=m, decoder=f"projection k={'3' if k == 3 else '3m'}", subset="+".join(S), **s_all, w2x_hold=s_hold["w2x"], A_w2x=float(np.mean(np.abs(rp.predict(project_out(X[testA], V)) - y[testA]) <= LOG2))))
        sys.stdout.flush()
    t = pd.DataFrame([r for r in rows if r["layer"] == layer and r["position"] == pos])
    # summary: mean over subsets then over held-out families, per m and decoder
    for dec in ["augmentation", "projection k=3", "projection k=3m"]:
        sub = t[t.decoder == dec]
        if sub.empty: continue
        g = sub.groupby(["held", "m"]).agg(slope=("slope", "mean"), pull=("pull", "mean"), w2x=("w2x", "mean"), w2x_hold=("w2x_hold", "mean"), A=("A_w2x", "mean")).reset_index()
        gm = g.groupby("m").agg(slope=("slope", "mean"), slope_sd=("slope", "std"), pull=("pull", "mean"), w2x=("w2x", "mean"), w2x_sd=("w2x", "std"), w2x_hold=("w2x_hold", "mean"), A=("A", "mean"))
        print(f"\n{dec}: mean over held-out families (sd across families) by number of training families m")
        print("  m   slope (sd)      pull    w2x on F (sd)   w2x F hold-cfgs   new-scenario w2x")
        for m, r in gm.iterrows():
            print(f"  {m}   {r.slope:+.3f} ({r.slope_sd:.3f})  {r.pull:+.2f}   {r.w2x:.2f} ({r.w2x_sd:.2f})       {r.w2x_hold:.2f}             {r.A:.2f}")
    ceil = t[t.decoder.str.startswith("own-family")]
    print(f"\nown-family ceiling (hold configs), mean over families: slope {ceil.slope.mean():+.3f}, pull {ceil.pull.mean():+.2f}, w2x {ceil.w2x.mean():.2f}")
    print("per held-out family at m=5 (augmentation): " + ", ".join(f"{h}: slope {r.slope:+.2f} w2x {r.w2x:.2f}" for h, r in t[(t.decoder == "augmentation") & (t.m == 5)].set_index("held").iterrows()))
    sys.stdout.flush()
pd.DataFrame(rows).to_csv(out / "scaling_test.csv", index=False); print("wrote", out)
