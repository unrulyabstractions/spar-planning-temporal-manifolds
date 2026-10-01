#!/usr/bin/env python
"""Does the horizon geometry appear in unsupervised PCA once nuisance concepts are linearly erased?

On the matched matrix run the pooled per-cell PCA puts rendering (then domain) on the leading
components and |rho(PC1, log H)| is 0.05–0.27. Here we fit a LEACE erasure of a nuisance block on the
TRAIN-split main rows, apply the fitted map to the TEST-split main rows (new scenario configs), and run
the standard per-cell PCA on the erased test rows. Nuisance blocks (the horizon is never in a block):
    rend    one-hot rendering (3)                       cell    one-hot rendering × domain (6; main effects + interaction)
    dom     one-hot domain (2)                          par     log short reward, log reward ratio, log short delay, log long delay, order
    cfg     one-hot test config (6)      — fitted IN-SAMPLE on the test rows (unseen configs cannot be fitted on train)
    prompt  one-hot config × cell (36)   — in-sample; the prompt text up to the horizon phrase
Config and prompt identity are balanced with horizon by design (every level × every horizon), so their
in-sample erasure is a centring within level and cannot remove horizon information through the design.
`orth` = the same block erased by the orthogonal projection (our earlier nuisance-subspace method).
Ceilings: PCA within each rendering × domain sub-population of the test rows (72 rows), and, with
--canonical-run, PCA on random 72-row subsamples of the canonical single-scenario run restricted to the
same 12 horizons (isolates the sample-size effect).

Per condition: |rho(PC_k, log H)| k=1..3, explained variance, number of top-10 PCs with |rho| > 0.3,
fraction of test variance removed, held-out probes on the erased data (rendering accuracy vs 1/3,
domain accuracy vs 1/2, ridge R² for log short reward, ridge R² for log H), and the PCA of the 12
horizon centroids of the erased test rows (evr top-3, |rho(PC1)|). Also a variance budget of the test
rows: main-effect shares of rendering, domain, rendering×domain cell, config, prompt, horizon, and the
horizon × cell interaction.

Usage: leace_check.py MATRIX_RUN OUT_DIR [--cells 0.55L:T3,0.92L:T3,0.55L:R0,0.72L:R0] [--sweep] [--canonical-run RUN]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from ptm.erasure import LEACE, OrthogonalErasure, one_hot
from ptm.store import RunData
from ptm.depth import resolve_cell, parse_layers

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("out_dir"); ap.add_argument("--cells", default="0.55L:T3,0.92L:T3,0.55L:R0,0.72L:R0")
ap.add_argument("--sweep", action="store_true", help="also sweep layers 2..40 step 2 at T3 and R0")
ap.add_argument("--canonical-run", default="runs/qwen3-14b_investment_n2000_s0")
ap.add_argument("--shrinkage", type=float, default=0.0); ap.add_argument("--n-sub", type=int, default=50)
a = ap.parse_args()
run = RunData(a.run_dir); df = run.index.reset_index(drop=True); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
canon = RunData(a.canonical_run) if a.canonical_run and Path(a.canonical_run).exists() else None
main = (df.condition == "main").to_numpy()
tr = main & (df.split == "train").to_numpy(); te = main & (df.split == "test").to_numpy()
t = np.log10(df.horizon_years.to_numpy(dtype=float))
rend_lab = df.rendering.astype(str).to_numpy(); dom_lab = df.domain.astype(str).to_numpy(); cfg_lab = df.config_id.astype(str).to_numpy()
cell_lab = np.char.add(np.char.add(rend_lab, "|"), dom_lab); prompt_lab = np.char.add(np.char.add(cfg_lab, "|"), cell_lab)
lsr = np.log10(df.short_reward.to_numpy(dtype=float))
blocks = {
    "rend": one_hot(rend_lab), "dom": one_hot(dom_lab), "cell": one_hot(cell_lab),
    "par": np.column_stack([lsr, np.log10(df.long_reward / df.short_reward), np.log10(df.short_delay_years), np.log10(df.long_delay_years), df.short_first.astype(float)]),
}
# (label, train-fitted blocks, in-sample blocks, method)
CONDS = [("none", [], [], "leace"), ("rend", ["rend"], [], "leace"), ("rend+dom", ["rend", "dom"], [], "leace"),
         ("cell", ["cell"], [], "leace"), ("cell+par", ["cell", "par"], [], "leace"), ("cell+par orth", ["cell", "par"], [], "orth"),
         ("cell+par +cfg(in)", ["cell", "par"], ["cfg"], "leace"), ("cell+par +prompt(in)", ["cell", "par"], ["prompt"], "leace")]
SWEEP_CONDS = ("none", "cell+par", "cell+par +prompt(in)")


def rho(a_, b_):
    return abs(float(spearmanr(a_, b_).statistic))


def pca_stats(Xs, ts):
    p = PCA(10, svd_solver="randomized", random_state=0).fit(Xs); Z = p.transform(Xs)
    r = [rho(Z[:, k], ts) for k in range(10)]
    return dict(rho1=r[0], rho2=r[1], rho3=r[2], evr1=p.explained_variance_ratio_[0], evr2=p.explained_variance_ratio_[1],
                evr3=p.explained_variance_ratio_[2], n_pc_h=int(sum(v > 0.3 for v in r)))


def centroid_stats(Xs, ts):
    tv = np.unique(ts); C = np.stack([Xs[ts == v].mean(0) for v in tv]); Cc = C - C.mean(0)
    U, s, Vt = np.linalg.svd(Cc, full_matrices=False); var = s ** 2 / (s ** 2).sum()
    return dict(c_evr1=var[0], c_evr2=var[1], c_evr3=var[2], c_rho1=rho(Cc @ Vt[0], tv))


def probes(Xtr, Xte):
    acc_r = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, rend_lab[tr]).score(Xte, rend_lab[te])
    acc_d = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, dom_lab[tr]).score(Xte, dom_lab[te])
    r2_sr = Ridge(1.0).fit(Xtr, lsr[tr]).score(Xte, lsr[te]); r2_h = Ridge(1.0).fit(Xtr, t[tr]).score(Xte, t[te])
    return dict(acc_rend=acc_r, acc_dom=acc_d, r2_short_reward=r2_sr, r2_h=r2_h)


def share(Xs, lab):
    """Between-level centroid variance as a share of total variance (main effect of a balanced factor)."""
    tot_ = ((Xs - Xs.mean(0)) ** 2).sum(); lv = np.unique(lab)
    C = np.stack([Xs[lab == v].mean(0) for v in lv]); n_ = np.array([(lab == v).sum() for v in lv])
    return float((n_[:, None] * (C - Xs.mean(0)) ** 2).sum() / tot_)


def budget(Xte_):
    labs = dict(rendering=rend_lab[te], domain=dom_lab[te], cell=cell_lab[te], config=cfg_lab[te], prompt=prompt_lab[te], horizon=t[te])
    b = {k: share(Xte_, v) for k, v in labs.items()}
    hc = np.char.add(np.char.add(cell_lab[te], "|"), t[te].astype(str))
    b["horizon×cell"] = share(Xte_, hc) - b["horizon"] - b["cell"]
    hp = np.char.add(np.char.add(prompt_lab[te], "|"), t[te].astype(str))     # one row per level: share = 1 by construction
    b["horizon×prompt (=residual)"] = share(Xte_, hp) - b["horizon"] - b["prompt"]
    return b


def evaluate(X, layer, pos, conds=CONDS, full=True):
    rows = []; Xtr, Xte = X[tr], X[te]; tot = ((Xte - Xte.mean(0)) ** 2).sum()
    for name, keys, in_keys, method in conds:
        Etr, Ete, rank = Xtr, Xte, 0
        if keys:
            Z = np.hstack([blocks[k] for k in keys])
            er = (LEACE(shrinkage=a.shrinkage) if method == "leace" else OrthogonalErasure()).fit(Xtr, Z[tr])
            Etr, Ete, rank = er.transform(Xtr), er.transform(Xte), er.rank
        for k in in_keys:
            lab = {"cfg": cfg_lab, "prompt": prompt_lab}[k][te]
            er = LEACE(shrinkage=a.shrinkage).fit(Ete, one_hot(lab)); Ete = er.transform(Ete); rank += er.rank
        rec = dict(layer=layer, position=pos, cond=name, rank=rank, removed=float(((Ete - Xte) ** 2).sum() / tot),
                   **pca_stats(Ete, t[te]), **centroid_stats(Ete, t[te]))
        if full: rec.update(probes(Etr, Ete))
        rows.append(rec)
    if full:
        sub = []
        for c in np.unique(cell_lab[te]):
            m = cell_lab[te] == c; st = pca_stats(Xte[m], t[te][m]); st["share_h"] = share(Xte[m], t[te][m]); sub.append(st)
            rows.append(dict(layer=layer, position=pos, cond=f"  sub-cell {c}", rank=0, removed=np.nan, **st))
        rows.append(dict(layer=layer, position=pos, cond="sub-cell ceiling (mean of 6)", rank=0, removed=np.nan, **pd.DataFrame(sub).mean().to_dict()))
        # is the horizon response the same direction in every cell? principal angles between the centroid-PC1
        # directions (and the top-2 centroid subspaces) of the 6 sub-cells, split by what differs between the pair
        dirs = {}
        for c in np.unique(cell_lab[te]):
            m = cell_lab[te] == c; tv = np.unique(t[te][m]); C = np.stack([Xte[m & (t[te] == v)].mean(0) for v in tv]); Cc = C - C.mean(0)
            _, _, Vt = np.linalg.svd(Cc, full_matrices=False); dirs[c] = Vt[:2].T
        ang1, ang2 = {"rendering": [], "domain": [], "both": []}, {"rendering": [], "domain": [], "both": []}
        keys = list(dirs)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ri, di = keys[i].split("|"); rj, dj = keys[j].split("|"); kind = "both" if (ri != rj and di != dj) else ("rendering" if ri != rj else "domain")
                ang1[kind].append(np.degrees(np.arccos(np.clip(abs(dirs[keys[i]][:, 0] @ dirs[keys[j]][:, 0]), 0, 1))))
                sv = np.linalg.svd(dirs[keys[i]].T @ dirs[keys[j]], compute_uv=False); ang2[kind].append(np.degrees(np.arccos(np.clip(sv, 0, 1))).max())
        rows.append(dict(layer=layer, position=pos, cond="centroid-direction angles", rank=0, removed=np.nan,
                         **{f"pc1_{k}": float(np.mean(v)) for k, v in ang1.items()}, **{f"sub2_{k}": float(np.mean(v)) for k, v in ang2.items()}))
        if canon is not None:
            cdf = canon.index; Xc = canon.get(layer, pos).astype(np.float64); tc = np.log10(cdf.horizon_years.to_numpy(dtype=float))
            ok = canon.valid_mask(pos) & np.isin(np.round(tc, 6), np.round(np.unique(t[te]), 6)); rng = np.random.default_rng(0); subs = []
            for _ in range(a.n_sub):
                idx = np.concatenate([rng.choice(np.flatnonzero(ok & (np.round(tc, 6) == v)), 6, replace=False) for v in np.round(np.unique(t[te]), 6)])
                subs.append(pca_stats(Xc[idx], tc[idx]))
            rows.append(dict(layer=layer, position=pos, cond=f"canonical 72-row subsample (mean of {a.n_sub})", rank=0, removed=np.nan, **pd.DataFrame(subs).mean().to_dict()))
            rows.append(dict(layer=layer, position=pos, cond=f"canonical all rows, 12 horizons (n={ok.sum()})", rank=0, removed=np.nan, **pca_stats(Xc[ok], tc[ok]), share_h=share(Xc[ok], tc[ok])))
    return rows


allrows = []
for cell in a.cells.split(","):
    layer, pos = resolve_cell(cell, run)
    X = run.get(layer, pos).astype(np.float64)
    rows = evaluate(X, layer, pos); allrows += rows
    b = budget(X[te]); allrows.append(dict(layer=layer, position=pos, cond="variance budget", **{f"share_{k}": v for k, v in b.items()}))
    print(f"\n=== L{layer} {pos}  (erasure fit on {tr.sum()} train rows; PCA on {te.sum()} test rows: 6 new configs × 2 domains × 12 horizons × 3 renderings) ===")
    print("variance budget of the test rows: " + "  ".join(f"{k} {v:.3f}" for k, v in b.items()))
    print(f"{'condition':44s} {'rank':>4s} {'removed':>7s} | {'|rho1|':>6s} {'|rho2|':>6s} {'|rho3|':>6s} {'evr1':>5s} {'evr2':>5s} {'evr3':>5s} {'nPCh':>4s} | {'cent evr1/2/3':>15s} {'c_rho1':>6s} | {'accR':>5s} {'accD':>5s} {'R2sr':>6s} {'R2h':>5s}")
    for r in rows:
        if r["cond"] == "centroid-direction angles":
            print(f"angles between sub-cell horizon directions (deg): centroid PC1 — pairs differing in rendering {r['pc1_rendering']:.0f}, in domain {r['pc1_domain']:.0f}, in both {r['pc1_both']:.0f}; "
                  f"largest principal angle between top-2 centroid subspaces — rendering {r['sub2_rendering']:.0f}, domain {r['sub2_domain']:.0f}, both {r['sub2_both']:.0f}   (random directions in R^5120: ~89)")
            continue
        pr = f"{r['acc_rend']:5.2f} {r['acc_dom']:5.2f} {r['r2_short_reward']:6.2f} {r['r2_h']:5.2f}" if "acc_rend" in r else ""
        ce = f"{r['c_evr1']:.2f}/{r['c_evr2']:.2f}/{r['c_evr3']:.2f}" if "c_evr1" in r else ""
        cr = f"{r['c_rho1']:6.2f}" if "c_rho1" in r else (f"  h-share {r['share_h']:.3f}" if "share_h" in r and not np.isnan(r["share_h"]) else " " * 6)
        print(f"{r['cond']:44s} {r['rank']:4d} {r['removed']:7.3f} | {r['rho1']:6.2f} {r['rho2']:6.2f} {r['rho3']:6.2f} {r['evr1']:5.2f} {r['evr2']:5.2f} {r['evr3']:5.2f} {int(r['n_pc_h']):4d} | {ce:>15s} {cr} | {pr}")
    sys.stdout.flush()
pd.DataFrame(allrows).to_csv(out / "leace_cells.csv", index=False)

if a.sweep:
    sw = []; sconds = [c for c in CONDS if c[0] in SWEEP_CONDS]
    for pos in ["T3", "R0"]:
        for layer in range(2, run.meta["n_layers"] + 1, max(1, round(run.meta["n_layers"] / 20))):
            X = run.get(layer, pos).astype(np.float64); rows = evaluate(X, layer, pos, sconds, full=False)
            b = budget(X[te]); [r.update(share_h=b["horizon"], share_cell=b["cell"], share_prompt=b["prompt"]) for r in rows]; sw += rows
            print(f"sweep L{layer:2d} {pos}: horizon share {b['horizon']:.3f}  " + "  ".join(f"{r['cond']} |rho1| {r['rho1']:.2f}" for r in rows)); sys.stdout.flush()
    sw = pd.DataFrame(sw); sw.to_csv(out / "leace_sweep.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, pos in zip(axes, ["T3", "R0"]):
        for cond, c in zip(SWEEP_CONDS, ["#9a9a9a", "#1b7f79", "#d9730d"]):
            d = sw[(sw.position == pos) & (sw.cond == cond)]; ax.plot(d.layer, d.rho1, "-o", ms=3, color=c, label=f"PCA after erasing: {cond}")
        d = sw[(sw.position == pos) & (sw.cond == "none")]; ax.plot(d.layer, d.share_h, "--", color="k", lw=1, label="horizon share of total variance")
        ax.set_title(f"position {pos}"); ax.set_xlabel("layer"); ax.set_ylim(0, 1); ax.grid(alpha=.3)
    axes[0].set_ylabel("|rho(PC1, log H)|  /  share"); axes[0].legend(fontsize=7)
    fig.suptitle("Matrix run, test split (new scenarios): pooled-PCA ordinality before and after LEACE erasure", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "leace_sweep.png", dpi=130)
print("\nwrote", out)
