#!/usr/bin/env python
"""Levina–Bickel (NIPS 2004) maximum-likelihood intrinsic dimension of the activation cloud at chosen cells.

Per cell, on the rows with a stated horizon and a valid position, exact duplicate activations removed (the same
prompt text yields the same activation and a zero neighbour distance):
  full          eq. (9) with k = k1..k2 (paper: 10..20), normalised by k-1 (eq. 8 as printed) and by k-2 (§3.1)
  gauss         the same estimator on a Gaussian sample with the cell's sample covariance (same n, same PCA
                spectrum, no manifold structure): how far the cloud is from filling its covariance ellipsoid
  calibration   N_m(0, I) at the same n, for reading an estimate against the paper's negative bias (Fig. 1b, 2b)
  nn_same_*     share of each point's k2 nearest neighbours that share its level of a prompt factor (chance =
                sum of squared level frequencies): which factors the neighbourhoods resolve
  ctr_*         estimate after subtracting the per-level mean of one prompt factor, and the same after subtracting
                the means of randomly permuted levels (same group sizes; mean ± SD over --n-perm draws)
plus the curve m_k over a wide k range (the paper's Fig. 1; k < n - 1), written per cell and drawn as a figure.
--query restricts the rows (pandas expression on index.parquet), e.g. one unit of a matched-duration set.
Factors with one level, or with a distinct level per row, are skipped for nn_same/ctr. When every value of --var
(default horizon_years; short_reward for the reward control) is distinct, i.e. a dense single-quantity set, also:
  ref_1d        the estimator on log10 var itself: an exact 1-D manifold with the same sampling density
  nn_rank_gap   mean |var rank difference| to the k2 nearest neighbours, activations vs ref_1d (ordered curve -> equal)
  local_*       per-point estimate (mean over k1..k2 of eq. 8): Spearman with log var and the mean per decade
  smooth_r2     5-fold held-out R² of a cubic spline in log10 var (ptm.curve.quantile_basis: 8 interior knots at quantiles),
                also by range of the rendered integer (1-9, 10-99, >=100), scored on interpolation only (smooth_r2_interp),
                and for a straight line in log10 var; 4 interior knots instead of 8 below 400 rows
  resid         the estimator on X minus the in-sample spline fit: what is left after the smooth curve
  resid_share_* share of residual variance carried by per-group means of the last digit and digit count of the rendered
                integer (--value-col), with the last digit permuted as the chance level; for the horizon also from the
                rendered number string (resid_share_last_rendered_digit, ..._rendered_digit_count: "3.08 years" -> 8, 3)
  with --group COL (e.g. horizon_unit in a pooled multi-unit set), all scored on interpolation only (test rows inside
  their level's training range; ptm.curve.cv_r2): held-out R² of one shared curve, + per-level offsets, per-level
  separate curves (own quantile knots: 8 interior, 4 below 400 rows), a line + offsets; per level n, estimate
  (calibration at that n is printed), its own spline and line R²; and for pairs of equal values rendered in different
  levels, median pair distance / median k2-th NN distance and how often the partner is among the k2 nearest neighbours.
The dense branch also runs when up to half the rows tie on --var (equal values rendered differently, 7 days = 1 week,
12 months = 1 year); ref_1d then uses the distinct values.

Activations come from the full shards when present, else from the tier-1 subset files.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from ptm.depth import parse_cells
from scipy.stats import spearmanr

from ptm.curve import Basis, CurveModel, cv_r2, quantile_basis

from ptm.intrinsic_dim import dedupe_rows, gaussian_matched, knn_distances, levina_bickel, local_mle, mle_curve
from ptm.store import RunData
from ptm.subset import load_subset

FACTORS = ["horizon_years", "short_reward", "short_delay_years", "long_reward", "long_delay_years", "short_first"]
KS = [3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300]
CAL_M = [1, 2, 3, 5, 8, 10, 12, 15, 20, 30]

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("out_dir")
ap.add_argument("--cells", default="0.55L:R0,0.72L:R0,0.92L:T3,0.55L:T3")
ap.add_argument("--k1", type=int, default=10); ap.add_argument("--k2", type=int, default=20)
ap.add_argument("--n-perm", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--factors", default=",".join(FACTORS), help="prompt factors for nn_same/ctr (comma-separated index columns)")
ap.add_argument("--var", default="horizon_years", help="dense coordinate: when its values are all distinct, the dense-set diagnostics use log10 of it")
ap.add_argument("--value-col", default="horizon_value", help="integer column whose rendered digits the residual is grouped by")
ap.add_argument("--query", default=None, help="restrict rows with a pandas expression on index.parquet, e.g. \"horizon_unit == 'days'\"")
ap.add_argument("--group", default=None, help="dense sets: column whose levels get curve offsets / per-level curves and per-level estimates (e.g. horizon_unit)")
ap.add_argument("--label", default=None, help="model name for figure titles (default: meta model_name)")
a = ap.parse_args()

run = RunData(a.run_dir); df = run.index
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
label = a.label or run.meta["model_name"]
L = run.n_layers
use_shards = any(run.dir.glob("acts_[0-9]*.safetensors"))      # RunData's glob also matches acts_subset_*
rng = np.random.default_rng(a.seed)
print(f"run {a.run_dir} ({label}, L={L}); activations from {'shards' if use_shards else 'subset'}; "
      f"k1..k2 = {a.k1}..{a.k2}", flush=True)


def lb(X, D=None, unbiased=False):
    return levina_bickel(X, a.k1, a.k2, unbiased=unbiased, D=D)


def center_by(X, g):
    return X - pd.DataFrame(X).groupby(g).transform("mean").to_numpy()


rows, curves, cal_cache = [], [], {}
for layer, pos in parse_cells(a.cells, run):
    X = run.get(layer, pos) if use_shards else load_subset(a.run_dir, layer, pos)
    has = run.valid_mask(pos) & df["horizon_years"].notna().to_numpy()
    if a.query:
        has &= df.eval(a.query).to_numpy(dtype=bool)
    idx = np.flatnonzero(has)
    keep = dedupe_rows(X[idx]); idx = idx[keep]
    X = X[idx].astype(np.float64); sub = df.iloc[idx].reset_index(drop=True); n = len(X)
    ks = [k for k in KS if k < n - 1]                         # the k-curve stops below n (small sets)
    D = knn_distances(X, max(ks))
    r = dict(layer=layer, depth=round(layer / L, 3), position=pos, n=n, n_dropped_duplicates=int(has.sum() - n),
             lb=lb(X, D), lb_unbiased=lb(X, D, True))
    G = gaussian_matched(X, rng); DG = knn_distances(G, max(ks)); r["gauss"] = lb(G, DG)
    if n not in cal_cache:
        cal_cache[n] = {m: lb(rng.standard_normal((n, m))) for m in CAL_M}
    # neighbour identities: the k2 nearest indices (knn_distances returns distances only)
    Xc = X - X.mean(0); sq = np.einsum("ij,ij->i", Xc, Xc)
    d2 = sq[:, None] + sq[None] - 2 * Xc @ Xc.T; np.fill_diagonal(d2, np.inf)
    nn = np.argpartition(d2, a.k2, axis=1)[:, : a.k2]
    factors = [f for f in a.factors.split(",") if f and 1 < sub[f].nunique() < n]
    for f in factors:
        v = sub[f].to_numpy()
        r[f"nn_same_{f}"] = float((v[nn] == v[:, None]).mean())
        r[f"nn_chance_{f}"] = float((pd.Series(v).value_counts(normalize=True) ** 2).sum())
        r[f"ctr_{f}"] = lb(center_by(X, v))
        perm = [lb(center_by(X, rng.permutation(v))) for _ in range(a.n_perm)]
        r[f"ctr_perm_{f}"] = float(np.mean(perm)); r[f"ctr_perm_sd_{f}"] = float(np.std(perm))
    lh = np.log10(sub["horizon_years"].to_numpy())
    r["nn_mean_abs_dlog10_horizon"] = float(np.abs(lh[nn] - lh[:, None]).mean())
    tv = np.log10(sub[a.var].to_numpy(dtype=float))           # the dense coordinate (default: horizon in years)
    sets = [("activations", D), ("gauss", DG)]
    tv_distinct = np.unique(np.round(tv, 9))                  # ties within 1e-9 decades: 14 days vs 2 weeks differ in the last bit
    n_unique = len(tv_distinct)
    if n_unique >= 0.5 * n:                                   # dense set (the canonical grid has ~1% distinct); ties = equal
                                                              # values rendered differently (7 days = 1 week, 12 months = 1 year)
        r["n_unique_var"] = n_unique
        X1 = (tv if n_unique == n else tv_distinct)[:, None]; D1 = knn_distances(X1, min(max(ks), len(X1) - 2)); r["ref_1d"] = lb(X1, D1); sets.append(("ref_1d", D1))
        rank = np.argsort(np.argsort(tv))
        nn1 = np.argpartition(np.abs(tv[:, None] - tv[None]) + np.diag(np.full(n, np.inf)), a.k2, axis=1)[:, : a.k2]
        r["nn_rank_gap"] = float(np.abs(rank[nn] - rank[:, None]).mean())
        r["nn_rank_gap_ref_1d"] = float(np.abs(rank[nn1] - rank[:, None]).mean())
        loc = np.mean([local_mle(D, k) for k in range(a.k1, a.k2 + 1)], axis=0)
        r["local_rho_log_var"] = float(spearmanr(loc, tv).statistic)
        dec = np.floor(tv).astype(int)
        r["local_by_decade"] = json.dumps({int(d): round(float(loc[dec == d].mean()), 2) for d in np.unique(dec)})
        val = sub[a.value_col].to_numpy().astype(np.int64)
        basis = lambda: quantile_basis(tv, 8 if n >= 400 else 4)    # 4 interior knots below 400 rows (overfit otherwise)
        r["smooth_r2"], err, err0 = cv_r2(X, tv, basis)
        r["smooth_r2_interp"], e_int, _ = cv_r2(X, tv, basis, interpolate_only=True)
        for lo, hi in ((1, 10), (10, 100), (100, None)):         # by range of the rendered integer: catches end blow-ups
            m = (val >= lo) & (val < hi) if hi else (val >= lo)
            r[f"smooth_r2_{lo}_{hi or 'up'}"] = float(1 - err[m].sum() / err0[m].sum())
        r["line_r2"] = cv_r2(X, tv, lambda: Basis("line"))[0]
        R = X - CurveModel(basis(), alpha=1e-6).fit(X, tv).predict(tv); r["resid"] = lb(R)
        share = lambda g: float((pd.DataFrame(R).groupby(g).transform("mean").to_numpy() ** 2).sum() / (R ** 2).sum())
        r["resid_share_last_digit"] = share(val % 10)
        r["resid_share_last_digit_perm"] = share(rng.permutation(val % 10))
        r["resid_share_digit_count"] = share(np.array([len(str(x)) for x in val]))
        if a.var == "horizon_years":                          # digits as rendered ("3.08 years" -> last 8, 3 digits)
            num = sub["horizon_text"].str.split().str[0].to_numpy()
            r["resid_share_last_rendered_digit"] = share(np.array([x[-1] for x in num]))
            # own generator: drawing from `rng` would shift every later draw and change earlier-logged columns
            rng_r = np.random.default_rng([a.seed, layer, sum(map(ord, pos))])
            r["resid_share_last_rendered_digit_perm"] = share(rng_r.permutation(np.array([x[-1] for x in num])))
            r["resid_share_rendered_digit_count"] = share(np.array([sum(c.isdigit() for c in x) for x in num]))
        if a.group:
            g = sub[a.group].astype(str).to_numpy()
            # grouped diagnostics score interpolation only (no test row outside its level's training range); per-level
            # curves use their own quantile knots, 8 interior (4 when the level has < 400 rows)
            lvl_basis = lambda tl: (lambda: quantile_basis(tl, 8 if len(tl) >= 400 else 4))
            for ctxname in ("none", "offset", "separate"):
                r2g, eg, _ = cv_r2(X, tv, basis, ctx=g, context=ctxname, interpolate_only=True,
                                   make_level_basis=lambda tl: lvl_basis(tl)())
                r[f"grp_r2_{ctxname}"] = r2g; r[f"grp_n_excluded_{ctxname}"] = int(np.isnan(eg).sum())
            # equal values rendered in different levels (7 days / 1 week): their distance against the k2-NN scale
            key = np.round(tv, 9); pairs = []
            for kv in np.unique(key):
                idx_k = np.flatnonzero(key == kv)
                pairs += [(i, j) for ii, i in enumerate(idx_k) for j in idx_k[ii + 1:] if g[i] != g[j]]
            r["tie_pairs"] = len(pairs)
            if pairs:
                pi_, pj_ = np.array(pairs).T
                dpair = np.linalg.norm(X[pi_] - X[pj_], axis=1)
                r["tie_dist_over_k2nn"] = float(np.median(dpair) / np.median(D[:, a.k2 - 1]))
                r["tie_partner_in_k2nn"] = float(np.mean([j in set(nn[i]) or i in set(nn[j]) for i, j in pairs]))
            r["grp_line_offset"] = cv_r2(X, tv, lambda: Basis("line"), ctx=g, context="offset", interpolate_only=True)[0]
            per = {}
            for lv in sorted(set(g), key=lambda v: -int((g == v).sum())):
                m = g == lv; nl = int(m.sum())
                if nl <= 3 * a.k2:
                    continue
                if nl not in cal_cache:
                    cal_cache[nl] = {mm: lb(rng.standard_normal((nl, mm))) for mm in CAL_M}
                tl = tv[m]
                per[lv] = dict(n=nl, m=round(lb(X[m]), 2),
                               spline_r2=round(cv_r2(X[m], tl, lvl_basis(tl), interpolate_only=True)[0], 3),
                               line_r2=round(cv_r2(X[m], tl, lambda: Basis("line"), interpolate_only=True)[0], 3),
                               log10_range=[round(float(tl.min()), 2), round(float(tl.max()), 2)])
            r["per_group"] = json.dumps(per)
    rows.append(r)
    for name, DD in sets:
        kk = [k for k in ks if k <= DD.shape[1]]
        for k, v in zip(kk, mle_curve(DD, kk)):
            curves.append(dict(layer=layer, position=pos, set=name, k=k, m_k=float(v)))
    print(f"L{layer} {pos} ({r['depth']:.2f} L) n={n} (dropped {r['n_dropped_duplicates']} duplicates): "
          f"m = {r['lb']:.2f} [k-2 form {r['lb_unbiased']:.2f}]  Gaussian, same covariance {r['gauss']:.2f}", flush=True)
    print("   m_k at k=" + ",".join(map(str, ks)) + ": "
          + " ".join(f"{c['m_k']:.2f}" for c in curves if c["layer"] == layer and c["position"] == pos and c["set"] == "activations"))
    if "ref_1d" in r:
        print(f"   1-D reference (log10 {a.var}, same sampling) {r['ref_1d']:.2f}; mean |rank gap| to {a.k2}-NN {r['nn_rank_gap']:.1f} "
              f"(1-D reference {r['nn_rank_gap_ref_1d']:.1f}); local estimate vs log {a.var}: Spearman {r['local_rho_log_var']:+.2f}, "
              f"by decade (log10 {a.var}) {r['local_by_decade']}")
        print(f"   spline in log {a.var} (quantile knots): held-out R² {r['smooth_r2']:.3f} (interpolation only "
              f"{r['smooth_r2_interp']:.3f}) [value 1-9 {r['smooth_r2_1_10']:.2f}, "
              f"10-99 {r['smooth_r2_10_100']:.2f}, >=100 {r['smooth_r2_100_up']:.2f}; line {r['line_r2']:.3f}]; estimate on the residual {r['resid']:.2f}; residual variance "
              f"share by last digit {r['resid_share_last_digit']:.3f} (permuted {r['resid_share_last_digit_perm']:.3f}), "
              f"by digit count {r['resid_share_digit_count']:.3f}")
        if "resid_share_last_rendered_digit" in r:
            print(f"   as rendered: last digit {r['resid_share_last_rendered_digit']:.3f} (permuted "
                  f"{r['resid_share_last_rendered_digit_perm']:.3f}), digit count {r['resid_share_rendered_digit_count']:.3f}")
        if n_unique < n:
            print(f"   {n - n_unique} rows tie on {a.var} (equal values in different renderings); ref_1d uses the {n_unique} distinct values")
        if a.group:
            print(f"   by {a.group} (interpolation only): spline R² shared curve {r['grp_r2_none']:.3f}, + per-level offsets "
                  f"{r['grp_r2_offset']:.3f}, per-level curves {r['grp_r2_separate']:.3f} (excluded rows {r['grp_n_excluded_none']}/"
                  f"{r['grp_n_excluded_offset']}/{r['grp_n_excluded_separate']}); line + offsets {r['grp_line_offset']:.3f}")
            print(f"   per level: {r['per_group']}")
            if r["tie_pairs"]:
                print(f"   {r['tie_pairs']} pairs of equal values in different levels: median pair distance / median {a.k2}-th NN distance "
                      f"{r['tie_dist_over_k2nn']:.2f}; partner among the {a.k2}-NN in {r['tie_partner_in_k2nn']:.2f} of pairs")
    for f in factors:
        print(f"   {f:18s} same-level share of {a.k2}-NN {r[f'nn_same_{f}']:.2f} (chance {r[f'nn_chance_{f}']:.2f})  "
              f"centered {r[f'ctr_{f}']:.2f}  permuted-centered {r[f'ctr_perm_{f}']:.2f} ± {r[f'ctr_perm_sd_{f}']:.2f}")
    print(f"   mean |Δ log10 H| to the {a.k2}-NN: {r['nn_mean_abs_dlog10_horizon']:.2f} decades", flush=True)

for n, cal in cal_cache.items():
    print(f"calibration N_m(0, I), n={n}, eq. (9) k={a.k1}..{a.k2}: " + "  ".join(f"m={m}: {v:.2f}" for m, v in cal.items()))
tab = pd.DataFrame(rows); tab.to_csv(out / "intrinsic_dim.csv", index=False)
cv = pd.DataFrame(curves); cv.to_csv(out / "intrinsic_dim_kcurve.csv", index=False)
json.dump({str(n): cal for n, cal in cal_cache.items()}, open(out / "intrinsic_dim_calibration.json", "w"), indent=1)

# figure: m_k vs k per cell (the paper's Fig. 1 form), with the matched Gaussian and N_m reference curves
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, REF = "#2a78d6", "#eb6834", "#b4b2a9"
cells = list(dict.fromkeys(zip(cv.layer, cv.position)))
fig, axes = plt.subplots(1, len(cells), figsize=(3.6 * len(cells), 3.4), sharey=True, squeeze=False)
n0 = next(iter(cal_cache))
ks0 = [k for k in KS if k < n0 - 1]
ref = {m: mle_curve(knn_distances(np.random.default_rng(1).standard_normal((n0, m)), max(ks0)), ks0) for m in (5, 10, 20)}
for ax, (l, p) in zip(axes[0], cells):
    for m, c in ref.items():
        ax.plot(ks0, c, color=REF, lw=1, zorder=1)
        ax.text(ks0[-1] * 1.05, c[-1], f"N$_{{{m}}}$", color="#6b6a63", fontsize=7, va="center")
    s1 = cv[(cv.layer == l) & (cv.position == p) & (cv.set == "ref_1d")]
    if len(s1):
        ax.plot(s1.k, s1.m_k, color="#6b6a63", lw=1.5, ls="--", label="1-D, same horizon sampling", zorder=2)
    for name, col, lab in (("activations", BLUE, "activations"), ("gauss", ORANGE, "Gaussian, same covariance")):
        s = cv[(cv.layer == l) & (cv.position == p) & (cv.set == name)]
        ax.plot(s.k, s.m_k, color=col, lw=2, marker="o", ms=3, label=lab, zorder=3)
    ax.axvspan(a.k1, a.k2, color="#e9e8e2", zorder=0)
    ax.set_xscale("log"); ax.set_title(f"L{l} {p} ({l / L:.2f} L)", fontsize=9); ax.set_xlabel("k (neighbours)")
    ax.grid(alpha=0.3, lw=0.5); ax.spines[["top", "right"]].set_visible(False)
axes[0][0].set_ylabel("$\\hat m_k$ (eq. 8, averaged over points)")
axes[0][0].legend(fontsize=7, frameon=False, loc="upper right")
fig.suptitle(f"{label}: Levina–Bickel MLE of intrinsic dimension vs k  (band: k={a.k1}..{a.k2}; gray: N$_m$(0,I) at n={n0})",
             fontsize=9)
fig.tight_layout(); fig.savefig(out / "intrinsic_dim_kcurve.png", dpi=150)
print(f"wrote {out / 'intrinsic_dim.csv'}, {out / 'intrinsic_dim_kcurve.csv'}, {out / 'intrinsic_dim_kcurve.png'}")
