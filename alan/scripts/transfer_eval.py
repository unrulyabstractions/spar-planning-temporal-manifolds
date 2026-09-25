#!/usr/bin/env python
"""Step 3: frozen ridge horizon decoder with held-out scenarios, renderings, horizons and domains,
against text baselines, on the matched matrix run.

Protocol
  train  : condition=main, split=train, horizon not withheld, renderings per --train-renderings
  select : dev configurations, all renderings, train horizons  -> choose (layer, position, alpha), then freeze
  test   : suites below, each evaluated once with the frozen decoder
Metrics (log10 years): RMSE, median |err|, Spearman rho, within factor 2 (|err| <= 0.301), worst horizon-bin within-2x.
Baselines: regex-first (first duration in the text), regex-horizon (duration in the sentence containing
"horizon"), text-ridge (TF-IDF 1-2grams -> ridge, trained on the same rows).

Usage: transfer_eval.py RUN_DIR OUT_DIR [--layers 14,18,22,26,29,33,37] [--positions T0,T1,T3,T5,T8,R0]
                        [--train-renderings structured | structured,plain,varied] [--train-domains investment,climate]
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from ptm.horizons import Horizon
from ptm.store import RunData

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("out_dir")
ap.add_argument("--layers", default="14,18,22,26,29,33,37"); ap.add_argument("--positions", default="T0,T1,T3,T5,T8,R0")
ap.add_argument("--alphas", default="1,10,100,1000,10000")
ap.add_argument("--train-renderings", default="structured"); ap.add_argument("--train-domains", default="investment,climate")
ap.add_argument("--tag", default=None)
ap.add_argument("--distractor-sweep", action="store_true",
                help="also fit a decoder per candidate cell (alpha 1) and report new-scenario within-2x against the pull of the mention/role distractors")
ap.add_argument("--sweep-positions", default="T0,T1,T2,T3,T4,T5,T6,T7,T8,R0")
a = ap.parse_args()
run = RunData(a.run_dir); df = run.index.reset_index(drop=True); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
tag = a.tag or f"train-{a.train_renderings.replace(',', '+')}-{a.train_domains.replace(',', '+')}"
layers = [int(x) for x in a.layers.split(",")]; positions = a.positions.split(","); alphas = [float(x) for x in a.alphas.split(",")]
train_r = a.train_renderings.split(","); train_d = a.train_domains.split(",")
LOG2 = np.log10(2)

y = np.log10(df.horizon_years.to_numpy(dtype=float))
main = (df.condition == "main").to_numpy(); heldout_h = df.horizon_heldout.fillna(False).astype(bool).to_numpy()
tr = main & (df.split == "train").to_numpy() & ~heldout_h & df.rendering.isin(train_r).to_numpy() & df.domain.isin(train_d).to_numpy()
dev = main & (df.split == "dev").to_numpy() & ~heldout_h
print(f"[{tag}] train rows {tr.sum()}  dev rows {dev.sum()}  candidate cells {len(layers) * len(positions)}  alphas {alphas}")
sys.stdout.flush()


def metrics(pred, yy, groups=None):
    err = pred - yy
    m = dict(n=len(yy), rmse=float(np.sqrt(np.mean(err ** 2))), bias=float(np.mean(err)), rmse_debiased=float(np.std(err)),
             medae=float(np.median(np.abs(err))),
             rho=float(spearmanr(pred, yy).statistic) if len(yy) > 2 and np.std(yy) > 0 else np.nan,
             within2x=float(np.mean(np.abs(err) <= LOG2)))
    if groups is not None:
        g = pd.Series(np.abs(err) <= LOG2).groupby(pd.Series(groups)).mean()
        m["worst_group"] = float(g.min()); m["worst_group_name"] = str(g.idxmin())
    return m


# ---------------------------------------------------------------- activations: load candidate layers once
acts = {l: run.get_layer(l) for l in layers}    # [n, n_pos, d] each
def X_of(layer, pos, mask):
    return acts[layer][mask, run.pos_index(pos)]

# ---------------------------------------------------------------- selection on dev
sel = []
for l in layers:
    for p in positions:
        Xtr = X_of(l, p, tr); Xdv = X_of(l, p, dev)
        for al in alphas:
            r = Ridge(alpha=al).fit(Xtr, y[tr]); pred = r.predict(Xdv)
            sel.append(dict(layer=l, position=p, alpha=al, **metrics(pred, y[dev])))
sel = pd.DataFrame(sel); sel.to_csv(out / f"selection_{tag}.csv", index=False)
best = sel.sort_values("rmse").iloc[0]
L, P, AL = int(best.layer), best.position, float(best.alpha)
print(f"selected on dev: layer {L} position {P} alpha {AL:g}  (dev rmse {best.rmse:.3f}, within2x {best.within2x:.2f})")
print("dev rmse by position (best layer/alpha each): " + ", ".join(f"{p} {sel[sel.position == p].rmse.min():.3f}" for p in positions))
print("dev rmse by layer (best position/alpha each): " + ", ".join(f"L{l} {sel[sel.layer == l].rmse.min():.3f}" for l in layers))
sys.stdout.flush()
decoder = Ridge(alpha=AL).fit(X_of(L, P, tr), y[tr])

# ---------------------------------------------------------------- text baselines
DUR = re.compile(r"(\d[\d,]*)\s+(minutes?|hours?|days?|weeks?|months?|years?|decades?|century|centuries)\b")
def regex_first(t):
    m = DUR.search(t); return np.log10(Horizon(float(m.group(1).replace(",", "")), m.group(2)).years) if m else np.nan
def regex_horizon(t):
    for sent in re.split(r"(?<=[.!?])\s+", t):
        if "horizon" in sent.lower():
            m = DUR.search(sent)
            if m: return np.log10(Horizon(float(m.group(1).replace(",", "")), m.group(2)).years)
    return regex_first(t)
text_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), Ridge(alpha=1.0)).fit(df.text[tr], y[tr])

def predict_all(mask):
    return dict(activation=decoder.predict(X_of(L, P, mask)),
                regex_first=np.array([regex_first(t) for t in df.text[mask]]),
                regex_horizon=np.array([regex_horizon(t) for t in df.text[mask]]),
                text_ridge=text_model.predict(df.text[mask]))

# ---------------------------------------------------------------- test suites
test = main & (df.split == "test").to_numpy()
trainsc = main & (df.split == "train").to_numpy()
suites = {
    "A new scenarios, structured, train horizons": test & (df.rendering == "structured").to_numpy() & ~heldout_h,
    "B familiar scenarios, plain, train horizons": trainsc & (df.rendering == "plain").to_numpy() & ~heldout_h,
    "C familiar scenarios, varied, train horizons": trainsc & (df.rendering == "varied").to_numpy() & ~heldout_h,
    "D new scenarios, plain, train horizons": test & (df.rendering == "plain").to_numpy() & ~heldout_h,
    "E new scenarios, varied, train horizons": test & (df.rendering == "varied").to_numpy() & ~heldout_h,
    "F withheld horizons, structured, train scenarios": trainsc & (df.rendering == "structured").to_numpy() & heldout_h,
    "G withheld horizons, all renderings, new scenarios": test & heldout_h,
}
for d in ["investment", "climate"]:
    if d not in train_d:
        suites[f"H domain transfer -> {d}, all renderings, all scenarios"] = main & (df.domain == d).to_numpy() & ~heldout_h
rows = []
print("\n=== test suites (frozen decoder; log10-year errors) ===")
print(f"{'suite':54s} {'n':>4s}  {'method':13s} {'rmse':>6s} {'bias':>6s} {'sd':>6s} {'medae':>6s} {'rho':>6s} {'w/in2x':>7s} {'worst bin':>9s}")
for name, mask in suites.items():
    if mask.sum() == 0: continue
    if name[0] in "BC" and len(train_r) > 1:
        name = name + " [IN TRAINING SET under this condition]"
    preds = predict_all(mask); yy = y[mask]; groups = df.horizon_text[mask].to_numpy()
    for meth, pr in preds.items():
        ok = ~np.isnan(pr)
        m = metrics(pr[ok], yy[ok], groups[ok]); m.update(suite=name, method=meth, tag=tag, n_nan=int((~ok).sum())); rows.append(m)
        print(f"{name:54s} {m['n']:4d}  {meth:13s} {m['rmse']:6.3f} {m['bias']:+6.2f} {m['rmse_debiased']:6.3f} {m['medae']:6.3f} {m['rho']:6.3f} {m['within2x']:7.2f} {m['worst_group']:6.2f} ({m['worst_group_name']})")
    sys.stdout.flush()
pd.DataFrame(rows).to_csv(out / f"suites_{tag}.csv", index=False)

# ---------------------------------------------------------------- controls
print("\n=== controls (frozen decoder) ===")
ctl = {}
for cond in ["mention", "role"]:
    mask = (df.condition == cond).to_numpy(); pr = decoder.predict(X_of(L, P, mask))
    D = np.log10(df.distractor_years[mask].to_numpy(dtype=float)); H = y[mask]
    m = metrics(pr, H, df.horizon_text[mask].to_numpy())
    rho_D = float(spearmanr(pr - H, D - H).statistic)   # does the residual follow the distractor?
    rf = np.array([regex_first(t) for t in df.text[mask]]); rh = np.array([regex_horizon(t) for t in df.text[mask]])
    print(f"{cond:8s} n={m['n']}  activation: rmse {m['rmse']:.3f} within2x {m['within2x']:.2f} rho(pred,H) {m['rho']:.3f}  rho(residual, D-H) {rho_D:+.3f}"
          f" | regex-first within2x {np.mean(np.abs(rf - H) <= LOG2):.2f}  regex-horizon within2x {np.mean(np.abs(rh - H) <= LOG2):.2f}")
    ctl[cond] = dict(**m, rho_resid_distractor=rho_D)
mask = (df.condition == "factual").to_numpy()
if P.startswith("T"):
    pr = decoder.predict(X_of(L, P, mask)); m = metrics(pr, y[mask], df.horizon_text[mask].to_numpy())
    print(f"factual  n={m['n']}  activation at {P}: rmse {m['rmse']:.3f} within2x {m['within2x']:.2f} rho {m['rho']:.3f}   (no decision task; same quantities)")
else:
    r2 = Ridge(alpha=AL).fit(X_of(L, "T3", tr), y[tr]); pr = r2.predict(X_of(L, "T3", mask)); m = metrics(pr, y[mask], df.horizon_text[mask].to_numpy())
    print(f"factual  n={m['n']}  activation at T3 (selected position {P} is a response position; refit at T3): rmse {m['rmse']:.3f} within2x {m['within2x']:.2f} rho {m['rho']:.3f}")
ctl["factual"] = m
mask = (df.condition == "scaling").to_numpy(); pr = decoder.predict(X_of(L, P, mask)); s = df[mask].assign(pred=pr)
slopes = []
for (cid, dom, base), g in s.groupby(["config_id", "domain", "scale_base_horizon"]):
    g = g.sort_values("scale"); slopes.append(np.polyfit(np.log10(g.scale), g.pred, 1)[0])
abs_err = np.abs(pr - y[mask]); rel = np.log10(df.horizon_years[mask] / df.short_delay_years[mask])
print(f"scaling  n={mask.sum()}  slope of prediction vs log10(scale factor): mean {np.mean(slopes):.2f} (1 = absolute horizon, 0 = relative to option delays); "
      f"within2x of absolute H {np.mean(abs_err <= LOG2):.2f}; rho(pred, absolute H) {spearmanr(pr, y[mask]).statistic:.3f}, rho(pred, H/short-delay) {spearmanr(pr, rel).statistic:.3f}")
ctl["scaling"] = dict(slope_mean=float(np.mean(slopes)), n=int(mask.sum()))
json.dump(dict(tag=tag, layer=L, position=P, alpha=AL, controls=ctl), open(out / f"controls_{tag}.json", "w"), indent=2, default=float)

# ---------------------------------------------------------------- distractor sweep (optional)
if a.distractor_sweep:
    sweep_pos = a.sweep_positions.split(",")
    ment, role = (df.condition == "mention").to_numpy(), (df.condition == "role").to_numpy()
    Dm = np.log10(df.distractor_years[ment].to_numpy(dtype=float)); Dr = np.log10(df.distractor_years[role].to_numpy(dtype=float))
    testA = test & ~heldout_h
    print(f"\n=== distractor sweep ({tag}; alpha=1 per cell): new-scenario within-2x | mention within-2x, rho(resid, D-H) | role within-2x, rho(resid, D-H) ===")
    print("layer  " + "  ".join(f"{p:>16s}" for p in sweep_pos))
    srows = []
    for l in layers:
        XL = acts[l] if l in acts else run.get_layer(l); line = f"L{l:2d}   "
        for p in sweep_pos:
            k = run.pos_index(p); r = Ridge(alpha=1.0).fit(XL[tr, k], y[tr])
            w2 = float(np.mean(np.abs(r.predict(XL[testA, k]) - y[testA]) <= LOG2))
            pm = r.predict(XL[ment, k]); pr_ = r.predict(XL[role, k])
            mw, mr = float(np.mean(np.abs(pm - y[ment]) <= LOG2)), float(spearmanr(pm - y[ment], Dm - y[ment]).statistic)
            rw, rr = float(np.mean(np.abs(pr_ - y[role]) <= LOG2)), float(spearmanr(pr_ - y[role], Dr - y[role]).statistic)
            srows.append(dict(tag=tag, layer=l, position=p, test_within2x=w2, mention_within2x=mw, mention_rho=mr, role_within2x=rw, role_rho=rr))
            line += f"  {w2:.2f}|{mw:.2f},{mr:+.2f}|{rw:.2f},{rr:+.2f}"
        print(line); sys.stdout.flush()
    st = pd.DataFrame(srows); st.to_csv(out / f"distractor_sweep_{tag}.csv", index=False)
    ok = st[st.test_within2x >= 0.85]
    print(f"cells with new-scenario within-2x >= 0.85: {len(ok)}/{len(st)}; min mention rho {st.mention_rho.min():+.2f}, min role rho {st.role_rho.min():+.2f} over all cells")
    print("least role-distractor pull among them:"); print(ok.sort_values("role_rho").head(5).round(3).to_string(index=False))
print("wrote", out)
