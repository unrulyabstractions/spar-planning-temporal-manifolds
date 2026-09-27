#!/usr/bin/env python
"""Relevance-aware horizon decoder: fit decoders, measure distractor pull per tier, apply the decision rules.

Decoders (ridge on one layer x position cell; alpha chosen on dev, then frozen):
  baseline : clean matrix prompts only (split=train, training horizons, all three renderings) = Alan's condition 2
  rad_all  : baseline rows + distractor rows of ALL five families (seen templates, split=train)
  lofo_<f> : baseline rows + distractor rows of every family except f (leave one family out)

Tiers (all on the 6 held-out TEST configurations, so every scenario is new to every decoder):
  A : rad_all on seen templates                  (trained wording, new scenarios)
  B : rad_all on unseen templates                (trained role, new wording)
  C : lofo_<f> on family f, all four templates   (role never seen in training)
  pre : baseline on the same pools               (the pull we are trying to remove)

Usage:
  evaluate.py RUN_DIR PROMPTS.parquet OUT_DIR [--layers 14,18,22,26,29,33,37] [--positions T0,...,T8,R0]
              [--alphas 1,10,100,1000,10000] [--n-boot 2000] [--cells-full L26:T1,auto]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rad  # noqa: E402,F401  (puts ../alan on sys.path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from ptm.store import RunData  # noqa: E402
from rad.analysis import (RidgePath, accuracy, behavior_pull, horizon_effect, markdown_table, pull_stats,  # noqa: E402
                          regex_first, regex_horizon)
from rad.build import EXTRA_COLUMNS  # noqa: E402
from rad.families import FAMILIES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("prompts"); ap.add_argument("out_dir")
ap.add_argument("--layers", default="14,18,22,26,29,33,37")
ap.add_argument("--positions", default="T0,T1,T2,T3,T4,T5,T6,T7,T8,R0")
ap.add_argument("--alphas", default="1,10,100,1000,10000")
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--cells-full", default="L26:T1,auto", help="cells reported with bootstrap CIs; 'auto' = best rad_all dev RMSE")
ap.add_argument("--behavior-gate", type=float, default=0.10, help="family is behaviorally irrelevant if its behavior-pull 95%% CI lies within ±gate")
ap.add_argument("--removed", type=float, default=0.10, help="pull counts as removed if its 95%% CI upper bound is below this")
ap.add_argument("--min-baseline-pull", type=float, default=0.20, help="families whose baseline pull is below this have nothing to remove")
ap.add_argument("--max-clean-drop", type=float, default=0.05, help="allowed drop in clean-test within-2x (rad_all vs baseline)")
a = ap.parse_args()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
layers = [int(x) for x in a.layers.split(",")]; positions = a.positions.split(","); alphas = [float(x) for x in a.alphas.split(",")]
FAMS = [f.name for f in FAMILIES]
T0 = time.time()
log_lines = []


def say(*args):
    s = " ".join(str(x) for x in args); print(s); sys.stdout.flush(); log_lines.append(s)


# ------------------------------------------------------------------ data
run = RunData(a.run_dir)
design = pd.read_parquet(a.prompts)[["sample_uid"] + EXTRA_COLUMNS]
df = run.index.reset_index(drop=True).merge(design, on="sample_uid", how="left", validate="1:1")
assert len(df) == len(run.index)
missing = set(design.sample_uid) - set(df.sample_uid)
say(f"run {a.run_dir}: {len(df)} rows ({df.condition.value_counts().to_dict()}); prompts missing from run: {len(missing)}")
say("format adherence by condition:", df.groupby("condition").choice.apply(lambda s: round(s.notna().mean(), 4)).to_dict())

y = np.log10(df.horizon_years.to_numpy(float))
cond, split, fam = df.condition.to_numpy(), df.split.to_numpy(), df.family.to_numpy()
group = df.template_group.to_numpy()
heldout = df.horizon_heldout.fillna(False).astype(bool).to_numpy()
main, dist = cond == "main", cond == "distractor"
row_of = {u: i for i, u in enumerate(df.sample_uid)}
twin = np.array([row_of.get(u, -1) if isinstance(u, str) else -1 for u in df.twin_uid])
assert (twin[dist] >= 0).all(), "every distractor prompt needs its clean twin in the run"
log_d = np.where(dist, np.log10(df.distractor_years.to_numpy(float)), np.nan)
clusters = df.scenario_id.to_numpy()

base_tr = main & (split == "train") & ~heldout
dev_clean = main & (split == "dev") & ~heldout
test_clean = main & (split == "test")


def dist_rows(s, fams):
    return dist & (split == s) & np.isin(fam, list(fams))


SPECS = {"baseline": (base_tr, dev_clean)}
SPECS["rad_all"] = (base_tr | dist_rows("train", FAMS), dev_clean | dist_rows("dev", FAMS))
for f in FAMS:
    rest = [g for g in FAMS if g != f]
    SPECS[f"lofo_{f}"] = (base_tr | dist_rows("train", rest), dev_clean | dist_rows("dev", rest))
for k, (tr, dv) in SPECS.items():
    say(f"decoder {k:24s} train rows {tr.sum():5d}  dev rows {dv.sum():4d}")

# tier pools: (tier, decoder, family) -> mask of distractor test rows
POOLS = []
for f in FAMS:
    for g, tier in (("seen", "A"), ("unseen", "B")):
        m = dist & (split == "test") & (fam == f) & (group == g)
        POOLS += [(tier, "rad_all", f, m), (f"pre-{tier}", "baseline", f, m)]
    mC = dist & (split == "test") & (fam == f)
    POOLS += [("C", f"lofo_{f}", f, mC), ("pre-C", "baseline", f, mC)]


def evaluate_preds(cell, preds: dict, n_boot: int) -> list[dict]:
    rows = []
    for k, p in preds.items():
        rows.append(dict(cell=cell, decoder=k, tier="clean", family="-", **accuracy(p[test_clean], y[test_clean])))
    for tier, k, f, m in POOLS:
        if k not in preds or not m.any():
            continue
        p = preds[k]
        st = pull_stats(p[m], p[twin[m]], y[m], log_d[m], clusters[m], n_boot=n_boot)
        rows.append(dict(cell=cell, decoder=k, tier=tier, family=f, **st))
    return rows


# ------------------------------------------------------------------ behavior gate (no activations needed)
clean_plain = main & (df.rendering == "plain").to_numpy()
beta_h = horizon_effect(df[clean_plain])
say(f"\nbehavior: beta_H = {beta_h:+.3f} short log-odds per decade of horizon (clean plain prompts)")
beh = []
for f in FAMS:
    m = dist & (fam == f)
    b = behavior_pull(df[m], df.iloc[twin[m]], beta_h, n_boot=a.n_boot)
    b["family"] = f
    b["irrelevant"] = bool(-a.behavior_gate < b["behavior_pull_lo"] and b["behavior_pull_hi"] < a.behavior_gate)
    beh.append(b)
    say(f"  {f:18s} n={b['n']:4d}  behavior pull {b['behavior_pull']:+.3f} [{b['behavior_pull_lo']:+.3f}, {b['behavior_pull_hi']:+.3f}]"
        f"  choice flips {b['choice_flip_rate']:.3f}  -> {'irrelevant' if b['irrelevant'] else 'NOT irrelevant: analysed separately'}")
beh = pd.DataFrame(beh); beh.to_csv(out / "behavior_gate.csv", index=False)
eligible = beh[beh.irrelevant].family.tolist()

# ------------------------------------------------------------------ text baselines
text = df.text.to_numpy()
text_preds = {"regex_first": np.array([regex_first(t) for t in text]), "regex_horizon": np.array([regex_horizon(t) for t in text])}
rows = []
for name, p in text_preds.items():
    rows += evaluate_preds(f"text:{name}", {k: p for k in SPECS}, n_boot=0)
tfidf = {}
for k, (tr, _) in SPECS.items():
    tfidf[k] = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), Ridge(alpha=1.0)).fit(text[tr], y[tr]).predict(text)
rows += evaluate_preds("text:tfidf_ridge", tfidf, n_boot=0)
pd.DataFrame(rows).to_csv(out / "text_baselines.csv", index=False)
say(f"text baselines done ({time.time() - T0:.0f}s)")

# ------------------------------------------------------------------ activation decoders: sweep all cells
sweep_rows, selection, all_preds = [], [], {}
for l in layers:
    XL = run.get_layer(l)                                    # float32 [n, n_pos, d]
    for pos in positions:
        X = XL[:, run.pos_index(pos)]
        cell = f"L{l}:{pos}"
        preds = {}
        for k, (tr, dv) in SPECS.items():
            rp = RidgePath(X[tr], y[tr])
            dev_rmse = {al: float(np.sqrt(np.mean((rp.predict(X[dv], al) - y[dv]) ** 2))) for al in alphas}
            best = min(dev_rmse, key=dev_rmse.get)
            preds[k] = rp.predict(X, best)
            selection.append(dict(cell=cell, decoder=k, alpha=best, dev_rmse=dev_rmse[best]))
        all_preds[cell] = preds
        sweep_rows += evaluate_preds(cell, preds, n_boot=0)
    del XL
    say(f"layer {l} done ({time.time() - T0:.0f}s)")
sel = pd.DataFrame(selection); sel.to_csv(out / "selection.csv", index=False)
sweep = pd.DataFrame(sweep_rows); sweep.to_csv(out / "sweep.csv", index=False)
np.savez_compressed(out / "predictions.npz", sample_uid=df.sample_uid.to_numpy(),
                    **{f"{c}|{k}": p.astype(np.float32) for c, d in all_preds.items() for k, p in d.items()})

# ------------------------------------------------------------------ headline cells with bootstrap CIs
cells = []
for c in a.cells_full.split(","):
    if c == "auto":
        c = sel[sel.decoder == "rad_all"].sort_values("dev_rmse").iloc[0].cell
    if c in all_preds and c not in cells:
        cells.append(c)
full = pd.DataFrame([r for c in cells for r in evaluate_preds(c, all_preds[c], n_boot=a.n_boot)])
full.to_csv(out / "headline_cells.csv", index=False)


# ------------------------------------------------------------------ decision rules
def verdict(tab: pd.DataFrame, f: str) -> str:
    g = lambda tier: tab[(tab.tier == tier) & (tab.family == f)].iloc[0]
    pre, A, B, C = g("pre-C"), g("A"), g("B"), g("C")
    if pre.pull < a.min_baseline_pull:
        return "no pull to remove (baseline pull below threshold)"
    removed = lambda r: r.pull_hi < a.removed
    if removed(C):
        return "SEPARABLE: pull removed for a role never seen in training"
    if removed(A) and removed(B):
        return "ROLE-LEVEL ONLY: removed for trained roles (new wording too), not for a new role"
    if removed(A):
        return "WORDING ONLY: removed for trained sentences, not for new wording"
    if A.pull >= 0.5 * pre.pull:
        return "ENTANGLED: pull persists even for trained sentences"
    return "PARTIAL: pull reduced but not removed"


md = ["# Relevance-aware decoder: results", "",
      f"Run `{a.run_dir}` · {len(df)} prompts · generated {time.strftime('%Y-%m-%d %H:%M')} · thresholds: "
      f"behavior gate ±{a.behavior_gate}, removed if CI upper < {a.removed}, min baseline pull {a.min_baseline_pull}, "
      f"max clean drop {a.max_clean_drop}", "",
      "## Behavior gate", "", markdown_table(beh[["family", "n", "behavior_pull", "behavior_pull_lo", "behavior_pull_hi", "choice_flip_rate", "irrelevant"]]), ""]
for c in cells:
    tab = full[full.cell == c]
    clean = tab[tab.tier == "clean"].set_index("decoder")
    drop = clean.loc["baseline", "within2x"] - clean.loc["rad_all", "within2x"]
    md += [f"## Cell {c}", "", f"Clean test within-2x: baseline {clean.loc['baseline', 'within2x']:.3f}, rad_all "
           f"{clean.loc['rad_all', 'within2x']:.3f} (drop {drop:+.3f}; {'OK' if drop <= a.max_clean_drop else 'TOO LARGE'})", "",
           "| family | pre (baseline) | A | B | C | verdict |", "|---|---|---|---|---|---|"]
    for f in FAMS:
        cellfmt = lambda tier: (lambda r: f"{r.pull:+.2f} [{r.pull_lo:+.2f}, {r.pull_hi:+.2f}]")(tab[(tab.tier == tier) & (tab.family == f)].iloc[0])
        v = verdict(tab, f) if f in eligible else "excluded: behavior moves with this distractor (see behavior gate)"
        md += [f"| {f} | {cellfmt('pre-C')} | {cellfmt('A')} | {cellfmt('B')} | {cellfmt('C')} | {v} |"]
    md += [""]
md += ["## Log", "", "```", *log_lines, "```"]
(out / "summary.md").write_text("\n".join(md))
json.dump(dict(cells=cells, eligible=eligible, beta_h=beta_h, args=vars(a)), open(out / "summary.json", "w"), indent=2)

# ------------------------------------------------------------------ sweep figure
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, (tier, title) in zip(axes, [("pre-C", "baseline decoder"), ("A", "rad_all, tier A"), ("C", "leave-one-family-out, tier C")]):
        s = sweep[(sweep.tier == tier) & sweep.family.isin(eligible or FAMS)].groupby("cell").pull.mean()
        M = np.array([[s.get(f"L{l}:{p}", np.nan) for p in positions] for l in layers])
        im = ax.imshow(M, vmin=-0.2, vmax=1.0, cmap="viridis", aspect="auto", origin="lower")
        ax.set_xticks(range(len(positions)), positions); ax.set_yticks(range(len(layers)), layers); ax.set_title(f"mean pull: {title}")
    axes[0].set_ylabel("layer"); fig.colorbar(im, ax=axes, label="pull fraction (0 = ignores distractor, 1 = follows it)")
    fig.savefig(out / "sweep_pull.png", dpi=130, bbox_inches="tight")
except Exception as e:  # figure is a convenience; never fail the run on it
    say("figure skipped:", e)
say(f"wrote {out} ({time.time() - T0:.0f}s)")
