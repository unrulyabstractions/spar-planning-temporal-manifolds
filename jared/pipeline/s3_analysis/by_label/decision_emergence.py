"""When is the choice readable, beyond what the prompt's surface predicts? (post hoc, 2026-11-03)

Saved activations only. Within the regimes where the thinking-on choice varies (none:
no payout arrives in time; both: both do, and the 8B still picks near on ~40% of the
old-bank prompts), a probe reads chose-near from the residual stream at each kept
position, from the prompt-final tokens before any reasoning to the answer slot.

Three readers, all out of fold (GroupKFold over base prompts, so renderings of one
base never straddle a split), scored by AUC within a regime:
  surface   logistic on the prompt's metadata: log near / far delay, log near reward,
            log reward ratio, near first, log unit size, log horizon, bank, domain,
            phrasing (one-hot). What a reader of the record alone gets.
  text      logistic on TF-IDF word 1-2 grams of the prompt text. Anything in the
            wording (situation, role, numbers as tokens) the metadata leaves out.
  acts      logistic on the top-64 PCs (fit on the training fold) of one cell.
  acts+surface  both.
The question is the gap acts - max(surface, text) at the prompt-final tokens: if the
eventual choice is readable before reasoning well beyond the surface readers, the
model has a disposition before it writes the reasoning (cf. "reasoning theater",
answer-before-CoT probes). The think positions then show how the readability grows.

Rows: natural closes only (a forced close cuts the reasoning and pulls the choice
toward 50/50), prompts with a horizon. 8B: full + units + fill pooled (anchors out),
as the pooled fits; 14B: full.

    python -m pipeline.s3_analysis.by_label.decision_emergence /home/Jared/SPAR/results/horizon OUT_DIR [on|off]
    -> OUT_DIR/decision_emergence.{json,txt,png}; decision_emergence_off.* for thinking off,
       the positive control: there the choice is made in the forward pass from the prompt,
       so information beyond the surface must be present by the answer slot
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

from pipeline.s3_analysis.by_label.confound_sweep import FEATURES, dataset
from pipeline.s1_data_vectors.analysis.probe_dimensionality import positions
from pipeline.s3_analysis.by_datatype.unit_overlap import load_run
from spar_horizon.e0_runs import run_records
from spar_horizon.runs import REGIONS

RUNS = {"Qwen3-8B": ("full", "units", "fill"), "Qwen3-14B": ("full",)}
# layer 0 is the token embedding: constant across prompts at the fixed-token positions
LAYERS = {"Qwen3-8B": (2, 4, 8, 12, 16, 20, 22, 24, 28, 32, 36),
          "Qwen3-14B": (2, 5, 10, 15, 20, 25, 27, 30, 35, 40)}
POSITIONS = {"on": ("suffix newline", "suffix last", "think@0.25", "think@0.5", "think@1", "think close", "pre-label"),
             # thinking off: the empty think block is prefilled, so "suffix last" is the token the answer starts from
             "off": ("suffix newline", "suffix last", "pre-label")}
REGIMES = ("none", "both")
N_FOLDS = 5
N_PCS = 64
BOOT = 200
SURFACE_NUMERIC = ("log near delay", "log far delay", "log reward", "log ratio", "near first", "log unit size")


def load(root, model, mode="on"):
    """Pooled rows (as confound_sweep.dataset) plus each row's record, for the surface readers."""
    parts, data = dataset(root, model, mode, RUNS[model])
    uids, banks = [], []
    for bank in RUNS[model]:
        _, labels = load_run(root, mode, bank, model)
        keep = ~labels["anchor"]
        uids.extend(labels["sample_uids"][keep])
        banks.extend([bank] * int(keep.sum()))
    records = {}
    for bank in RUNS[model]:
        records.update(run_records(root, f"exp0_{model}_think-{mode}_{bank}"))
    recs = [records[u] for u in uids]
    data["bank"] = np.array(banks)
    data["prompt"] = np.array([r["prompt"] for r in recs])
    data["domain"] = np.array([r.get("domain", "") for r in recs])
    data["phrasing"] = np.array([str(r.get("phrasing_id", "")) for r in recs])
    data["natural_close"] = data["features"][:, FEATURES.index("forced")] == 0
    return parts, data


def cell(parts, position, layer):
    blocks = []
    for z, rows in parts:
        meta = {"regions": {r: tuple(int(x) for x in z[f"region_{r}"]) for r in REGIONS},
                "tokens": [str(t) for t in z["tokens"]]}
        blocks.append(z[f"L{layer}"][rows, positions(meta)[position]].astype(np.float64))
    return np.concatenate(blocks)


def surface_matrix(data, rows):
    numeric = np.column_stack([data["features"][rows, FEATURES.index(f)] for f in SURFACE_NUMERIC] + [data["y"][rows]])
    categorical = np.column_stack([data["bank"][rows], data["domain"][rows], data["phrasing"][rows]])
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(categorical)
    return np.column_stack([StandardScaler().fit_transform(numeric), onehot])


def classifier():
    return LogisticRegressionCV(Cs=np.logspace(-4, 2, 13), cv=4, max_iter=5000, scoring="neg_log_loss")


def out_of_fold(make_features, y, groups):
    """Out-of-fold P(chose near). make_features(train_idx, test_idx) -> (X_train, X_test)."""
    pred = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=N_FOLDS).split(y, y, groups):
        X_train, X_test = make_features(train, test)
        pred[test] = classifier().fit(X_train, y[train]).predict_proba(X_test)[:, 1]
    return pred


def fixed_features(X):
    return lambda train, test: (X[train], X[test])


def text_features(texts):
    def make(train, test):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        return vectorizer.fit_transform(texts[train]), vectorizer.transform(texts[test])
    return make


def pca_features(X, extra=None):
    def make(train, test):
        model = make_pipeline(StandardScaler(), PCA(n_components=min(N_PCS, len(train) - 1), random_state=0))
        A_train, A_test = model.fit_transform(X[train]), model.transform(X[test])
        # PC1 at unit scale, so the PCs and the standardized surface columns share one penalty
        A_train, A_test = A_train / A_train[:, :1].std(), A_test / A_train[:, :1].std()
        if extra is not None:
            A_train, A_test = np.column_stack([A_train, extra[train]]), np.column_stack([A_test, extra[test]])
        return A_train, A_test
    return make


def auc_ci(y, pred, groups, rng):
    """AUC and a 95% CI that resamples base prompts."""
    point = roc_auc_score(y, pred)
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    draws = []
    for _ in range(BOOT):
        rows = np.concatenate([index[g] for g in rng.choice(unique, len(unique))])
        if 0 < y[rows].mean() < 1:
            draws.append(roc_auc_score(y[rows], pred[rows]))
    return [float(point), *map(float, np.percentile(draws, [2.5, 97.5]))]


def paired_gap(y, pred_a, pred_b, groups, rng):
    """AUC(a) - AUC(b) with a CI from the same base-prompt resamples."""
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    point = roc_auc_score(y, pred_a) - roc_auc_score(y, pred_b)
    draws = []
    for _ in range(BOOT):
        rows = np.concatenate([index[g] for g in rng.choice(unique, len(unique))])
        if 0 < y[rows].mean() < 1:
            draws.append(roc_auc_score(y[rows], pred_a[rows]) - roc_auc_score(y[rows], pred_b[rows]))
    return [float(point), *map(float, np.percentile(draws, [2.5, 97.5]))]


def run_model(root, model, rng, mode="on"):
    parts, data = load(root, model, mode)
    selections = {regime: np.flatnonzero((data["regime"] == regime) & data["natural_close"]) for regime in REGIMES}
    result = {}
    for regime, rows in selections.items():
        y, groups = data["chose_near"][rows].astype(int), data["pairs"][rows]
        surface = surface_matrix(data, rows)
        baseline = {"surface": out_of_fold(fixed_features(surface), y, groups),
                    "text": out_of_fold(text_features(data["prompt"][rows]), y, groups)}
        result[regime] = {"n": int(len(rows)), "near share": float(y.mean()),
                          "by bank": {b: [int((data["bank"][rows] == b).sum()), float(y[data["bank"][rows] == b].mean())]
                                      for b in RUNS[model] if (data["bank"][rows] == b).any()},
                          "surface": auc_ci(y, baseline["surface"], groups, rng),
                          "text": auc_ci(y, baseline["text"], groups, rng), "cells": {}}
        best_surface = max(baseline, key=lambda k: result[regime][k][0])
        result[regime]["best surface"] = best_surface
        baseline_pred = baseline[best_surface]
        result[regime]["_pred"] = {"y": y, "groups": groups, "baseline": baseline_pred, "surface_X": surface}
    jobs = [(p, layer) for p in POSITIONS[mode] for layer in LAYERS[model]]
    for position, layer in tqdm(jobs, desc=model):
        X_all = cell(parts, position, layer)
        for regime, rows in selections.items():
            keep = result[regime]["_pred"]
            y, groups = keep["y"], keep["groups"]
            acts = out_of_fold(pca_features(X_all[rows]), y, groups)
            both = out_of_fold(pca_features(X_all[rows], extra=keep["surface_X"]), y, groups)
            result[regime]["cells"][f"{position} L{layer}"] = {
                "acts": auc_ci(y, acts, groups, rng), "acts+surface": auc_ci(y, both, groups, rng),
                "acts - best surface": paired_gap(y, acts, keep["baseline"], groups, rng)}
    for regime in result:
        result[regime].pop("_pred")
    return result


def fmt(ci):
    return f"{ci[0]:.2f} [{ci[1]:.2f}, {ci[2]:.2f}]"


def summary_lines(results):
    lines = ["Decision emergence: AUC of chose-near within a regime, natural closes, out of fold (base-prompt groups)", ""]
    for model, by_regime in results.items():
        for regime, r in by_regime.items():
            lines.append(f"{model}  regime {regime}: n {r['n']}, near share {r['near share']:.2f}, by bank "
                         + ", ".join(f"{b} {n} ({s:.2f})" for b, (n, s) in r["by bank"].items()))
            lines.append(f"  surface {fmt(r['surface'])}   text {fmt(r['text'])}   (best: {r['best surface']})")
            lines.append(f"  {'cell':24s} {'acts':22s} {'acts+surface':22s} acts - best surface")
            for name, c in r["cells"].items():
                lines.append(f"  {name:24s} {fmt(c['acts']):22s} {fmt(c['acts+surface']):22s} {fmt(c['acts - best surface'])}")
            lines.append("")
    return lines


def plot(results, path, mode="on"):
    fig, axes = plt.subplots(len(results), len(REGIMES), figsize=(11, 3.6 * len(results)), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(POSITIONS[mode])))
    for i, (model, by_regime) in enumerate(results.items()):
        for j, regime in enumerate(REGIMES):
            ax, r = axes[i, j], by_regime[regime]
            for color, position in zip(colors, POSITIONS[mode]):
                layers = LAYERS[model]
                values = [r["cells"][f"{position} L{layer}"]["acts"][0] for layer in layers]
                ax.plot(layers, values, "-o", ms=3, color=color, label=position)
            for name, style in (("surface", "--"), ("text", ":")):
                ax.axhline(r[name][0], color="gray", ls=style, lw=1, label=f"{name} (prompt only)")
            ax.axhline(0.5, color="k", lw=0.5)
            ax.set_title(f"{model}, regime {regime} (n {r['n']}, near {r['near share']:.2f})", fontsize=10)
            ax.set_xlabel("layer")
            ax.set_ylabel("AUC, chose near")
            ax.set_ylim(0.4, 1.0)
    axes[0, -1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(root, out_dir, mode="on"):
    root, out_dir = Path(root), Path(out_dir)
    stem = "decision_emergence" if mode == "on" else f"decision_emergence_{mode}"
    rng = np.random.default_rng(0)
    results = {model: run_model(root, model, rng, mode) for model in RUNS}
    (out_dir / f"{stem}.json").write_text(json.dumps(results, indent=1))
    text = "\n".join([f"thinking {mode}"] + summary_lines(results))
    (out_dir / f"{stem}.txt").write_text(text)
    plot(results, out_dir / f"{stem}.png", mode)
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:4])
