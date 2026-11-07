"""Is the break-even rate computed before reasoning? (post hoc, 2026-11-03)

breakeven_rate.py: the choice follows log10 r*, the annual rate at which the near reward
reinvested until the far delay matches the far reward. decision_emergence.py: before
reasoning the activations read the choice no better than the prompt's surface. Here:
is r* itself readable from the prompt-end activations beyond what the surface
ingredients give?

Target log10 r* (capped as in breakeven_rate). Readers, out of fold (5 GroupKFolds over
base prompts), R²:
  ingredients   ridge on log ratio, log near delay, log far delay, log delay gap and their
                squares and pairwise products (a quadratic surface: r* is a smooth function
                of these, so this is a strong prompt-only baseline)
  acts          ridge on the top-64 PCs of the cell (z-scored dims, as decision_emergence)
  acts+ingredients
  (first run: the quadratic surface reaches R² 0.998, so "beyond ingredients" cannot be
  detected this way; the informative comparison is the next one)
  linear in true ingredients     ridge on the four ingredients, no products: how nonlinear r* is
  linear in decoded ingredients  each ingredient read from the cell by its own out-of-fold
                                 probe, then a linear map from those four to r* (out of fold)
If acts beats "linear in decoded ingredients", the cell holds a nonlinear combination of
the ingredients, read linearly: a sign that something like r* is computed before reasoning.
Rows: 8B thinking on, full + units (the fill's capped r* would dominate), natural and
forced closes (r* is a prompt property), prompts with a horizon.

    python analysis/breakeven_decode.py /home/Jared/SPAR/results/horizon OUT_DIR
    -> OUT_DIR/breakeven_decode.{json,txt}
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from breakeven_rate import log10_breakeven  # noqa: E402
from decision_emergence import N_PCS, cell  # noqa: E402
from unit_overlap import load_run  # noqa: E402
from written_unit import run_records  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
MODEL, BANKS = "Qwen3-8B", ("full", "units")
CELLS = [("suffix newline", layer) for layer in (8, 16, 20, 24, 32)] + [("suffix last", 20)]


def load(root):
    parts, rows = [], []
    for bank in BANKS:
        z, labels = load_run(root, "on", bank, MODEL)
        records = run_records(root, f"exp0_{MODEL}_think-on_{bank}")
        parts.append((z, labels["rows"]))
        for uid, pair in zip(labels["sample_uids"], labels["pairs"]):
            r = records[uid]
            ratio = r["rewards"][1] / r["rewards"][0]
            gap = r["far_delay_years"] - r["near_delay_years"]
            rows.append({"group": f"{bank}|{pair}", "log_rate": log10_breakeven(ratio, gap),
                         "ingredients": [np.log10(ratio), np.log10(r["near_delay_years"]),
                                         np.log10(r["far_delay_years"]), np.log10(gap)]})
    return parts, rows


def ingredient_model():
    return make_pipeline(StandardScaler(), PolynomialFeatures(2), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))


def oof_r2(make, X, y, groups):
    pred = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        pred[test] = make().fit(X[train], y[train]).predict(X[test])
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def acts_model():
    return make_pipeline(StandardScaler(), PCA(n_components=N_PCS, random_state=0), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-2, 4, 13)))


def acts_plus_model(n_ingredients):
    """PCs of the activation columns, plus the quadratic ingredient terms, in one ridge."""
    def make():
        return make_pipeline(ColumnTransformer([
            ("acts", make_pipeline(StandardScaler(), PCA(n_components=N_PCS, random_state=0)), slice(n_ingredients, None)),
            ("ingredients", make_pipeline(StandardScaler(), PolynomialFeatures(2)), slice(0, n_ingredients))]),
            StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
    return make


def linear_model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))


def oof_predictions(make, X, y, groups):
    pred = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        pred[test] = make().fit(X[train], y[train]).predict(X[test])
    return pred


def decoded_ingredients_r2(X, ingredients, y, groups):
    """r* from the ingredients as the cell encodes them, fully nested: in each outer fold the
    ingredient probes are fit on the training rows only (their training-row predictions come
    from an inner out-of-fold pass), then a linear map from decoded ingredients to r* is fit
    on the training rows and applied to the test rows' decoded ingredients."""
    pred = np.zeros(len(y))
    decoded_all = np.zeros_like(ingredients)
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        decoded_train = np.column_stack([oof_predictions(acts_model, X[train], ingredients[train, k], groups[train])
                                         for k in range(ingredients.shape[1])])
        decoded_test = np.column_stack([acts_model().fit(X[train], ingredients[train, k]).predict(X[test])
                                        for k in range(ingredients.shape[1])])
        decoded_all[test] = decoded_test
        pred[test] = linear_model().fit(decoded_train, y[train]).predict(decoded_test)
    r2_each = [float(1 - ((ingredients[:, k] - decoded_all[:, k]) ** 2).sum() / ((ingredients[:, k] - ingredients[:, k].mean()) ** 2).sum())
               for k in range(ingredients.shape[1])]
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()), r2_each


def main(root, out_dir):
    root, out_dir = Path(root), Path(out_dir)
    parts, rows = load(root)
    y = np.array([r["log_rate"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    ingredients = np.array([r["ingredients"] for r in rows])
    result = {"n": len(y), "ingredients R²": oof_r2(ingredient_model, ingredients, y, groups),
              "linear in true ingredients R²": oof_r2(linear_model, ingredients, y, groups), "cells": {}}
    for position, layer in tqdm(CELLS):
        X = cell(parts, position, layer)
        decoded_r2, probe_r2 = decoded_ingredients_r2(X, ingredients, y, groups)
        result["cells"][f"{position} L{layer}"] = {
            "acts R²": oof_r2(acts_model, X, y, groups),
            "linear in decoded ingredients R²": decoded_r2,
            "ingredient probe R² (ratio, near, far, gap)": probe_r2,
            "acts+ingredients R²": oof_r2(acts_plus_model(ingredients.shape[1]), np.column_stack([ingredients, X]), y, groups)}
    (out_dir / "breakeven_decode.json").write_text(json.dumps(result, indent=1))
    lines = [f"log10 r* from the prompt end, 8B thinking on, {'+'.join(BANKS)}, n {result['n']}; out-of-fold R²",
             f"ingredients (quadratic in log ratio, log near, log far, log gap): {result['ingredients R²']:.3f}; "
             f"linear in the true ingredients: {result['linear in true ingredients R²']:.3f}", ""]
    for name, c in result["cells"].items():
        lines.append(f"  {name:20s} acts {c['acts R²']:.3f}   linear in decoded ingredients {c['linear in decoded ingredients R²']:.3f}"
                     f"   (ingredient probes {', '.join(f'{v:.2f}' for v in c['ingredient probe R² (ratio, near, far, gap)'])})"
                     f"   acts+ingredients {c['acts+ingredients R²']:.3f}")
    text = "\n".join(lines)
    (out_dir / "breakeven_decode.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:3])
