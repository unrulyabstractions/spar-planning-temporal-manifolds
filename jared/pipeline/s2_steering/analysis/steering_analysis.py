"""E8 analysis: is the answer-slot "horizon subspace" the horizon or the decision?
And figures for both steering runs (spec: 07_spec_steering.md, results:
08_results_steering.md).

At the answer slot late in the network, horizon averages differ mostly because
the decision differs by horizon, so ablating their subspace could remove the
decision by construction. For each answer-slot cell this asks, on the saved E0
activations and the saved label logits (the decision, per prompt):
- how well the subspace predicts the decision (a-vs-b and near-vs-far log-odds)
  and the horizon;
- whether it still separates horizons among prompts that made the same choice;
- how much horizon information it adds beyond the decision itself.
All R² are 5-fold cross-validated ridge, from the top-4 horizon-average PCs.

    python -m pipeline.s2_steering.analysis.steering_analysis results/horizon
    -> <root>/analysis/steering_decision_check.json and .txt,
       steering_patching.png, steering_cells.png, steering_subspace.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from pipeline.s1_data_vectors.analysis.horizon_shape import cell_data
from spar_horizon.steering import bootstrap_mean

JARED = Path(__file__).resolve().parents[3]
INPUTS = JARED / "steering" / "inputs_Qwen3-8B_answer_slot.npz"
RUNS = {"first": "exp8_Qwen3-8B_full", "second": "exp8_Qwen3-8B_answer_slot"}
K = 4
NORMAL_LEAD = 3
BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893", "#52514e"


def cv_r2(features, target, folds=5, seed=0):
    """Pooled out-of-fold R² of a ridge (penalty by internal CV) on standardized features."""
    features = np.atleast_2d(np.asarray(features, float).T).T
    pred = np.empty(len(target))
    for train, test in KFold(folds, shuffle=True, random_state=seed).split(features):
        scaler = StandardScaler().fit(features[train])
        model = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(scaler.transform(features[train]), target[train])
        pred[test] = model.predict(scaler.transform(features[test]))
    return float(1 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def saved_decision(run, z):
    """The decision per horizon prompt of a saved E0 run (rows as in cell_data):
    a-vs-b log-odds, near-vs-far log-odds, chose_near. Rows whose lead is not NORMAL_LEAD
    have NaN log-odds: runs saved before extract.label_pair read the spaced ' a'/' b' logits
    there, but after a markdown lead like `**b)**` the label is unspaced, so those logits are
    for the wrong token pair. Callers drop the NaN rows."""
    answers = json.loads((run / "answers.json").read_text())
    keep = ~np.isnan(z["horizons"])
    rows = [a for a, k in zip(answers, keep) if k]
    ab = z["label_logits"][keep, 0] - z["label_logits"][keep, 1]
    ab[z["lead"][keep] != NORMAL_LEAD] = np.nan
    near_is_a = np.array([r["near_label"] == "a" for r in rows])
    near = np.where(near_is_a, ab, -ab)
    chose_near = np.array([r["label"] == r["near_label"] for r in rows])
    return ab, near, chose_near


def answer_slot(root, mode, layer, cell):
    """Subspace coordinates Z [n, K], log horizon, and the decision per horizon prompt:
    a-vs-b log-odds, near-vs-far log-odds, chose_near."""
    run = root / f"exp0_Qwen3-8B_think-{mode}_full"
    z = np.load(run / "activations.npz")
    X, log_h, _ = cell_data(z, "pre-label", layer)
    ab, near, chose_near = saved_decision(run, z)
    valid = np.isfinite(ab)
    Z = (X - cell["mean"]) @ cell["basis"][:K].T
    return Z[valid], log_h[valid], ab[valid], near[valid], chose_near[valid]


def answer_slot_activations(root, mode, layer):
    """The activations X [n, d] behind answer_slot's rows."""
    run = root / f"exp0_Qwen3-8B_think-{mode}_full"
    z = np.load(run / "activations.npz")
    X, _, _ = cell_data(z, "pre-label", layer)
    ab, _, _ = saved_decision(run, z)
    return X[np.isfinite(ab)]


def cv_r2_refit_subspace(X, log_h, target, extra=None, folds=5, seed=0):
    """cv_r2 on the top-K PCs of the horizon averages (as fit_cell), refit inside each fold on its
    training prompts only, so the test prompts never shape the subspace. `extra` columns are
    appended to the PC coordinates."""
    pred = np.empty(len(target))
    for train, test in KFold(folds, shuffle=True, random_state=seed).split(X):
        mean = X[train].mean(0)
        groups = np.round(log_h[train], 9)
        averages = np.stack([X[train][groups == h].mean(0) - mean for h in np.unique(groups)])
        basis = PCA(n_components=K).fit(averages).components_
        features = (X - mean) @ basis.T
        if extra is not None:
            features = np.c_[features, extra]
        scaler = StandardScaler().fit(features[train])
        model = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(scaler.transform(features[train]), target[train])
        pred[test] = model.predict(scaler.transform(features[test]))
    return float(1 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def decision_check(root, cells):
    out = {}
    for key in cells:
        mode, _, layer = key.split("|")
        X = answer_slot_activations(root, mode, int(layer))
        _, log_h, ab, near, chose_near = answer_slot(root, mode, int(layer), cells[key])
        decision = np.c_[ab, near]
        r2_dec = cv_r2(decision, log_h)
        out[key] = {
            "n": int(len(log_h)), "chose_near": int(chose_near.sum()),
            "horizon_from_subspace": cv_r2_refit_subspace(X, log_h, log_h),
            "horizon_from_decision": r2_dec,
            "horizon_gain_beyond_decision": cv_r2_refit_subspace(X, log_h, log_h, extra=decision) - r2_dec,
            "ab_from_subspace": cv_r2_refit_subspace(X, log_h, ab),
            "near_from_subspace": cv_r2_refit_subspace(X, log_h, near),
            "horizon_within_chose_near": cv_r2_refit_subspace(X[chose_near], log_h[chose_near], log_h[chose_near]),
            "horizon_within_chose_far": cv_r2_refit_subspace(X[~chose_near], log_h[~chose_near], log_h[~chose_near]),
        }
    return out


def check_text(check):
    lines = ["Answer-slot horizon subspace (top-4 horizon-average PCs): horizon or decision?",
             "5-fold CV R². 'decision' = a-vs-b and near-vs-far log-odds at the answer slot.", "",
             f"{'cell':18} {'h|sub':>6} {'h|dec':>6} {'gain':>6} {'ab|sub':>7} {'near|sub':>8} "
             f"{'h|sub, chose near':>18} {'h|sub, chose far':>17}"]
    for key, c in check.items():
        lines.append(f"{key:18} {c['horizon_from_subspace']:6.2f} {c['horizon_from_decision']:6.2f} "
                     f"{c['horizon_gain_beyond_decision']:+6.2f} {c['ab_from_subspace']:7.2f} "
                     f"{c['near_from_subspace']:8.2f} {c['horizon_within_chose_near']:18.2f} "
                     f"{c['horizon_within_chose_far']:17.2f}")
    return "\n".join(lines)


# --- figures ---------------------------------------------------------------------

def plot_patching(first, second, path):
    """Left: the first run's patching map. Right: transfer at the answer slot by layer."""
    patch = first["patch"]
    groups = [g.strip("'").replace("<|", "").replace("|>", "") for g in patch["groups"]]
    M = np.array([[v[0] for v in row] for row in patch["transfer"]])[1:]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.35, 1]})
    im = ax0.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="lower",
                    extent=(-0.5, len(groups) - 0.5, 0.5, M.shape[0] + 0.5))
    ax0.set_xticks(range(len(groups)), [g.replace("\n", "\\n") for g in groups], rotation=60, ha="right", fontsize=8)
    ax0.set_ylabel("layer")
    ax0.set_title("Where the horizon effect travels (thinking off, patch one position group)", fontsize=10)
    fig.colorbar(im, ax=ax0, fraction=0.035, label="share of the choice shift carried")
    col = patch["groups"].index("':'")
    series = [("patch ':' alone, prompts differing only in horizon (run 1)", [r[col] for r in patch["transfer"]], BLUE),
              ("answer slot across prompts, thinking off (run 2)", second["answer"]["off"]["transfer"], ORANGE),
              ("answer slot across prompts, thinking on (run 2)", second["answer"]["on"]["transfer"], AQUA)]
    for name, values, color in series:
        layers = np.arange(1, len(values))
        mean = np.array([v[0] for v in values[1:]])
        low, high = np.array([v[1] for v in values[1:]]), np.array([v[2] for v in values[1:]])
        ax1.fill_between(layers, low, high, color=color, alpha=0.18, linewidth=0)
        ax1.plot(layers, mean, color=color, lw=2, label=name)
    ax1.axhline(0, color=INK, lw=0.6)
    ax1.set_xlabel("layer")
    ax1.set_ylabel("share of the choice shift carried by the answer slot")
    ax1.set_title("The decision reaches the answer slot at layers 22-24, with or without thinking", fontsize=10)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    ax1.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cell_label(key, run):
    mode, label, layer = key.split("|")
    return f"{'think ' + mode}\n{label}\nL{layer}" + ("\n(run 1)" if run == "first" else "")


def toward(steer, direction, alpha):
    return np.asarray(steer[direction][str(alpha)]["toward"], float)


def plot_cells(first, second, path):
    """Ablation removed share by k, and steering effect toward the rational answer at 2 decades."""
    cells = [(k, "first", v) for k, v in first["cells"].items()] + [(k, "second", v) for k, v in second["cells"].items()]
    names = [cell_label(k, run) for k, run, _ in cells]
    x = np.arange(len(cells))
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 9.5), sharex=True)
    ks, shades = ("1", "2", "4", "8"), plt.get_cmap("Blues")(np.linspace(0.35, 0.95, 4))
    width = 0.18
    for j, (k, shade) in enumerate(zip(ks, shades)):
        values = [res["ablate"].get(k, {}).get("horizon", [np.nan])[0] for _, _, res in cells]
        ax0.bar(x + (j - 1.5) * width, values, width * 0.92, color=shade, label=f"horizon subspace, k={k}")
    random_max = [max((v[0] for n, v in row.items() if n.startswith("random")), default=np.nan)
                  for _, _, res in cells for row in [next(r for r in res["ablate"].values()
                                                          if any(n.startswith("random") for n in r))]]
    ax0.scatter(x + 0.45, random_max, marker="_", s=260, color=GRAY, linewidths=2.5,
                label="random subspace (worst of 3, at the judged k)", zorder=3)
    ax0.axhline(0.5, color=INK, lw=0.8, ls="--")
    ax0.text(-0.55, 0.52, "necessary: 0.5", ha="left", fontsize=8, color=INK)
    ax0.set_ylabel("share of the near-vs-both contrast removed")
    ax0.set_title("Ablating the horizon-average subspace at each cell", fontsize=10)
    ax0.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    ax0.grid(axis="y", alpha=0.2)
    directions = [("line", "probe line", BLUE), ("curve", "along the curve", ORANGE),
                  ("mean difference", "mean difference", AQUA), ("random 0", "random", GRAY)]
    width = 0.2
    for j, (direction, name, color) in enumerate(directions):
        stats = [bootstrap_mean(np.r_[toward(res["steer"], direction, 2), toward(res["steer"], direction, -2)])
                 for _, _, res in cells]
        mean = np.array([s[0] for s in stats])
        err = np.array([[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]])
        ax1.bar(x + (j - 1.5) * width, mean, width * 0.92, color=color, label=name, yerr=err,
                error_kw={"elinewidth": 0.8, "ecolor": INK})
    ax1.axhline(0, color=INK, lw=0.6)
    ax1.set_ylabel("effect toward the rational answer\n(share of contrast, 2 decades)")
    ax1.set_title("Steering by 2 decades: every direction moves as far as the curve does", fontsize=10)
    ax1.legend(fontsize=8, frameon=False, ncol=4, loc="upper left")
    ax1.grid(axis="y", alpha=0.2)
    ax1.set_xticks(x, names, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_subspace(root, cells, path):
    """Top-2 subspace coordinates at four answer-slot cells, colored by horizon,
    shaped by the choice: two clusters by choice means the subspace is the decision."""
    keys = [k for k in ("off|pre-label|24", "off|pre-label|30", "on|pre-label|24", "on|pre-label|30") if k in cells]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        mode, _, layer = key.split("|")
        Z, log_h, _, _, chose_near = answer_slot(root, mode, int(layer), cells[key])
        norm = plt.Normalize(log_h.min(), log_h.max())
        for mask, marker, name in ((chose_near, "o", "chose near"), (~chose_near, "^", "chose far")):
            points = ax.scatter(Z[mask, 0], Z[mask, 1], c=log_h[mask], cmap="Blues", norm=norm, s=16, marker=marker,
                                edgecolors=INK, linewidths=0.3, label=name)
        ax.set_title(f"thinking {mode}, answer slot, L{layer}", fontsize=9)
        ax.set_xlabel("horizon-average PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="best")
    bar = fig.colorbar(points, ax=axes[0].tolist(), fraction=0.02)
    ticks = {"1 min": 1 / 525_960, "1 h": 1 / 8_766, "1 day": 1 / 365.25, "1 mo": 1 / 12, "1 y": 1, "100 y": 100}
    bar.set_ticks([np.log10(v) for v in ticks.values()], labels=list(ticks))
    bar.set_label("horizon", fontsize=8)
    fig.suptitle("Answer slot: does the horizon subspace separate choices or horizons?", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(root):
    root = Path(root)
    meta = json.loads(INPUTS.with_suffix(".json").read_text())
    z = np.load(INPUTS)
    cells = {f"{c['mode']}|{c['label']}|{c['layer']}": {name: z[f"{c['mode']}|{c['label']}|{c['layer']}|{name}"]
                                                        for name in ("mean", "basis")} for c in meta["cells"]}
    first, second = (json.loads((root / run / "results.json").read_text()) for run in RUNS.values())
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    check = decision_check(root, cells)
    (out / "steering_decision_check.json").write_text(json.dumps(check, indent=1))
    text = check_text(check)
    (out / "steering_decision_check.txt").write_text(text)
    print(text)
    plot_patching(first, second, out / "steering_patching.png")
    plot_cells(first, second, out / "steering_cells.png")
    plot_subspace(root, cells, out / "steering_subspace.png")
    print(f"\nwrote {out}/steering_decision_check.json/.txt, steering_patching.png, steering_cells.png, "
          "steering_subspace.png")


if __name__ == "__main__":
    main(sys.argv[1])
