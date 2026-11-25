"""Spec 32, stage A: which Qwen3-8B MLP neurons track the horizon and the near-vs-far
decision at the suffix newline (L16) and the pre-label token (L22, L24), and are their
Neuron Lens labels about time or choice more often than chance?

A1 (read side): block L's neuron activations computed from the saved hidden_states[L]
with the MLP's own norm and weights, skipping block L's attention (not saved), so they
are approximate. Neurons are ranked by Spearman |rho| with each target; the pole whose
label applies is the sign the neuron's activation takes at the cell (its median), so a
neuron that fires positive and fades with the horizon is reported as pole + with rho < 0.

A2 (write side): cosine of each block L-1 neuron's write W_down[:, n] with the cell's
ridge direction for the target; the pole reported is the one whose write points toward
a larger target.

Thresholds: the 95th percentile, over permutations of the target (within unit for the
within-unit ranking) or over random directions (A2), of the largest |rho| or |cos| in the
cell. Labels come from the Neuron Lens catalog (stdlogits/neuron-lens, CC BY 4.0).

    python -m pipeline.s1_data_vectors.analysis.neuron_lens_compare $RESULTS
    -> <root>/analysis/neuron_lens_compare.json, .csv, .png

Stage B (`experiments/neuron_acts.py` on the box) gives the exact activations; with them,
A1 and B are ranked on the same prompts and compared (A2 does not change):

    python -m pipeline.s1_data_vectors.analysis.neuron_lens_compare $RESULTS \
        --exact $RESULTS/neuron_lens/neuron_acts_Qwen3-8B.npz
    -> <root>/analysis/neuron_lens_exact.json, .csv, .png
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import fisher_exact, rankdata
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

from pipeline.s1_data_vectors.analysis.horizon_shape import cell_data
from pipeline.s1_data_vectors.analysis.probe_dimensionality import units_of
from spar_horizon.e0_runs import saved_decision

RUN = "exp0_Qwen3-8B_think-on_full"
CELLS = (("suffix newline", 16, False), ("pre-label", 22, True), ("pre-label", 24, True))  # label, layer, natural only
RMS_EPS = 1e-6           # Qwen3-8B config.rms_norm_eps
N_PERMUTATIONS = 200
TOP = 20
RIDGE_ALPHAS = np.logspace(-1, 5, 13)
URL = "https://neuron-lens.com/qwen3-8b/L{layer}/n{neuron}"

# frozen in spec 32 before any result was seen
FAMILIES = {
    "time": ("time times timing duration durations period periods horizon horizons year years yearly annual "
             "month months monthly week weeks weekly day days daily hour hours hourly minute minutes decade "
             "decades century centuries millennium future later soon long-term short-term wait waiting delay "
             "delays delayed deadline temporal eventually tomorrow"),
    "choice": ("choice choices choose chooses chose option options decision decisions decide prefer preference "
               "select selection alternative alternatives reward rewards payout payouts"),
}
FAMILY_PATTERNS = {name: re.compile(r"(?<![\w-])(" + "|".join(re.escape(w) for w in words.split()) + r")(?![\w-])",
                                    re.IGNORECASE)
                   for name, words in FAMILIES.items()}


def load_bf16(path):
    """float32 array from the raw bf16 bits neuron_lens_fetch saved as uint16."""
    return (np.load(path).astype(np.uint32) << 16).view(np.float32)


def neuron_activations(H, gate, up, norm_weight, eps=RMS_EPS):
    """[n, n_neurons] activations silu(W_gate u) * (W_up u), u = RMSNorm(h): Qwen3's MLP
    neurons (the input of down_proj) when the MLP reads H [n, d]."""
    H = H.astype(np.float32)
    u = H / np.sqrt((H ** 2).mean(-1, keepdims=True) + eps) * norm_weight
    g = u @ gate.T
    return g / (1 + np.exp(-g)) * (u @ up.T)


def centre_within(values, groups):
    """`values` (rows = prompts) minus the mean of their group."""
    out = values.astype(np.float64).copy()
    for group in np.unique(groups):
        rows = groups == group
        out[rows] -= out[rows].mean(0)
    return out


def standardized_ranks(values):
    """Column ranks, centred and scaled so that x.T @ y / n is a Pearson correlation of ranks."""
    ranks = rankdata(values, axis=0)
    ranks -= ranks.mean(0)
    return ranks / ranks.std(0).clip(1e-12)


def spearman_columns(A, y):
    """Spearman rho of every column of A [n, m] with y [n]."""
    return standardized_ranks(A).T @ standardized_ranks(y[:, None])[:, 0] / len(y)


def permuted(y, groups, rng):
    """y shuffled, within each group when groups is given."""
    if groups is None:
        return rng.permutation(y)
    out = y.copy()
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        out[rows] = y[rng.permutation(rows)]
    return out


def max_abs_rho_threshold(A, y, groups, rng):
    """95th percentile of max_n |rho(A[:, n], permuted y)|."""
    R = standardized_ranks(A)
    maxima = [np.abs(R.T @ standardized_ranks(permuted(y, groups, rng)[:, None])[:, 0] / len(y)).max()
              for _ in range(N_PERMUTATIONS)]
    return float(np.quantile(maxima, 0.95))


def random_direction_threshold(W_down, rng):
    """95th percentile of max_n |cos(W_down[:, n], random direction)|."""
    columns = W_down / np.linalg.norm(W_down, axis=0)
    maxima = []
    for _ in range(N_PERMUTATIONS):
        v = rng.standard_normal(W_down.shape[0])
        maxima.append(np.abs(v @ columns).max() / np.linalg.norm(v))
    return float(np.quantile(maxima, 0.95))


def ridge_direction(X, y):
    """Unit-norm ridge coefficients of y on X (penalty by CV), oriented toward larger y."""
    coef = RidgeCV(alphas=RIDGE_ALPHAS).fit(X - X.mean(0), y - y.mean()).coef_
    return coef / np.linalg.norm(coef)


def matches(label_row, family):
    return bool(FAMILY_PATTERNS[family].search(f"{label_row['fires']} {label_row['writes']}"))


def enrichment(top_rows, layer_rows):
    """Per family: the matching share of the top poles vs all poles of the layer, one-sided Fisher p."""
    out = {}
    for family in FAMILIES:
        hits = sum(matches(r, family) for r in top_rows)
        base = sum(matches(r, family) for r in layer_rows)
        table = [[hits, len(top_rows) - hits], [base, len(layer_rows) - base]]
        out[family] = {"top": hits, "n_top": len(top_rows), "base_share": base / len(layer_rows),
                       "p": float(fisher_exact(table, alternative="greater")[1])}
    return out


def pole_rows(labels, layer, neurons, poles, scores):
    """Label rows of (neuron, pole), with the score and the explorer URL."""
    return [{**labels[(int(n), int(p))], "score": float(s), "url": URL.format(layer=layer, neuron=int(n))}
            for n, p, s in zip(neurons, poles, scores)]


def load_labels(lens_dir, block):
    rows = json.loads((lens_dir / f"labels_L{block}.json").read_text())
    return {(r["neuron"], r["pole"]): r for r in rows}, rows


def cell_targets(z, run, label, layer, natural_only):
    """X [n, d] at one cell and {target: (rows, y, groups or None)}, `rows` a mask over X's
    rows and y, groups over all of them; rows without a decision are left out of 'decision'."""
    X, log_h, forced = cell_data(z, label, layer)
    _, near_lo, _ = saved_decision(run, z)
    keep = ~forced if natural_only else np.ones(len(log_h), bool)
    return X, {"horizon raw": (keep, log_h, None),
               "horizon within unit": (keep, log_h, units_of(log_h)),
               "decision": (keep & np.isfinite(near_lo), near_lo, None)}


def read_side(A, y, groups, labels, label_rows, layer, rng):
    """Neurons ranked by Spearman |rho| of their activations A [n, neurons] with y."""
    pole = np.where(np.median(A, 0) >= 0, 1, -1)
    if groups is not None:
        A, y = centre_within(A, groups), centre_within(y[:, None], groups)[:, 0]
    rho = spearman_columns(A, y)
    threshold = max_abs_rho_threshold(A, y, groups, rng)
    order = np.argsort(-np.abs(rho))[:TOP]
    top = pole_rows(labels, layer, order, pole[order], rho[order])
    return {"n": len(y), "threshold": threshold, "n_pass": int((np.abs(rho) > threshold).sum()),
            "abs_rho": np.sort(np.abs(rho))[::-1], "rho": rho, "top": top, "enrichment": enrichment(top, label_rows)}


def write_side(X, y, groups, W_down, labels, label_rows, layer, rng):
    """A2 at one cell and target: block layer - 1's writes against the cell's ridge direction."""
    if groups is not None:
        X, y = centre_within(X, groups), centre_within(y[:, None], groups)[:, 0]
    direction = ridge_direction(X, y)
    cos = direction @ (W_down / np.linalg.norm(W_down, axis=0))
    threshold = random_direction_threshold(W_down, rng)
    order = np.argsort(-np.abs(cos))[:TOP]
    top = pole_rows(labels, layer - 1, order, np.sign(cos[order]).astype(int), np.abs(cos[order]))
    return {"n": len(y), "threshold": threshold, "n_pass": int((np.abs(cos) > threshold).sum()),
            "abs_cos": np.sort(np.abs(cos))[::-1], "top": top, "enrichment": enrichment(top, label_rows)}


def column_correlations(A, B):
    """Pearson r of each column of A with the same column of B."""
    A, B = A - A.mean(0), B - B.mean(0)
    return (A * B).sum(0) / np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0)).clip(1e-12)


def exact_cell(exact, label, layer, n_rows):
    """The box's arrays at one cell, as [n_rows, width] over cell_data rows (NaN where the
    prompt was not run), and the mask of rows it covers."""
    from pipeline.s1_data_vectors.experiments.neuron_acts import POSITIONS
    j, k = list(exact["blocks"]).index(layer), POSITIONS.index(label)
    present = np.zeros(n_rows, bool)
    present[exact["rows"]] = True
    out = {}
    for name in ("block_in", "mlp_in", "acts"):
        full = np.full((n_rows, exact[name].shape[-1]), np.nan, np.float32)
        full[exact["rows"]] = exact[name][:, j, k]
        out[name] = full
    return out, present


def agreement(X, approx, cell, rows, exact_side, approx_side):
    """How stage A compares with the box at one cell and target."""
    cos_block_in = (X * cell["block_in"][rows]).sum(1) / (
        np.linalg.norm(X, axis=1) * np.linalg.norm(cell["block_in"][rows], axis=1))
    exact_top = {r["neuron"] for r in exact_side["top"]}
    return {"median cos(saved activation, box block input)": float(np.median(cos_block_in)),
            "median per-neuron r(approx, exact)": float(np.median(column_correlations(approx, cell["acts"][rows]))),
            "r(rho approx, rho exact) over neurons": float(np.corrcoef(approx_side["rho"], exact_side["rho"])[0, 1]),
            "top-20 overlap": len(exact_top & {r["neuron"] for r in approx_side["top"]})}


def run_all(root, exact_path=None):
    """Stage A (A1 read, A2 write) or, with the box's activations, A1 and B (exact read) on
    the prompts the box ran, with their agreement."""
    root = Path(root)
    run, lens_dir = root / RUN, root / "neuron_lens"
    z = np.load(run / "activations.npz")
    exact = None if exact_path is None else np.load(exact_path)
    rng = np.random.default_rng(0)
    results = {}
    for label, layer, natural_only in tqdm(CELLS, desc="cells"):
        read_weights = [load_bf16(lens_dir / f"L{layer}_{name}.npy")
                        for name in ("mlp_gate_proj", "mlp_up_proj", "post_attention_layernorm")]
        read_labels, read_label_rows = load_labels(lens_dir, layer)
        X_all, targets = cell_targets(z, run, label, layer, natural_only)
        if exact is not None:
            cell, present = exact_cell(exact, label, layer, len(X_all))
        else:
            W_down = load_bf16(lens_dir / f"L{layer - 1}_mlp_down_proj.npy")
            write_labels, write_label_rows = load_labels(lens_dir, layer - 1)
        for target, (rows, y, groups) in targets.items():
            key = f"{label} L{layer} | {target}"
            if exact is not None:
                rows = rows & present
            X, y, groups = X_all[rows], y[rows], None if groups is None else groups[rows]
            approx = neuron_activations(X, *read_weights)
            approx_side = read_side(approx, y, groups, read_labels, read_label_rows, layer, rng)
            if exact is None:
                results[key] = {"A1 read": approx_side,
                                "A2 write": write_side(X, y, groups, W_down, write_labels, write_label_rows,
                                                       layer, rng)}
            else:
                exact_side = read_side(cell["acts"][rows], y, groups, read_labels, read_label_rows, layer, rng)
                results[key] = {"A1 read": approx_side, "B read": exact_side,
                                "agreement": agreement(X, approx, cell, rows, exact_side, approx_side)}
    return results


def sides(result):
    return [side for side in result if side != "agreement"]


def plot(results, path):
    """Sorted |rho| (read sides) and |cos| (A2) per cell and target, with each permutation threshold."""
    side_names = sides(next(iter(results.values())))
    fig, axes = plt.subplots(len(side_names), len(results), figsize=(2.6 * len(results), 2.75 * len(side_names)),
                             sharey="row")
    for column, (key, result) in enumerate(results.items()):
        for row, side in enumerate(side_names):
            ax = axes[row, column]
            ax.plot(result[side]["abs_rho" if "read" in side else "abs_cos"][:200], lw=1.2)
            ax.axhline(result[side]["threshold"], color="0.4", ls="--", lw=0.8)
            if row == 0:
                ax.set_title(key.replace(" | ", "\n"), fontsize=7)
            ax.set_xlabel("rank", fontsize=7)
            ax.tick_params(labelsize=6)
    for row, side in enumerate(side_names):
        axes[row, 0].set_ylabel(f"{side} |rho|" if "read" in side else f"{side} |cos(write, direction)|", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def save(results, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot(results, out_dir / f"{name}.png")
    with open(out_dir / f"{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell", "side", "rank", "layer", "neuron", "pole", "score", "fires", "fires_conf",
                         "writes", "writes_conf", "url"])
        for key, result in results.items():
            for side in sides(result):
                for rank, r in enumerate(result[side]["top"], 1):
                    writer.writerow([key, side, rank, r["layer"], r["neuron"], r["pole"], f"{r['score']:.3f}",
                                     r["fires"], f"{r['fires_conf']:.2f}", r["writes"], f"{r['writes_conf']:.2f}",
                                     r["url"]])
    summary = {key: {side: (result[side] if side == "agreement" else
                            {k: v for k, v in result[side].items() if k not in ("abs_rho", "abs_cos", "rho")})
                     for side in result} for key, result in results.items()}
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=1))
    for key, result in summary.items():
        for side in sides(result):
            r = result[side]
            hits = ", ".join(f"{f} {e['top']}/{e['n_top']} (base {e['base_share']:.3f}, p {e['p']:.3g})"
                             for f, e in r["enrichment"].items())
            print(f"{key:42s} {side:8s} n={r['n']:3d} thr={r['threshold']:.3f} pass={r['n_pass']:5d} | {hits}")
        if "agreement" in result:
            print(f"{'':42s} agreement " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                                    for k, v in result["agreement"].items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="results root, e.g. /home/Jared/SPAR/results/horizon")
    parser.add_argument("--exact", help="the box's neuron_acts npz (stage B); adds the exact read side")
    args = parser.parse_args()
    save(run_all(args.root, args.exact), Path(args.root) / "analysis",
         "neuron_lens_compare" if args.exact is None else "neuron_lens_exact")
