"""Duration, unit word or written number: what the prompt-end geometry tracks, by layer (post hoc, 2026-11-03)

Saved E9 activations only (exp9_Qwen3-8B_units: 22 durations, each written in 2-4 units,
in the same 24 contexts; suffix newline and last suffix token, every layer). E9 judged
"duration, not unit" on a probe readout (unit share 0.02 at L16). A probe can pick the
duration-aligned part of a code that is mostly text. This asks the question of the
whole residual stream, with representational distances:

  1. Cell means. Each rendering ("60 minutes") is averaged over a half of the contexts;
     each dimension is z-scored over all E9 rows first, so Qwen's outlier dimensions do
     not set the distances.
  2. Cross-validated squared distance between two renderings: the dot product of their
     difference in one half of the contexts with their difference in the other half
     (unbiased: noise does not inflate it). Averaged over N_SPLITS random halvings,
     divided by the mean over pairs, so 1 = a typical pair.
  3. Primary, model-free contrasts (mean distance of a pair type, 1 = mean pair):
       integration   same duration, other unit ("60 minutes" / "1 hour") against same
                     unit, a small gap of 0.1-0.5 decades ("2 years" / "3 years"). An
                     integrated duration code puts the first well below the second; a
                     unit-word code puts it above. Unit-neutral version: against
                     other-unit pairs at the same 0.1-0.5 decade gap ("3 days" / "1 week").
       numeral       same numeral, other duration ("60 minutes" / "60 months") against
                     pairs with another numeral matched on the duration gap (±0.25
                     decades) and also in different units. Below 1 = a shared number code.
  4. Secondary: pairs regressed on duration-gap bins (a curved code makes distance
     nonlinear in the gap; a synthetic check showed a linear gap term biases the rest),
     same unit, same numeral and |Δ log10 numeral|.

CIs: 95%, resampling the 24 contexts (BOOT draws), each draw with fresh halvings.

    python analysis/unit_number_rsa.py /home/Jared/SPAR/results/horizon OUT_DIR
    -> OUT_DIR/unit_number_rsa.{json,txt,png}
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spar_horizon.horizons import UNIT_YEARS, canonical_unit  # noqa: E402

RUN = "exp9_Qwen3-8B_units"
POSITIONS = ("newline", "last")          # suffix newline after <|im_end|>, last suffix token
N_SPLITS = 8
BOOT = 50
GAP_EDGES = (0.05, 0.5, 1, 2, 4)        # decades; the first bin (< 0.05) is the same duration, the reference
TERMS = ("gap 0.05-0.5", "gap 0.5-1", "gap 1-2", "gap 2-4", "gap > 4", "same unit", "same numeral", "|Δ log numeral|")
SMALL_GAP = (0.1, 0.5)                   # decades: the same-unit reference for the integration contrast
MATCH = 0.25                             # decades: gap matching for the numeral contrast
LAYERS = range(1, 37, 2)


def renderings(root):
    """Per-row rendering id and context, and the rendering table (text, log duration, unit, numeral)."""
    prompts = json.loads((root / RUN / "prompts.json").read_text())
    rows = [i for i, p in enumerate(prompts) if not p["original"]]
    first = {}
    for i in rows:
        first.setdefault(prompts[i]["text"], prompts[i])
    table = []
    for text, p in sorted(first.items(), key=lambda item: (item[1]["years"], item[0])):
        numeral, unit = text.split()
        table.append({"text": text, "duration": p["duration"], "log_years": float(np.log10(p["years"])),
                      "unit": canonical_unit(unit), "numeral": numeral, "log_numeral": float(np.log10(float(numeral)))})
    index = {t["text"]: k for k, t in enumerate(table)}
    rendering = np.array([index[prompts[i]["text"]] for i in rows])
    context = np.array([prompts[i]["record"] for i in rows])
    return np.array(rows), rendering, context, table


def pair_design(table):
    """Pairs of renderings and per-pair facts: gap (decades), same duration / unit / numeral, |Δ log numeral|."""
    pairs = np.array(list(combinations(range(len(table)), 2)))
    a, b = [table[i] for i in pairs[:, 0]], [table[j] for j in pairs[:, 1]]
    facts = {"gap": np.array([abs(x["log_years"] - y["log_years"]) for x, y in zip(a, b)]),
             "same duration": np.array([x["duration"] == y["duration"] for x, y in zip(a, b)]),
             "same unit": np.array([x["unit"] == y["unit"] for x, y in zip(a, b)]),
             "same numeral": np.array([x["numeral"] == y["numeral"] for x, y in zip(a, b)]),
             "numeral gap": np.array([abs(x["log_numeral"] - y["log_numeral"]) for x, y in zip(a, b)])}
    bins = np.digitize(facts["gap"], GAP_EDGES)
    X = np.column_stack([bins == k for k in range(1, len(GAP_EDGES) + 1)]
                        + [facts["same unit"], facts["same numeral"], facts["numeral gap"]]).astype(float)
    return pairs, X, facts


def context_cell_means(Y, rendering, context, n_cells):
    """(contexts, cells, dims): every rendering appears once in every context."""
    contexts = np.unique(context)
    M = np.zeros((len(contexts), n_cells, Y.shape[1]))
    M[np.searchsorted(contexts, context), rendering] = Y
    if np.bincount(np.searchsorted(contexts, context) * n_cells + rendering).max() != 1 or len(Y) != M.shape[0] * n_cells:
        raise ValueError("expected each rendering exactly once per context")
    return M


def cv_distances(M, pairs, rng, resample=False):
    """Cross-validated squared distances over pairs, averaged over random halvings of the contexts.
    With `resample`, each half is a bootstrap draw from its own contexts, so the halves stay disjoint."""
    n_contexts = M.shape[0]
    total = np.zeros(len(pairs))
    for _ in range(N_SPLITS):
        order = rng.permutation(n_contexts)
        halves = []
        for half in (order[: n_contexts // 2], order[n_contexts // 2:]):
            if resample:
                half = rng.choice(half, len(half))
            halves.append(M[half].mean(axis=0))
        # (a_i - a_j)·(b_i - b_j) from the Gram matrix of the two halves' cell means
        G = halves[0] @ halves[1].T
        i, j = pairs[:, 0], pairs[:, 1]
        total += G[i, i] + G[j, j] - G[i, j] - G[j, i]
    distance = total / N_SPLITS
    return distance / distance.mean()


def regression(distance, X):
    design = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(design, distance, rcond=None)
    fitted = design @ coef
    r2 = 1 - ((distance - fitted) ** 2).sum() / ((distance - distance.mean()) ** 2).sum()
    return coef[1:], float(r2)


def contrasts(distance, facts):
    gap, same_dur, same_unit, same_num = facts["gap"], facts["same duration"], facts["same unit"], facts["same numeral"]
    in_band = (gap >= SMALL_GAP[0]) & (gap <= SMALL_GAP[1])
    small, small_other_unit = same_unit & in_band, ~same_unit & in_band
    numeral_pairs = np.flatnonzero(same_num & ~same_dur)
    reference = ~same_num & ~same_unit
    matched = [distance[reference & (np.abs(gap - gap[k]) <= MATCH)].mean() for k in numeral_pairs]
    return {"same duration, other unit": float(distance[same_dur].mean()),
            f"same unit, gap {SMALL_GAP[0]}-{SMALL_GAP[1]}": float(distance[small].mean()),
            "integration ratio": float(distance[same_dur].mean() / distance[small].mean()),
            # unit-neutral version: the reference pairs also differ in unit
            f"other unit, gap {SMALL_GAP[0]}-{SMALL_GAP[1]}": float(distance[small_other_unit].mean()),
            "integration ratio, other-unit reference": float(distance[same_dur].mean() / distance[small_other_unit].mean()),
            "same numeral, other duration": float(distance[numeral_pairs].mean()),
            "gap-matched other numeral": float(np.mean(matched)),
            "numeral ratio": float(distance[numeral_pairs].mean() / np.mean(matched))}


def analyse_cell(Y, rendering, context, pairs, X, facts, n_cells, rng):
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)
    M = context_cell_means(Y, rendering, context, n_cells)
    point = cv_distances(M, pairs, rng)
    coef, r2 = regression(point, X)
    named = contrasts(point, facts)
    boot_coef, boot_named = [], []
    for _ in range(BOOT):
        d = cv_distances(M, pairs, rng, resample=True)
        boot_coef.append(regression(d, X)[0])
        boot_named.append(list(contrasts(d, facts).values()))
    lo_c, hi_c = np.percentile(boot_coef, [2.5, 97.5], axis=0)
    lo_n, hi_n = np.percentile(boot_named, [2.5, 97.5], axis=0)
    return {"r2": r2,
            "coef": {t: [float(c), float(l), float(h)] for t, c, l, h in zip(TERMS, coef, lo_c, hi_c)},
            "contrasts": {k: [v, float(l), float(h)] for (k, v), l, h in zip(named.items(), lo_n, hi_n)}}


def fmt(ci):
    return f"{ci[0]:+.2f} [{ci[1]:+.2f}, {ci[2]:+.2f}]"


def summary_lines(results, table, facts):
    lines = ["E9 renderings: cross-validated distances (1 = mean pair); 95% CI over contexts",
             f"{len(table)} renderings, {len(facts['gap'])} pairs: same duration {int(facts['same duration'].sum())}, "
             f"same unit {int(facts['same unit'].sum())}, same numeral {int(facts['same numeral'].sum())} "
             f"(other duration {int((facts['same numeral'] & ~facts['same duration']).sum())})",
             "regression: reference = same duration (gap < 0.05), other unit, other numeral", ""]
    for position, by_layer in results.items():
        lines.append(f"position: suffix {position}")
        lines.append("  regression (secondary):")
        lines.append("  layer  R²    " + "  ".join(f"{t:>22s}" for t in TERMS))
        for layer, r in by_layer.items():
            lines.append(f"  L{layer:<4d} {r['r2']:.2f}  " + "  ".join(f"{fmt(r['coef'][t]):>22s}" for t in TERMS))
        lines.append("  contrasts (mean distance, 1 = mean pair; ratios: integration < 1 = duration code, numeral < 1 = shared number code):")
        names = list(next(iter(by_layer.values()))["contrasts"])
        lines.append("  layer  " + "  ".join(f"{n:>30s}" for n in names))
        for layer, r in by_layer.items():
            lines.append(f"  L{layer:<4d} " + "  ".join(f"{fmt(r['contrasts'][n]):>30s}" for n in names))
        lines.append("")
    return lines


def plot(results, path):
    """Top: the two ratios per position. Bottom: the four pair types behind them."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    styles = {"newline": "-", "last": "--"}
    for position, by_layer in results.items():
        layers = list(by_layer)
        for color, name in (("#2a78d6", "integration ratio"), ("#1baf7a", "integration ratio, other-unit reference"),
                            ("#eb6834", "numeral ratio")):
            values = np.array([by_layer[layer]["contrasts"][name] for layer in layers])
            axes[0].plot(layers, values[:, 0], styles[position], marker="o", ms=3, color=color, label=f"{name}, suffix {position}")
            axes[0].fill_between(layers, values[:, 1], values[:, 2], color=color, alpha=0.12)
        names = [n for n in by_layer[layers[0]]["contrasts"] if "ratio" not in n.split()]
        for color, name in zip(("#2a78d6", "#7b4fd6", "#1baf7a", "#eb6834", "#eda100"), names):
            values = np.array([by_layer[layer]["contrasts"][name] for layer in layers])
            axes[1].plot(layers, values[:, 0], styles[position], marker="o", ms=3, color=color, label=f"{name}, {position}")
    axes[0].axhline(1, color="k", lw=0.5)
    axes[0].set_ylabel("ratio of mean distances")
    axes[0].set_title("integration < 1: '60 minutes' closer to '1 hour' than '2 years' to '3 years'\n"
                      "numeral < 1: '60 minutes' closer to '60 months' than gap-matched pairs", fontsize=9)
    axes[1].axhline(1, color="k", lw=0.5)
    axes[1].set_ylabel("mean distance (1 = mean pair)")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(root, out_dir):
    root, out_dir = Path(root), Path(out_dir)
    rows, rendering, context, table = renderings(root)
    pairs, X, facts = pair_design(table)
    z = np.load(root / RUN / "activations.npz")
    rng = np.random.default_rng(0)
    results = {}
    for position in POSITIONS:
        activations = z[position]            # (prompts, layers, dims), ~0.9 GB: read once per position
        results[position] = {}
        # layer 0 is the token embedding, identical across prompts at these fixed tokens
        for layer in tqdm(LAYERS, desc=position):
            Y = activations[rows, layer].astype(np.float64)
            results[position][layer] = analyse_cell(Y, rendering, context, pairs, X, facts, len(table), rng)
    payload = {"renderings": table, "results": results}
    (out_dir / "unit_number_rsa.json").write_text(json.dumps(payload, indent=1))
    text = "\n".join(summary_lines(results, table, facts))
    (out_dir / "unit_number_rsa.txt").write_text(text)
    plot(results, out_dir / "unit_number_rsa.png")
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:3])
