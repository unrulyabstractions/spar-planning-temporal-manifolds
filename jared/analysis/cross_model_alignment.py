"""Is the 14B horizon representation the 8B one? (10_spec_scale_14b.md section 5, S2 and S3.)

Both models ran the same 500-prompt bank, so the 48 horizon averages (horizon x d, at the suffix
newline and at pre-label) line up row by row. Thinking on uses the prompts that closed naturally in
both models (one shared prompt set, so the two models' averages are over the same prompts).

- S2: linear CKA between every 8B layer and every 14B layer; the best-matching 14B layer per 8B layer
  must rise with depth, sit near the same relative depth, and (pre-label) the one matching 8B L24 must
  be within 2 layers of the 14B hand-off.
- S3: at matched cells (suffix newline at each model's decode_peak, pre-label at each model's hand-off):
  CKA against two 1,000-shuffle nulls, a leave-one-horizon-out cross-model readout (next to the
  leave-one-horizon-out unit lookup on the same averages), the same CKA with unit-word means removed,
  and the rank order along the first aligned coordinate. The nulls:
  - row permutation: permutes one model's horizon averages, so it only asks whether the two models
    pair the same horizons (also true of any shared prompt-level structure);
  - label shuffle (the verdict's): shuffles horizon labels over prompts, the same shuffle in both
    models (within unit word for the within-unit CKA), then recomputes the averages. Prompt-level
    structure the two models share without any horizon signal stays in this null.

decode_peak: the layer whose ridge on the prompt-level top-8 horizon-average PCs predicts log horizon
best under leave-one-horizon-out (the PCs are refitted on each fold's training horizons).

    python analysis/cross_model_alignment.py results/horizon
    -> <root>/analysis/cross_model_alignment.json and cka_heatmap_<position>_<mode>.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from horizon_shape import cell_data  # noqa: E402  (same folder)
from probe_dimensionality import units_of  # noqa: E402
from spar_horizon.probes import r2, within_unit_r2  # noqa: E402
from stage_markers import MODELS, exp0_run, handoff, n_layers  # noqa: E402

POSITIONS = ("suffix newline", "pre-label")
MODES = ("off", "on")
K = 8
SHUFFLES = 1000
ANCHOR_LAYER = 24   # 8B layer whose best 14B match is compared with the 14B hand-off (S2)
CKA_MIN, READOUT_MIN, RANK_MIN = 0.8, 0.8, 0.95
SPEARMAN_MIN, DEPTH_GAP_MAX, HANDOFF_GAP_MAX = 0.9, 0.08, 2


# --- statistics ------------------------------------------------------------------

def horizon_averages(X, log_h):
    """Sorted horizons (log10) and the average activation per horizon [h, d]."""
    groups = np.round(log_h, 9)
    horizons = np.unique(groups)
    return horizons, np.stack([X[groups == h].mean(0) for h in horizons])


def linear_cka(A, B):
    """Linear CKA of two [h, d] matrices with matching rows (NaN when either has no variance)."""
    return gram_cka(gram(A), gram(B))


def gram(A):
    A = A - A.mean(0)
    return A @ A.T


def gram_cka(K, L):
    denominator = np.sqrt((K * K).sum() * (L * L).sum())
    return float((K * L).sum() / denominator) if denominator > 1e-12 else float("nan")


def cka_null(A, B, shuffles=SHUFFLES, seed=0):
    """CKA of A with B after permuting B's rows, `shuffles` times."""
    K, L = gram(A), gram(B)
    rng = np.random.default_rng(seed)
    return np.array([gram_cka(K, L[np.ix_(p, p)]) for p in (rng.permutation(len(L)) for _ in range(shuffles))])


def label_shuffle_null(X8, X14, log_h, within_unit, shuffles=SHUFFLES, seed=0):
    """CKA of the two models' horizon averages after shuffling horizon labels over prompts (within each
    unit word when `within_unit`, which also removes unit means before the CKA), the same shuffle in
    both models. X8 [n, d8], X14 [n, d14] hold the same prompts in the same order."""
    horizons, group = np.unique(np.round(log_h, 9), return_inverse=True)
    prompt_units = units_of(log_h)
    counts = np.bincount(group, minlength=len(horizons))[:, None]
    rng = np.random.default_rng(seed)
    out = np.empty(shuffles)
    for s in range(shuffles):
        if within_unit:
            shuffled = group.copy()
            for unit in np.unique(prompt_units):
                rows = np.flatnonzero(prompt_units == unit)
                shuffled[rows] = group[rng.permutation(rows)]
        else:
            shuffled = group[rng.permutation(len(group))]
        member = np.eye(len(horizons))[shuffled].T / counts
        A8, A14 = member @ X8, member @ X14
        if within_unit:
            A8, A14 = remove_unit_means(A8, horizons), remove_unit_means(A14, horizons)
        out[s] = linear_cka(A8, A14)
    return out


def shuffle_p_value(observed, null):
    """One-sided p: share of the null at least as large as the observed value (counting the observation)."""
    return float((1 + (null >= observed).sum()) / (1 + len(null)))


def remove_unit_means(values, horizons):
    """Subtract the mean over horizons sharing a unit word (rows of `values`, 1-D or 2-D)."""
    units = units_of(horizons)
    out = np.array(values, float)
    for unit in np.unique(units):
        out[units == unit] -= out[units == unit].mean(0)
    return out


def unit_lookup_r2(horizons, within_unit=False):
    """Leave-one-horizon-out R² of predicting each horizon (log10) by the mean of the other horizons
    sharing its unit word: what the unit word alone gives on these averages. With `within_unit`, the
    unit means are removed from target and prediction first (`probes.horizon_within_unit_r2`'s score)."""
    units = units_of(horizons)
    pred = np.array([horizons[(units == u) & (horizons != h)].mean() for h, u in zip(horizons, units)])
    if within_unit:
        return within_unit_r2(horizons, pred, units)
    return r2(horizons, pred)


def scaled_procrustes(source, target):
    """Orthogonal R and scale s minimising |s source R - target|."""
    R, total = orthogonal_procrustes(source, target)
    return R, total / (source ** 2).sum()


def cross_model_readout(A14, A8, log_h):
    """Leave one horizon out: top-K PCs of each model's training averages, scaled Procrustes 14B -> 8B
    on the training horizons, linear probe on the 8B coordinates, applied to the held-out 14B
    horizon. Returns R² over all held-out predictions."""
    pred = np.empty(len(log_h))
    for i in range(len(log_h)):
        train = np.arange(len(log_h)) != i
        pca8, pca14 = PCA(K).fit(A8[train]), PCA(K).fit(A14[train])
        a, b = pca8.transform(A8[train]), pca14.transform(A14[train])
        R, scale = scaled_procrustes(b, a)
        probe = LinearRegression().fit(a, log_h[train])
        pred[i] = probe.predict(scale * pca14.transform(A14[i:i + 1]) @ R)[0]
    return float(1 - ((log_h - pred) ** 2).sum() / ((log_h - log_h.mean()) ** 2).sum())


def first_aligned_coordinate(A14, A8, log_h):
    """Spearman with log horizon of the first coordinate of the shared (8B PC) space, in each model."""
    pca8, pca14 = PCA(K).fit(A8), PCA(K).fit(A14)
    a, b = pca8.transform(A8), pca14.transform(A14)
    R, scale = scaled_procrustes(b, a)
    mapped = scale * b @ R
    return float(spearmanr(a[:, 0], log_h)[0]), float(spearmanr(mapped[:, 0], log_h)[0])


def decode_r2(X, log_h):
    """Leave-one-horizon-out R² of a ridge (penalty by internal CV) on standardized coordinates on the
    top-K PCs of the training horizons' averages (refitted per fold, so the held-out horizon takes no
    part in choosing the coordinates)."""
    pred = np.empty(len(log_h))
    groups = np.round(log_h, 9)
    for train, test in LeaveOneGroupOut().split(X, groups=groups):
        mean = X[train].mean(0)
        averages = np.stack([X[train][groups[train] == h].mean(0) for h in np.unique(groups[train])]) - mean
        basis = PCA(K).fit(averages).components_
        Z_train, Z_test = (X[train] - mean) @ basis.T, (X[test] - mean) @ basis.T
        scaler = StandardScaler().fit(Z_train)
        model = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(scaler.transform(Z_train), log_h[train])
        pred[test] = model.predict(scaler.transform(Z_test))
    return float(1 - ((log_h - pred) ** 2).sum() / ((log_h - log_h.mean()) ** 2).sum())


# --- per-run collection -----------------------------------------------------------

def shared_drop(root, mode):
    """Mask over the kept prompts (those with a horizon) of the prompts force-closed in either model,
    so that both models are averaged over the same natural closes."""
    drop = None
    for model in MODELS:
        _, _, forced = cell_data(np.load(exp0_run(root, model, mode) / "activations.npz"), "suffix newline", 1)
        drop = forced if drop is None else drop | forced
    return drop


def collect(root, model, mode, drop):
    """{position: {"averages": {layer: [h, d]}, "decode_r2": {layer: R²}}} and the horizons (log10)
    for layers 1..n (layer 0 is the embedding: identical across horizons at these tokens). `drop`
    marks the prompts left out."""
    z = np.load(exp0_run(root, model, mode) / "activations.npz")
    out = {p: {"averages": {}, "decode_r2": {}} for p in POSITIONS}
    horizons = None
    for layer in tqdm(range(1, n_layers(z) + 1), desc=f"{model} {mode}"):
        for position in POSITIONS:
            X, log_h, _ = cell_data(z, position, layer)
            X, log_h = X[~drop], log_h[~drop]
            horizons, out[position]["averages"][layer] = horizon_averages(X, log_h)
            if position == "suffix newline":
                out[position]["decode_r2"][layer] = decode_r2(X, log_h)
    return out, horizons


# --- S2 / S3 ----------------------------------------------------------------------

def cka_matrix(averages8, averages14):
    """CKA [8B layer - 1, 14B layer - 1]."""
    grams14 = {l: gram(A) for l, A in averages14.items()}
    return np.array([[gram_cka(gram(A8), G) for G in grams14.values()] for A8 in averages8.values()])


def s2_summary(C, n8, n14, anchor_handoff=None):
    """Best-matching 14B layer per 8B layer (NaN rows skipped) and the S2 checks."""
    rows = [i for i in range(len(C)) if not np.isnan(C[i]).all()]
    layers8 = np.array(rows) + 1
    best = np.array([np.nanargmax(C[i]) for i in rows]) + 1
    rho = float(spearmanr(layers8, best)[0])
    gap = float(np.abs(layers8 / n8 - best / n14).mean())
    out = {"best_14b_layer": dict(zip(map(int, layers8), map(int, best))), "spearman": rho, "mean_depth_gap": gap,
           "monotone_ok": rho >= SPEARMAN_MIN, "depth_ok": gap <= DEPTH_GAP_MAX}
    if anchor_handoff is not None:
        match = int(best[list(layers8).index(ANCHOR_LAYER)])
        out.update(match_of_8b_L24=match, handoff_14b=anchor_handoff, handoff_ok=abs(match - anchor_handoff) <= HANDOFF_GAP_MAX)
    out["pass"] = out["monotone_ok"] and out["depth_ok"] and out.get("handoff_ok", True)
    return out


def matched_cell(A8, A14, horizons, X8, X14, log_h):
    """All S3 numbers for one matched cell, plus the within-unit CKA. A8, A14: horizon averages;
    X8, X14, log_h: the prompts they average, in the same order in both models."""
    cka, null = linear_cka(A8, A14), cka_null(A8, A14)
    unit8, unit14 = remove_unit_means(A8, horizons), remove_unit_means(A14, horizons)
    within, within_null = linear_cka(unit8, unit14), cka_null(unit8, unit14)
    shuffle = label_shuffle_null(X8, X14, log_h, within_unit=False)
    within_shuffle = label_shuffle_null(X8, X14, log_h, within_unit=True)
    rho8, rho14 = first_aligned_coordinate(A14, A8, horizons)
    out = {"cka": cka, "null_p95": float(np.percentile(null, 95)),
           "shuffle_mean": float(shuffle.mean()), "shuffle_p95": float(np.percentile(shuffle, 95)),
           "shuffle_p": shuffle_p_value(cka, shuffle),
           "readout_r2": cross_model_readout(A14, A8, horizons), "unit_lookup_r2": unit_lookup_r2(horizons),
           "within_unit_cka": within, "within_unit_null_p95": float(np.percentile(within_null, 95)),
           "within_unit_shuffle_mean": float(within_shuffle.mean()),
           "within_unit_shuffle_p95": float(np.percentile(within_shuffle, 95)),
           "within_unit_shuffle_p": shuffle_p_value(within, within_shuffle),
           "rank_spearman_8b": rho8, "rank_spearman_14b": rho14}
    out["across_unit_ok"] = cka >= CKA_MIN and cka > out["shuffle_p95"] and out["readout_r2"] >= READOUT_MIN
    out["within_unit_ok"] = within > out["within_unit_shuffle_p95"]
    out["rank_ok"] = min(abs(rho8), abs(rho14)) >= RANK_MIN
    out["pass"] = out["across_unit_ok"] and out["within_unit_ok"] and out["rank_ok"]
    return out


def matched_prompts(root, mode, position, layers, drop):
    """Both models' prompt activations [n, d] at the matched cell and the shared log horizons."""
    cells = []
    for model, layer in zip(MODELS, layers):
        X, log_h, _ = cell_data(np.load(exp0_run(root, model, mode) / "activations.npz"), position, layer)
        cells.append((X[~drop], log_h[~drop]))
    assert np.allclose(cells[0][1], cells[1][1]), "the two banks do not share the same prompts"
    return cells[0][0], cells[1][0], cells[0][1]


def plot_heatmap(C, n8, n14, title, path):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(C, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto",
                   extent=(0.5, C.shape[1] + 0.5, 0.5, C.shape[0] + 0.5))
    ax.plot([0.5, n14 + 0.5], [0.5, n8 + 0.5], "w--", lw=1, label="equal relative depth")
    ax.set_xlabel("Qwen3-14B layer")
    ax.set_ylabel("Qwen3-8B layer")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left", labelcolor="w")
    fig.colorbar(im, ax=ax, label="linear CKA")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(root):
    root = Path(root)
    out_dir = root / "analysis"
    out_dir.mkdir(exist_ok=True)
    n = {m: n_layers(np.load(exp0_run(root, m, "off") / "activations.npz")) for m in MODELS}
    results = {"S2": {}, "S3": {}}
    for mode in MODES:
        drop = shared_drop(root, mode)
        data8, h8 = collect(root, "Qwen3-8B", mode, drop)
        data14, h14 = collect(root, "Qwen3-14B", mode, drop)
        assert np.allclose(h8, h14), "the two banks do not share the same horizons"
        for position in POSITIONS:
            C = cka_matrix(data8[position]["averages"], data14[position]["averages"])
            plot_heatmap(C, n["Qwen3-8B"], n["Qwen3-14B"], f"CKA, {position}, thinking {mode}",
                         out_dir / f"cka_heatmap_{position.replace(' ', '_')}_{mode}.png")
            anchor = handoff(root, "Qwen3-14B", mode)[0] if position == "pre-label" else None
            results["S2"][f"{position}|{mode}"] = {**s2_summary(C, n["Qwen3-8B"], n["Qwen3-14B"], anchor), "cka": C.tolist()}
        cells = {"suffix newline": [max(d["suffix newline"]["decode_r2"], key=d["suffix newline"]["decode_r2"].get)
                                    for d in (data8, data14)],
                 "pre-label": [handoff(root, m, mode)[0] for m in MODELS]}
        for position, (layer8, layer14) in cells.items():
            cell = matched_cell(data8[position]["averages"][layer8], data14[position]["averages"][layer14], h8,
                                *matched_prompts(root, mode, position, (layer8, layer14), drop))
            results["S3"][f"{position}|{mode}"] = {"layer_8b": layer8, "layer_14b": layer14, **cell}
    results["S2_pass"] = all(v["pass"] for v in results["S2"].values())
    results["S3_pass"] = all(v["pass"] for v in results["S3"].values())
    (out_dir / "cross_model_alignment.json").write_text(json.dumps(results, indent=1))
    print_report(results)
    print(f"\nwrote {out_dir}/cross_model_alignment.json and cka_heatmap_*.png")


def print_report(results):
    print("S2 depth correspondence (best-matching 14B layer per 8B layer)")
    for key, s in results["S2"].items():
        extra = (f", 8B L{ANCHOR_LAYER} -> 14B L{s['match_of_8b_L24']} vs hand-off L{s['handoff_14b']}"
                 f" ({'ok' if s['handoff_ok'] else 'FAIL'})" if "handoff_14b" in s else "")
        print(f"  {key:22} Spearman {s['spearman']:.3f} ({'ok' if s['monotone_ok'] else 'FAIL'}), "
              f"mean |depth gap| {s['mean_depth_gap']:.3f} ({'ok' if s['depth_ok'] else 'FAIL'}){extra}"
              f" -> {'PASS' if s['pass'] else 'FAIL'}")
    print(f"S2: {'PASS' if results['S2_pass'] else 'FAIL'}\n")
    print("S3 matched cells (CKA nulls: row = permuted horizon averages; shuffle = horizon labels over prompts,")
    print("same shuffle in both models, the verdict's; p = share of the shuffle null >= observed)")
    print(f"  {'cell':22} {'8B/14B':>7} {'CKA':>6} {'row p95':>7} {'shuf mean':>9} {'shuf p95':>8} {'p':>6} "
          f"{'readout':>8} {'lookup':>7} {'unit CKA':>9} {'row p95':>7} {'shuf mean':>9} {'shuf p95':>8} {'p':>6} "
          f"{'rho 8B':>7} {'rho 14B':>8}")
    for key, s in results["S3"].items():
        print(f"  {key:22} L{s['layer_8b']}/L{s['layer_14b']:<3d} {s['cka']:6.3f} {s['null_p95']:7.3f} "
              f"{s['shuffle_mean']:9.3f} {s['shuffle_p95']:8.3f} {s['shuffle_p']:6.3f} "
              f"{s['readout_r2']:8.3f} {s['unit_lookup_r2']:7.3f} {s['within_unit_cka']:9.3f} {s['within_unit_null_p95']:7.3f} "
              f"{s['within_unit_shuffle_mean']:9.3f} {s['within_unit_shuffle_p95']:8.3f} {s['within_unit_shuffle_p']:6.3f} "
              f"{s['rank_spearman_8b']:7.3f} {s['rank_spearman_14b']:8.3f}  "
              f"across {'ok' if s['across_unit_ok'] else 'FAIL'}, within-unit {'ok' if s['within_unit_ok'] else 'FAIL'}, "
              f"rank {'ok' if s['rank_ok'] else 'FAIL'}")
    print(f"S3: {'PASS' if results['S3_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1])
