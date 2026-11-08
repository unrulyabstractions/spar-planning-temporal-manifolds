"""Probes and geometry on the saved chunks of a choice-bank E0 run (pipeline/s1_data_vectors/experiments/exp0_choice.py,
26_spec_overnight_choice_scan.md §6). Takes whatever chunks exist and reports n; a stopped run is fine.

    python -m pipeline.s3_analysis.by_label.choice_structure results/horizon/exp0_Qwen3-8B_think-on_choice4
    -> <run dir>/choice_structure/{results.json, results.txt, probes_*.png}

Every probe is a cross-validated linear model on standardized activations reduced to at most
MAX_PCS whitened principal components, with the scaler and PCA fitted inside each training fold.
Folds are GroupKFold by `pair_id` (the base), so renderings of a base never straddle folds.
Classifiers score balanced accuracy (chance 1 / classes), regressions the out-of-fold R^2 (chance
about 0, negative when worse than the mean). Seeds are fixed and each cell has a time cap.

- Prompt-level targets, per layer at the suffix newline, horizon phrase (number mean, unit), think
  positions, `</think>` and pre-label: half chosen (near / far), the chosen label's position, the
  truth option's position, log10 horizon, regime.
- Option-level targets, at every option's label token and line-end `.` pooled over options
  (groups = the prompt's base): the option's return %, log10 delay, half, rank in half, whether it
  is the chosen option, and its line index (a control: position is trivially decodable).
- Horizon PCA at the suffix newline and phrase positions: |r| of PC1-3 with log10 horizon and the
  adjusted R^2 of the rendered unit and of the duration on those PCs (weighted by variance).
- The same key probes on forced-closed and naturally closed prompts separately, and the position
  bias of the chosen label.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

MAX_PCS = 64
FOLDS = 5
SEED = 0
CELL_SECONDS = 30
MIN_ROWS = 24          # fewer rows than this in a cell and its scores are nan
MIN_GROUPS = 6
MAIN_POSITIONS = ("suffix newline", "phrase number", "phrase unit", "think@0.25", "think@0.5", "think@1",
                  "think close", "pre-label")
SPLIT_POSITIONS = ("suffix newline", "think@1", "think close", "pre-label")
SPLIT_TARGETS = ("half_chosen", "chosen_pos", "regime")
HORIZON_POSITIONS = ("suffix newline", "phrase number", "phrase unit")
PROMPT_TARGETS = {"half_chosen": "binary", "chosen_pos": "multiclass", "truth_pos": "multiclass",
                  "log_horizon": "regression", "regime": "binary"}
OPTION_TARGETS = {"return_pct": "regression", "log_delay": "regression", "half": "binary",
                  "rank_in_half": "regression", "is_chosen": "binary", "line_index": "multiclass"}
OPTION_KINDS = ("label", "end")


# --- loading -------------------------------------------------------------------------------

def position_indices(regions, think_fractions, n_options):
    """Name -> index in the kept layout."""
    pos = {"suffix newline": regions["suffix"][0] + 1}
    for k, f in enumerate(think_fractions):
        pos[f"think@{f:g}"] = regions["think"][0] + k
    pos["think close"] = regions["pre"][0]
    pos["pre-label"] = regions["prelabel"][0]
    start = regions["prompt"][0]
    pos["phrase number"], pos["phrase unit"] = start, start + 1
    for j in range(n_options):
        pos[f"option {j + 1} label"], pos[f"option {j + 1} end"] = start + 2 + 2 * j, start + 3 + 2 * j
    return pos


def load_chunks(run_dir):
    """The kept prompts of every finished chunk: `rows` (answers.json rows), `acts` (layer ->
    [n, n_positions, d], only the positions in `index`), `forced`, `option_order`, `index`, `layers`."""
    chunk_dir = Path(run_dir) / "chunks"
    stems = sorted(p.stem for p in chunk_dir.glob("chunk_*.json"))
    if not stems:
        raise SystemExit(f"no finished chunks under {chunk_dir}")
    rows, forced, order, acts, setup = [], [], [], None, None
    for stem in stems:
        meta = json.loads((chunk_dir / f"{stem}.json").read_text())
        rows += [r for r, k in zip(meta["rows"], meta["kept"]) if k]
        with np.load(chunk_dir / f"{stem}.npz") as z:
            if setup is None:
                regions = {k[len("region_"):]: tuple(int(x) for x in z[k]) for k in z.files if k.startswith("region_")}
                layers = [int(l) for l in z["layers"]]
                index = position_indices(regions, z["think_fractions"], z["option_order"].shape[1])
                wanted = sorted(set(index.values()))
                setup = (regions, layers, index, wanted)
                acts = {l: [] for l in layers}
            regions, layers, index, wanted = setup
            for l in layers:
                acts[l].append(z[f"L{l}"][:, wanted].astype(np.float32))
            forced.append(z["forced_mask"])
            order.append(z["option_order"])
    column = {name: wanted.index(i) for name, i in index.items()}
    return {"rows": rows, "acts": {l: np.concatenate(a) for l, a in acts.items()}, "forced": np.concatenate(forced),
            "option_order": np.concatenate(order), "index": column, "layers": layers, "n_chunks": len(stems)}


# --- targets -------------------------------------------------------------------------------

def chosen_index(row):
    """Position of the chosen label in the row's labels, -1 if the answer did not parse."""
    label = row.get("label")
    return row["labels"].index(label) if label in row["labels"] else -1


def prompt_targets(rows):
    """name -> (y, kind); classes are ints with -1 for an unparsed answer, regressions float."""
    chosen = np.array([chosen_index(r) for r in rows])
    near_pick = np.array([r["labels"][c] in r["near_labels"] if c >= 0 else False for r, c in zip(rows, chosen)])
    return {"half_chosen": (np.where(chosen >= 0, near_pick, -1).astype(int), "binary"),
            "chosen_pos": (chosen, "multiclass"),
            "truth_pos": (np.array([r["labels"].index(r["truth_label"]) for r in rows]), "multiclass"),
            "log_horizon": (np.log10([r["horizon"] for r in rows]), "regression"),
            "regime": (np.array([r["regime"] == "near" for r in rows], int), "binary")}


def option_targets(rows, option_order):
    """Targets for the pooled option rows, ordered line-major: all prompts' line 1, then line 2, ..."""
    n, k = option_order.shape
    chosen = [chosen_index(r) for r in rows]
    cols = {name: [] for name in OPTION_TARGETS}
    for j in range(k):
        for r, order, c in zip(rows, option_order, chosen):
            o = r["options"][order[j]]
            cols["return_pct"].append(o["return_pct"])
            cols["log_delay"].append(np.log10(o["delay_years"]))
            cols["half"].append(int(o["half"] == "near"))
            cols["rank_in_half"].append(o["rank_in_half"])
            cols["is_chosen"].append(-1 if c < 0 else int(o["label"] == r["labels"][c]))
            cols["line_index"].append(j)
    return {name: (np.array(cols[name], float if kind == "regression" else int), kind)
            for name, kind in OPTION_TARGETS.items()}


# --- cross-validated probes ----------------------------------------------------------------

def _valid(y, kind):
    return ~np.isnan(y) if kind == "regression" else y >= 0


def _fit_predict(kind, Ztr, ytr, Zte):
    if kind == "regression":
        return RidgeCV(alphas=np.logspace(0, 4, 9)).fit(Ztr, ytr).predict(Zte)
    if len(np.unique(ytr)) < 2:
        return np.full(len(Zte), ytr[0])
    model = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
    return model.fit(Ztr, ytr).predict(Zte)


def _score(y, pred, kind):
    if kind == "regression":
        return float(r2_score(y, pred)), 0.0
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan"), float("nan")
    return float(balanced_accuracy_score(y, pred)), 1 / len(classes)


def cv_probes(X, groups, targets, seconds=CELL_SECONDS):
    """Grouped-CV scores of every target (name -> (y, kind)) on the rows of X. Returns
    name -> {score, chance, n}; score is nan (with `reason`) when the cell is too small or runs
    past `seconds`."""
    X, groups = np.asarray(X, np.float64), np.asarray(groups)
    n_groups = len(set(groups.tolist()))
    empty = {name: {"score": float("nan"), "chance": float("nan"), "n": int(_valid(y, kind).sum())}
             for name, (y, kind) in targets.items()}
    if len(X) < MIN_ROWS or n_groups < MIN_GROUPS:
        return {name: {**cell, "reason": "too few rows or bases"} for name, cell in empty.items()}
    deadline = time.monotonic() + seconds
    preds = {name: np.full(len(X), np.nan) for name in targets}
    for train, test in GroupKFold(min(FOLDS, n_groups)).split(X, groups=groups):
        scaler = StandardScaler().fit(X[train])
        pca = PCA(min(MAX_PCS, len(train) - 1, X.shape[1]), whiten=True, random_state=SEED).fit(scaler.transform(X[train]))
        Ztr, Zte = (pca.transform(scaler.transform(X[i])) for i in (train, test))
        for name, (y, kind) in targets.items():
            ok_tr, ok_te = _valid(y, kind)[train], _valid(y, kind)[test]
            if ok_tr.sum() >= 2 and ok_te.any():
                preds[name][test[ok_te]] = _fit_predict(kind, Ztr[ok_tr], y[train][ok_tr], Zte[ok_te])
        if time.monotonic() > deadline:
            return {name: {**cell, "reason": f"over {seconds} s"} for name, cell in empty.items()}
    out = {}
    for name, (y, kind) in targets.items():
        ok = _valid(y, kind) & ~np.isnan(preds[name])
        score, chance = _score(y[ok], preds[name][ok], kind) if ok.sum() >= 2 else (float("nan"), float("nan"))
        out[name] = {"score": score, "chance": chance, "n": int(ok.sum())}
    return out


def subset_targets(targets, mask):
    return {name: (y[mask], kind) for name, (y, kind) in targets.items()}


def prompt_level(data, targets, groups, seconds, rows_mask=None):
    """target -> position -> layer -> cell, for the main positions."""
    mask = np.ones(len(groups), bool) if rows_mask is None else rows_mask
    positions = [p for p in MAIN_POSITIONS if p in data["index"]]
    result = {t: {p: {} for p in positions} for t in targets}
    for layer in tqdm(data["layers"], desc="prompt probes", unit="layer", dynamic_ncols=True, leave=False):
        for position in positions:
            X = data["acts"][layer][mask, data["index"][position]]
            cells = cv_probes(X, groups[mask], subset_targets(targets, mask), seconds)
            for t, cell in cells.items():
                result[t][position][layer] = cell
    return result


def option_level(data, rows, groups, seconds):
    """target -> kind ('label' / 'end') -> layer -> cell, probes pooled over the option lines."""
    k = data["option_order"].shape[1]
    targets = option_targets(rows, data["option_order"])
    pooled_groups = np.tile(groups, k)
    result = {t: {kind: {} for kind in OPTION_KINDS} for t in targets}
    for layer in tqdm(data["layers"], desc="option probes", unit="layer", dynamic_ncols=True, leave=False):
        for kind in OPTION_KINDS:
            X = np.concatenate([data["acts"][layer][:, data["index"][f"option {j + 1} {kind}"]] for j in range(k)])
            for t, cell in cv_probes(X, pooled_groups, targets, seconds).items():
                result[t][kind][layer] = cell
    return result


# --- horizon PCA, position bias --------------------------------------------------------------

def adjusted_r2(y, labels):
    """Adjusted R^2 of y on the one-hot of `labels` (descriptive, not cross-validated)."""
    labels = np.asarray(labels)
    levels = np.unique(labels)
    sse = sum(((y[labels == v] - y[labels == v].mean()) ** 2).sum() for v in levels)
    total = ((y - y.mean()) ** 2).sum()
    if total == 0 or len(y) <= len(levels):
        return float("nan")
    return float(1 - (sse / (len(y) - len(levels))) / (total / (len(y) - 1)))


def horizon_pca(data, rows):
    """position -> layer -> |r| of PC1-3 with log10 horizon, adjusted R^2 of rendered unit and of
    duration on each PC, and their variance-weighted means over the three PCs."""
    log_h = np.log10([r["horizon"] for r in rows])
    units, durations = [r["rendered_unit"] for r in rows], [r["duration_id"] for r in rows]
    if len(rows) < MIN_ROWS:
        return {}
    result = {}
    for position in (p for p in HORIZON_POSITIONS if p in data["index"]):
        result[position] = {}
        for layer in data["layers"]:
            X = StandardScaler().fit_transform(data["acts"][layer][:, data["index"][position]].astype(np.float64))
            pca = PCA(3, random_state=SEED).fit(X)
            Z, evr = pca.transform(X), pca.explained_variance_ratio_
            unit_r2 = np.array([adjusted_r2(Z[:, i], units) for i in range(3)])
            dur_r2 = np.array([adjusted_r2(Z[:, i], durations) for i in range(3)])
            result[position][layer] = {
                "evr": evr.tolist(), "abs_r_log_horizon": [float(abs(np.corrcoef(Z[:, i], log_h)[0, 1])) for i in range(3)],
                "unit_adj_r2": unit_r2.tolist(), "duration_adj_r2": dur_r2.tolist(),
                "unit_share": float(np.nansum(evr * unit_r2) / evr.sum()),
                "duration_share": float(np.nansum(evr * dur_r2) / evr.sum())}
    return result


def position_bias(rows):
    """Share of parsed answers per displayed label position, and of truths there; accuracy by truth position."""
    k = len(rows[0]["labels"])
    chosen = np.array([chosen_index(r) for r in rows])
    truth = np.array([r["labels"].index(r["truth_label"]) for r in rows])
    parsed = chosen >= 0
    return {"n_parsed": int(parsed.sum()),
            "pick_rate": (np.bincount(chosen[parsed], minlength=k) / max(parsed.sum(), 1)).tolist(),
            "truth_rate": (np.bincount(truth, minlength=k) / len(truth)).tolist(),
            "exact_by_truth_pos": [float(np.mean(chosen[parsed & (truth == p)] == p)) if (parsed & (truth == p)).any()
                                   else float("nan") for p in range(k)]}


# --- report --------------------------------------------------------------------------------

def best(cells):
    """(layer, score) of the best cell of a layer -> cell dict, (None, nan) if all nan."""
    scored = [(l, c["score"]) for l, c in cells.items() if not np.isnan(c["score"])]
    return max(scored, key=lambda t: t[1]) if scored else (None, float("nan"))


def _row(label, cells, layers, width=22):
    return f"{label:<{width}}" + "".join(f"{cells[l]['score']:>7.2f}" if l in cells and not np.isnan(cells[l]["score"])
                                         else f"{'-':>7}" for l in layers)


def format_report(res):
    layers = res["layers"]
    head = f"{'':<22}" + "".join(f"{'L' + str(l):>7}" for l in layers)
    lines = [f"choice structure: {res['run_dir']}",
             f"n = {res['n']} kept prompts from {res['n_chunks']} chunk(s), {res['n_bases']} bases, k = {res['k']}, "
             f"{res['n_forced']} force-closed / {res['n'] - res['n_forced']} natural",
             "scores: balanced accuracy (classifiers; chance in each table title) / out-of-fold R^2 (regressions); "
             f"grouped CV by base, <= {MAX_PCS} PCs"]
    lines += [f"NOTE: {note}" for note in res["notes"]]
    if res["small_n"]:
        return "\n".join(lines) + "\n"
    for t, by_position in res["prompt_probes"].items():
        chance = next((c["chance"] for cells in by_position.values() for c in cells.values()
                       if not np.isnan(c["chance"])), float("nan"))
        lines += ["", f"[{t}]  chance {chance:.2f}" if PROMPT_TARGETS[t] != "regression" else f"[{t}]  R^2", head]
        lines += [_row(p, cells, layers) for p, cells in by_position.items()]
    for t, by_kind in res["option_probes"].items():
        lines += ["", f"[option: {t}]" + (f"  chance {next(iter(by_kind['label'].values()))['chance']:.2f}"
                                           if OPTION_TARGETS[t] != "regression" else "  R^2"), head]
        lines += [_row(f"option {kind}", cells, layers) for kind, cells in by_kind.items()]
    lines += ["", "[horizon PCA] |r(PC1, log h)| / unit share / duration share (adjusted R^2 on PC1-3, variance-weighted)", head]
    for p, by_layer in res["horizon_pca"].items():
        for name, key in (("|r| PC1", None), ("unit share", "unit_share"), ("duration share", "duration_share")):
            cells = {l: {"score": (c["abs_r_log_horizon"][0] if key is None else c[key])} for l, c in by_layer.items()}
            lines.append(_row(f"{p}: {name}", cells, layers, 30))
    lines += ["", "[forced vs natural close] best layer score (layer)"]
    for subset, by_target in res["split"].items():
        for t, by_position in by_target.items():
            for p, cells in by_position.items():
                layer, score = best(cells)
                lines.append(f"  {subset:<8} {t:<12} {p:<15} {score:6.2f}  (L{layer})  n = "
                             f"{next(iter(cells.values()))['n']}")
    bias = res["position_bias"]
    lines += ["", f"[position bias] n parsed {bias['n_parsed']}",
              "  pick rate  " + " ".join(f"{x:.2f}" for x in bias["pick_rate"]),
              "  truth rate " + " ".join(f"{x:.2f}" for x in bias["truth_rate"]),
              "  exact | truth position " + " ".join(f"{x:.2f}" for x in bias["exact_by_truth_pos"])]
    return "\n".join(lines) + "\n"


def curve_figure(path, panels, layers):
    """One row of small panels; each is (title, chance, {series: layer -> cell})."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.0), squeeze=False)
    for ax, (title, chance, series) in zip(axes[0], panels):
        for name, cells in series.items():
            ax.plot(layers, [cells[l]["score"] if l in cells else np.nan for l in layers], marker="o", ms=3, label=name)
        if chance is not None:
            ax.axhline(chance, color="gray", ls=":", lw=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("hidden_states layer", fontsize=8)
    axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_figures(res, out_dir):
    layers = res["layers"]

    def chance_of(by_position):
        return next((c["chance"] for cells in by_position.values() for c in cells.values() if not np.isnan(c["chance"])), None)

    pp = res["prompt_probes"]
    curve_figure(out_dir / "probes_prompt_main.png",
                 [(f"{t} ({'R^2' if PROMPT_TARGETS[t] == 'regression' else 'bal. acc.'})",
                   None if PROMPT_TARGETS[t] == "regression" else chance_of(pp[t]), pp[t])
                  for t in ("half_chosen", "regime", "log_horizon")], layers)
    curve_figure(out_dir / "probes_position.png",
                 [(f"{t} (bal. acc.)", chance_of(pp[t]), pp[t]) for t in ("chosen_pos", "truth_pos")], layers)
    op = res["option_probes"]
    curve_figure(out_dir / "probes_options.png",
                 [(f"option {t}", None if OPTION_TARGETS[t] == "regression" else chance_of(op[t]), op[t])
                  for t in ("return_pct", "log_delay", "half", "rank_in_half", "is_chosen")], layers)


def json_safe(o):
    """Nested dict/list with nan as None (strict json) and numpy scalars as python numbers."""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    return None if isinstance(o, float) and np.isnan(o) else o


def write_results(res, out_dir):
    (out_dir / "results.json").write_text(json.dumps(json_safe(res), indent=1))
    (out_dir / "results.txt").write_text(format_report(res))


def analyze(run_dir, out_dir=None, seconds=CELL_SECONDS):
    out_dir = Path(out_dir) if out_dir else Path(run_dir) / "choice_structure"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_chunks(run_dir)
    rows = data["rows"]
    groups = np.array([r["pair_id"] for r in rows])
    n_bases = len(set(groups.tolist()))
    notes = []
    res = {"run_dir": str(run_dir), "n": len(rows), "n_chunks": data["n_chunks"], "n_bases": n_bases,
           "k": len(rows[0]["labels"]), "layers": data["layers"], "n_forced": int(data["forced"].sum()),
           "small_n": len(rows) < MIN_ROWS or n_bases < MIN_GROUPS, "notes": notes}
    if res["small_n"]:
        notes.append(f"only {len(rows)} prompts / {n_bases} bases (need {MIN_ROWS} / {MIN_GROUPS}): no probes were run")
        write_results(res, out_dir)
        return res
    targets = prompt_targets(rows)
    res["prompt_probes"] = prompt_level(data, targets, groups, seconds)
    res["option_probes"] = option_level(data, rows, groups, seconds)
    res["horizon_pca"] = horizon_pca(data, rows)
    res["position_bias"] = position_bias(rows)
    res["split"] = {}
    for subset, mask in (("forced", data["forced"]), ("natural", ~data["forced"])):
        if mask.sum() < MIN_ROWS or len(set(groups[mask].tolist())) < MIN_GROUPS:
            notes.append(f"{subset} closes: {int(mask.sum())} prompts, split probes skipped")
            continue
        sub = {t: targets[t] for t in SPLIT_TARGETS}
        split = prompt_level({**data, "index": {p: data["index"][p] for p in SPLIT_POSITIONS if p in data["index"]}},
                             sub, groups, seconds, mask)
        res["split"][subset] = split
    write_results(res, out_dir)
    write_figures(res, out_dir)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--out", default=None, help="default: <run dir>/choice_structure")
    parser.add_argument("--cell-seconds", type=float, default=CELL_SECONDS, help="time cap per (layer, position) cell")
    args = parser.parse_args()
    analyze(args.run_dir, args.out, args.cell_seconds)
    print((Path(args.out) if args.out else Path(args.run_dir) / "choice_structure").joinpath("results.txt").read_text())
