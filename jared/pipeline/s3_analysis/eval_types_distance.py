"""How far is each eval type from Alan's prompt, inside Qwen3-8B? Ranks the types and labels every
row, so later steering runs (E8 answer slot, E13 suffix, E14 horizon phrase) can be read against
distance.

    python -m pipeline.s3_analysis.eval_types_distance $RESULTS/evaltypes_Qwen3-8B_think-on \
        $RESULTS/exp0_Qwen3-8B_think-on_full

Matched original. Each row is compared with the same decision written in Alan's template:
- types with twins: a `base` row against its item's `twin` row (same options, same order, horizon
  stated in Alan's CONSTRAINT line);
- the wording ladder: `t0_w1_cash` / `t0_w3_clear` rows against the `t0_w0` row with the same item
  and variant (`t0_w0` is Alan's wording, so it is the reference at distance 0);
- `free_form` / `free_form_anchor` have no twin; they get only the unpaired distance below.

Positions: suffix_end = last prompt token (where E13 steers; the stand-in for E14's horizon phrase,
which is not saved); prelabel = the token before the answer label (where E8 steers).

Distance, per layer and position, in three spaces:
- full: the whole 4096-d hidden state;
- horizon: its coordinates in the steering subspace, the top PCs of the full bank's per-horizon
  averages (`steering.fit_cell`, the basis E8/E13 steer in; 8 dimensions);
- bank_pca: its coordinates in the full bank's top principal components covering 90% of the bank's
  variance (the space Alan's prompts span).
In each: cosine distance after subtracting the mean over all eval rows, divided by the median
distance between two *different* original prompts of the same regime. So 0 is "same state as Alan's
prompt" and 1 is "as far as another of Alan's prompts".

Two more per row: probe_shift = |readout of the row - readout of its original| in decades, from the
full bank's ridge probe for log horizon (`fit_cell`'s w); outside_bank_pca = the share of the row's
variance about the bank mean that lies outside the bank's top PCs.

Unpaired distance (full space), for every row: cosine distance to the mean of the original prompts,
divided by the median of the same for the originals themselves (originals sit near 1).

A type's score is the median over its rows, with a bootstrap 90% interval over rows. The ranking
uses the full space at L18 suffix_end. Writes `eval_distance.json`, `eval_distance.md` and
`eval_distance_rows.json`.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from spar_horizon.eval_types import TYPES
from spar_horizon.steering import fit_cell

POSITIONS = {"suffix_end": 4, "prelabel": 10}   # indices into the 17 saved tokens (same in the bank)
SPACES = ("full", "horizon", "bank_pca")
BANK_VARIANCE = 0.90
RANK_CELL = "L18_suffix_end"
WORDING_REFERENCE = {"t0_w1_cash": "t0_w0", "t0_w3_clear": "t0_w0"}
REFERENCE_TYPE = "t0_w0"
BOOTSTRAP = 2000

# What the prompt changes relative to the twin (Alan's template, same options and horizon).
CHANGES = {
    "t0_w0": "none: Alan's template (reference)",
    "t0_w1_cash": "adds 'money is kept as cash and earns no interest'",
    "t0_w3_clear": "cash sentence + receive-payment TASK + end-of-horizon CONSTRAINT",
    "roi_equal_10": "4 options of equal value at 10%/yr; rate and 10,000 stake in TASK",
    "roi_unequal": "none in the text: 8 options already in Alan's template",
    "roi_unequal_clear": "8 options + clear wording",
    "free_form": "names a deposit term instead of choosing a label",
    "free_form_anchor": "free_form with the horizon as an implicit life event",
    "unit_mismatch": "horizon written in a different unit (e.g. 24 months)",
    "horizon_revised": "stale horizon in CONSTRAINT, true one in an UPDATE line",
    "horizon_revised_clear": "horizon_revised + clear wording",
    "loss_framing": "costs instead of payouts; budget constraint",
    "json_io": "whole prompt and answer as JSON",
    "system_prompt": "constraint moved into the system message",
    "system_prompt_clear": "system_prompt + clear wording",
    "role_fund_manager": "fund-manager role",
    "role_assistant": "'you are my assistant' advisory role",
    "sycophancy_lean": "assistant role + user leans towards one option",
    "sycophancy_lean_clear": "sycophancy_lean + cash sentence and end-of-horizon CONSTRAINT",
    "prose_natural": "plain prose request, no SITUATION/TASK headers",
    "multiturn_analyst": "decision after earlier COST/BENEFIT/TIMING turns",
    "implicit_concrete": "horizon implied by a life event (exact)",
    "implicit_vague": "horizon implied by a life event (a range)",
    "city_no_values": "city projects, delivery dates only, no values",
    "city_values": "city projects with cost and benefit",
    "public_health": "regional health programs",
    "coding_agent": "coding agent choosing engineering projects",
    "distractor_temporal": "irrelevant duration added to SITUATION",
}

# Where each steering modality can act. E14 steers the horizon phrase.
E14_SITE = {
    "implicit_concrete": "life-event phrase", "implicit_vague": "life-event phrase",
    "free_form_anchor": "life-event phrase",
    "horizon_revised": "two phrases (stale, updated)", "horizon_revised_clear": "two phrases (stale, updated)",
}


def family(type_id):
    return TYPES[type_id].render.__module__.rsplit(".", 1)[-1]


def steering_sites(type_id):
    label = TYPES[type_id].answer_kind == "label"
    return {"E8": "yes" if label else "no (duration answer)", "E13": "yes",
            "E14": E14_SITE.get(type_id, "stated horizon")}


# ---- loading ----

def load_rows(run_dir):
    """{type_id: rows} for every type directory with rows and activations."""
    out = {}
    for path in sorted(Path(run_dir).glob("*/rows.json")):
        rows = [r for r in json.loads(path.read_text()) if r.get("act_file")]
        if rows and path.parent.name in TYPES:
            out[path.parent.name] = rows
    return out


def load_states(run_dir, rows_by_type):
    """(layers, {sample_uid: array (n_layers, n_positions, d)}) at the POSITIONS."""
    states, layers = {}, None
    pos = list(POSITIONS.values())
    for type_id, rows in rows_by_type.items():
        files = {}
        for r in rows:
            if r["act_file"] not in files:
                z = np.load(Path(run_dir) / type_id / r["act_file"])
                file_layers = tuple(int(x) for x in z["layers"])
                assert layers in (None, file_layers), f"{type_id}: layers {file_layers} != {layers}"
                layers = file_layers
                files[r["act_file"]] = np.stack([z[f"L{n}"][:, pos] for n in layers], axis=1)
            states[r["sample_uid"]] = files[r["act_file"]][r["act_row"]].astype(np.float32)
    return layers, states


# ---- pairing ----

def item_key(row):
    return row["twin_id"].rsplit("_", 1)[-1], row["variant"]


def match_originals(rows_by_type):
    """{sample_uid: (original sample_uid or None, pairing)} for every row."""
    out = {}
    for type_id, rows in rows_by_type.items():
        twins = {r["twin_id"]: r["sample_uid"] for r in rows if r["variant"] == "twin"}
        reference = rows_by_type.get(WORDING_REFERENCE.get(type_id))
        reference = {item_key(r): r["sample_uid"] for r in reference} if reference else {}
        for r in rows:
            uid = r["sample_uid"]
            if r["variant"] == "twin":
                out[uid] = (None, "is original (twin)")
            elif type_id == REFERENCE_TYPE:
                out[uid] = (uid, "is original (t0_w0)")
            elif type_id in WORDING_REFERENCE and item_key(r) in reference:
                out[uid] = (reference[item_key(r)], f"same item in {WORDING_REFERENCE[type_id]}")
            elif r["variant"] == "base" and r["twin_id"] in twins:
                out[uid] = (twins[r["twin_id"]], "twin")
            else:
                out[uid] = (None, "unpaired")
    return out


def original_prompts(rows_by_type):
    """Distinct original prompts: twin rows and t0_w0 rows, one row per prompt text."""
    seen = {}
    for type_id, rows in rows_by_type.items():
        for r in rows:
            if r["variant"] == "twin" or type_id == REFERENCE_TYPE:
                seen.setdefault(r["prompt"], r)
    return list(seen.values())


# ---- distances ----

def cosine_distance(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return 1.0 - np.sum(a * b, axis=-1)


def originals_scale(X, regimes):
    """Median cosine distance between two different originals of the same regime."""
    d = []
    for regime in set(regimes):
        idx = [i for i, g in enumerate(regimes) if g == regime]
        for j, i in enumerate(idx):
            d.extend(cosine_distance(X[idx[j + 1:]], X[i]))
    return float(np.median(d))


def cell_distances(X, uids, originals, regimes_of_originals, matched):
    """Paired and unpaired distances of every row for one (layer, position).

    X: (n_rows, d) states in `uids` order; `originals` lists the uids of the distinct originals;
    `matched` maps a uid to its original's uid (or None)."""
    Xc = X - X.mean(axis=0)
    index = {u: i for i, u in enumerate(uids)}
    O = Xc[[index[u] for u in originals]]
    scale = originals_scale(O, regimes_of_originals)
    centroid = O.mean(axis=0)
    centroid_scale = float(np.median(cosine_distance(O, centroid)))
    paired = {}
    for u, orig in matched.items():
        if orig is not None:
            paired[u] = float(cosine_distance(Xc[index[u]], Xc[index[orig]])) / scale
    unpaired = cosine_distance(Xc, centroid) / centroid_scale
    return paired, {u: float(unpaired[i]) for i, u in enumerate(uids)}, scale


def median_ci(values, rng):
    values = np.asarray(values)
    if len(values) == 0:
        return None
    boot = np.median(rng.choice(values, size=(BOOTSTRAP, len(values))), axis=1)
    return {"n": len(values), "median": round(float(np.median(values)), 3),
            "ci90": [round(float(q), 3) for q in np.percentile(boot, [5, 95])]}


# ---- the bank's spaces ----

def load_bank_layer(bank_dir, layer):
    """Full-bank states at the POSITIONS [n, positions, d] and log10 horizons, prompts with a horizon."""
    z = np.load(Path(bank_dir) / "activations.npz")
    keep = z["kept"] & np.isfinite(z["horizons"])
    X = np.asarray(z[f"L{layer}"][:, list(POSITIONS.values())], dtype=np.float32)[keep]
    return X, np.log10(z["horizons"][keep])


def fit_bank_spaces(Xb, log_h):
    """The steering cell (mean, probe w and b, horizon basis) and the bank's top PCs."""
    cell = fit_cell(Xb, log_h)
    pca = PCA(n_components=BANK_VARIANCE, svd_solver="full").fit(Xb)
    return {"cell": cell, "pca_mean": pca.mean_, "pca_basis": pca.components_}


def project(bank, X):
    """The rows of X [n, d] in each space, the probe readout and the share outside the bank PCs."""
    cell = bank["cell"]
    about_bank = X - bank["pca_mean"]
    coords = about_bank @ bank["pca_basis"].T
    outside = 1.0 - np.sum(coords ** 2, axis=1) / np.sum(about_bank ** 2, axis=1)
    return {"full": X, "horizon": (X - cell["mean"]) @ cell["basis"].T, "bank_pca": coords,
            "readout": (X - cell["mean"]) @ cell["w"] + cell["b"], "outside": outside}


# ---- the run ----

def compute(run_dir, bank_dir):
    rows_by_type = load_rows(run_dir)
    layers, states = load_states(run_dir, rows_by_type)
    match = match_originals(rows_by_type)
    matched = {u: orig for u, (orig, _) in match.items()}
    prompt_of = {r["sample_uid"]: r["prompt"] for rows in rows_by_type.values() for r in rows}
    originals = original_prompts(rows_by_type)
    original_uids = [r["sample_uid"] for r in originals]
    original_regimes = [r["regime"] for r in originals]
    uids = list(states)
    index = {u: i for i, u in enumerate(uids)}
    stacked = np.stack([states[u] for u in uids])            # (n, layers, positions, d)
    repeats = repeated_prompts(uids, prompt_of)

    distance = {u: {} for u in uids}
    cells, repeat = {}, {}
    for li, layer in enumerate(layers):
        Xb_layer, log_h = load_bank_layer(bank_dir, layer)
        for pi, pos in enumerate(POSITIONS):
            name = f"L{layer}_{pos}"
            bank = fit_bank_spaces(Xb_layer[:, pi], log_h)
            proj = project(bank, stacked[:, li, pi])
            cells[name] = {"bank_pca_k": int(bank["pca_basis"].shape[0]),
                           "horizon_k": int(bank["cell"]["basis"].shape[0]),
                           "originals_outside_bank_pca": round(float(np.median(
                               proj["outside"][[index[u] for u in original_uids]])), 4)}
            for u in uids:
                orig = matched[u]
                distance[u][name] = {"probe_shift": None if orig is None else
                                     round(float(abs(proj["readout"][index[u]] - proj["readout"][index[orig]])), 4),
                                     "outside_bank_pca": round(float(proj["outside"][index[u]]), 4)}
            for space in SPACES:
                paired, unpaired, scale = cell_distances(proj[space], uids, original_uids, original_regimes, matched)
                cells[name][f"scale_{space}"] = round(scale, 4)
                for u in uids:
                    distance[u][name][space] = {"paired": round(paired[u], 4) if u in paired else None}
                    if space == "full":
                        distance[u][name][space]["unpaired"] = round(unpaired[u], 4)
                if layer == 18:
                    repeat[f"{name} {space}"] = repeat_gaps(proj[space], repeats, scale)
            if layer == 18:
                gaps = [abs(proj["readout"][i] - proj["readout"][j]) for i, j in repeats]
                repeat[f"{name} probe_shift"] = {"n": len(gaps), "median": round(float(np.median(gaps)), 4),
                                                 "max": round(float(max(gaps)), 4)}
        print(f"L{layer} done", flush=True)

    row_labels = []
    for type_id, rows in rows_by_type.items():
        for r in rows:
            u = r["sample_uid"]
            orig = matched[u]
            row_labels.append({
                "sample_uid": u, "type_id": type_id, "family": family(type_id), "variant": r["variant"],
                "twin_id": r["twin_id"], "regime": r["regime"], "horizon": r["horizon"],
                "horizon_text": r["horizon_text"], "original_uid": orig, "pairing": match[u][1],
                "prompt_identical_to_original": None if orig is None else r["prompt"] == prompt_of[orig],
                "distance": distance[u]})
    return layers, cells, row_labels, repeat


def repeated_prompts(uids, prompt_of):
    """Index pairs of twin rows with the same prompt: a v2 type's twin is its original type's twin,
    rerun a day later on another box."""
    first, pairs = {}, []
    for i, u in enumerate(uids):
        if u.endswith("_twin"):
            j = first.setdefault(prompt_of[u], i)
            if j != i:
                pairs.append((i, j))
    return pairs


def repeat_gaps(Z, pairs, scale):
    """Scaled distance between the two runs of each repeated prompt: the noise floor. Centred like
    every other distance."""
    Zc = Z - Z.mean(axis=0)
    gaps = [float(cosine_distance(Zc[i], Zc[j])) / scale for i, j in pairs]
    return {"n": len(gaps), "median": round(float(np.median(gaps)), 4), "max": round(max(gaps), 4)}


def type_table(row_labels, run_dir):
    rng = np.random.default_rng(0)
    summary = _json(Path(run_dir) / "summary.json")
    probe = _json(Path(run_dir) / "probe_transfer.json").get("types", {})
    out = {}
    for type_id in dict.fromkeys(r["type_id"] for r in row_labels):
        rows = [r for r in row_labels if r["type_id"] == type_id and r["variant"] != "twin"]
        by_cell = {}
        for cell in rows[0]["distance"]:
            d = [r["distance"][cell] for r in rows]
            by_cell[cell] = {space: {"paired": median_ci([x[space]["paired"] for x in d
                                                          if x[space]["paired"] is not None], rng)}
                             for space in SPACES}
            by_cell[cell]["full"]["unpaired"] = median_ci([x["full"]["unpaired"] for x in d], rng)
            by_cell[cell]["probe_shift"] = median_ci([x["probe_shift"] for x in d if x["probe_shift"] is not None], rng)
            by_cell[cell]["outside_bank_pca"] = median_ci([x["outside_bank_pca"] for x in d], rng)
        s = summary.get(type_id, {})
        out[type_id] = {
            "family": family(type_id), "changes": CHANGES.get(type_id, ""), "steering": steering_sites(type_id),
            "pairing": sorted({r["pairing"] for r in rows}),
            "accuracy": (s.get("accuracy") or {}).get("p"),
            "probe_spearman": {c: (probe.get(type_id, {}).get(c) or {}).get("spearman")
                               for c in ("L18_suffix_end", "L18_prelabel")},
            "distance": by_cell}
    for space in SPACES:
        key = "rank" if space == "full" else f"rank_{space}"
        ranked = sorted((t for t in out if out[t]["distance"][RANK_CELL][space]["paired"]),
                        key=lambda t: out[t]["distance"][RANK_CELL][space]["paired"]["median"])
        for i, t in enumerate(ranked, 1):
            out[t][key] = i
    by_unpaired = sorted(out, key=lambda t: out[t]["distance"][RANK_CELL]["full"]["unpaired"]["median"])
    for i, t in enumerate(by_unpaired, 1):
        out[t]["rank_unpaired"] = i
    return out


def _json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _value(types, t, cell, measure, kind="paired"):
    """A type's median of one measure: a space name (paired/unpaired) or probe_shift/outside_bank_pca."""
    v = types[t]["distance"][cell][measure]
    if measure in SPACES:
        v = v[kind]
    return None if v is None else v["median"]


def _spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    rho, p = spearmanr(*zip(*pairs))
    return {"n": len(pairs), "spearman": round(float(rho), 3), "p": round(float(p), 3)}


def checks(types, layers, row_labels):
    """Rank stability across layers, agreement between spaces, identical prompts, and the
    correlations with behaviour used as a preview."""
    out = {"rank_stability": {}, "space_agreement": {}, "identical_prompts": {}, "preview": {}}
    paired_types = [t for t in types if "rank" in types[t] and t != REFERENCE_TYPE]
    for space in SPACES:
        for pos in POSITIONS:
            cells = [f"L{n}_{pos}" for n in layers if n in (14, 18, 22)]
            for a, b in zip(cells, cells[1:]):
                out["rank_stability"][f"{space} {a} vs {b}"] = _spearman(
                    [_value(types, t, a, space) for t in paired_types],
                    [_value(types, t, b, space) for t in paired_types])["spearman"]
    for pos in POSITIONS:
        cell = f"L18_{pos}"
        for a, b in (("full", "horizon"), ("full", "bank_pca"), ("horizon", "bank_pca"), ("horizon", "probe_shift")):
            out["space_agreement"][f"{cell} {a} vs {b}"] = _spearman(
                [_value(types, t, cell, a) for t in paired_types], [_value(types, t, cell, b) for t in paired_types])
        same = [r["distance"][cell]["full"]["paired"] for r in row_labels
                if r["prompt_identical_to_original"] and r["pairing"] == "twin"]
        out["identical_prompts"][cell] = {"n": len(same), "max_full": round(max(same), 4) if same else None}
    out["paired_vs_unpaired_rank"] = _spearman([_value(types, t, RANK_CELL, "full") for t in paired_types],
                                               [_value(types, t, RANK_CELL, "full", "unpaired") for t in paired_types])
    for cell in ("L18_suffix_end", "L18_prelabel"):
        for measure in (*SPACES, "probe_shift", "outside_bank_pca"):
            for target in ("accuracy", "probe_spearman"):
                ys = [types[t]["accuracy"] if target == "accuracy" else types[t]["probe_spearman"][cell]
                      for t in paired_types]
                out["preview"][f"{cell} {measure} vs {target}"] = _spearman(
                    [_value(types, t, cell, measure) for t in paired_types], ys)
    return out


def _fmt(v, ci=True):
    if v is None:
        return "-"
    return f"{v['median']:.2f} [{v['ci90'][0]:.2f}, {v['ci90'][1]:.2f}]" if ci else f"{v['median']:.2f}"


def markdown(types, cells, check, repeat):
    order = sorted(types, key=lambda t: (types[t].get("rank", 99), types[t]["rank_unpaired"]))
    lines = [
        "# Eval types ranked by distance from Alan's prompt (Qwen3-8B, thinking on)",
        "",
        "Distance = cosine distance of the hidden state to the same decision in Alan's template, after "
        "centring, divided by the median distance between two different Alan prompts of the same regime "
        "(0 = same state, 1 = as far as another Alan prompt). Median over the type's rows [bootstrap 90%]. "
        f"Ranked by the full space at {RANK_CELL}. E14's horizon-phrase position is not saved: suffix_end "
        "stands in for it.",
        "",
        "## Full 4096-d state",
        "",
        "| rank | type | family | what changes | n | L18 suffix | L18 prelabel | L14 suffix | L22 suffix "
        "| unpaired L18 suffix (rank) | E8 / E13 / E14 | accuracy | probe ρ L18 suffix |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in order:
        x = types[t]
        d = x["distance"]
        n = (d[RANK_CELL]["full"]["paired"] or {}).get("n", 0)
        acc = "-" if x["accuracy"] is None else f"{x['accuracy']:.2f}"
        rho = x["probe_spearman"]["L18_suffix_end"]
        st = x["steering"]
        lines.append(
            f"| {x.get('rank', '-')} | {t} | {x['family']} | {x['changes']} | {n} "
            f"| {_fmt(d['L18_suffix_end']['full']['paired'])} | {_fmt(d['L18_prelabel']['full']['paired'])} "
            f"| {_fmt(d['L14_suffix_end']['full']['paired'])} | {_fmt(d['L22_suffix_end']['full']['paired'])} "
            f"| {d[RANK_CELL]['full']['unpaired']['median']:.2f} ({x['rank_unpaired']}) "
            f"| {st['E8']} / {st['E13']} / {st['E14']} | {acc} | {'-' if rho is None else f'{rho:.2f}'} |")

    c18s, c18p = cells["L18_suffix_end"], cells["L18_prelabel"]
    lines += [
        "", "## Three spaces at L18", "",
        f"horizon = the {c18s['horizon_k']}-d steering subspace (top PCs of the full bank's per-horizon averages); "
        f"bank PCA = the full bank's top PCs covering {BANK_VARIANCE:.0%} of its variance "
        f"(k = {c18s['bank_pca_k']} at suffix, {c18p['bank_pca_k']} at prelabel); outside = share of the row's "
        f"variance outside them (Alan's prompts in this run: {c18s['originals_outside_bank_pca']:.2f} at suffix, "
        f"{c18p['originals_outside_bank_pca']:.2f} at prelabel); probe shift = |probe readout - original's| in decades. "
        "Rank in brackets.",
        "",
        "| type | suffix: full | horizon | bank PCA | outside | probe shift | prelabel: full | horizon | bank PCA | outside | probe shift |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in order:
        x, row = types[t], [f"| {t} "]
        for cell in ("L18_suffix_end", "L18_prelabel"):
            d = x["distance"][cell]
            for space in SPACES:
                rank = x.get("rank" if space == "full" else f"rank_{space}") if cell == RANK_CELL else None
                row.append(f"| {_fmt(d[space]['paired'], ci=False)}{f' ({rank})' if rank else ''} ")
            row.append(f"| {_fmt(d['outside_bank_pca'], ci=False)} | {_fmt(d['probe_shift'], ci=False)} ")
        lines.append("".join(row) + "|")

    lines += ["", "## Checks", ""]
    lines.append("Rank agreement between layers (Spearman over paired types): "
                 + ", ".join(f"{k}: {v:.2f}" for k, v in check["rank_stability"].items()) + ".")
    lines.append("")
    lines.append("Agreement between spaces at L18 (Spearman over paired types): "
                 + ", ".join(f"{k}: {v['spearman']:.2f}" for k, v in check["space_agreement"].items()) + ".")
    lines.append("")
    lines.append(f"Paired vs unpaired ranking at {RANK_CELL}: Spearman {check['paired_vs_unpaired_rank']['spearman']:.2f}.")
    for cell, v in check["identical_prompts"].items():
        lines.append(f"Rows whose prompt is word for word their twin's, {cell}: n = {v['n']}, max full distance {v['max_full']}.")
    lines += ["", "Noise floor: the same twin prompt run on two boxes (v2 type vs its original), L18:", "",
              "| cell, measure | n | median | max |", "|---|---|---|---|"]
    for k, v in repeat.items():
        lines.append(f"| {k} | {v['n']} | {v['median']:.3f} | {v['max']:.3f} |")
    lines += ["", "## Preview: distance vs behaviour (Spearman over paired types, uncorrected)", "",
              "| cell | measure | against | n | Spearman | p |", "|---|---|---|---|---|---|"]
    for k, v in check["preview"].items():
        cell, measure, _, target = k.split(" ")
        lines.append(f"| {cell} | {measure} | {target} | {v['n']} | {v['spearman']:.2f} | {v['p']:.3f} |")
    lines += ["", "Scales (median distance between two Alan prompts), L18: "
              + ", ".join(f"{c} {k.removeprefix('scale_')} {v:.3f}" for c in ("L18_suffix_end", "L18_prelabel")
                          for k, v in cells[c].items() if k.startswith("scale_")) + "."]
    return "\n".join(lines) + "\n"


def main(run_dir, bank_dir):
    run_dir = Path(run_dir)
    layers, cells, row_labels, repeat = compute(run_dir, bank_dir)
    types = type_table(row_labels, run_dir)
    check = checks(types, layers, row_labels)
    check["repeated_prompts"] = repeat
    (run_dir / "eval_distance_rows.json").write_text(json.dumps(row_labels, indent=1))
    (run_dir / "eval_distance.json").write_text(json.dumps(
        {"rank_cell": RANK_CELL, "layers": layers, "cells": cells, "checks": check, "types": types}, indent=1))
    md = markdown(types, cells, check, repeat)
    (run_dir / "eval_distance.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
