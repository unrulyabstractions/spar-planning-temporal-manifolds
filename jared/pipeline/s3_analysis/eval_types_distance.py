"""How far is each eval type from Alan's prompt, inside Qwen3-8B? Ranks the types and labels every
row, so later steering runs (E8 answer slot, E13 suffix, E14 horizon phrase) can be read against
distance.

    python -m pipeline.s3_analysis.eval_types_distance $RESULTS/evaltypes_Qwen3-8B_think-on

Matched original. Each row is compared with the same decision written in Alan's template:
- types with twins: a `base` row against its item's `twin` row (same options, same order, horizon
  stated in Alan's CONSTRAINT line);
- the wording ladder: `t0_w1_cash` / `t0_w3_clear` rows against the `t0_w0` row with the same item
  and variant (`t0_w0` is Alan's wording, so it is the reference at distance 0);
- `free_form` / `free_form_anchor` have no twin; they get only the unpaired distance below.

Distance, per layer and position (suffix_end = last prompt token, where E13 steers and the stand-in
for E14's horizon phrase, which is not saved; prelabel = the token before the answer label, where
E8 steers): cosine distance between the two hidden states after subtracting the mean over all rows,
divided by the median distance between two different original prompts of the same regime. So 0 is
"same state as Alan's prompt" and 1 is "as far as another of Alan's prompts".

Unpaired distance, for every row: cosine distance to the mean of the original prompts, divided by
the median of the same for the originals themselves (originals sit near 1).

A type's score is the median over its rows, with a bootstrap 90% interval over rows. The ranking
uses L18 suffix_end. Writes `eval_distance.json`, `eval_distance.md` and `eval_distance_rows.json`.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from spar_horizon.eval_types import TYPES

POSITIONS = {"suffix_end": 4, "prelabel": 10}   # indices into the 17 saved tokens
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


# ---- the run ----

def compute(run_dir):
    rows_by_type = load_rows(run_dir)
    layers, states = load_states(run_dir, rows_by_type)
    match = match_originals(rows_by_type)
    matched = {u: orig for u, (orig, _) in match.items()}
    prompt_of = {r["sample_uid"]: r["prompt"] for rows in rows_by_type.values() for r in rows}
    originals = original_prompts(rows_by_type)
    original_uids = [r["sample_uid"] for r in originals]
    original_regimes = [r["regime"] for r in originals]
    uids = list(states)
    stacked = np.stack([states[u] for u in uids])            # (n, layers, positions, d)

    rows_out = {u: {"distance": {}} for u in uids}
    scales = {}
    for li, layer in enumerate(layers):
        for pi, pos_name in enumerate(POSITIONS):
            cell = f"L{layer}_{pos_name}"
            paired, unpaired, scales[cell] = cell_distances(stacked[:, li, pi], uids, original_uids,
                                                            original_regimes, matched)
            for u in uids:
                rows_out[u]["distance"][cell] = {"paired": round(paired[u], 4) if u in paired else None,
                                                 "unpaired": round(unpaired[u], 4)}

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
                "distance": rows_out[u]["distance"]})
    return layers, scales, row_labels, repeat_check(stacked, uids, prompt_of, layers, scales)


def repeat_check(stacked, uids, prompt_of, layers, scales):
    """The same prompt run twice (a v2 type's twin is its original type's twin, rerun a day later):
    scaled distance between the two states at each position of L18. Should be near 0."""
    first, gaps = {}, {pos: [] for pos in POSITIONS}
    li = layers.index(18)
    means = stacked[:, li].mean(axis=0)                      # centred like every other distance
    for i, u in enumerate(uids):
        if not u.endswith("_twin"):
            continue
        j = first.setdefault(prompt_of[u], i)
        if j != i:
            for pi, pos in enumerate(POSITIONS):
                d = cosine_distance(stacked[i, li, pi] - means[pi], stacked[j, li, pi] - means[pi])
                d /= scales[f"L18_{pos}"]
                gaps[pos].append(float(d))
    return {pos: {"n": len(g), "median": round(float(np.median(g)), 4) if g else None,
                  "max": round(max(g), 4) if g else None} for pos, g in gaps.items()}


def type_table(layers, row_labels, run_dir):
    rng = np.random.default_rng(0)
    summary = _json(Path(run_dir) / "summary.json")
    probe = _json(Path(run_dir) / "probe_transfer.json").get("types", {})
    out = {}
    for type_id in dict.fromkeys(r["type_id"] for r in row_labels):
        rows = [r for r in row_labels if r["type_id"] == type_id and r["variant"] != "twin"]
        by_cell = {}
        for cell in rows[0]["distance"]:
            paired = [r["distance"][cell]["paired"] for r in rows if r["distance"][cell]["paired"] is not None]
            by_cell[cell] = {"paired": median_ci(paired, rng),
                             "unpaired": median_ci([r["distance"][cell]["unpaired"] for r in rows], rng)}
        s = summary.get(type_id, {})
        out[type_id] = {
            "family": family(type_id), "changes": CHANGES.get(type_id, ""), "steering": steering_sites(type_id),
            "pairing": sorted({r["pairing"] for r in rows}),
            "accuracy": (s.get("accuracy") or {}).get("p"),
            "probe_spearman": {c: (probe.get(type_id, {}).get(c) or {}).get("spearman")
                               for c in ("L18_suffix_end", "L18_prelabel")},
            "distance": by_cell}
    ranked = sorted((t for t in out if out[t]["distance"][RANK_CELL]["paired"]),
                    key=lambda t: out[t]["distance"][RANK_CELL]["paired"]["median"])
    for i, t in enumerate(ranked, 1):
        out[t]["rank"] = i
    by_unpaired = sorted(out, key=lambda t: out[t]["distance"][RANK_CELL]["unpaired"]["median"])
    for i, t in enumerate(by_unpaired, 1):
        out[t]["rank_unpaired"] = i
    return out


def _json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _median(types, t, cell, kind="paired"):
    v = types[t]["distance"][cell][kind]
    return None if v is None else v["median"]


def checks(types, layers, row_labels):
    """Rank stability across layers, identical prompts, and the correlations used as a preview."""
    out = {"rank_stability": {}, "identical_prompts": {}, "preview": {}}
    paired_types = [t for t in types if "rank" in types[t] and t != REFERENCE_TYPE]
    for pos in POSITIONS:
        cells = [f"L{n}_{pos}" for n in layers if n in (14, 18, 22)]
        for a, b in zip(cells, cells[1:]):
            rho = spearmanr([_median(types, t, a) for t in paired_types],
                            [_median(types, t, b) for t in paired_types])[0]
            out["rank_stability"][f"{a} vs {b}"] = round(float(rho), 3)
    for pos in POSITIONS:
        cell = f"L18_{pos}"
        same = [r["distance"][cell]["paired"] for r in row_labels
                if r["prompt_identical_to_original"] and r["pairing"] == "twin"]
        differ = [r["distance"][cell]["paired"] for r in row_labels
                  if r["prompt_identical_to_original"] is False]
        out["identical_prompts"][cell] = {"n_identical": len(same),
                                          "max_identical": round(max(same), 4) if same else None,
                                          "median_different": round(float(np.median(differ)), 3)}
    rho = spearmanr([_median(types, t, RANK_CELL) for t in paired_types],
                    [_median(types, t, RANK_CELL, "unpaired") for t in paired_types])[0]
    out["paired_vs_unpaired_rank"] = round(float(rho), 3)
    for cell in ("L18_suffix_end", "L18_prelabel"):
        for target in ("accuracy", "probe_spearman"):
            pairs = [(_median(types, t, cell),
                      types[t]["accuracy"] if target == "accuracy" else types[t]["probe_spearman"][cell])
                     for t in paired_types]
            pairs = [(d, y) for d, y in pairs if y is not None]
            rho, p = spearmanr(*zip(*pairs))
            out["preview"][f"{cell} vs {target}"] = {"n": len(pairs), "spearman": round(float(rho), 3),
                                                      "p": round(float(p), 3)}
    return out


def _fmt(v):
    if v is None:
        return "-"
    return f"{v['median']:.2f} [{v['ci90'][0]:.2f}, {v['ci90'][1]:.2f}]"


def markdown(types, scales, check):
    lines = [
        "# Eval types ranked by distance from Alan's prompt (Qwen3-8B, thinking on)",
        "",
        "Distance = cosine distance of the hidden state to the same decision in Alan's template, after "
        "centring, divided by the median distance between two different Alan prompts of the same regime "
        "(0 = same state, 1 = as far as another Alan prompt). Median over the type's rows [bootstrap 90%]. "
        "Unpaired = distance to the mean Alan prompt (originals sit near 1). "
        f"Ranked by {RANK_CELL}. E14's horizon-phrase position is not saved: suffix_end stands in for it.",
        "",
        "| rank | type | family | what changes | n | L18 suffix | L18 prelabel | L14 suffix | L22 suffix "
        "| unpaired L18 suffix (rank) | E8 / E13 / E14 | accuracy | probe ρ L18 suffix |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = sorted(types, key=lambda t: (types[t].get("rank", 99), types[t]["rank_unpaired"]))
    for t in order:
        x = types[t]
        d = x["distance"]
        n = (d[RANK_CELL]["paired"] or {}).get("n", 0)
        acc = "-" if x["accuracy"] is None else f"{x['accuracy']:.2f}"
        rho = x["probe_spearman"]["L18_suffix_end"]
        st = x["steering"]
        lines.append(
            f"| {x.get('rank', '-')} | {t} | {x['family']} | {x['changes']} | {n} | {_fmt(d['L18_suffix_end']['paired'])} "
            f"| {_fmt(d['L18_prelabel']['paired'])} | {_fmt(d['L14_suffix_end']['paired'])} "
            f"| {_fmt(d['L22_suffix_end']['paired'])} | {d[RANK_CELL]['unpaired']['median']:.2f} ({x['rank_unpaired']}) "
            f"| {st['E8']} / {st['E13']} / {st['E14']} | {acc} | {'-' if rho is None else f'{rho:.2f}'} |")
    lines += ["", "## Checks", "",
              "Rank agreement between layers (Spearman over paired types): "
              + ", ".join(f"{k}: {v:.2f}" for k, v in check["rank_stability"].items()) + ".", "",
              f"Paired vs unpaired ranking at {RANK_CELL}: Spearman {check['paired_vs_unpaired_rank']:.2f}.", ""]
    for cell, v in check["identical_prompts"].items():
        lines.append(f"Rows whose prompt is word for word their twin's, {cell}: n = {v['n_identical']}, "
                     f"max distance {v['max_identical']} (rows that differ: median {v['median_different']}).")
    for pos, v in check["repeated_prompts"].items():
        lines.append(f"Same twin prompt in two runs (v2 type vs its original), L18_{pos}: n = {v['n']}, "
                     f"median distance {v['median']}, max {v['max']}: the noise floor at this position.")
    lines += ["", "## Preview: distance vs behaviour (Spearman over paired types)", "",
              "| distance | against | n | Spearman | p |", "|---|---|---|---|---|"]
    for k, v in check["preview"].items():
        cell, target = k.split(" vs ")
        lines.append(f"| {cell} | {target} | {v['n']} | {v['spearman']:.2f} | {v['p']:.3f} |")
    lines += ["", "Scale (median distance between two Alan prompts): "
              + ", ".join(f"{c} {s:.3f}" for c, s in scales.items() if c.startswith(("L14", "L18", "L22"))) + "."]
    return "\n".join(lines) + "\n"


def main(run_dir):
    run_dir = Path(run_dir)
    layers, scales, row_labels, repeats = compute(run_dir)
    types = type_table(layers, row_labels, run_dir)
    check = checks(types, layers, row_labels)
    check["repeated_prompts"] = repeats
    (run_dir / "eval_distance_rows.json").write_text(json.dumps(row_labels, indent=1))
    (run_dir / "eval_distance.json").write_text(json.dumps(
        {"rank_cell": RANK_CELL, "layers": layers, "scales": scales, "checks": check, "types": types}, indent=1))
    md = markdown(types, scales, check)
    (run_dir / "eval_distance.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1])
