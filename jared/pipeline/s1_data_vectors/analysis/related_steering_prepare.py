"""E23 inputs (EvalsRelatedQualities.md §3), prepared on the box after `exp0_evaltypes --suite related` (CPU):

    python -m pipeline.s1_data_vectors.analysis.related_steering_prepare $WORKSPACE/results/horizon
    -> jared/steering/related_box/inputs_Qwen3-8B_related.npz and .json

Reads <root>/related_<model>_think-on/<set_id>/{rows.json, acts_*.npz}, <root>/gen_cache and the horizon
cell from related_cells.py. A row is kept (readable) when it has an a/b label after "I choose:", its
answer starts with that label, and its saved generation is the one the activations were taken on
(evaltypes_steering_prepare.on_sequence, as in E19); the others are listed as unreadable with the reason.
Per kept row, fit and eval split alike:
- the thinking-on sequence cut before the label, and the label token pair the model used;
- near_first = (matching_label == "a"), so score_last_edits' "near log-odds" is the behaviour log-odds,
  logit(matching label) - logit(other); the matching option is the behaviour-seeking one (power, survival,
  wealth) or the reward hack (sorh), so + = more seeking/hacking;
- log_h, the start horizon: curve_readout of the row's own pre-label activation at the cell's layer
  (these prompts state no horizon; spec decision).
Behaviour CAA per set (the positive control): the mean pre-label activation of the fit rows whose unsteered
behaviour log-odds (E0's label logits) is > 0, minus that of those < 0. It needs MIN_CLASS rows per class;
a set short of that is recorded as unavailable, with the counts.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from pipeline.s1_data_vectors.analysis.evaltypes_steering_prepare import eval_settings, on_sequence, pack
from spar_horizon.steering import curve_readout

MODEL = "Qwen/Qwen3-8B"
PREFIX = "I choose:"
MIN_CLASS = 10
OUT = Path(__file__).resolve().parents[3] / "steering" / "related_box"
CELL = OUT / "cells_Qwen3-8B_on_L24.npz"
CELL_FIELDS = ("mean", "w", "b", "basis", "curve_logh", "curve_coords", "caa")
KEEP = ("sample_uid", "set_id", "variant", "twin_id", "split", "matching_label", "label_logits", "forced",
        "cheat_method")


def load_cell(path):
    """The fit_cell arrays and the meta related_cells wrote next to them."""
    z = np.load(path)
    cell = {key: z[key] for key in CELL_FIELDS}
    cell["b"] = float(cell["b"])
    return cell, json.loads(Path(path).with_suffix(".json").read_text())


def unreadable_reason(row):
    """Why the row cannot be scored from its rows.json entry alone, or None."""
    if row["labels"] != ["a", "b"] or row["choice_prefix"] != PREFIX:
        return "not an a/b row after 'I choose:'"
    if row.get("act_file") is None:
        return "generation too short to keep"
    if not re.match(r"\s*[ab]\b", row["answer"]):
        return "answer does not start with a or b"
    return None


def behaviour_log_odds(row):
    """logit(matching label) - logit(other) from E0's label logits (in `labels` order, a then b)."""
    a_minus_b = row["label_logits"][0] - row["label_logits"][1]
    return a_minus_b if row["matching_label"] == "a" else -a_minus_b


def behaviour_caa(rows, X, lo):
    """{set_id: CAA vector [d]} for the sets that have one, and {set_id: availability and class counts}."""
    vectors, info = {}, {}
    for set_id in dict.fromkeys(r["set_id"] for r in rows):
        fit = np.array([r["set_id"] == set_id and r["split"] == "fit" for r in rows])
        matching, other = fit & (lo > 0), fit & (lo < 0)
        available = bool(min(matching.sum(), other.sum()) >= MIN_CLASS)
        info[set_id] = {"available": available, "n_matching": int(matching.sum()), "n_other": int(other.sum())}
        if available:
            vectors[set_id] = X[matching].mean(0) - X[other].mean(0)
    return vectors, info


def load_inputs(path):
    """What main() wrote: meta, cell, {set_id: CAA}, per-row ids lists, label pairs, near_first, log_h."""
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    z = np.load(path)
    cell = {key: z[f"cell|{key}"] for key in CELL_FIELDS}
    cell["b"] = float(cell["b"])
    offsets = z["offsets"]
    return {"meta": meta, "cell": cell,
            "caa": {s: z[f"caa|{s}"] for s, info in meta["caa"].items() if info["available"]},
            "ids": [z["ids"][a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])],
            "labels": [tuple(int(t) for t in pair) for pair in z["labels"]],
            "near_first": z["near_first"].tolist(), "log_h": z["log_h"]}


def main(root, out_dir, cell_path=CELL, model=MODEL):
    root = Path(root)
    run = root / f"related_{model.split('/')[-1]}_think-on"
    cfg_on = eval_settings(run)
    cfg_on.cache_dir = str(root / "gen_cache")
    tokenizer = AutoTokenizer.from_pretrained(model)
    cell, cell_meta = load_cell(cell_path)
    layer = cell_meta["layer"]

    rows, ids, labels, states, unreadable, acts = [], [], [], [], [], {}
    for rows_file in sorted(run.glob("*/rows.json")):
        for r in json.loads(rows_file.read_text()):
            reason = unreadable_reason(r)
            if reason is None:
                act_path = rows_file.parent / r["act_file"]
                if act_path not in acts:
                    # read each array once: indexing a lazy NpzFile re-reads the whole array per row, and the
                    # row slices kept as views pin every copy (OOM on the full run, 400 rows x ~110 MB per file)
                    with np.load(act_path) as z:
                        acts[act_path] = {key: z[key] for key in z.files}
                seq = on_sequence(tokenizer, r, cfg_on, acts[act_path])
                if seq is None:
                    reason = "cached generation is not the one the activations were taken on"
            if reason is not None:
                unreadable.append({**{k: r.get(k) for k in ("sample_uid", "set_id", "variant", "twin_id", "split")},
                                   "reason": reason})
                continue
            rows.append(r)
            ids.append(seq[0])
            labels.append(seq[2])
            a = acts[act_path]
            states.append(a[f"L{layer}"][r["act_row"], a["region_prelabel"][0]])
    if not rows:
        raise ValueError(f"no readable rows in {run} ({len(unreadable)} unreadable)")

    X = np.stack(states).astype(np.float64)
    log_h = curve_readout(cell, X)
    caa, caa_info = behaviour_caa(rows, X, np.array([behaviour_log_odds(r) for r in rows]))

    arrays = {f"cell|{key}": np.asarray(value) for key, value in cell.items()}
    arrays["ids"], arrays["offsets"] = pack(ids)
    arrays["labels"] = np.array(labels)
    arrays["near_first"] = np.array([r["matching_label"] == "a" for r in rows])
    arrays["log_h"] = log_h
    arrays.update({f"caa|{set_id}": vector for set_id, vector in caa.items()})
    meta = {"model": model, "source": str(root), "run": str(run), "cell": str(cell_path), "cell_meta": cell_meta,
            "layer": layer, "caa": caa_info, "unreadable": unreadable,
            "rows": [{k: r[k] for k in KEEP if k in r} for r in rows]}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"inputs_{model.split('/')[-1]}_related"
    np.savez(out_dir / f"{stem}.npz", **arrays)
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    for set_id, info in caa_info.items():
        print(f"{set_id}: {sum(r['set_id'] == set_id for r in rows)} rows, behaviour CAA "
              f"{'available' if info['available'] else 'UNAVAILABLE'} ({info['n_matching']} matching, "
              f"{info['n_other']} other fit rows), start log_h median "
              f"{np.median(log_h[[r['set_id'] == set_id for r in rows]]):.2f}")
    print(f"{len(rows)} rows kept, {len(unreadable)} unreadable; wrote {out_dir / stem}.npz and .json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", help="results/horizon: the related run and gen_cache/")
    p.add_argument("--cell", default=str(CELL), help="cells_*.npz from related_cells.py")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()
    main(args.root, args.out, args.cell, args.model)
