"""E19 inputs, prepared locally (EvalsPromptsV2.md §8): the full bank's answer-slot cells and the
eval-type prompts to steer with them, in both modes.

    python -m pipeline.s1_data_vectors.analysis.evaltypes_steering_prepare ~/SPAR/results/horizon
    -> jared/steering/inputs_Qwen3-8B_evaltypes.npz and .json   (shipped to the box by `just push`)

Cells: `fit_cell` on the full bank's saved activations at the pre-label token, thinking off and on,
at L24 (the vectors E8 steered with, `steering_prepare` defaults without the decision split) and the
other LAYERS of E21's sweep.

Rows: the near- and both-regime rows (base, swap and twin) of every eval type whose answer is a or
b after "I choose:" and whose rational answer follows the horizon rule (so not `loss_framing`).
- thinking off: the prompt rendered with thinking off and "I choose:" prefilled, so the last token
  is the pre-label ':' and the next is the label;
- thinking on: the generation E0 saved in gen_cache, cut before the label, kept only when it is the
  one E0's activations were taken on (think length and kept token ids match), as in E8's T4.

Needs only CPU, the bank runs (exp0_Qwen3-8B_think-{off,on}_full), the eval-type run, gen_cache/ and
the tokenizer.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from pipeline.s1_data_vectors.analysis.horizon_shape import cell_data
from spar_horizon.config import Settings
from spar_horizon.e0_runs import run_dir
from spar_horizon.model_io import cache_load, label_ids, think_end_id
from spar_horizon.steering import encode, fit_cell

MODEL = "Qwen/Qwen3-8B"
# L24 is the cell E19 steers at; the others are E21's layer sweep
LAYERS = (16, 20, 22, 24, 26, 28, 32)
CELLS = tuple((mode, "pre-label", layer) for mode in ("off", "on") for layer in LAYERS)
PREFIX = "I choose:"
EXCLUDED = {"loss_framing": "costs: a longer horizon makes the near option rational, the reverse of the rule"}
REGIMES = ("near", "both")
VARIANTS = ("base", "swap", "twin")
OUT = Path(__file__).resolve().parents[3] / "steering"


def eval_settings(run):
    """The Settings of the eval-type run (its settings.json, minus the run's own extras)."""
    saved = json.loads((Path(run) / "settings.json").read_text())
    fields = {f.name for f in dataclasses.fields(Settings)}
    return Settings(**{k: v for k, v in saved.items() if k in fields})


def steerable(row):
    return (row["regime"] in REGIMES and row["variant"] in VARIANTS and row.get("near_label") in ("a", "b")
            and row["answer_kind"] == "label" and row["labels"] == ["a", "b"] and row["choice_prefix"] == PREFIX)


def eval_rows(run):
    """{type_id: steerable rows} for the types E19 can steer; also {type_id: reason} for those left out."""
    rows, left_out = {}, dict(EXCLUDED)
    for path in sorted(Path(run).glob("*/rows.json")):
        type_id = path.parent.name
        if type_id in EXCLUDED:
            continue
        all_rows = json.loads(path.read_text())
        # the type's own rows decide; its twins are Alan's template and always qualify
        if not any(steerable(r) for r in all_rows if r["variant"] != "twin"):
            left_out[type_id] = "no near/both rows with an a/b label after 'I choose:'"
            continue
        rows[type_id] = [r for r in all_rows if steerable(r) and r.get("act_file")]
    return rows, left_out


def bank_cells(root, model):
    """{key: fit_cell arrays} at CELLS, fitted on the full bank."""
    cells = {}
    for mode, label, layer in CELLS:
        z = np.load(run_dir(root, model, mode) / "activations.npz")
        X, log_h, _ = cell_data(z, label, layer)
        cells[f"{mode}|{label}|{layer}"] = fit_cell(X, log_h)
        print(f"{mode}|{label}|{layer}: fitted on {len(log_h)} bank prompts")
    return cells


def off_sequence(tokenizer, row, cfg_off):
    ids = encode(tokenizer, row["prompt"], cfg_off)
    if tokenizer.decode([ids[-1]]) != ":":
        raise ValueError(f"{row['sample_uid']}: thinking-off sequence does not end at the pre-label ':'")
    return ids


def on_sequence(tokenizer, row, cfg_on, acts):
    """The saved generation cut before the label, the position of </think> and the label pair the
    model used; None when the cache lacks it or it is not the one the activations were taken on."""
    c = dataclasses.replace(cfg_on, choice_prefix=row["choice_prefix"])
    g = cache_load(c, encode(tokenizer, row["prompt"], c), None, "cpu")
    if g is None:
        return None
    ids = g.ids.tolist()
    positions = acts["keep_idx"][row["act_row"]]
    if not (g.think_len == acts["think_len"][row["act_row"]] and positions.max() < len(ids)
            and np.array_equal(np.asarray(ids)[positions], acts["keep_ids"][row["act_row"]])):
        return None
    close_id = think_end_id(tokenizer)
    closes = [p for p in range(g.prompt_len + g.think_len, g.answer_start) if ids[p] == close_id]
    if len(closes) != 1:
        raise ValueError(f"{row['sample_uid']}: {len(closes)} </think> tokens before the label")
    spaced = tokenizer.decode([ids[g.answer_start]]).startswith(" ")
    labels = label_ids(tokenizer, cfg=c) if spaced else label_ids(tokenizer)
    return ids[:g.answer_start], closes[0], labels


def pack(sequences):
    """Concatenated ids and offsets, like E8's cached reasoning."""
    lengths = [len(s) for s in sequences]
    return np.concatenate(sequences).astype(np.int32), np.r_[0, np.cumsum(lengths)].astype(np.int64)


def main(root, out_dir, model=MODEL, name="evaltypes"):
    root = Path(root)
    run = root / f"evaltypes_{model.split('/')[-1]}_think-on"
    cfg_on = eval_settings(run)
    cfg_on.cache_dir = str(root / "gen_cache")
    cfg_off = dataclasses.replace(cfg_on, thinking="off", prefill=PREFIX, choice_prefix="")
    tokenizer = AutoTokenizer.from_pretrained(model)
    by_type, left_out = eval_rows(run)
    rows = [r for type_rows in by_type.values() for r in type_rows]

    off_ids = [off_sequence(tokenizer, r, cfg_off) for r in rows]
    off_labels = label_ids(tokenizer, cfg=cfg_off)
    on_rows, on_ids, on_close, on_labels, dropped = [], [], [], [], []
    acts = {}
    for i, r in enumerate(rows):
        key = (r["type_id"], r["act_file"])
        if key not in acts:
            acts[key] = np.load(run / r["type_id"] / r["act_file"])
        seq = on_sequence(tokenizer, r, cfg_on, acts[key])
        if seq is None:
            dropped.append(r["sample_uid"])
            continue
        on_rows.append(i)
        on_ids.append(seq[0])
        on_close.append(seq[1])
        on_labels.append(seq[2])

    arrays = {}
    cells = bank_cells(root, model)
    for key, cell in cells.items():
        for field, value in cell.items():
            arrays[f"{key}|{field}"] = np.asarray(value)
    arrays["off|ids"], arrays["off|offsets"] = pack(off_ids)
    arrays["off|row"] = np.arange(len(rows))
    arrays["off|labels"] = np.array([off_labels] * len(rows))
    arrays["on|ids"], arrays["on|offsets"] = pack(on_ids)
    arrays["on|row"] = np.array(on_rows)
    arrays["on|labels"] = np.array(on_labels)
    arrays["on|close"] = np.array(on_close)

    keep = ("sample_uid", "type_id", "variant", "twin_id", "regime", "horizon", "horizon_text", "near_label",
            "truth_labels", "label_logits", "forced")
    meta = {"model": model, "source": str(root), "eval_run": str(run),
            "cells": [{"mode": m, "label": lab, "layer": layer, "k": int(cells[f"{m}|{lab}|{layer}"]["basis"].shape[0])}
                      for m, lab, layer in CELLS],
            "types": list(by_type), "left_out": left_out, "dropped_on": dropped,
            "rows": [{k: r.get(k) for k in keep} for r in rows]}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"inputs_{model.split('/')[-1]}_{name}"
    np.savez(out_dir / f"{stem}.npz", **arrays)
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(by_type)} types, {len(rows)} rows thinking off, {len(on_rows)} thinking on "
          f"({len(dropped)} dropped: cached generation is not E0's); left out: {', '.join(left_out)}")
    print(f"wrote {out_dir / stem}.npz and .json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", help="results/horizon: the bank runs, the eval-type run and gen_cache/")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()
    main(args.root, args.out, args.model)
