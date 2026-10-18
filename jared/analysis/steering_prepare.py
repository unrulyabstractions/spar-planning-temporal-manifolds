"""E8 inputs, prepared locally (spec: 07_spec_steering.md, section 6): per-cell
directions fitted on a run's saved activations, and the cached thinking-on
reasoning of its near- and both-regime prompts.

    python analysis/steering_prepare.py results/horizon
    -> jared/steering/inputs_Qwen3-8B_answer_slot.npz and .json   (shipped to the box by `just push`)

    python analysis/steering_prepare.py results/horizon --name first_run \
        --cells "off:suffix newline:16,off:pre-label:21,on:think close:19,on:pre-label:21"
    -> the first run's cells (07_spec_steering.md section 3)

    python analysis/steering_prepare.py results/horizon --name decision_L24 --cells on:pre-label:24 --natural-only
    -> fitted on natural think closes only, as in 07_spec_steering.md section 10

Every cell also carries `decision`: the near-vs-far direction inside its horizon
subspace (steering.decision_direction), so E8 can split the curve into its
decision part and the rest.

Needs only CPU, the run folders (exp0_*_think-off_full, exp0_*_think-on_full,
gen_cache/) and the model's tokenizer.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from horizon_shape import cell_data  # noqa: E402  (same folder)
from steering_analysis import saved_decision  # noqa: E402  (same folder)
from spar_horizon.banks import build_records  # noqa: E402
from spar_horizon.config import Settings  # noqa: E402
from spar_horizon.model_io import cache_load, label_ids, think_end_id  # noqa: E402
from spar_horizon.steering import decision_direction, encode, fit_cell  # noqa: E402

# Second run (07_spec_steering.md section 9): the answer slot where patching says the decision lives
CELLS = "off:pre-label:24,off:pre-label:27,off:pre-label:30,on:pre-label:24,on:pre-label:27,on:pre-label:30"
NAME = "answer_slot"
OUT = Path(__file__).resolve().parent.parent / "steering"


def parse_cells(text):
    """"mode:label:layer,..." -> [(mode, label, layer)]."""
    cells = []
    for item in text.split(","):
        mode, label, layer = item.split(":")
        if mode not in ("off", "on"):
            raise ValueError(f"cell {item!r}: mode must be off or on")
        cells.append((mode, label, int(layer)))
    return cells


def run_settings(root, mode):
    """The Settings of the full-bank E0 run in `root` for `mode`, cache under root."""
    manifest = json.loads(next(root.glob(f"exp0_*_think-{mode}_full/manifest.json")).read_text())
    known = {f.name for f in dataclasses.fields(Settings)}
    cfg = Settings(**{k: v for k, v in manifest["settings"].items() if k in known})
    cfg.cache_dir = str(root / "gen_cache")
    return cfg


def cached_reasoning(cfg, records, selected, tokenizer):
    """Arrays for the thinking-on prompts in `selected` whose generation is cached:
    their ids up to the label (concatenated, with offsets), the position of the
    </think> token, think length, forced flag, the label token pair in the form
    the model used, and the index into `selected`. Also the indices it could not
    find. A forced close is "\n</think>\n\n" appended to the cut-off reasoning,
    so </think> is located, not computed, as in extract()."""
    rows, missing = [], []
    close_id = think_end_id(tokenizer)
    with_space, without_space = label_ids(tokenizer, cfg=cfg), label_ids(tokenizer)
    for j, i in enumerate(selected):
        g = cache_load(cfg, encode(tokenizer, records[i]["prompt"], cfg), None, "cpu")
        if g is None:
            missing.append(i)
            continue
        ids = g.ids.tolist()
        closes = [p for p in range(g.prompt_len + g.think_len, g.answer_start) if ids[p] == close_id]
        if len(closes) != 1:
            raise ValueError(f"prompt {i}: {len(closes)} </think> tokens between the think span and the label")
        label_text = tokenizer.decode([ids[g.answer_start]])
        rows.append({"ids": ids[:g.answer_start], "close": closes[0], "think_len": g.think_len,
                     "forced": g.forced, "labels": with_space if label_text.startswith(" ") else without_space,
                     "record": j})
    if not rows:
        return None, missing
    lengths = [len(r["ids"]) for r in rows]
    arrays = {"ids": np.concatenate([r["ids"] for r in rows]).astype(np.int32),
              "offsets": np.r_[0, np.cumsum(lengths)].astype(np.int64),
              "close": np.array([r["close"] for r in rows]),
              "think_len": np.array([r["think_len"] for r in rows]),
              "forced": np.array([r["forced"] for r in rows]),
              "labels": np.array([r["labels"] for r in rows]),
              "record": np.array([r["record"] for r in rows])}
    return arrays, missing


def main(root, out_dir, cells_spec, name, natural_only=False):
    root = Path(root)
    cfg_on = run_settings(root, "on")
    records = build_records(cfg_on)
    selected = [i for i, r in enumerate(records) if r["regime"] in ("near", "both")]
    arrays, cells = {}, []
    for mode, label, layer in parse_cells(cells_spec):
        run = next(root.glob(f"exp0_*_think-{mode}_full"))
        z = np.load(run / "activations.npz")
        X, log_h, forced = cell_data(z, label, layer)
        _, near_lo, _ = saved_decision(run, z)
        # forced closes separate perfectly from natural ones at these cells and their rate
        # varies with the horizon, so directions fitted on all prompts partly read "was cut off"
        keep = ~forced if natural_only else np.ones(len(log_h), bool)
        cell = fit_cell(X[keep], log_h[keep])
        cell["decision"], decision_r2 = decision_direction(X[keep], near_lo[keep], cell)
        key = f"{mode}|{label}|{layer}"
        for field, value in cell.items():
            arrays[f"{key}|{field}"] = np.asarray(value)
        cells.append({"mode": mode, "label": label, "layer": layer, "k": int(cell["basis"].shape[0]),
                      "n": int(keep.sum()), "natural_only": natural_only, "decision_r2": decision_r2})
        print(f"{key}: {keep.sum()} prompts{' (natural closes)' if natural_only else ''}, "
              f"horizon-average basis k={cell['basis'].shape[0]}, near log-odds from the basis R² {decision_r2:.2f}")
    tokenizer = AutoTokenizer.from_pretrained(cfg_on.model)
    post, missing = cached_reasoning(cfg_on, records, selected, tokenizer)
    if post is not None:
        arrays.update({f"post|{name}": value for name, value in post.items()})
    print(f"cached reasoning: {0 if post is None else len(post['record'])} of {len(selected)} near/both prompts"
          + (f"; missing {len(missing)} (thinking-on test T4 will skip them)" if missing else ""))
    meta = {"model": cfg_on.model, "n_prompts": cfg_on.n_prompts, "seed": cfg_on.seed, "limit": cfg_on.limit,
            "selected": selected, "sample_uids": [records[i]["sample_uid"] for i in selected], "cells": cells,
            "n_post": 0 if post is None else int(len(post["record"])), "source": str(root)}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"inputs_{cfg_on.model.split('/')[-1]}_{name}.npz"
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) and {path.with_suffix('.json').name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="folder with the full-bank E0 runs and gen_cache/")
    parser.add_argument("--out", type=Path, default=OUT, help=f"output folder (default {OUT})")
    parser.add_argument("--cells", default=CELLS, help='"mode:label:layer,..." (default: the answer slot at L24/27/30)')
    parser.add_argument("--name", default=NAME, help=f"suffix of the inputs file and of E8's output folder ({NAME})")
    parser.add_argument("--natural-only", action="store_true",
                        help="fit directions on prompts whose think block closed on its own")
    args = parser.parse_args()
    main(args.root, args.out, args.cells, args.name, args.natural_only)
