"""E19: does the full bank's answer-slot steering move the choice on the eval types?
(EvalsPromptsV2.md §8)

    python -m pipeline.s2_steering.experiments.ablate_steer.exp19_evaltypes_steer \\
        --inputs steering/inputs_Qwen3-8B_evaltypes.npz --out /workspace/results/horizon
    python -m pipeline.s2_steering.experiments.ablate_steer.exp19_evaltypes_steer \\
        --inputs steering/inputs_Qwen3-8B_evaltypes.npz --limit 4 --out /workspace/results/horizon/smoke   # smoke

Inputs from pipeline/s1_data_vectors/analysis/evaltypes_steering_prepare.py. Per mode (thinking off:
the prompt with "I choose:" prefilled; thinking on: E0's own reasoning, cut before the label), one
forward pass per row and condition, the edit at the last position (the pre-label ':') at L24:
- steer: none, the curve of horizon averages +-2 decades, a random direction of the same length +-2
  (E8's T3/T4 directions and step sizes, the bank's cells);
- patch: E8's T5 within each type, the pre-label state of a prompt from the other regime (same near
  label) written into a recipient, at every layer; transfer fraction per layer, the type's ceiling.
Outcome: near log-odds, logit(near label) - logit(far label). Writes <out>/exp19_<inputs name>/
results.json (per row: sample_uid, base, every condition; per type and mode: patch transfer),
saved after each mode.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pipeline.s2_steering.experiments.ablate_steer.exp8_steering import Runner, adder, delta_rows, run_answer
from spar_horizon.config import Settings
from spar_horizon.model_io import load_model
from spar_horizon.runs import code_hash, out_base, stamp

CONDITIONS = tuple((direction, alpha) for direction in ("curve", "random 0") for alpha in (-2, 2))
PATCH_PAIRS = 12       # per type and mode
LAYER = 24


def load(path):
    """meta, cells {(mode, layer): arrays}, sequences {mode: (ids lists, labels, row index)}."""
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    z = np.load(path)
    cells = {}
    for c in meta["cells"]:
        key = f"{c['mode']}|{c['label']}|{c['layer']}"
        cell = {name: z[f"{key}|{name}"] for name in ("mean", "w", "basis", "curve_logh", "curve_coords", "caa")}
        cells[c["mode"], c["layer"]] = {**cell, "b": float(z[f"{key}|b"])}
    sequences = {}
    for mode in ("off", "on"):
        offsets = z[f"{mode}|offsets"]
        ids = [z[f"{mode}|ids"][a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
        labels = [tuple(int(t) for t in pair) for pair in z[f"{mode}|labels"]]
        sequences[mode] = (ids, labels, z[f"{mode}|row"].tolist())
    return meta, cells, sequences


def limited(meta, sequences, limit):
    """The first `limit` rows of each type and variant (smoke)."""
    seen, keep_rows = {}, set()
    for i, r in enumerate(meta["rows"]):
        key = (r["type_id"], r["variant"])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= limit:
            keep_rows.add(i)
    out = {}
    for mode, (ids, labels, rows) in sequences.items():
        keep = [j for j, row in enumerate(rows) if row in keep_rows]
        out[mode] = ([ids[j] for j in keep], [labels[j] for j in keep], [rows[j] for j in keep])
    return out


def prompt_set(ids, labels, near_first, batch_size):
    variant = {"ids": ids, "labels": labels, "near_first": near_first,
               "positions": {"pre-label": [[len(seq) - 1] for seq in ids]}}
    return {"variants": [variant], "batch_size": batch_size}


def run_mode(runner, mode, cell, meta, ids, labels, rows, batch_size, patch):
    records = [meta["rows"][i] for i in rows]
    near_first = [r["near_label"] == "a" for r in records]
    log_h = np.log10([r["horizon"] for r in records])
    prompts = prompt_set(ids, labels, near_first, batch_size)
    edits = [None] + [adder(delta_rows(cell, direction, alpha, log_h), runner.device) for direction, alpha in CONDITIONS]
    scored = runner.score_many(prompts, "pre-label", LAYER, edits)
    out = {"rows": [{"sample_uid": r["sample_uid"], "base": float(scored[0, j]),
                     **{f"{d} {a:+d}": float(scored[1 + c, j]) for c, (d, a) in enumerate(CONDITIONS)}}
                    for j, r in enumerate(records)]}
    if patch:
        out["patch"] = {}
        for type_id in dict.fromkeys(r["type_id"] for r in records):
            idx = [j for j, r in enumerate(records) if r["type_id"] == type_id and r["variant"] != "twin"]
            variant = prompt_set([ids[j] for j in idx], [labels[j] for j in idx],
                                 [near_first[j] for j in idx], batch_size)["variants"][0]
            out["patch"][type_id] = run_answer(runner, variant, [records[j]["regime"] for j in idx],
                                               [records[j]["near_label"] for j in idx], PATCH_PAIRS)
            stamp(f"{mode} patch {type_id}: transfer at L{LAYER} {out['patch'][type_id]['transfer'][LAYER][0]:.2f} "
                  f"({out['patch'][type_id]['n_used']} pairs)")
    return out


def main(args):
    meta, cells, sequences = load(args.inputs)
    if args.limit:
        sequences = limited(meta, sequences, args.limit)
    cfg = Settings(model=meta["model"], thinking="off", prefill="I choose:", bank="full")
    tokenizer, model, device = load_model(cfg)
    runner = Runner(model, device, tokenizer.pad_token_id, args.batch_size, args.long_batch_size)
    out_dir = out_base(args.out) / Path(args.inputs).stem.replace("inputs_", "exp19_", 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"inputs": str(args.inputs), "code_hash": code_hash(), "layer": LAYER,
               "conditions": [f"{d} {a:+d}" for d, a in CONDITIONS], "modes": {}, "phases": {}}
    for mode in args.modes.split(","):
        ids, labels, rows = sequences[mode]
        t0 = time.time()
        stamp(f"thinking {mode}: {len(ids)} rows ...")
        batch = args.batch_size if mode == "off" else args.long_batch_size
        results["modes"][mode] = run_mode(runner, mode, cells[mode, LAYER], meta, ids, labels, rows, batch,
                                          patch=not args.no_patch)
        results["phases"][mode] = round(time.time() - t0, 1)
        stamp(f"thinking {mode} done, {results['phases'][mode]} s")
        (out_dir / "results.json").write_text(json.dumps(results))   # incremental
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--inputs", required=True)
    p.add_argument("--modes", default="off,on")
    p.add_argument("--limit", type=int, default=None, help="rows per type and variant (smoke)")
    p.add_argument("--no-patch", action="store_true")
    p.add_argument("--batch-size", type=int, default=32, help="batch for thinking-off prompts")
    p.add_argument("--long-batch-size", type=int, default=8, help="batch for cached reasoning")
    p.add_argument("--out", default=None)
    main(p.parse_args())
