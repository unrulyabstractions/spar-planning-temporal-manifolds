"""Part B of the overnight scan: steer Qwen3-8B (thinking on) at one (layer, position, direction)
cell at a time and see whether the choice moves (26_spec_overnight_choice_scan.md §2). A SCREEN
on 10 prompts per cell, the same 10 in every cell; nothing here is a significance claim.

    python experiments/exp15_scan.py --dir /workspace/results/horizon/exp15_Qwen3-8B_scan
    python experiments/exp15_scan.py --dir out/exp15_Qwen3.5-0.8B_scan --limit 2 --batch-size 2 \\
        --score-batch-size 2 --max-think-tokens 32                      # local smoke (plumbing only)

Reads <dir>/inputs.npz and inputs.json (analysis/scan_prepare.py). A unit of work is saved as
conditions/<name>.json and .npz when it finishes, skipped on a rerun, and claimed through a .lock
folder, so several copies (one per GPU) share the queue, as in E10 and E14. Units, in this order:

  baseline, noop 1, noop 2     the unsteered prompts, then two reruns whose batches differ from
                               the baseline's (another order and a smaller batch: E10 found 12/96
                               labels changing from batch composition alone), the noise floor
  L{layer} {position} {direction}
                               for the layers of the inputs (18, 14, 22, 10, 26), the positions
                               number, unit, suffix, answer, and the directions manifold, random:
                               a manifold cell and its random twin run back to back

Positions and how each cell steers:
  number, unit   the step is added to the horizon phrase's number tokens, or to its unit token, on
                 every prefill pass (E10's prompt patch, as E14); the model then regenerates its
                 reasoning and label
  suffix         the same prefill edit at the suffix newline after <|im_end|> (as E13 stage B)
  answer         NO regeneration: the baseline's saved reasoning is scored again with the step
                 added at the pre-label position (the token before the label) at that layer, as E8's
                 answer-slot edits. The baseline unit's saved generations are the cached reasoning,
                 so these cells wait for the baseline unit. The row's reference is an unedited score
                 of the same fixed reasoning on the same pass, and its label is the option with the
                 higher logit, so an answer-slot flip is a different kind of event from a
                 regenerating cell's flip.

Rows hold the label, the near log-odds before it, forced close and think length (and the reasoning
text for regenerated cells); analysis/scan_summary.py counts flips against the baseline.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp8_steering import selected_records, suffix_newlines  # noqa: E402  (same folder)
from exp10_patch_reasoning import generate, generation_arrays, run_claimed, score  # noqa: E402
from exp14_phrase_states import record_phrase  # noqa: E402
from scan_summary import BASELINE, NOOPS, cell_name, flip_measures, load_unit, unit_file  # noqa: E402  (analysis/)
from spar_horizon import steering as S  # noqa: E402
from spar_horizon.config import Settings  # noqa: E402
from spar_horizon.model_io import label_ids, load_model, suffix_length  # noqa: E402
from spar_horizon.runs import code_hash, stamp  # noqa: E402

# (shuffle seed of the prompt order, number of batches the prompts are split into); the baseline
# and every cell use the inputs' order and --batch-size
NOOP_BATCHES = {"noop 1": (1, 2), "noop 2": (2, 3)}


def planned_units(layers, positions, directions):
    """[(name, kind, layer, position, direction)] in run order: baseline, the reruns, then each
    layer's positions with every direction back to back. kind is baseline, noop or cell."""
    units = [(BASELINE, "baseline", None, None, None)] + [(name, "noop", None, None, None) for name in NOOPS]
    units += [(cell_name(layer, position, direction), "cell", layer, position, direction)
              for layer in layers for position in positions for direction in directions]
    return units


def rerun_batching(name, n):
    """(prompt order, batch size) of a no-op rerun over `n` prompts."""
    seed, parts = NOOP_BATCHES[name]
    return np.random.default_rng(seed).permutation(n), -(-n // parts)


def edit_positions(position, ctx, i):
    """Unpadded token positions a prefill edit at `position` touches in prompt i."""
    if position == "number":
        return ctx["phrase"][i][:-1]
    if position == "unit":
        return ctx["phrase"][i][-1:]
    return ctx["newline"][i]


def cell_patch(ctx, layer, position, direction):
    """patch(idx) for generate and score: the prompts' step added at `position` of `layer`."""
    steps = ctx["steps"][f"step|L{layer}|{direction}|{position}"]
    positions = [edit_positions(position, ctx, i) for i in range(len(ctx["ids"]))]
    deltas = [torch.tensor(steps[i], dtype=torch.float32)[None].expand(len(p), -1) for i, p in enumerate(positions)]

    def patch(idx):
        return S.prompt_patch(ctx["model"], layer, [positions[i] for i in idx], [deltas[i] for i in idx], add=True)

    return patch


def base_row(ctx, i):
    p, r = ctx["prompts"][i], ctx["records"][i]
    return {"sample_uid": r["sample_uid"], "bank": ctx["banks"][i], "transition": p["transition"],
            "near_label": r["near_label"], "log_h": p["log_h"], "target_log_h": p["target_log_h"]}


def run_generated(ctx, order, batch_size, patch=None):
    """Generate and score the prompts in `order` (batched in that order); rows and arrays come
    back in prompt order."""
    model, device, tokenizer, cfg = ctx["model"], ctx["device"], ctx["tokenizer"], ctx["cfg"]
    generations = generate(model, device, tokenizer, cfg, [ctx["ids"][i] for i in order], batch_size, patch)
    near_first = [ctx["records"][i]["near_label"] == "a" for i in order]
    near_lo, labels = score(model, device, tokenizer, cfg, generations, near_first, ctx["score_batch_size"], patch)
    back = np.argsort(order)
    generations, near_lo, labels = [generations[j] for j in back], near_lo[back], [labels[j] for j in back]
    rows = [{**base_row(ctx, i), "near_lo": float(lo), "label": label, "forced": g.forced, "think_len": g.think_len,
             "think_text": g.think_text(tokenizer)}
            for i, (g, lo, label) in enumerate(zip(generations, near_lo, labels))]
    return rows, generation_arrays(generations)


def baseline_sequences(out_dir, tokenizer, n):
    """The baseline unit's saved generations as (ids up to the label, the label as written) per prompt."""
    z = np.load(unit_file(out_dir, BASELINE).with_suffix(".npz"))
    if len(z["answer_start"]) != n:
        raise ValueError(f"the baseline unit holds {len(z['answer_start'])} prompts, this run {n}")
    seqs, written = [], []
    for start, stop, answer in zip(z["offsets"][:-1], z["offsets"][1:], z["answer_start"]):
        full = z["ids"][start:stop]
        seqs.append(full[:answer].tolist())
        written.append(tokenizer.decode([int(full[answer])]))
    return seqs, written


def label_from_log_odds(near_lo, near_label):
    """The option with the higher logit: the near label above zero, else the other."""
    other = "b" if near_label == "a" else "a"
    return near_label if near_lo > 0 else other


def run_answer_cell(ctx, layer, direction):
    """Re-score the baseline's reasoning with the step added at the pre-label position; no generation."""
    model, device, tokenizer, cfg = ctx["model"], ctx["device"], ctx["tokenizer"], ctx["cfg"]
    n = len(ctx["ids"])
    seqs, written = baseline_sequences(ctx["out_dir"], tokenizer, n)
    with_space, without_space = label_ids(tokenizer, cfg=cfg), label_ids(tokenizer)
    labels = [with_space if w.startswith(" ") else without_space for w in written]
    near_first = [r["near_label"] == "a" for r in ctx["records"]]
    delta = torch.tensor(ctx["steps"][f"step|L{layer}|{direction}|answer"], dtype=torch.float32, device=device)
    edits = [None, (layer, lambda rows: S.Add(delta[rows]))]
    ref_lo, near_lo = S.score_last_edits(model, device, seqs, near_first, labels, tokenizer.pad_token_id,
                                         ctx["score_batch_size"], edits)
    rows = []
    for i, r in enumerate(ctx["records"]):
        rows.append({**base_row(ctx, i), "near_lo": float(near_lo[i]), "label": label_from_log_odds(near_lo[i], r["near_label"]),
                     "ref_near_lo": float(ref_lo[i]), "ref_label": label_from_log_odds(ref_lo[i], r["near_label"])})
    return rows, {"near_lo": near_lo, "ref_near_lo": ref_lo}


def run_unit(unit, ctx):
    """(result, arrays) of one unit."""
    name, kind, layer, position, direction = unit
    n = len(ctx["ids"])
    if kind == "baseline":
        rows, arrays = run_generated(ctx, np.arange(n), ctx["batch_size"])
    elif kind == "noop":
        rows, arrays = run_generated(ctx, *rerun_batching(name, n))
    elif position == "answer":
        rows, arrays = run_answer_cell(ctx, layer, direction)
    else:
        rows, arrays = run_generated(ctx, np.arange(n), ctx["batch_size"], cell_patch(ctx, layer, position, direction))
    return {"name": name, "kind": kind, "layer": layer, "position": position, "direction": direction,
            "rows": rows}, arrays


def report(out_dir, name):
    """One line for a finished unit: mean near log-odds and flips against the baseline."""
    rows = load_unit(out_dir, name)["rows"]
    line = f"  {name:22} n {len(rows):3}  near log-odds {np.mean([r['near_lo'] for r in rows]):+.2f}"
    baseline = load_unit(out_dir, BASELINE)
    if baseline is not None and name != BASELINE:
        m = flip_measures(rows, baseline["rows"])
        line += f"  flips {m['flips']} ({m['toward']} toward near)  d near log-odds {m['near_lo_shift']:+.2f}"
    print(line, flush=True)


def main(args):
    out_dir = Path(args.dir)
    meta = json.loads((out_dir / "inputs.json").read_text())
    steps = np.load(out_dir / "inputs.npz")
    generation = {**meta["generation"],
                  "max_think_tokens": args.max_think_tokens or meta["generation"]["max_think_tokens"]}
    cfg = Settings(**generation, batch_size=args.batch_size)
    records = selected_records(meta)[:args.limit]
    tokenizer, model, device = load_model(cfg)
    ids = [S.encode(tokenizer, r["prompt"], cfg) for r in records]
    ctx = {"model": model, "device": device, "tokenizer": tokenizer, "cfg": cfg, "steps": steps, "out_dir": out_dir,
           "records": records, "prompts": meta["prompts"], "banks": meta["selected_banks"], "ids": ids,
           "phrase": [record_phrase(tokenizer, cfg, r) for r in records],
           "newline": suffix_newlines(ids, tokenizer, suffix_length(tokenizer, records[0]["prompt"], cfg)),
           "batch_size": args.batch_size, "score_batch_size": args.score_batch_size}
    (out_dir / "conditions").mkdir(exist_ok=True)
    only = {name.replace(" ", "_") for name in args.only.split(";")} if args.only else None
    todo = [u for u in planned_units(meta["layers"], meta["positions"], meta["directions"])
            if only is None or u[0].replace(" ", "_") in only]
    stamp(f"{len(records)} prompts, {len(todo)} units planned, think budget {cfg.max_think_tokens}")
    (out_dir / "scan_manifest.json").write_text(json.dumps({
        "settings": cfg.to_dict(), "prompts": len(records), "code_hash": code_hash()}, indent=1))
    for unit in tqdm(todo, desc="units", unit="unit", dynamic_ncols=True):
        name = unit[0]
        if unit[3] == "answer" and not unit_file(out_dir, BASELINE).exists():
            stamp(f"{name}: the baseline unit is not finished (it caches the reasoning); skipped, rerun later")
            continue
        if run_claimed(out_dir, name, lambda: run_unit(unit, ctx)):
            report(out_dir, name)
    stamp(f"wrote {out_dir}")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", required=True, help="<out>/exp15_<model>_scan, with inputs.npz and inputs.json")
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts (smoke)")
    parser.add_argument("--max-think-tokens", type=int, default=None,
                        help="think budget (default: the inputs' generation settings)")
    parser.add_argument("--batch-size", type=int, default=12, help="generation batch (E0's on the 4090)")
    parser.add_argument("--score-batch-size", type=int, default=8, help="batch for the scoring passes")
    parser.add_argument("--only", default=None, help='";"-separated unit names to run, e.g. "baseline;L18 number manifold"')
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
