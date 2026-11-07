"""Where does the horizon phrase carry its number? Patch parts of the phrase of same-unit crossing
pairs and let Qwen3-8B (thinking on) reason (27_spec_phrase_number_patch.md §2). A SCREEN on 24
pairs; nothing here is a significance claim.

    python experiments/exp16_phrase_patch.py --dir /workspace/results/horizon/exp16_Qwen3-8B_patch
    python experiments/exp16_phrase_patch.py --dir out/exp16_Qwen3.5-0.8B_patch --limit 2 --batch-size 2 \\
        --score-batch-size 2 --max-think-tokens 32                      # local smoke (plumbing only)

Reads <dir>/inputs.npz and inputs.json (analysis/patch_prepare.py). A unit of work is saved as
conditions/<name>.json and .npz when it finishes, skipped on a rerun, and claimed through a .lock
folder, as in E10, E14 and E15. Units, in this order (patch_summary.planned_units):

  baseline, rerun       the unpatched recipient prompts in the inputs' order, then again in another
                        order (the noise floor: batch composition alone changes some labels)
  self                  the recipient's own phrase states put back (plumbing check)
  donor phrase          the donor-rendered prompt's states at every phrase token
  donor number          ... at the number tokens only
  manifold number       the pair's manifold step added to each number token
  manifold phrase       the number step on the number tokens and the unit step on the unit token
  random number         a norm-matched Gaussian step on the number tokens
  donor unit            the donor's state at the unit token only
  subspace number       only the top-8 horizon-subspace component of the donor's number states
  inspan number         a norm-matched random step inside that subspace, on the number tokens
  donor number L12/L22  the number-token donor patch at the other depths

All at the patch layer (16) unless named. A patch replaces (or, for a step, adds to) the output of
decoder layer L-1 at the prompt's token positions on every prefill pass, so it covers the reasoning
pass and the answer pass, and never a decode step (steering.prompt_patch). Donor states come from a
forward pass of the donor-rendered prompt, whose phrase has the recipient's token count (the pairs
are same-unit, equal-length); the prompts are identical elsewhere.

Rows hold the label, the near log-odds before it, forced close, think length and the reasoning text;
analysis/patch_summary.py counts what the traces state and the flips against the baseline.
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
from exp8_steering import selected_records  # noqa: E402  (same folder)
from exp10_patch_reasoning import generate, generation_arrays, names_horizon, run_claimed, score, states_at  # noqa: E402
from exp14_phrase_states import record_phrase  # noqa: E402
from patch_summary import BASELINE, RERUN, planned_units, unit_measures  # noqa: E402  (analysis/)
from scan_summary import load_unit  # noqa: E402
from spar_horizon import steering as S  # noqa: E402
from spar_horizon.config import Settings  # noqa: E402
from spar_horizon.full_bank import render_variant  # noqa: E402
from spar_horizon.model_io import load_model  # noqa: E402
from spar_horizon.runs import code_hash, stamp  # noqa: E402

RERUN_SEED, RERUN_PARTS = 1, 2     # the rerun's prompt order and number of batches


def rerun_batching(n):
    """(prompt order, batch size) of the rerun over `n` pairs."""
    return np.random.default_rng(RERUN_SEED).permutation(n), -(-n // RERUN_PARTS)


def token_positions(phrase, where):
    """The unpadded positions a patch at `where` (phrase, number or unit) touches in one phrase."""
    return {"phrase": phrase, "number": phrase[:-1], "unit": phrase[-1:]}[where]


def phrase_values(states, where):
    """The rows of a phrase's [len(phrase), d] states that `where` patches."""
    return {"phrase": states, "number": states[:-1], "unit": states[-1:]}[where]


def source_states(ctx, source, layer):
    """Per pair, the [len(phrase), d] states at the whole phrase of the recipient (`self`) or of the
    donor-rendered prompt (`donor`, `subspace`), at the output of layer - 1 hidden_states[layer]."""
    ids = ctx["ids"] if source == "self" else ctx["donor_ids"]
    return states_at(ctx["model"], ctx["device"], ctx["tokenizer"], ids, ctx["phrase"], [layer],
                     ctx["score_batch_size"])[layer]


def step_values(ctx, layer, source, where):
    """Per pair, the [n_positions, d] step added at `where`: the number step on each number token
    (manifold, random, inspan) and, for the manifold step on the whole phrase, the unit step on the unit token."""
    steps = ctx["steps"]
    number = torch.tensor(steps[f"step|L{layer}|{source}|number"], dtype=torch.float32)
    unit = torch.tensor(steps[f"step|L{layer}|manifold|unit"], dtype=torch.float32)
    values = []
    for i, phrase in enumerate(ctx["phrase"]):
        row = number[i].expand(len(phrase) - 1, -1)
        values.append(torch.cat([row, unit[i:i + 1]]) if where == "phrase" else row)
    return values


def unit_patch(ctx, source, where, layer):
    """patch(idx) for generate and score: the unit's edit at `where` of `layer`."""
    positions = [token_positions(phrase, where) for phrase in ctx["phrase"]]
    basis, add = None, source in ("manifold", "random", "inspan")
    if add:
        values = step_values(ctx, layer, source, where)
    else:
        values = [phrase_values(s, where) for s in source_states(ctx, source, layer)]
        if source == "subspace":
            basis = torch.tensor(ctx["steps"][f"basis|L{layer}"], dtype=torch.float32)

    def patch(idx):
        return S.prompt_patch(ctx["model"], layer, [positions[i] for i in idx], [values[i] for i in idx], basis, add)

    return patch


def base_row(ctx, i):
    p, r = ctx["prompts"][i], ctx["records"][i]
    return {"sample_uid": r["sample_uid"], "bank": ctx["banks"][i], "transition": p["transition"],
            "start_regime": p["start_regime"], "target_regime": p["target_regime"], "horizon_text": r["horizon_text"],
            "donor_horizon_text": p["donor_horizon_text"], "log_h": p["log_h"], "target_log_h": p["target_log_h"],
            "donor_log_h": p["donor_log_h"], "near_label": r["near_label"],
            "near_delay_years": r["near_delay_years"], "far_delay_years": r["far_delay_years"]}


def run_generated(ctx, order, batch_size, patch=None):
    """Generate and score the pairs in `order` (batched in that order); rows and arrays come back in
    pair order. A patch is indexed by position in `order`, so it needs the identity order."""
    model, device, tokenizer, cfg = ctx["model"], ctx["device"], ctx["tokenizer"], ctx["cfg"]
    generations = generate(model, device, tokenizer, cfg, [ctx["ids"][i] for i in order], batch_size, patch)
    near_first = [ctx["records"][i]["near_label"] == "a" for i in order]
    near_lo, labels = score(model, device, tokenizer, cfg, generations, near_first, ctx["score_batch_size"], patch)
    back = np.argsort(order)
    generations, near_lo, labels = [generations[j] for j in back], near_lo[back], [labels[j] for j in back]
    rows = []
    for i, (g, lo, label) in enumerate(zip(generations, near_lo, labels)):
        think = g.think_text(tokenizer)
        rows.append({**base_row(ctx, i), "near_lo": float(lo), "label": label, "forced": g.forced,
                     "think_len": g.think_len, "think_text": think,
                     "names_own": names_horizon(think, ctx["records"][i]["horizon_text"]),
                     "names_donor": names_horizon(think, ctx["prompts"][i]["donor_horizon_text"])})
    return rows, generation_arrays(generations)


def run_unit(unit, ctx):
    """(result, arrays) of one unit."""
    name, number, source, where, layer = unit
    n = len(ctx["ids"])
    if name == BASELINE:
        rows, arrays = run_generated(ctx, np.arange(n), ctx["batch_size"])
    elif name == RERUN:
        rows, arrays = run_generated(ctx, *rerun_batching(n))
    else:
        rows, arrays = run_generated(ctx, np.arange(n), ctx["batch_size"], unit_patch(ctx, source, where, layer))
    return {"name": name, "condition": number, "source": source, "positions": where, "layer": layer,
            "rows": rows}, arrays


def report(out_dir, name):
    """One line for a finished unit: what its traces state and the flips against the baseline."""
    rows = load_unit(out_dir, name)["rows"]
    line = f"  {name:26} n {len(rows):3}  near log-odds {np.mean([r['near_lo'] for r in rows]):+.2f}"
    baseline = load_unit(out_dir, BASELINE)
    if baseline is not None and name != BASELINE:
        m = unit_measures(rows, baseline["rows"])
        line += (f"  states donor {m['states_donor']}  lands target {m['lands_target']}  flips {m['flips']} "
                 f"({m['flips_toward_near']} to near)  d near log-odds {m['near_lo_shift']:+.2f}")
    print(line, flush=True)


def donor_inputs(tokenizer, cfg, records, prompts):
    """Per pair, the donor-rendered prompt's ids and its phrase positions, which must be the recipient's."""
    ids, phrases = [], []
    for record, p in zip(records, prompts):
        donor = {**record, "horizon_text": p["donor_horizon_text"]}
        donor["prompt"] = render_variant(donor)
        ids.append(S.encode(tokenizer, donor["prompt"], cfg))
        phrases.append(record_phrase(tokenizer, cfg, donor))
    return ids, phrases


def main(args):
    out_dir = Path(args.dir)
    meta = json.loads((out_dir / "inputs.json").read_text())
    steps = np.load(out_dir / "inputs.npz")
    generation = {**meta["generation"],
                  "max_think_tokens": args.max_think_tokens or meta["generation"]["max_think_tokens"]}
    cfg = Settings(**generation, batch_size=args.batch_size)
    records = selected_records(meta)[:args.limit]
    prompts = meta["prompts"][:len(records)]
    tokenizer, model, device = load_model(cfg)
    phrase = [record_phrase(tokenizer, cfg, r) for r in records]
    donor_ids, donor_phrase = donor_inputs(tokenizer, cfg, records, prompts)
    for r, own, donor in zip(records, phrase, donor_phrase):
        if own != donor:
            raise AssertionError(f"{r['sample_uid']}: donor phrase at {donor}, recipient's at {own}")
    ctx = {"model": model, "device": device, "tokenizer": tokenizer, "cfg": cfg, "steps": steps,
           "records": records, "prompts": prompts, "banks": meta["selected_banks"][:len(records)],
           "ids": [S.encode(tokenizer, r["prompt"], cfg) for r in records], "donor_ids": donor_ids, "phrase": phrase,
           "batch_size": args.batch_size, "score_batch_size": args.score_batch_size}
    (out_dir / "conditions").mkdir(exist_ok=True)
    only = {name.replace(" ", "_") for name in args.only.split(";")} if args.only else None
    todo = [u for u in planned_units(meta["patch_layer"], meta["depth_layers"])
            if only is None or u[0].replace(" ", "_") in only]
    stamp(f"{len(records)} pairs, {len(todo)} units planned, think budget {cfg.max_think_tokens}")
    (out_dir / "patch_manifest.json").write_text(json.dumps({
        "settings": cfg.to_dict(), "pairs": len(records), "code_hash": code_hash()}, indent=1))
    for unit in tqdm(todo, desc="units", unit="unit", dynamic_ncols=True):
        if run_claimed(out_dir, unit[0], lambda: run_unit(unit, ctx)):
            report(out_dir, unit[0])
    stamp(f"wrote {out_dir}")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", required=True, help="<out>/exp16_<model>_patch, with inputs.npz and inputs.json")
    parser.add_argument("--limit", type=int, default=None, help="only the first N pairs (smoke)")
    parser.add_argument("--max-think-tokens", type=int, default=None,
                        help="think budget (default: the inputs' generation settings)")
    parser.add_argument("--batch-size", type=int, default=12, help="generation batch (E0's on the 4090)")
    parser.add_argument("--score-batch-size", type=int, default=8, help="batch for the scoring passes")
    parser.add_argument("--only", default=None, help='";"-separated unit names to run, e.g. "baseline;donor number L16"')
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
