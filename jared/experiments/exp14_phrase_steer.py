"""E14 step 3: steer the horizon phrase toward a per-prompt target horizon, then let the
model reason (21_spec_phrase_steering.md §3). Qwen3-8B, thinking on.

    python experiments/exp14_phrase_steer.py --dir /workspace/results/horizon/exp14_Qwen3-8B_phrase
    python experiments/exp14_phrase_steer.py --dir out/exp14_Qwen3.5-0.8B_phrase --limit 2 \\
        --batch-size 2 --score-batch-size 2 --max-think-tokens 32      # local smoke (plumbing only)

Reads <dir>/inputs.npz and inputs.json (steering_prepare.py --phrase). A unit of work is
one (transition, condition) over that transition's prompts, run in this order within a
transition: noop, effects L16, random L16, donor L16, invariant L16, opposite L16 (none and
both starts only), effects L14, effects L18, random L14, random L18.

A steering condition adds the prompt's step at the phrase tokens of the condition's layer on
every prefill pass (E10's prompt patch): the number step on each number token, the unit step
on the last token. The donor condition replaces the phrase states at L16 with those of the
same prompt rendered with the donor horizon (E10's full-phrase patch); prompts without a
donor are left out of that unit. Noop is the unedited rerun, the batch-noise baseline.

Writes <dir>/conditions/<transition>__<condition>.json and .npz. As in E10, each unit is
saved when it finishes and skipped on a rerun, and several copies (one per GPU) share the
queue by claiming units with a .lock folder.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp8_steering import selected_records  # noqa: E402  (same folder)
from exp10_patch_reasoning import generate, generation_arrays, names_horizon, run_claimed, score, states_at  # noqa: E402
from exp13_suffix_steer import names_duration  # noqa: E402
from exp14_phrase_states import record_phrase  # noqa: E402
from spar_horizon import steering as S  # noqa: E402
from spar_horizon.config import Settings  # noqa: E402
from spar_horizon.full_bank import render_variant  # noqa: E402
from spar_horizon.model_io import load_model  # noqa: E402
from spar_horizon.runs import code_hash, stamp  # noqa: E402

# (kind, layer) in run order; L16 first so that a run cut short still holds the main test
UNITS = (("noop", None), ("effects", 16), ("random", 16), ("donor", 16), ("invariant", 16), ("opposite", 16),
         ("effects", 14), ("effects", 18), ("random", 14), ("random", 18))
DONOR_LAYER = 16


def condition_name(kind, layer):
    return kind if layer is None else f"{kind} L{layer}"


def unit_name(transition, kind, layer):
    return f"{transition}__{condition_name(kind, layer)}"


def planned_units(transition, start_regime):
    """[(name, kind, layer)] for one transition; the opposite step needs a start inside
    the none or both regime (a near start has no step in the other direction)."""
    return [(unit_name(transition, kind, layer), kind, layer) for kind, layer in UNITS
            if kind != "opposite" or start_regime in ("none", "both")]


def step_deltas(steps, layer, kind, rows, n_positions):
    """Per row, the [n_positions, d] delta of a steering condition: the number step on
    every number token, the unit step on the last. `rows` index the inputs' drawn prompts."""
    number = torch.tensor(steps[f"step|L{layer}|{kind}|number"][rows], dtype=torch.float32)
    unit = torch.tensor(steps[f"step|L{layer}|{kind}|unit"][rows], dtype=torch.float32)
    return [torch.cat([number[j].expand(n - 1, -1), unit[j:j + 1]]) for j, n in enumerate(n_positions)]


def donor_states(ctx, rows, positions):
    """The donor prompts' L16 states at their phrase positions, which must be the
    recipient's (the prompts are identical up to the phrase, and its token count matches)."""
    model, device, tokenizer, cfg = ctx["model"], ctx["device"], ctx["tokenizer"], ctx["cfg"]
    donor_ids, donor_positions = [], []
    for i, recipient_positions in zip(rows, positions):
        record = ctx["records"][i]
        donor = {**record, "horizon_text": ctx["prompts"][i]["donor_horizon_text"]}
        donor["prompt"] = render_variant(donor)
        donor_ids.append(S.encode(tokenizer, donor["prompt"], cfg))
        donor_positions.append(record_phrase(tokenizer, cfg, donor))
        if donor_positions[-1] != recipient_positions:
            raise AssertionError(f"{record['sample_uid']}: donor phrase at {donor_positions[-1]}, "
                                 f"recipient's at {recipient_positions}")
    return states_at(model, device, tokenizer, donor_ids, donor_positions, [DONOR_LAYER],
                     ctx["score_batch_size"])[DONOR_LAYER]


def run_unit(transition, kind, layer, ctx):
    """Generate and score one (transition, condition); returns the per-prompt record and the ids."""
    model, device, tokenizer, cfg, prompts = ctx["model"], ctx["device"], ctx["tokenizer"], ctx["cfg"], ctx["prompts"]
    rows = runnable_rows(transition, kind, ctx)
    records = [ctx["records"][i] for i in rows]
    positions = [ctx["phrase"][i] for i in rows]
    patch = None
    if kind == "donor":
        values = donor_states(ctx, rows, positions)

        def patch(idx):
            return S.prompt_patch(model, layer, [positions[i] for i in idx], [values[i] for i in idx])
    elif kind != "noop":
        deltas = step_deltas(ctx["steps"], layer, kind, rows, [len(p) for p in positions])

        def patch(idx):
            return S.prompt_patch(model, layer, [positions[i] for i in idx], [deltas[i] for i in idx], add=True)

    ids = [ctx["ids"][i] for i in rows]
    generations = generate(model, device, tokenizer, cfg, ids, ctx["batch_size"], patch)
    near_first = [r["near_label"] == "a" for r in records]
    near_lo, labels = score(model, device, tokenizer, cfg, generations, near_first, ctx["score_batch_size"], patch)
    out_rows = []
    for i, r, g, lo, label in zip(rows, records, generations, near_lo, labels):
        p, think = prompts[i], g.think_text(tokenizer)
        delays = (r["near_delay_years"], r["far_delay_years"])
        out_rows.append({
            "sample_uid": r["sample_uid"], "bank": ctx["banks"][i], "transition": transition,
            "condition": condition_name(kind, layer), "layer": layer,
            "start_regime": p["start_regime"], "target_regime": p["target_regime"],
            "horizon_text": r["horizon_text"], "log_h": p["log_h"], "target_log_h": p["target_log_h"],
            "near_label": r["near_label"], "near_delay_years": delays[0], "far_delay_years": delays[1],
            "near_lo": float(lo), "label": label, "forced": g.forced, "think_len": g.think_len,
            "think_text": think, "names_own": names_horizon(think, r["horizon_text"]),
            "names_target": names_duration(think, 10 ** p["target_log_h"], delays)})
    result = {"name": unit_name(transition, kind, layer), "transition": transition,
              "condition": condition_name(kind, layer), "layer": layer,
              "n_without_donor": len(ctx["transition_rows"][transition]) - len(rows) if kind == "donor" else 0,
              "rows": out_rows}
    return result, generation_arrays(generations)


def runnable_rows(transition, kind, ctx):
    """The transition's prompts a unit runs on: all, but for the donor patch only those
    with a donor."""
    rows = ctx["transition_rows"][transition]
    if kind == "donor":
        rows = [i for i in rows if ctx["prompts"][i]["donor_horizon_text"] is not None]
    return rows


def report(out_dir, name):
    """One line for a finished unit: mean near log-odds, names-target and forced rates."""
    result = json.loads((out_dir / "conditions" / f"{name.replace(' ', '_')}.json").read_text())
    rows = result["rows"]
    print(f"  {name:28} n {len(rows):3}  near log-odds {np.mean([r['near_lo'] for r in rows]):+.2f}  "
          f"names target {np.mean([r['names_target'] for r in rows]):.2f}  "
          f"forced {np.mean([r['forced'] for r in rows]):.2f}", flush=True)


def transition_rows(prompts, transitions, limit):
    """{transition: indices of its drawn prompts, in order (the first `limit`)}; transitions
    with no prompts are left out."""
    rows = {t: [i for i, p in enumerate(prompts) if p["transition"] == t][:limit] for t in transitions}
    return {t: r for t, r in rows.items() if r}


def main(args):
    out_dir = Path(args.dir)
    meta = json.loads((out_dir / "inputs.json").read_text())
    steps = np.load(out_dir / "inputs.npz")
    by_transition = transition_rows(meta["prompts"], meta["transitions"], args.limit)
    generation = {**meta["generation"],
                  "max_think_tokens": args.max_think_tokens or meta["generation"]["max_think_tokens"]}
    cfg = Settings(**generation, batch_size=args.batch_size)
    all_records = selected_records(meta)
    kept = sorted(i for rows in by_transition.values() for i in rows)
    tokenizer, model, device = load_model(cfg)
    ctx = {"model": model, "device": device, "tokenizer": tokenizer, "cfg": cfg, "steps": steps,
           "prompts": meta["prompts"], "banks": meta["selected_banks"], "transition_rows": by_transition,
           "records": {i: all_records[i] for i in kept},
           "ids": {i: S.encode(tokenizer, all_records[i]["prompt"], cfg) for i in kept},
           "phrase": {i: record_phrase(tokenizer, cfg, all_records[i]) for i in kept},
           "batch_size": args.batch_size, "score_batch_size": args.score_batch_size}
    (out_dir / "conditions").mkdir(exist_ok=True)
    stamp(f"{len(kept)} prompts in {len(by_transition)} transitions, think budget {cfg.max_think_tokens}")
    (out_dir / "steer_manifest.json").write_text(json.dumps({
        "settings": cfg.to_dict(), "prompts": len(kept), "code_hash": code_hash()}, indent=1))
    only = {name.replace(" ", "_") for name in args.only.split(";")} if args.only else None
    for transition, rows in by_transition.items():
        for name, kind, layer in planned_units(transition, meta["prompts"][rows[0]]["start_regime"]):
            if only is not None and name.replace(" ", "_") not in only:
                continue
            if not runnable_rows(transition, kind, ctx):
                stamp(f"{name}: no prompts to run; skipped")
                continue
            if run_claimed(out_dir, name, lambda: run_unit(transition, kind, layer, ctx)):
                report(out_dir, name)
    stamp(f"wrote {out_dir}")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", required=True, help="<out>/exp14_<model>_phrase, with inputs.npz and inputs.json")
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts per transition (smoke)")
    parser.add_argument("--max-think-tokens", type=int, default=None,
                        help="think budget (default: the inputs' generation settings)")
    parser.add_argument("--batch-size", type=int, default=12, help="generation batch (E0's on the 4090)")
    parser.add_argument("--score-batch-size", type=int, default=8, help="batch for the scoring passes")
    parser.add_argument("--only", default=None, help='";"-separated unit names to run, e.g. "both-near__effects L16"')
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
