"""E20 transfer: does the horizon phrase steer the choice in the eval-type prompts (EvalsSteeringSpec.md)?
Qwen3-8B, thinking on, E14's phrase patch applied to eval-type prompts.

    python -m pipeline.s2_steering.experiments.manifold_steer.exp20_evaltype_transfer \\
        --types city_no_values,city_values,public_health,coding_agent \\
        --evaltypes-dir $RESULTS/evaltypes_Qwen3-8B_think-on --exp14-dir $RESULTS/exp14_Qwen3-8B_phrase
    python -m pipeline.s2_steering.experiments.manifold_steer.exp20_evaltype_transfer --types city_values --limit 2 \\
        --arms donor,noop --model Qwen/Qwen3.5-0.8B --max-think-tokens 32 --evaltypes-dir ... --out out   # local smoke

Each row of a type is steered toward the answer a different horizon regime makes rational: the target
is the log-midpoint across the nearest delay (near -> both, both -> near, none -> near), as in E14.
Arms, run in this order (EvalsSteeringSpec.md §3):
  donor    the phrase states at L16 replaced by those of the same prompt with the donor horizon (the
           grid horizon nearest the target in the target regime, with the phrase's token count)
  noop     the unedited rerun, the bf16 batch-noise baseline
  effects  E14's effects step at L16 (the bank fit, <exp14 dir>/states.npz), from the row's horizon
           to the target: the number step on each number token, the unit step on the last token
  random   a random direction of the same length per row and role
Rows without a donor are left out of every arm. Generation settings are the eval run's settings.json
(E0's, with its answer_window).

Writes <out>/exp20_<model>_evaltype_transfer/<type>/conditions/<arm>.json and .npz and <type>/summary.json.
As in E10, each (type, arm) is saved when it finishes and skipped on a rerun, and several copies
(one per GPU) share the queue by claiming units with a .lock folder.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from pipeline.s1_data_vectors.analysis.evaltypes_steering_prepare import eval_settings
from pipeline.s1_data_vectors.analysis.steering_prepare import (
    OUT, PHRASE_ROLES, phrase_columns, fit_phrase_cell, random_like, transition_target, varying_factors)
from pipeline.s1_data_vectors.experiments.exp14_phrase_states import record_phrase
from pipeline.s2_steering.experiments.manifold_steer.exp13_suffix_steer import names_duration
from pipeline.s2_steering.experiments.manifold_steer.exp14_phrase_steer import DONOR_LAYER, step_deltas
from pipeline.s2_steering.experiments.patch.exp10_patch_reasoning import (
    generate, generation_arrays, names_horizon, run_claimed, score, states_at)
from spar_horizon import steering as S
from spar_horizon.banks import selected_records
from spar_horizon.eval_types.records import eval_records, find_donor
from spar_horizon.model_io import load_model
from spar_horizon.runs import code_hash, out_base, stamp

ARMS = ("donor", "noop", "effects", "random")   # run order: the main measurement first
LAYER = DONOR_LAYER
TARGET_REGIME = {"near": "both", "both": "near", "none": "near"}
SEED = 0


def select_rows(records, rows_per_type, seed=SEED):
    """All rows, or `rows_per_type` // 2 near and as many both rows, drawn with a fixed seed; all of a
    regime's rows when it has fewer. Called on the rows that have a donor, so the draw is not thinned later."""
    if rows_per_type is None:
        return records
    rng = np.random.default_rng(seed)
    chosen = []
    for regime in ("near", "both"):
        pool = [r for r in records if r["regime"] == regime]
        n = min(rows_per_type // 2, len(pool))
        chosen += [pool[i] for i in sorted(rng.choice(len(pool), n, replace=False))]
    return chosen


def plan_rows(records, tokenizer, cfg):
    """(rows with their target and donor, number of rows without a donor)."""
    rows = []
    for r in records:
        target_regime = TARGET_REGIME[r["regime"]]
        log_near, log_far = np.log10(r["near_delay_years"]), np.log10(r["far_delay_years"])
        target_log_h = float(transition_target(target_regime, log_near, log_far))
        donor = find_donor(r, target_log_h, target_regime, tokenizer, cfg)
        if donor is None:
            continue
        far_label = next(label for label in r["labels"] if label != r["near_label"])
        rows.append({**r, "transition": f"{r['regime']}-{target_regime}", "start_regime": r["regime"],
                     "target_regime": target_regime, "target_label": r["near_label"] if target_regime == "near" else far_label,
                     "log_h": float(np.log10(r["horizon"])), "target_log_h": target_log_h,
                     "donor_horizon_text": donor[0], "donor_prompt": donor[2]})
    return rows, len(records) - len(rows)


def bank_cells(exp14_dir, prompts_file=None):
    """{role: the bank's pooled fit at L16}, refitted from the E14 phrase states as steering_prepare
    --phrase did: inputs.npz keeps only the steps of the bank's own prompts, not the curve."""
    exp14_dir = Path(exp14_dir)
    states_meta = json.loads((exp14_dir / "states.json").read_text())
    states = np.load(exp14_dir / "states.npz")
    prompts_path = Path(prompts_file or states_meta["prompts_file"])
    if not prompts_path.exists():
        prompts_path = OUT / prompts_path.name
    records = selected_records(json.loads(prompts_path.read_text()))[:states_meta["n"]]
    if [r["sample_uid"] for r in records] != [str(u) for u in states["sample_uids"]]:
        raise ValueError("the states are not those of the prompts file's first rows")
    columns = phrase_columns(records)
    factors = varying_factors(columns)
    return {role: fit_phrase_cell(states[f"L{LAYER}|{role}"], columns, factors) for role in PHRASE_ROLES}


def bank_steps(cells, rows):
    """E14's input arrays for `rows`: the effects step from each row's horizon to its target on the
    bank's curve, and a random step as long as it."""
    log_h = np.array([r["log_h"] for r in rows])
    target_log_h = np.array([r["target_log_h"] for r in rows])
    steps = {}
    for role_index, role in enumerate(PHRASE_ROLES):
        effects = S.curve_delta(cells[role], log_h, target_log_h - log_h)
        steps[f"step|L{LAYER}|effects|{role}"] = effects
        steps[f"step|L{LAYER}|random|{role}"] = random_like(effects, np.random.default_rng([SEED, LAYER, role_index]))
    return steps


def donor_states(ctx):
    """The donor prompts' L16 states at their phrase positions, which must be the row's own
    (the prompts are identical up to the phrase, and its token count matches)."""
    tokenizer, cfg = ctx["tokenizer"], ctx["cfg"]
    donor_ids, donor_positions = [], []
    for r, row_positions in zip(ctx["rows"], ctx["positions"]):
        donor = {**r, "prompt": r["donor_prompt"], "horizon_text": r["donor_horizon_text"]}
        donor_ids.append(S.encode(tokenizer, donor["prompt"], cfg))
        donor_positions.append(record_phrase(tokenizer, cfg, donor))
        if donor_positions[-1] != row_positions:
            raise AssertionError(f"{r['sample_uid']}: donor phrase at {donor_positions[-1]}, row's at {row_positions}")
    return states_at(ctx["model"], ctx["device"], tokenizer, donor_ids, donor_positions, [LAYER],
                     ctx["score_batch_size"])[LAYER]


def run_arm(arm, ctx):
    """Generate and score one arm over the type's rows; returns the record and the ids."""
    model, device, tokenizer, cfg, rows, positions = (ctx[k] for k in ("model", "device", "tokenizer", "cfg", "rows", "positions"))
    patch = None
    if arm == "donor":
        values = donor_states(ctx)

        def patch(idx):
            return S.prompt_patch(model, LAYER, [positions[i] for i in idx], [values[i] for i in idx])
    elif arm != "noop":
        deltas = step_deltas(bank_steps(ctx["cells"], rows), LAYER, arm, np.arange(len(rows)), [len(p) for p in positions])

        def patch(idx):
            return S.prompt_patch(model, LAYER, [positions[i] for i in idx], [deltas[i] for i in idx], add=True)

    generations = generate(model, device, tokenizer, cfg, ctx["ids"], ctx["batch_size"], patch)
    near_first = [r["near_label"] == "a" for r in rows]
    near_lo, labels = score(model, device, tokenizer, cfg, generations, near_first, ctx["score_batch_size"], patch)
    out_rows = []
    for r, g, lo, label in zip(rows, generations, near_lo, labels):
        think = g.think_text(tokenizer)
        delays = (r["near_delay_years"], r["far_delay_years"])
        out_rows.append({
            "sample_uid": r["sample_uid"], "type_id": r["type_id"], "arm": arm, "transition": r["transition"],
            "start_regime": r["start_regime"], "target_regime": r["target_regime"], "horizon_text": r["horizon_text"],
            "donor_horizon_text": r["donor_horizon_text"], "log_h": r["log_h"], "target_log_h": r["target_log_h"],
            "near_label": r["near_label"], "target_label": r["target_label"],
            "near_lo": float(lo), "label": label, "forced": g.forced, "think_len": g.think_len, "think_text": think,
            "names_own": names_horizon(think, r["horizon_text"]),
            "names_target": names_duration(think, 10 ** r["target_log_h"], delays),
            "e0_near_lo": r["near_lo"], "e0_label": r["label"]})
    result = {"name": arm, "arm": arm, "type_id": ctx["type_id"], "n_skipped_no_donor": ctx["n_skipped"], "rows": out_rows}
    return result, generation_arrays(generations)


def flipped_toward_target(rows):
    """Share of the rows whose E0 label was not the target regime's answer that now are; NaN if none."""
    flippable = [r for r in rows if r["e0_label"] is not None and r["e0_label"] != r["target_label"]]
    if not flippable:
        return float("nan")
    return float(np.mean([r["label"] == r["target_label"] for r in flippable]))


def summarize(results):
    """Per arm: the flip rate toward the target regime (E0's label to the arm's), the near log-odds
    shift against noop signed toward the target (+ for a near target, - for a far one) with a
    bootstrap CI, the forced-close and names-target rates, and the counts."""
    noop = {r["sample_uid"]: r for r in results["noop"]["rows"]} if "noop" in results else {}
    noop_flipped = flipped_toward_target(list(noop.values()))
    out = {}
    for arm, result in results.items():
        rows = result["rows"]
        entry = {"n": len(rows), "n_skipped_no_donor": result["n_skipped_no_donor"],
                 "flipped_toward": flipped_toward_target(rows),
                 "forced": float(np.mean([r["forced"] for r in rows])),
                 "names_target": float(np.mean([r["names_target"] for r in rows]))}
        if arm != "noop" and noop:
            sign = np.array([1.0 if r["target_regime"] == "near" else -1.0 for r in rows])
            shift = np.array([r["near_lo"] - noop[r["sample_uid"]]["near_lo"] for r in rows])
            entry["toward_log_odds_vs_noop"] = S.bootstrap_mean(sign * shift)
            entry["flipped_toward_minus_noop"] = entry["flipped_toward"] - noop_flipped
        out[arm] = entry
    return out


def report(type_dir, arm=None):
    """<type>/summary.json over the arms finished so far; prints one arm's line, or every arm's."""
    results = {}
    for file in sorted((type_dir / "conditions").glob("*.json")):
        result = json.loads(file.read_text())
        results[result["arm"]] = result
    summary = summarize(results)
    (type_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    for name in ([arm] if arm else [a for a in ARMS if a in summary]):
        s = summary[name]
        shift = s.get("toward_log_odds_vs_noop")
        print(f"  {type_dir.name:20} {name:8} n {s['n']:3} (no donor {s['n_skipped_no_donor']})  "
              f"flipped toward target {s['flipped_toward']:.2f}  "
              + (f"log-odds toward target vs noop {shift[0]:+.2f} [{shift[1]:+.2f}, {shift[2]:+.2f}]  " if shift else "")
              + f"forced {s['forced']:.2f}", flush=True)


def main(args):
    run_dir = Path(args.evaltypes_dir)
    cfg = eval_settings(run_dir)
    cfg = dataclasses.replace(cfg, model=args.model or cfg.model, batch_size=args.batch_size,
                              max_think_tokens=args.max_think_tokens or cfg.max_think_tokens)
    unknown = set(args.arms.split(",")) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arms {sorted(unknown)}; choose from {ARMS}")
    arms = [a for a in ARMS if a in args.arms.split(",")]
    needs_bank_fit = {"effects", "random"} & set(arms)
    if needs_bank_fit and not args.exp14_dir:
        raise ValueError("the effects and random arms need --exp14-dir")
    cells = bank_cells(args.exp14_dir, args.bank_prompts) if needs_bank_fit else None
    out_dir = out_base(args.out) / f"exp20_{cfg.model.split('/')[-1]}_evaltype_transfer"
    tokenizer, model, device = load_model(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({
        "settings": cfg.to_dict(), "args": vars(args), "code_hash": code_hash()}, indent=1))
    for type_id in args.types.split(","):
        records = eval_records(run_dir / type_id / "rows.json")
        if {r["choice_prefix"] for r in records} != {cfg.choice_prefix}:
            raise ValueError(f"{type_id}: choice prefixes differ from the run's {cfg.choice_prefix!r}")
        rows, n_skipped = plan_rows(records, tokenizer, cfg)
        rows = select_rows(rows, args.rows_per_type)[:args.limit]
        regimes = {g: sum(r["regime"] == g for r in rows) for g in ("near", "both", "none")}
        stamp(f"{type_id}: {len(rows)} rows {regimes}, {n_skipped} of {len(records)} without a donor, "
              f"think budget {cfg.max_think_tokens}")
        type_dir = out_dir / type_id
        (type_dir / "conditions").mkdir(parents=True, exist_ok=True)
        ctx = {"model": model, "device": device, "tokenizer": tokenizer, "cfg": cfg, "cells": cells,
               "type_id": type_id, "rows": rows, "n_skipped": n_skipped,
               "ids": [S.encode(tokenizer, r["prompt"], cfg) for r in rows],
               "positions": [record_phrase(tokenizer, cfg, r) for r in rows],
               "batch_size": args.batch_size, "score_batch_size": args.score_batch_size}
        for arm in arms:
            if run_claimed(type_dir, arm, lambda: run_arm(arm, ctx)):
                report(type_dir, arm)
        report(type_dir)
    stamp(f"wrote {out_dir}")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--types", required=True, help="comma-separated eval types, e.g. city_values,coding_agent")
    parser.add_argument("--arms", default=",".join(ARMS), help="subset of donor,noop,effects,random (run in that order)")
    parser.add_argument("--evaltypes-dir", required=True, help="the eval-type run: <type>/rows.json and settings.json")
    parser.add_argument("--exp14-dir", default=None, help="E14 dir with states.json/states.npz; needed for effects and random")
    parser.add_argument("--bank-prompts", default=None,
                        help="E14's prompts file when states.json's path does not exist here (phrase_prompts_<model>.json)")
    parser.add_argument("--rows-per-type", type=int, default=None,
                        help="N/2 near and N/2 both rows per type among those with a donor, fixed seed (default: all rows)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows with a donor per type (smoke)")
    parser.add_argument("--model", default=None, help="model to run (default: the eval run's; a smaller one for smokes)")
    parser.add_argument("--max-think-tokens", type=int, default=None, help="think budget (default: the eval run's)")
    parser.add_argument("--batch-size", type=int, default=12, help="generation batch (E0's on the 4090)")
    parser.add_argument("--score-batch-size", type=int, default=8, help="batch for the scoring passes")
    parser.add_argument("--out", default=None, help="output base (default: out/ or $WORKSPACE/results/horizon)")
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
