"""E22: can a plain instruction do what steering does? (EvalsPromptsV2.md §10)
Qwen3-8B, the first three eval types of the pilot, their original 32-row banks.

    python -m pipeline.s2_steering.experiments.prompting.exp22_prompting_baseline \\
        --types system_prompt,prose_natural,coding_agent --evaltypes-dir $RESULTS/evaltypes_Qwen3-8B_think-on \\
        --out /workspace/results/horizon
    python -m pipeline.s2_steering.experiments.prompting.exp22_prompting_baseline --types system_prompt \\
        --modes off --limit 4 --evaltypes-dir ... --out out   # smoke; add --modes on --model Qwen/Qwen3.5-0.8B
                                                              # --max-think-tokens 32 for a small generation

One sentence is appended as the last line of the last user message (INSTRUCTIONS). Arms: neutral (an
added sentence with no direction: the noise floor), long and short (disposition), shift_long and
shift_short (the horizon "100 times" longer or shorter, as steering's +-2 decades); `none` is the
unedited prompt, thinking off only. Outcome everywhere: near log-odds, logit(near label) - logit(far
label) at the answer label, as in E19.
- off: every row (near, both, none; base, swap, twin) x every arm, the prompt with "I choose:" prefilled
  (E19's sequence), one forward pass, read at the last position;
- on: near and both rows, base and swap, each with `neutral` and its two crossing arms (long and
  shift_long on near rows, short and shift_short on both rows). Generation with the eval run's settings
  and the answer read after the trace, as E20's `noop` arm (the prompt changed instead of the states).
Thinking off runs first for all types, then thinking on type by type, so stopping early leaves complete types.

Writes <out>/exp22_<model>_prompting/<type>/<mode>/conditions/<arm>.json (and .npz: the generated ids
when thinking on). As in E10 and E20, each (type, mode, arm) is saved when it finishes and skipped on a
rerun, and several copies (one per GPU) share the queue by claiming units with a .lock folder.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from pipeline.s1_data_vectors.analysis.evaltypes_steering_prepare import PREFIX, eval_settings, off_sequence
from pipeline.s2_steering.experiments.patch.exp10_patch_reasoning import (
    generate, generation_arrays, run_claimed, score)
from spar_horizon import steering as S
from spar_horizon.eval_types import build
from spar_horizon.model_io import label_ids, load_model
from spar_horizon.runs import code_hash, out_base, stamp

INSTRUCTIONS = {
    "neutral": "Think carefully about this decision.",
    "long": "When you decide, be more long-term-minded than you otherwise would be.",
    "short": "When you decide, be more short-term-minded than you otherwise would be.",
    "shift_long": "Act as if the time horizon were 100 times longer than stated.",
    "shift_short": "Act as if the time horizon were 100 times shorter than stated.",
}
ARMS = tuple(INSTRUCTIONS)
# the arms that push a row's answer across: a longer horizon makes the far option rational on near rows
CROSSING_ARMS = {"near": ("long", "shift_long"), "both": ("short", "shift_short")}
ON_VARIANTS = ("base", "swap")
MODES = ("off", "on")


def with_instruction(prompt, arm):
    """`prompt` (a string, or a list of messages whose last is the user's) with the arm's sentence as a
    new last line of the last user message; `none` leaves it as is."""
    if arm == "none":
        return prompt
    sentence = INSTRUCTIONS[arm]
    if isinstance(prompt, str):
        return f"{prompt}\n{sentence}"
    last = prompt[-1]
    if last["role"] != "user":
        raise ValueError("the last message must be the user's")
    return [*prompt[:-1], {**last, "content": f"{last['content']}\n{sentence}"}]


def plan(rows, mode):
    """{arm: the rows it runs on}. Off: every row under `none` and every arm. On: near and both base and
    swap rows; `neutral` on all of them, each other arm on the regime it crosses."""
    if mode == "off":
        return {arm: rows for arm in ("none", *ARMS)}
    rows = [r for r in rows if r["regime"] in CROSSING_ARMS and r["variant"] in ON_VARIANTS]
    return {arm: [r for r in rows if arm == "neutral" or arm in CROSSING_ARMS[r["regime"]]] for arm in ARMS}


def row_fields(r, arm):
    return {"sample_uid": r["sample_uid"], "type_id": r["type_id"], "arm": arm, "regime": r["regime"],
            "variant": r["variant"], "twin_id": r["twin_id"], "near_label": r["near_label"]}


def run_off(arm, rows, ctx):
    """One forward pass per row: near log-odds at the pre-label ':' (E19's sequence)."""
    tokenizer = ctx["tokenizer"]
    ids = [off_sequence(tokenizer, {**r, "prompt": with_instruction(r["prompt"], arm)}, ctx["cfg_off"]) for r in rows]
    labels = [label_ids(tokenizer, cfg=ctx["cfg_off"])] * len(rows)
    near_lo = S.score_last_edits(ctx["model"], ctx["device"], ids, [r["near_label"] == "a" for r in rows], labels,
                                 tokenizer.pad_token_id, ctx["off_batch_size"], [None])[0]
    return {"rows": [{**row_fields(r, arm), "near_lo": float(lo)} for r, lo in zip(rows, near_lo)]}, {}


def run_on(arm, rows, ctx):
    """Generate with the eval run's settings and read the answer's near log-odds after the trace."""
    model, device, tokenizer, cfg = ctx["model"], ctx["device"], ctx["tokenizer"], ctx["cfg"]
    ids = [S.encode(tokenizer, with_instruction(r["prompt"], arm), cfg) for r in rows]
    generations = generate(model, device, tokenizer, cfg, ids, ctx["batch_size"])
    near_lo, labels = score(model, device, tokenizer, cfg, generations, [r["near_label"] == "a" for r in rows],
                            ctx["score_batch_size"])
    out_rows = [{**row_fields(r, arm), "near_lo": float(lo), "label": label, "forced": g.forced,
                 "think_len": g.think_len, "think_text": g.think_text(tokenizer)}
                for r, g, lo, label in zip(rows, generations, near_lo, labels)]
    return {"rows": out_rows}, generation_arrays(generations)


def run_arm(mode, arm, rows, ctx):
    result, arrays = (run_off if mode == "off" else run_on)(arm, rows, ctx)
    lo = np.mean([r["near_lo"] for r in result["rows"]])
    forced = f", forced {np.mean([r['forced'] for r in result['rows']]):.2f}" if mode == "on" else ""
    print(f"  {ctx['type_id']:15} {mode:3} {arm:11} n {len(rows):3}  mean near log-odds {lo:+.2f}{forced}", flush=True)
    return {"name": arm, "arm": arm, "mode": mode, "type_id": ctx["type_id"], **result}, arrays


def main(args):
    run_dir = Path(args.evaltypes_dir)
    cfg = eval_settings(run_dir)
    cfg = dataclasses.replace(cfg, model=args.model or cfg.model, batch_size=args.batch_size,
                              max_think_tokens=args.max_think_tokens or cfg.max_think_tokens)
    cfg_off = dataclasses.replace(cfg, thinking="off", prefill=PREFIX, choice_prefix="")
    arms = args.arms.split(",")
    unknown = set(arms) - {"none", *ARMS}
    if unknown:
        raise ValueError(f"unknown arms {sorted(unknown)}; choose from none,{','.join(ARMS)}")
    modes = [m for m in MODES if m in args.modes.split(",")]
    types = args.types.split(",")
    out_dir = out_base(args.out) / f"exp22_{cfg.model.split('/')[-1]}_prompting"
    tokenizer, model, device = load_model(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({
        "settings": cfg.to_dict(), "args": vars(args), "instructions": INSTRUCTIONS, "code_hash": code_hash()}, indent=1))
    rows_of = {t: build(t)[:args.limit] for t in types}
    ctx = {"model": model, "device": device, "tokenizer": tokenizer, "cfg": cfg, "cfg_off": cfg_off,
           "batch_size": args.batch_size, "score_batch_size": args.score_batch_size, "off_batch_size": args.off_batch_size}
    units = [(t, "off") for t in types if "off" in modes] + [(t, "on") for t in types if "on" in modes]
    for type_id, mode in units:
        by_arm = {arm: rows for arm, rows in plan(rows_of[type_id], mode).items() if arm in arms and rows}
        stamp(f"{type_id} thinking {mode}: {sum(map(len, by_arm.values()))} passes over {len(rows_of[type_id])} rows")
        mode_dir = out_dir / type_id / mode
        (mode_dir / "conditions").mkdir(parents=True, exist_ok=True)
        for arm, rows in by_arm.items():
            run_claimed(mode_dir, arm, lambda: run_arm(mode, arm, rows, {**ctx, "type_id": type_id}))
    stamp(f"wrote {out_dir}")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--types", required=True, help="comma-separated eval types, e.g. system_prompt,prose_natural")
    parser.add_argument("--modes", default="off,on", help="off,on (off always runs first)")
    parser.add_argument("--arms", default=",".join(("none", *ARMS)), help="subset of none,neutral,long,short,shift_long,shift_short")
    parser.add_argument("--evaltypes-dir", required=True, help="the eval-type run: its settings.json (E0's settings)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows per type (smoke)")
    parser.add_argument("--model", default=None, help="model to run (default: the eval run's; a smaller one for smokes)")
    parser.add_argument("--max-think-tokens", type=int, default=None, help="think budget (default: the eval run's)")
    parser.add_argument("--batch-size", type=int, default=12, help="generation batch (E0's on the 4090)")
    parser.add_argument("--score-batch-size", type=int, default=8, help="batch for the scoring passes after a trace")
    parser.add_argument("--off-batch-size", type=int, default=32, help="batch for the thinking-off passes")
    parser.add_argument("--out", default=None, help="output base (default: out/ or $WORKSPACE/results/horizon)")
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
