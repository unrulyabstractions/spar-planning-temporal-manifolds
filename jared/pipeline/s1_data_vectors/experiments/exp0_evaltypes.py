"""E0 on the eval-type banks (spar_horizon/eval_types, EvalsPromptTypes.md): generate with thinking
on, locate each answer after its type's choice prefix, keep the reduced activation set.

    python -m pipeline.s1_data_vectors.experiments.exp0_evaltypes --model Qwen/Qwen3-8B --types all --batch-size 12
    python -m pipeline.s1_data_vectors.experiments.exp0_evaltypes --model Qwen/Qwen3.5-0.8B --types all --limit 2 \\
        --max-think-tokens 32 --batch-size 4 --out out/smoke                       # plumbing smoke

Run from jared/. Writes `<out>/evaltypes_<model>_think-<mode>/<type_id>/` (`related_...` with `--suite related`):
  acts_<g>.npz   per group g of the type's records (records sharing a choice prefix and a label
                 count go through one extract call): activations at `--layers` (keyed L<layer>), the
                 kept layout (regions, tokens, keep_idx, keep_ids), label logits, forced mask, think length
  rows.json      one row per record, in the type's record order: the record, `act_file` / `act_row`
                 (row in that npz, None when the generation was too short to keep), the answer from the
                 label token on, the response after the think block, think text and length, forced
The rows file is written last, so its presence marks the type done; a rerun skips done types.
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np

from spar_horizon import related_qualities
from spar_horizon.config import Settings
from spar_horizon.eval_types import TYPES, build
from spar_horizon.extract import extract
from spar_horizon.model_io import load_model
from spar_horizon.runs import out_base, stamp

LAYERS = (10, 14, 16, 18, 22, 24, 26, 30)
ANSWER_WINDOW = 200     # tokens after the think block: room for a sentence before the choice prefix


def _atomic_write(path, write):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def groups(records):
    """Record indices grouped by (choice prefix, number of labels), in first-seen order. Records
    without labels (free-form durations) score the a/b pair, which extract uses by default."""
    out = {}
    for i, r in enumerate(records):
        out.setdefault((r["choice_prefix"], len(r["labels"])), []).append(i)
    return list(out.items())


def extract_record(r):
    """The record as extract sees it: no `labels` key when it has none, so the a/b default applies."""
    return r if r["labels"] else {k: v for k, v in r.items() if k != "labels"}


def run_type(type_id, records, cfg, model_bundle, layers, type_dir):
    tokenizer, model, device = model_bundle
    rows = [dict(r, act_file=None, act_row=None) for r in records]
    for g, ((prefix, _), idx) in enumerate(groups(records)):
        gcfg = dataclasses.replace(cfg, choice_prefix=prefix)
        chunk = [extract_record(records[i]) for i in idx]
        stamp(f"{type_id} group {g}: {len(idx)} prompts, prefix {prefix!r}")
        ext = extract(tokenizer, model, device, [r["prompt"] for r in chunk], gcfg, layers=layers, records=chunk)
        regions = {f"region_{k}": np.asarray(v, int) for k, v in ext.regions.items()}
        name = f"acts_{g}.npz"
        _atomic_write(type_dir / name, lambda f: np.savez(
            f, layers=np.asarray(layers, int), tokens=np.array(ext.tokens), kept=ext.kept,
            think_fractions=np.asarray(ext.think_fractions, float), forced_mask=ext.forced_mask,
            think_len=ext.think_len, lead=ext.lead, keep_idx=ext.keep_idx, keep_ids=ext.keep_ids,
            label_logits=ext.label_logits, **regions, **{f"L{l}": a for l, a in zip(layers, ext.activations)}))
        kept_rows = iter(range(int(ext.kept.sum())))
        for i, answer, kept in zip(idx, ext.answers, ext.kept):
            if not kept:
                rows[i]["answer"] = None
                continue
            j = next(kept_rows)
            gen = ext.generations[j]
            rows[i].update(act_file=name, act_row=j, answer=answer, forced=bool(ext.forced_mask[j]),
                           think_len=int(ext.think_len[j]), think_text=ext.think_text[j],
                           response_text=tokenizer.decode(gen.ids[gen.prompt_len + gen.think_len:]),
                           label_logits=ext.label_logits[j].tolist())
    _atomic_write(type_dir / "rows.json", lambda f: f.write(json.dumps(rows, indent=1).encode()))
    stamp(f"{type_id}: {sum(r['answer'] is not None for r in rows)}/{len(rows)} kept, "
          f"{sum(r.get('forced', False) for r in rows)} force-closed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=Settings.model)
    p.add_argument("--suite", choices=("evaltypes", "related"), default="evaltypes",
                   help="eval-type banks, or the related-quality sets (spar_horizon/related_qualities.py)")
    p.add_argument("--types", default="all", help="comma-separated type ids, or 'all'")
    p.add_argument("--limit", type=int, default=None, help="records per type (smoke)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-think-tokens", type=int, default=Settings.max_think_tokens)
    p.add_argument("--batch-size", type=int, default=Settings.batch_size)
    p.add_argument("--layers", default=",".join(map(str, LAYERS)))
    p.add_argument("--out", default=None, help="output base dir (default: out/ or $WORKSPACE/results/horizon)")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    suite_types = TYPES if args.suite == "evaltypes" else related_qualities.TYPES
    type_ids = list(suite_types) if args.types == "all" else args.types.split(",")
    base = out_base(args.out)
    cfg = Settings(model=args.model, thinking="on", bank="full", prefill="", choice_prefix="I choose:",
                   max_think_tokens=args.max_think_tokens, batch_size=args.batch_size, seed=args.seed,
                   answer_window=ANSWER_WINDOW, cache_dir=None if args.no_cache else str(base / "gen_cache"))
    run(type_ids, cfg, base, tuple(int(x) for x in args.layers.split(",")), args.limit, args.suite)


def run(type_ids, cfg, base, layers, limit=None, suite="evaltypes"):
    """Run every type in `type_ids` that has no rows.json yet; returns the run directory."""
    build_records = build if suite == "evaltypes" else related_qualities.build
    run_dir = Path(base) / f"{suite}_{cfg.model.split('/')[-1]}_think-{cfg.thinking}"
    todo = []
    for type_id in type_ids:
        type_dir = run_dir / type_id
        if (type_dir / "rows.json").exists():
            stamp(f"{type_id}: done earlier, skipped")
            continue
        records = build_records(type_id, cfg.seed)[:limit]
        type_dir.mkdir(parents=True, exist_ok=True)
        todo.append((type_id, records, type_dir))
    stamp(f"{len(todo)} of {len(type_ids)} types to run, {sum(len(r) for _, r, _ in todo)} prompts, into {run_dir}")
    if not todo:
        return run_dir
    bundle = load_model(cfg)
    n_layers = bundle[1].config.get_text_config().num_hidden_layers
    layers = tuple(l for l in layers if l < n_layers)
    (run_dir / "settings.json").write_text(json.dumps({**cfg.to_dict(), "layers": layers, "types": type_ids}, indent=1))
    for type_id, records, type_dir in todo:
        run_type(type_id, records, cfg, bundle, layers, type_dir)
    return run_dir

if __name__ == "__main__":
    main()
