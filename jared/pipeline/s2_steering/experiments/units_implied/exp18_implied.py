"""E18: implied horizons (29_spec_implied_horizon_side.md). Qwen3-8B, thinking on, greedy.

    python -m pipeline.s2_steering.experiments.units_implied.exp18_implied --out /workspace/results/horizon                  # M1, M2, M3, M4
    python -m pipeline.s2_steering.experiments.units_implied.exp18_implied --out /workspace/results/horizon --modules M1    # one module
    python -m pipeline.s2_steering.experiments.units_implied.exp18_implied --model Qwen/Qwen3.5-0.8B --limit 6 --max-think-tokens 24 \\
        --batch-size 4 --out out                                    # local smoke (plumbing only)

The bank (`spar_horizon.implied_bank`) is cut into units of up to --batch-size records that keep each
base's renderings together (implied, explicit twin, null), module by module: M1 (families A and C,
near/both), M2 (B), M3 (none regime), M4 (vague). A unit is generated and extracted on its own and
written to `<run dir>/units/<module>_<NNN>.npz` and `.json` through a temp file and a rename, the json
last; a rerun skips finished units, so a killed run loses at most the unit in flight, and any prefix of
units is analysable (pipeline/s2_steering/analysis/implied_summary.py). --limit keeps that many records per module (whole
bases, evenly spaced; smoke only).

The json holds each record's fields (without the prompt) and, for the kept ones, the answer, label,
forced close, think length and think text. The npz holds, per kept record, the states at the layers
--layers (hidden_states indices) at the suffix tokens (the newline after <|im_end|> is suffix_1),
`think@1` (the last think token) and `prelabel` (the token before the label), float32, plus label
logits (a, b), forced and think length. An out-of-memory error steps the batch down for the rest of the
run (generation inside a unit steps down on its own, see model_io.generate_batch).
"""

import argparse
import json
from itertools import groupby

import numpy as np
import torch
from tqdm import tqdm

from pipeline.s1_data_vectors.experiments.exp0_choice import _atomic_write, usable_layers
from spar_horizon.banks import build_records
from spar_horizon.config import Settings
from spar_horizon.extract import extract
from spar_horizon.full_bank import evenly_spaced_bases
from spar_horizon.implied_bank import MODULES
from spar_horizon.model_io import _free_memory, load_model
from spar_horizon.runs import Manifest, out_base, resolve_out_dir, stamp

EXP = "exp18"
LAYERS = (12, 16, 20, 24)
BATCH = 12


def settings(args):
    cache = None if args.no_cache else str(out_base(args.out) / "gen_cache")
    return Settings(model=args.model, thinking="on", max_think_tokens=args.max_think_tokens,
                    batch_size=args.batch_size, prefill="", choice_prefix="I choose:", bank="implied",
                    seed=args.seed, think_fractions=(1.0,), cache_dir=cache)


def module_groups(records, module, limit=None):
    """The module's bases as lists of records (run order); `limit` keeps about that many records."""
    mine = [r for r in records if r["module"] == module]
    groups = [list(g) for _, g in groupby(mine, key=lambda r: r["twin_id"])]
    return evenly_spaced_bases(groups, limit) if groups else []


def pack_units(groups, size):
    """Consecutive groups packed into units of at most `size` records, never splitting a group."""
    units, current = [], []
    for group in groups:
        if current and len(current) + len(group) > size:
            units.append(current)
            current = []
        current = current + group
    return units + ([current] if current else [])


def planned_units(records, modules, size, limit=None):
    """[(name, module, records)] in run order."""
    return [(f"{module}_{n:03d}", module, unit)
            for module in modules for n, unit in enumerate(pack_units(module_groups(records, module, limit), size))]


def unit_paths(unit_dir, name):
    return unit_dir / f"{name}.npz", unit_dir / f"{name}.json"


def finished_units(unit_dir):
    """The json is renamed last, so its presence means the npz is complete."""
    return sorted(p.stem for p in unit_dir.glob("*.json"))


def check_unit_matches(unit_dir, name, unit):
    uids = json.loads(unit_paths(unit_dir, name)[1].read_text())["sample_uids"]
    if uids != [r["sample_uid"] for r in unit]:
        raise ValueError(f"{unit_paths(unit_dir, name)[1]} was written for other prompts (another --limit or "
                         f"--batch-size?); use another --out or delete {unit_dir}")


def state_columns(ext):
    """Kept-layout columns of the saved states and their names: every suffix token, think@1 (the last
    think token, the only think fraction extracted), the pre-label token."""
    suffix = list(range(*ext.regions["suffix"]))
    cols = suffix + [ext.regions["think"][1] - 1, ext.regions["prelabel"][0]]
    return cols, [f"suffix_{i}" for i in range(len(suffix))] + ["think@1", "prelabel"]


def answer_row(rec, answer, kept_info):
    row = {k: v for k, v in rec.items() if k != "prompt"}
    row["answer"] = answer
    if kept_info is not None:
        forced, think_len, think_text, logits = kept_info
        row.update(label=answer.strip()[:1] if answer else None, forced=bool(forced), think_len=int(think_len),
                   think_text=think_text, label_logits=[float(x) for x in logits])
    return row


def write_unit(unit_dir, name, module, unit, ext, layers):
    cols, names = state_columns(ext)
    arrays = {"layers": np.asarray(layers, int), "cols": np.array(names), "col_tokens": np.array([ext.tokens[c] for c in cols]),
              "kept": ext.kept, "forced_mask": ext.forced_mask, "think_len": ext.think_len,
              "label_logits": ext.label_logits,
              "sample_uids": np.array([r["sample_uid"] for r, k in zip(unit, ext.kept) if k]),
              **{f"L{layer}": a[:, cols].astype(np.float32) for layer, a in zip(layers, ext.activations)}}
    kept_info = iter(zip(ext.forced_mask, ext.think_len, ext.think_text, ext.label_logits))
    rows = [answer_row(r, a, next(kept_info) if k else None) for r, a, k in zip(unit, ext.answers, ext.kept)]
    npz_path, json_path = unit_paths(unit_dir, name)
    _atomic_write(npz_path, lambda f: np.savez(f, **arrays))
    meta = {"unit": name, "module": module, "sample_uids": [r["sample_uid"] for r in unit],
            "kept": ext.kept.tolist(), "rows": rows}
    _atomic_write(json_path, lambda f: f.write(json.dumps(meta, indent=1).encode()))


def extract_with_stepdown(tokenizer, model, device, unit, cfg, layers):
    while True:
        try:
            return extract(tokenizer, model, device, [r["prompt"] for r in unit], cfg, layers=layers, records=unit)
        except torch.OutOfMemoryError:
            if cfg.batch_size == 1:
                raise
            _free_memory(device)
            cfg.batch_size = max(1, cfg.batch_size * 2 // 3)
            stamp(f"out of memory; continuing at batch size {cfg.batch_size}")


def run_units(cfg, out_dir, units, layers):
    unit_dir = out_dir / "units"
    unit_dir.mkdir(exist_ok=True)
    done = set(finished_units(unit_dir))
    for name, _, unit in units:
        if name in done:
            check_unit_matches(unit_dir, name, unit)
    todo = [u for u in units if u[0] not in done]
    stamp(f"{sum(len(u[2]) for u in units)} records in {len(units)} units; {len(units) - len(todo)} done, "
          f"{len(todo)} to run")
    if not todo:
        return
    tokenizer, model, device = load_model(cfg)
    layers = usable_layers(model, layers)
    for name, module, unit in tqdm(todo, desc="units", unit="unit", dynamic_ncols=True):
        stamp(f"{name}: {len(unit)} records")
        ext = extract_with_stepdown(tokenizer, model, device, unit, cfg, layers)
        write_unit(unit_dir, name, module, unit, ext, layers)
        stamp(f"{name} written: {int(ext.kept.sum())}/{len(unit)} kept, {ext.forced}/{len(unit)} force-closed")


def run(cfg, out=None, modules=MODULES, limit=None, layers=LAYERS):
    unknown = [m for m in modules if m not in MODULES]
    if unknown:
        raise ValueError(f"unknown modules {unknown}; choose from {MODULES}")
    out_dir = resolve_out_dir(EXP, cfg, out)
    records = build_records(cfg)
    units = planned_units(records, modules, cfg.batch_size, limit)
    print(f"{len(records)} prompts in the bank, model {cfg.model}, modules {','.join(modules)}")
    manifest = Manifest(EXP, cfg, out_dir)
    try:
        run_units(cfg, out_dir, units, layers)
    finally:
        manifest.add(modules=list(modules), limit=limit, layers=list(layers), n_units_planned=len(units),
                     units_done=finished_units(out_dir / "units"))
        manifest.write()
    return out_dir


def make_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=Settings.model)
    p.add_argument("--modules", default=",".join(MODULES), help="comma-separated, run in this order")
    p.add_argument("--limit", type=int, default=None, help="records per module (smoke tests)")
    p.add_argument("--max-think-tokens", type=int, default=Settings.max_think_tokens)
    p.add_argument("--batch-size", type=int, default=BATCH)
    p.add_argument("--layers", default=",".join(map(str, LAYERS)), help="hidden_states indices to keep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output base dir (default: out/ or $WORKSPACE/results/horizon)")
    p.add_argument("--no-cache", action="store_true", help="do not read or write the generation cache")
    return p


if __name__ == "__main__":
    args = make_parser().parse_args()
    run(settings(args), args.out, tuple(m for m in args.modules.split(",") if m), args.limit,
        tuple(int(l) for l in args.layers.split(",")))
