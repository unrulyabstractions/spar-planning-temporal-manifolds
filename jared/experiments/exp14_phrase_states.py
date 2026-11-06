"""E14 step 1: hidden states at the horizon phrase of the pooled thinking-on prompts
(21_spec_phrase_steering.md §7). One prompt-only forward pass per prompt, no generation.

    python experiments/exp14_phrase_states.py --prompts steering/phrase_prompts_Qwen3-8B.json \\
        --out /workspace/results/horizon
    python experiments/exp14_phrase_states.py --prompts steering/phrase_prompts_Qwen3-8B.json \\
        --model Qwen/Qwen3.5-0.8B --limit 8 --batch-size 4 --out out      # local smoke (plumbing only)

The phrase has two roles: "number" is the mean over all its tokens but the last
("59", "1.8"), "unit" is the last token (" years"). Writes
<out>/exp14_<model short>_phrase/states.npz (keys `L{layer}|{role}` float32 [n, d] for
L12, L14, L16, L18, `n_number` [n], `sample_uids`; rows in the prompts file's order) and
states.json. Skipped when states.npz exists, so a queue can rerun it.

With --layers other than the default the same keys are written for those layers to
states_L<layers joined by "-">.npz (and .json), e.g. --layers 10,22,26 -> states_L10-22-26.npz,
which has its own skip rule (26_spec_overnight_choice_scan.md §2).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp8_steering import selected_records  # noqa: E402  (same folder)
from exp10_patch_reasoning import states_at  # noqa: E402
from spar_horizon import steering as S  # noqa: E402
from spar_horizon.config import Settings  # noqa: E402
from spar_horizon.model_io import chat_text, load_model  # noqa: E402
from spar_horizon.runs import code_hash, out_base, stamp  # noqa: E402

LAYERS = (12, 14, 16, 18)   # L12 is kept for a follow-up; the steering fits use L14, L16, L18
ROLES = ("number", "unit")


def record_phrase(tokenizer, cfg, record):
    """Token positions of the record's horizon phrase; the last one is the unit."""
    positions = S.phrase_positions(tokenizer, chat_text(tokenizer, record["prompt"], cfg), record["horizon_text"])
    if len(positions) < 2:
        raise ValueError(f"{record['sample_uid']}: horizon {record['horizon_text']!r} is {len(positions)} token(s); "
                         "need at least a number and a unit")
    return positions


def states_name(layers):
    """File stem of the states of `layers`: "states" for the default layers, else states_L10-22-26."""
    return "states" if tuple(layers) == LAYERS else "states_L" + "-".join(str(layer) for layer in layers)


def phrase_states(model, device, tokenizer, cfg, records, batch_size, layers=LAYERS):
    """{"L{layer}|{role}": float32 [n, d]} and the number-token counts, in `records` order."""
    ids = [S.encode(tokenizer, r["prompt"], cfg) for r in records]
    positions = [record_phrase(tokenizer, cfg, r) for r in records]
    n_layers = len(model.model.layers)
    if max(layers) >= n_layers:
        raise ValueError(f"the model has {n_layers} layers; the states need layer {max(layers)}")
    rows = {f"L{layer}|{role}": [None] * len(records) for layer in layers for role in ROLES}
    by_length = np.argsort([len(x) for x in ids])   # similar lengths per batch: less padding
    for start in tqdm(range(0, len(records), batch_size), desc="phrase states", unit="batch", dynamic_ncols=True):
        batch = by_length[start:start + batch_size]
        states = states_at(model, device, tokenizer, [ids[i] for i in batch], [positions[i] for i in batch],
                           layers, len(batch))
        for layer in layers:
            for i, phrase in zip(batch, states[layer]):
                rows[f"L{layer}|number"][i] = phrase[:-1].mean(0).numpy()
                rows[f"L{layer}|unit"][i] = phrase[-1].numpy()
    arrays = {key: np.stack(values).astype(np.float32) for key, values in rows.items()}
    arrays["n_number"] = np.array([len(p) - 1 for p in positions])
    return arrays


def main(args):
    meta = json.loads(Path(args.prompts).read_text())
    model_name = args.model or meta["model"]
    out_dir = out_base(args.out) / f"exp14_{model_name.split('/')[-1]}_phrase"
    layers = tuple(int(x) for x in args.layers.split(",")) if args.layers else LAYERS
    stem = states_name(layers)
    if (out_dir / f"{stem}.npz").exists():
        stamp(f"{out_dir / f'{stem}.npz'} exists; skipping")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    records = selected_records(meta)
    if args.limit:
        records = records[:args.limit]
    cfg = Settings(**{**meta["generation"], "model": model_name})
    tokenizer, model, device = load_model(cfg)
    stamp(f"{len(records)} prompts, {model_name}, layers {layers}")
    arrays = phrase_states(model, device, tokenizer, cfg, records, args.batch_size, layers)
    arrays["sample_uids"] = np.array([r["sample_uid"] for r in records])
    (out_dir / f"{stem}.json").write_text(json.dumps({
        "model": model_name, "layers": list(layers), "prompts_file": str(args.prompts), "n": len(records),
        "code_hash": code_hash()}, indent=1))
    # write then rename, so a reader never sees half a file
    np.savez(out_dir / f"{stem}.tmp.npz", **arrays)
    (out_dir / f"{stem}.tmp.npz").replace(out_dir / f"{stem}.npz")
    stamp(f"wrote {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prompts", required=True, help="steering/phrase_prompts_<model>.json")
    parser.add_argument("--model", default=None, help="model to run (default: the prompts file's); for the 0.8B smoke")
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts (smoke)")
    parser.add_argument("--layers", default=None,
                        help=f'comma-separated layers to extract (default {",".join(map(str, LAYERS))}); '
                             "others go to states_L<layers>.npz")
    parser.add_argument("--batch-size", type=int, default=16, help="prompts per forward pass")
    parser.add_argument("--out", default=None, help="output base (default: out/ or $WORKSPACE/results/horizon)")
    main(parser.parse_args())
