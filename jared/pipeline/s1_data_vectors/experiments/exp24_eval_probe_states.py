"""E24 step 1: hidden states for the eval-awareness probe (EvalsRelatedQualities.md §4). Forward passes
over the eval/deploy training pairs and the implicit benchmark/natural pairs, no generation.

    python -m pipeline.s1_data_vectors.experiments.exp24_eval_probe_states --model Qwen/Qwen3-8B --batch-size 16
    python -m pipeline.s1_data_vectors.experiments.exp24_eval_probe_states --model Qwen/Qwen3.5-0.8B --limit 6 \\
        --layers 12,16 --out out/smoke                                   # plumbing smoke

Each prompt is encoded with the thinking-on chat template and read at its last token: the "\\n" after
`<|im_start|>assistant`, the last position of the T0 "suffix" region, so a probe fitted here can be read
on the saved T0 and eval-type activations. Writes <out>/eval_probe_states_<model short>/:
  states.npz  `layers` [L]; `train_X` [n, L, d] float32, `train_label` (eval = 1), `train_request_id`;
              `implicit_X` [m, L, d], `implicit_label` (benchmark = 1)
  rows.json   the records of both sets with `text`, the message contents joined, for the bag-of-words baseline
"""

import argparse
import json

import numpy as np
from tqdm import tqdm

from spar_horizon.config import Settings
from spar_horizon.eval_awareness import implicit_pairs, training_pairs
from spar_horizon.model_io import chat_messages, load_model
from spar_horizon.runs import out_base, stamp
from spar_horizon.steering import encode, states_at

LAYERS = (16, 22, 24, 26, 28, 30, 32)


def prompt_text(prompt):
    """The prompt's message contents joined, without the chat markup."""
    return "\n".join(m["content"] for m in chat_messages(prompt))


def last_token_states(model, device, tokenizer, cfg, prompts, layers, batch_size):
    """float32 [len(prompts), len(layers), d]: each prompt's hidden state at its last token."""
    ids = [encode(tokenizer, p, cfg) for p in prompts]
    n_layers = len(model.model.layers)
    if max(layers) >= n_layers:
        raise ValueError(f"the model has {n_layers} layers; the states need layer {max(layers)}")
    rows = [None] * len(prompts)
    by_length = np.argsort([len(x) for x in ids])   # similar lengths per batch: less padding
    for start in tqdm(range(0, len(prompts), batch_size), desc="probe states", unit="batch", dynamic_ncols=True):
        batch = by_length[start:start + batch_size]
        states = states_at(model, device, tokenizer, [ids[i] for i in batch], [[len(ids[i]) - 1] for i in batch],
                           layers, len(batch))
        for slot, i in enumerate(batch):
            rows[i] = np.stack([states[layer][slot][0].numpy() for layer in layers])
    return np.stack(rows).astype(np.float32)


def main(args):
    layers = tuple(int(x) for x in args.layers.split(","))
    out_dir = out_base(args.out) / f"eval_probe_states_{args.model.split('/')[-1]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train, implicit = training_pairs(), implicit_pairs()
    if args.limit:
        train, implicit = train[:args.limit], implicit[:args.limit]
    cfg = Settings(model=args.model, thinking="on")
    tokenizer, model, device = load_model(cfg)
    stamp(f"{len(train)} training and {len(implicit)} implicit prompts, {args.model}, layers {layers}")
    train_X = last_token_states(model, device, tokenizer, cfg, [r["prompt"] for r in train], layers, args.batch_size)
    implicit_X = last_token_states(model, device, tokenizer, cfg, [r["prompt"] for r in implicit], layers,
                                   args.batch_size)
    np.savez(out_dir / "states.npz", layers=np.asarray(layers), train_X=train_X, implicit_X=implicit_X,
             train_label=np.array([r["framing"] == "eval" for r in train], dtype=int),
             train_request_id=np.array([r["request_id"] for r in train]),
             implicit_label=np.array([r["cue"] == "benchmark" for r in implicit], dtype=int))
    rows = {"train": train, "implicit": implicit}
    for records in rows.values():
        for r in records:
            r["text"] = prompt_text(r["prompt"])
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=1))
    stamp(f"wrote {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=Settings.model)
    parser.add_argument("--layers", default=",".join(map(str, LAYERS)), help="hidden_states indices")
    parser.add_argument("--batch-size", type=int, default=16, help="prompts per forward pass")
    parser.add_argument("--limit", type=int, default=None, help="prompts per set (smoke)")
    parser.add_argument("--out", default=None, help="output base (default: out/ or $WORKSPACE/results/horizon)")
    main(parser.parse_args())
