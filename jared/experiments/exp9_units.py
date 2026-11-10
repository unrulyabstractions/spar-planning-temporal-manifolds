"""E9: does the horizon representation track the duration or the unit word?
(E3 on the full bank, thinking on; spec: 07_spec_steering.md section 11.)

Every duration in `full_bank.UNIT_SETS` is rendered in each of its units ("1 year",
"12 months", "52 weeks", "365 days") inside the same full-bank prompts (the first
`--contexts` records with a horizon, delays, rewards and option order unchanged).
Each context's own prompt is added unchanged, to check the extraction against the
saved E0 activations. One forward pass up to the end of the chat-templated prompt,
no generation: the hidden states at the suffix newline after <|im_end|> and at the
last suffix token, every layer, are what reasoning starts from.

    python experiments/exp9_units.py --out /workspace/results/horizon
    python experiments/exp9_units.py --model Qwen/Qwen3.5-0.8B --contexts 2 --out out   # local smoke

Writes <out>/exp9_<model>_units/activations.npz (newline, last: [n, layers + 1, d]
float32) and prompts.json (one row per prompt: context, rendering, duration, unit).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spar_horizon import steering as S  # noqa: E402
from spar_horizon.banks import build_records  # noqa: E402
from spar_horizon.config import Settings  # noqa: E402
from spar_horizon.full_bank import UNIT_SETS, render_variant  # noqa: E402
from spar_horizon.horizons import Horizon  # noqa: E402
from spar_horizon.model_io import load_model, suffix_length  # noqa: E402
from spar_horizon.runs import code_hash, out_base, stamp  # noqa: E402


def unit_prompts(records, n_contexts):
    """One row per (context, rendering), plus each context's original prompt. The
    contexts are the first `n_contexts` records with a horizon."""
    contexts = [(i, r) for i, r in enumerate(records) if r["horizon"] is not None][:n_contexts]
    rows = []
    for index, rec in contexts:
        rows.append({"record": index, "sample_uid": rec["sample_uid"], "original": True,
                     "duration": None, "text": rec["horizon_text"], "prompt": rec["prompt"]})
        for duration, rewrites in UNIT_SETS.items():
            for text in (duration, *rewrites):
                rows.append({"record": index, "sample_uid": rec["sample_uid"], "original": False,
                             "duration": duration, "text": text,
                             "prompt": render_variant({**rec, "horizon_text": text})})
    for row in rows:
        horizon = Horizon.parse(row["text"])
        row.update(years=horizon.years, unit=horizon.unit)
    return rows


@torch.no_grad()
def extract(model, device, tokenizer, cfg, prompts, batch_size):
    """Hidden states [n, layers + 1, d] at the suffix newline and the last suffix token."""
    n_suffix = suffix_length(tokenizer, prompts[0], cfg)
    ids = [S.encode(tokenizer, p, cfg) for p in prompts]
    if any(tokenizer.decode([seq[len(seq) - n_suffix + 1]]) != "\n" for seq in ids):
        raise ValueError("the token after <|im_end|> is not a newline; the suffix layout changed")
    newline, last = [], []
    for start in tqdm(range(0, len(ids), batch_size), desc="forward", unit="batch"):
        # left padding: the suffix sits at the same offset from the end in every row
        _, hidden, _ = S.forward(model, device, ids[start:start + batch_size], tokenizer.pad_token_id, hidden=True)
        newline.append(torch.stack([h[:, -n_suffix + 1] for h in hidden], 1).float().cpu())
        last.append(torch.stack([h[:, -1] for h in hidden], 1).float().cpu())
    return torch.cat(newline).numpy(), torch.cat(last).numpy()


def main(args):
    cfg = Settings(model=args.model, thinking="on", bank="full", prefill="", choice_prefix="I choose:",
                   n_prompts=500, seed=0)
    rows = unit_prompts(build_records(cfg), args.contexts)
    stamp(f"{len(rows)} prompts: {args.contexts} contexts x {sum(1 + len(v) for v in UNIT_SETS.values())} "
          f"renderings of {len(UNIT_SETS)} durations, plus each context's original")
    tokenizer, model, device = load_model(cfg)
    t0 = time.time()
    newline, last = extract(model, device, tokenizer, cfg, [r["prompt"] for r in rows], args.batch_size)
    out_dir = out_base(args.out) / f"exp9_{cfg.model.split('/')[-1]}_units"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "activations.npz", newline=newline, last=last)
    (out_dir / "prompts.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "prompt"} for r in rows]))
    (out_dir / "manifest.json").write_text(json.dumps({
        "model": cfg.model, "thinking": cfg.thinking, "contexts": args.contexts, "n": len(rows),
        "code_hash": code_hash(), "seconds": round(time.time() - t0, 1), "shape": list(newline.shape)}, indent=1))
    stamp(f"wrote {out_dir} ({newline.nbytes * 2 / 1e9:.2f} GB of activations)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--contexts", type=int, default=24, help="full-bank prompts each rendering is placed in")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default=None, help="output base (default: out/ or $WORKSPACE/results/horizon)")
    main(parser.parse_args())
