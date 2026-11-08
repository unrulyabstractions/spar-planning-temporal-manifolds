"""Does E8's shared-prefix scoring (steering.score_last_edits) give the same near
log-odds as one full forward pass per edit, on a real model in its real dtype?
On the tiny float32 test model they agree to 1e-5; in bfloat16 the cached and
full passes round differently, and this measures by how much.

    python -m pipeline.s1_data_vectors.experiments.check_prefix_cache                        # Qwen3-0.6B (all full attention)
    python -m pipeline.s1_data_vectors.experiments.check_prefix_cache --model Qwen/Qwen3-8B
    python -m pipeline.s1_data_vectors.experiments.check_prefix_cache --float32               # the logic alone, without bf16 rounding

Short rows are thinking-off full-bank prompts teacher-forced through "I choose:";
long rows join several of them, to reach the ~2.5k tokens of a reasoning trace.
Passes if every edit's largest difference is below --tolerance log-odds (E8's
effects of interest are ~1 log-odds and up).
"""

import argparse
import sys

import numpy as np
import torch

from spar_horizon import steering as S
from spar_horizon.banks import build_records
from spar_horizon.config import Settings
from spar_horizon.model_io import label_ids, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--float32", action="store_true",
                        help="upcast the model: in bf16 even the unedited cached and full passes differ by one "
                             "logit rounding step (0.125 log-odds for logits of 16-32)")
    args = parser.parse_args()
    cfg = Settings(model=args.model, thinking="off", prefill="I choose:", bank="full", limit=8)
    tokenizer, model, device = load_model(cfg)
    if args.float32:
        model = model.float()
    records = build_records(cfg)
    short = [S.encode(tokenizer, r["prompt"], cfg) for r in records]
    long = [sum(short[i:] + short[:i], [])[-2500:] for i in range(4)]   # ends in a real prompt's answer slot
    labels = label_ids(tokenizer, cfg=cfg)
    d = model.config.get_text_config().hidden_size
    layer = len(model.model.layers) // 2
    gen = torch.Generator().manual_seed(0)
    delta = (4 * torch.randn(len(short), d, generator=gen)).to(device)
    basis = torch.linalg.qr(torch.randn(d, 4, generator=gen))[0].T.to(device)
    mean = torch.zeros(d, device=device)
    donor = (8 * torch.randn(len(short), d, generator=gen)).to(device)
    edits = {"none": None,
             "add": (layer, lambda rows: S.Add(delta[rows])),
             "mean ablate": (layer, lambda rows: S.MeanAblate(mean, basis)),
             "replace": (layer, lambda rows: S.Replace(donor[rows]))}
    ok = True
    for name, seqs, batch in (("short", short, 4), ("long", long, 2)):
        near_first = [True] * len(seqs)
        rows = [labels] * len(seqs)
        cached = S.score_last_edits(model, device, seqs, near_first, rows, tokenizer.pad_token_id, batch,
                                    list(edits.values()))
        for e, (edit_name, spec) in enumerate(edits.items()):
            if spec is None:
                full = S.score(model, device, seqs, near_first, rows, tokenizer.pad_token_id, batch)
            else:
                full = S.score(model, device, seqs, near_first, rows, tokenizer.pad_token_id, batch, spec[0],
                               [[len(s) - 1] for s in seqs], spec[1])
            gap = float(np.abs(cached[e] - full).max())
            ok &= gap < args.tolerance
            print(f"prefix cache {name:5} ({max(map(len, seqs))} tokens) {edit_name:11}: "
                  f"max |cached - full| {gap:.4f} log-odds (values {np.round(full, 2)})")
    print(f"prefix cache: {'PASS' if ok else 'FAIL'} at tolerance {args.tolerance}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
