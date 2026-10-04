#!/usr/bin/env python
"""Token lengths of prompt datasets under a model's chat template, against its sliding-attention window.

Encodes every prompt with `ptm.chat.encode_user_turn` (the capture path; no-thinking unless --thinking) and
reports per dataset: n, min / median / max prompt tokens, and the longest full sequence (prompt + new tokens)
for each --max-new-tokens value. With a sliding window W in the model config, a sequence of length <= W attends
identically in sliding and full layers; longer ones do not. Tokenizer and config only; no weights.

Usage: prompt_lengths.py --model google/gemma-4-31B-it [--max-new-tokens 48,160] [--thinking] [PARQUET ...]
       (default: every *.parquet under data/prompts)
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from transformers import AutoConfig, AutoTokenizer
from ptm.chat import encode_user_turn

ap = argparse.ArgumentParser(); ap.add_argument("parquets", nargs="*"); ap.add_argument("--model", required=True)
ap.add_argument("--max-new-tokens", default="48,160"); ap.add_argument("--thinking", action="store_true")
a = ap.parse_args()
files = [Path(p) for p in a.parquets] or sorted(Path("data/prompts").rglob("*.parquet"))
news = [int(x) for x in a.max_new_tokens.split(",")]
tok = AutoTokenizer.from_pretrained(a.model)
cfg = AutoConfig.from_pretrained(a.model); tc = getattr(cfg, "text_config", cfg)
W = getattr(tc, "sliding_window", None)
print(f"{a.model}: sliding_window = {W}; thinking = {a.thinking}", flush=True)
worst = 0
for f in files:
    df = pd.read_parquet(f)
    if "text" not in df.columns:
        print(f"{f}: no 'text' column, skipped"); continue
    n = np.array([encode_user_turn(tok, t, a.thinking).prompt_len for t in df.text])
    worst = max(worst, int(n.max()))
    tot = "  ".join(f"max+{m} = {n.max() + m}" for m in news)
    print(f"{str(f):58s} n={len(n):5d}  prompt tokens min {n.min()} / median {int(np.median(n))} / max {n.max()}  |  {tot}", flush=True)
if W is not None:
    for m in news:
        print(f"longest sequence with {m} new tokens: {worst + m} {'<=' if worst + m <= W else '>'} window {W}"
              f"{'' if worst + m <= W else '  (sliding layers see a truncated context)'}")
