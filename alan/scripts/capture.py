#!/usr/bin/env python
"""Capture residual-stream activations for a prompt dataset.

Usage: capture.py PROMPTS.parquet RUN_DIR [--model Qwen/Qwen3-14B] [--limit N] [--batch-size B]
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd
from transformers import AutoConfig

from ptm.capture import CaptureConfig, run_capture
from ptm.chat import position_labels
from ptm.prompts import from_frame
from ptm.store import RunWriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--n-response", type=int, default=8)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-memory-gib", default=None, help="per-GPU cap in GiB, e.g. 14 or 13.9,14.7")
    ap.add_argument("--split", type=int, default=None, help="explicit 2-GPU map: decoder layers on GPU 0 (Qwen3-14B: 19)")
    a = ap.parse_args()

    df = pd.read_parquet(a.prompts)
    df = df.iloc[a.offset : (a.offset + a.limit) if a.limit else None]
    samples = from_frame(df)
    cfg = CaptureConfig(model_name=a.model, enable_thinking=a.thinking, max_new_tokens=a.max_new_tokens,
                        n_response=a.n_response, batch_size=a.batch_size,
                        max_memory_gib=([float(x) for x in a.max_memory_gib.split(',')] if a.max_memory_gib else None),
                        split_layers=a.split)
    hf_cfg = AutoConfig.from_pretrained(a.model)
    n_layers, d_model = hf_cfg.num_hidden_layers, hf_cfg.hidden_size
    n_trans = 5 if a.thinking else 9   # verified for Qwen3 in tests/test_chat_tokens.py; asserted again at capture time
    meta = dict(
        model_name=a.model, n_layers=n_layers, d_model=d_model,
        position_labels=position_labels(n_trans, a.n_response),
        capture=cfg.to_dict(), prompts_file=str(a.prompts), n_prompts=len(samples),
        started=dt.datetime.now().isoformat(timespec="seconds"),
        layer_convention="layer 0 = embeddings; layer l = residual stream after decoder layer l-1 (resid_post of l-1); no final norm",
    )
    writer = RunWriter(a.run_dir, meta)
    writer.register_samples(samples)

    def sink(results):
        for r in results:
            assert len(r.transition_tokens) == n_trans, f"transition window {r.transition_tokens} != expected length {n_trans}"
        writer(results)

    try:
        run_capture(samples, cfg, sink)
    finally:
        writer.meta["finished"] = dt.datetime.now().isoformat(timespec="seconds")
        writer.close()
    idx = writer.rows
    n_ok = sum(r["choice"] is not None for r in idx)
    print(f"done: {len(idx)} samples, format adherence {n_ok}/{len(idx)} = {n_ok/len(idx):.3f}")


if __name__ == "__main__":
    main()
