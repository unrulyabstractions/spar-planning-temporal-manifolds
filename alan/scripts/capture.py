#!/usr/bin/env python
"""Capture residual-stream activations for a prompt dataset.

Usage: capture.py PROMPTS.parquet RUN_DIR [--model Qwen/Qwen3-14B] [--limit N] [--batch-size B]
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd
from transformers import AutoConfig, AutoTokenizer

from ptm.capture import CaptureConfig, run_capture
from ptm.chat import ANCHOR_LABELS, position_labels, encode_user_turn
from ptm.prompts import from_frame
from ptm.store import RunWriter



def _software_versions() -> dict:
    """Versions that change the numerics of a capture. The gated-DeltaNet kernels (flash-linear-attention) and
    their reference fallback are each deterministic but not bit-identical to each other (Qwen3.5-9B, 64 prompts:
    generations differ on some prompts), so the kernel path is part of a run's identity."""
    import importlib.metadata as md, torch, transformers
    def ver(name):
        try: return md.version(name)
        except md.PackageNotFoundError: return None
    return dict(torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__,
                flash_linear_attention=ver("flash-linear-attention"), causal_conv1d=ver("causal-conv1d"),
                gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


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
    ap.add_argument("--anchors", action="store_true", help="also store M0 (first reasoning token), E0 (last generated token), MEAN over M0..E0")
    ap.add_argument("--max-memory-gib", default=None, help="per-GPU cap in GiB, e.g. 14 or 13.9,14.7")
    ap.add_argument("--split", type=int, default=None, help="explicit 2-GPU map: decoder layers on GPU 0 (Qwen3-14B: 19)")
    a = ap.parse_args()

    df = pd.read_parquet(a.prompts)
    df = df.iloc[a.offset : (a.offset + a.limit) if a.limit else None]
    samples = from_frame(df)
    cfg = CaptureConfig(model_name=a.model, enable_thinking=a.thinking, max_new_tokens=a.max_new_tokens,
                        n_response=a.n_response, batch_size=a.batch_size, anchors=a.anchors,
                        max_memory_gib=([float(x) for x in a.max_memory_gib.split(',')] if a.max_memory_gib else None),
                        split_layers=a.split)
    hf_cfg = AutoConfig.from_pretrained(a.model)
    tcfg = getattr(hf_cfg, "text_config", hf_cfg)          # multimodal wrappers (Qwen3.5) keep the text depth/width in text_config
    n_layers, d_model = tcfg.num_hidden_layers, tcfg.hidden_size
    # transition-window length from the model's own template (Qwen3: 9 no-think / 5 think; Qwen3.5: 9 / 7),
    # verified per family in tests/test_chat_tokens.py and asserted again for every prompt at capture time
    n_trans = len(encode_user_turn(AutoTokenizer.from_pretrained(a.model), "x", enable_thinking=a.thinking).transition_tokens)
    meta = dict(
        model_name=a.model, n_layers=n_layers, d_model=d_model,
        position_labels=position_labels(n_trans, a.n_response) + (ANCHOR_LABELS if a.anchors else []),
        capture=cfg.to_dict(), prompts_file=str(a.prompts), n_prompts=len(samples),
        started=dt.datetime.now().isoformat(timespec="seconds"),
        layer_convention="layer 0 = embeddings; layer l = residual stream after decoder layer l-1 (resid_post of l-1); no final norm",
        software=_software_versions(),
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
