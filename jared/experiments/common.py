"""Shared CLI for every experiment script."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spar_horizon.config import Settings  # noqa: E402


def parse_args(description, extra=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", default=Settings.model)
    p.add_argument("--thinking", choices=["on", "off"], default=Settings.thinking)
    p.add_argument("--limit", type=int, default=None, help="prompt subset for smoke tests")
    p.add_argument("--max-think-tokens", type=int, default=Settings.max_think_tokens)
    p.add_argument("--batch-size", type=int, default=Settings.batch_size)
    p.add_argument("--prefill", default=Settings.prefill, help='assistant prefill; "" to disable')
    p.add_argument("--out", default=None, help="output base dir (default: out/ or $WORKSPACE/results/horizon)")
    if extra:
        extra(p)
    args = p.parse_args()
    cfg = Settings(model=args.model, thinking=args.thinking, limit=args.limit,
                   max_think_tokens=args.max_think_tokens, batch_size=args.batch_size,
                   prefill=args.prefill)
    return args, cfg
