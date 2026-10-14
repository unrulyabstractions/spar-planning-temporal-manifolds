"""Shared CLI for every experiment script."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spar_horizon.config import Settings  # noqa: E402
from spar_horizon.runs import out_base  # noqa: E402


def parse_args(description, extra=None):
    """Parsed args and the Settings they describe. The generation cache lives
    under the output base unless --no-cache is given."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", default=Settings.model)
    p.add_argument("--thinking", choices=["on", "off"], default=Settings.thinking)
    p.add_argument("--limit", type=int, default=None, help="prompt subset for smoke tests")
    p.add_argument("--max-think-tokens", type=int, default=Settings.max_think_tokens)
    p.add_argument("--batch-size", type=int, default=Settings.batch_size)
    p.add_argument("--bank", choices=["starter", "full"], default=Settings.bank,
                   help="starter: the 116-prompt starter bank; full: Alan's generator (default prefill off, "
                        "label located after 'I choose:')")
    p.add_argument("--n-prompts", type=int, default=Settings.n_prompts, help="full bank size")
    p.add_argument("--behavior-n", type=int, default=None,
                   help="run the order/label stability tests on this many evenly spaced prompts (default: all)")
    p.add_argument("--prefill", default=None,
                   help='assistant prefill; "" to disable (default: "I choose:" on the starter bank, off on the full bank)')
    p.add_argument("--think-fractions", default=",".join(map(str, Settings.think_fractions)),
                   help="comma-separated fractions of the think span to keep (thinking on)")
    p.add_argument("--out", default=None, help="output base dir (default: out/ or $WORKSPACE/results/horizon)")
    p.add_argument("--no-cache", action="store_true", help="do not read or write the generation cache")
    p.add_argument("--attn", default=None, help='attention implementation, e.g. "eager" (default: library sdpa)')
    if extra:
        extra(p)
    args = p.parse_args()
    fractions = tuple(float(f) for f in args.think_fractions.split(",") if f)
    cache = None if args.no_cache else str(out_base(args.out) / "gen_cache")
    prefill = args.prefill if args.prefill is not None else ("" if args.bank == "full" else Settings.prefill)
    choice_prefix = "I choose:" if args.bank == "full" and not prefill else ""
    cfg = Settings(model=args.model, thinking=args.thinking, limit=args.limit,
                   max_think_tokens=args.max_think_tokens, batch_size=args.batch_size,
                   prefill=prefill, choice_prefix=choice_prefix, bank=args.bank, n_prompts=args.n_prompts,
                   behavior_n=args.behavior_n,
                   think_fractions=fractions, cache_dir=cache, attn=args.attn)
    return args, cfg
