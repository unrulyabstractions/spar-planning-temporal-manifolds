"""How many token-cap hits in a generation run are repetition loops (CPU only, reads prompt_bank outputs).

A response counts as looping when, in its last `--window` generated tokens, some block of `--n` consecutive tokens
occurs at least `--repeats` times. Long-but-progressing reasoning does not repeat 64-token blocks verbatim, so the
default thresholds separate "stuck" from "slow" cleanly in practice (Qwen3-1.7B thinking on, greedy: 323/337 cap hits loop).

Why it matters: greedy decoding with thinking on is known to loop (Qwen3 model cards say not to use it in thinking
mode; they recommend temperature 0.6, top-p 0.95, top-k 20). Loops cost the full token budget and lose the answer.

Usage: python check_loops.py --run ../prompt_bank/runs/<model>__<bank> [--bank ../prompt_bank/data/<bank>]
  --bank adds a breakdown by regime (field `regime` of prompts.jsonl).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def is_looping(ids, n=64, repeats=3, window=1024):
    """True if some n-token block occurs >= `repeats` times in the last `window` tokens."""
    tail = ids[-window:]
    if len(tail) < n * repeats:
        return False
    counts = Counter(tuple(tail[i:i + n]) for i in range(len(tail) - n + 1))
    return max(counts.values()) >= repeats


def summarize(rows, n, repeats, window):
    s = {"responses": len(rows), "answered": 0, "hit_cap": 0, "hit_cap_looping": 0, "answered_looping": 0}
    for r in rows:
        loop = is_looping(r["generated_ids"], n, repeats, window)
        if r["parse_status"] == "ok":
            s["answered"] += 1
            s["answered_looping"] += loop
        if r["hit_token_cap"]:
            s["hit_cap"] += 1
            s["hit_cap_looping"] += loop
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="prompt_bank run dir (has responses.jsonl)")
    ap.add_argument("--bank", type=Path, help="bank dir (has prompts.jsonl), for a breakdown by regime")
    ap.add_argument("--n", type=int, default=64, help="block length in tokens")
    ap.add_argument("--repeats", type=int, default=3, help="minimum occurrences of the block")
    ap.add_argument("--window", type=int, default=1024, help="tokens from the end that are searched")
    a = ap.parse_args()

    rows = read_jsonl(a.run / "responses.jsonl")
    print(f"{a.run.name}: loop = a {a.n}-token block repeated >= {a.repeats}x in the last {a.window} tokens")
    groups = {"all": rows}
    if a.bank:
        regime = {p["uid"]: p.get("regime", "?") for p in read_jsonl(a.bank / "prompts.jsonl")}
        by = defaultdict(list)
        for r in rows:
            by[regime.get(r["uid"], "?")].append(r)
        groups.update(sorted(by.items()))
    print(f"{'group':<14} {'responses':>9} {'answered':>8} {'hit cap':>7} {'cap+loop':>8} {'answered+loop':>13}")
    for name, g in groups.items():
        s = summarize(g, a.n, a.repeats, a.window)
        print(f"{name:<14} {s['responses']:>9} {s['answered']:>8} {s['hit_cap']:>7} {s['hit_cap_looping']:>8} "
              f"{s['answered_looping']:>13}")


if __name__ == "__main__":
    main()
