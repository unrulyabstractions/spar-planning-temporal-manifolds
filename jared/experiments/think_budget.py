"""Does a model close its think block on our prompts within a budget?

    python experiments/think_budget.py --model Qwen/Qwen3-4B --max-think-tokens 3072 --limit 4 --batch-size 4

Prints seconds per batch, and per prompt the think length, whether it was
force-closed, the answer, and the tail of the reasoning. Use before a full
thinking-on E0 on a new model to pick `max_think_tokens` and `batch_size`.
"""

import time

from common import parse_args
from spar_horizon.banks import build_records
from spar_horizon.model_io import generate_batch, load_model


def run(cfg):
    tokenizer, model, device = load_model(cfg)
    records = build_records(cfg)
    t0 = time.time()
    gens = generate_batch(tokenizer, model, device, [r["prompt"] for r in records], cfg)
    forced = sum(g.forced for g in gens)
    print(f"{cfg.model} budget {cfg.max_think_tokens}: {time.time() - t0:.0f} s for {len(records)} prompts, "
          f"force-closed {forced}/{len(records)}")
    for r, g in zip(records, gens):
        horizon = r.get("horizon_text") or ("none" if r["horizon"] is None else f"{r['horizon']:g} y")
        print(f"  horizon={horizon:>11}  regime={r.get('regime', '-'):>5}  near={r.get('near_label', 'a')}  "
              f"think_len={g.think_len:>5}  forced={g.forced}  answer={g.answer_text(tokenizer)[:30]!r}")
        print("     think tail:", repr(g.think_text(tokenizer)[-160:]))


if __name__ == "__main__":
    args, cfg = parse_args(__doc__)
    cfg.thinking = "on"
    run(cfg)
