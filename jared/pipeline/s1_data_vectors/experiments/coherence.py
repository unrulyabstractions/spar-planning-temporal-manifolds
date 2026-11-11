"""The paper's three coherence tests (temporal reasoning, order stability, label
stability; spar_horizon/behavior.py) on prompts whose base generation is already
in the cache. E0 runs them itself only with thinking off (thinking on passes
--skip-behavior); this runs them with thinking on, on an idle GPU while E0's
thinking-on generation is still filling the cache. Only the swapped and
relabeled variants are generated, so it never regenerates or rewrites a base
generation E0 is about to use.

    CUDA_VISIBLE_DEVICES=1 python -m pipeline.s1_data_vectors.experiments.coherence --model Qwen/Qwen3-14B --thinking on --bank full \
        --out /workspace/results/horizon --behavior-n 80

Writes <out>/coherence_<model>_think-<mode>_full/coherence.json.
"""

import json
from collections import Counter

from pipeline.s1_data_vectors.experiments.common import parse_args
from spar_horizon.banks import build_records, variant_builder
from spar_horizon.behavior import behavior
from spar_horizon.model_io import cache_load, chat_text, load_model
from spar_horizon.prompts import subset
from spar_horizon.runs import resolve_out_dir, stamp


def cached_base(tokenizer, records, cfg):
    """{record index: base answer text} for every record whose base generation is cached.
    The answer is None when the generation is shorter than the kept answer window, as in extract()."""
    answers = {}
    for i, rec in enumerate(records):
        ids = tokenizer.encode(chat_text(tokenizer, rec["prompt"], cfg), add_special_tokens=False)
        g = cache_load(cfg, ids, None, "cpu")
        if g is not None:
            answers[i] = None if g.ids.shape[0] < g.answer_start + cfg.n_response else g.answer_text(tokenizer)
    return answers


def main():
    args, cfg = parse_args(__doc__)
    if cfg.behavior_n is None:
        raise ValueError("pass --behavior-n: every prompt costs two generations")
    records = build_records(cfg)
    tokenizer, model, device = load_model(cfg)
    base = cached_base(tokenizer, records, cfg)
    idx = subset(sorted(base), cfg.behavior_n)
    chosen = [records[i] for i in idx]
    stamp(f"{len(base)} of {len(records)} base generations cached; coherence tests on {len(idx)} of them")
    result = behavior(tokenizer, model, device, chosen, [base[i] for i in idx], cfg, variant_builder(cfg))
    out_dir = resolve_out_dir("coherence", cfg, args.out)
    summary = {"model": cfg.model, "thinking": cfg.thinking, "n_cached": len(base), "n": len(idx),
               "sample_uids": [r["sample_uid"] for r in chosen],
               "regimes": dict(Counter(r["regime"] for r in chosen)), **result}
    (out_dir / "coherence.json").write_text(json.dumps(summary, indent=1))
    stamp(f"wrote {out_dir / 'coherence.json'}")


if __name__ == "__main__":
    main()
