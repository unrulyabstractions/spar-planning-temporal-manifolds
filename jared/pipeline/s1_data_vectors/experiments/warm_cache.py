"""Fill the generation cache for one shard of an E0 run's prompts, so the
generation can be split over several GPUs (10_spec_scale_14b.md section 3).
E0 reads the cache once when it starts, so start the shards together, wait for
all of them, then run E0 itself: it finds every prompt cached and only does
the forward passes.

    CUDA_VISIBLE_DEVICES=0 python -m pipeline.s1_data_vectors.experiments.warm_cache --model Qwen/Qwen3-14B --thinking on --bank full --shard 0/2
    CUDA_VISIBLE_DEVICES=1 python -m pipeline.s1_data_vectors.experiments.warm_cache --model Qwen/Qwen3-14B --thinking on --bank full --shard 1/2

Takes the same arguments as exp0_baseline.py, so the cache keys match.
"""

from pipeline.s1_data_vectors.experiments.common import parse_args
from spar_horizon.banks import build_records
from spar_horizon.model_io import cache_load, chat_text, generate_batch, load_model
from spar_horizon.runs import stamp


def shard_slice(items, shard, n_shards):
    """Contiguous part `shard` of `n_shards` near-equal parts of `items`."""
    start = len(items) * shard // n_shards
    stop = len(items) * (shard + 1) // n_shards
    return items[start:stop]


def uncached(tokenizer, prompts, cfg):
    """Prompts with no cached generation, in bank order."""
    ids = [tokenizer.encode(chat_text(tokenizer, p, cfg), add_special_tokens=False) for p in prompts]
    return [p for p, i in zip(prompts, ids) if cache_load(cfg, i, None, "cpu") is None]


def main():
    args, cfg = parse_args(__doc__, lambda p: p.add_argument("--shard", required=True, help='"k/n", e.g. 0/2'))
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    if not 0 <= shard < n_shards:
        raise ValueError(f"--shard {args.shard}: need 0 <= k < n")
    tokenizer, model, device = load_model(cfg)
    prompts = [r["prompt"] for r in build_records(cfg)]
    # every shard sees the same uncached list when they start together, so the parts are disjoint
    todo = uncached(tokenizer, prompts, cfg)
    mine = shard_slice(todo, shard, n_shards)
    stamp(f"shard {shard}/{n_shards}: {len(todo)} of {len(prompts)} prompts uncached, generating {len(mine)}")
    if mine:
        generate_batch(tokenizer, model, device, mine, cfg)
    stamp(f"shard {shard}/{n_shards}: done")


if __name__ == "__main__":
    main()
