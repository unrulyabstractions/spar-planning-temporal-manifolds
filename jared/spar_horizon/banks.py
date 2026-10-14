"""Which prompt bank a run uses, from `Settings.bank`."""

from .full_bank import build_full_prompts, render_variant
from .prompts import build_prompts, choice_prompt


def build_records(cfg):
    """The prompt records for `cfg`: the starter bank or the full bank, with
    `cfg.limit` applied."""
    if cfg.bank == "full":
        return build_full_prompts(seed=cfg.seed, n=cfg.n_prompts, limit=cfg.limit)
    return build_prompts(seed=cfg.seed, limit=cfg.limit)


def starter_variant(rec, **kw):
    return choice_prompt(rec["rewards"], rec["delays"], rec["horizon"], **kw)


def variant_builder(cfg):
    """`prompt_fn(rec, swap=, labels=)` for the behavior tests on `cfg`'s bank."""
    return render_variant if cfg.bank == "full" else starter_variant
