"""Which prompt bank a run uses, from `Settings.bank`."""

from .full_bank import (CHOICE_BANKS, CHOICE_PER_DURATION, build_choice_prompts, build_fill_prompts, build_full_prompts,
                        build_units_prompts, choice_variant, render_variant)
from .prompts import build_prompts, choice_prompt


def build_records(cfg):
    """The prompt records for `cfg`: the starter, full, units, fill or choice bank, with
    `cfg.limit` applied."""
    if cfg.bank == "full":
        return build_full_prompts(seed=cfg.seed, n=cfg.n_prompts, limit=cfg.limit)
    if cfg.bank == "units":
        return build_units_prompts(seed=cfg.seed, limit=cfg.limit)
    if cfg.bank == "fill":
        return build_fill_prompts(seed=cfg.seed, limit=cfg.limit)
    if cfg.bank in CHOICE_BANKS:
        k, roi_explicit = CHOICE_BANKS[cfg.bank]
        return build_choice_prompts(k, roi_explicit, cfg.seed, CHOICE_PER_DURATION, cfg.limit)
    return build_prompts(seed=cfg.seed, limit=cfg.limit)


def starter_variant(rec, **kw):
    return choice_prompt(rec["rewards"], rec["delays"], rec["horizon"], **kw)


def variant_builder(cfg):
    """`prompt_fn(rec, swap=, labels=)` for the behavior tests on `cfg`'s bank."""
    if cfg.bank == "starter":
        return starter_variant
    return choice_variant if cfg.bank in CHOICE_BANKS else render_variant
