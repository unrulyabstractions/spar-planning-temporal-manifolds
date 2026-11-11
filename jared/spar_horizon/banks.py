"""Which prompt bank a run uses, from `Settings.bank`."""

from .config import Settings
from .full_bank import (CHOICE_BANKS, CHOICE_PER_DURATION, build_choice_prompts, build_fill_prompts, build_full_prompts,
                        build_units_prompts, choice_variant, render_variant)
from .implied_bank import build_implied_prompts, implied_variant
from .prompts import build_prompts, choice_prompt


def build_records(cfg):
    """The prompt records for `cfg`: the starter, full, units, fill, implied or choice bank, with
    `cfg.limit` applied."""
    if cfg.bank == "full":
        return build_full_prompts(seed=cfg.seed, n=cfg.n_prompts, limit=cfg.limit)
    if cfg.bank == "units":
        return build_units_prompts(seed=cfg.seed, limit=cfg.limit)
    if cfg.bank == "fill":
        return build_fill_prompts(seed=cfg.seed, limit=cfg.limit)
    if cfg.bank == "implied":
        return build_implied_prompts(seed=cfg.seed, limit=cfg.limit)
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
    if cfg.bank == "implied":
        return implied_variant
    return choice_variant if cfg.bank in CHOICE_BANKS else render_variant


def selected_records(meta):
    """The inputs' prompt records, in their order. Pooled inputs name each record's
    bank and the settings its run was built with; older inputs are all full bank."""
    if "bank_settings" in meta:
        banks, settings = meta["selected_banks"], meta["bank_settings"]
    else:
        banks = ["full"] * len(meta["selected"])
        settings = {"full": {k: meta[k] for k in ("n_prompts", "seed", "limit")}}
    built = {bank: build_records(Settings(bank=bank, **settings[bank])) for bank in set(banks)}
    records = [built[bank][i] for bank, i in zip(banks, meta["selected"])]
    if [r["sample_uid"] for r in records] != meta["sample_uids"]:
        raise ValueError("the prompt bank differs from the one the inputs were prepared on")
    return records
