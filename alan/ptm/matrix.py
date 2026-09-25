"""Matched prompt matrix: scenario × horizon × rendering, plus separate control sets.

Main design (step 2 of the 2026-09-23 plan): every option configuration appears in both domains,
at every horizon of a 12-point grid, in three meaning-preserving renderings. Splits are assigned at
the configuration level so all renderings, horizons and domains of a held-out decision leave
training together. Four interior horizons are withheld for interpolation tests.

Control sets are generated separately (never crossed into the main matrix):
  mention   – an irrelevant duration is mentioned; the label is still the stated horizon
  role      – a duration in a different temporal role ("you have 2 hours to prepare")
  factual   – the same quantities with no decision task and no response format
  scaling   – horizon and both option delays multiplied by 0.1 / 1 / 10 (ratios preserved)
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Optional

from .horizons import UNIT_YEARS, Horizon
from .prompts import (LONG_DELAYS, REWARD_MULTIPLIERS, SHORT_DELAYS, SHORT_REWARDS, DOMAINS, PromptFormat,
                      PromptSample, _fmt_reward)

TRAIN_HORIZONS = tuple(Horizon.parse(t) for t in ["1 day", "1 week", "1 month", "6 months", "2 years", "10 years", "30 years", "100 years"])
HELDOUT_HORIZONS = tuple(Horizon.parse(t) for t in ["2 weeks", "3 months", "5 years", "20 years"])
MATRIX_HORIZONS = tuple(sorted(TRAIN_HORIZONS + HELDOUT_HORIZONS, key=lambda h: h.years))
RENDERINGS = ("structured", "plain", "varied")


@dataclass(frozen=True)
class DomainProse:
    role: str            # "the head of a household planning its finances"
    role_np: str         # "the household"
    situation: str       # one sentence of context
    entity: str          # for the mention control: "the household's savings account"
    thing: str           # "investments" / "emissions programs"
    verb: str            # "yields" / "prevents"


PROSE: dict[str, DomainProse] = {
    "investment": DomainProse(role="the head of a household planning its finances", role_np="the household",
                              situation="The household is deciding how to invest.", entity="the household's savings account",
                              thing="investments", verb="pays out"),
    "climate": DomainProse(role="the administration of a city planning its climate policy", role_np="the city",
                           situation="The city is deciding which emissions program to fund.", entity="the city's climate office",
                           thing="emissions programs", verb="delivers"),
}


@dataclass(frozen=True)
class OptionConfig:
    config_id: str
    short_reward: float
    short_delay: Horizon
    long_reward: float
    long_delay: Horizon
    short_first: bool


def sample_configs(n: int, seed: int) -> list[OptionConfig]:
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        sd, ld = rng.choice(SHORT_DELAYS), rng.choice(LONG_DELAYS)
        sr = rng.choice(SHORT_REWARDS); lr = sr * rng.choice(REWARD_MULTIPLIERS)
        key = (sd, ld, sr, lr)
        if key in seen:
            continue
        seen.add(key)
        out.append(OptionConfig(config_id=f"c{seed}_{len(out):03d}", short_reward=float(sr), short_delay=sd,
                                long_reward=float(lr), long_delay=ld, short_first=len(out) % 2 == 0))
    return out


def assign_splits(configs: list[OptionConfig], n_test: int, n_dev: int, seed: int) -> dict[str, str]:
    ids = [c.config_id for c in configs]
    random.Random(seed + 1).shuffle(ids)
    split = {}
    for i, cid in enumerate(ids):
        split[cid] = "test" if i < n_test else ("dev" if i < n_test + n_dev else "train")
    return split


# ------------------------------------------------------------------ renderings

def _options_prose(s: PromptSample, style: int) -> str:
    unit = DOMAINS[s.domain].reward_unit
    first, second = (s.short_reward, s.short_delay_text), (s.long_reward, s.long_delay_text)
    if not s.short_first:
        first, second = second, first
    if style == 0:
        return f"{s.label_a} {_fmt_reward(first[0])} {unit} in {first[1]}, or {s.label_b} {_fmt_reward(second[0])} {unit} in {second[1]}"
    verb = PROSE[s.domain].verb
    return (f"option {s.label_a} {verb} {_fmt_reward(first[0])} {unit} after {first[1]}, while option {s.label_b} "
            f"{verb} {_fmt_reward(second[0])} {unit} after {second[1]}")


def _horizon_prose(h: str, style: int) -> str:
    if style == 0:
        return f"Your time horizon for this decision is {h}: choose the option that provides the greatest benefit within it."
    return f"Choose the option with the greatest benefit over a time horizon of {h}."


FORMAT_PROSE = "Respond in this format: I choose: <{a} or {b}>. My reasoning: <1-3 sentences>"


def render_plain(s: PromptSample) -> str:
    d = PROSE[s.domain]
    return (f"You are {d.role}. {d.situation} Two options are available: {_options_prose(s, 0)}. "
            f"{_horizon_prose(s.horizon_text, 0)} Select one of the two options and give your reasoning. "
            + FORMAT_PROSE.format(a=s.label_a, b=s.label_b))


def render_varied(s: PromptSample, variant: int) -> str:
    """8 variants = 4 sentence orders × 2 option phrasings; horizon wording follows the option phrasing."""
    d = PROSE[s.domain]
    order, style = variant % 4, variant // 4
    role = f"You are {d.role}."
    sit = d.situation
    opts = f"Two options are available: {_options_prose(s, style)}."
    opts_first = f"Two {d.thing} are on the table for {d.role_np}: {_options_prose(s, style)}."
    hor = _horizon_prose(s.horizon_text, style)
    task = "Select one of the two options and give your reasoning."
    fmt = FORMAT_PROSE.format(a=s.label_a, b=s.label_b)
    # none of the four orders equals the plain rendering [role, sit, opts, hor, task, fmt]
    if order == 0:
        parts = [role, sit, hor, opts, task, fmt]
    elif order == 1:
        parts = [hor, role, sit, opts, task, fmt]
    elif order == 2:
        parts = [opts_first, sit, hor, task, fmt]
    else:
        parts = [role, sit, opts, task, hor, fmt]
    return " ".join(parts)


def render(s: PromptSample, rendering: str, fmt: PromptFormat, variant: int = 0) -> str:
    if rendering == "structured":
        return fmt.render(s)
    if rendering == "plain":
        return render_plain(s)
    if rendering == "varied":
        return render_varied(s, variant)
    raise ValueError(rendering)


# ------------------------------------------------------------------ builders

def _base_sample(uid: str, domain: str, c: OptionConfig, h: Optional[Horizon], fmt: PromptFormat) -> PromptSample:
    return PromptSample(
        sample_uid=uid, domain=domain,
        horizon_text=None if h is None else str(h), horizon_years=float("nan") if h is None else h.years,
        short_reward=c.short_reward, short_delay_text=str(c.short_delay), short_delay_years=c.short_delay.years,
        long_reward=c.long_reward, long_delay_text=str(c.long_delay), long_delay_years=c.long_delay.years,
        short_first=c.short_first, label_a="a)", label_b="b)", format_name=fmt.name,
        horizon_canonical=None if h is None else str(h), horizon_unit=None if h is None else h.unit,
        horizon_value=float("nan") if h is None else h.value, phrasing_id=0,
        config_id=c.config_id, scenario_id=f"{c.config_id}/{domain}",
    )


def _finish(s: PromptSample, text: str, rendering: str, condition: str, split: str) -> PromptSample:
    s.text = text; s.rendering = rendering; s.condition = condition; s.split = split
    s.prompt_id = hashlib.sha1(text.encode()).hexdigest()[:12]
    return s


@dataclass
class MatrixConfig:
    n_configs: int = 30
    n_test_configs: int = 6
    n_dev_configs: int = 4
    seed: int = 0
    domains: tuple[str, ...] = ("investment", "climate")
    renderings: tuple[str, ...] = RENDERINGS
    n_control_configs: int = 10          # controls use the first n training configs
    scaling_horizons: tuple[str, ...] = ("1 week", "1 month", "6 months", "2 years", "10 years", "30 years")
    scaling_factors: tuple[float, ...] = (0.1, 1.0, 10.0)


def build_matrix(cfg: MatrixConfig, fmt: Optional[PromptFormat] = None) -> list[PromptSample]:
    fmt = fmt or PromptFormat()
    configs = sample_configs(cfg.n_configs, cfg.seed)
    split = assign_splits(configs, cfg.n_test_configs, cfg.n_dev_configs, cfg.seed)
    out: list[PromptSample] = []
    for c in configs:
        for domain in cfg.domains:
            for hi, h in enumerate(MATRIX_HORIZONS):
                for rendering in cfg.renderings:
                    variant = int(hashlib.sha1(f"{c.config_id}|{domain}|{hi}".encode()).hexdigest(), 16) % 8
                    s = _base_sample(f"m{cfg.seed}_{c.config_id}_{domain}_h{hi:02d}_{rendering}", domain, c, h, fmt)
                    s.variant = variant if rendering == "varied" else 0
                    s.horizon_heldout = h in HELDOUT_HORIZONS
                    out.append(_finish(s, render(s, rendering, fmt, variant), rendering, "main", split[c.config_id]))
    return out


MENTION_DURATIONS = MATRIX_HORIZONS
ROLE_DURATIONS = tuple(Horizon.parse(t) for t in ["1 hour", "2 hours", "3 days", "1 week"])


def _mention_sentence(domain: str, d: Horizon) -> str:
    return f"{PROSE[domain].entity[0].upper() + PROSE[domain].entity[1:]} was established {d} ago."


def _role_sentence(d: Horizon) -> str:
    return f"You have {d} to prepare your recommendation."


def render_factual(s: PromptSample) -> str:
    d = PROSE[s.domain]; unit = DOMAINS[s.domain].reward_unit
    return (f"{d.role_np[0].upper() + d.role_np[1:]} holds two {d.thing}: one {d.verb} {_fmt_reward(s.short_reward)} {unit} after "
            f"{s.short_delay_text}, the other {d.verb} {_fmt_reward(s.long_reward)} {unit} after {s.long_delay_text}. "
            f"Its planning horizon is {s.horizon_text}.")


def scale_horizon(h: Horizon, factor: float, tol: float = 0.05) -> Horizon:
    """h × factor rendered as an integer in the LARGEST unit whose rounding error is within `tol`
    (value in [1, 9999]); falls back to the smallest-error unit if none is within tolerance."""
    target = h.years * factor
    best = None
    for unit in ["years", "months", "weeks", "days", "hours"]:      # the grid's own units; no decades/centuries (unit shifts geometry and behavior)
        v = target / UNIT_YEARS[unit]
        if 1 <= v <= 9999:
            r = round(v); err = abs(r * UNIT_YEARS[unit] / target - 1)
            if err <= tol:
                return Horizon(r, unit)
            if best is None or err < best[0]:
                best = (err, Horizon(r, unit))
    if best is None:
        raise ValueError(f"cannot render {target} years")
    return best[1]


def build_controls(cfg: MatrixConfig, fmt: Optional[PromptFormat] = None) -> list[PromptSample]:
    fmt = fmt or PromptFormat()
    configs = sample_configs(cfg.n_configs, cfg.seed)
    split = assign_splits(configs, cfg.n_test_configs, cfg.n_dev_configs, cfg.seed)
    train_cfgs = [c for c in configs if split[c.config_id] == "train"][: cfg.n_control_configs]
    rng = random.Random(cfg.seed + 7)
    out: list[PromptSample] = []
    for c in train_cfgs:
        for domain in cfg.domains:
            for hi, h in enumerate(MATRIX_HORIZONS):
                # mention without relevance
                d = rng.choice([x for x in MENTION_DURATIONS if x != h])
                s = _base_sample(f"k{cfg.seed}_{c.config_id}_{domain}_h{hi:02d}_mention", domain, c, h, fmt)
                s.distractor_text, s.distractor_years = str(d), d.years
                text = render_plain(s).replace(" Two options are available:", f" {_mention_sentence(domain, d)} Two options are available:")
                out.append(_finish(s, text, "plain", "mention", "control"))
                # different temporal role
                d = rng.choice(ROLE_DURATIONS)
                s = _base_sample(f"k{cfg.seed}_{c.config_id}_{domain}_h{hi:02d}_role", domain, c, h, fmt)
                s.distractor_text, s.distractor_years = str(d), d.years
                text = render_plain(s).replace(" Select one of the two options", f" {_role_sentence(d)} Select one of the two options")
                out.append(_finish(s, text, "plain", "role", "control"))
                # factual description, no task
                s = _base_sample(f"k{cfg.seed}_{c.config_id}_{domain}_h{hi:02d}_factual", domain, c, h, fmt)
                out.append(_finish(s, render_factual(s), "factual", "factual", "control"))
            # scaling set: horizon and both delays × factor
            for ht in cfg.scaling_horizons:
                h0 = Horizon.parse(ht)
                for f in cfg.scaling_factors:
                    h = scale_horizon(h0, f)
                    cs = OptionConfig(config_id=c.config_id, short_reward=c.short_reward, long_reward=c.long_reward,
                                      short_delay=scale_horizon(c.short_delay, f), long_delay=scale_horizon(c.long_delay, f),
                                      short_first=c.short_first)
                    s = _base_sample(f"k{cfg.seed}_{c.config_id}_{domain}_{ht.replace(' ', '')}_x{f:g}", domain, cs, h, fmt)
                    s.scale = f; s.scale_base_horizon = ht
                    out.append(_finish(s, render_plain(s), "plain", "scaling", "control"))
    return out
