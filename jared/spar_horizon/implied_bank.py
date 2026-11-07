"""The implied-horizon bank (29_spec_implied_horizon_side.md): the constraint line names an anchor
("until the lease expires") instead of a duration, rendered with an explicit twin that names the
anchor's log-midpoint, and once per anchor without any constraint line (the null twin).

Families: A end points computed from stated dates, B world-knowledge anchors with no number in the
prompt, C events whose duration is stated elsewhere in the prompt, D vague quantifiers (no truth).
Each non-D anchor has an interval [lo, hi] in years. The delays come from the crossed generator's
machinery with the delay nearest the interval at least IMPLIED_MARGIN[0] decades outside it, so every
reading inside the interval gives the same regime and truth (`near` -> near, `both` -> far; `none`
has no truth). Deterministic in `seed`.

Records carry the usual 2-option keys (`record` in full_bank) plus family, anchor_id, anchor_lo/hi,
rendering (implied, explicit, null, vague), twin_id (shared by the renderings of one base), module
(M1 A and C near/both; M2 B near/both; M3 none regime; M4 vague), us_convention (B anchors that
assume US school and retirement ages), context (the sentence added to SITUATION, "" if none) and
explicit_text (the twin's horizon). They are emitted in run order, group by group (a group is the
renderings of one base, contiguous), so any prefix of whole groups is balanced.
"""

import hashlib
import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .full_bank import (DELAY_LIMITS, GAP, MAX_DRAWS, REWARD_MULTIPLIERS, SHORT_REWARDS, TRUTH, DOMAINS,
                        PromptFormat, crossed_renderings, delays_for, evenly_spaced_bases, natural_horizon,
                        realism_labels, regime, render_variant)
from .horizons import DAYS_PER_YEAR, Horizon

IMPLIED_MARGIN = (0.3, 1.5)        # decades from the interval to the nearer delay (the crossed bank uses 0.2)
FAMILIES = ("A", "B", "C", "D")
MODULES = ("M1", "M2", "M3", "M4")
BASES_NEAR_BOTH = 4                # per anchor: two near and two both
VAGUE_BASES = 3
VAGUE_LOG_REFERENCE = (-3.0, 2.0)  # log10 years from which a vague base's delays are placed
DATE_TOLERANCE = 0.10              # interval half-width around a date span
STATED_TOLERANCE = 0.05            # ... around a duration the prompt states


@dataclass(frozen=True)
class Anchor:
    family: str
    anchor_id: str
    phrase: str                    # after "for this time horizon: "
    context: str                   # sentence added to SITUATION, "" if none
    lo: Optional[float]            # years; None for the vague family
    hi: Optional[float]
    us_convention: bool = False

    @property
    def log_mid(self):
        return (math.log10(self.lo) + math.log10(self.hi)) / 2


def date_text(d):
    return f"{d.day} {d:%B %Y}"


def clock_text(d):
    return f"{d.hour % 12 or 12}:{d.minute:02d} {'am' if d.hour < 12 else 'pm'}"


def span_years(now, end):
    return (end - now).total_seconds() / (DAYS_PER_YEAR * 86400)


def around(years, tolerance):
    return years * (1 - tolerance), years * (1 + tolerance)


def computed_anchor(anchor_id, now, end, context, phrase, text):
    """Family A: the prompt states `now` and the end point; the interval is their exact span."""
    lo, hi = around(span_years(now, end), DATE_TOLERANCE)
    fields = {"now": text(now), "end": text(end)}
    return Anchor("A", anchor_id, phrase.format(**fields), context.format(**fields), lo, hi)


def stated_anchor(anchor_id, duration, context, phrase):
    """Family C: the duration appears in `context`, not in the constraint phrase."""
    lo, hi = around(duration.years, STATED_TOLERANCE)
    return Anchor("C", anchor_id, phrase, context.format(duration=duration), lo, hi)


def world_anchor(anchor_id, phrase, lo, hi, us_convention=False):
    return Anchor("B", anchor_id, phrase, "", lo, hi, us_convention)


ANCHORS: tuple[Anchor, ...] = (
    computed_anchor("A1", datetime(2026, 3, 2, 9, 15), datetime(2026, 3, 2, 16, 30), "It is {now}.",
                    "until {end} today", clock_text),
    computed_anchor("A2", datetime(2026, 3, 2, 8, 0), datetime(2026, 3, 6, 18, 0), "It is {now} on Monday.",
                    "until {end} on Friday", clock_text),
    computed_anchor("A3", datetime(2026, 2, 1), datetime(2026, 6, 30), "Today is {now}.",
                    "until the end of the fiscal year, which closes on {end}", date_text),
    computed_anchor("A4", datetime(2026, 3, 12), datetime(2026, 9, 12), "Today is {now}.", "until {end}", date_text),
    computed_anchor("A5", datetime(2026, 2, 14), datetime(2028, 11, 5), "Today is {now}.",
                    "until the next US presidential election ({end})", date_text),
    computed_anchor("A6", datetime(2026, 1, 3), datetime(2035, 12, 31), "Today is {now}.", "until {end}", date_text),
    computed_anchor("A7", datetime(2026, 7, 1), datetime(2076, 7, 1), "Today is {now}.",
                    "until the family trust matures on {end}", date_text),
    computed_anchor("A8", datetime(2026, 3, 20), datetime(2126, 3, 20), "Today is {now}.",
                    "until the endowment matures on {end}", date_text),
    world_anchor("B1", "over the coming fortnight", 12 / DAYS_PER_YEAR, 16 / DAYS_PER_YEAR),
    world_anchor("B2", "over the coming semester", 0.28, 0.40),
    world_anchor("B3", "over a generation", 20, 30),
    world_anchor("B4", "over the coming weekend", 1.5 / DAYS_PER_YEAR, 3 / DAYS_PER_YEAR),
    world_anchor("B5", "before your newborn daughter starts kindergarten", 4, 6, us_convention=True),
    world_anchor("B6", "by the time your newborn grandchild retires", 60, 70, us_convention=True),
    world_anchor("B7", "until your newborn son finishes high school", 17, 19, us_convention=True),
    world_anchor("B8", "until your newborn's first birthday", 0.95, 1.05),
    stated_anchor("C1", Horizon(18, "months"), "The household's lease, signed today, runs for {duration}.",
                  "until the lease expires"),
    stated_anchor("C2", Horizon(3, "months"),
                  "The household's investment committee reviews the portfolio every {duration} and met today.",
                  "before the next committee review"),
    stated_anchor("C3", Horizon(5, "years"), "The household's car loan, taken out today, has a term of {duration}.",
                  "until the loan is repaid"),
    stated_anchor("C4", Horizon(25, "years"), "The household signed a ground lease today for {duration}.",
                  "until the ground lease expires"),
    stated_anchor("C5", Horizon(60, "days"), "The household's mortgage rate lock, set today, holds for {duration}.",
                  "until the rate lock expires"),
    stated_anchor("C6", Horizon(6, "weeks"),
                  "The household filed its tax return today, and the refund office quotes {duration} to process it.",
                  "until the refund is processed"),
    stated_anchor("C7", Horizon(3, "years"), "The apprenticeship starting today for the household's son lasts {duration}.",
                  "until the apprenticeship ends"),
    stated_anchor("C8", Horizon(10, "days"), "The household's trial subscription, started today, runs for {duration}.",
                  "until the trial ends"),
    *(Anchor("D", f"D{i + 1}", phrase, "", None, None) for i, phrase in enumerate((
        "in the short term", "in the long run", "very soon", "for the foreseeable future",
        "in the medium term", "for the near future", "over the long term", "right away"))),
)
ANCHOR_BY_ID = {a.anchor_id: a for a in ANCHORS}


def family_anchors(family):
    return [a for a in ANCHORS if a.family == family]


def base_regimes(index, anchor):
    """(slot, regime) of an anchor's bases: near and both alternate (starting with the other one at
    the next anchor), then `none` last for the three interval families. Vague bases cycle all three."""
    if anchor.family == "D":
        return [(b, ("near", "both", "none")[(b + index) % 3]) for b in range(VAGUE_BASES)]
    return [(b, ("near", "both")[(b + index) % 2]) for b in range(BASES_NEAR_BOTH)] + [(BASES_NEAR_BOTH, "none")]


def feasible(lo_log, hi_log, regime_name):
    """Whether delays at the minimum margins (and, for two delays on one side, the minimum gap) fit
    inside DELAY_LIMITS."""
    low, high = (math.log10(y) for y in DELAY_LIMITS)
    m = IMPLIED_MARGIN[0]
    if regime_name == "near":
        return lo_log - m >= low and hi_log + m <= high
    if regime_name == "both":
        return lo_log - m - GAP[0] >= low
    return hi_log + m + GAP[0] <= high


def delay_logs(rng, lo_log, hi_log, regime_name):
    """(near, far) delays in log10 years, clear of [lo, hi] by drawn margins."""
    margin = rng.uniform(*IMPLIED_MARGIN)
    if regime_name == "near":
        return lo_log - margin, hi_log + rng.uniform(*IMPLIED_MARGIN)
    if regime_name == "both":
        return delays_for(lo_log, "both", margin, 0.0, rng.uniform(*GAP))
    return delays_for(hi_log, "none", margin, 0.0, rng.uniform(*GAP))


def clear_of_interval(near, far, lo_log, hi_log, regime_name):
    """The rendered delays keep the margin: rounding to a rendering must not eat into it."""
    m = IMPLIED_MARGIN[0] - 1e-9
    n, f = math.log10(near.years), math.log10(far.years)
    if regime_name == "near":
        return n <= lo_log - m and f >= hi_log + m
    if regime_name == "both":
        return f <= lo_log - m
    return n >= hi_log + m


def draw_implied_delays(rng, lo_log, hi_log, regime_name):
    """Rendered (near, far) delays for `regime_name`, redrawn until both lie within the limits and
    the renderings keep the margin from the interval."""
    low, high = (math.log10(y) for y in DELAY_LIMITS)
    for _ in range(MAX_DRAWS):
        near_log, far_log = delay_logs(rng, lo_log, hi_log, regime_name)
        if not (low <= near_log and far_log <= high):
            continue
        near, far = (rng.choice(crossed_renderings(10 ** d)) for d in (near_log, far_log))
        if DELAY_LIMITS[0] <= near.years and far.years <= DELAY_LIMITS[1] \
                and clear_of_interval(near, far, lo_log, hi_log, regime_name):
            return near, far
    raise ValueError(f"no {regime_name} delays for log interval [{lo_log:.2f}, {hi_log:.2f}] in {MAX_DRAWS} draws")


def with_context(text, context):
    """`text` (a rendered prompt) with `context` appended to the SITUATION line."""
    if not context:
        return text
    situation = f"{PromptFormat().situation_marker} {DOMAINS['investment'].situation}"
    return text.replace(situation, f"{situation} {context}", 1)


def implied_variant(rec, swap=False, labels=("a", "b")):
    """The record's prompt with the options swapped and/or relabeled (`prompt_fn` of the behavior tests)."""
    return with_context(render_variant(rec, swap=swap, labels=labels), rec["context"])


def base_record(anchor, regime_name, rendering, phrase, base_id, drawn):
    near, far, short_reward, long_reward, short_first = drawn
    mid_years = None if anchor.family == "D" else 10 ** anchor.log_mid
    horizon = mid_years if rendering in ("implied", "explicit") and mid_years else None
    reg = regime(horizon, near.years, far.years) if horizon else ("vague" if rendering == "vague" else "null")
    rec = {"rewards": (int(short_reward), int(long_reward)), "delays": (str(near), str(far)), "horizon": horizon,
           "near_delay_years": near.years, "far_delay_years": far.years,
           "near_label": "a" if short_first else "b", "regime": reg, "truth": TRUTH.get(reg),
           "sample_uid": f"{base_id}_{rendering}", "domain": "investment", "horizon_text": phrase,
           "short_first": short_first, "phrasing_id": 0, "context": anchor.context}
    rec["prompt"] = implied_variant(rec)
    rec["prompt_id"] = hashlib.sha1(rec["prompt"].encode()).hexdigest()[:12]
    rec.update(family=anchor.family, anchor_id=anchor.anchor_id, anchor_lo=anchor.lo, anchor_hi=anchor.hi,
               rendering=rendering, twin_id=base_id, pair_id=base_id, duration_id=anchor.anchor_id,
               us_convention=anchor.us_convention, base_regime=regime_name, log_h=math.log10(horizon) if horizon else None,
               **realism_labels(near.years, far.years))
    return rec


def base_group(rng, anchor, regime_name, base_id, with_null):
    """The renderings of one base: implied, explicit twin, and the null twin when `with_null`
    (vague anchors: the single vague rendering)."""
    if anchor.family == "D":
        reference = rng.uniform(*VAGUE_LOG_REFERENCE)
        lo_log = hi_log = reference
    else:
        lo_log, hi_log = math.log10(anchor.lo), math.log10(anchor.hi)
    near, far = draw_implied_delays(rng, lo_log, hi_log, regime_name)
    short_reward = rng.choice(SHORT_REWARDS)
    drawn = (near, far, short_reward, short_reward * rng.choice(REWARD_MULTIPLIERS), rng.random() < 0.5)
    if anchor.family == "D":
        return [base_record(anchor, regime_name, "vague", anchor.phrase, base_id, drawn)]
    explicit = str(natural_horizon(10 ** anchor.log_mid))
    group = [base_record(anchor, regime_name, "implied", anchor.phrase, base_id, drawn),
             base_record(anchor, regime_name, "explicit", explicit, base_id, drawn)]
    if with_null:
        group.append(base_record(anchor, regime_name, "null", None, base_id, drawn))
    for rec in group:
        rec["explicit_text"] = explicit
        if rec["horizon"] is not None:
            assert rec["regime"] == regime_name, (rec["sample_uid"], rec["regime"], regime_name)
    return group


def module_of(anchor, regime_name):
    if anchor.family == "D":
        return "M4"
    if regime_name == "none":
        return "M3"
    return "M2" if anchor.family == "B" else "M1"


def all_groups(seed):
    """{(anchor_id, slot): group}, drawn in the anchors' canonical order from the bank's own RNG."""
    rng = random.Random(f"implied-{seed}")
    groups = {}
    for index, anchor in enumerate(ANCHORS):
        in_family = family_anchors(anchor.family).index(anchor)
        for slot, regime_name in base_regimes(in_family, anchor):
            if anchor.family != "D" and not feasible(math.log10(anchor.lo), math.log10(anchor.hi), regime_name):
                continue
            base_id = f"i{seed}_{anchor.anchor_id}_{slot}"
            group = base_group(rng, anchor, regime_name, base_id, with_null=slot == 0)
            for rec in group:
                rec["module"] = module_of(anchor, regime_name)
                rec["slot"] = slot
            groups[(anchor.anchor_id, slot)] = group
    return groups


def run_order(groups):
    """Groups in run order: M1, M2, M3, M4; within a module round by round (slot), and within a round
    interleaving the anchors, so every prefix covers the anchors and both regimes evenly."""
    a, b, c, d = (family_anchors(f) for f in FAMILIES)
    plan = [(x.anchor_id, slot) for slot in range(BASES_NEAR_BOTH) for pair in zip(a, c) for x in pair]
    plan += [(x.anchor_id, slot) for slot in range(BASES_NEAR_BOTH) for x in b]
    plan += [(x.anchor_id, BASES_NEAR_BOTH) for trio in zip(a, b, c) for x in trio]
    plan += [(x.anchor_id, slot) for slot in range(VAGUE_BASES) for x in d]
    return [groups[key] for key in plan if key in groups]


def build_implied_groups(seed=0, limit=None):
    """The bank as groups (lists of records, one per base). `limit` keeps evenly spaced whole groups,
    about `limit` records."""
    return evenly_spaced_bases(run_order(all_groups(seed)), limit)


def build_implied_prompts(seed=0, limit=None):
    """The implied bank as records in run order (288 with every regime feasible)."""
    return [rec for group in build_implied_groups(seed, limit) for rec in group]
