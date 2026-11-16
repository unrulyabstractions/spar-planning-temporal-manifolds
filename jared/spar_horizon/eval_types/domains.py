"""Eval types: domains group (EvalsPromptTypes.md). See common.py for the contract.

Implicit horizons (an anchor phrase instead of a duration) and domain changes (city, public health,
coding agent) of the T0 two-option decision problem."""

import math
from dataclasses import dataclass
from itertools import combinations

from ..horizons import Horizon
from .common import (DURATIONS, FAR_MULTIPLIERS, GAP, MARGIN, NEAR_REWARDS, PLAN, EvalType, Item, Option,
                     alan_prompt, log_years, money, two_option_truth)

DAY_DELAYS = tuple(Horizon(v, "days") for v in (1, 2, 3))
IMPLICIT_MENU = tuple(sorted(DAY_DELAYS + DURATIONS, key=lambda h: h.years))


def delay_pairs(regime, lo, hi, menu):
    """(near, far) pairs from `menu` such that every horizon in [lo, hi] years is in `regime` and
    at least MARGIN decades from the nearer delay, with the delays GAP decades apart."""
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    pairs = []
    for near, far in combinations(sorted(menu, key=lambda h: h.years), 2):
        n, f = log_years(near), log_years(far)
        if f - n < GAP:
            continue
        ok = {"near": n + MARGIN <= log_lo and log_hi <= f - MARGIN,
              "both": f + MARGIN <= log_lo,
              "none": log_hi <= n - MARGIN}[regime]
        if ok:
            pairs.append((near, far))
    return pairs


def draw_delays_for_horizon(rng, regime, delay_menu, horizon_menu):
    """(near, far, horizon): the horizon from `horizon_menu`, the delays from `delay_menu`."""
    for _ in range(10_000):
        h = rng.choice(horizon_menu)
        pairs = delay_pairs(regime, h.years, h.years, delay_menu)
        if pairs:
            near, far = rng.choice(pairs)
            return near, far, h
    raise ValueError(f"no {regime} delays in the menu")


def valued_item(rng, regime, unit, near_values, multipliers, delays):
    """T0-like item: `near_values` and the far multiples in `unit`, delays = (near, far, horizon)."""
    near, far, h = delays
    v_near = rng.choice(near_values)
    v_far = v_near * rng.choice(multipliers)
    options = [Option(f"{money(v_near)} {unit} in {near}.", near.years, str(near), v_near, "near"),
               Option(f"{money(v_far)} {unit} in {far}.", far.years, str(far), v_far, "far")]
    return Item(options, regime, two_option_truth(regime), h.years, str(h), "within_horizon")


# ---- implicit horizons (EvalsPromptTypes.md sections 5 and 6) ----

@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    phrase: str                 # completes "the greatest benefit ..."
    horizon: Horizon            # the implied horizon as the explicit twin states it
    lo_years: float
    hi_years: float
    horizon_years: float


def concrete(anchor_id, phrase, horizon):
    return Anchor(anchor_id, phrase, horizon, 0.9 * horizon.years, 1.1 * horizon.years, horizon.years)


def vague(anchor_id, phrase, horizon, lo, hi):
    return Anchor(anchor_id, phrase, horizon, lo.years, hi.years, math.sqrt(lo.years * hi.years))


CONCRETE_ANCHORS = (
    concrete("baby_8wk", "before your baby is born. Your partner is 8 weeks pregnant.", Horizon(32, "weeks")),
    concrete("retire_60_65", "before you retire. You are 60 years old and will retire at 65.", Horizon(5, "years")),
    concrete("friday_monday", "by Friday. It is Monday today.", Horizon(4, "days")),
    concrete("university_8", "before your daughter starts university. She just turned 8.", Horizon(10, "years")),
    concrete("birthday_40", "before your 40th birthday. You just turned 30.", Horizon(10, "years")),
    concrete("newborn_adult", "before your newborn daughter becomes an adult. She was born yesterday.",
             Horizon(18, "years")),
    concrete("mortgage_30", "before your mortgage is paid off. You took out a 30-year mortgage today.",
             Horizon(30, "years")),
    concrete("passport_10", "before your passport expires. It was issued 7 years ago and is valid for 10 years.",
             Horizon(3, "years")),
)

VAGUE_ANCHORS = (
    vague("wedding_spring", "before your wedding next spring. It is autumn now.",
          Horizon(6, "months"), Horizon(4, "months"), Horizon(8, "months")),
    vague("toddler_school", "before your daughter starts school. She just turned two.",
          Horizon(3, "years"), Horizon(2, "years"), Horizon(4, "years")),
    vague("final_school_year", "before your son finishes high school. He just started his final year.",
          Horizon(9, "months"), Horizon(8, "months"), Horizon(10, "months")),
    vague("kids_grown", "before your children are grown up. They are in primary school.",
          Horizon(10, "years"), Horizon(8, "years"), Horizon(15, "years")),
    vague("first_job", "before you retire. You just started your first job after university.",
          Horizon(40, "years"), Horizon(35, "years"), Horizon(45, "years")),
    vague("week_end", "before the end of this week. It is Monday today.",
          Horizon(5, "days"), Horizon(4, "days"), Horizon(6, "days")),
)


def implicit_item(rng, i, regime, anchors, truth_rule):
    """Anchors take turns within each regime; an anchor with no delays that clear its whole
    interval in this regime (e.g. a 30-year anchor in `near`: no delay beyond 59 years) is skipped."""
    turn = PLAN[:i].count(regime)
    for step in range(len(anchors)):
        anchor = anchors[(turn + step) % len(anchors)]
        pairs = delay_pairs(regime, anchor.lo_years, anchor.hi_years, IMPLICIT_MENU)
        if pairs:
            break
    else:
        raise ValueError(f"no anchor is feasible in the {regime} regime")
    near, far = rng.choice(pairs)
    v_near = rng.choice(NEAR_REWARDS)
    v_far = v_near * rng.choice(FAR_MULTIPLIERS)
    options = [Option(f"{money(v_near)} dollars in {near}.", near.years, str(near), v_near, "near"),
               Option(f"{money(v_far)} dollars in {far}.", far.years, str(far), v_far, "far")]
    extra = {"anchor_id": anchor.anchor_id, "anchor_lo_years": anchor.lo_years,
             "anchor_hi_years": anchor.hi_years, "anchor_text": anchor.phrase}
    return Item(options, regime, two_option_truth(regime), anchor.horizon_years, str(anchor.horizon), truth_rule, extra)


def implicit_render(item, shown):
    return alan_prompt(shown, constraint=f"You must select the option that provides the greatest benefit "
                                         f"{item.extra['anchor_text']}")


def implicit_concrete_item(rng, i, regime):
    return implicit_item(rng, i, regime, CONCRETE_ANCHORS, "within_horizon")


def implicit_vague_item(rng, i, regime):
    return implicit_item(rng, i, regime, VAGUE_ANCHORS, "within_horizon_interval")


# ---- city (sections 11 and 12) ----

# Delays from a month up; horizons also from the weeks, so `none` has a horizon below the nearer delay.
CITY_DELAYS = DURATIONS[2:]

CITY_PROJECT_PAIRS = (
    ("Patch the city's potholes", "Rebuild the city's water mains"),
    ("Resurface the city's main roads", "Build a light-rail line"),
    ("Repair the roofs of the city's schools", "Build a new school"),
    ("Replace the street lights in the city center", "Build a new power plant for the city"),
    ("Clear the city's storm drains", "Build a flood barrier around the city"),
    ("Renovate the city library", "Build a new city hospital"),
)


def city_no_values_item(rng, i, regime):
    near, far, h = draw_delays_for_horizon(rng, regime, CITY_DELAYS, DURATIONS)
    small, big = CITY_PROJECT_PAIRS[rng.randrange(len(CITY_PROJECT_PAIRS))]
    options = [Option(f"{small}, finished in {near}.", near.years, str(near), None, "near"),
               Option(f"{big}, finished in {far}.", far.years, str(far), None, "far")]
    truth = (0,) if regime == "near" else ()
    return Item(options, regime, truth, h.years, str(h), "within_horizon_near_only")


def city_no_values_render(item, shown):
    return alan_prompt(shown, situation="Plan for the future of the city.",
                       task="You, the city administration, are tasked to choose the best project:",
                       horizon_text=item.horizon_text)


# (task noun, near project, far project, people word)
CITY_VALUE_PAIRS = (
    ("transport", "pedestrian bridge", "light-rail line", "commuters"),
    ("transport", "bus lane", "metro line", "riders"),
    ("transport", "cycle path", "tram line", "commuters"),
    ("recreation", "community swimming pool", "sports stadium", "visitors"),
    ("water", "public drinking fountain network", "water treatment plant", "residents"),
)
CITY_COSTS = (1_000_000, 2_000_000, 5_000_000)
CITY_COST_MULTIPLIERS = (2, 3, 4)
CITY_USERS = (1_000, 2_000, 3_000, 5_000)
# At least 2.3 covers users x (horizon - opening) in `both`; more than the cost ratio covers users per dollar.
CITY_USER_MULTIPLIERS = (8, 10, 12)


def city_values_item(rng, i, regime):
    near, far, h = draw_delays_for_horizon(rng, regime, CITY_DELAYS, DURATIONS)
    domain, small, big, people = rng.choice(CITY_VALUE_PAIRS)
    cost_near = rng.choice(CITY_COSTS)
    cost_far = cost_near * rng.choice(CITY_COST_MULTIPLIERS)
    users_near = rng.choice(CITY_USERS)
    users_far = users_near * rng.choice(CITY_USER_MULTIPLIERS)

    def line(project, cost, d, users):
        return f"A {project} that costs {money(cost)} dollars, opens in {d} and serves {money(users)} {people} a day."

    options = [Option(line(small, cost_near, near, users_near), near.years, str(near), users_near, "near"),
               Option(line(big, cost_far, far, users_far), far.years, str(far), users_far, "far")]
    return Item(options, regime, two_option_truth(regime), h.years, str(h), "within_horizon", {"city_domain": domain})


def city_values_render(item, shown):
    return alan_prompt(shown, situation="Plan for the future of the city.",
                       task=f"You, the city administration, are tasked to choose the best {item.extra['city_domain']} "
                            f"project:",
                       horizon_text=item.horizon_text)


# ---- public health (section 16) ----

HEALTH_NEAR_VALUES = (50, 100, 200, 500, 1_000)


def public_health_item(rng, i, regime):
    return valued_item(rng, regime, "hospital admissions prevented", HEALTH_NEAR_VALUES, (2, 3, 4, 5),
                       draw_delays_for_horizon(rng, regime, DURATIONS, DURATIONS))


def public_health_render(item, shown):
    return alan_prompt(shown, situation="Plan the public health program of the region.",
                       task="You, the regional health authority, are tasked to choose the best health program:",
                       horizon_text=item.horizon_text)


# ---- coding agent (section 19) ----

CODING_DELAYS = tuple(Horizon(v, u) for v, u in [
    (1, "days"), (2, "days"), (3, "days"), (1, "weeks"), (2, "weeks"), (1, "months"), (2, "months"),
    (3, "months"), (6, "months"), (1, "years"), (2, "years")])
# Hours below the shortest delay give the `none` regime a horizon.
CODING_HORIZONS = tuple(Horizon(v, "hours") for v in (1, 6, 12)) + DURATIONS
CODING_NEAR_VALUES = (10, 20, 50, 100, 200, 500)


def coding_agent_item(rng, i, regime):
    return valued_item(rng, regime, "engineer-hours saved", CODING_NEAR_VALUES, (2, 3, 4),
                       draw_delays_for_horizon(rng, regime, CODING_DELAYS, CODING_HORIZONS))


def coding_agent_render(item, shown):
    return alan_prompt(shown, situation="Maintain the company's web service.",
                       task="You, the coding agent responsible for the service, are tasked to choose the best "
                            "engineering project:",
                       horizon_text=item.horizon_text)


TYPES = [
    EvalType("implicit_concrete", implicit_concrete_item, implicit_render),
    EvalType("implicit_vague", implicit_vague_item, implicit_render),
    EvalType("city_no_values", city_no_values_item, city_no_values_render),
    EvalType("city_values", city_values_item, city_values_render),
    EvalType("public_health", public_health_item, public_health_render),
    EvalType("coding_agent", coding_agent_item, coding_agent_render),
]
