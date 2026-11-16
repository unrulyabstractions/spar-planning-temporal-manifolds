"""Eval types: investment group (EvalsPromptTypes.md). See common.py for the contract."""

from ..horizons import Horizon
from .common import (
    DURATIONS, MARGIN, EvalType, Item, Option, alan_prompt, draw_two_delays, log_years,
    max_value_within, money, standard_item, two_option_truth, two_rewards)

PRINCIPALS = (5_000, 10_000, 20_000)

# 12 near, 4 both, 4 none, interleaved so a prefix is balanced.
MULTI_PLAN = ("near", "both", "near", "none") * 4 + ("near",) * 4


def near_index(plan, i):
    """How many near items come before slot i: cycles the number of options inside the horizon."""
    return plan[:i].count("near")


def horizon_clear_of(delays, menu=DURATIONS):
    return [h for h in menu if all(abs(log_years(h) - log_years(d)) >= MARGIN for d in delays)]


def draw_delays_and_horizon(rng, n_delays, n_inside):
    """Sorted delays (Horizons) and a horizon at least MARGIN decades from each, with exactly
    `n_inside` delays at or before the horizon."""
    for _ in range(100_000):
        delays = sorted(rng.sample(DURATIONS, n_delays), key=log_years)
        h = rng.choice(horizon_clear_of(delays) or [None])
        if h is not None and sum(d.years <= h.years for d in delays) == n_inside:
            return delays, h
    raise ValueError(f"no {n_delays} delays with {n_inside} inside the horizon")


def inside_count(regime, n_delays, k_near):
    return {"near": k_near, "both": n_delays, "none": 0}[regime]


# 1. roi_equal

def roi_equal_type(type_id, rate):
    """Payouts compound at `rate` per year, so every option is worth the same at the horizon if it
    pays inside it; truth is every option with delay <= horizon."""
    rate_text = f"{rate * 100:g}%"

    def make_item(rng, i, regime):
        k_near = 1 + near_index(MULTI_PLAN, i) % 3
        delays, h = draw_delays_and_horizon(rng, 4, inside_count(regime, 4, k_near))
        principal = rng.choice(PRINCIPALS)
        options = [Option(f"{money(payout)} dollars in {d}.", d.years, str(d), payout)
                   for d in delays for payout in [round(principal * (1 + rate) ** d.years)]]
        truth = tuple(j for j, o in enumerate(options) if o.delay_years <= h.years)
        return Item(options, regime, truth, h.years, str(h), "within_horizon_equal_roi",
                    {"principal": principal, "rate": rate, "n_inside": len(truth)})

    def render(item, shown):
        task = (f"You, the head of the household, have {money(item.extra['principal'])} dollars to invest and are "
                f"tasked to choose the best investment. Any payout can be reinvested at {rate_text} per year, "
                f"compounded yearly:")
        return alan_prompt(shown, task=task, horizon_text=item.horizon_text)

    return EvalType(type_id, make_item, render, plan=MULTI_PLAN, shuffle_base=False)


# 2. roi_unequal

def unequal_payouts(rng, delays, h):
    """Payouts that rise on average with delay but not monotonically, with a unique best option
    inside the horizon that is not the latest one inside (when two or more are inside)."""
    base = rng.choice(PRINCIPALS)
    inside = [j for j, d in enumerate(delays) if d.years <= h.years]
    for _ in range(10_000):
        values = [round(base * (1 + 0.12 * j) * rng.uniform(0.85, 1.15), -2) for j in range(len(delays))]
        if values == sorted(values):
            continue
        if inside:
            best = max(values[j] for j in inside)
            winners = [j for j in inside if values[j] == best]
            if len(winners) > 1 or (len(inside) >= 2 and winners[0] == inside[-1]):
                continue
        return values
    raise ValueError("no unequal payouts found")


def roi_unequal_item(rng, i, regime):
    k_near = 2 + near_index(MULTI_PLAN, i) % 5
    delays, h = draw_delays_and_horizon(rng, 8, inside_count(regime, 8, k_near))
    values = unequal_payouts(rng, delays, h)
    options = [Option(f"{money(v)} dollars in {d}.", d.years, str(d), v) for d, v in zip(delays, values)]
    return Item(options, regime, max_value_within(options, h.years), h.years, str(h), "max_value_within_horizon")


def roi_unequal_render(item, shown):
    return alan_prompt(shown, horizon_text=item.horizon_text)


# 3. free_form

FREE_FORM_PLAN = ("term",) * 20
FREE_FORM_HORIZONS = DURATIONS[1:13]    # 2 weeks .. 30 years


def free_form_item(rng, i, regime):
    h = FREE_FORM_HORIZONS[i] if i < len(FREE_FORM_HORIZONS) else rng.choice(FREE_FORM_HORIZONS)
    return Item([], regime, (), h.years, str(h), "term_equals_horizon",
                {"principal": rng.choice(PRINCIPALS), "truth_term_years": h.years})


def free_form_render(item, shown):
    task = (f"You, the head of the household, have {money(item.extra['principal'])} dollars to put in a fixed-term "
            f"deposit and are tasked to choose its term. The deposit pays 5% per year, compounded yearly, for any "
            f"term from 1 week to 30 years. The money cannot be withdrawn before the term ends.")
    return alan_prompt(shown, task=task, objective="Think deeply about which term is preferable.",
                       constraint=("You must select the term that provides the greatest benefit for this time horizon: "
                                   f"{item.horizon_text}."),
                       action="ACTION: Choose one term. Provide reasoning.",
                       fmt="FORMAT: I choose a term of: <number and unit>. My reasoning: <1-3 sentences>")


# 4. unit_mismatch

DELAY_YEARS = (1, 2, 3, 5, 10, 20, 30)
RATIO = 1.15    # the horizon is at least this factor away from each delay


def mismatched_horizon_menu():
    """(rendered text, years) of horizons in months or weeks that are not a whole number of years."""
    menu = [(str(Horizon(m, "months")), Horizon(m, "months").years) for m in range(3, 361, 3) if m % 12]
    menu += [(str(Horizon(w, "weeks")), Horizon(w, "weeks").years) for w in range(8, 1569, 4)]
    return menu


MISMATCHED_HORIZONS = mismatched_horizon_menu()


def unit_mismatch_item(rng, i, regime):
    for _ in range(10_000):
        near, far = sorted(rng.sample(DELAY_YEARS, 2))
        if far < 1.5 * near:
            continue
        fits = {"near": lambda y: near * RATIO <= y <= far / RATIO,
                "both": lambda y: y >= far * RATIO,
                "none": lambda y: y <= near / RATIO}[regime]
        candidates = [(text, y) for text, y in MISMATCHED_HORIZONS if fits(y)]
        if candidates:
            break
    else:
        raise ValueError(f"no unit-mismatch horizon for {regime}")
    rendered, years = rng.choice(candidates)
    years = round(years, 2)
    v_near, v_far = two_rewards(rng)
    options = [Option(f"{money(v_near)} dollars in {Horizon(near, 'years')}.", near, str(Horizon(near, "years")), v_near, "near"),
               Option(f"{money(v_far)} dollars in {Horizon(far, 'years')}.", far, str(Horizon(far, "years")), v_far, "far")]
    return Item(options, regime, two_option_truth(regime), years, f"{years:g} years", "within_horizon",
                {"rendered_horizon_text": rendered})


def unit_mismatch_render(item, shown):
    return alan_prompt(shown, horizon_text=item.extra["rendered_horizon_text"])


# 5. horizon_revised

STALE_REGIMES = {"near": ("both",), "both": ("near",), "none": ("near", "both")}


def stale_candidates(item, stale_regime):
    near, far = item.options[0].delay_years, item.options[1].delay_years
    fits = {"near": lambda x: near <= x < far, "both": lambda x: x >= far}[stale_regime]
    return [h for h in DURATIONS if fits(h.years) and all(abs(log_years(h) - log_years(d)) >= MARGIN for d in (near, far))]


def horizon_revised_item(rng, i, regime):
    while True:
        item = standard_item(rng, regime)
        stale_regime = rng.choice(STALE_REGIMES[regime])
        candidates = stale_candidates(item, stale_regime)
        if candidates:
            break
    stale = rng.choice(candidates)
    item.extra = {"stale_horizon_text": str(stale), "stale_horizon_years": stale.years, "stale_regime": stale_regime}
    return item


def horizon_revised_render(item, shown):
    update = f"UPDATE: The household's plans have changed. The time horizon is now {item.horizon_text}."
    return alan_prompt(shown, horizon_text=item.extra["stale_horizon_text"], extra_lines=[update])


# 6. loss_framing

def loss_framing_item(rng, i, regime):
    near, far, h = draw_two_delays(rng, regime)
    c_near, c_far = two_rewards(rng)
    options = [Option(f"Pay {money(c_near)} dollars in {near}.", near.years, str(near), c_near, "near"),
               Option(f"Pay {money(c_far)} dollars in {far}.", far.years, str(far), c_far, "far")]
    truth = {"near": (1,), "both": (0,), "none": ()}[regime]
    return Item(options, regime, truth, h.years, str(h), "min_cost_within_horizon")


def loss_framing_render(item, shown):
    constraint = (f"Only payments made within the next {item.horizon_text} count against your budget. "
                  f"Choose the option that costs the least.")
    return alan_prompt(shown, situation="Plan the budget of the household.",
                       task="You, the head of the household, are tasked to choose the best way to pay for a required purchase:",
                       constraint=constraint)


TYPES = [
    roi_equal_type("roi_equal_10", 0.10),
    EvalType("roi_unequal", roi_unequal_item, roi_unequal_render, plan=MULTI_PLAN, shuffle_base=False),
    EvalType("free_form", free_form_item, free_form_render, choice_prefix="I choose a term of:",
             answer_kind="duration", plan=FREE_FORM_PLAN, twins=False),
    EvalType("unit_mismatch", unit_mismatch_item, unit_mismatch_render),
    EvalType("horizon_revised", horizon_revised_item, horizon_revised_render),
    EvalType("loss_framing", loss_framing_item, loss_framing_render),
]
