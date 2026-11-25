"""Eval types: investment group (EvalsPromptTypes.md). See common.py for the contract."""

from ..horizons import Horizon
from .common import (
    DURATIONS, MARGIN, PLAN, EvalType, Item, Option, alan_prompt, draw_two_delays, log_years,
    max_value_within, money, standard_item, two_option_truth, two_rewards)
from .framing import expansion_index

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


def horizon_fits(regime, near, far):
    """Whether a horizon of y years is in `regime` and at least RATIO away from both delays (years)."""
    return {"near": lambda y: near * RATIO <= y <= far / RATIO,
            "both": lambda y: y >= far * RATIO,
            "none": lambda y: y <= near / RATIO}[regime]


# The expanded banks use these from item len(PLAN) on, one per item in turn: (unit of the option
# delays, their values, unit of the stated horizon, its values). The original bank only has delays in
# years against a horizon in months or weeks.
NEW_UNIT_PAIRS = (
    ("weeks", (2, 4, 6, 8, 12, 26, 52, 104), "days", (4, 6, 9, 12, 15, 20, 25, 40, 50, 75, 100, 150, 200, 300, 500, 800)),
    ("weeks", (2, 4, 6, 8, 12, 26, 52, 104), "months", (2, 3, 4, 5, 7, 9, 10, 14, 18, 30, 48)),
    ("weeks", (4, 8, 12, 26, 52, 104, 260), "years", (1, 2, 3, 4, 5, 6, 8, 12)),
    ("months", (1, 2, 3, 6, 9, 12, 24, 60), "days", (10, 20, 40, 50, 75, 100, 150, 200, 300, 500, 800, 1500, 2000)),
    ("months", (1, 2, 3, 6, 9, 12, 24, 60), "weeks", (3, 5, 6, 10, 14, 20, 30, 40, 70, 90, 130, 200, 300)),
    ("months", (3, 6, 9, 12, 24, 60, 120), "years", (1, 2, 3, 4, 5, 6, 8, 12, 14, 15, 25)),
    ("years", (1, 2, 3, 4, 7, 15, 25, 40), "days", (60, 100, 200, 300, 400, 500, 730, 1000, 1500, 2000, 3000, 5000, 8000)),
    ("years", (1, 2, 3, 4, 7, 15, 25, 40), "decades", (0.2, 0.3, 0.5, 0.6, 1.5, 2.5, 3.5, 4.5, 6, 7)),
    ("decades", (1, 2, 3, 5), "years", (2, 4, 5, 6, 8, 14, 16, 25, 35, 40, 60, 80, 150)),
    ("decades", (1, 2, 3, 5), "months", (24, 48, 60, 90, 150, 200, 300, 500, 700)),
    ("decades", (1, 2, 3, 5), "weeks", (100, 200, 300, 500, 800, 1500, 2000, 3500)),
    ("centuries", (1, 2, 3, 5), "years", (10, 25, 40, 60, 80, 150, 250, 350, 400, 700)),
    ("centuries", (1, 2, 3, 5), "decades", (1, 2, 3, 5, 7, 15, 25, 35, 70)),
    ("days", (7, 10, 14, 21, 30, 60, 90, 180, 365), "weeks", (1, 2, 3, 5, 6, 8, 12, 20, 40, 60)),
    ("days", (7, 10, 14, 21, 30, 60, 90, 180, 365), "months", (1, 2, 3, 5, 7, 9, 14, 18, 30)),
    ("days", (90, 180, 365, 730, 1095, 1825), "years", (1, 2, 3, 4, 5, 8)),
)


def unit_mismatch_item(rng, i, regime):
    if i >= len(PLAN):
        return new_unit_mismatch_item(rng, i, regime)
    for _ in range(10_000):
        near, far = sorted(rng.sample(DELAY_YEARS, 2))
        if far < 1.5 * near:
            continue
        fits = horizon_fits(regime, near, far)
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


def new_unit_mismatch_item(rng, i, regime):
    delay_unit, delay_values, horizon_unit, horizon_values = NEW_UNIT_PAIRS[expansion_index(len(NEW_UNIT_PAIRS), i)]
    delays = [Horizon(v, delay_unit) for v in delay_values]
    horizons = [Horizon(v, horizon_unit) for v in horizon_values]
    for _ in range(10_000):
        near, far = sorted(rng.sample(delays, 2), key=lambda d: d.years)
        if far.years < 1.5 * near.years:
            continue
        candidates = [h for h in horizons if horizon_fits(regime, near.years, far.years)(h.years)]
        if candidates:
            break
    else:
        raise ValueError(f"no unit-mismatch horizon for {regime} with {delay_unit} against {horizon_unit}")
    horizon = rng.choice(candidates)
    v_near, v_far = two_rewards(rng)
    options = [Option(f"{money(v_near)} dollars in {near}.", near.years, str(near), v_near, "near"),
               Option(f"{money(v_far)} dollars in {far}.", far.years, str(far), v_far, "far")]
    # The twin states the horizon in the options' unit, as the original items' twins do (their options are in years).
    in_delay_unit = Horizon(round(horizon.years / Horizon(1, delay_unit).years, 2), delay_unit)
    return Item(options, regime, two_option_truth(regime), horizon.years, str(in_delay_unit), "within_horizon",
                {"rendered_horizon_text": str(horizon), "unit_pair": f"{delay_unit}/{horizon_unit}"})


def unit_mismatch_render(item, shown):
    return alan_prompt(shown, horizon_text=item.extra["rendered_horizon_text"])


# 5. horizon_revised

STALE_REGIMES = {"near": ("both",), "both": ("near",), "none": ("near", "both")}


def stale_candidates(item, stale_regime):
    near, far = item.options[0].delay_years, item.options[1].delay_years
    fits = {"near": lambda x: near <= x < far, "both": lambda x: x >= far}[stale_regime]
    return [h for h in DURATIONS if fits(h.years) and all(abs(log_years(h) - log_years(d)) >= MARGIN for d in (near, far))]


UPDATE_LINE = "UPDATE: The household's plans have changed. The time horizon is now {h}."
NEW_UPDATE_LINES = (
    "UPDATE: Plans have changed since the last note. The new time horizon is {h}.",
    "UPDATE: The household has revised its plans. The time horizon is now {h}.",
    "UPDATE: Circumstances have changed for the household. The horizon to use is now {h}.",
    "UPDATE: The household's situation has shifted, and the time horizon is now {h}.",
    "UPDATE: Please disregard the earlier horizon. The time horizon is now {h}.",
    "UPDATE: The household has reconsidered its plans. The relevant time horizon is now {h}.",
    "UPDATE: The earlier horizon no longer applies. The household now plans for {h}.",
    "UPDATE: There has been a change of plans in the household. The time horizon is now {h}.",
    "UPDATE: The household's timetable has changed. The time horizon is now {h}.",
    "UPDATE: Following a family discussion, the time horizon is now {h}.",
    "UPDATE: The household's priorities have changed. The time horizon is now {h}.",
    "UPDATE: A change in circumstances means the household's time horizon is now {h}.",
    "UPDATE: The household has updated its plans. The time horizon to use is now {h}.",
    "UPDATE: The previous time horizon is out of date. The household's time horizon is now {h}.",
    "UPDATE: The household's plans have been revised, and the time horizon is now {h}.",
    "UPDATE: What the household needs has changed. The time horizon is now {h}.",
    "UPDATE: Things have changed for the household. The time horizon has been reset to {h}.",
)
UPDATE_LINES = (UPDATE_LINE, *NEW_UPDATE_LINES)


def horizon_revised_item(rng, i, regime):
    while True:
        item = standard_item(rng, regime)
        stale_regime = rng.choice(STALE_REGIMES[regime])
        candidates = stale_candidates(item, stale_regime)
        if candidates:
            break
    stale = rng.choice(candidates)
    item.extra = {"stale_horizon_text": str(stale), "stale_horizon_years": stale.years, "stale_regime": stale_regime}
    if i >= len(PLAN):
        item.extra["update_phrasing_id"] = 1 + expansion_index(len(NEW_UPDATE_LINES), i)
    return item


def update_line(item):
    return UPDATE_LINES[item.extra.get("update_phrasing_id", 0)].format(h=item.horizon_text)


def horizon_revised_render(item, shown):
    return alan_prompt(shown, horizon_text=item.extra["stale_horizon_text"], extra_lines=[update_line(item)])


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
