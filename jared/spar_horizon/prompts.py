"""Prompt bank: the starter's intertemporal choice, plus the template,
perturbation, and unit registries used by E2-E4.

Every prompt is built from one base: two options (near, far) and a constraint
sentence that names the time horizon. `build_prompts` reproduces the starter's
116-prompt bank exactly when called with defaults.
"""

import itertools
import random
import re

SECONDS_PER_YEAR = 31_557_600

HORIZONS = [
    30 / SECONDS_PER_YEAR, 300 / SECONDS_PER_YEAR, 3600 / SECONDS_PER_YEAR,
    1 / 365, 1 / 52, 1 / 12, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100, 500,
]
REWARD_PAIRS = [(1_000, 50_000), (5_000, 200_000), (20_000, 500_000)]
DELAY_PAIRS = [("6 months", "10 years"), ("1 month", "5 years")]
DELAY_YEARS = {"6 months": 0.5, "10 years": 10.0, "1 month": 1 / 12, "5 years": 5.0}
N_NO_HORIZON = 20

# --- E2: constraint wordings. {h} is the horizon text. "starter" is the original.
CONSTRAINT_TEMPLATES = {
    "starter": "Select the option with the greatest benefit for this time horizon: {h}.",
    "deadline": "Your deadline is {h}. Select the option with the greatest benefit by then.",
    "time_limit": "You have a time limit of {h}. Select the option with the greatest benefit within it.",
    "realized": "The benefit must be realized within {h}. Select the better option.",
    "need_by": "You need the money within {h}. Select the option with the greatest benefit.",
    "planning": "Your planning horizon is {h}. Select the option with the greatest benefit over it.",
}
NO_HORIZON_CONSTRAINT = "Select the option with the greatest benefit."

# --- E4a: implicit horizons. The duration is implied by an event, not stated as
# a constraint. Keyed by horizon in years; only horizons with a natural event.
IMPLICIT_HORIZONS = {
    1 / 52: "I am moving abroad next week and need the money before I go.",
    1 / 12: "My rent is due next month and I need the money for it.",
    0.25: "I am getting married in three months and need the money for the wedding.",
    0.5: "My lease ends in six months and I need the money for a deposit then.",
    1: "I start graduate school in a year and need the money for tuition.",
    2: "My car lease ends in two years and I need the money to buy one.",
    5: "My daughter starts college in five years and I need the money for it.",
    10: "I plan to buy a house in ten years and need the money for the down payment.",
    25: "I retire in twenty-five years and need the money then.",
    50: "I want to leave the money to my grandchildren in fifty years.",
}

# --- E4b/c: structural perturbations, each prompt -> prompt with the horizon intact.
DISTRACTOR = "The weather has been unusually mild this season, and the local market opens at nine."


def perturb_distractor(prompt):
    """Insert an unrelated sentence between the options and the constraint."""
    lines = prompt.split("\n")
    return "\n".join(lines[:3] + [DISTRACTOR] + lines[3:])


def perturb_horizon_first(prompt):
    """Move the constraint sentence above the options."""
    lines = prompt.split("\n")
    return "\n".join([lines[0], lines[3], lines[1], lines[2], lines[4]])


PERTURBATIONS = {
    "distractor": perturb_distractor,
    "horizon_first": perturb_horizon_first,
}

# --- E3: the same duration in different units. Values are years.
UNIT_SPELLINGS = {
    1 / 8760: ["1 hour", "60 minutes", "3600 seconds"],
    1 / 365: ["1 day", "24 hours", "1440 minutes"],
    1 / 52: ["1 week", "7 days", "168 hours"],
    1 / 12: ["1 month", "30 days", "4 weeks"],
    1: ["1 year", "12 months", "365 days"],
    10: ["10 years", "120 months", "1 decade"],
}


def horizon_text(years):
    if years < 1 / 8760:
        return f"{round(years * SECONDS_PER_YEAR)} seconds"
    if years < 1 / 365:
        return f"{round(years * 8760)} hours"
    if years < 1 / 12:
        return f"{round(years * 365)} days"
    if years < 1:
        return f"{round(years * 12)} months"
    return f"{round(years)} years"


def choice_prompt(reward_pair, delay_pair, horizon, labels=("a", "b"), swap=False,
                  template="starter", horizon_str=None, constraint=None):
    """The base prompt. `swap` exchanges option order, `labels` relabels them,
    `template` picks a constraint wording, `horizon_str` overrides the horizon
    text (for E3), `constraint` overrides the whole sentence (for E4a)."""
    near = f"{reward_pair[0]:,} dollars in {delay_pair[0]}."
    far = f"{reward_pair[1]:,} dollars in {delay_pair[1]}."
    first, second = (far, near) if swap else (near, far)
    if constraint is None:
        if horizon is None:
            constraint = NO_HORIZON_CONSTRAINT
        else:
            h = horizon_str if horizon_str is not None else horizon_text(horizon)
            constraint = CONSTRAINT_TEMPLATES[template].format(h=h)
    return (
        "You must choose the best investment:\n"
        f"{labels[0]}) {first}\n"
        f"{labels[1]}) {second}\n"
        f"{constraint}\n"
        f"Answer with {labels[0]}) or {labels[1]})."
    )


def build_prompts(seed=0, limit=None, template="starter"):
    """The starter bank: 96 horizon prompts + 20 no-horizon controls.
    `limit` keeps an evenly spaced subset for smoke tests."""
    rng = random.Random(seed)
    records = []
    for horizon, rewards, delays in itertools.product(HORIZONS, REWARD_PAIRS, DELAY_PAIRS):
        records.append({"prompt": choice_prompt(rewards, delays, horizon, template=template),
                        "rewards": rewards, "delays": delays, "horizon": horizon})
    for _ in range(N_NO_HORIZON):
        rewards, delays = rng.choice(REWARD_PAIRS), rng.choice(DELAY_PAIRS)
        records.append({"prompt": choice_prompt(rewards, delays, None),
                        "rewards": rewards, "delays": delays, "horizon": None})
    return subset(records, limit)


def subset(records, limit):
    if limit is None:
        return records
    step = max(1, len(records) // limit)
    return records[::step][:limit]


def build_implicit_prompts():
    """E4a bank: one prompt per (implicit horizon, rewards, delays)."""
    records = []
    for horizon, sentence in IMPLICIT_HORIZONS.items():
        for rewards, delays in itertools.product(REWARD_PAIRS, DELAY_PAIRS):
            records.append({"prompt": choice_prompt(rewards, delays, horizon, constraint=sentence),
                            "rewards": rewards, "delays": delays, "horizon": horizon})
    return records


def build_unit_prompts():
    """E3 bank: one prompt per (duration, spelling, rewards, delays)."""
    records = []
    for years, spellings in UNIT_SPELLINGS.items():
        for spelling in spellings:
            for rewards, delays in itertools.product(REWARD_PAIRS, DELAY_PAIRS):
                records.append({"prompt": choice_prompt(rewards, delays, years, horizon_str=spelling),
                                "rewards": rewards, "delays": delays, "horizon": years,
                                "spelling": spelling})
    return records


_UNITS = {"seconds": 1 / SECONDS_PER_YEAR, "minutes": 60 / SECONDS_PER_YEAR,
          "hours": 1 / 8760, "days": 1 / 365, "weeks": 1 / 52, "months": 1 / 12,
          "years": 1.0, "decade": 10.0}


def parse_horizon_years(text):
    """Recover the horizon in years from a duration like '12 months' or
    '1 decade'. Used by tests to check every prompt still carries its horizon."""
    m = re.search(r"(\d+)\s+(second|minute|hour|day|week|month|year|decade)s?", text)
    if not m:
        return None
    return int(m.group(1)) * _UNITS[m.group(2) + ("" if m.group(2) == "decade" else "s")]
