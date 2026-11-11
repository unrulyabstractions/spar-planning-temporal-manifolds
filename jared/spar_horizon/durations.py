"""Durations written in text: "<number> <unit>" with the number as digits or an English word
("three-year", "fifty-nine years", "a year"), which of them match a given horizon, and which one a
reasoning trace states as its horizon.

Shared by the stages (the written-unit audit, the trace audits, the phrase steering and the
patching summaries), so it lives here rather than in any one stage's scripts.
"""

import re

from .horizons import UNIT_YEARS, canonical_unit

TOLERANCE = 0.10

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen").split()
TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
WORD_VALUES = {**{w: i for i, w in enumerate(ONES)}, **{w: 20 + 10 * i for i, w in enumerate(TENS)},
               "a": 1, "an": 1, "hundred": 100, "thousand": 1000}

# a number (digits, a word, or a hyphenated compound like "fifty-nine"), then a time unit,
# joined by spaces or a hyphen ("three-year")
DURATION = re.compile(r"\b(?P<number>\d[\d,]*(?:\.\d+)?|[a-z]+(?:-[a-z]+)?)[\s-]+"
                      r"(?P<unit>seconds?|minutes?|hours?|days?|weeks?|months?|years?|decades?|century|centuries)\b",
                      re.IGNORECASE)


def parse_number(text):
    """The value of a digit string or an English number word ("fifty-nine"), or None."""
    text = text.lower().replace(",", "")
    if text[0].isdigit():
        return float(text)
    if text in WORD_VALUES:
        return float(WORD_VALUES[text])
    tens, _, ones = text.partition("-")
    if tens in TENS and ones in ONES[1:10]:
        return float(WORD_VALUES[tens] + WORD_VALUES[ones])
    return None


def durations(text):
    """Every "<number> <unit>" in `text` as (years, canonical unit), in order."""
    out = []
    for match in DURATION.finditer(text):
        value = parse_number(match["number"])
        if value is not None:
            unit = canonical_unit(match["unit"])
            out.append((value * UNIT_YEARS[unit], unit))
    return out


def close_to(years, target, tolerance=TOLERANCE):
    return abs(years / target - 1) <= tolerance


def horizon_units(text, horizon_years):
    """The units of the mentions in `text` whose duration matches the horizon, in order."""
    return [unit for years, unit in durations(text) if close_to(years, horizon_years)]


# post hoc: a duration stated as the horizon ("a 13-year time horizon", "the horizon is 2.5 years")
HORIZON_AFTER = re.compile(r"[\s-]+(?:time[\s-]+)?horizon", re.IGNORECASE)
HORIZON_BEFORE = re.compile(r"horizon (?:given |here |stated )?(?:is|of) (?:about |approximately |roughly )?$",
                            re.IGNORECASE)
MOVED = 0.05                # decades (12%): further than any rendering of the same horizon


def first_stated_horizon(text):
    """(years, unit, number as written) of the first duration the trace states as its horizon, or None."""
    for match in DURATION.finditer(text):
        before = text[max(0, match.start() - 40):match.start()]
        if HORIZON_AFTER.match(text, match.end()) or HORIZON_BEFORE.search(before):
            value = parse_number(match["number"])
            if value is not None:
                unit = canonical_unit(match["unit"])
                return value * UNIT_YEARS[unit], unit, match["number"]
    return None
