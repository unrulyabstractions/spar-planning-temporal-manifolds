"""Durations in text -> years. Handles digits and words, ranges (upper end), "half a decade", "a couple of years",
"a quarter of a year", "0.25 months". Used for the canonical horizons and for parsing what the model writes."""

from __future__ import annotations

import re

UNIT_YEARS = {
    "minute": 1 / 525_960, "hour": 1 / 8_766, "day": 1 / 365.25, "week": 7 / 365.25, "fortnight": 14 / 365.25,
    "month": 1 / 12, "quarter": 0.25, "year": 1.0, "decade": 10.0, "century": 100.0,
}
WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "hundred": 100, "half": 0.5, "couple": 2, "a couple of": 2, "half a": 0.5,
    "half an": 0.5, "a quarter of a": 0.25, "a quarter of an": 0.25,
}
_WORDS = "|".join(sorted((re.escape(w) for w in WORD_NUMBERS), key=len, reverse=True))
_NUM = rf"(\d+(?:,\d{{3}})*(?:\.\d+)?|{_WORDS})"
_UNIT = r"(minute|hour|day|week|fortnight|month|quarter|year|decade|centur(?:y|ie))s?"
DURATION = re.compile(rf"\b{_NUM}(?:\s*(?:-|–|—|to)\s*{_NUM})?[\s-]*{_UNIT}\b", re.IGNORECASE)


def _num(s: str) -> float:
    s = s.lower()
    return WORD_NUMBERS[s] if s in WORD_NUMBERS else float(s.replace(",", ""))


def _unit(s: str) -> str:
    s = s.lower()
    return "century" if s.startswith("centur") else s


def parse_years(text: str | None) -> float | None:
    """First duration in `text`, in years; a range gives its upper end. None if there is none."""
    if not text:
        return None
    m = DURATION.search(text)
    if not m:
        return None
    return _num(m.group(2) or m.group(1)) * UNIT_YEARS[_unit(m.group(3))]


def has_duration(text: str) -> bool:
    return DURATION.search(text) is not None
