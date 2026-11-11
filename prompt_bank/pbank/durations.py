"""Durations: exact arithmetic, the rendering rule, the 48-point horizon grid, and unit rewrites.

Everything in the prompt bank that is a length of time (a horizon, an option's delay) is a `Duration`:
a value and a unit, exactly as it is written in the prompt ("84 minutes", "0.7 years").

Two choices here matter for correctness downstream:

1. **Exact arithmetic.** Durations are compared in days using `fractions.Fraction`, never floats. Whether an
   option "pays off within the horizon" is a comparison of two durations, and ties are real in this bank
   (a horizon of "3 months" with an option that pays "in 3 months"). With floats, 12 * (1/12) years versus
   1 year can land on either side of the tie; with fractions it cannot.

2. **The written value is the duration.** A grid point targets a log-spaced duration but is written rounded
   ("84 minutes" for 83.6 minutes). Every comparison uses the rounded, written value, because that is what
   the model reads.

Calendar convention (same as the earlier project code): 1 year = 365.25 days, 1 month = 1/12 year,
1 week = 7 days.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

# Length of one unit, in days, as an exact fraction.
UNIT_DAYS: dict[str, Fraction] = {
    "seconds": Fraction(1, 86400),
    "minutes": Fraction(1, 1440),
    "hours": Fraction(1, 24),
    "days": Fraction(1),
    "weeks": Fraction(7),
    "months": Fraction(36525, 1200),   # 365.25 / 12
    "years": Fraction(36525, 100),     # 365.25
}

# The units a horizon may be written in, largest first. (No quarters, decades or centuries: earlier runs
# showed that unusual units change both the geometry and the behaviour, and quarters were read by some
# models as per-period payments.)
NATURAL_UNITS: tuple[str, ...] = ("years", "months", "weeks", "days", "hours", "minutes", "seconds")

DAYS_PER_YEAR = UNIT_DAYS["years"]


@dataclass(frozen=True)
class Duration:
    """A length of time as written in a prompt: `value` (exact) in `unit` (plural form, e.g. "months")."""

    value: Fraction
    unit: str

    def __post_init__(self) -> None:
        if self.unit not in UNIT_DAYS:
            raise ValueError(f"unknown unit {self.unit!r}; expected one of {list(UNIT_DAYS)}")
        if self.value <= 0:
            raise ValueError(f"a duration must be positive, got {self.value}")
        object.__setattr__(self, "value", Fraction(self.value))

    @property
    def days(self) -> Fraction:
        """Exact length in days. Use this for every comparison between durations."""
        return self.value * UNIT_DAYS[self.unit]

    @property
    def years(self) -> float:
        """Approximate length in years. For display, logs and plots only, never for comparisons."""
        return float(self.days / DAYS_PER_YEAR)

    def __str__(self) -> str:
        """The text that appears in the prompt: "1 year", "84 minutes", "0.7 years"."""
        if self.value.denominator == 1:
            number = str(self.value.numerator)
        else:
            # Exact decimal expansion; values in this bank have at most two decimals.
            number = str((Decimal(self.value.numerator) / Decimal(self.value.denominator)).normalize())
        unit = self.unit[:-1] if self.value == 1 else self.unit   # "1 year", "2 years"
        return f"{number} {unit}"

    @classmethod
    def parse(cls, text: str) -> "Duration":
        """Inverse of `str()`: "84 minutes" -> Duration(84, "minutes"); also accepts "1 year", "0.7 years"."""
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s+([a-z]+?)s?\s*", text)
        if not m:
            raise ValueError(f"cannot parse duration {text!r}")
        return cls(Fraction(Decimal(m.group(1))), m.group(2) + "s")


# ---------------------------------------------------------------------------------------------------------------
# The rendering rule and the horizon grid (report, Appendix A.1 and Table 2)
# ---------------------------------------------------------------------------------------------------------------

def round_for_display(x: float) -> int:
    """Round a count to a whole number, keeping two significant figures above 100 (108 -> 110, 326 -> 330)."""
    if x <= 100:
        return round(x)
    step = 10 ** (math.floor(math.log10(x)) - 1)
    return round(x / step) * step


def natural_duration(days: float) -> Duration:
    """`days` written as people would: in the largest unit in which the count is at least 2, rounded by
    `round_for_display`. Example: 0.058 days -> "84 minutes"; 51,000 days -> "140 years".

    Because the unit only ever grows with the duration, unit and magnitude stay ordered along the grid."""
    for unit in NATURAL_UNITS[:-1]:
        count = days / float(UNIT_DAYS[unit])
        if count >= 2:
            return Duration(Fraction(round_for_display(count)), unit)
    return Duration(Fraction(max(1, round_for_display(days / float(UNIT_DAYS["seconds"])))), "seconds")


def log_spaced_grid(first: Duration, last_years: float, n: int) -> tuple[Duration, ...]:
    """`n` durations evenly spaced in log time from `first` to `last_years` years, each written naturally."""
    lo, hi = math.log10(float(first.days)), math.log10(last_years * float(DAYS_PER_YEAR))
    return tuple(natural_duration(10 ** (lo + i * (hi - lo) / (n - 1))) for i in range(n))


# The 48 stated horizons of the T0 bank: 30 seconds to 500 years, each about 1.53 times the previous
# (about 5.4 per decade). tests/test_durations.py checks this against Table 2 of the report value by value.
HORIZON_GRID: tuple[Duration, ...] = log_spaced_grid(Duration(Fraction(30), "seconds"), 500.0, 48)


# ---------------------------------------------------------------------------------------------------------------
# Unit rewrites: the same duration written in a different unit ("23 months" as "100 weeks")
# ---------------------------------------------------------------------------------------------------------------

REWRITE_TOLERANCE = 0.02       # a rewrite may differ from the original duration by at most 2%
REWRITE_MAX_VALUE = 1000       # and must be written with a number no larger than this (no "5040 seconds")


def unit_rewrites(d: Duration, tolerance: float = REWRITE_TOLERANCE,
                  max_value: int = REWRITE_MAX_VALUE) -> tuple[Duration, ...]:
    """Every way to write `d` as a whole number of another natural unit, within `tolerance` of its length
    and with a number from 1 to `max_value`. Ordered largest unit first. Example: "3 months" -> ("13 weeks",
    "91 days"). Durations with no such rewrite (most sub-minute and multi-century ones) return ()."""
    out = []
    for unit in NATURAL_UNITS:
        if unit == d.unit:
            continue
        count = round(d.days / UNIT_DAYS[unit])
        if not 1 <= count <= max_value:
            continue
        candidate = Duration(Fraction(count), unit)
        if abs(float(candidate.days / d.days) - 1) <= tolerance:
            out.append(candidate)
    return tuple(out)
