"""Time-horizon values: parsing, unit conversion, formatting, and the standard grid.

A horizon is a positive duration. Internally everything converts to years (astronomical
year, 365.25 days) so that horizons from different units compare and sort correctly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

DAYS_PER_YEAR = 365.25

# canonical plural unit -> years per unit
UNIT_YEARS: dict[str, float] = {
    "seconds": 1.0 / (DAYS_PER_YEAR * 86400),
    "minutes": 1.0 / (DAYS_PER_YEAR * 1440),
    "hours": 1.0 / (DAYS_PER_YEAR * 24),
    "days": 1.0 / DAYS_PER_YEAR,
    "weeks": 7.0 / DAYS_PER_YEAR,
    "months": 1.0 / 12.0,
    "years": 1.0,
    "decades": 10.0,
    "centuries": 100.0,
}

_ALIASES: dict[str, str] = {}
for _u in UNIT_YEARS:
    _ALIASES[_u] = _u
    _ALIASES[_u[:-1]] = _u  # singular
_ALIASES.update({"century": "centuries", "centurys": "centuries"})


def canonical_unit(unit: str) -> str:
    key = unit.strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"unknown time unit: {unit!r}")
    return _ALIASES[key]


@dataclass(frozen=True, order=False)
class Horizon:
    value: float
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", canonical_unit(self.unit))
        if not (self.value > 0):
            raise ValueError(f"horizon value must be positive, got {self.value}")

    @property
    def years(self) -> float:
        return self.value * UNIT_YEARS[self.unit]

    @property
    def log10_years(self) -> float:
        return math.log10(self.years)

    def __str__(self) -> str:
        v = int(self.value) if float(self.value).is_integer() else self.value
        unit = self.unit
        if v == 1:
            unit = "century" if unit == "centuries" else unit[:-1]
        return f"{v} {unit}"

    def __lt__(self, other: "Horizon") -> bool:
        return self.years < other.years

    @classmethod
    def parse(cls, text: str) -> "Horizon":
        """Parse strings like '6 months', '1 year', '2.5 weeks'."""
        m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]+)\s*", text)
        if not m:
            raise ValueError(f"cannot parse horizon: {text!r}")
        return cls(float(m.group(1)), m.group(2))


# The 17-point grid used by the prior work's highly-formatted prompts: 1 day to 100 years.
STANDARD_HORIZONS: tuple[Horizon, ...] = tuple(
    Horizon(v, u)
    for v, u in [
        (1, "days"), (3, "days"), (1, "weeks"), (2, "weeks"),
        (1, "months"), (2, "months"), (3, "months"), (6, "months"),
        (1, "years"), (2, "years"), (3, "years"), (5, "years"),
        (10, "years"), (20, "years"), (30, "years"), (50, "years"), (100, "years"),
    ]
)


# Extended grid: four sub-day values below and four multi-century values above the standard grid.
# Sub-day values must use hours/minutes; the top end stays in years to avoid introducing a new unit
# (unit rewrites measurably shift geometry and behavior; see progress-20260922.md).
EXTENDED_HORIZONS: tuple[Horizon, ...] = tuple(
    [Horizon(v, u) for v, u in [(1, "minutes"), (10, "minutes"), (1, "hours"), (6, "hours")]]
    + list(STANDARD_HORIZONS)
    + [Horizon(v, u) for v, u in [(300, "years"), (1000, "years"), (3000, "years"), (10000, "years")]]
)

GRIDS: dict[str, tuple[Horizon, ...]] = {"standard": STANDARD_HORIZONS, "extended": EXTENDED_HORIZONS}
