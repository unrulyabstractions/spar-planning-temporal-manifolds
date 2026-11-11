"""Durations: the horizon grid against the report's Table 2, exact comparison, rendering, unit rewrites."""

from fractions import Fraction

from pbank.durations import (HORIZON_GRID, REWRITE_MAX_VALUE, REWRITE_TOLERANCE, Duration, natural_duration,
                             round_for_display, unit_rewrites)

# Table 2 of the midterm report ("Horizon grid: the 48 stated horizons, grouped by the unit they are written in"),
# transcribed from midterm_report/latex/report.tex.
REPORT_TABLE_2 = {
    "seconds": [30, 46, 71, 110],
    "minutes": [3, 4, 6, 10, 15, 23, 36, 55, 84],
    "hours": [2, 3, 5, 8, 12, 18, 28, 43],
    "days": [3, 4, 6, 10],
    "weeks": [2, 3, 5, 8],
    "months": [3, 4, 6, 10, 15, 23],
    "years": [3, 5, 7, 11, 16, 25, 39, 59, 91, 140, 210, 330, 500],
}


def test_grid_matches_report_table_2():
    by_unit = {}
    for d in HORIZON_GRID:
        by_unit.setdefault(d.unit, []).append(int(d.value))
    for unit, values in REPORT_TABLE_2.items():
        print(f"{unit:>8}: grid {by_unit.get(unit)}  report {values}")
    assert by_unit == REPORT_TABLE_2
    assert len(HORIZON_GRID) == 48


def test_grid_is_increasing_and_log_spaced():
    days = [float(d.days) for d in HORIZON_GRID]
    assert days == sorted(days) and len(set(days)) == 48
    ratios = [b / a for a, b in zip(days, days[1:])]
    print(f"successive ratios of the written values: min {min(ratios):.3f}, max {max(ratios):.3f} (target 1.533)")
    # Rounding moves each value by at most 10% either way, so the ratio of two neighbours lies within
    # 1.533 * 0.9/1.1 = 1.254 and 1.533 * 1.1/0.9 = 1.874 (e.g. 3 -> 4 minutes is 1.333).
    assert 1.533 * 0.9 / 1.1 < min(ratios) and max(ratios) < 1.533 * 1.1 / 0.9


def test_rendering_error_is_at_most_ten_percent():
    import math
    lo, hi = math.log10(30 / 86400), math.log10(500 * 365.25)
    worst = 0.0
    for i, d in enumerate(HORIZON_GRID):
        target = 10 ** (lo + i * (hi - lo) / 47)
        worst = max(worst, abs(float(d.days) / target - 1))
    print(f"largest relative rounding error on the grid: {worst:.3%}")
    assert worst <= 0.10


def test_round_for_display():
    assert [round_for_display(x) for x in (2.4, 99.6, 108.4, 139.2, 213.0, 326.0, 500.0)] == [2, 100, 110, 140, 210, 330, 500]


def test_natural_unit_is_largest_with_count_at_least_two():
    assert str(natural_duration(1.5 / 24)) == "90 minutes"     # 1.5 hours: fewer than 2 hours, so minutes
    assert str(natural_duration(1.9 / 24)) == "110 minutes"    # 114 minutes, above 100: two significant figures
    assert str(natural_duration(2.0 / 24)) == "2 hours"
    assert str(natural_duration(700.0)) == "23 months"


def test_str_and_parse():
    for text in ("1 year", "2 years", "84 minutes", "1 week", "0.7 years", "17.86 years", "140 years"):
        d = Duration.parse(text)
        print(f"{text!r} -> {d!r} -> {str(d)!r}")
        assert str(d) == text
    assert str(Duration(Fraction(1), "months")) == "1 month"


def test_exact_comparison_has_no_float_error():
    # 12 months, 1 year and 365.25 days are the same length exactly; 52 weeks is shorter.
    assert Duration.parse("12 months").days == Duration.parse("1 year").days == Fraction(36525, 100)
    assert Duration.parse("52 weeks").days < Duration.parse("1 year").days
    assert Duration.parse("60 months").days == Duration.parse("5 years").days


def test_unit_rewrites_respect_tolerance_and_bounds():
    total = 0
    for d in HORIZON_GRID:
        rewrites = unit_rewrites(d)
        total += len(rewrites)
        if rewrites:
            print(f"{str(d):>12} -> {', '.join(str(r) for r in rewrites)}")
        for r in rewrites:
            assert r.unit != d.unit
            assert r.value.denominator == 1 and 1 <= r.value <= REWRITE_MAX_VALUE
            assert abs(float(r.days / d.days) - 1) <= REWRITE_TOLERANCE
    print(f"{sum(bool(unit_rewrites(d)) for d in HORIZON_GRID)} of 48 grid values have a rewrite; {total} in all")
    # The report's own example of a unit rewrite.
    assert Duration.parse("100 weeks") in unit_rewrites(Duration.parse("23 months"))
    # Calendar approximations within 2% are offered: 3 months = 13.04 weeks, so "13 weeks" is 0.3% short.
    assert Duration.parse("13 weeks") in unit_rewrites(Duration.parse("3 months"))
    # A count above the cap is never offered.
    assert Duration.parse("1380 seconds") not in unit_rewrites(Duration.parse("23 minutes"))
