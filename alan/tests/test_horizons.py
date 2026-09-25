import math
import pytest
from ptm.horizons import EXTENDED_HORIZONS, GRIDS, Horizon, STANDARD_HORIZONS


def test_parse_and_years():
    assert Horizon.parse("6 months").years == pytest.approx(0.5)
    assert Horizon.parse("1 year").years == 1.0
    assert Horizon.parse("2 weeks").years == pytest.approx(14 / 365.25)
    assert Horizon.parse("1 century").years == 100.0


def test_str_roundtrip():
    for h in STANDARD_HORIZONS:
        assert Horizon.parse(str(h)) == h, str(h)
    assert str(Horizon(1, "years")) == "1 year"
    assert str(Horizon(6, "months")) == "6 months"


def test_grid_is_sorted_and_spans():
    ys = [h.years for h in STANDARD_HORIZONS]
    assert ys == sorted(ys)
    assert len(STANDARD_HORIZONS) == 17
    assert math.isclose(ys[0], 1 / 365.25) and ys[-1] == 100.0


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        Horizon.parse("soon")
    with pytest.raises(ValueError):
        Horizon(0, "days")


def test_extended_grid():
    ys = [h.years for h in EXTENDED_HORIZONS]
    assert ys == sorted(ys) and len(EXTENDED_HORIZONS) == 25
    assert EXTENDED_HORIZONS[4:21] == STANDARD_HORIZONS
    assert str(EXTENDED_HORIZONS[0]) == "1 minute" and str(EXTENDED_HORIZONS[-1]) == "10000 years"
    assert {h.unit for h in EXTENDED_HORIZONS[-4:]} == {"years"}
    assert GRIDS["extended"] is EXTENDED_HORIZONS
