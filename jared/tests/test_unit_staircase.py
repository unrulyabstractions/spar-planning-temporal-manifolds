import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from unit_staircase import staircase  # noqa: E402


def renderings():
    """Durations 1..1000 days in two units: days below 100, then 'hundreds' (100 days)."""
    days = np.logspace(0, 3, 13)
    units = np.where(days < 100, "days", "hundreds")
    number = np.where(days < 100, days, days / 100)
    return np.log10(days), units, np.log10(number)


def test_a_duration_code_has_no_steps():
    logh, units, _ = renderings()
    overall, within, boundary, inside = staircase(0.9 * logh, logh, units)
    assert np.isclose(overall, 0.9) and np.isclose(within, 0.9)
    assert abs(boundary) < 1e-9 and abs(inside) < 1e-9


def test_a_number_line_is_steeper_within_units_and_drops_at_the_boundary():
    logh, units, number = renderings()
    readout = 0.8 * number + 0.4 * (logh - number)       # a > b
    overall, within, boundary, inside = staircase(readout, logh, units)
    assert within > overall and np.isclose(within, 0.8)
    assert boundary < 0 < inside
