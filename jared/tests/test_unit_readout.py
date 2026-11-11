"""pipeline/s3_analysis/by_datatype/unit_readout.py: the number weight on readouts planted on the real units bank."""

import numpy as np
import pytest

import pipeline.s3_analysis.by_datatype.unit_readout as ur
from spar_horizon.full_bank import UNIT_YEARS, build_units_prompts


@pytest.fixture(scope="module")
def bank():
    records = build_units_prompts()
    return {"y": np.log10([r["horizon"] for r in records]),
            "units": np.array([r["rendered_unit"] for r in records]),
            "pairs": np.array([r["pair_id"] for r in records]),
            "natural": np.array([r["natural"] for r in records])}


def codes(bank):
    """Readouts of a pure number, duration and unit-word code, with a little noise."""
    noise = 0.02 * np.random.default_rng(1).normal(size=len(bank["y"]))
    return {"number": ur.log_numbers(bank["y"], bank["units"]) + noise, "duration": bank["y"] + noise,
            "unit": np.log10([UNIT_YEARS[u] for u in bank["units"]]) + noise}


def test_log_numbers_are_the_written_integers(bank):
    numbers = 10 ** ur.log_numbers(bank["y"], bank["units"])
    assert np.allclose(numbers, np.round(numbers)) and numbers.min() >= 1 and numbers.max() <= 1000


def test_number_index_and_pair_slope_are_plus_one_zero_minus_one_for_number_duration_unit_codes(bank):
    for name, expected in (("number", 1), ("duration", 0), ("unit", -1)):
        pred = codes(bank)[name]
        for value, low, high in (ur.number_index(pred, bank, np.random.default_rng(0)),
                                 ur.pair_slope(pred, bank, np.random.default_rng(0))):
            assert abs(value - expected) < 0.05 and low <= value <= high, name


def test_a_weak_probe_keeps_its_number_index_but_not_its_pair_slope(bank):
    c = codes(bank)
    pred = 0.3 * (c["duration"] + c["number"])   # a = 0.6, b = 0.3: index 1/3 at any scale
    index, _, _ = ur.number_index(pred, bank, np.random.default_rng(0))
    slope, _, _ = ur.pair_slope(pred, bank, np.random.default_rng(0))
    assert abs(index - 1 / 3) < 0.05
    assert abs(slope - 0.3) < 0.05   # a - b, in raw decades
