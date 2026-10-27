"""analysis/unit_overlap.py on planted codes, laid out as the real units bank."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import unit_overlap as uo  # noqa: E402
from spar_horizon.full_bank import build_units_prompts  # noqa: E402


@pytest.fixture(scope="module")
def bank():
    records = build_units_prompts()
    natural_unit = {r["pair_id"]: r["rendered_unit"] for r in records if r["natural"]}
    return {"y": np.log10([r["horizon"] for r in records]),
            "durations": np.array([r["duration_id"] for r in records]),
            "units": np.array([r["rendered_unit"] for r in records]),
            "natural_units": np.array([natural_unit[r["pair_id"]] for r in records]),
            "pairs": np.array([r["pair_id"] for r in records]),
            "natural": np.array([r["natural"] for r in records])}


def measures(bank, X):
    keys = ("y", "durations", "units", "natural_units", "pairs", "natural")
    return uo.cell_measures(X, *(bank[k] for k in keys), ks=(2,))[2]


def test_a_duration_code_reads_other_renderings_as_well_as_natural_ones(bank):
    rng = np.random.default_rng(0)
    X = np.outer(bank["y"], rng.normal(size=32)) + 0.05 * rng.normal(size=(len(bank["y"]), 32))
    m = measures(bank, X)
    assert m["r2_natural"] > 0.95 and m["r2_other"] > 0.95
    assert m["shift_abs"] < 0.05
    # unit and duration are largely confounded in the bank, so the unique parts are small; their contrast is the reading
    assert m["share"]["unit"] < 0.01 and m["share"]["duration"] > 0.05
    assert m["within_horizon"] > 0.9


def test_a_unit_word_code_misreads_other_renderings_by_their_unit(bank):
    rng = np.random.default_rng(0)
    unit_vectors = {u: rng.normal(size=32) for u in np.unique(bank["units"])}
    X = np.stack([unit_vectors[u] for u in bank["units"]]) + 0.05 * rng.normal(size=(len(bank["y"]), 32))
    m = measures(bank, X)
    assert m["r2_natural"] - m["r2_other"] > 0.3
    assert m["shift_smaller_unit"] < -0.3 and m["shift_larger_unit"] > 0.3   # "65 weeks" read as weeks-sized
    assert m["share"]["unit"] > 0.2 and m["share"]["duration"] < 0.01
    # no within-unit credit; below 0, since holding a duration out shifts its unit's mean against it
    assert m["within_horizon"] < 0.2


def test_pair_shifts_net_out_the_true_difference():
    pred = np.array([1.0, 1.5, 2.0, 2.1])
    y = np.array([1.0, 1.1, 2.0, 2.0])
    rows, shift = uo.pair_shifts(pred, y, np.array(["p", "p", "q", "q"]), np.array([True, False, True, False]))
    assert rows.tolist() == [1, 3]
    assert np.allclose(shift, [0.4, 0.1])


def test_behavior_counts_pairs_with_one_choice():
    pairs = np.array(["p", "p", "q", "q", "q"])
    natural = np.array([True, False, True, False, False])
    units = np.array(["months", "weeks", "hours", "minutes", "days"])
    natural_units = np.array(["months", "months", "hours", "hours", "hours"])
    b = uo.behavior(np.array([2.0, 1.0, -1.0, -1.5, 0.5]), np.array([1, 1, 0, 0, 1]), pairs, natural,
                    units, natural_units)
    assert b["pairs"] == 2 and b["same_choice"] == 0.5
    assert np.isclose(b["diff_smaller_unit"][0], (-1.0 - 0.5) / 2)   # weeks for months, minutes for hours
    assert np.isclose(b["diff_larger_unit"][0], 1.5)                 # days for hours


def test_shift_size_averages_each_rendering_before_taking_its_size():
    shift = np.array([0.5, -0.5, 0.3, 0.3])
    size = uo.rendering_shift_size(shift, np.array(["1 h", "1 h", "2 h", "2 h"]), np.array(["min", "min", "min", "min"]))
    assert np.isclose(size, (0.0 + 0.3) / 2)   # prompt scatter around 0 is not a shift
