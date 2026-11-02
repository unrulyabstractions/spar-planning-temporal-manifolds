import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from written_unit import durations, horizon_units, parse_number, summarize  # noqa: E402


@pytest.mark.parametrize("text, value", [("36", 36), ("1,200", 1200), ("1.5", 1.5), ("three", 3),
                                         ("fifty-nine", 59), ("a", 1), ("Twenty", 20), ("per", None),
                                         ("twenty-ten", None)])
def test_parse_number(text, value):
    assert parse_number(text) == value


def test_conversion_in_reasoning_is_found_in_order():
    text = "The time horizon given is 36 months, which is three years. A three-year horizon."
    assert horizon_units(text, 3.0) == ["months", "years", "years"]


def test_delays_and_rates_are_not_horizon_mentions():
    # horizon 3 years; "five years" is a delay, "per year" has no number
    text = "Wait five years for the money, at 5% per year, but the horizon is 3 years."
    assert horizon_units(text, 3.0) == ["years"]


def test_mentions_within_ten_percent_count():
    # "23 months" is 4% short of 2 years; "20 months" is 17% short
    assert horizon_units("about 23 months, not 20 months", 2.0) == ["months"]


def test_durations_converts_units_to_years():
    (years, unit), = durations("a decade")
    assert unit == "decades" and years == pytest.approx(10.0)


def test_summarize_main_unit_breaks_ties_by_first_mention():
    s = summarize(["months", "years", "years", "months"], "months")
    assert s["main"] == "months" and s["first"] == "months" and s["last"] == "months"
    assert s["converts"] is True
    assert summarize([], "months")["main"] is None
