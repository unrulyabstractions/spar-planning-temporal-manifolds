"""pipeline/s3_analysis/split.py on synthetic answers.json files laid out as exp0_baseline writes them."""

import json
import math

import pytest

from pipeline.s3_analysis.by_datatype.choice_by_datatype import summarize as summarize_datatype
from pipeline.s3_analysis.by_label.choice_by_label import summarize as summarize_label
from pipeline.s3_analysis.split import (COLUMNS, bank_of, load_run, load_runs, near_share, number_format,
                                        split_by)

UNITS_ROWS = [
    {"horizon": 0.25, "regime": "none", "truth": None, "near_label": "a", "rendered_unit": "months",
     "duration_id": "3 months", "delay_scale": "days-weeks", "label": "a", "forced": False},
    {"horizon": 0.25, "regime": "near", "truth": "near", "near_label": "b", "rendered_unit": "weeks",
     "duration_id": "3 months", "delay_scale": "months-years", "label": "a", "forced": True},
    {"horizon": 2.0, "regime": "both", "truth": None, "near_label": "a", "rendered_unit": "years",
     "duration_id": "2 years", "delay_scale": "months-years", "label": "b", "forced": False},
    {"horizon": 2.0, "regime": "both", "near_label": "a", "rendered_unit": "years", "duration_id": "2 years"},
]
CHOICE_ROWS = [
    {"horizon": 1.0, "regime": "both", "truth": "far", "n_options": 4, "near_labels": ["a", "b"],
     "label": "b", "forced": False},
    {"horizon": 1.0, "regime": "both", "truth": "far", "n_options": 4, "near_labels": ["a", "b"],
     "label": "d", "forced": True},
]


@pytest.fixture
def root(tmp_path):
    for name, rows in (("exp0_Qwen3-8B_think-on_units", UNITS_ROWS), ("exp0_Qwen3-8B_think-on_choice4", CHOICE_ROWS)):
        run = tmp_path / name
        run.mkdir()
        (run / "answers.json").write_text(json.dumps(rows))
    return tmp_path


@pytest.mark.parametrize("name, bank", [("exp0_Qwen3-8B_think-on_units", "units"),
                                        ("exp0_Qwen3-8B_think-off_choice4_roi", "choice4_roi"),
                                        ("exp0_Qwen3-8B_think-on", "starter")])
def test_bank_comes_from_the_run_folder_name(name, bank):
    assert bank_of(name) == bank


def test_every_row_has_the_documented_columns_and_unkept_prompts_are_left_out(root):
    rows = load_run(root / "exp0_Qwen3-8B_think-on_units")
    assert len(rows) == 3
    assert all(tuple(r) == COLUMNS for r in rows)
    assert rows[1] == {"regime": "near", "unit": "weeks", "horizon": 0.25, "delay": "months-years", "truth": "near",
                       "n_choices": 2, "forced_close": True, "bank": "units", "number_format": "digits",
                       "chose_near": False}


def test_choice_bank_counts_a_pick_in_the_near_half_as_near(root):
    rows = load_run(root / "exp0_Qwen3-8B_think-on_choice4")
    assert [r["n_choices"] for r in rows] == [4, 4]
    assert [r["chose_near"] for r in rows] == [True, False]
    assert rows[0]["unit"] is None


def test_number_format_is_digits_or_written_and_none_without_horizon():
    assert number_format({"horizon": 3.0, "duration_id": "3 years"}) == "digits"
    assert number_format({"horizon": 3.0, "duration_id": "three years"}) == "written"
    assert number_format({"horizon": None}) is None
    assert number_format({"horizon": 3.0}) == "digits"


def test_load_runs_concatenates_banks_and_skips_missing_ones(root):
    rows = load_runs(root, "Qwen3-8B", "on", ["units", "fill", "choice4"])
    assert split_by(rows, "bank").keys() == {"units", "choice4"}
    assert len(rows) == 5


def test_split_by_keeps_every_row_and_groups_none_values(root):
    rows = load_runs(root, "Qwen3-8B", "on", ["units", "choice4"])
    by_regime = split_by(rows, "regime")
    assert {k: len(v) for k, v in by_regime.items()} == {"none": 1, "near": 1, "both": 3}
    by_unit = split_by(rows, "unit")
    assert len(by_unit[None]) == 2
    assert sum(len(v) for v in by_unit.values()) == len(rows)


def test_near_share_ignores_unknown_choices_and_is_nan_for_none():
    rows = [{"chose_near": True}, {"chose_near": False}, {"chose_near": True}, {"chose_near": None}]
    assert near_share(rows) == pytest.approx(2 / 3)
    assert math.isnan(near_share([{"chose_near": None}]))
    assert math.isnan(near_share([]))


def test_label_summary_separates_natural_and_forced_closes(root):
    rows = load_runs(root, "Qwen3-8B", "on", ["units"])
    summary = {value: (n, natural, forced) for value, n, natural, forced in summarize_label(rows, "regime")}
    assert summary["none"][:2] == (1, 1.0)
    assert math.isnan(summary["none"][2])
    assert summary["near"][0] == 1 and summary["near"][2] == 0.0


def test_datatype_summary_has_one_line_per_bank(root):
    rows = load_runs(root, "Qwen3-8B", "on", ["units", "choice4"])
    assert summarize_datatype(rows, "bank") == [("choice4", 2, 0.5), ("units", 3, pytest.approx(1 / 3))]
