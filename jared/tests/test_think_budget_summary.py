"""P0, C1, C2, C3 of 28_spec_choice4_side.md on hand-made rows, and the chunk reading."""

import json

import pytest

import pipeline.s2_steering.experiments.think_budget.exp17_think_budget as exp17
import pipeline.s2_steering.analysis.think_budget_summary as summary
from pipeline.s2_steering.analysis.think_budget_stage0 import accuracy, position_stats

LABELS = ["a", "b", "c", "d"]
INFO = {"a": ["near", 1], "b": ["near", 2], "c": ["far", 1], "d": ["far", 2]}
# (truth, original forced label, loop, last committed label)
SPECS = [("a", "d", False, "a"), ("a", "c", False, None), ("b", "b", False, "b"),
         ("c", "d", True, None), ("d", "a", True, "d"), ("a", "a", False, "a")]


def make_stage0(bank="choice4"):
    rows = [{"sample_uid": f"u{i}", "regime": "near" if i % 2 else "both", "truth_label": truth,
             "truth": INFO[truth][0], "labels": LABELS, "option_info": INFO, "orig_label": orig, "loop": loop,
             "tentative": tentative} for i, (truth, orig, loop, tentative) in enumerate(SPECS)]
    origs = [r["orig_label"] for r in rows]
    side = {"accuracy": accuracy(rows, origs), "position": position_stats(rows, origs)}
    return {"bank": bank, "rows": rows, "reference": {"forced": side, "natural": side}}


def unit_row(uid, label, forced, new=100):
    return {"sample_uid": uid, "label": label, "forced": forced, "new_think_len": new}


UNIT1 = {f"u{i}": unit_row(f"u{i}", lab, True, 0) for i, lab in enumerate("dcbdba")}
UNIT2 = {"u0": unit_row("u0", "a", False, 500), "u1": unit_row("u1", "a", False, 700),
         "u2": unit_row("u2", "b", False, 900), "u3": unit_row("u3", "d", True, 3072),
         "u4": unit_row("u4", "a", True, 3072), "u5": unit_row("u5", "c", False, 100)}


def test_unit_one_is_the_noise_floor_and_p0_checks_the_85_percent_line():
    s = summary.summarize_bank("choice4", make_stage0(), {1: UNIT1})
    assert s["unit1"]["same_label"] == [5, 6]
    assert s["criteria"]["P0"]["pass"] is False                      # 5/6 = 0.83 < 0.85
    assert s["criteria"]["C1"]["pass"] is None and s["unit2"] is None
    same = {uid: unit_row(uid, SPECS[int(uid[1])][1], True, 0) for uid in UNIT1}
    assert summary.summarize_bank("choice4", make_stage0(), {1: same})["criteria"]["P0"]["pass"] is True


def test_continuation_counts_and_criteria():
    s = summary.summarize_bank("choice4", make_stage0(), {1: UNIT1, 2: UNIT2})["unit2"]
    assert s["closed"] == [4, 6]
    assert s["closed_by_loop"] == {"loop": [0, 2], "no loop": [4, 4]}
    assert s["closed_by_regime"] == {"both": [2, 3], "near": [2, 3]}
    assert (s["wrong_to_right"], s["right_to_wrong"], s["net_fixes"]) == (2, 1, 1)
    assert s["accuracy_closed"] == {"n": 4, "exact": 3, "half": 3, "within_half": 3}
    assert s["new_tokens_closed"]["median"] == 600
    assert s["agrees_with_tentative"] == {"forced_label": [2, 3], "continued_label": [2, 3]}
    assert s["still_open_accuracy"]["n"] == 2
    c = summary.summarize_bank("choice4", make_stage0(), {1: UNIT1, 2: UNIT2})["criteria"]
    assert c["C1"]["pass"] is False                                   # closed 0.67 ok, accuracy 0.75 < 0.85
    assert c["C1"]["net_min"] == pytest.approx(10 * 6 / 105)
    assert c["C2"]["excess_late_closed"] == pytest.approx(0.25) and c["C2"]["pass"] is False
    assert c["C3"]["pass"] is True


def test_c2_uses_half_the_banks_forced_excess_outside_choice4():
    stage0 = make_stage0("choice8")
    forced = stage0["reference"]["forced"]["position"]["excess_late"]
    c2 = summary.summarize_bank("choice8", stage0, {2: UNIT2})["criteria"]["C2"]
    assert c2["max"] == pytest.approx(forced / 2)


def test_unit_three_counts_the_open_traces_it_closed():
    unit3 = {"u3": unit_row("u3", "c", False, 3500), "u4": unit_row("u4", "a", True, 5000)}
    s = summary.summarize_bank("choice4", make_stage0(), {2: UNIT2, 3: unit3})["unit3"]
    assert (s["open_after_unit2"], s["attempted"], s["closed_in_unit3"]) == (2, 2, 1)
    assert s["closed_within_8192_of_all_unit2"] == [5, 6]


def test_a_prefix_of_chunks_is_summarised_with_its_n_and_files_are_read(tmp_path):
    inputs, root = tmp_path / "inputs", tmp_path / "run"
    inputs.mkdir()
    (inputs / "stage0_choice4.json").write_text(json.dumps(make_stage0()))
    unit2 = root / "choice4" / "unit2"
    exp17.write_chunk(unit2, 0, 2, ["u0", "u1"], [UNIT2["u0"], UNIT2["u1"]], [[1], [2]], 0.0)
    exp17.write_chunk(unit2, 1, 2, ["u2", "u3"], [UNIT2["u2"], UNIT2["u3"]], [[1], [2]], 0.0)
    result, text = summary.summarize_root(root, inputs)
    assert result["choice4"]["unit2"]["n"] == 4 and result["choice4"]["unit2"]["closed"] == [3, 4]
    assert "unit 2: 2" in text and "C1:" in text and "C3:" in text
    assert "choice8" not in result
