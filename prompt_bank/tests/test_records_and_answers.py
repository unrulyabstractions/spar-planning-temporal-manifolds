"""The decision rule (regime, ties), record round-trips, and the answer parser and scorer."""

import pytest

from pbank.answers import parse_choice, score
from pbank.durations import Duration
from pbank.records import load_prompts, prompts_sha256, record_to_dict, regime, write_jsonl
from pbank.t0 import build_verification_grid, make_record

D = Duration.parse


@pytest.mark.parametrize("horizon, near, far, expected", [
    ("3 days", "1 week", "2 years", "none"),
    ("3 months", "3 months", "5 years", "near_only"),     # tie with the near delay: it pays
    ("13 weeks", "3 months", "5 years", "none"),          # 13 weeks is 0.3% shorter than 3 months
    ("5 years", "1 year", "5 years", "both"),             # tie with the far delay: it pays
    ("60 months", "1 year", "5 years", "both"),           # the same length in another unit, exactly
    ("4.99 years", "1 year", "5 years", "near_only"),
    (None, "1 year", "5 years", "no_horizon"),
])
def test_regime_rule(horizon, near, far, expected):
    got = regime(None if horizon is None else D(horizon), D(near), D(far))
    print(f"horizon {horizon!s:>10}, near {near:>8}, far {far:>8} -> {got}")
    assert got == expected


def test_far_only_is_impossible_and_raises():
    with pytest.raises(ValueError):
        regime(D("2 years"), D("3 years"), D("2 years"))


def test_labels_map_to_options_in_both_orders():
    common = dict(uid="t", split="verification", near_delay=D("1 year"), far_delay=D("10 years"), near_reward=1000,
                  multiplier=5, labels=("1)", "2)"), horizon=D("20 years"), grid_value=D("20 years"), phrasing_id=1)
    near_first, far_first = make_record(near_first=True, **common), make_record(near_first=False, **common)
    assert (near_first.near_label, near_first.far_label, near_first.correct_label) == ("1)", "2)", "2)")
    assert (far_first.near_label, far_first.far_label, far_first.correct_label) == ("2)", "1)", "1)")
    assert far_first.option_for_label("1)") == "far" and far_first.option_for_label("c)") is None


def test_jsonl_round_trip_and_hash(tmp_path):
    records = build_verification_grid()
    write_jsonl(tmp_path / "prompts.jsonl", (record_to_dict(r) for r in records))
    loaded = load_prompts(tmp_path)
    assert loaded == records
    assert prompts_sha256(loaded) == prompts_sha256(records)
    loaded[0].text += " "
    assert prompts_sha256(loaded) != prompts_sha256(records)


@pytest.mark.parametrize("answer, labels, label, status", [
    ("I choose: a). My reasoning: only a pays in time.", ("a)", "b)"), "a)", "ok"),
    ("I choose: b", ("a)", "b)"), "b)", "ok"),
    ("I choose: **2)**. My reasoning: ...", ("1)", "2)"), "2)", "ok"),
    ("i choose: (A)", ("a)", "b)"), "a)", "ok"),
    ("I choose: `b)`", ("a)", "b)"), "b)", "ok"),
    ("I choose: a). ... To restate, I choose: a)", ("a)", "b)"), "a)", "ok"),
    ("I choose: a) ... no wait, I choose: b)", ("a)", "b)"), None, "conflicting"),
    ("I choose: and then reconsider", ("a)", "b)"), None, "format_failure"),   # "and" is not label "a"
    ("I choose: <a) or b)>", ("a)", "b)"), None, "format_failure"),          # the template echoed back
    ("Option a) is better.", ("a)", "b)"), None, "format_failure"),
    ("I choose: c)", ("a)", "b)"), None, "format_failure"),
    (None, ("a)", "b)"), None, "no_answer"),
])
def test_parse_choice(answer, labels, label, status):
    parsed = parse_choice(answer, labels)
    print(f"{answer!r:<55} -> {parsed}")
    assert (parsed.label, parsed.status) == (label, status)


def test_score():
    rec = build_verification_grid()[0]                 # near first, near_only: the near option is correct
    assert rec.correct_option == "near"
    good = score(rec, parse_choice(f"I choose: {rec.near_label}", rec.labels))
    bad = score(rec, parse_choice(f"I choose: {rec.far_label}", rec.labels))
    none = score(rec, parse_choice("no idea", rec.labels))
    assert (good.chosen_option, good.correct) == ("near", True)
    assert (bad.chosen_option, bad.correct) == ("far", False)
    assert (none.chosen_option, none.correct) == (None, None)
