"""Tests for label_audit.py. Run (from evaluation/): python -m pytest -q test_label_audit.py"""

import json
from pathlib import Path

import pytest

import label_audit as la
from pbank.durations import Duration
from pbank.records import read_jsonl, record_to_dict
from pbank.t0 import CONSTRAINT_PHRASINGS, make_record

T0_BANK = Path(__file__).resolve().parents[1] / "prompt_bank" / "data" / "t0_s0"
TIES = {"x0004", "x0042", "x0224", "x0234", "x0311"}
NEAR_DELAY_REWRITES = {"x0238", "x0272", "x0290", "x0340"}


def record(uid="r1", horizon="2 years", near="1 week", far="5 years", near_first=True, phrasing_id=0, **kw) -> dict:
    return record_to_dict(make_record(
        uid=uid, split="verification", near_delay=Duration.parse(near), far_delay=Duration.parse(far),
        near_reward=1000, multiplier=5, near_first=near_first, labels=("a)", "b)"),
        horizon=None if horizon is None else Duration.parse(horizon),
        grid_value=None if horizon is None else Duration.parse(horizon), phrasing_id=phrasing_id, pair_id="item", **kw))


@pytest.fixture(scope="module")
def t0_audit():
    return la.audit_bank(read_jsonl(T0_BANK / "prompts.jsonl"))


def test_t0_has_no_mismatch_and_no_gap(t0_audit):
    assert t0_audit["n"] == 700
    assert all(uids == [] for uids in t0_audit["mismatches"].values())
    assert t0_audit["unset"] == {} and t0_audit["unparsed"] == []


def test_t0_ties_and_near_delay_rewrites_flagged(t0_audit):
    flagged = {d["uid"]: d["kinds"] for d in t0_audit["debatable"]}
    assert {u for u, k in flagged.items() if "tie" in k} == TIES
    assert all("near_boundary" in flagged[u] and "rewrite" in flagged[u] for u in NEAR_DELAY_REWRITES)
    assert sum("rewrite" in k for k in flagged.values()) == 172


def test_null_twin_is_not_a_gap():
    audit = la.audit_bank([record(horizon=None, phrasing_id=None)])
    assert audit["unset"] == {} and audit["mismatches"]["regime"] == []
    assert audit["labels"]["phrasing_id"]["null"] == 1


@pytest.mark.parametrize("phrasing_id", range(4))
def test_phrasing_found_in_text_with_wildcard_horizon(phrasing_id):
    row = record(phrasing_id=phrasing_id, horizon="2.24 years")
    assert la.phrasing_in_text(row["text"]) == phrasing_id
    system_message = CONSTRAINT_PHRASINGS[phrasing_id].format(H="3 months")
    assert la.phrasing_in_text(f"[system]\n{system_message}\n\n[user]\nTASK") == phrasing_id
    assert la.phrasing_in_text("no constraint here") is None


def test_null_phrasing_id_with_phrasing_in_text_is_unset():
    row = {**record(), "phrasing_id": None}
    audit = la.audit_bank([row])
    assert audit["unset"] == {"phrasing_id": ["r1"]}


def test_wrong_phrasing_id_is_a_mismatch():
    audit = la.audit_bank([{**record(phrasing_id=0), "phrasing_id": 2}])
    assert audit["mismatches"]["phrasing_id"] == ["r1"]


def test_missing_key_counted_and_reported():
    row = record()
    del row["near_first"]
    audit = la.audit_bank([row])
    assert audit["labels"]["near_first"] == {"present": 0, "null": 0, "missing": 1}
    assert audit["unset"]["near_first"] == ["r1"]


def test_regime_and_correct_option_rechecked_from_delays():
    row = record(horizon="2 years")                     # near pays, far (5 years) does not
    assert row["regime"] == "near_only"
    bad = {**row, "regime": "both", "correct_option": "far", "correct_label": "b)"}
    audit = la.audit_bank([row, {**bad, "uid": "bad"}])
    assert audit["mismatches"]["regime"] == ["bad"]
    assert audit["mismatches"]["correct_option"] == ["bad"]
    assert audit["mismatches"]["correct_label"] == ["bad"]


@pytest.mark.parametrize("near_first, correct_label", [(True, "a)"), (False, "b)")])
def test_correct_label_follows_the_option_order(near_first, correct_label):
    row = record(near_first=near_first)
    assert row["correct_label"] == correct_label
    flipped = {**row, "correct_label": "b)" if correct_label == "a)" else "a)"}
    assert la.audit_bank([row])["mismatches"]["correct_label"] == []
    assert la.audit_bank([flipped])["mismatches"]["correct_label"] == ["r1"]


def test_tie_flagged_and_recomputed_flag_checked():
    row = record(horizon="5 years", far="5 years")
    assert row["horizon_equals_a_delay"] and row["regime"] == "both"
    audit = la.audit_bank([row, {**row, "uid": "bad", "horizon_equals_a_delay": False}])
    assert [d["uid"] for d in audit["debatable"]] == ["r1", "bad"]
    assert all(d["kinds"] == ["tie"] for d in audit["debatable"])
    assert audit["mismatches"]["horizon_equals_a_delay"] == ["bad"]


def test_horizon_within_one_percent_is_near_boundary_but_two_percent_is_not():
    near = record(uid="close", horizon="183 days", near="6 months")
    far = record(uid="far", horizon="1.5 years", near="6 months")
    audit = la.audit_bank([near, far])
    assert {d["uid"]: d["kinds"] for d in audit["debatable"]} == {"close": ["near_boundary"]}


def test_without_reward_values_both_regime_has_no_correct_option():
    row = record(horizon="10 years")
    row["near"]["reward"] = row["far"]["reward"] = row["reward_multiplier"] = None
    row["correct_option"] = row["correct_label"] = None
    audit = la.audit_bank([row])
    assert row["regime"] == "both"
    assert all(uids == [] for uids in audit["mismatches"].values()) and "correct_option" not in audit["unset"]


def test_decades_and_centuries_parse():
    assert la.parse_duration("1.5 centuries").days == Duration.parse("150 years").days
    assert la.parse_duration("1 decade").days == Duration.parse("10 years").days
    assert la.parse_duration("3 months").days == Duration.parse("3 months").days


def test_evaltype_keys_expected_in_notes():
    row = {**record(), "tier": "evaltype", "notes": {"type_id": "t", "variant": "base"}}
    audit = la.audit_bank([row])
    assert audit["unset"]["notes.twin_id"] == ["r1"] and "notes.type_id" not in audit["unset"]
    assert audit["evaltype_keys"]["domain"]["missing"] == 1


def test_main_writes_json_and_markdown(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [record(), record("r2", near_first=False)]))
    la.main([str(bank), "--out", str(tmp_path / "out")])
    audits = json.loads((tmp_path / "out" / "label_audit.json").read_text())
    assert audits["bank"]["n"] == 2
    assert "## bank" in (tmp_path / "out" / "label_audit.md").read_text()
