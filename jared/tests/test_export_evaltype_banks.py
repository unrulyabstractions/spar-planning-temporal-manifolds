"""The eval-type banks exported in prompt_bank's schema: fields, the record mapping, conversations, and that pbank's
own loader reads them (when a prompt_bank checkout is found above this one)."""

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from pipeline.s1_data_vectors.analysis.export_evaltype_banks import (EXPORTED, constraint_phrasing_id, main,
                                                                    prompts_sha256, to_prompt_record)
from spar_horizon.eval_types import build
from spar_horizon.full_bank import CONSTRAINT_PHRASINGS

PROMPT_RECORD_FIELDS = {"uid", "tier", "split", "text", "horizon", "horizon_grid_value", "horizon_unit",
                        "horizon_years", "rewritten", "phrasing_id", "near", "far", "reward_multiplier", "near_first",
                        "labels", "regime", "correct_option", "correct_label", "horizon_equals_a_delay",
                        "h_over_far_delay", "pair_id", "notes"}


def pbank_records():
    """pbank.records from the nearest prompt_bank/ above this file (the main checkout), or skip."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "prompt_bank" / "pbank" / "records.py").exists():
            sys.path.insert(0, str(parent / "prompt_bank"))
            from pbank import records
            return records
    pytest.skip("no prompt_bank checkout above this one")


def test_exported_types_are_the_gated_ones():
    sys.path.insert(0, str(Path(__file__).parent))
    from test_eval_types_expanded import GATED
    assert set(EXPORTED) == set(GATED)


def test_record_mapping():
    recs = build("t0_w0", n_base=100)
    for rec in recs:
        row = to_prompt_record(rec)
        assert set(row) == PROMPT_RECORD_FIELDS
        assert row["uid"] == rec["sample_uid"] and row["text"] == rec["prompt"]
        assert (row["tier"], row["split"]) == ("evaltype", "transfer")
        assert row["labels"] == [f"{lab})" for lab in rec["labels"]]
        near_label = row["labels"][0] if row["near_first"] else row["labels"][1]
        assert near_label == rec["near_label"] + ")"
        assert row["regime"] == {"near": "near_only", "both": "both", "none": "none"}[rec["regime"]]
        assert row["correct_option"] == rec.get("truth")
        if row["correct_option"] is not None:
            assert row["correct_label"] == rec["truth_labels"][0] + ")"
            assert (row["correct_label"] == near_label) == (row["correct_option"] == "near")
        assert row["h_over_far_delay"] == pytest.approx(rec["horizon"] / row["far"]["delay_years"])
        assert (row["h_over_far_delay"] < 1) == (row["regime"] != "both")
        assert row["notes"] == {k: v for k, v in rec.items() if k != "prompt"}
        assert f'{row["near"]["reward"]:,}' in row["text"] and row["near"]["delay"] in row["text"]


@pytest.mark.parametrize("phrasing_id", range(len(CONSTRAINT_PHRASINGS)))
@pytest.mark.parametrize("horizon", ["2 months", "0.7 years", "84 minutes"])
def test_constraint_phrasing_found_in_text(phrasing_id, horizon):
    line = CONSTRAINT_PHRASINGS[phrasing_id].format(horizon=horizon)
    assert constraint_phrasing_id(f"TASK: x\nCONSTRAINT: {line}\nACTION: y") == phrasing_id
    assert constraint_phrasing_id(f"[system]\n{line}\n\n[user]\nz") == phrasing_id
    assert constraint_phrasing_id("TASK: x\nACTION: y") is None


def test_phrasing_id_is_set_exactly_where_the_text_has_a_constraint_phrasing():
    """system_prompt keeps the phrasing in its system message; t0_w3_clear words the constraint differently
    (money at the end of the horizon), so it has none."""
    rows = [to_prompt_record(r) for r in build("system_prompt", n_base=100)]
    conversation = [r for r in rows if "messages" in r["notes"]]
    assert conversation and {r["phrasing_id"] for r in conversation} == {0}
    assert all((r["phrasing_id"] == 0) == (CONSTRAINT_PHRASINGS[0].format(horizon=r["horizon"]) in r["text"])
               for r in rows)
    assert {to_prompt_record(r)["phrasing_id"] for r in build("t0_w3_clear", n_base=100)} == {None}


def test_pairs_are_the_two_option_orders():
    rows = [to_prompt_record(r) for r in build("city_values", n_base=100)]
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair_id"], []).append(r)
    for members in pairs.values():
        variants = sorted(r["notes"]["variant"] for r in members)
        assert variants in (["base"], ["base", "swap"], ["twin"])
        if len(members) == 2:
            assert members[0]["near_first"] != members[1]["near_first"]
    assert sum(len(m) == 2 for m in pairs.values()) == 40


def test_conversation_kept_in_notes():
    rec = next(r for r in build("system_prompt", n_base=100) if not isinstance(r["prompt"], str))
    row = to_prompt_record(rec)
    assert row["notes"]["messages"] == rec["prompt"]
    assert isinstance(row["text"], str)
    for m in rec["prompt"]:                          # the transcript holds every turn, so the hash covers it
        assert m["content"] in row["text"]


def test_types_without_values_or_truth():
    rows = [to_prompt_record(r) for r in build("city_no_values", n_base=100)]
    no_values = [r for r in rows if r["notes"]["variant"] != "twin"]
    assert all(r["near"]["reward"] is None and r["reward_multiplier"] is None for r in no_values)
    assert {r["correct_option"] for r in no_values if r["regime"] == "both"} == {None}


def test_export_loads_with_pbank(tmp_path):
    main(["--types", "t0_w0", "multiturn_analyst", "--out", str(tmp_path)])
    records = pbank_records()
    assert {f.name for f in dataclasses.fields(records.PromptRecord)} == PROMPT_RECORD_FIELDS
    for type_id, n in (("t0_w0", 140), ("multiturn_analyst", 160)):
        loaded = records.load_prompts(tmp_path / type_id)
        manifest = json.loads((tmp_path / type_id / "manifest.json").read_text())
        assert len(loaded) == n == manifest["n_prompts"]
        assert records.prompts_sha256(loaded) == manifest["prompts_sha256"]
        assert manifest["conversation"] == (type_id == "multiturn_analyst")
        for r in loaded:
            if r.correct_label is not None:
                assert r.option_for_label(r.correct_label) == r.correct_option


def test_hash_matches_pbank_formula():
    rows = [{"uid": "u1", "text": "a"}, {"uid": "u2", "text": "b\nc"}]
    records = pbank_records()
    assert prompts_sha256(rows) == records.prompts_sha256([type("R", (), r) for r in rows])
