"""Tests for the boundary eval types (spar_horizon/eval_types/boundary.py): the templated multi-turn and
no-horizon types, the implied-event wrapper, and the scaffold loader (on tests/fixtures/boundary_scaffolds.json)."""

import copy
import json
import math
import warnings
from pathlib import Path

import pytest

from spar_horizon.config import Settings
from spar_horizon.eval_types import TYPES, build
from spar_horizon.eval_types.boundary import (DURATION_TEXT, FILLER, HORIZONS_PER_SCAFFOLD, INTERVAL_MARGIN,
                                             load_boundary_types)
from spar_horizon.eval_types.common import SWAP_PER_REGIME, TWIN_PER_REGIME, build_records
from spar_horizon.full_bank import regime
from spar_horizon.model_io import chat_text, last_user_text

FIXTURE = Path(__file__).parent / "fixtures" / "boundary_scaffolds.json"
TEMPLATED = ["mt_first", "mt_last", "mt_revised", "mt_long", "mt_assistant", "no_horizon", "implied_event"]
LLM = ["llm_prose", "llm_competing", "llm_document"]
PERSONA = ["persona_user", "persona_model"]
EXPLICIT_TEMPLATED = [t for t in TEMPLATED if t not in ("no_horizon", "implied_event")]


@pytest.fixture(scope="module")
def loaded():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {et.type_id: et for et in load_boundary_types(FIXTURE)}


def records_of(type_id, loaded):
    return build_records(loaded[type_id]) if type_id in loaded else build(type_id)


def bases(recs):
    return [r for r in recs if r["variant"] == "base"]


def counts(recs):
    out = {}
    for r in recs:
        out[r["regime"]] = out.get(r["regime"], 0) + 1
    return out


def delays(rec):
    near = next(o for o in rec["options"] if o["kind"] == "near")
    far = next(o for o in rec["options"] if o["kind"] == "far")
    return near["delay_years"], far["delay_years"]


def messages(rec):
    return [{"role": "user", "content": rec["prompt"]}] if isinstance(rec["prompt"], str) else rec["prompt"]


def whole_text(rec):
    return "\n".join(m["content"] for m in messages(rec))


# ---- shape of every type ----

@pytest.mark.parametrize("type_id", TEMPLATED)
def test_templated_plan_swaps_and_twins(type_id):
    recs = build(type_id)
    assert len(bases(recs)) == 48
    expected = {"null": 48} if type_id == "no_horizon" else {"near": 16, "both": 16, "none": 16}
    assert counts(bases(recs)) == expected
    swaps = [r for r in recs if r["variant"] == "swap"]
    twins = [r for r in recs if r["variant"] == "twin"]
    assert len(swaps) == 2 * SWAP_PER_REGIME
    assert len(twins) == (0 if type_id == "no_horizon" else 2 * TWIN_PER_REGIME)
    assert len({r["sample_uid"] for r in recs}) == len(recs)


@pytest.mark.parametrize("type_id", EXPLICIT_TEMPLATED + ["implied_event"])
def test_templated_types_use_48_distinct_delay_problems_and_many_horizons(type_id):
    assert len({r["horizon_text"] for r in bases(build(type_id))}) >= 10


@pytest.mark.parametrize("type_id", EXPLICIT_TEMPLATED)
def test_templated_types_share_items(type_id):
    reference = bases(build("mt_first"))
    for a, b in zip(reference, bases(build(type_id))):
        assert (a["horizon"], a["options"]) == (b["horizon"], b["options"])


@pytest.mark.parametrize("type_id", TEMPLATED + LLM + PERSONA)
def test_truth_follows_regime_and_prompts_end_with_the_format_line(type_id, loaded):
    for r in records_of(type_id, loaded):
        near, far = delays(r)
        if r["regime"] == "null":
            assert r["horizon"] is None and r["truth_labels"] == []
        else:
            assert regime(r["horizon"], near, far) == r["regime"], r["sample_uid"]
            if "interval_years" in r:
                lo, hi = r["interval_years"]
                assert lo < r["horizon"] < hi
                assert {regime(x, near, far) for x in (lo, r["horizon"], hi)} == {r["regime"]}
            expected = {"near": [r["near_label"]], "both": [x for x in r["labels"] if x != r["near_label"]],
                        "none": []}[r["regime"]]
            assert r["truth_labels"] == expected, r["sample_uid"]
        assert last_user_text(r["prompt"]).endswith(f"FORMAT: I choose: <{r['labels'][0]}) or {r['labels'][1]})>. "
                                                    "My reasoning: <1-3 sentences>"), r["sample_uid"]


@pytest.mark.parametrize("type_id", TEMPLATED + LLM + PERSONA)
def test_cue_occurs_once_in_the_stated_message(type_id, loaded):
    for r in records_of(type_id, loaded):
        if type_id == "no_horizon":
            assert r["cue_text"] is None and r["cue_message"] is None
            continue
        parts = [m["content"] for m in messages(r)]
        assert sum(p.count(r["cue_text"]) for p in parts) == 1, r["sample_uid"]
        assert r["cue_text"] in parts[r["cue_message"]]


@pytest.mark.parametrize("type_id", EXPLICIT_TEMPLATED + LLM)
def test_explicit_cue_ends_with_the_horizon_phrase_or_carries_it(type_id, loaded):
    for r in records_of(type_id, loaded):
        assert r["horizon_text"] in r["cue_text"]
        if r["variant"] == "twin":
            assert r["cue_text"] == f"for this time horizon: {r['horizon_text']}"


# ---- multi-turn types ----

def user_turns(rec):
    return [m for m in messages(rec) if m["role"] == "user"]


def test_mt_first_is_the_analyst_conversation_with_the_horizon_in_turn_one():
    for r in bases(build("mt_first")):
        turns = user_turns(r)
        assert len(turns) == 3 and [m["role"] for m in r["prompt"]] == ["user", "assistant"] * 2 + ["user"]
        assert r["horizon_text"] in turns[0]["content"] and "horizon" not in turns[2]["content"].lower()
        assert r["cue_message"] == 0


def test_mt_last_has_the_horizon_only_in_the_last_turn():
    for r in bases(build("mt_last")):
        assert len(user_turns(r)) == 3 and r["cue_message"] == 4
        assert all("horizon" not in m["content"].lower() for m in r["prompt"][:-1])
        assert r["horizon_text"] in r["prompt"][-1]["content"]


def test_mt_revised_labels_the_new_horizon():
    recs = bases(build("mt_revised"))
    different_regime = 0
    for r in recs:
        first, last = r["prompt"][0]["content"], r["prompt"][-1]["content"]
        old, new = r["old_horizon_years"], r["horizon"]
        assert abs(math.log10(old) - math.log10(new)) >= 1 - 1e-9
        assert f"horizon: {r['old_horizon_text']}." in first and f"horizon: {r['horizon_text']}." not in first
        assert last.startswith("Actually, my horizon has changed:")
        assert f"horizon: {r['horizon_text']}." in last and r["old_horizon_text"] not in last
        assert r["cue_text"] == f"for this time horizon: {r['horizon_text']}" and r["cue_message"] == 4
        near, far = delays(r)
        different_regime += regime(old, near, far) != r["regime"]
    assert different_regime >= 0.75 * len(recs)


def test_mt_long_filler_and_length():
    for r in bases(build("mt_long")):
        turns = r["prompt"]
        assert [m["role"] for m in turns] == ["user", "assistant"] * 9 + ["user"]
        assert r["horizon_text"] in turns[0]["content"] and r["cue_message"] == 0
        assert not any(r["horizon_text"] in m["content"] for m in turns[1:])
        assert len(set(r["filler_ids"])) == 8
    assert len({tuple(r["filler_ids"]) for r in bases(build("mt_long"))}) > 10


def test_filler_has_no_durations():
    for question, answer in FILLER:
        assert not DURATION_TEXT.search(question + " " + answer), question


def test_mt_assistant_horizon_only_in_an_assistant_turn():
    for r in build("mt_assistant"):
        if r["variant"] == "twin":
            continue
        roles = [m["role"] for m in r["prompt"]]
        assert roles[r["cue_message"]] == "assistant"
        for m in r["prompt"]:
            if m["role"] == "user":
                assert "horizon" not in m["content"].lower() and r["cue_text"] not in m["content"]
                assert not DURATION_TEXT.search(m["content"].split("Here are the options:")[0])


def test_no_horizon_has_no_duration_text():
    for r in build("no_horizon"):
        assert r["horizon"] is None and r["horizon_text"] is None and r["regime"] == "null"
        assert r["truth_labels"] == [] and r["truth_rule"] == "none"
        text = r["prompt"]
        assert "CONSTRAINT" not in text and "horizon" not in text.lower()
        without_options = "\n".join(line for line in text.splitlines() if not line[:2] in ("a)", "b)"))
        assert not DURATION_TEXT.search(without_options)


# ---- implied events ----

def test_implied_event_labels_are_intervals_with_clearance():
    for r in build("implied_event"):
        lo, hi = r["interval_years"]
        near, far = delays(r)
        assert lo <= r["horizon"] <= hi and (lo, hi) == (r["anchor_lo_years"], r["anchor_hi_years"])
        for o in (near, far):
            assert min(abs(math.log10(x / o)) for x in (lo, hi)) >= 0.3 - 1e-9, r["sample_uid"]
        if r["variant"] != "twin":
            assert f"time horizon: {r['anchor_text']}." in r["prompt"]
            assert r["cue_text"] == r["anchor_text"]
        else:
            assert r["context"] in r["prompt"] if r["context"] else True


# ---- scaffold loader ----

def test_loader_registers_all_five_types(loaded):
    assert set(loaded) == set(LLM + PERSONA)


@pytest.mark.parametrize("type_id", LLM)
def test_llm_types_fill_six_horizons_per_scaffold(type_id, loaded):
    recs = bases(build_records(loaded[type_id]))
    scaffolds = {r["scaffold_id"] for r in recs}
    assert len(recs) == HORIZONS_PER_SCAFFOLD * len(scaffolds)
    for sid in scaffolds:
        own = [r for r in recs if r["scaffold_id"] == sid]
        assert len({r["horizon_text"] for r in own}) == HORIZONS_PER_SCAFFOLD
        assert counts(own) == {"near": 2, "both": 2, "none": 2}
        for r in own:
            assert r["horizon_text"] in whole_text(r) and "{" not in whole_text(r).replace("{options}", "")


@pytest.mark.parametrize("type_id", PERSONA)
def test_persona_types_skip_infeasible_scaffolds_and_hide_the_horizon(type_id, loaded):
    recs = build_records(loaded[type_id])
    own = bases(recs)
    assert len(own) == 6
    assert sorted(counts(own).values()) == [2, 2, 2]
    for r in recs:
        lo, hi = r["interval_years"]
        assert r["horizon"] == pytest.approx(math.sqrt(lo * hi))
        near, far = delays(r)
        for o in (near, far):
            assert min(abs(math.log10(x / o)) for x in (lo, hi)) >= INTERVAL_MARGIN - 1e-9
        if r["variant"] != "twin":
            text = whole_text(r).split("Here are the options:")[0]
            assert not DURATION_TEXT.search(text) and "horizon" not in whole_text(r).lower()


def test_persona_user_skips_the_unplaceable_interval():
    with pytest.warns(UserWarning, match="persona_user_07"):
        types = {et.type_id: et for et in load_boundary_types(FIXTURE)}
    assert len(types["persona_user"].plan) == 6


@pytest.mark.parametrize("type_id", PERSONA)
def test_persona_twin_is_the_same_scaffold_plus_the_constraint(type_id, loaded):
    recs = build_records(loaded[type_id])
    base = {r["twin_id"]: r for r in recs if r["variant"] == "base"}
    twins = [r for r in recs if r["variant"] == "twin"]
    assert twins
    for t in twins:
        constraint = f"\nYou must select the option that provides the greatest benefit for this time horizon: {t['horizon_text']}."
        assert whole_text(t).replace(constraint, "") == whole_text(base[t["twin_id"]])
        assert t["truth_labels"] == base[t["twin_id"]]["truth_labels"] or base[t["twin_id"]]["regime"] == "none"


# ---- malformed scaffolds ----

def valid(type_id):
    return copy.deepcopy(next(s for s in json.loads(FIXTURE.read_text()) if s["type_id"] == type_id))


def set_message(scaffold, text, index=-1):
    scaffold["messages"][index]["content"] = text
    return scaffold


MALFORMED = {
    "unknown_type": lambda: {**valid("llm_prose"), "type_id": "llm_poem"},
    "no_messages": lambda: {**valid("llm_prose"), "messages": []},
    "last_not_user": lambda: {**valid("llm_prose"), "messages": valid("llm_prose")["messages"]
                              + [{"role": "assistant", "content": "ok"}]},
    "options_missing": lambda: set_message(valid("llm_prose"), "We need the money back within {horizon}."),
    "options_twice": lambda: set_message(valid("llm_prose"), "We need the money back within {horizon}. {options} {options}"),
    "options_not_in_last_message": lambda: {**valid("llm_document"), "messages": [
        {"role": "system", "content": "{options}"}, {"role": "user", "content": valid("llm_document")["messages"][-1]["content"].replace("{options}", "x")}]},
    "horizon_missing": lambda: set_message(valid("llm_prose"), "We need the money back soon. {options}"),
    "horizon_twice": lambda: set_message(valid("llm_prose"), "We need the money back within {horizon} or {horizon}. {options}"),
    "horizon_in_persona": lambda: set_message(valid("persona_user"), "I'm 24 and saving for my retirement. {horizon} {options}"),
    "cue_absent": lambda: {**valid("llm_prose"), "cue": "no such phrase {horizon}"},
    "cue_twice": lambda: set_message(valid("persona_user"), "I'm 24 and saving for my retirement. I'm 24 and saving for my retirement. {options}"),
    "cue_without_horizon": lambda: {**valid("llm_prose"), "cue": "We need the money back"},
    "persona_cue_absent": lambda: {**valid("persona_user"), "cue": "I am nobody"},
    "persona_digit_unit": lambda: set_message(valid("persona_user"), "I'm 24 and saving for my retirement in 6 months. {options}"),
    "persona_year_old": lambda: set_message(valid("persona_user"), "I'm 24 and saving for my retirement. I'm a 24-year-old. {options}"),
    "persona_week_word": lambda: set_message(valid("persona_user"), "I'm 24 and saving for my retirement, due next week. {options}"),
    "persona_daily_word": lambda: set_message(valid("persona_user"), "I'm 24 and saving for my retirement, daily. {options}"),
    "persona_decade_system": lambda: set_message(valid("persona_model"), "You advise for a decade. I'm 24.", 0),
    "interval_missing": lambda: {k: v for k, v in valid("persona_user").items() if k != "interval_years"},
    "interval_reversed": lambda: {**valid("persona_user"), "interval_years": [8, 3]},
    "interval_negative": lambda: {**valid("persona_user"), "interval_years": [-1, 3]},
    "interval_not_a_pair": lambda: {**valid("persona_user"), "interval_years": [3]},
    "interval_strings": lambda: {**valid("persona_user"), "interval_years": ["3", "8"]},
}


@pytest.mark.parametrize("case", MALFORMED)
def test_loader_rejects_malformed_scaffold_naming_it(case, tmp_path):
    scaffold = MALFORMED[case]()
    scaffold["scaffold_id"] = f"bad_{case}"
    path = tmp_path / "scaffolds.json"
    path.write_text(json.dumps([scaffold]))
    with pytest.raises(ValueError, match=f"bad_{case}"):
        load_boundary_types(path)


def test_loader_rejects_duplicate_ids(tmp_path):
    scaffolds = json.loads(FIXTURE.read_text())
    scaffolds.append(scaffolds[0])
    path = tmp_path / "scaffolds.json"
    path.write_text(json.dumps(scaffolds))
    with pytest.raises(ValueError, match=scaffolds[0]["scaffold_id"]):
        load_boundary_types(path)


def test_missing_scaffold_file_leaves_templated_types_registered():
    assert set(TEMPLATED) <= set(TYPES)


# ---- real tokenizer ----

@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.5-0.8B tokenizer not cached")


def token_covering_last_char(tok, text, cue):
    start = text.index(cue)
    last_char = start + len(cue) - 1
    offsets = tok(text, return_offsets_mapping=True)["offset_mapping"]
    return next(i for i, (s, e) in enumerate(offsets) if s <= last_char < e), offsets


@pytest.mark.parametrize("type_id", TEMPLATED + LLM + PERSONA)
def test_chat_template_holds_cue_and_format_line(type_id, loaded, tok):
    cfg = Settings(bank="full", prefill="", choice_prefix="I choose:")
    for r in records_of(type_id, loaded)[:12]:
        text = chat_text(tok, r["prompt"], cfg)
        assert "<|im_start|>assistant\n" in text[-60:]
        assert "FORMAT: I choose: <" in text
        if r["cue_text"] is None:
            continue
        assert text.count(r["cue_text"]) == 1
        index, offsets = token_covering_last_char(tok, text, r["cue_text"])
        assert offsets[index][1] <= text.index(r["cue_text"]) + len(r["cue_text"]) + 1
        assert text[offsets[index][0]:offsets[index][1]].strip() in r["cue_text"].split()[-1] + "."


def test_mt_long_is_about_1500_tokens(tok):
    cfg = Settings(bank="full", prefill="", choice_prefix="I choose:")
    lengths = [len(tok(chat_text(tok, r["prompt"], cfg))["input_ids"]) for r in bases(build("mt_long"))]
    assert all(1200 <= n <= 1900 for n in lengths), (min(lengths), max(lengths))
