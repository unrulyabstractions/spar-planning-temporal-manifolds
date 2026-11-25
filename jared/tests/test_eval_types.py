"""Contract tests for every registered eval type (spar_horizon/eval_types). Pure python, plus the
Qwen3-8B tokenizer (cached locally, no weights) for the chat-template checks."""

import pytest

from spar_horizon.config import Settings
from spar_horizon.eval_types import TYPES, build
from spar_horizon.eval_types.common import PLAN, SWAP_PER_REGIME, TWIN_PER_REGIME
from spar_horizon.full_bank import regime
from spar_horizon.model_io import chat_messages, chat_text, last_user_text, suffix_length

TYPE_IDS = sorted(TYPES)


@pytest.fixture(scope="module", params=TYPE_IDS)
def records(request):
    return request.param, build(request.param)


def test_types_registered():
    assert "distractor_temporal" in TYPES


def test_record_counts_and_unique_ids(records):
    type_id, recs = records
    plan = TYPES[type_id].plan
    bases = [r for r in recs if r["variant"] == "base"]
    assert len(bases) == len(plan)
    assert len({r["sample_uid"] for r in recs}) == len(recs)
    assert {r["variant"] for r in recs} <= {"base", "swap", "twin"}
    for name in ("near", "both"):
        n = sum(p == name for p in plan)
        assert sum(r["variant"] == "swap" and r["regime"] == name for r in recs) <= min(n, SWAP_PER_REGIME)
        assert sum(r["variant"] == "twin" and r["regime"] == name for r in recs) <= min(n, TWIN_PER_REGIME)


def test_deterministic(records):
    type_id, recs = records
    assert build(type_id) == recs
    assert build(type_id, seed=1) != recs


def test_labels_and_truth(records):
    _, recs = records
    for r in recs:
        assert r["truth_labels"] == sorted(set(r["truth_labels"]))
        assert set(r["truth_labels"]) <= set(r["labels"]), r["sample_uid"]
        assert sorted(o["label"] for o in r["options"]) == sorted(r["labels"])
        if r["answer_kind"] == "label":
            text = last_user_text(r["prompt"]) if isinstance(r["prompt"], list) else r["prompt"]
            whole = text if isinstance(r["prompt"], str) else "\n".join(m["content"] for m in r["prompt"])
            for o in r["options"]:
                assert f"{o['label']})" in whole, (r["sample_uid"], o["label"])
        if r["answer_kind"] == "label" and r["variant"] != "twin":
            assert r["choice_prefix"].rstrip(":").strip('"') in last_user_text(r["prompt"]), r["sample_uid"]


def test_swap_keeps_options_and_truth(records):
    _, recs = records
    base = {r["twin_id"]: r for r in recs if r["variant"] == "base"}
    for s in (r for r in recs if r["variant"] == "swap"):
        b = base[s["twin_id"]]
        by_text = lambda rec: {o["text"]: o["label"] for o in rec["options"]}
        assert set(by_text(s)) == set(by_text(b))
        assert by_text(s) != by_text(b), "the swap must change which label each option has"
        truth_text = lambda rec: {o["text"] for o in rec["options"] if o["label"] in rec["truth_labels"]}
        assert truth_text(s) == truth_text(b)
        assert s["prompt"] != b["prompt"]


def test_twin_states_the_horizon(records):
    _, recs = records
    for t in (r for r in recs if r["variant"] == "twin"):
        prompt = t["prompt"] if isinstance(t["prompt"], str) else "\n".join(m["content"] for m in t["prompt"])
        assert f"time horizon: {t['horizon_text']}." in prompt
        assert t["choice_prefix"] == "I choose:" and t["answer_kind"] == "label"
        assert t["truth_labels"], t["sample_uid"]


def test_two_option_regime_matches_delays(records):
    _, recs = records
    for r in recs:
        if r["truth_rule"] == "within_horizon" and len(r["options"]) == 2 and "near_label" in r:
            near = next(o for o in r["options"] if o["kind"] == "near")
            far = next(o for o in r["options"] if o["kind"] == "far")
            assert regime(r["horizon"], near["delay_years"], far["delay_years"]) == r["regime"], r["sample_uid"]


def test_message_prompts_are_well_formed(records):
    _, recs = records
    for r in recs:
        if isinstance(r["prompt"], list):
            messages = chat_messages(r["prompt"])
            assert all(m["role"] in ("system", "user", "assistant") for m in messages)
            assert all(isinstance(m["content"], str) and m["content"] for m in messages)
            assert messages[-1]["role"] == "user"


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3-8B tokenizer not cached")


def test_chat_template_renders_every_prompt(records, tok):
    _, recs = records
    cfg = Settings(bank="full", prefill="", choice_prefix="I choose:")
    reference = suffix_length(tok, "x", cfg)
    for r in recs:
        text = chat_text(tok, r["prompt"], cfg)
        assert text.endswith("<|im_start|>assistant\n")
        assert last_user_text(r["prompt"]) in text
        assert suffix_length(tok, r["prompt"], cfg) == reference, r["sample_uid"]


def test_messages_match_string_prompt_with_system(tok):
    cfg = Settings(bank="full", prefill="", choice_prefix="I choose:")
    as_messages = [{"role": "system", "content": "S."}, {"role": "user", "content": "U."}]
    assert chat_text(tok, as_messages, cfg) == chat_text(tok, "U.", cfg, system="S.")
    with pytest.raises(ValueError):
        chat_messages(as_messages, system="S.")
