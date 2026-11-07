"""Stage 0 of 28_spec_choice4_side.md on synthetic answers.json rows and a synthetic generation cache (the
cached Qwen3.5-0.8B tokenizer): loop flag, last committed label, ids from the cache or re-tokenised, and
the predictions."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import think_budget_stage0 as stage0  # noqa: E402
from spar_horizon import model_io as M  # noqa: E402
from spar_horizon.config import Settings  # noqa: E402
from test_exp14_states import GENERATION, tok  # noqa: E402,F401

LABELS = ["a", "b", "c", "d"]
OPTIONS = [{"label": lab, "half": half, "rank_in_half": rank}
           for lab, half, rank in (("a", "near", 1), ("b", "near", 2), ("c", "far", 1), ("d", "far", 2))]
BLOCK = "Wait, let me reconsider the whole problem from the start and compare every option again. " * 2


def make_answer(uid, think_text, forced=True, label="d", truth="b"):
    return {"horizon": 1.0, "prompt": f"Prompt {uid}.", "answer": f" {label}).", "regime": "near", "truth": "near",
            "sample_uid": uid, "pair_id": uid, "duration_id": "1 year", "rendered_unit": "years",
            "delay_scale": "days-weeks", "options": OPTIONS, "labels": LABELS, "near_labels": ["a", "b"],
            "far_labels": ["c", "d"], "truth_label": truth, "n_options": 4, "label": label, "forced": forced,
            "think_len": 0, "think_text": think_text}


@pytest.fixture
def cfg(tmp_path):
    return Settings(**{**GENERATION, "bank": "choice4", "cache_dir": str(tmp_path / "gen_cache")})


def store(tok, cfg, answer, think_text):
    """Put the generation of `answer` in the cache, as E0 would have (think ids exactly as generated)."""
    prompt = tok.encode(M.chat_text(tok, answer["prompt"], cfg), add_special_tokens=False)
    think = tok.encode(think_text, add_special_tokens=False)
    ids = torch.tensor(prompt + think + [1, 2, 3])
    M.cache_store(cfg, prompt, None, M.Generation(ids, len(prompt), len(prompt) + len(think) + 1, len(think), True, 2, 0))
    answer["think_len"] = len(think)
    return think


def test_a_block_repeated_three_times_in_the_last_4000_characters_is_a_loop():
    block = "x" * 100 + "unique one " + "y" * 39
    assert stage0.has_loop("start " + block * 3)
    assert not stage0.has_loop("start " + block * 2 + "z" * 200)
    assert not stage0.has_loop(block * 3 + "".join(f"{i} varied filler. " for i in range(300)))   # outside the window


def test_the_last_commitment_in_the_final_1500_characters_wins():
    labels = LABELS
    assert stage0.last_committed_label("so the best is b. But wait. Therefore the answer is d)", labels) == "d"
    assert stage0.last_committed_label("I think I would choose option c.", labels) == "c"
    assert stage0.last_committed_label("So option b is the best.", labels) == "b"
    assert stage0.last_committed_label("So the best is b." + " filler" * 400, labels) is None   # older than 1,500 chars


def test_hypotheses_comparisons_and_the_article_are_not_commitments():
    labels = LABELS
    assert stage0.last_committed_label("If you choose option a, you get 15k.", labels) is None
    assert stage0.last_committed_label("Suppose we pick b, then the delay is long.", labels) is None
    assert stage0.last_committed_label("The best is a bigger payout, I would choose a larger sum.", labels) is None
    assert stage0.last_committed_label("option a is better than option d", labels) is None
    assert stage0.last_committed_label("the answer is a)", labels) == "a"
    assert stage0.last_committed_label("the best is h", labels) is None          # not a label of this bank
    assert stage0.last_committed_label("the best is h", tuple("abcdefgh")) == "h"


def test_ids_come_from_the_cache_and_a_missing_entry_is_retokenised(tok, cfg):
    cached = make_answer("c0", "Okay. So the best is b.")
    missing = make_answer("c1", "Hmm, let me think about it for a moment.")
    natural = make_answer("c2", "Short.", forced=False, label="b")
    cached_think = store(tok, cfg, cached, cached["think_text"])
    record, prompts, thinks = stage0.build([natural, cached, missing], tok, cfg)
    assert [r["sample_uid"] for r in record["rows"]] == ["c0", "c1"] and [r["bank_index"] for r in record["rows"]] == [1, 2]
    assert [r["ids_source"] for r in record["rows"]] == ["cache", "retokenised"]
    assert thinks[0] == cached_think and thinks[1] == tok.encode(missing["think_text"], add_special_tokens=False)
    assert prompts[0] == tok.encode(M.chat_text(tok, cached["prompt"], cfg), add_special_tokens=False)
    assert record["rows"][0]["tentative"] == "b" and record["rows"][0]["orig_label"] == "d"
    assert record["rows"][0]["option_info"]["c"] == ["far", 1]


def test_a_cache_entry_that_disagrees_with_answers_json_is_an_error(tok, cfg):
    answer = make_answer("c0", "Okay. So the best is b.")
    store(tok, cfg, answer, "A different trace.")
    with pytest.raises(ValueError, match="does not match"):
        stage0.build([answer], tok, cfg)


def test_reference_splits_forced_and_natural_closes():
    forced = [make_answer(f"f{i}", "t", label="d", truth="a") for i in range(3)]
    natural = [make_answer("n0", "t", forced=False, label="b", truth="b")]
    ref = stage0.reference(forced + natural)
    assert ref["forced"]["accuracy"] == {"n": 3, "exact": 0, "half": 0, "within_half": 0}
    assert ref["natural"]["accuracy"] == {"n": 1, "exact": 1, "half": 1, "within_half": 0}
    assert ref["forced"]["position"]["pick_rate"] == [0, 0, 0, 1] and ref["forced"]["position"]["excess_late"] == pytest.approx(1.0)


def test_predictions_say_not_loop_closes_and_the_committed_label_wins(tok, cfg):
    looping = make_answer("l0", "Start. " + BLOCK * 30, truth="a")
    calm = make_answer("l1", "Okay. So the best is b.", truth="b")
    record, _, _ = stage0.build([looping, calm], tok, cfg)
    pred = stage0.predictions(record)
    assert [(p["predicted_close"], p["predicted_label"]) for p in pred["per_trace"]] == [(False, "d"), (True, "b")]
    assert pred["total"]["n_loop"] == 1 and pred["total"]["predicted_close_rate"] == 0.5
    assert pred["total"]["forced_label_agrees_with_tentative"] == [0, 1] and pred["total"]["tentative_is_truth"] == [1, 1]
    text = stage0.predictions_text(pred, record)
    assert "loop flag" in text and "1/2" in text
    json.dumps(pred)


def test_the_hand_check_sample_is_seeded_and_capped_at_twenty(tok, cfg):
    answers = [make_answer(f"s{i}", f"Okay {i}. So the best is b. More text.") for i in range(30)]
    record, _, _ = stage0.build(answers, tok, cfg)
    text = stage0.tentative_check_text(answers, record)
    assert text.count("== s") == 20 and text == stage0.tentative_check_text(answers, record)


def test_concat_roundtrips_through_offsets():
    flat, offsets = stage0.concat([[1, 2, 3], [], [4]])
    assert flat.tolist() == [1, 2, 3, 4] and offsets.tolist() == [0, 3, 3, 4]
    assert np.issubdtype(flat.dtype, np.int32)
