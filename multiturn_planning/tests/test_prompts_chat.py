import math

import pytest

from mtp.chat import N_PRE, cut_step_reply, encode_prompt, messages_for_turn, pre_window, reply_positions, turn_positions
from mtp.prompts import (HORIZONS, SCENARIOS, WORDINGS, build_specs, outline_mentions_time,
                         parse_duration_years, parse_step)


def test_spec_counts_and_uniqueness():
    specs = build_specs(16)
    hor = [s for s in specs if s.condition == "horizon"]
    none = [s for s in specs if s.condition == "none"]
    assert len(hor) == len(SCENARIOS) * len(HORIZONS) * len(WORDINGS) == 288
    assert len(none) == len(SCENARIOS) * 16
    assert len({s.conv_id for s in specs}) == len(specs)
    assert all(s.greedy for s in hor) and not any(s.greedy for s in none)


def test_horizon_text_only_in_horizon_condition():
    for s in build_specs(2):
        if s.condition == "horizon":
            assert s.h_target_text in s.first_user
            assert math.isclose(parse_duration_years(s.h_target_text), s.h_target_years, rel_tol=0.02)
        else:
            assert parse_duration_years(s.first_user.split("\n\n")[0]) is None   # no duration in the goal text


@pytest.mark.parametrize("text,years", [
    ("18 months", 1.5), ("2 years", 2.0), ("6-12 months", 1.0), ("0–6 months", 0.5), ("three weeks", 21 / 365.25),
    ("1.5 years", 1.5), ("a decade", 10.0), ("5 to 10 years", 10.0), ("48 hours", 48 / 8766), ("no idea", None),
])
def test_parse_duration(text, years):
    got = parse_duration_years(text)
    assert (got is None and years is None) or math.isclose(got, years, rel_tol=1e-3)


def test_parse_step_formats():
    r = "Step 3: Expand the menu\nTime horizon: 9 months\nDetails: Add seasonal items."
    assert parse_step(r) == {"step_no": 3, "h_step_text": "9 months", "h_step_years": 0.75}
    r = "**Step 2: Hire staff**\n**Time horizon:** 2 years\nDetails: ..."
    assert parse_step(r)["h_step_years"] == 2.0
    assert parse_step("Plan Completed")["h_step_years"] is None
    assert outline_mentions_time("1. Audit (first 3 months)") and not outline_mentions_time("1. Audit\n2. Build")


# ---- tokenizer-level checks (Qwen3 family; the 1.7B tokenizer is the 14B/32B tokenizer) ------------------

@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    except OSError:
        pytest.skip("Qwen3 tokenizer not available offline")


def test_pre_window_tokens_every_turn(tok):
    spec = build_specs(1)[0]
    replies = []
    for k in range(1, 4):
        ids = encode_prompt(tok, messages_for_turn(spec.first_user, replies))
        win = [tok.decode([ids[i]]) for i in pre_window(tok, ids)]
        assert win == ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n"]
        assert len(win) == N_PRE
        replies.append(f"Step {k}: X\nTime horizon: {k} years\nDetails: y")


def test_history_drops_empty_think_blocks(tok):
    """Earlier assistant turns are rendered without the think block (why we capture per turn)."""
    spec = build_specs(1)[0]
    text = tok.decode(encode_prompt(tok, messages_for_turn(spec.first_user, ["A1", "A2"])))
    assert text.count("<think>") == 1 and text.rstrip().endswith("</think>")


def test_reply_positions_point_at_header_value_end(tok):
    reply = "Step 2: Hire staff\nTime horizon: 18 months\nDetails: Recruit two bakers.<|im_end|>"
    ids = tok(reply, add_special_tokens=False)["input_ids"]
    pos = reply_positions(tok, ids)
    assert ":" in tok.decode([ids[pos["H"]]])
    assert tok.decode(ids[: pos["V"] + 1]).endswith("18 months")
    assert tok.decode([ids[pos["E"]]]) == "<|im_end|>"
    prompt = encode_prompt(tok, messages_for_turn("hi", []))
    full = turn_positions(tok, prompt, ids)
    assert full["U"] == full["P0"] - 1 and full["E"] == len(prompt) + pos["E"]


def test_cut_step_reply_keeps_one_step(tok):
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    one = "Step 1: A\nTime horizon: 2 years\nDetails: x."
    ids = tok(one + "\n\nStep 2: B\nTime horizon: 3 years\nDetails: y.<|im_end|>", add_special_tokens=False)["input_ids"]
    kept, cut = cut_step_reply(tok, ids)
    assert cut and kept[-1] == im_end and tok.decode(kept[:-1]).rstrip() == one
    ids = tok(one + "\n\nPlan Completed<|im_end|>", add_special_tokens=False)["input_ids"]
    kept, cut = cut_step_reply(tok, ids)
    assert cut and tok.decode(kept[:-1]).rstrip() == one
    ids = tok(one + "<|im_end|>", add_special_tokens=False)["input_ids"]
    assert cut_step_reply(tok, ids) == (ids, False)
