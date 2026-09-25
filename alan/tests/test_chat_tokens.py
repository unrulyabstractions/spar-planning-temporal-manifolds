"""Asserts the exact captured token strings per model family. Tokenizer-only; runs offline."""
import os
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402

from ptm.chat import encode_user_turn, find_choice, label_first_token_id, position_labels  # noqa: E402

QWEN3_NOTHINK_WINDOW = ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n"]
QWEN3_THINK_WINDOW = ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"]


@pytest.fixture(scope="module", params=["Qwen/Qwen3-14B", "Qwen/Qwen3-8B"])
def tok(request):
    try:
        return AutoTokenizer.from_pretrained(request.param)
    except Exception as e:  # not cached
        pytest.skip(f"{request.param} not available offline: {e}")


def test_transition_window_nothink(tok):
    enc = encode_user_turn(tok, "SITUATION: x\nTASK: y", enable_thinking=False)
    print("transition tokens:", enc.transition_tokens)
    assert enc.transition_tokens == QWEN3_NOTHINK_WINDOW
    assert enc.input_ids[enc.transition_start] == tok.convert_tokens_to_ids("<|im_end|>")
    assert enc.transition_positions == list(range(enc.prompt_len - 9, enc.prompt_len))


def test_transition_window_think(tok):
    enc = encode_user_turn(tok, "hello", enable_thinking=True)
    assert enc.transition_tokens == QWEN3_THINK_WINDOW


def test_user_text_with_im_end_lookalike_is_not_confused(tok):
    enc = encode_user_turn(tok, "the string im_end appears here", enable_thinking=False)
    assert enc.transition_tokens == QWEN3_NOTHINK_WINDOW


def test_labels():
    assert position_labels(9, 3) == ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "R0", "R1", "R2"]


def test_choice_readout(tok):
    ia, ib = label_first_token_id(tok, "a)"), label_first_token_id(tok, "b)")
    assert ia != ib
    resp = tok("I choose: b). My reasoning: a short horizon favors a) here.", add_special_tokens=False)["input_ids"]
    j, c = find_choice(tok, resp, "a)", "b)")
    assert c == "b" and resp[j] == ib
    assert tok.decode(resp[:j]).rstrip().endswith("I choose:")
    # an article ' a' before the prefix must not be picked up
    resp2 = tok("As a household we weigh this. I choose: a). My reasoning: ...", add_special_tokens=False)["input_ids"]
    j2, c2 = find_choice(tok, resp2, "a)", "b)")
    assert c2 == "a" and tok.decode(resp2[:j2]).rstrip().endswith("I choose:")
    assert find_choice(tok, tok("Sure, here is my plan.", add_special_tokens=False)["input_ids"], "a)", "b)") == (None, None)
    # markdown-bold label, as Qwen3-14B produces ~5% of the time
    resp3 = tok("I choose: **a) 250,000 dollars in 2 years**. My reasoning: ...", add_special_tokens=False)["input_ids"]
    j3, c3 = find_choice(tok, resp3, "a)", "b)")
    assert c3 == "a" and tok.decode([resp3[j3]]) == "a", [tok.decode([t]) for t in resp3[:8]]
    print("bold-label tokens:", [tok.decode([t]) for t in resp3[:8]], "choice index", j3)
