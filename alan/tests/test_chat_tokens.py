"""Asserts the exact captured token strings per model family. Tokenizer-only; runs offline."""
import os
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402

from ptm.chat import encode_user_turn, end_of_turn_id, find_choice, label_first_token_id, position_labels  # noqa: E402

QWEN3_NOTHINK_WINDOW = ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n"]
QWEN3_THINK_WINDOW = ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"]
# Qwen3.5 opens the think block in the template itself when thinking is enabled (7 tokens); the no-think window is identical to Qwen3.
# Qwen3.8 (same Qwen3_5 architecture class, identical vocab) renders the no-think turn identically to Qwen3.5; its thinking
# template differs earlier in the prompt but ends in the same 7-token window.
QWEN35_THINK_WINDOW = QWEN3_THINK_WINDOW + ["<think>", "\n"]
# Gemma 4 (-it): no-think prefill is an empty thought channel; thinking mode adds `<|think|>` in a system turn
# before the user turn, so the window after the user's end-of-turn is the bare generation prompt.
GEMMA4_NOTHINK_WINDOW = ["<turn|>", "\n", "<|turn>", "model", "\n", "<|channel>", "thought", "\n", "<channel|>"]
GEMMA4_THINK_WINDOW = ["<turn|>", "\n", "<|turn>", "model", "\n"]

MODELS = ["Qwen/Qwen3-14B", "Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.8-27B", "google/gemma-4-31B-it"]


def is_gemma4(tok):
    return "gemma-4" in tok.name_or_path


def nothink_window(tok):
    return GEMMA4_NOTHINK_WINDOW if is_gemma4(tok) else QWEN3_NOTHINK_WINDOW


@pytest.fixture(scope="module", params=MODELS)
def tok(request):
    try:
        return AutoTokenizer.from_pretrained(request.param)
    except Exception as e:  # not cached
        pytest.skip(f"{request.param} not available offline: {e}")


def test_transition_window_nothink(tok):
    enc = encode_user_turn(tok, "SITUATION: x\nTASK: y", enable_thinking=False)
    print(tok.name_or_path, "transition tokens:", enc.transition_tokens)
    assert enc.transition_tokens == nothink_window(tok)
    assert enc.input_ids[enc.transition_start] == end_of_turn_id(tok)
    assert enc.transition_positions == list(range(enc.prompt_len - 9, enc.prompt_len))


def test_transition_window_think(tok):
    enc = encode_user_turn(tok, "hello", enable_thinking=True)
    expected = GEMMA4_THINK_WINDOW if is_gemma4(tok) else QWEN35_THINK_WINDOW if ("Qwen3.5" in tok.name_or_path or "Qwen3.8" in tok.name_or_path) else QWEN3_THINK_WINDOW
    print(tok.name_or_path, "thinking-mode transition tokens:", enc.transition_tokens)
    assert enc.transition_tokens == expected


def test_user_text_with_end_of_turn_lookalike_is_not_confused(tok):
    enc = encode_user_turn(tok, "the strings im_end, turn| and <turn appear here", enable_thinking=False)
    assert enc.transition_tokens == nothink_window(tok)


def test_single_bos(tok):
    """The template text carries its own BOS (Gemma); encoding must not add a second one."""
    enc = encode_user_turn(tok, "SITUATION: x", enable_thinking=False)
    if tok.bos_token_id is not None and is_gemma4(tok):
        assert enc.input_ids[0] == tok.bos_token_id and enc.input_ids.count(tok.bos_token_id) == 1
    else:
        assert tok.bos_token_id is None or tok.bos_token_id not in enc.input_ids, "unexpected BOS in a Qwen prompt"


def test_end_of_turn_id_rejects_unknown_family():
    class Fake:
        unk_token_id = 0
        def convert_tokens_to_ids(self, t):
            return 0
    with pytest.raises(ValueError, match="unverified chat template family"):
        end_of_turn_id(Fake())


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


def test_find_anchor_first_reasoning_token(tok):
    from ptm.chat import find_anchor
    ids = tok("I choose: a). My reasoning: Over a 10-year horizon the larger sum wins.", add_special_tokens=False)["input_ids"]
    j = find_anchor(tok, ids, "My reasoning:")
    assert j is not None and tok.decode(ids[:j]).rstrip().endswith("My reasoning:") and tok.decode([ids[j]]).strip() == "Over"
    assert find_anchor(tok, tok("I choose: a). No explanation.", add_special_tokens=False)["input_ids"], "My reasoning:") is None
    assert find_anchor(tok, tok("I choose: a). My reasoning:", add_special_tokens=False)["input_ids"], "My reasoning:") is None   # nothing follows
