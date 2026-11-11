"""Finding the answer-label token, on the real Gemma 4 and Qwen3.5 tokenizers (skipped if not cached).

Replies are built the way a model's generated ids look: thinking, the end-of-thinking token, then the answer.
"""

import pytest

from steer.sites import locate_label


def tokenizer(model_id):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except OSError:
        pytest.skip(f"{model_id} tokenizer not cached")


FAMILIES = {   # model id -> (thinking prefix tokens, end-of-thinking token)
    "Qwen/Qwen3.5-2B": ("", "</think>"),
    "google/gemma-4-31B-it": ("<|channel>", "<channel|>"),
}


def reply_ids(tok, model_id, thinking_text, answer):
    opener, eot = FAMILIES[model_id]
    ids = [tok.convert_tokens_to_ids(opener)] if opener else []
    ids += tok.encode(("thought\n" if opener else "") + thinking_text, add_special_tokens=False)
    ids += [tok.convert_tokens_to_ids(eot)]
    return ids + tok.encode(answer, add_special_tokens=False)


CASES = [   # (answer, labels, expected label char)
    ("\n\nI choose: b). My reasoning: only b pays in time.", ("a)", "b)"), "b"),
    ("I choose: **2)**. My reasoning: both pay.", ("1)", "2)"), "2"),
    ("I choose: (a). My reasoning: it is sooner.", ("a)", "b)"), "a"),
    ("I choose: and then... I choose: a). My reasoning: x.", ("a)", "b)"), "a"),
]


@pytest.mark.parametrize("model_id", list(FAMILIES))
@pytest.mark.parametrize("answer,labels,char", CASES)
def test_label_token_found(model_id, answer, labels, char):
    tok = tokenizer(model_id)
    # The thinking mentions a different choice: it must not be taken for the answer.
    ids = reply_ids(tok, model_id, "Maybe I choose: a) here? No, think again.", answer)
    site = locate_label(tok, ids, labels, thinking=True)
    token = tok.decode([ids[site.label_index]])
    before = tok.decode(ids[site.answer_start:site.label_index])
    print(f"{model_id}: {answer[:28]!r} -> label token {token!r}, preceded by {before[-14:]!r}")
    assert site.label_char == char
    assert char in token
    assert char not in before.split("choose")[-1]          # nothing between "I choose" and the label carries it


@pytest.mark.parametrize("model_id", list(FAMILIES))
def test_no_site_without_answer_or_choice(model_id):
    tok = tokenizer(model_id)
    # Thinking never closed (cut off at the token cap): no answer, even though the thinking says "I choose: a)".
    no_eot = tok.encode("thinking and I choose: a) but never closing", add_special_tokens=False)
    assert locate_label(tok, no_eot, ("a)", "b)"), thinking=True) is None
    no_choice = reply_ids(tok, model_id, "hmm", "Neither option pays in time.")
    assert locate_label(tok, no_choice, ("a)", "b)"), thinking=True) is None
    wrong_labels = reply_ids(tok, model_id, "hmm", "I choose: c).")
    assert locate_label(tok, wrong_labels, ("a)", "b)"), thinking=True) is None


def test_thinking_off():
    tok = tokenizer("Qwen/Qwen3.5-2B")
    ids = tok.encode("I choose: 1). My reasoning: sooner.", add_special_tokens=False)
    site = locate_label(tok, ids, ("1)", "2)"), thinking=False)
    print(f"thinking off: label token {tok.decode([ids[site.label_index]])!r} at {site.label_index}")
    assert site.answer_start == 0 and "1" in tok.decode([ids[site.label_index]])
