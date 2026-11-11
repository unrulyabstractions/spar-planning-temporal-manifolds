"""Splitting a generation into thinking and answer, with the real tokenizers of the model families in use.

Tokenizer-only (no weights, no GPU). Each test is skipped if that tokenizer is not in the local Hugging Face
cache; run with HF_HUB_OFFLINE=1 to make sure nothing is downloaded.
"""

import pytest

from pbank.local_model import split_reply


def tokenizer(model_id):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(model_id)
    except OSError:
        pytest.skip(f"{model_id} tokenizer not cached")


def ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def test_gemma4_thinking_template_and_split():
    tok = tokenizer("google/gemma-4-31B-it")
    chat = tok.apply_chat_template([{"role": "user", "content": "Q"}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)
    print(repr(chat))
    assert "<|think|>" in chat and chat.endswith("<|turn>model\n")     # thinking on; the model opens its channel
    generated = ids(tok, "<|channel>thought\nOnly a pays within 8 years.<channel|>I choose: a). My reasoning: x")
    thought, answer = split_reply(tok, generated, thinking=True)
    print(f"thinking={thought!r}\nanswer={answer!r}")
    assert thought == "Only a pays within 8 years." and answer == "I choose: a). My reasoning: x"
    # Never closed: all of it is thinking and there is no answer.
    assert split_reply(tok, ids(tok, "<|channel>thought\nStill thinking about it"), thinking=True)[1] is None


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-2B"])
def test_qwen35_thinking_template_and_split(model_id):
    tok = tokenizer(model_id)
    chat = tok.apply_chat_template([{"role": "user", "content": "Q"}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)
    print(repr(chat))
    assert chat.endswith("<think>\n")                                 # the template opens the thinking block
    generated = ids(tok, "Both pay; b is larger.\n</think>\n\nI choose: b). My reasoning: y")
    thought, answer = split_reply(tok, generated, thinking=True)
    print(f"thinking={thought!r}\nanswer={answer!r}")
    assert thought == "Both pay; b is larger." and answer == "I choose: b). My reasoning: y"
    # A thought that happens to start with the word "thought" is not trimmed (that is Gemma-only).
    assert split_reply(tok, ids(tok, "thought about it</think>ok"), thinking=True) == ("thought about it", "ok")


def test_unverified_family_is_refused():
    tok = tokenizer("gpt2")
    with pytest.raises(ValueError, match="not been verified"):
        split_reply(tok, ids(tok, "hello"), thinking=True)
