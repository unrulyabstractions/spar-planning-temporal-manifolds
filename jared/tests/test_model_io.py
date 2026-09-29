"""Uses the real Qwen3.5-0.8B tokenizer (cached locally). No model weights."""

import pytest
from transformers import AutoTokenizer

from spar_horizon.config import Settings
from spar_horizon.model_io import chat_text, label_ids, prefill_ids, suffix_length, think_end_id
from spar_horizon.prompts import build_prompts

STARTER_MODEL = "Qwen/Qwen3.5-0.8B"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(STARTER_MODEL)


def test_think_end_token_exists(tok):
    assert think_end_id(tok) is not None


def test_suffix_off_matches_starter_constant(tok):
    cfg = Settings(model=STARTER_MODEL, thinking="off", prefill="")
    prompt = build_prompts()[0]["prompt"]
    assert suffix_length(tok, prompt, cfg) == 9
    assert chat_text(tok, prompt, cfg).endswith("<think>\n\n</think>\n\n")


def test_suffix_off_with_prefill(tok):
    cfg = Settings(model=STARTER_MODEL, thinking="off")
    prompt = build_prompts()[0]["prompt"]
    n_pre = len(prefill_ids(tok, cfg))
    assert n_pre >= 2
    assert suffix_length(tok, prompt, cfg) == 9 + n_pre
    assert chat_text(tok, prompt, cfg).endswith("</think>\n\nI choose:")


def test_label_ids_follow_prefill(tok):
    a, b = label_ids(tok, cfg=Settings(model=STARTER_MODEL))
    assert tok.decode([a]) == " a" and tok.decode([b]) == " b"
    a2, _ = label_ids(tok, cfg=Settings(model=STARTER_MODEL, prefill=""))
    assert tok.decode([a2]) == "a"


def test_suffix_on_is_open_think_block(tok):
    cfg = Settings(model=STARTER_MODEL, thinking="on")
    prompt = build_prompts()[0]["prompt"]
    assert suffix_length(tok, prompt, cfg) == 7
    assert chat_text(tok, prompt, cfg).endswith("<think>\n")


def test_suffix_is_prompt_independent(tok):
    cfg = Settings(model=STARTER_MODEL, thinking="off", prefill="")
    lengths = {suffix_length(tok, r["prompt"], cfg) for r in build_prompts(limit=10)}
    assert lengths == {9}


def test_system_prompt_goes_before_user(tok):
    cfg = Settings(model=STARTER_MODEL, thinking="off")
    text = chat_text(tok, "USER TEXT", cfg, system="SYSTEM TEXT")
    assert text.index("SYSTEM TEXT") < text.index("USER TEXT")


def test_settings_rejects_bad_thinking():
    with pytest.raises(ValueError):
        Settings(thinking="maybe")
