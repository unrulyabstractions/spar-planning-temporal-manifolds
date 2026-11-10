"""Uses the real Qwen3.5-0.8B tokenizer (cached locally). No model weights."""

import pytest
import torch
from transformers import AutoTokenizer

from spar_horizon.config import Settings
from spar_horizon.extract import label_pair
from spar_horizon.model_io import (Generation, cache_key, cache_load, cache_store, chat_text, label_ids, prefill_ids,
                                   suffix_length, think_end_id)
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


FULL = dict(model=STARTER_MODEL, thinking="off", prefill="", choice_prefix="I choose:")


def test_label_ids_follow_choice_prefix(tok):
    a, b = label_ids(tok, cfg=Settings(**FULL))
    assert tok.decode([a]) == " a" and tok.decode([b]) == " b"


def test_locate_label_after_choice_prefix(tok):
    from spar_horizon.model_io import locate_label
    cfg = Settings(**FULL)
    prompt = tok.encode("PROMPT", add_special_tokens=False)
    # after a markdown lead the label token has no leading space, as in the prefill path;
    # digits are their own tokens, so the space before "2" is a lead token too
    for answer, expect in [("I choose: a). My reasoning: fine.", " a"),
                           ("**I choose: b)**. My reasoning: fine.", " b"),
                           ("I choose: **a)**. My reasoning:", "a"),
                           ("Sure.\nI choose: 2). My reasoning:", "2")]:
        ids = prompt + tok.encode(answer, add_special_tokens=False)
        i = locate_label(tok, ids, len(prompt), cfg)
        assert tok.decode([ids[i]]) == expect, (answer, tok.decode([ids[i]]))
    ids = prompt + tok.encode("**b)** is better.", add_special_tokens=False)
    assert tok.decode([ids[locate_label(tok, ids, len(prompt), cfg)]]) == "b"


def test_suffix_off_full_bank_has_no_prefill(tok):
    from spar_horizon.full_bank import build_full_prompts
    cfg = Settings(**FULL)
    prompt = build_full_prompts(n=4)[0]["prompt"]
    assert suffix_length(tok, prompt, cfg) == 9
    assert chat_text(tok, prompt, cfg).endswith("<think>\n\n</think>\n\n")


def make_generation(token):
    return Generation(torch.tensor([1, 2, token]), 1, 2, 0, False, 0, 0)


def test_cache_store_keeps_the_first_entry(tmp_path):
    cfg = Settings(cache_dir=str(tmp_path))
    cache_store(cfg, [1, 2], None, make_generation(7))
    cache_store(cfg, [1, 2], None, make_generation(8))
    assert cache_load(cfg, [1, 2], None, "cpu").ids.tolist() == [1, 2, 7]
    assert [p.name for p in tmp_path.iterdir()] == [f"{cache_key(cfg, [1, 2], None)}.npz"]   # no temp files left


def test_label_pair_follows_the_written_label_form(tok):
    spaced, unspaced = label_ids(tok, cfg=Settings(model=STARTER_MODEL)), label_ids(tok)
    assert label_pair(tok, tok.encode(" b", add_special_tokens=False)[0], spaced, unspaced) == spaced
    assert label_pair(tok, tok.encode("b", add_special_tokens=False)[0], spaced, unspaced) == unspaced
