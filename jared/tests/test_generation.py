"""Batched generation parity and force-close, on the real 0.8B (CPU, ~2 min)."""

import pytest
import torch
from transformers import AutoTokenizer

from spar_horizon.config import Settings
from spar_horizon.model_io import close_ids, generate_batch, load_model, think_end_id
from spar_horizon.prompts import build_prompts

STARTER_MODEL = "Qwen/Qwen3.5-0.8B"


def test_close_ids_contains_think_end():
    tok = AutoTokenizer.from_pretrained(STARTER_MODEL)
    ids = close_ids(tok)
    assert think_end_id(tok) in ids
    assert tok.decode(ids).strip() == "</think>"


@pytest.fixture(scope="module")
def loaded():
    return load_model(Settings(model=STARTER_MODEL, thinking="off"))


def test_batched_matches_single_greedy(loaded):
    tok, model, dev = loaded
    prompts = [r["prompt"] for r in build_prompts(limit=6)]
    single = generate_batch(tok, model, dev, prompts, Settings(model=STARTER_MODEL, thinking="off", batch_size=1))
    batched = generate_batch(tok, model, dev, prompts, Settings(model=STARTER_MODEL, thinking="off", batch_size=4))
    for s, b in zip(single, batched):
        assert s.prompt_len == b.prompt_len and s.answer_start == s.prompt_len + s.pre_len
        assert s.pre_len <= 3 and any(c.isalnum() for c in tok.decode([int(s.ids[s.answer_start])]))
        assert torch.equal(s.ids, b.ids), (tok.decode(s.ids[s.prompt_len:]), tok.decode(b.ids[b.prompt_len:]))
        assert "a)" in s.answer_text(tok) or "b)" in s.answer_text(tok)


def test_force_close_at_tiny_budget(loaded):
    tok, model, dev = loaded
    prompts = [r["prompt"] for r in build_prompts(limit=3)]
    cfg = Settings(model=STARTER_MODEL, thinking="on", max_think_tokens=24, batch_size=2)
    gens = generate_batch(tok, model, dev, prompts, cfg)
    tid = think_end_id(tok)
    pre = close_ids(tok) + tok.encode(cfg.prefill, add_special_tokens=False)
    for g in gens:
        assert g.forced and g.think_len >= 24
        assert g.pre_len == len(pre)
        assert g.ids[g.answer_start - g.pre_len:g.answer_start].tolist() == pre
        assert tid in g.ids[g.answer_start - g.pre_len:g.answer_start].tolist()
        assert g.ids.shape[0] >= g.answer_start + cfg.n_response
