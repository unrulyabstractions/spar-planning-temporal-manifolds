"""Batched generation parity and force-close, on the real 0.8B (CPU, ~2 min)."""

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from spar_horizon.config import Settings
from spar_horizon.extract import extract, think_positions
from spar_horizon.model_io import close_ids, generate_batch, load_model, suffix_length, think_end_id
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


def test_think_positions_forced_close(loaded):
    tok, model, dev = loaded
    prompts = [r["prompt"] for r in build_prompts(limit=2)]
    cfg = Settings(model=STARTER_MODEL, thinking="on", max_think_tokens=32, batch_size=2)
    for g in generate_batch(tok, model, dev, prompts, cfg):
        pos = think_positions(g.prompt_len, g.think_len, cfg.think_fractions)
        assert all(b > a for a, b in zip(pos, pos[1:]))
        assert all(g.prompt_len <= p < g.answer_start - g.pre_len for p in pos)
        assert pos[-1] == g.prompt_len + g.think_len - 1
        assert think_positions(g.prompt_len, g.think_len, (0.0, 1.0))[0] == g.prompt_len


def test_extract_thinking_off_layout(loaded):
    tok, model, dev = loaded
    prompts = [r["prompt"] for r in build_prompts(limit=3)]
    cfg = Settings(model=STARTER_MODEL, thinking="off", batch_size=3)
    ext = extract(tok, model, dev, prompts, cfg, layers=[0, 1])
    n_suffix = suffix_length(tok, prompts[0], cfg)
    assert ext.regions["think"] == (n_suffix, n_suffix) and ext.regions["pre"] == (n_suffix, n_suffix)
    assert ext.regions["prelabel"] == (n_suffix, n_suffix + 1)
    assert ext.regions["answer"] == (n_suffix + 1, n_suffix + 1 + cfg.n_response)
    assert ext.n_suffix == n_suffix + 1 and len(ext.tokens) == n_suffix + 1 + cfg.n_response
    assert not any(t.startswith("think@") for t in ext.tokens) and ext.think_fractions == ()
    assert ext.kept.all() and ext.activations[0].shape == (3, len(ext.tokens), model.config.hidden_size)
    assert np.all(np.diff(ext.keep_idx, axis=1) >= 0) and ext.label_logits.shape == (3, 2)
    assert ext.keep_idx[:, n_suffix].tolist() == [g.answer_start - 1 for g in ext.generations]


def test_extract_thinking_on_layout(loaded):
    tok, model, dev = loaded
    prompts = [r["prompt"] for r in build_prompts(limit=3)]
    cfg = Settings(model=STARTER_MODEL, thinking="on", max_think_tokens=32, batch_size=3)
    ext = extract(tok, model, dev, prompts, cfg, layers=[0])
    n_suffix = suffix_length(tok, prompts[0], cfg)
    n_pre = 2 + len(tok.encode(cfg.prefill, add_special_tokens=False))
    assert ext.regions["think"] == (n_suffix, n_suffix + 4)
    assert ext.tokens[n_suffix:n_suffix + 4] == ["think@0.25*", "think@0.5*", "think@0.75*", "think@1*"]
    assert ext.regions["pre"] == (n_suffix + 4, n_suffix + 4 + n_pre)
    assert ext.tokens[n_suffix + 4] == "</think>"
    assert ext.forced_mask.all() and ext.kept.all() and (ext.think_len >= 32).all()
    assert ext.n_suffix == ext.regions["answer"][0]
    assert all(t.strip() == "" for t in ext.think_text) is False


def test_generation_cache_roundtrip(loaded, tmp_path):
    tok, model, dev = loaded
    prompts = [r["prompt"] for r in build_prompts(limit=2)]
    cfg = Settings(model=STARTER_MODEL, thinking="off", batch_size=2, cache_dir=str(tmp_path))
    first = generate_batch(tok, model, dev, prompts, cfg)
    assert len(list(tmp_path.glob("*.npz"))) == 2
    second = generate_batch(tok, model, dev, prompts, cfg)
    for a, b in zip(first, second):
        assert torch.equal(a.ids, b.ids) and (a.answer_start, a.lead, a.forced) == (b.answer_start, b.lead, b.forced)
