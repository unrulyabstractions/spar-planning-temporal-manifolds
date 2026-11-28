"""The horizon read sites of extract.py: which tokens, NaN rows without a horizon, and the mean (Qwen3-8B tokenizer)."""

import numpy as np
import pytest

import extract
from fixtures import BANK
from pbank.records import load_prompts
from steer.activations import HORIZON_POSITIONS, POSITIONS
from steer.sites import locate_horizon

MODEL_ID = "Qwen/Qwen3-8B"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except OSError:
        pytest.skip(f"{MODEL_ID} tokenizer not cached")


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def test_positions_include_the_horizon_sites():
    assert HORIZON_POSITIONS == ("horizon_last", "horizon_mean") and set(HORIZON_POSITIONS) <= set(POSITIONS)


def test_horizon_sites_are_the_phrase_tokens(tok):
    recs = [r for r in load_prompts(BANK) if r.horizon is not None]
    assert recs
    for r in recs:
        site = locate_horizon(tok, chat(tok, r.text), r.text, r.horizon)
        assert site.decoded.strip() == r.horizon
        # horizon_last is the unit token, the last of the phrase
        assert site.tokens[-1].strip() and site.token_indices[-1] == max(site.token_indices)
        assert not site.tokens[-1].strip().isdigit()


def test_horizon_vectors_last_and_mean():
    rows = np.random.default_rng(0).normal(size=(3, 5)).astype(np.float32)
    last, mean = extract.horizon_vectors(rows, 5)
    assert np.array_equal(last, rows[-1])
    np.testing.assert_allclose(mean, rows.mean(axis=0), rtol=1e-6)
    assert last.dtype == mean.dtype == np.float32


def test_single_token_phrase_mean_equals_last():
    rows = np.arange(4, dtype=np.float32)[None]
    last, mean = extract.horizon_vectors(rows, 4)
    assert np.array_equal(last, mean)


def test_no_horizon_gives_nan_rows():
    last, mean = extract.horizon_vectors(np.zeros((0, 5), np.float32), 5)
    assert last.shape == mean.shape == (5,) and np.isnan(last).all() and np.isnan(mean).all()
    assert last.dtype == mean.dtype == np.float32


def test_no_horizon_prompts_exist_in_the_fixture_bank():
    assert any(r.horizon is None for r in load_prompts(BANK))
