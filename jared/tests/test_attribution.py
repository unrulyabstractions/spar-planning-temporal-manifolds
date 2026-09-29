"""Attribution patching on a tiny random Qwen3, no download."""

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from spar_horizon.attribution import COMPONENTS, aggregate, attribute_pair, logit_diff


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=64, head_dim=8)
    return Qwen3ForCausalLM(cfg).eval()


def test_identical_inputs_score_zero(tiny):
    ids = torch.tensor([[3, 5, 7, 9, 11]])
    s = attribute_pair(tiny, ids, ids.clone(), (1, 2))
    for c in COMPONENTS:
        assert s.noising[c].shape == (3, 5)
        assert np.allclose(s.noising[c], 0) and np.allclose(s.denoising[c], 0)
    assert s.clean_metric == s.corrupt_metric


def test_untouched_prefix_scores_zero(tiny):
    """Causal model: positions before the first differing token are identical
    in both runs, so their scores must be exactly zero."""
    clean = torch.tensor([[3, 5, 7, 9, 11]])
    corrupt = torch.tensor([[3, 5, 8, 9, 11]])
    s = attribute_pair(tiny, clean, corrupt, (1, 2))
    for c in COMPONENTS:
        assert np.allclose(s.noising[c][:, :2], 0), c
        assert not np.allclose(s.noising[c][:, 2:], 0), c


def test_length_mismatch_rejected(tiny):
    with pytest.raises(ValueError):
        attribute_pair(tiny, torch.tensor([[1, 2, 3]]), torch.tensor([[1, 2]]), (1, 2))


def test_params_stay_frozen_and_metric_matches(tiny):
    clean = torch.tensor([[3, 5, 7, 9, 11]])
    corrupt = torch.tensor([[3, 5, 8, 9, 11]])
    s = attribute_pair(tiny, clean, corrupt, (1, 2))
    assert all(not p.requires_grad for p in tiny.parameters())
    with torch.no_grad():
        expected = float(logit_diff(tiny(input_ids=clean).logits, (1, 2)))
    assert abs(s.clean_metric - expected) < 1e-5


def test_aggregate_aligns_from_end(tiny):
    a = attribute_pair(tiny, torch.tensor([[3, 5, 7, 9, 11]]), torch.tensor([[3, 5, 8, 9, 11]]), (1, 2))
    b = attribute_pair(tiny, torch.tensor([[2, 3, 5, 7, 9, 11]]), torch.tensor([[2, 3, 5, 8, 9, 11]]), (1, 2))
    agg, width = aggregate([a, b])
    assert width == 5
    assert agg["noising"]["resid_post"].shape == (3, 5)
