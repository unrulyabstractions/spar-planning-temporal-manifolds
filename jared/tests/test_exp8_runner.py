"""exp8's Runner.score_many on a tiny random Qwen3 (no download): the shared-prefix
path for last-position cells must equal one full pass per edit."""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from exp8_steering import Runner  # noqa: E402
from spar_horizon import steering as S  # noqa: E402

SEQS = [[3, 5, 7, 9, 11, 13], [4, 6, 8, 10], [7, 7, 9, 12, 15]]
PAD = 0


def test_score_many_matches_full_passes_and_baseline():
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                         num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                                         head_dim=8)).eval()
    runner = Runner(model, "cpu", PAD, batch_size=2, long_batch_size=2)
    variant = {"ids": SEQS, "labels": [(1, 2)] * 3, "near_first": [True, False, True],
               "positions": {"pre-label": [[len(s) - 1] for s in SEQS], "early": [[1]] * 3}}
    prompts = {"variants": [variant], "batch_size": 2}
    delta = torch.randn(3, 32)
    makes = [lambda rows: S.Add(delta[rows]), lambda rows: S.Add(-delta[rows])]
    fast = runner.score_many(prompts, "pre-label", 2, makes)
    for make, row in zip(makes, fast):
        np.testing.assert_allclose(row, runner.score(prompts, "pre-label", 2, make), atol=1e-5)
    full_baseline = S.score(model, "cpu", SEQS, variant["near_first"], variant["labels"], PAD, 2)
    np.testing.assert_allclose(runner.score(prompts), full_baseline, atol=1e-5)
    # a cell that is not the last position falls back to full passes
    early = runner.score_many(prompts, "early", 2, makes)
    np.testing.assert_allclose(early[0], runner.score(prompts, "early", 2, makes[0]), atol=1e-6)
