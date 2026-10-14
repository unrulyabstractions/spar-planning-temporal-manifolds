"""generate_batch steps the batch size down on out-of-memory. Generation is faked;
only the real 0.8B tokenizer is used, so this runs without model weights."""

import pytest
import torch
from transformers import AutoTokenizer

from spar_horizon import model_io
from spar_horizon.config import Settings
from spar_horizon.prompts import build_prompts

STARTER_MODEL = "Qwen/Qwen3.5-0.8B"


@pytest.fixture
def tok():
    return AutoTokenizer.from_pretrained(STARTER_MODEL)


def fake_greedy(tok, max_rows, sizes):
    """A _greedy that records batch sizes, runs out of memory above `max_rows`,
    and otherwise appends ' a) fine.' to every prompt."""
    answer = tok.encode(" a) fine. And more text here.", add_special_tokens=False)

    def greedy(model, tokenizer, device, id_lists, max_new, min_new):
        sizes.append(len(id_lists))
        if len(id_lists) > max_rows:
            raise torch.OutOfMemoryError("fake OOM")
        return [torch.tensor(ids + answer) for ids in id_lists]
    return greedy


def test_oom_steps_down_and_finishes(tok, monkeypatch):
    sizes = []
    monkeypatch.setattr(model_io, "_greedy", fake_greedy(tok, 2, sizes))
    prompts = [r["prompt"] for r in build_prompts(limit=6)]
    cfg = Settings(model=STARTER_MODEL, thinking="off", batch_size=3)
    gens = model_io.generate_batch(tok, None, "cpu", prompts, cfg)
    assert sizes == [3, 2, 2, 2]
    assert all(g is not None and tok.decode([int(g.ids[g.answer_start])]) == " a" for g in gens)


def test_oom_at_batch_one_raises(tok, monkeypatch):
    monkeypatch.setattr(model_io, "_greedy", fake_greedy(tok, 0, []))
    cfg = Settings(model=STARTER_MODEL, thinking="off", batch_size=1)
    with pytest.raises(torch.OutOfMemoryError):
        model_io.generate_batch(tok, None, "cpu", [build_prompts()[0]["prompt"]], cfg)
