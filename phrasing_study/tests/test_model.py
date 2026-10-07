import os

import pytest

from phr.items import build_choice, load_config
from phr.model import PREFILL, encode_chat, label_readout, transition_window


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    except OSError:
        pytest.skip("Qwen3 tokenizer not available offline")


@pytest.mark.parametrize("sep", [" ", " **"])
@pytest.mark.parametrize("labels", [("a)", "b)"), ("A)", "B)"), ("1)", "2)"), ("b)", "a)")])
def test_label_readout(tok, sep, labels):
    pre, ia, ib = label_readout(tok, PREFILL + sep, *labels)
    assert ia != ib
    assert tok.decode(pre + [ia]).endswith(labels[0][0]) and tok.decode(pre + [ib]).endswith(labels[1][0])
    assert tok.decode(pre).startswith(PREFILL)


def test_transition_window_every_item(tok):
    df = build_choice(load_config())
    for t in df.drop_duplicates("variant").text:
        ids = encode_chat(tok, t)
        w = transition_window(tok, ids)
        assert tok.decode([ids[w[0]]]) == "<|im_end|>" and w[-1] == len(ids) - 1


@pytest.mark.skipif(os.environ.get("PHR_MODEL_TEST") != "1", reason="set PHR_MODEL_TEST=1 (loads Qwen3-1.7B in fp32 on CPU)")
def test_batched_equals_unbatched():
    """Right-padded batches give the same readout and activations as one prompt at a time. Checked in fp32 on CPU,
    where it is exact (2026-10-06: max |diff| 1e-5); in bf16 on GPU, kernel rounding alone moves log-odds by up to ~0.35."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from phr.model import Capture, forward_choice
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32).eval()
    df = build_choice(load_config()).iloc[[0, 500, 3000, 6000, 8000, 8900]]      # different lengths
    pre, ia, ib = label_readout(tok, PREFILL + " **", "a)", "b)")
    seqs = [encode_chat(tok, t) + pre for t in df.text]
    pos = [transition_window(tok, s[: -len(pre)]) + [len(s) - 1] for s in seqs]
    cap = Capture(model, [0, 14, 28])
    la, lb, _, acts = forward_choice(tok, model, cap, seqs, pos, [(ia, ib)] * len(seqs))
    for i in range(len(seqs)):
        la1, lb1, _, a1 = forward_choice(tok, model, cap, seqs[i:i + 1], pos[i:i + 1], [(ia, ib)])
        assert abs(float(la[i] - lb[i]) - float(la1[0] - lb1[0])) < 1e-3
        assert torch.allclose(acts[i].float(), a1[0].float(), rtol=1e-2, atol=1e-2)
