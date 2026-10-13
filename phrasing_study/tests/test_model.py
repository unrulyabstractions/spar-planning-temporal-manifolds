import os

import pandas as pd
import pytest

from phr.items import build_choice, load_config
from phr.model import PREFILL, encode_chat, label_readout, plan_readout, readout_paths, transition_window


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


@pytest.mark.parametrize("labels", [("a)", "b)"), ("A)", "B)"), ("1)", "2)"), ("b)", "a)")])
def test_readout_paths_share_one_context(tok, labels):
    """Both separators are scored from the same point after 'I choose:'; letters need one sequence, digits two."""
    paths = readout_paths(tok, [" ", " **"], *labels)
    assert len(paths) == 2
    seqs, where, k = plan_readout([p[1] for p in paths])
    assert tok.decode(paths[0][1][:k]) == PREFILL
    assert len(seqs) == (2 if labels[0][0].isdigit() else 1)
    for (sep, pre, ia, ib), j in zip(paths, where):
        assert seqs[j][: len(pre)] == pre
        assert tok.decode(pre + [ia]) == PREFILL + sep + labels[0][0]


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
    from phr.model import Capture, forward_logprobs
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32).eval()
    df = build_choice(load_config()).iloc[[0, 500, 3000, 6000, 8000, 8900]]      # different lengths
    pre, ia, ib = label_readout(tok, PREFILL + " **", "a)", "b)")
    seqs = [encode_chat(tok, t) + pre for t in df.text]
    pos = [transition_window(tok, s[: -len(pre)]) + [len(s) - 1] for s in seqs]
    reads = [[(len(s) - 2, pre[-1]), (len(s) - 1, ia), (len(s) - 1, ib)] for s in seqs]
    cap = Capture(model, [0, 14, 28])
    vals, acts = forward_logprobs(tok, model, cap, seqs, pos, reads)
    for i in range(len(seqs)):
        v1, a1 = forward_logprobs(tok, model, cap, seqs[i:i + 1], pos[i:i + 1], reads[i:i + 1])
        assert torch.allclose(vals[i], v1[0], atol=1e-3)
        assert torch.allclose(acts[i].float(), a1[0].float(), rtol=1e-2, atol=1e-2)
    assert all((v <= 0).all() for v in vals)


@pytest.mark.skipif(os.environ.get("PHR_MODEL_TEST") != "1", reason="set PHR_MODEL_TEST=1 (loads Qwen3-1.7B in fp32 on CPU)")
def test_summed_readout_matches_direct_probabilities():
    """The summed readout equals probabilities computed directly, one separator path at a time, and the ' ' path
    alone reproduces the old single-separator log-odds."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from phr.single import _combine, _prep_choice
    from phr.model import forward_logprobs
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32).eval()
    df = build_choice(load_config())
    df = pd.concat([df[df.label_short == l].head(1) for l in ("a)", "1)")])
    rows = _prep_choice(tok, df, [" ", " **"])
    assert len(rows) == 3                                       # letters: 1 sequence, digits: 2
    lp = [dict() for _ in range(len(df))]
    for r in rows:
        vals, _ = forward_logprobs(tok, model, None, [r["seq"]], [r["pos"]], [r["reads"]])
        for (s, what), v in zip(r["keys"], vals[0].tolist()):
            d = lp[r["n"]].setdefault(s, {"sep": 0.0})
            d["sep"] = d["sep"] + v if what == "sep" else d["sep"]
            if what != "sep":
                d[what] = v

    def direct(text, cont):                                     # log P(cont | prompt + "I choose:"), token by token
        ids = encode_chat(tok, text) + tok(PREFILL, add_special_tokens=False)["input_ids"]
        c = tok(PREFILL + cont, add_special_tokens=False)["input_ids"][len(ids) - len(encode_chat(tok, text)):]
        with torch.no_grad():
            logp = torch.log_softmax(model(torch.tensor([ids + c])).logits[0].float(), -1)
        return sum(float(logp[len(ids) - 1 + t, c[t]]) for t in range(len(c)))

    for n, r in enumerate(df.itertuples()):
        got = _combine(lp[n], {" ": "space", " **": "bold"})
        pa = sum(np.exp(direct(r.text, s + r.label_short[0])) for s in (" ", " **"))
        pb = sum(np.exp(direct(r.text, s + r.label_long[0])) for s in (" ", " **"))
        assert abs(got["mass_ab"] - (pa + pb)) < 1e-4
        assert abs(got["logit_diff"] - np.log(pa / pb)) < 1e-3
