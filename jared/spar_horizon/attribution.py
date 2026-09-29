"""Standard attribution patching: score = (clean - corrupt) . grad.

Approximates the effect of patching one component's activation at one
position from the corrupt run into the clean run, for every (layer, position,
component) in one forward and one backward pass. Follows the prior repo's
design (`temporal-awareness/src/attribution_patching/standard_attribution.py`):

- metric: logit(label_a) - logit(label_b) at the position that predicts the
  answer token (the last prompt token, thinking off)
- components: resid_post (decoder layer output), attn_out (self-attention or
  linear-attention module output), mlp_out
- noising: grads from the clean run, score = (corrupt - clean) . grad_clean,
  i.e. how much the metric moves when corrupt content is inserted
- denoising: grads from the corrupt run, score = (clean - corrupt) . grad_corrupt

Params are frozen; only activation grads are kept. Clean and corrupt must
tokenize to the same length, so positions line up.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

COMPONENTS = ("resid_post", "attn_out", "mlp_out")
_ATTN_NAMES = ("self_attn", "linear_attn")


def _submodule(layer, names):
    for n in names:
        if hasattr(layer, n):
            return getattr(layer, n)
    raise AttributeError(f"no attention module among {names} in {type(layer).__name__}")


@contextmanager
def capture(model, retain_grad):
    """Register forward hooks on every decoder layer's output, attention, and
    MLP. Yields a dict component -> list of tensors (one per layer)."""
    store = {c: [None] * len(model.model.layers) for c in COMPONENTS}
    handles = []

    def hook(component, i):
        def fn(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if retain_grad:
                t.retain_grad()
            store[component][i] = t
        return fn

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(hook("resid_post", i)))
        handles.append(_submodule(layer, _ATTN_NAMES).register_forward_hook(hook("attn_out", i)))
        handles.append(layer.mlp.register_forward_hook(hook("mlp_out", i)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def logit_diff(logits, ids, position=-1):
    return logits[0, position, ids[0]] - logits[0, position, ids[1]]


@dataclass
class PairScores:
    noising: dict      # component -> [n_layers, seq]
    denoising: dict
    clean_metric: float
    corrupt_metric: float
    seq_len: int


def _run(model, input_ids, ids, with_grad):
    """One pass. Returns (metric value, detached activations, grads or None)."""
    torch.set_grad_enabled(with_grad)
    try:
        # Params are frozen, so the graph has to start at the embeddings.
        embeds = model.get_input_embeddings()(input_ids).detach()
        if with_grad:
            embeds.requires_grad_(True)
        with capture(model, retain_grad=with_grad) as store:
            out = model(inputs_embeds=embeds)
            metric = logit_diff(out.logits.float(), ids)
            if with_grad:
                metric.backward()
        acts = {c: [t.detach().float() for t in ts] for c, ts in store.items()}
        grads = ({c: [t.grad.float() for t in ts] for c, ts in store.items()}
                 if with_grad else None)
    finally:
        torch.set_grad_enabled(False)
    return float(metric), acts, grads


def attribute_pair(model, clean_ids, corrupt_ids, label_token_ids):
    """Scores for one clean/corrupt pair. Inputs are [1, seq] tensors of equal
    length. Returns PairScores with arrays [n_layers, seq] per component."""
    if clean_ids.shape != corrupt_ids.shape:
        raise ValueError(f"length mismatch: {clean_ids.shape} vs {corrupt_ids.shape}")
    for p in model.parameters():
        p.requires_grad_(False)
    m_clean, a_clean, g_clean = _run(model, clean_ids, label_token_ids, with_grad=True)
    m_corr, a_corr, g_corr = _run(model, corrupt_ids, label_token_ids, with_grad=True)

    def scores(a_from, a_to, grads):
        out = {}
        for c in COMPONENTS:
            rows = [((a_to[c][i] - a_from[c][i]) * grads[c][i]).sum(-1)[0].cpu().numpy()
                    for i in range(len(grads[c]))]
            out[c] = np.stack(rows)
        return out

    return PairScores(noising=scores(a_clean, a_corr, g_clean),
                      denoising=scores(a_corr, a_clean, g_corr),
                      clean_metric=m_clean, corrupt_metric=m_corr,
                      seq_len=int(clean_ids.shape[1]))


def aggregate(pair_scores):
    """Mean over pairs, per direction and component. Pairs may differ in
    length; positions are aligned from the END (the suffix and answer region
    is what every prompt shares), so the output width is the shortest pair."""
    width = min(p.seq_len for p in pair_scores)
    agg = {"noising": {}, "denoising": {}}
    for direction in agg:
        for c in COMPONENTS:
            agg[direction][c] = np.mean(
                [getattr(p, direction)[c][:, -width:] for p in pair_scores], axis=0)
    return agg, width
