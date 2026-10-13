"""Model loading, residual capture, choice readout and batched generation for the single-turn protocols.

Residual convention (as in alan/ptm/capture.py and multiturn_planning): layer 0 = embedding output, layer l = output
of decoder layer l-1, no final norm. Choice forward passes are batched with RIGHT padding: under causal attention the
real tokens never see the pads, and position ids start at 0 for every row, so each row is computed as if alone
(up to kernel-level float differences; tests/test_model.py compares with unbatched passes).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PREFILL = "I choose:"
N_PRE = 9          # Qwen3 transition window: <|im_end|> \n <|im_start|> assistant \n <think> \n\n </think> \n\n


def load(model_name: str, dtype=torch.bfloat16, gpu_gib: float | None = None):
    """gpu_gib: cap GPU memory and offload the rest to CPU (local tests of models that don't fit)."""
    tok = AutoTokenizer.from_pretrained(model_name)
    placement = (dict(device_map="auto", max_memory={0: f"{gpu_gib}GiB", "cpu": "24GiB"}) if gpu_gib
                 else dict(device_map="cuda"))
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **placement)
    model.eval()
    return tok, model


def default_layers(n_layers: int) -> list[int]:
    """Residual layers kept: 0.2 .. 0.9 of depth in steps of 0.1, plus the last layer."""
    return sorted({round(f * n_layers) for f in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)} | {n_layers})


def encode_chat(tok, user_text: str) -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def transition_window(tok, ids: list[int]) -> list[int]:
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    last = max(i for i, t in enumerate(ids) if t == im_end)
    idx = list(range(last, len(ids)))
    if len(idx) != N_PRE:
        raise ValueError(f"transition window has {len(idx)} tokens, expected {N_PRE}: {[tok.decode([t]) for t in ids[last:]]}")
    return idx


def label_readout(tok, prefix: str, label_a: str, label_b: str) -> tuple[list[int], int, int]:
    """(prefill ids, token id for a, token id for b). `prefix + label` is tokenised for both labels (prefix = "I choose:"
    plus the model's own separator, e.g. " " or " **"); the shared leading tokens are prefilled (this also covers
    Qwen's ' ' before digits: ' 1)' -> ' ', '1', ')') and the first differing token is read."""
    ta = tok(prefix + label_a, add_special_tokens=False)["input_ids"]
    tb = tok(prefix + label_b, add_special_tokens=False)["input_ids"]
    c = 0
    while c < min(len(ta), len(tb)) and ta[c] == tb[c]:
        c += 1
    if c >= min(len(ta), len(tb)):
        raise ValueError(f"labels {label_a!r} / {label_b!r} have no distinguishing token")
    return ta[:c], ta[c], tb[c]


class Capture:
    """Hooks on the embedding and the decoder layers; keeps chosen layers at chosen positions per batch row."""

    def __init__(self, model, layers: list[int]):
        mods = [model.model.embed_tokens] + list(model.model.layers)
        self.layers, self.index, self.out = layers, None, {}
        self.handles = [mods[l].register_forward_hook(self._hook(l)) for l in layers]

    def _hook(self, l):
        def fn(_m, _inp, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            if self.index is not None:
                rows, pos = self.index
                self.out[l] = h[rows.to(h.device), pos.to(h.device)].to(torch.float16).cpu()
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


def readout_paths(tok, seps, label_a: str, label_b: str) -> list[tuple[str, list[int], int, int]]:
    """One readout path per separator the model may write after "I choose:" (" ", " **", ...): (sep, prefill ids,
    token id for a, token id for b). Paths with identical tokens are kept once (the first separator's name)."""
    out, seen = [], set()
    for s in seps:
        pre, ia, ib = label_readout(tok, PREFILL + s, label_a, label_b)
        if (tuple(pre), ia, ib) not in seen:
            seen.add((tuple(pre), ia, ib))
            out.append((s, pre, ia, ib))
    return out


def plan_readout(prefills: list[list[int]]) -> tuple[list[list[int]], list[int], int]:
    """Fewest sequences that hold every path's prefill as a prefix: P(' a' | 'I choose:') is read inside the
    'I choose: **' sequence, so letter labels need one forward pass; ' 1' vs ' **1' need two.
    Returns (sequences, index of the sequence holding each prefill, k = length of the prefix all prefills share).
    Tokens from k on are scored, so every path's probability is conditional on the same context."""
    k = 0
    while all(len(p) > k for p in prefills) and len({p[k] for p in prefills}) == 1:
        k += 1
    seqs: list[list[int]] = []
    for p in sorted(prefills, key=len, reverse=True):
        if not any(s[: len(p)] == p for s in seqs):
            seqs.append(p)
    where = [next(j for j, s in enumerate(seqs) if s[: len(p)] == p) for p in prefills]
    return seqs, where, k


def _logits_fp32(h: torch.Tensor, W: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """h @ W.T in float32 (bf16 logits are only good to ~0.1 at |logit| ~ 20), in vocabulary chunks."""
    h = h.float()
    return torch.cat([h @ W[i:i + chunk].float().T for i in range(0, W.shape[0], chunk)], -1)


@torch.no_grad()
def forward_logprobs(tok, model, cap: Capture | None, seqs: list[list[int]], positions: list[list[int]],
                     reads: list[list[tuple[int, int]]]):
    """One right-padded batch. reads[i] = [(pos, token id), ...] for row i. Returns (per row, a float32 tensor of
    log p(token | seq[: pos + 1]) = full-vocab log-softmax of the final-normed last-layer state at pos;
    acts [B, n_layers, n_pos, d] or None)."""
    B, T = len(seqs), max(map(len, seqs))
    pad = tok.pad_token_id
    ids = torch.full((B, T), pad, dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
    dev = model.get_input_embeddings().weight.device
    if cap is not None:
        cap.index = (torch.arange(B)[:, None].expand(B, len(positions[0])), torch.tensor(positions))
        cap.out = {}
    hid = model.model(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state   # final norm applied
    if cap is not None:
        cap.index = None
    rows = torch.tensor([i for i, r in enumerate(reads) for _ in r], device=hid.device)
    pos = torch.tensor([p for r in reads for p, _ in r], device=hid.device)
    W = model.lm_head.weight
    lp = torch.log_softmax(_logits_fp32(hid[rows, pos].to(W.device), W), -1)
    toks = torch.tensor([t for r in reads for _, t in r], device=W.device)
    vals = lp[torch.arange(len(toks), device=W.device), toks].cpu()
    acts = torch.stack([cap.out[l] for l in cap.layers], 1) if cap is not None else None
    return list(vals.split([len(r) for r in reads])), acts


@torch.no_grad()
def generate(tok, model, prompts: list[list[int]], max_new: int, greedy: bool = True, seed: int = 0,
             sampling: dict | None = None) -> list[list[int]]:
    """Left-padded batched generation; reply ids per prompt cut after the first <|im_end|> (kept)."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    side = tok.padding_side
    tok.padding_side = "left"
    batch = tok.pad({"input_ids": prompts}, return_tensors="pt").to(model.get_input_embeddings().weight.device)
    tok.padding_side = side
    kw = dict(do_sample=False) if greedy else dict(do_sample=True, **(sampling or {}))
    if not greedy:
        torch.manual_seed(seed)
    out = model.generate(**batch, max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                         eos_token_id=[im_end, tok.eos_token_id], **kw)
    replies = []
    for row in out[:, batch["input_ids"].shape[1]:].tolist():
        replies.append(row[: row.index(im_end) + 1] if im_end in row else [t for t in row if t != tok.pad_token_id])
    return replies
