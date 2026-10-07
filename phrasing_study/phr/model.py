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


@torch.no_grad()
def forward_choice(tok, model, cap: Capture | None, seqs: list[list[int]], positions: list[list[int]],
                   label_ids: list[tuple[int, int]]):
    """One right-padded batch. seqs end with the prefill; the last token predicts the label.
    Returns (logit_a, logit_b, mass_ab, acts [B, n_layers, n_pos, d] or None). Logits in float32 from the final-normed
    last-layer state (two rows of lm_head); mass_ab = full-vocab softmax probability of the two label tokens."""
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
    last = torch.tensor([len(s) - 1 for s in seqs], device=hid.device)
    h = hid[torch.arange(B, device=hid.device), last]                                       # [B, d]
    W = model.lm_head.weight
    h = h.to(W.device)
    full = (h @ W.T).float()                                                                  # [B, V]
    ia = torch.tensor([a for a, _ in label_ids], device=W.device)
    ib = torch.tensor([b for _, b in label_ids], device=W.device)
    la = (h.float() * W[ia].float()).sum(-1)
    lb = (h.float() * W[ib].float()).sum(-1)
    p = torch.softmax(full, -1)
    mass = p.gather(1, ia[:, None]).squeeze(1) + p.gather(1, ib[:, None]).squeeze(1)
    acts = torch.stack([cap.out[l] for l in cap.layers], 1) if cap is not None else None
    return la.cpu(), lb.cpu(), mass.cpu(), acts


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
