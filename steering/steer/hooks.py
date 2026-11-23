"""Applying an edit (steer/edits.py) inside the model: during generation, and in a teacher-forced replay.

    with steering(model, edit, x=0.1, layer=38, positions="generated"):
        replies = local_chat_model.generate(texts, thinking=True, max_new_tokens=8192)

    logits = replay_logits(model, prompt_ids, answer_ids, edit, x=0.1, layer=38)   # next-token logits after answer_ids

Which positions are edited (`positions`):
  "all"        every real token: the prompt and everything generated
  "prompt"     the prompt only: the prefill pass, so the edited prompt states are what generation attends to
  "generated"  generated tokens only: each decode step's token (the prompt's states, and so the first generated
               token, are unedited)
  a callable   ctx -> [batch, tokens] bool, given the EditContext with roles already filled in;
               at_generated(index) builds one that edits a single token per row (e.g. the answer-label token);
               at_prompt(indices) builds one that edits chosen prompt tokens per row (e.g. the horizon's tokens)
Padding is never edited.

How roles are known. A forward pre-hook on the model reads, for each pass, the absolute index of each of its tokens
(left padding included). The pass whose first position is 0 is the prefill: its tokens are the prompt, and its length
is the prompt length; later passes (with the key-value cache) are generated tokens. The index comes from the first of
these that generate() passes:
  cache_position   the absolute indices themselves (passed by some transformers versions; 5.22.2 does not);
  attention_mask   [batch, all tokens so far], 0 at left padding: this pass is its last seq_len columns;
  position_ids     without an attention mask there is no padding, so position ids are the absolute indices
                   (transformers 5.22.2 drops an all-ones mask, e.g. at batch 1); with padding they are not, which
                   is why the mask comes first;
and, with none of them, a single pass from position 0. replay_logits runs one pass over prompt + answer and states the
prompt length itself.

The edit is computed in its own dtype (float32 unless set otherwise) and cast back to the model's dtype once:
h_new = where(mask, (h + delta).to(dtype), h). Unedited positions keep their original bits, and an edit whose
delta is exactly 0 (CAA at strength 0, `displace` at shift 0) reproduces the unhooked model exactly.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Union

import torch

from steer.edits import ROLE_GENERATED, ROLE_PAD, ROLE_PROMPT, EditContext
from steer.layers import check_layer, decoder_layers

POSITIONS = ("all", "prompt", "generated")


def _mask(positions: Union[str, Callable], ctx: EditContext) -> torch.Tensor:
    if callable(positions):
        mask = positions(ctx)
    elif positions == "all":
        mask = ctx.role != ROLE_PAD
    elif positions == "prompt":
        mask = ctx.role == ROLE_PROMPT
    elif positions == "generated":
        mask = ctx.role == ROLE_GENERATED
    else:
        raise ValueError(f"positions must be one of {POSITIONS} or a callable")
    return mask & (ctx.role != ROLE_PAD)


def at_generated(index) -> Callable[[EditContext], torch.Tensor]:
    """A `positions` callable that edits one token per row: generated token index[i] of row i (0 = the first
    generated token; -1 = the last prompt token, whose state predicts the first generated token).

    The token at generated index j sits at absolute position prompt_len + j, where prompt_len is the (padded) prompt
    length. It is read from the first pass of each run, recognised by its first position being 0: in generate() that
    pass is the prefill (no generated tokens, so prompt_len is its length); in a replay pass (steering(...,
    prompt_len=...)) it is the position of the first generated token. Index from a reply of the same prompt batch,
    e.g. steer.sites.locate_label(...).label_index for the label site, minus 1 for the prelabel site. The token at that
    index is only the label token if generation up to it is unchanged, which holds when nothing earlier is edited.
    """
    index = torch.as_tensor(index, dtype=torch.long)
    state: dict = {}

    def mask(ctx: EditContext) -> torch.Tensor:
        pos = ctx.position
        if int(pos[0, 0]) == 0:                                    # first pass of a run: (re)read prompt_len
            gen = ctx.role == ROLE_GENERATED
            state["prompt_len"] = int(pos[gen].min()) if gen.any() else int(pos.max()) + 1
        if "prompt_len" not in state:
            raise RuntimeError("at_generated: no pass starting at position 0 seen, so the prompt length is unknown")
        if index.shape != (pos.shape[0],):
            raise ValueError(f"at_generated: {index.shape[0]} indices for a batch of {pos.shape[0]}")
        return pos == (state["prompt_len"] + index.to(pos.device))[:, None]

    return mask


def at_prompt(indices) -> Callable[[EditContext], torch.Tensor]:
    """A `positions` callable that edits chosen prompt tokens: indices[i] lists row i's token indices in its own
    model input, counted without padding (e.g. the horizon's tokens, `token_indices` in horizon_sites.jsonl; an empty
    list edits nothing in that row).

    In a batch the prompts are left-padded, so row i's token k sits at absolute position pads_i + k; pads_i is read
    from the roles of the pass that starts at position 0 (the prefill, or a replay pass). Later passes hold generated
    tokens only and are not edited. The indices must come from the same model input (same chat template and thinking
    setting) that is being run.
    """
    indices = [sorted(int(k) for k in row) for row in indices]

    def mask(ctx: EditContext) -> torch.Tensor:
        out = torch.zeros_like(ctx.role, dtype=torch.bool)
        if int(ctx.position[0, 0]) != 0:
            return out
        if len(indices) != out.shape[0]:
            raise ValueError(f"at_prompt: {len(indices)} index lists for a batch of {out.shape[0]}")
        pads = (ctx.role == ROLE_PAD).sum(1).tolist()
        n_prompt = (ctx.role == ROLE_PROMPT).sum(1).tolist()
        for i, row in enumerate(indices):
            if row and not 0 <= row[0] <= row[-1] < n_prompt[i]:
                raise ValueError(f"at_prompt: row {i} has indices {row[0]}..{row[-1]} but only {n_prompt[i]} prompt tokens")
            out[i, [pads[i] + k for k in row]] = True
        return out

    return mask


class _PassState:
    """What the pre-hook learned about the current forward pass."""

    def __init__(self, prompt_len: Optional[int]):
        self.fixed_prompt_len = prompt_len
        self.prompt_len: Optional[int] = prompt_len
        self.cache_position: Optional[torch.Tensor] = None
        self.attention_mask: Optional[torch.Tensor] = None
        self.position_ids: Optional[torch.Tensor] = None

    def pre_hook(self, module, args, kwargs):
        self.cache_position = kwargs.get("cache_position")
        self.attention_mask = kwargs.get("attention_mask")
        self.position_ids = kwargs.get("position_ids")
        if self.fixed_prompt_len is None:
            first = self._positions(self._seq_len(args, kwargs), None)
            if first is not None and int(first[0]) == 0:          # a prefill: this pass is the whole prompt
                self.prompt_len = self._seq_len(args, kwargs)

    @staticmethod
    def _seq_len(args, kwargs) -> int:
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            ids = kwargs["inputs_embeds"]
        return ids.shape[1]

    def _positions(self, seq_len: int, device) -> Optional[torch.Tensor]:
        """Absolute index [seq_len] of each token of this pass."""
        if self.cache_position is not None:
            return self.cache_position.to(device) if device is not None else self.cache_position
        if self.attention_mask is not None and self.attention_mask.ndim == 2:
            total = self.attention_mask.shape[1]
            return torch.arange(total - seq_len, total, device=device)
        if self.position_ids is not None and self.position_ids.shape[-1] == seq_len:
            # [batch, seq_len], or [k, batch, seq_len] for multi-axis rotary positions; no padding, so every row and
            # axis of a text-only pass holds the same indices.
            pos = self.position_ids.reshape(-1, seq_len)[0]
            return pos.to(device) if device is not None else pos
        return torch.arange(seq_len, device=device)

    def context(self, h: torch.Tensor, rows) -> EditContext:
        batch, seq_len = h.shape[:2]
        pos = self._positions(seq_len, h.device)
        if self.prompt_len is None:
            raise RuntimeError("steering hook: no prefill pass seen, so the prompt length is unknown")
        role = torch.where(pos < self.prompt_len, ROLE_PROMPT, ROLE_GENERATED).expand(batch, seq_len).clone()
        if self.attention_mask is not None and self.attention_mask.ndim == 2:
            am = self.attention_mask.to(h.device)
            real = am.gather(1, pos.clamp(max=am.shape[1] - 1).expand(batch, seq_len)) != 0
            role[~real] = ROLE_PAD
        return EditContext(role=role, position=pos.expand(batch, seq_len), mask=torch.zeros_like(role, dtype=torch.bool),
                           rows=rows)


@contextmanager
def steering(model, edit, x, layer: int, positions: Union[str, Callable] = "all", rows: Optional[list[dict]] = None,
             prompt_len: Optional[int] = None, record: Optional[list] = None) -> Iterator[None]:
    """Apply `edit` with request `x` at `layer` (steer.layers convention) on every forward pass inside the block.

    rows: per-row facts passed to the edit in ctx.rows (in batch order). prompt_len: set it for a single
    teacher-forced pass (replay); during generate() it is read from the prefill pass. record: if a list is given,
    each pass appends its EditContext (with the mask filled in) and the L2 norm of the applied displacement per token
    [batch, tokens] (0 where unedited), for audits and for matching random controls' sizes token by token.
    """
    check_layer(model, layer)
    state = _PassState(prompt_len)

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        ctx = state.context(h, rows)
        ctx.mask = _mask(positions, ctx)
        if ctx.mask.any():
            delta = edit.delta(h, x, ctx)
            edited = (h.to(delta.dtype) + delta).to(h.dtype)
            new = torch.where(ctx.mask[..., None], edited, h)
        else:
            delta, new = None, h
        if record is not None:
            size = torch.zeros(h.shape[:2], device=h.device) if delta is None else \
                (delta.norm(dim=-1) * ctx.mask).float()
            record.append({"ctx": ctx, "delta_norm": size.cpu()})
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new

    handles = [model.register_forward_pre_hook(state.pre_hook, with_kwargs=True),
               decoder_layers(model)[layer - 1].register_forward_hook(hook)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def replay_logits(model, prompt_ids: list[int], answer_ids: list[int], edit, x, layer: int,
                  positions: Union[str, Callable] = "all", rows: Optional[list[dict]] = None) -> torch.Tensor:
    """Next-token logits (float32 [vocab]) after prompt_ids + answer_ids, in one teacher-forced pass (batch 1) with
    the edit applied. With answer_ids = the model's own reply up to the token before its answer label, these are
    the label probabilities at the answer slot, steered (the "replay readout"). positions "prompt" / "generated"
    split the sequence at len(prompt_ids)."""
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(answer_ids)], device=device)
    with steering(model, edit, x, layer, positions, rows, prompt_len=len(prompt_ids)):
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    return out.logits[0, -1].float()
