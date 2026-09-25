"""Activation capture: generate greedily, then re-run teacher-forced and hook the residual stream.

Residual-stream convention (stored axis `layer`, length n_layers + 1):
  layer 0        = output of the token embedding (input to decoder layer 0)
  layer l >= 1   = output of decoder layer l-1, i.e. the residual stream after l layers
                   (the prior work's `resid_post` of layer l-1)
No final norm is applied to any stored layer. Choice logits are computed by applying the model's
final norm and lm_head to the layer-n_layers vector at the position that predicts the label token.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .chat import ChatEncoding, encode_user_turn, find_choice, label_first_token_id, position_labels
from .prompts import PromptSample


@dataclass
class CaptureConfig:
    model_name: str = "Qwen/Qwen3-14B"
    enable_thinking: bool = False
    max_new_tokens: int = 48
    n_response: int = 8            # response positions R0..R{n-1} to store
    batch_size: int = 4
    dtype: str = "bfloat16"
    max_memory_gib: Optional[object] = None  # per-GPU cap in GiB: a number, or a list per GPU; None = accelerate decides
    split_layers: Optional[int] = None       # explicit 2-GPU map: embeddings + this many decoder layers on GPU 0, rest on GPU 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SampleResult:
    sample_uid: str
    gen_text: str
    gen_ids: list[int] = field(default_factory=list)
    n_generated: int = 0
    n_transition: int = 0
    transition_tokens: list[str] = field(default_factory=list)
    response_tokens: list[str] = field(default_factory=list)
    choice: Optional[str] = None            # 'a' | 'b' | None
    choice_gen_index: Optional[int] = None  # index into gen_ids
    logit_a: float = math.nan               # float32 two-row logits from the normed residual
    logit_b: float = math.nan
    p_a: float = math.nan                   # full-vocab softmax mass on the label token (model dtype)
    p_b: float = math.nan
    acts: Optional[torch.Tensor] = None     # [n_layers+1, n_pos, d] bf16, zero-filled for missing positions
    pos_valid: Optional[list[bool]] = None


class ResidualHooks:
    """Forward hooks that copy the residual stream at requested positions to CPU as it is produced."""

    def __init__(self, model):
        base = model.model
        self.modules = [base.embed_tokens] + list(base.layers)
        self.n_layers = len(base.layers)
        self.positions: Optional[torch.Tensor] = None   # [B, n_pos] long, on CPU
        self.out: Optional[torch.Tensor] = None         # [B, L+1, n_pos, d]
        self._handles = [m.register_forward_hook(self._make_hook(i)) for i, m in enumerate(self.modules)]

    def _make_hook(self, layer_index: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, (tuple, list)) else output   # [B, S, d]
            if self.positions is None:
                return
            B = h.shape[0]
            idx = self.positions.to(h.device)
            gathered = h[torch.arange(B, device=h.device)[:, None], idx]     # [B, n_pos, d]
            if self.out is None:
                self.out = torch.zeros((B, self.n_layers + 1, idx.shape[1], h.shape[-1]), dtype=h.dtype)
            self.out[:, layer_index] = gathered.to("cpu")
        return hook

    def arm(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.out = None

    def disarm(self) -> None:
        self.positions = None

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


def explicit_device_map(model_name: str, split_layers: int) -> dict:
    """Embeddings and decoder layers [0, split) on cuda:0; the rest, final norm and lm_head on cuda:1."""
    from transformers import AutoConfig
    n = AutoConfig.from_pretrained(model_name).num_hidden_layers
    if not (0 < split_layers < n):
        raise ValueError(f"split_layers must be in (0, {n}), got {split_layers}")
    dm = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1}
    for i in range(n):
        dm[f"model.layers.{i}"] = 0 if i < split_layers else 1
    return dm


def load_model(cfg: CaptureConfig):
    dtype = getattr(torch, cfg.dtype)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    kwargs = dict(dtype=dtype, device_map="auto")
    if cfg.split_layers is not None:
        kwargs["device_map"] = explicit_device_map(cfg.model_name, cfg.split_layers)
    elif cfg.max_memory_gib is not None:
        caps = cfg.max_memory_gib if isinstance(cfg.max_memory_gib, (list, tuple)) else [cfg.max_memory_gib] * torch.cuda.device_count()
        kwargs["max_memory"] = {i: f"{c}GiB" for i, c in enumerate(caps)}
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    model.eval()
    bad = sorted({str(p.device) for p in model.parameters() if p.device.type != "cuda"})
    if bad:
        raise RuntimeError(f"model did not fit on the GPUs; some parameters are on {bad}. "
                           f"Raise max_memory_gib or use a smaller model / quantization.")
    return model, tok


def _left_pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, L - len(s):] = torch.tensor(s)
        attn[b, L - len(s):] = 1
    return ids, attn


def _right_pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, : len(s)] = torch.tensor(s)
        attn[b, : len(s)] = 1
    return ids, attn


@torch.no_grad()
def capture_batch(model, tok, hooks: ResidualHooks, samples: list[PromptSample], cfg: CaptureConfig) -> list[SampleResult]:
    encs: list[ChatEncoding] = [encode_user_turn(tok, s.text, cfg.enable_thinking) for s in samples]
    n_trans = len(encs[0].transition_tokens)
    for e in encs:
        assert len(e.transition_tokens) == n_trans, "transition window length differs across prompts; template drift"
    pad_id = tok.pad_token_id
    eos_ids = sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
    device = model.get_input_embeddings().weight.device

    # 1) greedy generation, left-padded so all sequences end at the same index
    ids, attn = _left_pad([e.input_ids for e in encs], pad_id)
    gen = model.generate(
        input_ids=ids.to(device), attention_mask=attn.to(device),
        max_new_tokens=cfg.max_new_tokens, do_sample=False,
        pad_token_id=pad_id, eos_token_id=eos_ids,
    )
    gen_ids: list[list[int]] = []
    for b in range(len(encs)):
        g = gen[b, ids.shape[1]:].tolist()
        cut = len(g)
        for k, t in enumerate(g):
            if t in eos_ids or t == pad_id:
                cut = k
                break
        gen_ids.append(g[:cut])

    # 2) teacher-forced pass over prompt + generation, one unpadded sequence at a time.
    #    Unpadded single-sequence passes reproduce the library's own hidden states bit-exactly;
    #    padded batches differ by up to ~8% relative L2 at deep layers (bf16 kernel noise).
    n_pos = n_trans + cfg.n_response
    acts_list, pos_valid, last_raw_list = [], [], []
    for e, g in zip(encs, gen_ids):
        positions = torch.zeros((1, n_pos), dtype=torch.long)
        valid = []
        for k in range(n_pos):
            if k < n_trans:
                positions[0, k] = e.transition_start + k
                valid.append(True)
            elif (k - n_trans) < len(g):
                positions[0, k] = e.prompt_len + (k - n_trans)
                valid.append(True)
            else:
                positions[0, k] = 0   # gathered then zeroed
                valid.append(False)
        hooks.arm(positions)
        last_raw: dict = {}
        hh = model.model.layers[-1].register_forward_hook(
            lambda m, i, o: last_raw.setdefault("x", (o[0] if isinstance(o, (tuple, list)) else o)))
        try:
            model.model(input_ids=torch.tensor([e.input_ids + g], device=device))
        finally:
            hh.remove()
            hooks.disarm()
        a = hooks.out[0].clone()   # [L+1, n_pos, d]
        for k, v in enumerate(valid):
            if not v:
                a[:, k] = 0
        acts_list.append(a)
        pos_valid.append(valid)
        last_raw_list.append(last_raw["x"][0])   # [S, d] raw residual after the last layer (pre-norm)

    # 3) behavior readout: logits over the two labels at the position predicting the choice token
    results: list[SampleResult] = []
    for b, (s, e, g) in enumerate(zip(samples, encs, gen_ids)):
        r = SampleResult(
            sample_uid=s.sample_uid,
            gen_text=tok.decode(g),
            gen_ids=g,
            n_generated=len(g),
            n_transition=n_trans,
            transition_tokens=e.transition_tokens,
            response_tokens=[tok.decode([t]) for t in g[: cfg.n_response]],
            acts=acts_list[b],
            pos_valid=pos_valid[b],
        )
        j, choice = find_choice(tok, g, s.label_a, s.label_b)
        if j is not None:
            k = n_trans + (j - 1)          # stored position R{j-1} predicts the label token
            pos = e.prompt_len + j - 1
            if k < n_pos and pos_valid[b][k]:
                raw = acts_list[b][-1, k]
            else:                          # choice beyond the stored response window: pull it now
                raw = last_raw_list[b][pos].to("cpu")
            norm = model.model.norm
            h = norm(raw.to(norm.weight.device, dtype=norm.weight.dtype))
            W = model.lm_head.weight
            # the actual emitted token decides the spacing variant; its counterpart uses the same variant
            emitted = g[j]
            spaced = emitted == label_first_token_id(tok, s.label_a if choice == "a" else s.label_b)
            ia = tok((" " if spaced else "") + s.label_a, add_special_tokens=False)["input_ids"][0]
            ib = tok((" " if spaced else "") + s.label_b, add_special_tokens=False)["input_ids"][0]
            h32 = h.to(W.device).float()
            # two-row logits in float32 (exact given the bf16 residual); full-vocab softmax in model dtype
            logit_a = float(h32 @ W[ia].float())
            logit_b = float(h32 @ W[ib].float())
            probs = torch.softmax(model.lm_head(h.to(W.device)).float(), dim=-1)
            r.choice, r.choice_gen_index = choice, j
            r.logit_a, r.logit_b = logit_a, logit_b
            r.p_a, r.p_b = float(probs[ia]), float(probs[ib])
        results.append(r)
    return results


def run_capture(samples: list[PromptSample], cfg: CaptureConfig, sink, log=print) -> None:
    """Capture all samples in batches, handing each batch's results to `sink(results)`."""
    model, tok = load_model(cfg)
    hooks = ResidualHooks(model)
    log(f"loaded {cfg.model_name}: n_layers={hooks.n_layers} devices={sorted({str(p.device) for p in model.parameters()})}")
    sys.stdout.flush()
    t0 = time.time()
    try:
        for i in range(0, len(samples), cfg.batch_size):
            batch = samples[i : i + cfg.batch_size]
            tb = time.time()
            res = capture_batch(model, tok, hooks, batch, cfg)
            sink(res)
            done = i + len(batch)
            n_ok = sum(r.choice is not None for r in res)
            log(f"[{done}/{len(samples)}] batch {time.time()-tb:.1f}s total {time.time()-t0:.0f}s format_ok {n_ok}/{len(batch)}")
            sys.stdout.flush()
    finally:
        hooks.remove()
