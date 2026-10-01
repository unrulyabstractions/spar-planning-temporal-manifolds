"""Hidden-state extraction at fixed positions of every prompt's sequence.

Positions kept per prompt, in order (regions, each a (start, end) pair in
the kept layout):
- suffix: the template (+ prefill with thinking off) tokens after the user
  content, i.e. the tokens before `prompt_len`
- think (thinking on only): one position per `cfg.think_fractions` entry,
  at that fraction of the prompt's own think span
- pre (thinking on only): `</think>`, the following whitespace, and the
  prefill tokens
- prelabel: the single token right before the label, whatever the model put
  there (the prefill colon or a markdown lead); its state predicts the label
- answer: `cfg.n_response` tokens from the label token

The layout is the same width for every prompt, so no prompt is skipped for
its answer formatting; force-closed and natural think closes share it.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm

from .model_io import generate_batch, label_ids, prefill_ids, suffix_length


@dataclass
class Extraction:
    activations: list          # per layer: [n_kept, n_positions, d_model]
    tokens: list               # majority token per position, '*' if it varies
    answers: list              # decoded answer per prompt, None if skipped
    kept: np.ndarray           # bool per prompt
    n_suffix: int              # index of the answer token in the kept layout
    think_lens: list           # per prompt, kept or not
    forced: int = 0            # number of force-closed prompts
    regions: dict = field(default_factory=dict)   # name -> (start, end) in the kept layout
    think_fractions: tuple = ()
    forced_mask: np.ndarray = None   # per kept prompt
    think_len: np.ndarray = None     # per kept prompt
    lead: np.ndarray = None          # per kept prompt
    keep_idx: np.ndarray = None      # [n_kept, n_positions] absolute token index
    keep_ids: np.ndarray = None      # [n_kept, n_positions] token id
    label_logits: np.ndarray = None  # [n_kept, 2] logits of the a/b label ids at the prelabel token
    think_text: list = field(default_factory=list)   # per kept prompt
    generations: list = field(default_factory=list)  # Generation per kept prompt (not saved)

    @property
    def n_layers(self):
        return len(self.activations)


def think_positions(prompt_len, think_len, fractions):
    """Absolute token indices at `fractions` of the think span
    `[prompt_len, prompt_len + think_len)`; f = 1 is the last think token."""
    return [prompt_len + round(f * (think_len - 1)) for f in fractions]


def layout(n_suffix, n_think, n_pre, n_response):
    """Region boundaries for the kept layout, in order."""
    bounds, start = {}, 0
    for name, n in (("suffix", n_suffix), ("think", n_think), ("pre", n_pre),
                    ("prelabel", 1), ("answer", n_response)):
        bounds[name] = (start, start + n)
        start += n
    return bounds


def keep_positions(g, cfg, n_suffix, n_pre):
    """Absolute indices of the kept positions for one Generation."""
    think = think_positions(g.prompt_len, g.think_len, cfg.think_fractions) if cfg.thinking == "on" else []
    close = g.answer_start - g.lead - n_pre
    return np.r_[np.arange(g.prompt_len - n_suffix, g.prompt_len), np.array(think, int),
                 np.arange(close, close + n_pre), g.answer_start - 1,
                 np.arange(g.answer_start, g.answer_start + cfg.n_response)]


def extract(tokenizer, model, device, prompts, cfg, layers=None):
    """Generate every answer (batched), then one forward pass per prompt keeping
    hidden states at the positions described above. `layers` restricts which
    hidden_states indices are kept. Returns an Extraction."""
    n_suffix = suffix_length(tokenizer, prompts[0], cfg)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    labels = label_ids(tokenizer, cfg=cfg)
    gens = generate_batch(tokenizer, model, device, prompts, cfg)
    n_think = len(cfg.think_fractions) if cfg.thinking == "on" else 0
    n_pre = 2 + len(prefill_ids(tokenizer, cfg)) if cfg.thinking == "on" else 0
    regions = layout(n_suffix, n_think, n_pre, cfg.n_response)
    n_positions = regions["answer"][1]
    activations, label_counts = None, [Counter() for _ in range(n_positions)]
    turn_len, answers, kept = cfg.n_response, [], []
    forced_mask, think_len, lead, keep_idx, keep_ids, label_logits, think_text = ([] for _ in range(7))
    for g in tqdm(gens, desc="extract", unit="prompt", mininterval=2, dynamic_ncols=True):
        start = g.answer_start
        if g.ids.shape[0] < start + cfg.n_response:
            kept.append(False)
            answers.append(None)
            continue
        kept.append(True)
        ids = g.ids[: start + cfg.n_response].unsqueeze(0)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        response = g.ids[start:start + cfg.n_response].tolist()
        answers.append(g.answer_text(tokenizer))
        if end_id in response:
            turn_len = min(turn_len, response.index(end_id) + 1)
        keep = keep_positions(g, cfg, n_suffix, n_pre)
        for pos, tok_id in enumerate(g.ids[keep]):
            label_counts[pos][tokenizer.decode(tok_id)] += 1
        hs = out.hidden_states if layers is None else [out.hidden_states[l] for l in layers]
        per_layer = [h[0, keep].float().cpu().numpy() for h in hs]
        if activations is None:
            activations = [[] for _ in per_layer]
        for layer, vectors in zip(activations, per_layer):
            layer.append(vectors)
        forced_mask.append(g.forced)
        think_len.append(g.think_len)
        lead.append(g.lead)
        keep_idx.append(keep)
        keep_ids.append(g.ids[keep].cpu().numpy())
        label_logits.append(out.logits[0, start - 1, list(labels)].float().cpu().numpy())
        think_text.append(g.think_text(tokenizer) if cfg.thinking == "on" else "")
    skipped = len(prompts) - sum(kept)
    forced = sum(g.forced for g in gens)
    think_lens = [g.think_len for g in gens]
    answer_pos = regions["answer"][0]
    tokens = [_token_label(c) for c in label_counts]
    for k, f in enumerate(cfg.think_fractions if n_think else ()):
        tokens[regions["think"][0] + k] = f"think@{f:g}*"
    print(f"suffix tokens: {n_suffix}  think positions: {n_think}  pre tokens: {n_pre}  "
          f"answers: {dict(label_counts[answer_pos])}  (turn closes after {turn_len} tokens)  "
          f"skipped: {skipped}")
    if cfg.thinking == "on":
        print(f"think length: median {int(np.median(think_lens))}, max {max(think_lens)}  "
              f"force-closed at budget: {forced}/{len(gens)}")
    first = label_counts[answer_pos]
    if first and first.most_common(1)[0][0].strip() not in ("a", "b", "1", "2"):
        print(f"WARNING: the token at the answer position is usually {first.most_common(1)[0][0]!r}, "
              f"not a label; the answer-token geometry is not about the choice.")
    return Extraction([np.stack(a) for a in activations], tokens, answers, np.array(kept), answer_pos,
                      think_lens, forced, regions, tuple(cfg.think_fractions) if n_think else (),
                      np.array(forced_mask), np.array(think_len), np.array(lead), np.stack(keep_idx),
                      np.stack(keep_ids), np.stack(label_logits), think_text,
                      [g for g, k in zip(gens, kept) if k])


def dense_track(model, ext, layer, coef, intercept):
    """Probe reading at every think token of every kept prompt, from a second
    forward pass reading hidden_states[layer]. Returns [n_kept, max_think_len],
    nan past each prompt's own think length."""
    out = np.full((len(ext.generations), max(ext.think_len)), np.nan)
    for i, g in enumerate(tqdm(ext.generations, desc="dense track", unit="prompt", mininterval=2,
                               dynamic_ncols=True)):
        ids = g.ids[: g.prompt_len + g.think_len].unsqueeze(0)
        with torch.no_grad():
            h = model(input_ids=ids, output_hidden_states=True).hidden_states[layer][0, g.prompt_len:]
        out[i, : g.think_len] = h.float().cpu().numpy() @ coef + intercept
    return out


def _token_label(counter):
    """Majority token at a position, starred when prompts disagree."""
    tok, n = counter.most_common(1)[0]
    return tok if n == sum(counter.values()) else tok + "*"
