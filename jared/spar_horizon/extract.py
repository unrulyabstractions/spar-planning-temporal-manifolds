"""Hidden-state extraction at the turn suffix, the pre-answer region, and the
answer tokens.

Positions kept per prompt, in order:
- suffix: the template (+ prefill) tokens after the user content, i.e. the
  `n_suffix` tokens before `prompt_len`
- pre: any leading formatting tokens before the label (e.g. ' **'); with
  thinking on also `</think>`, the following newline, and the prefill
- answer: `cfg.n_response` tokens from the label token
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from .model_io import generate_batch, suffix_length


@dataclass
class Extraction:
    activations: list          # per layer: [n_kept, n_positions, d_model]
    tokens: list               # majority token per position, '*' if it varies
    answers: list              # decoded answer per prompt, None if skipped
    kept: np.ndarray           # bool per prompt
    n_suffix: int              # suffix + pre positions, i.e. index of the answer token
    think_lens: list
    forced: int = 0

    @property
    def n_layers(self):
        return len(self.activations)


def extract(tokenizer, model, device, prompts, cfg, layers=None, log_every=20):
    """Generate every answer (batched), then one forward pass per prompt keeping
    hidden states at the positions described above. `layers` restricts which
    hidden_states indices are kept."""
    n_suffix = suffix_length(tokenizer, prompts[0], cfg)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    gens = generate_batch(tokenizer, model, device, prompts, cfg,
                          log_every=max(1, log_every // cfg.batch_size))
    n_pre = Counter(g.pre_len for g in gens).most_common(1)[0][0]
    n_positions = n_suffix + n_pre + cfg.n_response
    activations, label_counts = None, [Counter() for _ in range(n_positions)]
    turn_len, answers, kept = cfg.n_response, [], []
    for i, g in enumerate(gens):
        start = g.answer_start
        if g.ids.shape[0] < start + cfg.n_response or g.pre_len != n_pre:
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
        keep = np.r_[np.arange(g.prompt_len - n_suffix, g.prompt_len),
                     np.arange(start - n_pre, start + cfg.n_response)]
        for pos, tok_id in enumerate(g.ids[keep]):
            label_counts[pos][tokenizer.decode(tok_id)] += 1
        hs = out.hidden_states if layers is None else [out.hidden_states[l] for l in layers]
        per_layer = [h[0, keep].float().cpu().numpy() for h in hs]
        if activations is None:
            activations = [[] for _ in per_layer]
        for layer, vectors in zip(activations, per_layer):
            layer.append(vectors)
        if (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(prompts)} extracted")
    skipped = len(prompts) - sum(kept)
    forced = sum(g.forced for g in gens)
    think_lens = [g.think_len for g in gens]
    answer_pos = n_suffix + n_pre
    print(f"suffix tokens: {n_suffix}  pre-answer tokens: {n_pre}  "
          f"answers: {dict(label_counts[answer_pos])}  (turn closes after {turn_len} tokens)  "
          f"skipped: {skipped}")
    if cfg.thinking == "on":
        print(f"think length: median {int(np.median(think_lens))}, max {max(think_lens)}  "
              f"force-closed at budget: {forced}/{len(gens)}")
    first = label_counts[answer_pos]
    if first and first.most_common(1)[0][0].strip() not in ("a", "b", "1", "2"):
        print(f"WARNING: the token at the answer position is usually {first.most_common(1)[0][0]!r}, "
              f"not a label; the answer-token geometry is not about the choice.")
    if skipped:
        print(f"note: {skipped} prompts skipped because their pre-answer length differed from "
              f"the majority ({n_pre}), e.g. markdown before the label on some prompts only.")
    if activations is None:
        raise RuntimeError("every prompt was skipped")
    n_keep = answer_pos + turn_len
    tokens = [c.most_common(1)[0][0] + ("*" if len(c) > 1 else "")
              for c in label_counts[:n_keep]]
    return Extraction([np.stack(layer)[:, :n_keep] for layer in activations],
                      tokens, answers, np.array(kept), answer_pos, think_lens, forced)
