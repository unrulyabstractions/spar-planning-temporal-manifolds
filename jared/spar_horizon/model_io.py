"""Model loading, chat templating, positions, and generation.

Positions are derived from the tokenizer, not hardcoded: the turn suffix is
every template token after the user content ends.

Assistant prefill (`cfg.prefill`, default "I choose:") pins the answer token:
the token generated right after the prefill is the label. With thinking off
the prefill is part of the prompt text, so the turn suffix runs from
`<|im_end|>` through the prefill. With thinking on the model reasons first;
the think block is force-closed at `cfg.max_think_tokens` if needed, then
`</think>\\n\\n` + prefill is appended and the answer is generated. Rows whose
block was force-closed are marked `forced`.

Generation is batched (left-padded, greedy) and verified token-identical to
single-prompt generation.
"""

from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CLOSE_TEXT = "\n</think>\n\n"


def load_model(cfg):
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=cfg.dtype)
    model.to(device).eval()
    return tokenizer, model, device


def think_end_id(tokenizer):
    tid = tokenizer.convert_tokens_to_ids("</think>")
    return None if tid is None or tid == tokenizer.unk_token_id else tid


def close_ids(tokenizer):
    ids = tokenizer.encode(CLOSE_TEXT, add_special_tokens=False)
    assert think_end_id(tokenizer) in ids, "close text does not tokenize to </think>"
    return ids


def prefill_ids(tokenizer, cfg):
    return tokenizer.encode(cfg.prefill, add_special_tokens=False) if cfg.prefill else []


def template_kwargs(tokenizer, cfg):
    if cfg.thinking == "on" and think_end_id(tokenizer) is None:
        raise ValueError(f"{cfg.model} has no native </think> token; thinking='on' needs one")
    return {"enable_thinking": cfg.thinking == "on"}


def chat_text(tokenizer, prompt, cfg, system=None):
    """Chat-templated prompt. With thinking off the prefill is appended here."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs(tokenizer, cfg))
    if cfg.thinking == "off" and cfg.prefill:
        text += cfg.prefill
    return text


def suffix_length(tokenizer, prompt, cfg):
    """Number of template (+ prefill) tokens after the user content."""
    text = chat_text(tokenizer, prompt, cfg)
    end_char = text.index(prompt) + len(prompt)
    enc = tokenizer(text, return_offsets_mapping=True)
    return sum(1 for s, _ in enc["offset_mapping"] if s >= end_char)


def label_ids(tokenizer, labels=("a", "b"), cfg=None):
    """Token ids of the label tokens as the model will emit them: with a
    leading space after a prefill ending in ':' and no space otherwise."""
    space = " " if (cfg is not None and cfg.prefill) else ""
    ids = []
    for lab in labels:
        toks = tokenizer.encode(space + lab, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"label {space + lab!r} is {len(toks)} tokens, need 1")
        ids.append(toks[0])
    return tuple(ids)


@dataclass
class Generation:
    ids: torch.Tensor          # [T] unpadded full sequence
    prompt_len: int
    answer_start: int          # index of the label token
    think_len: int
    forced: bool
    pre_len: int = 0           # tokens between prompt_len (off) or the think close (on) and the label:
                               # '</think>\n\n' + prefill (on only) + any leading formatting like ' **'
    regions: dict = field(default_factory=dict)

    def answer_text(self, tokenizer):
        return tokenizer.decode(self.ids[self.answer_start:])


def _greedy(model, tokenizer, device, id_lists, max_new, min_new):
    """Left-padded greedy generation. Returns unpadded [prompt+generated] per row."""
    tokenizer.padding_side = "left"
    batch = tokenizer.pad({"input_ids": id_lists}, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=max_new, min_new_tokens=min_new,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id)
    width = batch["input_ids"].shape[1]
    return [torch.cat([torch.tensor(ids, device=device), out[i, width:]])
            for i, ids in enumerate(id_lists)]


MAX_LEAD = 3


def lead_length(tokenizer, ids, start, max_lead=MAX_LEAD):
    """Number of leading tokens from `start` that carry no letter or digit,
    e.g. a markdown ' **' before the label. Capped at `max_lead`."""
    n = 0
    for tok in ids[start:start + max_lead]:
        if any(ch.isalnum() for ch in tokenizer.decode([int(tok)])):
            break
        n += 1
    return n


def _is_whitespace_token(tokenizer, tok_id):
    return tokenizer.decode([int(tok_id)]).strip() == ""


def generate_batch(tokenizer, model, device, prompts, cfg, system=None, log_every=None):
    """Greedy answers for a list of prompts, in batches of cfg.batch_size."""
    texts = [chat_text(tokenizer, p, cfg, system) for p in prompts]
    id_lists = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    results = [None] * len(prompts)
    if cfg.thinking == "on":
        tid, closer, pre = think_end_id(tokenizer), close_ids(tokenizer), prefill_ids(tokenizer, cfg)
    for b in range(0, len(prompts), cfg.batch_size):
        idx = list(range(b, min(b + cfg.batch_size, len(prompts))))
        if cfg.thinking == "off":
            rows = _greedy(model, tokenizer, device, [id_lists[i] for i in idx],
                           cfg.n_response + MAX_LEAD, cfg.n_response + MAX_LEAD)
            for i, row in zip(idx, rows):
                plen = len(id_lists[i])
                lead = lead_length(tokenizer, row, plen)
                results[i] = Generation(row, plen, plen + lead, 0, False, lead)
        else:
            rows = _greedy(model, tokenizer, device, [id_lists[i] for i in idx],
                           cfg.max_think_tokens, 0)
            second, meta = [], []
            for i, row in zip(idx, rows):
                plen = len(id_lists[i])
                resp = row[plen:].tolist()
                if tid in resp:
                    k = resp.index(tid)
                    tail = [tid]
                    nxt = plen + k + 1
                    # keep the model's own whitespace after </think>, if any
                    if nxt < row.shape[0] and _is_whitespace_token(tokenizer, row[nxt]):
                        tail.append(int(row[nxt]))
                    else:
                        tail += tokenizer.encode("\n\n", add_special_tokens=False)
                    head = row[:plen + k].tolist()
                    forced = False
                else:
                    k = len(resp)
                    head = row.tolist()
                    tail = closer
                    forced = True
                seq = head + tail + pre
                second.append(seq)
                meta.append((i, plen, k, forced, len(tail) + len(pre)))
            rows2 = _greedy(model, tokenizer, device, second, cfg.n_response + MAX_LEAD,
                            cfg.n_response + MAX_LEAD)
            for j, ((i, plen, k, forced, pre_len), row) in enumerate(zip(meta, rows2)):
                start = len(second[j])
                lead = lead_length(tokenizer, row, start)
                results[i] = Generation(row, plen, start + lead, k, forced, pre_len + lead)
        if log_every and (b // cfg.batch_size + 1) % log_every == 0:
            print(f"  {min(b + cfg.batch_size, len(prompts))}/{len(prompts)} generated")
    return results


def generate_answer(tokenizer, model, device, prompt, cfg, system=None):
    """Decoded answer text for one prompt."""
    return generate_batch(tokenizer, model, device, [prompt], cfg, system)[0].answer_text(tokenizer)
