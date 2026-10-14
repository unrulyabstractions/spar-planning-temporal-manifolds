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

Without a prefill but with `cfg.choice_prefix` (the full bank: the FORMAT line
asks for `I choose: <a) or b)>`), the model writes the prefix itself and the
label is located after it in a `cfg.answer_window`-token window, as Alan's
parser reads it (`locate_label`).

Generation is batched (left-padded, greedy) and verified token-identical to
single-prompt generation. With `cfg.cache_dir` set, every prompt's
`Generation` is stored under a key of the settings and its input ids, so a
re-run skips generation; greedy decoding makes this exact.
"""

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

CLOSE_TEXT = "\n</think>\n\n"


def pick_device():
    """cuda, then Intel xpu, then Apple mps, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(cfg):
    device = pick_device()
    print(f"device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"attn_implementation": cfg.attn} if cfg.attn else {}
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=cfg.dtype, **kwargs)
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
    leading space after a prefill or choice prefix ending in ':' and no
    space otherwise."""
    space = " " if (cfg is not None and (cfg.prefill or cfg.choice_prefix)) else ""
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
    lead: int = 0              # leading formatting tokens before the label, e.g. ' **'

    def answer_text(self, tokenizer):
        return tokenizer.decode(self.ids[self.answer_start:])

    def think_text(self, tokenizer):
        return tokenizer.decode(self.ids[self.prompt_len:self.prompt_len + self.think_len])


_INT_FIELDS = ("prompt_len", "answer_start", "think_len", "pre_len", "lead")


def cache_key(cfg, ids, system):
    """sha256 of everything that determines a greedy generation."""
    h = hashlib.sha256()
    h.update(repr((cfg.model, cfg.thinking, cfg.max_think_tokens, cfg.prefill, cfg.n_response,
                   cfg.choice_prefix, cfg.answer_window if cfg.choice_prefix else None, system)).encode())
    h.update(np.asarray(ids, dtype=np.int64).tobytes())
    return h.hexdigest()


def cache_load(cfg, ids, system, device):
    """The cached Generation for these input ids, or None."""
    if not cfg.cache_dir:
        return None
    p = Path(cfg.cache_dir) / f"{cache_key(cfg, ids, system)}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return Generation(torch.tensor(z["ids"], device=device), int(z["prompt_len"]), int(z["answer_start"]),
                      int(z["think_len"]), bool(z["forced"]), int(z["pre_len"]), int(z["lead"]))


def cache_store(cfg, ids, system, gen):
    if not cfg.cache_dir:
        return
    d = Path(cfg.cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{cache_key(cfg, ids, system)}.npz", ids=gen.ids.cpu().numpy(), forced=gen.forced,
             **{f: getattr(gen, f) for f in _INT_FIELDS})


class _StepBar(StoppingCriteria):
    """Never stops; ticks a tqdm bar once per decode step so long batches show
    progress. Hidden for short answer generations."""

    def __init__(self, total, show):
        self.bar = tqdm(total=total, desc="decode", unit="tok", leave=False, mininterval=2,
                        dynamic_ncols=True, disable=not show)

    def __call__(self, input_ids, scores, **kwargs):
        self.bar.update(1)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def _greedy(model, tokenizer, device, id_lists, max_new, min_new):
    """Left-padded greedy generation. Returns unpadded [prompt+generated] per row."""
    tokenizer.padding_side = "left"
    batch = tokenizer.pad({"input_ids": id_lists}, return_tensors="pt").to(device)
    step_bar = _StepBar(max_new, show=max_new > 64)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=max_new, min_new_tokens=min_new,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id,
                             stopping_criteria=StoppingCriteriaList([step_bar]))
    step_bar.bar.close()
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


def locate_label(tokenizer, ids, start, cfg):
    """Index of the label token at or after `start`. With `cfg.choice_prefix`
    the label is the first letter-or-digit token after the model's own
    prefix (markdown in between allowed), as Alan's parser reads it; if the
    prefix never appears, or without one, it is the first letter-or-digit
    token from `start` (a short markdown lead allowed)."""
    if cfg.choice_prefix:
        for j in range(start + 1, len(ids) + 1):
            if cfg.choice_prefix in tokenizer.decode(ids[start:j]):
                return j + lead_length(tokenizer, ids, j, max_lead=len(ids) - j)
    return start + lead_length(tokenizer, ids, start)


def answer_tokens(cfg):
    """Tokens generated after the turn (or the think close): enough to hold
    the model's own choice prefix when the label is located, else just the
    answer window after the pinned prefill."""
    return cfg.answer_window if cfg.choice_prefix else cfg.n_response + MAX_LEAD


def _is_whitespace_token(tokenizer, tok_id):
    return tokenizer.decode([int(tok_id)]).strip() == ""


def _generate_rows(tokenizer, model, device, id_lists, idx, cfg, results):
    """Generate the prompts `idx` as one batch and store a Generation per
    prompt in `results`."""
    if cfg.thinking == "off":
        rows = _greedy(model, tokenizer, device, [id_lists[i] for i in idx],
                       answer_tokens(cfg), answer_tokens(cfg))
        for i, row in zip(idx, rows):
            plen = len(id_lists[i])
            lead = locate_label(tokenizer, row, plen, cfg) - plen
            results[i] = Generation(row, plen, plen + lead, 0, False, lead, lead)
        return
    tid, closer, pre = think_end_id(tokenizer), close_ids(tokenizer), prefill_ids(tokenizer, cfg)
    rows = _greedy(model, tokenizer, device, [id_lists[i] for i in idx], cfg.max_think_tokens, 0)
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
        second.append(head + tail + pre)
        meta.append((i, plen, k, forced, len(tail) + len(pre)))
    rows2 = _greedy(model, tokenizer, device, second, answer_tokens(cfg), answer_tokens(cfg))
    for j, ((i, plen, k, forced, pre_len), row) in enumerate(zip(meta, rows2)):
        start = len(second[j])
        lead = locate_label(tokenizer, row, start, cfg) - start
        results[i] = Generation(row, plen, start + lead, k, forced, pre_len + lead, lead)


def _free_memory(device):
    backend = getattr(torch, str(device).split(":")[0], None)
    if hasattr(backend, "empty_cache"):
        backend.empty_cache()


def generate_batch(tokenizer, model, device, prompts, cfg, system=None):
    """Greedy answers for a list of prompts, in batches of cfg.batch_size.
    Returns one Generation per prompt; cached prompts are not regenerated.
    If a batch runs out of device memory, it is retried and the rest of the
    run continues at two thirds of the batch size, so a too-large
    batch_size costs one batch, not the run."""
    texts = [chat_text(tokenizer, p, cfg, system) for p in prompts]
    id_lists = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    results = [cache_load(cfg, ids, system, device) for ids in id_lists]
    todo = [i for i, r in enumerate(results) if r is None]
    if len(todo) < len(prompts):
        print(f"generation cache: {len(prompts) - len(todo)}/{len(prompts)} prompts cached")
    batch_size, done = cfg.batch_size, 0
    bar = tqdm(total=len(todo), desc="generate", unit="prompt", mininterval=2, dynamic_ncols=True,
               disable=not todo)
    while done < len(todo):
        idx = todo[done:done + batch_size]
        try:
            _generate_rows(tokenizer, model, device, id_lists, idx, cfg, results)
        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            _free_memory(device)
            batch_size = max(1, batch_size * 2 // 3)
            print(f"out of memory at batch size {len(idx)}; continuing at batch size {batch_size}",
                  flush=True)
            continue
        for i in idx:
            g = results[i]
            # a label token with no letter or digit means the logits were garbage (e.g. NaN -> '!'):
            # keep the result for this run but never cache it
            if any(ch.isalnum() for ch in tokenizer.decode([int(g.ids[g.answer_start])])):
                cache_store(cfg, id_lists[i], system, g)
        done += len(idx)
        bar.update(len(idx))
    bar.close()
    return results


def generate_answer(tokenizer, model, device, prompt, cfg, system=None):
    """Decoded answer text for one prompt."""
    return generate_batch(tokenizer, model, device, [prompt], cfg, system)[0].answer_text(tokenizer)
