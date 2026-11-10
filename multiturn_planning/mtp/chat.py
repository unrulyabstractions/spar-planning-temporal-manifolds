"""Chat encoding and named capture positions for one turn.

For turn k the model sees the chat template rendered over the history (user 1, assistant 1, "Continue", ...,
user k) with the generation prompt. With thinking off, Qwen3 appends an empty think block to the *current*
generation prompt only and drops it from earlier assistant turns, so each turn is captured on exactly the
sequence it was generated from (prompt_k + reply_k), never on a re-tokenised transcript.

Positions (index into prompt_k + reply_k):
  P0..P8  pre-reply boundary window: last `<|im_end|>` of user turn k through the end of the generation prompt
          (`<|im_end|> \\n <|im_start|> assistant \\n <think> \\n\\n </think> \\n\\n` on Qwen3; asserted in tests)
  U       last token of user turn k (the token before that `<|im_end|>`)
  H       token ending "Time horizon:" in the reply (predicts the value), step turns only
  V       last token of the horizon duration in the reply, step turns only
  E       the `<|im_end|>` that ends the reply (absent if the reply was truncated)
"""

from __future__ import annotations

import re

from .prompts import _DURATION, _HORIZON_LINE, CONTINUE, DONE, SYSTEM

N_PRE = 9
POSITIONS = [f"P{i}" for i in range(N_PRE)] + ["U", "H", "V", "E"]


def messages_for_turn(first_user: str, replies: list[str]) -> list[dict]:
    """History for turn k = len(replies) + 1."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": first_user}]
    for r in replies:
        msgs += [{"role": "assistant", "content": r}, {"role": "user", "content": CONTINUE}]
    return msgs


def encode_prompt(tok, messages: list[dict]) -> list[int]:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def pre_window(tok, prompt_ids: list[int]) -> list[int]:
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    last = max(i for i, t in enumerate(prompt_ids) if t == im_end)
    idx = list(range(last, len(prompt_ids)))
    if len(idx) != N_PRE:
        raise ValueError(f"pre-reply window has {len(idx)} tokens, expected {N_PRE}: "
                         f"{[tok.decode([t]) for t in prompt_ids[last:]]}")
    return idx


def _char_ends(tok, ids: list[int]) -> list[int]:
    """End character offset (in the decoded reply) of each token, by incremental decoding."""
    return [len(tok.decode(ids[: i + 1])) for i in range(len(ids))]


def _token_at(ends: list[int], char_index: int) -> int:
    """Index of the token whose decoded span contains `char_index`."""
    for i, e in enumerate(ends):
        if e > char_index:
            return i
    return len(ends) - 1


def reply_positions(tok, reply_ids: list[int]) -> dict[str, int | None]:
    """H, V, E as offsets into reply_ids (None where absent)."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    out = {"H": None, "V": None, "E": reply_ids.index(im_end) if im_end in reply_ids else None}
    body = reply_ids[: out["E"]] if out["E"] is not None else reply_ids
    text = tok.decode(body)
    m = _HORIZON_LINE.search(text)
    if m:
        ends = _char_ends(tok, body)
        out["H"] = _token_at(ends, m.start(1))
        d = _DURATION.search(text, m.start(2))
        if d and d.start() < m.end(2):
            out["V"] = _token_at(ends, d.end() - 1)
    return out


def turn_positions(tok, prompt_ids: list[int], reply_ids: list[int]) -> dict[str, int | None]:
    """All named positions as indices into prompt_ids + reply_ids."""
    pre = pre_window(tok, prompt_ids)
    pos = {f"P{i}": p for i, p in enumerate(pre)}
    pos["U"] = pre[0] - 1
    n = len(prompt_ids)
    for k, v in reply_positions(tok, reply_ids).items():
        pos[k] = None if v is None else n + v
    return pos


def clean_reply(text: str) -> str:
    """What goes back into the history: the reply without trailing special tokens/whitespace."""
    return re.sub(r"(<\|im_end\|>|<\|endoftext\|>)+\s*$", "", text).strip()


_NEXT_STEP = re.compile(r"\n\s*[#*]*\s*step\s*\d", re.IGNORECASE)


def cut_step_reply(tok, reply_ids: list[int]) -> tuple[list[int], bool]:
    """A step reply must hold one step. If the model starts another step or appends "Plan Completed",
    cut before it and close the turn with <|im_end|> (logged as `cut`). Other replies pass unchanged."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    body = reply_ids[: reply_ids.index(im_end)] if im_end in reply_ids else reply_ids
    text = tok.decode(body)
    starts = [m.start() for m in [_NEXT_STEP.search(text, 1)] if m] + [i for i in [text.find(DONE, 1)] if i > 0]
    if not starts:
        return reply_ids, False
    cut_char = min(starts)
    ends = _char_ends(tok, body)
    starts_at = [0] + ends[:-1]                      # a token straddling the cut (e.g. ".\n\n") is kept
    keep = [t for t, st in zip(body, starts_at) if st < cut_char]
    while keep and tok.decode([keep[-1]]).strip() == "":
        keep.pop()
    return keep + [im_end], True
