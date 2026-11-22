"""Finding the answer-label token in a generated reply: the site where activations are read.

The report (Appendix A.3) reads the residual stream "at the answer-label token". In a reply such as

    <thinking ...> </think>  I choose: b). My reasoning: ...

that is the token carrying the "b" of the model's own "I choose: b)". The token just before it (normally the
space or ":" after "I choose") is also kept, as the "prelabel" site: its state is the one that *predicts* the
label, while the label token's state has already seen it. Which of the two to fit on is a choice for fit.py.

The label is found with the same regular expression that pbank's parser uses (pbank.answers.CHOICE_PATTERN),
applied to the same text (the decoded answer, after the end-of-thinking token), and the first "I choose:" naming
one of the prompt's two labels is taken. The parser only accepts a reply whose "I choose:" statements all agree,
so that first one is the parsed choice; `locate_label` checks this and raises if they ever differ.

Character position -> token: the answer is decoded one token longer at a time until the decoded prefix covers the
label character. Decoding prefixes rather than using the tokenizer's offset mapping works for any tokenizer and
reads exactly the text the parser read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pbank.answers import CHOICE_PATTERN
from pbank.local_model import END_OF_THINKING_TOKENS, single_token_id


@dataclass
class LabelSite:
    label_index: int      # index in the generated ids of the token that carries the label character
    answer_start: int     # index in the generated ids where the answer begins (after end-of-thinking; 0 if none)
    label_char: str       # the label character as the model wrote it, e.g. "b" or "2"


def locate_label(tok, generated_ids: list[int], labels: tuple[str, str], thinking: bool) -> Optional[LabelSite]:
    """Where the model wrote its choice, or None if the reply has no answer or no "I choose: <label>"."""
    if thinking:
        eot = single_token_id(tok, END_OF_THINKING_TOKENS, "end-of-thinking")
        if eot not in generated_ids:
            return None
        start = generated_ids.index(eot) + 1
    else:
        start = 0
    answer_ids = generated_ids[start:]
    answer = tok.decode(answer_ids, skip_special_tokens=True)

    label_chars = {label.rstrip(")").lower() for label in labels}
    match = next((m for m in CHOICE_PATTERN.finditer(answer) if m.group(1).lower() in label_chars), None)
    if match is None:
        return None
    c = match.start(1)   # character index of the label in the answer

    for j in range(len(answer_ids)):
        prefix = tok.decode(answer_ids[:j + 1], skip_special_tokens=True)
        if len(prefix) > c and prefix[:c + 1] == answer[:c + 1]:
            return LabelSite(label_index=start + j, answer_start=start, label_char=match.group(1))
    raise AssertionError("decoded prefixes never covered the label character")   # decode(all) == answer


def replay_sites(prompt_len: int, site: LabelSite, thinking: bool) -> dict[str, int]:
    """Each extraction site's index in the replayed sequence, which is the model input (`prompt_len` tokens)
    followed by the generated ids up to and including the label token.

    prompt_end is the last token of the model input: its state predicts the first generated token. think_end is the
    end-of-thinking token, the one before the answer begins: its state predicts the first token of the answer. It
    exists only for a thinking run. With thinking off, a label that is the first generated token makes prelabel and
    prompt_end the same token.
    """
    label = prompt_len + site.label_index
    where = {"label": label, "prelabel": label - 1, "prompt_end": prompt_len - 1}
    if thinking:
        if site.answer_start < 1:
            raise ValueError("a thinking reply's answer starts after its end-of-thinking token")
        where["think_end"] = prompt_len + site.answer_start - 1
    if not 0 <= where["prompt_end"] <= where.get("think_end", where["prompt_end"]) <= where["prelabel"]:
        raise ValueError(f"extraction sites out of order: {where}")
    return where


# ---- the horizon site: the prompt tokens that state the horizon ----------------------------------------------------

CONSTRAINT_PREFIX = "CONSTRAINT: "


@dataclass
class HorizonSite:
    char_start: int            # the horizon's characters in the prompt text: text[char_start:char_end] == horizon
    char_end: int
    token_indices: list[int]   # its tokens, as indices into the tokenized model input (chat template included)
    token_ids: list[int]
    tokens: list[str]          # each token decoded on its own, e.g. ["1", "5", " minutes"]
    n_input_tokens: int        # length of the tokenized model input

    @property
    def decoded(self) -> str:
        return "".join(self.tokens)


def horizon_char_span(text: str, horizon: str) -> tuple[int, int]:
    """Where the horizon is written in the prompt: its one occurrence in the CONSTRAINT line. (The same words can
    also appear in an option, "10,000 dollars in 3 months", which is not the horizon.)"""
    lines = [m for m in range(len(text)) if text.startswith(CONSTRAINT_PREFIX, m) and (m == 0 or text[m - 1] == "\n")]
    if len(lines) != 1:
        raise ValueError(f"expected one CONSTRAINT line, found {len(lines)}")
    end = text.find("\n", lines[0])
    end = len(text) if end < 0 else end
    if text.count(horizon, lines[0], end) != 1:
        raise ValueError(f"the CONSTRAINT line states {horizon!r} {text.count(horizon, lines[0], end)} times, not once")
    start = text.index(horizon, lines[0], end)
    return start, start + len(horizon)


def locate_horizon(tok, chat: str, text: str, horizon: str) -> HorizonSite:
    """The tokens of `chat` (the model input: the chat template around the prompt `text`) that state `horizon`.

    They are the tokens whose characters overlap the horizon's span, found with the tokenizer's offset mapping (a
    fast tokenizer is required). Tokenize `chat` exactly as generation does: no special tokens added, since the
    template already contains them. Whether the tokens found spell the horizon and nothing else is for the caller to
    check (`site.decoded.strip() == horizon`): a tokenizer could merge the unit with the following full stop.
    """
    if chat.count(text) != 1:
        raise ValueError(f"the prompt text appears {chat.count(text)} times in the model input, not once")
    start, end = horizon_char_span(text, horizon)
    base = chat.index(text)
    enc = tok(chat, add_special_tokens=False, return_offsets_mapping=True)
    idx = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if a < base + end and b > base + start]
    if not idx or idx != list(range(idx[0], idx[-1] + 1)):
        raise ValueError(f"the horizon's tokens are not one contiguous run: {idx}")
    ids = [enc["input_ids"][k] for k in idx]
    return HorizonSite(char_start=start, char_end=end, token_indices=idx, token_ids=ids,
                       tokens=[tok.decode([i]) for i in ids], n_input_tokens=len(enc["input_ids"]))
