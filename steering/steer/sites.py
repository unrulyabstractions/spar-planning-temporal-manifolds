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
