"""Chat-template encoding and the token positions we capture.

Positions are named, never assumed: the transition window is every token from the last
end-of-turn token of the user turn (`<|im_end|>` for Qwen3/Qwen3.5, `<turn|>` for Gemma 4) through
the end of the generation prompt (including the no-thinking prefill), and the response window is
the first `n_response` generated tokens. `tests/test_chat_tokens.py` asserts the exact token strings
for each model family used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# End-of-turn token of each chat-template family in use: Qwen3 / Qwen3.5, Gemma 4. A tokenizer must have exactly one.
END_OF_TURN_TOKENS = ("<|im_end|>", "<turn|>")


def end_of_turn_id(tok) -> int:
    """Id of the tokenizer's end-of-turn token. Unknown tokens map to None (Qwen) or to the unk id (Gemma)."""
    found = {t: i for t in END_OF_TURN_TOKENS
             if (i := tok.convert_tokens_to_ids(t)) is not None and i != tok.unk_token_id}
    if len(found) != 1:
        raise ValueError(f"expected exactly one of {END_OF_TURN_TOKENS} in the vocabulary, found {sorted(found)}; "
                         f"unverified chat template family")
    return next(iter(found.values()))


@dataclass
class ChatEncoding:
    input_ids: list[int]
    transition_start: int          # index of the last end-of-turn token in the prompt
    transition_tokens: list[str]   # decoded token strings, one per transition position

    @property
    def prompt_len(self) -> int:
        return len(self.input_ids)

    @property
    def transition_positions(self) -> list[int]:
        return list(range(self.transition_start, self.prompt_len))


def encode_user_turn(tok, user_text: str, enable_thinking: bool = False) -> ChatEncoding:
    """Apply the model's own chat template to a single user turn and locate the transition window."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    ids = tok(text, add_special_tokens=False)["input_ids"]
    eot = end_of_turn_id(tok)
    occurrences = [i for i, t in enumerate(ids) if t == eot]
    if not occurrences:
        raise ValueError(f"no {tok.decode([eot])!r} found in encoded prompt; wrong chat template family?")
    start = occurrences[-1]
    toks = [tok.decode([t]) for t in ids[start:]]
    return ChatEncoding(input_ids=ids, transition_start=start, transition_tokens=toks)


def position_labels(n_transition: int, n_response: int) -> list[str]:
    return [f"T{i}" for i in range(n_transition)] + [f"R{i}" for i in range(n_response)]


def label_first_token_id(tok, label: str) -> int:
    """Token id the model emits first when writing ` {label}` after `I choose:`."""
    return tok(" " + label, add_special_tokens=False)["input_ids"][0]


def label_first_token_ids(tok, label: str) -> set[int]:
    """First-token ids for the label with and without a leading space (` a` after `:`; `a` after ` **`)."""
    return {tok(" " + label, add_special_tokens=False)["input_ids"][0], tok(label, add_special_tokens=False)["input_ids"][0]}


def find_choice(tok, gen_ids: list[int], label_a: str, label_b: str, prefix: str = "I choose:") -> tuple[Optional[int], Optional[str]]:
    """Locate the choice token in generated ids.

    Returns (index into gen_ids of the label's first token, 'a' | 'b'), or (None, None) when the
    response does not follow the format. The decoded prefix must end with `prefix` so an incidental
    ' a' article elsewhere is not mistaken for the choice.
    """
    ia, ib = label_first_token_ids(tok, label_a), label_first_token_ids(tok, label_b)
    if ia & ib:
        raise ValueError(f"labels {label_a!r} and {label_b!r} share a first token; choice readout is ambiguous")
    for j in range(1, len(gen_ids)):
        if gen_ids[j] in ia or gen_ids[j] in ib:
            # allow markdown emphasis between the prefix and the label: `I choose: **a)`
            before = tok.decode(gen_ids[:j]).rstrip().rstrip("*_").rstrip()
            if before.endswith(prefix):
                return j, ("a" if gen_ids[j] in ia else "b")
    return None, None


def find_anchor(tok, gen_ids: list[int], prefix: str) -> Optional[int]:
    """Index into gen_ids of the first token AFTER the generated text ends with `prefix` (e.g. "My reasoning:"),
    or None if the prefix never occurs (or nothing follows it)."""
    for j in range(1, len(gen_ids)):
        if tok.decode(gen_ids[:j]).rstrip().endswith(prefix):
            return j
    return None


ANCHOR_LABELS = ["M0", "E0", "MEAN"]   # first reasoning token, last generated token, mean over M0..E0
