"""Reading a model's choice out of its answer, and scoring it against the prompt's correct answer.

The prompt asks for "I choose: <a) or b)>. My reasoning: ...". This parser reads only the *answer* (the text
after the model closed its thinking; see pbank/local_model.py), so nothing the model wrote while deliberating
can be mistaken for its choice.

Accepted forms, all for label "a)": "I choose: a)", "I choose: a", "I choose: **a)**", "I choose: (a)",
"I choose: `a)`". Matching is case-insensitive on "I choose". Two outcomes are flagged rather than guessed:

* no "I choose:" followed by one of the prompt's two labels -> format failure (choice None);
* several "I choose:" lines that name *different* labels -> conflicting (choice None).

This is a heuristic tuned on this template's two label styles (a)/b) and 1)/2)). Hand-audit a sample of its
output for every new model or template before relying on its counts (the review UI shows the raw answer next
to the parsed choice for exactly this purpose).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .records import PromptRecord

# "I choose:" then optional markdown/brackets, then a single label character that does not start a longer word
# (so "I choose: and ..." is not read as "a"), then an optional ")".
_CHOICE = re.compile(r"I\s+choose\s*:\s*[*_`\s]*\(?\s*([A-Za-z0-9])(?![A-Za-z0-9])\s*\)?", re.IGNORECASE)


@dataclass
class ParsedChoice:
    label: Optional[str]     # the chosen label as written in the prompt ("a)", "2)"), or None
    status: str              # "ok" | "no_answer" (thinking never closed) | "format_failure" | "conflicting"
    n_mentions: int          # how many "I choose:" statements the answer contains


def parse_choice(answer: Optional[str], labels: tuple[str, str]) -> ParsedChoice:
    """The label the answer chooses, among the prompt's two `labels` (e.g. ("a)", "b)"))."""
    if answer is None:
        return ParsedChoice(None, "no_answer", 0)
    by_char = {label.rstrip(")").lower(): label for label in labels}   # "a" -> "a)"
    found = [by_char[m.group(1).lower()] for m in _CHOICE.finditer(answer) if m.group(1).lower() in by_char]
    if not found:
        return ParsedChoice(None, "format_failure", 0)
    if len(set(found)) > 1:
        return ParsedChoice(None, "conflicting", len(found))
    return ParsedChoice(found[0], "ok", len(found))


@dataclass
class Score:
    chosen_option: Optional[str]   # "near" | "far" | None
    correct: Optional[bool]        # None when the prompt has no correct answer or no choice was read


def score(record: PromptRecord, parsed: ParsedChoice) -> Score:
    chosen = None if parsed.label is None else record.option_for_label(parsed.label)
    if chosen is None or record.correct_option is None:
        return Score(chosen, None)
    return Score(chosen, chosen == record.correct_option)
