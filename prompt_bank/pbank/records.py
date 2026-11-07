"""The prompt record (one row of the bank), the rule that decides the correct answer, and file I/O.

A bank is a directory:

    data/<bank name>/
        prompts.jsonl          one PromptRecord per line (all splits)
        manifest.json          generation settings, counts, and a SHA-256 over the prompts
        checks/<checker>.jsonl one CheckResult per prompt, per checker model (pbank/checker.py)
        reviews/<reviewer>.jsonl  human annotations from the review UI (pbank/annotations.py)

JSON Lines (one JSON object per line) is used throughout so that every file can be read with `head`,
diffed in git, and appended to safely by a long run that may be interrupted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .durations import Duration

# ---------------------------------------------------------------------------------------------------------------
# The decision rule (report, Methods "Tasks and data")
# ---------------------------------------------------------------------------------------------------------------
#
# An option *pays off within the horizon* when its delay is at most the horizon (delay <= horizon; a payout
# that arrives exactly at the end of the horizon counts). Then:
#
#   regime        which options pay off      correct answer
#   ----------    -----------------------    ---------------------------------------------
#   "none"        neither                    none (used for extraction, not for scoring)
#   "near_only"   only the near option       the near option
#   "both"        both                       the far option (its reward is always larger)
#   "no_horizon"  (no horizon is stated)     none (the null control)
#
# The far option's reward is the near reward times 2-50, so it is always the larger one, and its delay
# (2-50 years) is always longer than the near delay (1 week - 1 year); so "only the far option pays" cannot occur.

REGIMES = ("none", "near_only", "both", "no_horizon")


def pays_off(delay: Duration, horizon: Duration) -> bool:
    """True when a payout after `delay` arrives within `horizon` (exact comparison; a tie counts)."""
    return delay.days <= horizon.days


def regime(horizon: Optional[Duration], near_delay: Duration, far_delay: Duration) -> str:
    """Which options pay off within the horizon; see the table above."""
    if horizon is None:
        return "no_horizon"
    near, far = pays_off(near_delay, horizon), pays_off(far_delay, horizon)
    if near and far:
        return "both"
    if near:
        return "near_only"
    if far:   # unreachable with this bank's delay grids; kept so a future grid fails loudly instead of silently
        raise ValueError(f"far option ({far_delay}) pays off but near option ({near_delay}) does not")
    return "none"


CORRECT_OPTION = {"near_only": "near", "both": "far"}   # regimes absent here have no correct answer


# ---------------------------------------------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Option:
    """One of the two investment options as shown in the prompt."""
    reward: int              # dollars, e.g. 10000 (rendered "10,000 dollars")
    delay: str               # as written, e.g. "1 year"
    delay_years: float       # approximate, for display and sorting only


@dataclass
class PromptRecord:
    """Everything about one prompt. The rendered `text` is what the model sees; every other field is a
    parameter that produced it or a fact derived from those parameters, so any surface feature can be used to
    filter, group, or control for in analysis."""

    uid: str                         # stable id, e.g. "x0042" (extraction), "x0042-null", "v017-far-first"
    tier: str                        # "T0" (structured); T1-T3 later
    split: str                       # "extraction" (fit steering vectors) | "verification" (score behaviour)
    text: str                        # the full prompt, exactly as sent to the model

    # The horizon. All None for the null control.
    horizon: Optional[str]           # as written, e.g. "100 weeks"
    horizon_grid_value: Optional[str]  # the grid point it renders, e.g. "23 months" (same as `horizon` unless rewritten)
    horizon_unit: Optional[str]      # unit as written, e.g. "weeks"
    horizon_years: Optional[float]   # approximate, display only
    rewritten: bool                  # True if the horizon is written in a unit other than its natural one
    phrasing_id: Optional[int]       # which CONSTRAINT phrasing (0-3), see pbank/t0.py

    # The options, in near/far terms (not in display order).
    near: Option
    far: Option
    reward_multiplier: int           # far reward / near reward
    near_first: bool                 # True if the near option is listed first
    labels: tuple[str, str]          # the two labels in display order, e.g. ("a)", "b)") or ("1)", "2)")

    # Derived facts.
    regime: str                      # see REGIMES
    correct_option: Optional[str]    # "near" | "far" | None
    correct_label: Optional[str]     # the label of the correct option, e.g. "b)"; None when no answer is correct
    horizon_equals_a_delay: bool     # the horizon is exactly as long as one of the delays (a tie; counts as paying)
    h_over_far_delay: Optional[float]  # horizon / far delay; the verification grid spans 0.35-2.8

    pair_id: Optional[str] = None    # shared by prompts built as a pair (a null twin and its source; both orders)
    notes: dict = field(default_factory=dict)   # free-form extra parameters (e.g. verification ratio target)

    @property
    def near_label(self) -> str:
        return self.labels[0] if self.near_first else self.labels[1]

    @property
    def far_label(self) -> str:
        return self.labels[1] if self.near_first else self.labels[0]

    def option_for_label(self, label: str) -> Optional[str]:
        """"near" or "far" for a label as it appears in the prompt ("a)"), or None if it is neither."""
        return {self.near_label: "near", self.far_label: "far"}.get(label)


# ---------------------------------------------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------------------------------------------

def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write `rows` as JSON Lines, replacing the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Append `rows` and flush, so an interrupted run keeps everything written so far."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict]:
    """All rows of a JSON Lines file ([] if it does not exist)."""
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def record_to_dict(r: PromptRecord) -> dict:
    return asdict(r)


def record_from_dict(d: dict) -> PromptRecord:
    d = dict(d)
    d["near"], d["far"] = Option(**d["near"]), Option(**d["far"])
    d["labels"] = tuple(d["labels"])
    return PromptRecord(**d)


def load_prompts(bank_dir: Path) -> list[PromptRecord]:
    """The bank's prompts, in file order."""
    return [record_from_dict(d) for d in read_jsonl(Path(bank_dir) / "prompts.jsonl")]


def prompts_sha256(records: Iterable[PromptRecord]) -> str:
    """SHA-256 over "uid|text" lines. Two banks with this hash present byte-identical prompts under the same
    ids; baseline runs and checks record it so their results can be matched to the exact prompts they saw."""
    h = hashlib.sha256()
    for r in records:
        h.update(f"{r.uid}|{r.text}\n".encode())
    return h.hexdigest()
