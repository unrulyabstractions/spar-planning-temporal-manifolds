"""Human review annotations, as written by the review UI (review.py).

Each reviewer has one append-only file, `<bank>/reviews/<reviewer-slug>.jsonl`. Each line records one
judgement:

    {"time": "2026-11-07T18:02:11Z", "reviewer": "Alan McIntyre", "uid": "x0042",
     "target": "prompt", "verdict": "problem", "note": "'1380 seconds' reads oddly"}

`target` says what was judged: "prompt" for the prompt itself, or the name of a baseline run (e.g.
"gemma-4-31B-it__t0_s0") for that model's response to it. Changing your mind appends a new line; the latest
line per (reviewer, uid, target) is the current judgement, and the history stays in the file.

One file per reviewer means several people can review at once and commit their files without merge
conflicts. There is no authentication: the reviewer is whoever is selected in the UI's drop-down.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from .records import append_jsonl, read_jsonl

VERDICTS = ("ok", "problem", "unsure")


def reviewer_slug(name: str) -> str:
    """'Alan McIntyre' -> 'alan-mcintyre' (the file name for that reviewer's annotations)."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise ValueError(f"reviewer name {name!r} has no letters or digits")
    return slug


def add_annotation(bank_dir: Path, reviewer: str, uid: str, target: str, verdict: str, note: str) -> dict:
    """Append one judgement to the reviewer's file and return it."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    row = {"time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "reviewer": reviewer,
           "uid": uid, "target": target, "verdict": verdict, "note": note.strip()}
    append_jsonl(Path(bank_dir) / "reviews" / f"{reviewer_slug(reviewer)}.jsonl", [row])
    return row


def current_annotations(bank_dir: Path) -> dict[tuple[str, str], dict[str, dict]]:
    """{(uid, target): {reviewer: their latest judgement}} over every reviewer's file."""
    out: dict[tuple[str, str], dict[str, dict]] = {}
    for path in sorted((Path(bank_dir) / "reviews").glob("*.jsonl")):
        for row in read_jsonl(path):          # file order is time order, so later lines overwrite earlier ones
            out.setdefault((row["uid"], row["target"]), {})[row["reviewer"]] = row
    return out
