"""Model review of prompts: a second model reads every prompt and reports problems, for every tier alike.

Why: the generator is code, and code carries its author's assumptions. A reader that sees only the prompt
text catches what the parameters hide: a horizon that reads ambiguously, a rewrite that looks odd, a
CONSTRAINT phrasing that admits a second meaning. The same questionnaire is used for every tier (T0 now,
T1-T3 later), so problems are found and categorized the same way at every level of realism.

What the checker is asked (CHECKER_INSTRUCTIONS below):
  1. the stated horizon, and each option's label, payout and timing, as written;
  2. which options pay out within the horizon, and so the correct answer, under the study's rule;
  3. problems, each in one of a fixed set of categories, and an overall verdict.

How its output is used: (2) is compared with the generator's own answer (`agrees`). A disagreement means
either the generator or the checker is wrong about that prompt, and both cases need a human look. (3) is shown
to reviewers next to the prompt. Nothing is filtered out of the bank automatically.

Checkers: a local Hugging Face model (`local:<model id>`, default `local:Qwen/Qwen3.5-27B`, thinking on) or a
hosted one through OpenRouter (`openrouter:<model id>`, e.g. `openrouter:anthropic/claude-haiku-5.5`). Both get
the identical prompt; the hosted one is also given the JSON schema so its provider can enforce it.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .local_model import Reply, log
from .records import PromptRecord, append_jsonl, read_jsonl

PROBLEM_CATEGORIES = ("grammar", "ambiguity", "inconsistency", "implausible", "instructions", "other")

CHECKER_INSTRUCTIONS = """\
You are checking prompts written for a research study. In the study, a language model is given each prompt and \
asked to choose between investment options. Your job is NOT to answer the prompt as the study's model would. \
Your job is to check that the prompt is clear and well-formed, and to work out its correct answer under the \
study's rule.

The study's rule: an option counts only if its payout arrives within the stated time horizon. A payout that \
arrives exactly at the end of the horizon counts. If exactly one option pays out within the horizon, that option \
is correct. If several do, the one with the largest payout is correct. If none does, no option is correct. If \
the prompt states no time horizon, no option is correct. When comparing durations written in different units, \
use 1 year = 365.25 days, 1 month = 1/12 year, 1 week = 7 days.

Read the prompt between the markers and report:
- stated_horizon: the time horizon exactly as the prompt states it, or "none" if it states none.
- options: for each option, its label exactly as written (for example "a)" or "2)"), its payout as written, and \
when it pays out as written.
- options_within_horizon: the labels of the options that pay out within the horizon (an empty list if none does).
- expected_answer: the label of the correct option exactly as written, or "none" if no option is correct, or \
"no_horizon" if the prompt states no time horizon.
- problems: anything that could make a careful reader misread the prompt or disagree about its correct answer. \
Each problem has a category and a one-sentence detail. Categories: "grammar" (spelling, grammar, number or unit \
formatting), "ambiguity" (a part of the prompt has more than one reasonable reading), "inconsistency" (parts of \
the prompt contradict each other), "implausible" (numbers or scenario unrealistic enough to distract a reader), \
"instructions" (the requested answer format is unclear), "other". Use an empty list if there are no problems.
- verdict: "ok" if there are no problems, otherwise "problem".

These are intended and are not problems: the scenario is short and stylized; the payouts are lump sums; in some \
prompts no option is correct; some prompts state no horizon at all.

Reply with only a JSON object with exactly the keys stated_horizon, options (a list of objects with keys label, \
payout, timing), options_within_horizon, expected_answer, problems (a list of objects with keys category, \
detail), and verdict. Do not add any other text.

<<<PROMPT
{prompt}
PROMPT>>>"""

CHECK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["stated_horizon", "options", "options_within_horizon", "expected_answer", "problems", "verdict"],
    "properties": {
        "stated_horizon": {"type": "string"},
        "options": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["label", "payout", "timing"],
            "properties": {"label": {"type": "string"}, "payout": {"type": "string"}, "timing": {"type": "string"}}}},
        "options_within_horizon": {"type": "array", "items": {"type": "string"}},
        "expected_answer": {"type": "string"},
        "problems": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["category", "detail"],
            "properties": {"category": {"type": "string", "enum": list(PROBLEM_CATEGORIES)},
                           "detail": {"type": "string"}}}},
        "verdict": {"type": "string", "enum": ["ok", "problem"]},
    },
}


def checker_prompt(record: PromptRecord) -> str:
    return CHECKER_INSTRUCTIONS.format(prompt=record.text)


# ---------------------------------------------------------------------------------------------------------------
# Reading the checker's reply
# ---------------------------------------------------------------------------------------------------------------

def extract_json(answer: Optional[str]) -> tuple[Optional[dict], Optional[str]]:
    """(the first JSON object in `answer` that has the required keys, None) or (None, why not).
    Tolerates code fences and text around the object, which local models often add despite instructions."""
    if not answer:
        return None, "empty answer"
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", answer):
        try:
            obj, _ = decoder.raw_decode(answer, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and all(k in obj for k in CHECK_SCHEMA["required"]):
            return obj, None
    return None, "no JSON object with the required keys"


def normalize_answer(label: Optional[str]) -> str:
    """Comparable form of an answer: "a)", "a", "(A)" -> "a"; "none" and "no_horizon" unchanged."""
    if label is None:
        return "none"
    return label.strip().strip("()").strip().lower()


def generator_answer(record: PromptRecord) -> str:
    """The generator's answer in the checker's vocabulary."""
    if record.regime == "no_horizon":
        return "no_horizon"
    return record.correct_label or "none"


@dataclass
class CheckResult:
    uid: str
    checker: str                       # e.g. "local:Qwen/Qwen3.5-27B"
    agrees: Optional[bool]             # checker's expected answer == generator's; None if the reply was unreadable
    checker_answer: Optional[str]      # as the checker wrote it
    generator_answer: str              # see generator_answer()
    verdict: Optional[str]             # "ok" | "problem" | None (unreadable)
    problems: list                     # [{"category": ..., "detail": ...}, ...]
    parsed: Optional[dict]             # the checker's full JSON reply
    parse_error: Optional[str]         # why the reply could not be read, if it could not
    request_error: Optional[str]       # hosted models: the request failed (will be retried on the next run)
    raw_answer: Optional[str]
    thinking: Optional[str]
    n_generated: int
    hit_token_cap: bool


def interpret(record: PromptRecord, checker: str, reply: Reply) -> CheckResult:
    if reply.error:                    # the request failed: nothing to read (recorded in request_error)
        parsed, err = None, None
    elif reply.answer is None:         # a local model never closed its thinking
        parsed, err = None, "thinking never closed"
    else:
        parsed, err = extract_json(reply.answer)
    checker_answer = None if parsed is None else str(parsed["expected_answer"])
    agrees = None if parsed is None else normalize_answer(checker_answer) == normalize_answer(generator_answer(record))
    problems = [] if parsed is None else [p for p in parsed.get("problems", []) if isinstance(p, dict)]
    return CheckResult(
        uid=record.uid, checker=checker, agrees=agrees, checker_answer=checker_answer,
        generator_answer=generator_answer(record), verdict=None if parsed is None else parsed.get("verdict"),
        problems=problems, parsed=parsed, parse_error=err, request_error=reply.error,
        raw_answer=reply.answer, thinking=reply.thinking, n_generated=reply.n_generated,
        hit_token_cap=reply.hit_token_cap)


# ---------------------------------------------------------------------------------------------------------------
# Running a checker over a bank
# ---------------------------------------------------------------------------------------------------------------

def checker_file_name(checker: str) -> str:
    """'local:Qwen/Qwen3.5-27B' -> 'local__Qwen__Qwen3.5-27B.jsonl'."""
    return checker.replace(":", "__").replace("/", "__") + ".jsonl"


def latest_by_uid(rows: list[dict]) -> dict[str, dict]:
    """The last row written for each uid (a rerun appends; the newest row wins)."""
    return {row["uid"]: row for row in rows}


def load_model(checker: str, *, reasoning_effort: Optional[str]):
    """The model behind a checker spec "local:<hf id>" or "openrouter:<openrouter id>"."""
    backend, _, model_id = checker.partition(":")
    if backend == "local":
        from .local_model import LocalChatModel
        return LocalChatModel(model_id)
    if backend == "openrouter":
        from .hosted_model import OpenRouterModel
        return OpenRouterModel(model_id, reasoning_effort=reasoning_effort)
    raise ValueError(f"checker must be 'local:<model id>' or 'openrouter:<model id>', got {checker!r}")


def run_checker(records: list[PromptRecord], checker: str, out_path: Path, *, model=None, batch_size: int = 8,
                max_new_tokens: int = 8192, thinking: bool = True, reasoning_effort: Optional[str] = "medium",
                prompts_sha256: str = "") -> list[dict]:
    """Check every record not already checked in `out_path`, appending one CheckResult row per prompt, and
    write the checker's settings next to it (`<name>.meta.json`).

    Resumable: prompts whose latest row in `out_path` succeeded are skipped; prompts whose request failed are
    retried. A resume with different settings is refused, so one file never mixes settings. Pass `model` to
    reuse an already-loaded model (tests pass a fake one)."""
    hosted = checker.startswith("openrouter:")
    # Everything that changes what the checker writes. (Batch size matters for local models only: batching
    # can change greedy generation on near-tied steps.)
    settings = {"checker": checker, "instructions": CHECKER_INSTRUCTIONS, "max_new_tokens": max_new_tokens,
                "thinking": None if hosted else thinking, "reasoning_effort": reasoning_effort if hosted else None,
                "batch_size": None if hosted else batch_size}
    meta_path = out_path.with_suffix(".meta.json")
    if meta_path.exists():
        previous = json.loads(meta_path.read_text()).get("settings")
        if previous != settings:
            changed = sorted(k for k in settings if (previous or {}).get(k) != settings[k])
            raise SystemExit(f"{out_path} was written with different checker settings ({', '.join(changed)}). "
                             f"Resume with the same settings, or move that file aside to start over.")

    done = {uid for uid, row in latest_by_uid(read_jsonl(out_path)).items() if not row.get("request_error")}
    todo = [r for r in records if r.uid not in done]
    log(f"checker {checker}: {len(done)} already checked, {len(todo)} to check -> {out_path}")
    if not todo:
        return read_jsonl(out_path)
    model = model or load_model(checker, reasoning_effort=reasoning_effort)
    meta = {"settings": settings, "prompts_sha256": prompts_sha256, "model": model.describe()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        prompts = [checker_prompt(r) for r in batch]
        if hosted:
            replies = model.generate(prompts, max_new_tokens=max_new_tokens, json_schema=CHECK_SCHEMA)
        else:
            replies = model.generate(prompts, thinking=thinking, max_new_tokens=max_new_tokens)
        results = [interpret(r, checker, reply) for r, reply in zip(batch, replies)]
        append_jsonl(out_path, [asdict(c) for c in results])
        n_disagree = sum(c.agrees is False for c in results)
        n_unreadable = sum(c.agrees is None for c in results)
        log(f"  [{i + len(batch)}/{len(todo)}] disagree {n_disagree}, unreadable {n_unreadable}, "
            f"flagged {sum(c.verdict == 'problem' for c in results)} in this batch")
    return read_jsonl(out_path)
