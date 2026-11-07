"""The checker: reading its JSON reply, comparing with the generator, and resuming a run (with a fake model)."""

import json

from pbank.checker import (CHECK_SCHEMA, checker_file_name, checker_prompt, extract_json, generator_answer,
                           interpret, latest_by_uid, normalize_answer, run_checker)
from pbank.local_model import Reply
from pbank.records import read_jsonl
from pbank.t0 import build_extraction_bank, build_verification_grid


def reply(answer, error=None):
    return Reply(thinking="...", answer=answer, n_generated=10, hit_token_cap=False, seconds=0.1, error=error)


def check_json(expected, verdict="ok", problems=()):
    return json.dumps({"stated_horizon": "8 years", "options": [], "options_within_horizon": [],
                       "expected_answer": expected, "problems": list(problems), "verdict": verdict})


def test_extract_json_tolerates_fences_and_prose():
    body = check_json("a)")
    for wrapped in (body, f"```json\n{body}\n```", f"Here is my review:\n{body}\nDone.", "{oops} " + body):
        obj, err = extract_json(wrapped)
        assert err is None and obj["expected_answer"] == "a)"
    assert extract_json('{"verdict": "ok"}') == (None, "no JSON object with the required keys")
    assert extract_json("")[1] == "empty answer"


def test_normalize_and_generator_answer():
    assert normalize_answer("a)") == normalize_answer(" (A) ") == normalize_answer("a") == "a"
    bank = build_extraction_bank()
    by_regime = {r.regime: r for r in bank}
    assert generator_answer(by_regime["no_horizon"]) == "no_horizon"
    assert generator_answer(by_regime["none"]) == "none"
    assert generator_answer(by_regime["both"]) == by_regime["both"].far_label


def test_interpret_agree_disagree_unreadable():
    rec = build_verification_grid()[0]
    agree = interpret(rec, "fake", reply(check_json(rec.correct_label)))
    disagree = interpret(rec, "fake", reply(check_json(rec.far_label, "problem",
                                                       [{"category": "ambiguity", "detail": "x"}])))
    unreadable = interpret(rec, "fake", reply("I think it's fine"))
    unclosed = interpret(rec, "fake", reply(None))
    failed = interpret(rec, "fake", reply(None, error="RateLimitError: slow down"))
    for name, c in [("agree", agree), ("disagree", disagree), ("unreadable", unreadable), ("unclosed", unclosed),
                    ("failed", failed)]:
        print(f"{name:<10} agrees={c.agrees} verdict={c.verdict} parse_error={c.parse_error} "
              f"request_error={c.request_error}")
    assert agree.agrees is True and agree.verdict == "ok"
    assert disagree.agrees is False and disagree.problems[0]["category"] == "ambiguity"
    assert unreadable.agrees is None and unreadable.parse_error
    assert unclosed.parse_error == "thinking never closed"
    assert failed.agrees is None and failed.request_error and failed.parse_error is None


def test_prompt_and_schema_are_consistent():
    rec = build_verification_grid()[0]
    text = checker_prompt(rec)
    assert rec.text in text
    for key in CHECK_SCHEMA["required"]:
        assert key in text, key
    assert checker_file_name("local:Qwen/Qwen3.5-27B") == "local__Qwen__Qwen3.5-27B.jsonl"


class FakeHostedModel:
    """Answers correctly, except that the first call fails for one prompt (as a rate-limited request would)."""

    def __init__(self, records, fail_uid):
        self.by_text = {r.text: r for r in records}
        self.fail_uid, self.calls = fail_uid, 0

    def generate(self, prompts, max_new_tokens, json_schema=None):
        assert json_schema is CHECK_SCHEMA            # hosted checkers are given the schema
        self.calls += 1
        out = []
        for p in prompts:
            rec = next(r for t, r in self.by_text.items() if t in p)
            if rec.uid == self.fail_uid and self.calls == 1:
                out.append(reply(None, error="RateLimitError"))
            else:
                out.append(reply(check_json(generator_answer(rec))))
        return out

    def describe(self):
        return {"model_id": "fake"}


def test_run_checker_resumes_and_retries_failed_requests(tmp_path):
    records = build_verification_grid()[:6]
    out = tmp_path / "checks" / "fake.jsonl"
    model = FakeHostedModel(records, fail_uid=records[2].uid)
    run_checker(records, "openrouter:fake", out, model=model, batch_size=10, prompts_sha256="abc")
    first = latest_by_uid(read_jsonl(out))
    assert first[records[2].uid]["request_error"] and sum(r["agrees"] is True for r in first.values()) == 5

    run_checker(records, "openrouter:fake", out, model=model, batch_size=10)   # only the failed prompt again
    rows = read_jsonl(out)
    latest = latest_by_uid(rows)
    print(f"{len(rows)} rows written for {len(latest)} prompts after the retry")
    assert len(rows) == 7 and all(r["agrees"] is True for r in latest.values())
    assert json.loads(out.with_suffix(".meta.json").read_text())["model"] == {"model_id": "fake"}


def test_resume_with_different_settings_is_refused(tmp_path):
    import pytest
    records = build_verification_grid()[:2]
    out = tmp_path / "checks" / "fake.jsonl"
    run_checker(records, "openrouter:fake", out, model=FakeHostedModel(records, fail_uid=None))
    with pytest.raises(SystemExit, match="reasoning_effort"):
        run_checker(records, "openrouter:fake", out, model=FakeHostedModel(records, fail_uid=None),
                    reasoning_effort="high")
