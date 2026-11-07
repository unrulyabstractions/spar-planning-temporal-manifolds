"""The review UI end to end with Flask's test client: list, filters, item page, choosing a reviewer, annotating."""

import json

import pytest

pytest.importorskip("flask")

from pbank.annotations import current_annotations, reviewer_slug
from pbank.records import append_jsonl, prompts_sha256, record_to_dict, write_jsonl
from pbank.t0 import build_extraction_bank, build_verification_grid
from review import REVIEWER_COOKIE, create_app

REVIEWERS = ["Alan McIntyre", "Augusto Nicola"]


@pytest.fixture
def setup(tmp_path):
    """A small bank (10 extraction + 4 verification prompts), one checker file, one baseline run."""
    records = build_extraction_bank()[:10] + build_verification_grid()[:4]
    bank = tmp_path / "data" / "tiny"
    write_jsonl(bank / "prompts.jsonl", (record_to_dict(r) for r in records))
    disagree_uid = records[0].uid

    def problems(r):
        """Two prompts get problems: one ambiguity on the first, an ambiguity and a grammar on the second."""
        if r.uid == records[0].uid:
            return [{"category": "ambiguity", "detail": "The time frame could mean an evaluation period."}]
        if r.uid == records[1].uid:
            return [{"category": "ambiguity", "detail": "Unclear whether payouts are lump sums."},
                    {"category": "grammar", "detail": "Odd unit: 1380 seconds."}]
        return []

    append_jsonl(bank / "checks" / "local__fake.jsonl", [
        {"uid": r.uid, "checker": "local:fake", "agrees": r.uid != disagree_uid,
         "checker_answer": r.correct_label or "none", "generator_answer": r.correct_label or "none",
         "verdict": "problem" if problems(r) else "ok", "problems": problems(r),
         "parsed": {"stated_horizon": r.horizon or "none", "options": [
             {"label": r.labels[0], "payout": "some", "timing": "soon"},
             {"label": r.labels[1].rstrip(")"), "payout": "more", "timing": "later"}],   # "b" written without ")"
             "options_within_horizon": [], "expected_answer": r.correct_label or "none",
             "problems": problems(r), "verdict": "problem" if problems(r) else "ok"},
         "parse_error": None, "request_error": None, "raw_answer": "{}", "thinking": None, "n_generated": 5,
         "hit_token_cap": False}
        for r in records])
    run = tmp_path / "runs" / "fake-model__tiny"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"settings": {"bank_prompts_sha256": prompts_sha256(records)}}))
    wrong = records[10]                                   # a verification prompt, answered wrongly
    append_jsonl(run / "responses.jsonl", [
        {"uid": wrong.uid, "model": "fake", "thinking": "hmm", "answer": f"I choose: {wrong.far_label}",
         "choice_label": wrong.far_label, "parse_status": "ok", "n_choice_mentions": 1, "chosen_option": "far",
         "correct": False, "n_generated": 50, "hit_token_cap": False, "seconds": 1.0}])
    other = tmp_path / "runs" / "other-bank"              # a run of a different bank must not be picked up
    other.mkdir()
    (other / "run.json").write_text(json.dumps({"settings": {"bank_prompts_sha256": "different"}}))
    app = create_app(bank, tmp_path / "runs", REVIEWERS)
    app.config["TESTING"] = True
    return app.test_client(), bank, records, disagree_uid, wrong.uid


def test_index_and_filters(setup):
    client, _, records, disagree_uid, wrong_uid = setup
    page = client.get("/").get_data(as_text=True)
    assert "14 of 14 prompts" in page and "fake-model__tiny" in page and "other-bank" not in page
    page = client.get("/?checker=disagree").get_data(as_text=True)
    assert "1 of 14 prompts" in page and disagree_uid in page
    page = client.get("/?run=incorrect").get_data(as_text=True)
    assert "1 of 14 prompts" in page and wrong_uid in page
    page = client.get("/?split=verification").get_data(as_text=True)
    assert "4 of 14 prompts" in page


def test_item_page_shows_prompt_checker_and_run(setup):
    client, _, records, _, wrong_uid = setup
    page = client.get(f"/item/{wrong_uid}?split=verification").get_data(as_text=True)
    rec = next(r for r in records if r.uid == wrong_uid)
    assert rec.text.splitlines()[0] in page and "incorrect" in page and "1 of 4 in this filter" in page
    assert "Choose your name" in page                     # no reviewer chosen yet: no forms
    assert client.get("/item/nope").status_code == 404


def test_choose_reviewer_and_annotate(setup):
    client, bank, records, _, wrong_uid = setup
    r = client.post("/reviewer", data={"reviewer": "Alan McIntyre", "back": "/"})
    assert REVIEWER_COOKIE in r.headers.get("Set-Cookie", "")
    client.post("/annotate", data={"uid": wrong_uid, "target": "prompt", "verdict": "ok", "note": "fine"})
    client.post("/annotate", data={"uid": wrong_uid, "target": "fake-model__tiny", "verdict": "problem",
                                   "note": "  wrong despite the clear horizon  "})
    client.post("/annotate", data={"uid": wrong_uid, "target": "prompt", "verdict": "unsure", "note": "changed my mind"})
    lines = (bank / "reviews" / f"{reviewer_slug('Alan McIntyre')}.jsonl").read_text().splitlines()
    print("\n".join(lines))
    assert len(lines) == 3                                # append-only: the history is kept
    notes = current_annotations(bank)
    assert notes[(wrong_uid, "prompt")]["Alan McIntyre"]["verdict"] == "unsure"          # latest wins
    assert notes[(wrong_uid, "fake-model__tiny")]["Alan McIntyre"]["note"] == "wrong despite the clear horizon"
    page = client.get(f"/item/{wrong_uid}").get_data(as_text=True)
    assert "changed my mind" in page and "Save as Alan McIntyre" in page
    assert "1 of 14 prompts" in client.get("/?review=marked+problem+by+anyone").get_data(as_text=True)
    assert "13 of 14 prompts" in client.get("/?review=not+reviewed+by+me").get_data(as_text=True)


def test_annotation_requires_a_known_reviewer_and_target(setup):
    client, _, records, _, wrong_uid = setup
    assert client.post("/annotate", data={"uid": wrong_uid, "target": "prompt", "verdict": "ok"}).status_code == 400
    client.post("/reviewer", data={"reviewer": "Augusto Nicola"})
    assert client.post("/annotate", data={"uid": wrong_uid, "target": "no-such-run", "verdict": "ok"}).status_code == 400
    client.post("/reviewer", data={"reviewer": "Mallory"})                       # not on the list: cookie cleared
    assert client.post("/annotate", data={"uid": wrong_uid, "target": "prompt", "verdict": "ok"}).status_code == 400


def test_feedback_page_groups_problems_by_category(setup):
    client, _, records, _, _ = setup
    page = client.get("/feedback").get_data(as_text=True)
    # ambiguity: 2 prompts (2 reports); grammar: 1 prompt. Ambiguity is listed first (more prompts).
    assert "2 prompts, 2 reports" in page and "1 prompts, 1 reports" in page
    assert page.index(">ambiguity<") < page.index(">grammar<")
    assert "The time frame could mean an evaluation period." in page and records[1].uid in page
    assert "disagrees on" in page and "checked 14 of 14 prompts" in page


def test_problem_filter_search_and_feedback_column(setup):
    client, _, records, _, _ = setup
    page = client.get("/?problem=ambiguity").get_data(as_text=True)
    assert "2 of 14 prompts" in page and "Unclear whether payouts are lump sums." in page   # shown in the list
    assert "1 of 14 prompts" in client.get("/?problem=grammar").get_data(as_text=True)
    assert "2 of 14 prompts" in client.get("/?problem=any").get_data(as_text=True)
    assert "1 of 14 prompts" in client.get("/?q=1380+seconds").get_data(as_text=True)     # searches checker text


def test_item_page_sets_checker_reading_beside_generator(setup):
    client, _, records, _, _ = setup
    rec = records[1]
    page = client.get(f"/item/{rec.uid}").get_data(as_text=True)
    assert "How the checker read the prompt" in page and "Odd unit: 1380 seconds." in page
    # The checker wrote the second label without ")"; it still matches the generator's option.
    assert "no such label" not in page
