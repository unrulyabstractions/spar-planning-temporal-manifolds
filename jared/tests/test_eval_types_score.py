"""Eval-type scorer on hand-written rows: label and term parsing, accuracy, order agreement."""

import pytest

from pipeline.s3_analysis.eval_types_score import parse_label, parse_term_years, score_type
from spar_horizon.eval_types import build


def row(response, prefix="I choose:", labels=("a", "b")):
    return {"response_text": response, "choice_prefix": prefix, "labels": list(labels)}


@pytest.mark.parametrize("response,expected", [
    ("</think>\n\nI choose: a). My reasoning: ...", "a"),
    ("</think>\n\n**I choose: b)**", "b"),
    ("</think>\n\nI choose: **B)** because", "b"),
    ("</think>\n\nI choose: <a)>", "a"),
    ("</think>\n\nLet me think. I choose: c)", None),          # not a label of this record
    ("</think>\n\nThe answer is a).", None),                   # prefix missing
    ("thinking about I choose: b) ...\n</think>\n\nI choose: a)", "a"),   # only text after the think block
    ("</think>\n\nI choose: about this", None),
])
def test_parse_label(response, expected):
    assert parse_label(row(response)) == expected


def test_parse_json_label():
    r = row('</think>\n\n```json\n{\n  "choice": "b",\n  "reasoning": "x"\n}\n```', prefix='"choice":')
    assert parse_label(r) == "b"
    assert parse_label(row('</think>\n\n{"reasoning": "x"}', prefix='"choice":')) is None


def test_parse_term():
    r = row("</think>\n\nI choose a term of: 18 months. My reasoning", prefix="I choose a term of:", labels=())
    assert parse_term_years(r) == pytest.approx(1.5)
    assert parse_term_years(row("</think>\n\nI choose a term of: forever", prefix="I choose a term of:")) is None


def answered(rec, label):
    return {**rec, "answer": f"{label})", "forced": False, "think_len": 100, "think_text": rec["horizon_text"] or "",
            "response_text": f"</think>\n\nI choose: {label}). My reasoning: x", "label_logits": [1.0, 0.0]}


def test_score_type_on_always_truth_and_always_a():
    recs = [r for r in build("distractor_temporal") if r["choice_prefix"] == "I choose:"]
    perfect = score_type([answered(r, (r["truth_labels"] or ["a"])[0]) for r in recs])
    assert perfect["accuracy"]["p"] == 1.0 and perfect["twin_accuracy"]["p"] == 1.0
    assert perfect["order_agreement"]["p"] == 1.0 and perfect["parsed"]["p"] == 1.0
    assert perfect["near_rate_by_regime"]["near"]["p"] == 1.0 and perfect["near_rate_by_regime"]["both"]["p"] == 0.0
    always_a = score_type([answered(r, "a") for r in recs])
    assert always_a["order_agreement"]["p"] == 0.0     # "a" names the other option after the swap
