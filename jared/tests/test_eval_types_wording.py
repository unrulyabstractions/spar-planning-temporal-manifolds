"""v2 wording types: shared items across the ladder, pairing with the overnight types, the clear wording."""

import math

import pytest

from spar_horizon.eval_types import build
from spar_horizon.eval_types.wording import CASH, END_CONSTRAINT, RECEIVE_TASK

LADDER = ("t0_w0", "t0_w1_cash", "t0_w3_clear")
PAIRED = {"roi_unequal_clear": "roi_unequal", "system_prompt_clear": "system_prompt",
          "sycophancy_lean_clear": "sycophancy_lean", "horizon_revised_clear": "horizon_revised"}


def text(record):
    p = record["prompt"]
    return p if isinstance(p, str) else "\n".join(m["content"] for m in p)


def decision(record):
    return record["variant"], record["regime"], record["labels"], record["options"], record["truth_labels"]


def test_ladder_shares_items_and_orders():
    banks = [build(t) for t in LADDER]
    assert len({len(b) for b in banks}) == 1
    for rows in zip(*banks):
        assert len({repr(decision(r)) for r in rows}) == 1
        assert len({text(r) for r in rows}) == 3


def test_ladder_wording():
    w0, w1, w3 = (build(t)[0] for t in LADDER)
    assert CASH not in text(w0) and CASH in text(w1) and CASH in text(w3)
    assert RECEIVE_TASK in text(w3) and RECEIVE_TASK not in text(w1)
    assert END_CONSTRAINT.format(horizon=w3["horizon_text"]) in text(w3)


def test_ladder_plan_and_implied_rate():
    rows = [r for r in build("t0_w0") if r["variant"] == "base"]
    assert [sum(r["regime"] == g for r in rows) for g in ("near", "both", "none")] == [12, 24, 4]
    r = rows[0]
    near = next(o for o in r["options"] if o["kind"] == "near")
    far = next(o for o in r["options"] if o["kind"] == "far")
    expected = (far["value"] / near["value"]) ** (1 / (far["delay_years"] - near["delay_years"])) - 1
    assert r["implied_rate"] == pytest.approx(expected)


@pytest.mark.parametrize("clear,original", PAIRED.items())
def test_clear_types_pair_with_the_overnight_types(clear, original):
    new, old = build(clear), build(original)
    assert [decision(r) for r in new] == [decision(r) for r in old]
    for n, o in zip(new, old):
        if n["variant"] == "twin":
            assert n["prompt"] == o["prompt"]          # same reference prompt: cached generations reused
        else:
            assert CASH in text(n) and n["prompt"] != o["prompt"]


def test_horizon_revised_clear_keeps_stale_then_update():
    r = build("horizon_revised_clear")[0]
    t = text(r)
    assert END_CONSTRAINT.format(horizon=r["stale_horizon_text"]) in t
    assert t.index("CONSTRAINT:") < t.index(f"The time horizon is now {r['horizon_text']}.")


def test_free_form_anchor_hides_the_horizon():
    rows = build("free_form_anchor")
    assert len(rows) == 21 and {r["answer_kind"] for r in rows} == {"duration"}
    for r in rows:
        assert r["anchor_text"] in r["prompt"]
        assert "time horizon" not in r["prompt"]
        assert math.isclose(r["truth_term_years"], r["horizon"])
    assert len({r["anchor_id"] for r in rows}) == 7
