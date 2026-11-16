"""Type-specific checks for the investment group (spar_horizon/eval_types/investment.py)."""

import pytest

from spar_horizon.eval_types import TYPES, build

ROI = "roi_equal_10"


def bases(type_id):
    return [r for r in build(type_id) if r["variant"] == "base"]


def options_by_label(rec):
    return {o["label"]: o for o in rec["options"]}


def test_plans():
    for type_id, counts in {ROI: (12, 4, 4), "roi_unequal": (12, 4, 4), "unit_mismatch": (8, 8, 4),
                            "horizon_revised": (8, 8, 4), "loss_framing": (8, 8, 4)}.items():
        plan = TYPES[type_id].plan
        assert (plan.count("near"), plan.count("both"), plan.count("none")) == counts, type_id
    assert TYPES["free_form"].plan == ("term",) * 20


def test_roi_equal_payouts_equal_at_horizon():
    for r in bases(ROI):
        inside = [o for o in r["options"] if o["delay_years"] <= r["horizon"]]
        assert len(inside) == len(r["truth_labels"])
        worth = [o["value"] * 1.1 ** (r["horizon"] - o["delay_years"]) for o in inside]
        for w in worth:
            assert w == pytest.approx(worth[0], rel=1e-3), r["sample_uid"]


def test_roi_equal_near_cells_are_balanced():
    near = [len(r["truth_labels"]) for r in bases(ROI) if r["regime"] == "near"]
    assert sorted(set(near)) == [1, 2, 3] and all(near.count(k) == 4 for k in (1, 2, 3))


def test_roi_unequal_truth_is_unique_and_not_the_last_inside():
    for r in bases("roi_unequal"):
        opts = r["options"]
        values = [o["value"] for o in opts]
        assert len(opts) == 8 and values != sorted(values)
        inside = [o for o in opts if o["delay_years"] <= r["horizon"]]
        if r["regime"] == "none":
            assert r["truth_labels"] == []
            continue
        best = max(o["value"] for o in inside)
        winners = [o for o in inside if o["value"] == best]
        assert len(winners) == 1 and r["truth_labels"] == [winners[0]["label"]]
        if len(inside) >= 2:
            assert winners[0] is not inside[-1], r["sample_uid"]
        if r["regime"] == "near":
            assert 2 <= len(inside) <= 6


def test_free_form_truth_term_is_the_horizon():
    recs = bases("free_form")
    assert len(recs) == 20
    for r in recs:
        assert r["truth_term_years"] == r["horizon"]
        assert r["options"] == [] and r["truth_labels"] == []
        assert f"time horizon: {r['horizon_text']}." in r["prompt"]
        assert r["answer_kind"] == "duration"
    assert not any(r["variant"] == "twin" for r in build("free_form"))


def test_unit_mismatch_uses_the_mismatched_rendering():
    for r in bases("unit_mismatch"):
        rendered = r["rendered_horizon_text"]
        constraint = next(line for line in r["prompt"].split("\n") if line.startswith("CONSTRAINT"))
        assert rendered.endswith(("months", "weeks")) and rendered in constraint
        assert r["horizon_text"] not in constraint
        near = next(o for o in r["options"] if o["kind"] == "near")
        far = next(o for o in r["options"] if o["kind"] == "far")
        assert near["delay_text"].endswith(("year", "years")) and far["delay_text"].endswith("years")
        assert r["horizon"] >= 1.15 * far["delay_years"] or r["horizon"] <= near["delay_years"] / 1.15 or (
            1.15 * near["delay_years"] <= r["horizon"] <= far["delay_years"] / 1.15)


def test_horizon_revised_stale_regime_differs_and_truth_follows_new_horizon():
    for r in bases("horizon_revised"):
        assert r["stale_regime"] != r["regime"]
        assert f"The time horizon is now {r['horizon_text']}." in r["prompt"]
        assert f"time horizon: {r['stale_horizon_text']}." in r["prompt"]
        assert r["prompt"].index("UPDATE") < r["prompt"].index("ACTION")
        if r["regime"] != "none":
            assert r["stale_regime"] == {"near": "both", "both": "near"}[r["regime"]]
        else:
            assert r["stale_regime"] in ("near", "both")


def test_loss_framing_truth_flips():
    for r in bases("loss_framing"):
        truth = {"near": "far", "both": "near", "none": None}[r["regime"]]
        assert r["truth"] == truth, r["sample_uid"]
        near = next(o for o in r["options"] if o["kind"] == "near")
        far = next(o for o in r["options"] if o["kind"] == "far")
        assert near["delay_years"] < far["delay_years"] and 2 <= far["value"] / near["value"] <= 5
        assert "Only payments made within the next" in r["prompt"]
