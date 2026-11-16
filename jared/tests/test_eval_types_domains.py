"""Type-specific tests for the domains eval types (spar_horizon/eval_types/domains.py)."""

import math
import re

import pytest

from spar_horizon.eval_types import build
from spar_horizon.eval_types.domains import CONCRETE_ANCHORS, VAGUE_ANCHORS
from spar_horizon.full_bank import regime

CLEARANCE_DECADES = 0.25 - 1e-9


def bases(type_id):
    return [r for r in build(type_id) if r["variant"] == "base"]


def delays(rec):
    near = next(o for o in rec["options"] if o["kind"] == "near")
    far = next(o for o in rec["options"] if o["kind"] == "far")
    return near, far


@pytest.mark.parametrize("type_id", ["implicit_concrete", "implicit_vague"])
def test_every_reading_gives_the_regime_with_clearance(type_id):
    for rec in bases(type_id):
        near, far = delays(rec)
        lo, hi = rec["anchor_lo_years"], rec["anchor_hi_years"]
        assert lo <= rec["horizon"] <= hi
        for reading in (lo, math.sqrt(lo * hi), hi):
            assert regime(reading, near["delay_years"], far["delay_years"]) == rec["regime"], rec["sample_uid"]
        for o in (near, far):
            gap = min(abs(math.log10(x) - math.log10(o["delay_years"])) for x in (lo, hi))
            assert gap >= CLEARANCE_DECADES, rec["sample_uid"]


@pytest.mark.parametrize("type_id", ["implicit_concrete", "implicit_vague"])
def test_horizon_is_not_stated_in_the_constraint(type_id):
    for rec in build(type_id):
        constraint = next(line for line in rec["prompt"].splitlines() if line.startswith("CONSTRAINT:"))
        if rec["variant"] == "twin":
            assert f"time horizon: {rec['horizon_text']}." in constraint
        else:
            assert "time horizon" not in constraint
            assert rec["anchor_text"] in constraint


def test_vague_phrases_have_no_digits():
    assert all(not re.search(r"\d", a.phrase) for a in VAGUE_ANCHORS)
    for rec in bases("implicit_vague"):
        constraint = next(line for line in rec["prompt"].splitlines() if line.startswith("CONSTRAINT:"))
        assert not re.search(r"\d", constraint)


def test_concrete_anchors_and_vague_ranges_are_well_formed():
    for a in CONCRETE_ANCHORS:
        assert a.horizon_years == pytest.approx(a.horizon.years)
        assert a.lo_years == pytest.approx(0.9 * a.horizon_years) and a.hi_years == pytest.approx(1.1 * a.horizon_years)
    for a in VAGUE_ANCHORS:
        assert a.lo_years <= a.horizon.years <= a.hi_years


@pytest.mark.parametrize("type_id", ["implicit_concrete", "implicit_vague"])
def test_anchors_vary_and_plan_is_filled(type_id):
    recs = bases(type_id)
    assert len(recs) == 20
    assert len({r["anchor_id"] for r in recs}) >= 4


def test_city_no_values_truth_only_in_near():
    for rec in bases("city_no_values"):
        assert all(o["value"] is None for o in rec["options"])
        assert rec["truth_rule"] == "within_horizon_near_only"
        assert len(rec["truth_labels"]) == (1 if rec["regime"] == "near" else 0)


def parse_city_line(option):
    cost, users = re.search(r"costs ([\d,]+) dollars, .* serves ([\d,]+) ", option["text"]).groups()
    return int(cost.replace(",", "")), int(users.replace(",", ""))


def test_city_values_far_option_dominates_in_both():
    both = [r for r in bases("city_values") if r["regime"] == "both"]
    assert both
    for rec in both:
        near, far = delays(rec)
        (cost_n, users_n), (cost_f, users_f) = parse_city_line(near), parse_city_line(far)
        h = rec["horizon"]
        assert users_f > users_n
        assert users_f * (h - far["delay_years"]) > users_n * (h - near["delay_years"])
        assert users_f / cost_f > users_n / cost_n
        assert rec["truth"] == "far"


def test_city_values_far_option_delivers_nothing_in_near():
    near_recs = [r for r in bases("city_values") if r["regime"] == "near"]
    assert near_recs
    for rec in near_recs:
        near, far = delays(rec)
        assert far["delay_years"] > rec["horizon"] > near["delay_years"]
        assert rec["truth"] == "near"


def test_public_health_values_and_unit():
    for rec in bases("public_health"):
        assert all("hospital admissions prevented" in o["text"] for o in rec["options"])
        assert all(50 <= o["value"] <= 5_000 for o in rec["options"])


def test_coding_agent_delays_and_values():
    for rec in build("coding_agent"):
        if rec["variant"] == "twin":
            continue
        assert all(o["delay_years"] <= 2 for o in rec["options"]), rec["sample_uid"]
        assert all(10 <= o["value"] <= 2_000 for o in rec["options"])
        assert all("engineer-hours saved" in o["text"] for o in rec["options"])


@pytest.mark.parametrize("type_id", ["implicit_concrete", "implicit_vague", "city_no_values", "city_values",
                                     "public_health", "coding_agent"])
def test_plan_counts(type_id):
    counts = {}
    for rec in bases(type_id):
        counts[rec["regime"]] = counts.get(rec["regime"], 0) + 1
    assert counts == {"near": 8, "both": 8, "none": 4}
