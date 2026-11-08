"""The implied bank (29_spec_implied_horizon_side.md §3): sizes, determinism, twin pairing, the interval/delay
margin that fixes regime and truth, number-free family B, displaced duration in family C, wiring."""

import math
import re
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import pytest

from pipeline.s1_data_vectors.experiments import common
from pipeline.s3_analysis.by_datatype.written_unit import durations
from spar_horizon import implied_bank as ib
from spar_horizon.banks import build_records, variant_builder
from spar_horizon.config import Settings
from spar_horizon.full_bank import TRUTH, regime
from spar_horizon.horizons import DAYS_PER_YEAR

CONSTRAINT = re.compile(r"^CONSTRAINT: .*$", re.MULTILINE)


@pytest.fixture(scope="module")
def records():
    return ib.build_implied_prompts()


def of(records, **match):
    return [r for r in records if all(r[k] == v for k, v in match.items())]


def test_sizes_per_module_and_rendering(records):
    assert len(records) == 288
    assert Counter((r["module"], r["rendering"]) for r in records) == {
        ("M1", "implied"): 64, ("M1", "explicit"): 64, ("M1", "null"): 16,
        ("M2", "implied"): 32, ("M2", "explicit"): 32, ("M2", "null"): 8,
        ("M3", "implied"): 24, ("M3", "explicit"): 24, ("M4", "vague"): 24}
    modules = [r["module"] for r in records]
    assert modules == sorted(modules)   # run order: M1, M2, M3, M4


def test_each_anchor_has_two_near_two_both_and_one_none_base_and_one_null(records):
    for anchor in ib.ANCHORS:
        mine = of(records, anchor_id=anchor.anchor_id)
        if anchor.family == "D":
            assert len(mine) == 3
            continue
        implied = of(mine, rendering="implied")
        assert Counter(r["regime"] for r in implied) == {"near": 2, "both": 2, "none": 1}
        assert len(of(mine, rendering="null")) == 1


def test_deterministic_in_seed_and_seed_changes_delays(records):
    again = ib.build_implied_prompts()
    assert [r["prompt"] for r in again] == [r["prompt"] for r in records]
    other = ib.build_implied_prompts(seed=1)
    assert [r["prompt"] for r in other] != [r["prompt"] for r in records]


def test_a_twin_differs_from_its_implied_prompt_in_the_constraint_phrase_only(records):
    for implied in of(records, rendering="implied"):
        twin = next(r for r in records if r["twin_id"] == implied["twin_id"] and r["rendering"] == "explicit")
        assert implied["prompt"].replace(implied["horizon_text"], twin["horizon_text"]) == twin["prompt"]
        for key in ("rewards", "delays", "short_first", "near_label", "context", "anchor_id", "regime", "truth",
                    "near_delay_years", "far_delay_years"):
            assert implied[key] == twin[key], key
        assert twin["horizon_text"] == twin["explicit_text"] == implied["explicit_text"]


def test_the_null_twin_is_the_implied_prompt_without_its_constraint_line(records):
    nulls = of(records, rendering="null")
    assert len(nulls) == 24
    for null in nulls:
        implied = next(r for r in records if r["twin_id"] == null["twin_id"] and r["rendering"] == "implied")
        assert "CONSTRAINT" not in null["prompt"] and "time horizon" not in null["prompt"]
        assert CONSTRAINT.sub("", implied["prompt"]).replace("\n\n", "\n") == null["prompt"]
        assert null["horizon"] is None and null["regime"] == "null" and null["truth"] is None


def test_every_reading_inside_the_interval_gives_the_same_regime_and_truth(records):
    for r in records:
        if r["rendering"] not in ("implied", "explicit"):
            continue
        lo, hi = math.log10(r["anchor_lo"]), math.log10(r["anchor_hi"])
        near, far = math.log10(r["near_delay_years"]), math.log10(r["far_delay_years"])
        for log_h in np.linspace(lo, hi, 25):
            assert regime(10 ** log_h, r["near_delay_years"], r["far_delay_years"]) == r["regime"]
        assert regime(10 ** math.log10(r["horizon"]), r["near_delay_years"], r["far_delay_years"]) == r["regime"]
        assert r["truth"] == TRUTH.get(r["regime"])
        margin = {"near": min(lo - near, far - hi), "both": lo - far, "none": near - hi}[r["regime"]]
        assert margin >= 0.3 - 1e-6, (r["sample_uid"], margin)
        assert near < far


def test_margins_exceed_the_crossed_banks_and_the_explicit_twin_names_the_log_midpoint(records):
    assert ib.IMPLIED_MARGIN[0] == 0.3
    for r in of(records, rendering="explicit"):
        anchor = ib.ANCHOR_BY_ID[r["anchor_id"]]
        [(years, _)] = durations(r["horizon_text"])
        assert abs(math.log10(years) - anchor.log_mid) <= 0.05, r["horizon_text"]


def test_family_b_has_no_number_in_the_context_or_phrase_and_flags_only_the_us_conventions(records):
    for anchor in ib.family_anchors("B"):
        assert anchor.context == "" and not re.search(r"\d", anchor.phrase)
    assert {a.anchor_id for a in ib.ANCHORS if a.us_convention} == {"B5", "B6", "B7"}
    assert all(r["us_convention"] == ib.ANCHOR_BY_ID[r["anchor_id"]].us_convention for r in records)


def test_family_c_states_its_duration_in_the_context_and_not_in_the_phrase():
    for anchor in ib.family_anchors("C"):
        [(years, _)] = durations(anchor.context)
        assert math.isclose((anchor.lo + anchor.hi) / 2, years, rel_tol=1e-9)
        assert not re.search(r"\d", anchor.phrase)
        assert anchor.context in ib.with_context("SITUATION: Plan for the future of the household.", anchor.context)


@pytest.mark.parametrize("anchor_id,days", [("A1", 7.25 / 24), ("A2", 4 + 10 / 24), ("A3", 149), ("A4", 184)])
def test_family_a_interval_is_the_span_between_the_stated_dates(anchor_id, days):
    anchor = ib.ANCHOR_BY_ID[anchor_id]
    assert math.isclose((anchor.lo + anchor.hi) / 2 * DAYS_PER_YEAR, days, rel_tol=1e-6)
    assert math.isclose(anchor.hi / anchor.lo, 1.1 / 0.9)


def test_family_a_prompt_states_now_and_the_end_point():
    a3 = ib.ANCHOR_BY_ID["A3"]
    assert a3.context == "Today is 1 February 2026." and a3.phrase.endswith("closes on 30 June 2026")
    assert ib.ANCHOR_BY_ID["A1"].context == "It is 9:15 am." and ib.ANCHOR_BY_ID["A1"].phrase == "until 4:30 pm today"
    assert ib.span_years(datetime(2026, 2, 14), datetime(2028, 11, 5)) == pytest.approx(2.73, abs=0.01)


def test_vague_family_has_no_truth_and_no_horizon(records):
    vague = of(records, rendering="vague")
    assert len(vague) == 24 and {r["family"] for r in vague} == {"D"}
    assert all(r["regime"] == "vague" and r["truth"] is None and r["horizon"] is None for r in vague)
    assert all("CONSTRAINT" in r["prompt"] and r["horizon_text"] in r["prompt"] for r in vague)


def test_every_prefix_of_m1_covers_both_regimes_and_both_families_early(records):
    m1 = [r for r in records if r["module"] == "M1" and r["rendering"] == "implied"]
    first = m1[:8]
    assert {r["regime"] for r in first} == {"near", "both"} and {r["family"] for r in first} == {"A", "C"}
    assert len({r["anchor_id"] for r in m1[:16]}) == 16


def test_variants_rebuild_the_prompt_swap_the_options_and_keep_the_context(records):
    rec = next(r for r in records if r["family"] == "C" and r["rendering"] == "implied")
    assert ib.implied_variant(rec) == rec["prompt"]
    swapped = ib.implied_variant(rec, swap=True)
    assert swapped != rec["prompt"] and rec["context"] in swapped and rec["horizon_text"] in swapped
    relabeled = ib.implied_variant(rec, labels=("1", "2"))
    assert "1) " in relabeled and "a) " not in relabeled and rec["context"] in relabeled


def test_limit_keeps_whole_groups_and_the_bank_is_wired_into_settings_and_the_cli(monkeypatch):
    cfg = Settings(bank="implied", prefill="", choice_prefix="I choose:", limit=30)
    kept = build_records(cfg)
    assert 20 <= len(kept) <= 45
    for twin_id in {r["twin_id"] for r in kept}:
        group = [r for r in kept if r["twin_id"] == twin_id]
        assert {"implied", "explicit"} <= {r["rendering"] for r in group} or group[0]["rendering"] == "vague"
    assert variant_builder(cfg) is ib.implied_variant
    assert cfg.tag == "Qwen3-8B_think-on_implied"
    monkeypatch.setattr(sys, "argv", ["x", "--bank", "implied"])
    _, parsed = common.parse_args("t")
    assert (parsed.bank, parsed.prefill, parsed.choice_prefix) == ("implied", "", "I choose:")
