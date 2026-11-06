"""The k-option choice banks (18_spec_multichoice.md, 26_spec_overnight_choice_scan.md §6)."""

import hashlib
import json
import math
from collections import Counter, defaultdict

import pytest

from spar_horizon import full_bank as F

TASK_ALAN = "TASK: You, the head of the household, are tasked to choose the best investment:"


@pytest.fixture(scope="module", params=[4, 8])
def bank(request):
    return request.param, F.build_choice_prompts(request.param, False, 0, 2, None)


def bases(records):
    out = defaultdict(list)
    for r in records:
        out[r["pair_id"]].append(r)
    return out


def test_durations_kept_have_both_regimes_and_regimes_are_equal_per_duration(bank):
    _, records = bank
    kept = {str(h) for h in F.HORIZONS if {"near", "both"} <= set(F.feasible_regimes(h.log10_years))}
    assert {r["duration_id"] for r in records} == kept
    assert "30 seconds" not in kept and "108 seconds" not in kept
    per = defaultdict(Counter)
    for base in bases(records).values():
        per[base[0]["duration_id"]][base[0]["regime"]] += 1
    assert all(c["near"] == c["both"] == 1 for c in per.values())


def test_every_regime_has_both_halves_and_no_none(bank):
    _, records = bank
    assert {r["regime"] for r in records} == {"near", "both"}


def test_prefixes_are_balanced_over_regime_and_duration(bank):
    _, records = bank
    pairs = list(dict.fromkeys(r["pair_id"] for r in records))
    regime = {r["pair_id"]: r["regime"] for r in records}
    for n in (8, 20, 40):
        prefix = pairs[:n]
        assert abs(Counter(regime[p] for p in prefix)["near"] - n / 2) <= 1
    years = {r["pair_id"]: r["horizon"] for r in records}
    first = sorted(years[p] for p in pairs[:20])
    assert len({r["duration_id"] for r in records if r["pair_id"] in pairs[:20]}) >= 10   # near and both share a duration
    assert first[0] < 0.01 and first[-1] > 50 and first[len(first) // 2] > 0.01    # the whole range, not one end


def test_truth_positions_balanced_within_one_per_regime(bank):
    k, records = bank
    for regime in ("near", "both"):
        counts = Counter(base[0]["labels"].index(base[0]["truth_label"])
                         for base in bases(records).values() if base[0]["regime"] == regime)
        assert set(counts) == set(range(k)) and max(counts.values()) - min(counts.values()) <= 1


def test_delays_are_clear_of_every_horizon_rendering_and_regime_matches(bank):
    _, records = bank
    for r in records:
        near, far, h = r["near_delay_years"], r["far_delay_years"], r["horizon"]
        assert near < far
        for d in (near, far):
            assert F.DELAY_LIMITS[0] <= d <= F.DELAY_LIMITS[1] * 1.0001
            assert all(abs(math.log10(d / x["horizon"])) >= F.MIN_DISTANCE - 1e-9 for x in bases(records)[r["pair_id"]])
        assert r["regime"] == ("near" if h < far else "both") and h > near


def test_payouts_far_above_near_and_distinct_within_half(bank):
    k, records = bank
    for r in records:
        by_half = defaultdict(list)
        for o in r["options"]:
            by_half[o["half"]].append(o["payout"])
            assert o["payout"] == int(o["payout"]) and o["payout"] * 100 == r["principal"] * (100 + o["return_pct"])
        assert len(by_half["near"]) == len(by_half["far"]) == k // 2
        assert min(by_half["far"]) > max(by_half["near"])
        assert all(len(set(v)) == len(v) for v in by_half.values())
        assert {o["return_pct"] for o in r["options"] if o["half"] == "near"} <= set(F.NEAR_RETURNS)
        assert {o["return_pct"] for o in r["options"] if o["half"] == "far"} <= set(F.FAR_RETURNS)
        assert r["principal"] in F.SHORT_REWARDS


def test_truth_matches_the_rule(bank):
    _, records = bank
    for r in records:
        on_time = [o for o in r["options"] if o["delay_years"] <= r["horizon"]]
        best = max(on_time, key=lambda o: o["payout"])
        assert best["label"] == r["truth_label"] and best["half"] == r["truth"] == F.TRUTH[r["regime"]]
        second = sorted((o["payout"] for o in on_time if o["half"] == best["half"]), reverse=True)[1]
        assert r["gap_in_half"] == pytest.approx(best["payout"] / second)
        assert r["near_labels"] + r["far_labels"] != [] and set(r["near_labels"]) | set(r["far_labels"]) == set(r["labels"])


def test_renderings_of_a_base_differ_only_in_the_horizon_phrase(bank):
    _, records = bank
    for base in bases(records).values():
        masked = {"\n".join(line.replace(r["horizon_text"], "<h>") if line.startswith("CONSTRAINT:") else line
                            for line in r["prompt"].split("\n")) for r in base}
        assert len(masked) == 1
        assert len({json.dumps(r["options"]) for r in base}) == 1
        assert len({r["horizon_text"] for r in base}) == len(base)


def test_implicit_task_line_is_alans_and_roi_twins_differ_only_in_it(bank):
    k, records = bank
    explicit = F.build_choice_prompts(k, True, 0, 2, None)
    assert len(explicit) == len(records)
    for a, b in zip(records, explicit):
        assert a["prompt"].split("\n")[1] == TASK_ALAN
        assert (a["sample_uid"], a["pair_id"], a["options"]) == (b["sample_uid"], b["pair_id"], b["options"])
        assert (a["roi_explicit"], b["roi_explicit"]) == (False, True)
        la, lb = a["prompt"].split("\n"), b["prompt"].split("\n")
        assert [i for i, (x, y) in enumerate(zip(la, lb)) if x != y] == [1] and len(la) == len(lb)
        assert lb[1] == (f"TASK: You, the head of the household, have {a['principal']:,} dollars to invest "
                         "and are tasked to choose the best investment:")


def test_golden_prompt_for_four_options():
    r = F.build_choice_prompts(4, False, 0, 2, None)[0]
    assert r["prompt"] == (
        "SITUATION: Plan for the future of the household.\n"
        "TASK: You, the head of the household, are tasked to choose the best investment:\n"
        "a) 600,000 dollars in 7 minutes.\n"
        "b) 200,000 dollars in 7 minutes.\n"
        "c) 104,000 dollars in 126 seconds.\n"
        "d) 112,000 dollars in 126 seconds.\n"
        "OBJECTIVE: Think deeply about which option is preferable.\n"
        "CONSTRAINT: You must select the option that provides the greatest benefit for this time horizon: 4 minutes.\n"
        "ACTION: Select one of the four options. Provide reasoning.\n"
        "FORMAT: I choose: <a), b), c) or d)>. My reasoning: <1-3 sentences>")
    assert (r["regime"], r["truth"], r["truth_label"]) == ("near", "near", "d")


def test_eight_option_format_line():
    r = F.build_choice_prompts(8, False, 0, 2, None)[0]
    assert r["prompt"].endswith("FORMAT: I choose: <a), b), c), d), e), f), g) or h)>. My reasoning: <1-3 sentences>")
    assert "Select one of the eight options." in r["prompt"]


def test_counts_and_limit():
    for k in (4, 8):
        records = F.build_choice_prompts(k, False, 0, 2, None)
        assert 230 <= len(records) <= 270 and len(bases(records)) == 86
    small = F.build_choice_prompts(4, False, 0, 2, 5)
    full = F.build_choice_prompts(4, False, 0, 2, None)
    assert 5 <= len(small) <= 8 and small == full[:len(small)]


def test_variant_reverses_order_and_relabels():
    r = F.build_choice_prompts(4, False, 0, 2, None)[0]
    assert F.choice_variant(r) == r["prompt"]
    swapped = F.choice_variant(r, swap=True).split("\n")[2:6]
    assert [l[3:] for l in swapped] == [l[3:] for l in r["prompt"].split("\n")[2:6]][::-1]
    relabeled = F.choice_variant(r, labels=("1", "2")).split("\n")
    assert [l[:2] for l in relabeled[2:6]] == ["1)", "2)", "3)", "4)"] and relabeled[-1].startswith("FORMAT: I choose: <1), 2), 3) or 4)>")


def test_the_other_banks_are_unchanged():
    def digest(records):
        return hashlib.sha1(json.dumps([r["prompt"] for r in records[:50]]).encode()).hexdigest()[:12]
    # hashes of the first 50 prompts, computed before the choice bank existed
    pinned = ("c2ca990efd7c", "df79b82b912b", "0aee688bfd52", "2dfbc7a00453")
    F.build_choice_prompts(4, False, 0, 2, None)
    F.build_choice_prompts(8, True, 0, 2, None)
    now = (digest(F.build_full_prompts()), digest(F.build_units_prompts()), digest(F.build_crossed_prompts()),
           digest(F.build_fill_prompts()))
    assert now == pinned
