"""The full bank: a faithful copy of Alan's generator, plus our record columns."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from spar_horizon import full_bank as F
from spar_horizon.horizons import STANDARD_HORIZONS, Horizon
from spar_horizon.prompts import build_prompts

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "alan_prompts.json").read_text())


@pytest.mark.parametrize("case", list(FIXTURE))
def test_generator_matches_alans_output(case):
    """Texts and ids equal what `../alan/ptm/prompts.py` produced for the same
    config (fixture dumped 2026-10-02 from his snapshot)."""
    cfg = F.DatasetConfig(horizons=STANDARD_HORIZONS, **FIXTURE[case]["config"])
    ours = F.generate(cfg)
    theirs = FIXTURE[case]["samples"]
    assert len(ours) == len(theirs)
    for s, t in zip(ours, theirs):
        assert (s.sample_uid, s.text, s.horizon_text, s.short_first, s.pair_id, s.phrasing_id) == \
            (t["sample_uid"], t["text"], t["horizon_text"], t["short_first"], t["pair_id"], t["phrasing_id"])


def test_our_horizon_grid_is_48_log_spaced_in_natural_units():
    hs = F.HORIZONS
    assert len(hs) == 48 and len({str(h) for h in hs}) == 48
    assert str(hs[0]) == "30 seconds" and str(hs[-1]) == "500 years"
    steps = [b.years / a.years for a, b in zip(hs, hs[1:])]
    assert all(1.3 < s < 1.7 for s in steps)            # target ratio 1.53, distorted only by rounding
    units = [F.NATURAL_UNITS.index(h.unit) for h in hs]
    assert units == sorted(units, reverse=True)          # units only grow with the duration
    assert all(h.value == int(h.value) and (h.value >= 2 or h.unit == "seconds") for h in hs)


def test_natural_horizon():
    assert str(F.natural_horizon(Horizon(84, "minutes").years)) == "84 minutes"
    assert str(F.natural_horizon(Horizon(130, "minutes").years)) == "2 hours"
    assert str(F.natural_horizon(1.25)) == "15 months"
    assert str(F.natural_horizon(Horizon(30, "seconds").years)) == "30 seconds"


def test_records_carry_starter_keys_and_analysis_columns():
    recs = F.build_full_prompts(n=200)
    starter_keys = set(build_prompts()[0])
    assert len(recs) == 200
    for r in recs:
        assert starter_keys <= set(r)
        assert r["near_label"] in ("a", "b") and r["regime"] in ("null", "none", "near", "both")
        assert r["near_delay_years"] < r["far_delay_years"]
        assert (r["horizon"] is None) == (r["regime"] == "null") == (r["horizon_text"] is None)
        if r["regime"] == "none":
            assert r["horizon"] < r["near_delay_years"] and r["truth"] is None
        if r["regime"] == "near":
            assert r["near_delay_years"] <= r["horizon"] < r["far_delay_years"] and r["truth"] == "near"
        if r["regime"] == "both":
            assert r["horizon"] >= r["far_delay_years"] and r["truth"] == "far"
        if r["horizon_text"] is not None:
            assert r["horizon_text"] in r["prompt"] and math.isclose(Horizon.parse(r["horizon_text"]).years, r["horizon"])
    assert len({r["prompt"] for r in recs}) == 200


def test_near_label_is_balanced():
    recs = F.build_full_prompts(n=500)
    counts = Counter(r["near_label"] for r in recs)
    assert 0.4 <= counts["a"] / len(recs) <= 0.6
    lines = recs[0]["prompt"].split("\n")
    near_line = lines[2] if recs[0]["near_label"] == "a" else lines[3]
    assert recs[0]["delays"][0] in near_line


def test_limit_keeps_an_even_subset():
    recs = F.build_full_prompts(n=500, limit=6)
    full = F.build_full_prompts(n=500)
    assert len(recs) == 6 and recs[0] == full[0] and recs[1] == full[83]


def test_render_variant_swaps_and_relabels():
    rec = F.build_full_prompts(n=20)[0]
    base = rec["prompt"].split("\n")
    swapped = F.render_variant(rec, swap=True).split("\n")
    assert swapped[2] == "a) " + base[3][3:] and swapped[3] == "b) " + base[2][3:]
    assert swapped[:2] + swapped[4:] == base[:2] + base[4:]
    assert F.render_variant(rec) == rec["prompt"]
    relabeled = F.render_variant(rec, labels=("1", "2")).split("\n")
    assert relabeled[2].startswith("1) ") and relabeled[3].startswith("2) ")
    assert relabeled[-1] == "FORMAT: I choose: <1) or 2)>. My reasoning: <1-3 sentences>"


def test_regime_boundaries():
    assert F.regime(None, 0.5, 10) == "null"
    assert F.regime(0.25, 0.5, 10) == "none"
    assert F.regime(0.5, 0.5, 10) == "near"
    assert F.regime(10, 0.5, 10) == "both"


def test_unit_sets_render_one_duration_in_different_units():
    for duration, rewrites in F.UNIT_SETS.items():
        canonical = Horizon.parse(duration)
        units = {canonical.unit}
        for text in rewrites:
            h = Horizon.parse(text)
            assert abs(math.log10(h.years / canonical.years)) < 0.05, text   # within ~12% ("4 weeks" for "1 month")
            units.add(h.unit)
        assert len(units) == 1 + len(rewrites), duration                   # every rendering is a different unit


def test_render_variant_reproduces_the_prompt_and_swaps_only_the_horizon():
    rec = next(r for r in F.build_full_prompts(n=40) if r["horizon"] is not None)
    assert F.render_variant(rec) == rec["prompt"]
    rewritten = F.render_variant({**rec, "horizon_text": "52 weeks"})
    assert rewritten.replace("52 weeks", rec["horizon_text"]) == rec["prompt"]


# --- the units bank (14_spec_unit_overlap.md) ---

def test_full_bank_is_unchanged_by_the_units_bank():
    """sha1 of the 500 full-bank records at a93acf9, before DatasetConfig.rewrites existed."""
    import hashlib
    records = F.build_full_prompts()
    assert hashlib.sha1(json.dumps(records, sort_keys=True).encode()).hexdigest() == \
        "bb53819cf546b9bcd38f0553d0c30809d86b5dfc"


def test_neighbour_renderings_follow_the_spec():
    assert len(F.UNIT_DURATIONS) == 33
    assert Counter(len(v) for v in F.NEIGHBOUR_REWRITES.values()) == {1: 31, 2: 2}
    assert F.NEIGHBOUR_REWRITES["8 weeks"] == ("2 months", "56 days")
    assert F.NEIGHBOUR_REWRITES["15 months"] == ("65 weeks",)
    for natural, others in F.NEIGHBOUR_REWRITES.items():
        h = Horizon.parse(natural)
        units = F.NATURAL_UNITS
        for text in others:
            r = Horizon.parse(text)
            assert r.value == int(r.value) and 1 <= r.value <= 1000
            assert abs(r.years / h.years - 1) <= 0.10
            assert abs(units.index(r.unit) - units.index(h.unit)) == 1


def test_units_bank_pairs_differ_only_in_the_horizon_phrase():
    records = F.build_units_prompts()
    pairs = {}
    for r in records:
        pairs.setdefault(r["pair_id"], []).append(r)
    assert len(pairs) == F.UNITS_BASE
    per_duration = Counter(p[0]["duration_id"] for p in pairs.values())
    assert set(per_duration) == {str(h) for h in F.UNIT_DURATIONS} and max(per_duration.values()) - min(per_duration.values()) <= 1
    for p in pairs.values():
        assert len(p) == 1 + len(F.NEIGHBOUR_REWRITES[p[0]["duration_id"]])
        assert sum(r["natural"] for r in p) == 1
        assert len({r["regime"] for r in p}) == 1
        assert len({r["rendered_unit"] for r in p}) == len(p)
        # substitute inside the constraint line only: the horizon text can also be a delay ("5 years")
        without_horizon = {"\n".join(line.replace(r["horizon_text"], "<h>") if line.startswith("CONSTRAINT:") else line
                                     for line in r["prompt"].split("\n")) for r in p}
        assert len(without_horizon) == 1


def test_units_bank_limit_keeps_whole_pairs():
    records = F.build_units_prompts(limit=8)
    counts = Counter(r["pair_id"] for r in records)
    assert len(counts) == 4
    assert all(n == 1 + len(F.NEIGHBOUR_REWRITES[r["duration_id"]]) for r in records for n in [counts[r["pair_id"]]])


def test_build_records_routes_the_units_bank_with_full_bank_variants():
    from spar_horizon.banks import build_records, variant_builder
    from spar_horizon.config import Settings
    cfg = Settings(bank="units", prefill="", choice_prefix="I choose:", limit=8)
    records = build_records(cfg)
    assert all(r["pair_id"] and r["duration_id"] and r["rendered_unit"] for r in records)
    assert variant_builder(cfg)(records[0]) == records[0]["prompt"]


def test_crossed_renderings_follow_the_spec():
    assert str(F.crossed_rendering(1.0, "days")) == "365 days"          # 365.25: within 1% of an integer
    assert str(F.crossed_rendering(Horizon(1, "weeks").years, "hours")) == "168 hours"
    assert str(F.crossed_rendering(Horizon(8, "weeks").years, "months")) == "1.8 months"
    assert str(F.crossed_rendering(Horizon(30, "seconds").years, "minutes")) == "0.5 minutes"
    assert F.crossed_rendering(Horizon(30, "seconds").years, "hours") is None   # 0.0083 hours
    assert F.crossed_rendering(500.0, "months") is None                         # 6000 months
    per_duration = [F.crossed_renderings(h.years) for h in F.HORIZONS]
    assert sum(map(len, per_duration)) == 133
    assert Counter(map(len, per_duration)) == {1: 5, 2: 14, 3: 16, 4: 13}
    for h, rs in zip(F.HORIZONS, per_duration):
        assert str(h) in {str(r) for r in rs}                           # the natural rendering is one of them
        assert len({r.unit for r in rs}) == len(rs)
        assert all(0.1 <= r.value <= 1000 and abs(r.years / h.years - 1) <= 0.05 for r in rs)


def test_feasible_regimes_at_the_ends_of_the_grid():
    log = lambda h: Horizon.parse(h).log10_years
    assert F.feasible_regimes(log("30 seconds")) == ["none"]            # a near delay can reach 1 minute
    assert F.feasible_regimes(log("108 seconds")) == ["none", "near"]
    assert F.feasible_regimes(log("1 years")) == ["none", "near", "both"]
    assert F.feasible_regimes(log("500 years")) == ["near", "both"]     # none needs a far delay past 1000 years


def crossed_bases(**kw):
    bases = {}
    for r in F.build_crossed_prompts(**kw):
        bases.setdefault(r["pair_id"], []).append(r)
    return bases


def test_crossed_bank_bases_differ_only_in_the_horizon_phrase():
    bases = crossed_bases()
    assert len(bases) == 48 * F.CROSSED_PER_DURATION
    assert set(Counter(b[0]["duration_id"] for b in bases.values()).values()) == {F.CROSSED_PER_DURATION}
    for b in bases.values():
        renderings = F.crossed_renderings(Horizon.parse(b[0]["duration_id"]).years)
        assert [r["horizon_text"] for r in b] == [str(r) for r in renderings]
        assert sum(r["natural"] for r in b) == 1
        assert len({r["regime"] for r in b}) == 1
        without_horizon = {"\n".join(line.replace(r["horizon_text"], "<h>") if line.startswith("CONSTRAINT:") else line
                                     for line in r["prompt"].split("\n")) for r in b}
        assert len(without_horizon) == 1


def test_crossed_bank_delays_sit_at_the_spec_margins():
    lo, hi = F.DELAY_LIMITS
    rounding = 0.03                                                     # 2 significant figures move a delay < 0.022 decades
    for r in F.build_crossed_prompts():
        h, near, far = (math.log10(y) for y in (r["horizon"], r["near_delay_years"], r["far_delay_years"]))
        assert lo <= r["near_delay_years"] < r["far_delay_years"] <= hi
        assert min(abs(near - h), abs(far - h)) >= F.MIN_DISTANCE
        margin, gap = {"none": (near - h, far - near), "near": (h - near, None), "both": (h - far, far - near)}[r["regime"]]
        assert F.MARGIN[0] - rounding <= margin <= F.MARGIN[1] + rounding
        if gap is not None:
            assert F.GAP[0] - 2 * rounding <= gap <= F.GAP[1] + 2 * rounding
        if r["regime"] == "near":
            assert F.MARGIN[0] - rounding <= far - h <= F.MARGIN[1] + rounding


def test_crossed_bank_gives_every_duration_each_feasible_regime():
    bases = crossed_bases()
    for h in F.HORIZONS:
        regimes = Counter(b[0]["regime"] for b in bases.values() if b[0]["duration_id"] == str(h))
        assert set(regimes) == set(F.feasible_regimes(h.log10_years))
        assert max(regimes.values()) - min(regimes.values()) <= F.CROSSED_PER_DURATION - len(regimes) + 1
    overall = Counter(b[0]["regime"] for b in bases.values())
    assert max(overall.values()) - min(overall.values()) <= 16          # the edges lose regimes, the rest rotate


def test_crossed_bank_delay_renderings_are_crossed_with_the_horizon_units():
    records = F.build_crossed_prompts()
    near_units = {Horizon.parse(r["delays"][0]).unit for r in records}
    assert near_units == set(F.NATURAL_UNITS)
    pairs = Counter((r["rendered_unit"], Horizon.parse(r["delays"][0]).unit) for r in records)
    assert len(pairs) >= 30                                             # of 49 unit pairs; many are infeasible by magnitude


def test_realism_labels():
    day, month, week = (Horizon(1, u).years for u in ("days", "months", "weeks"))
    assert F.realism_labels(0.5 * day, 1.0)["delay_scale"] == "minutes-hours"
    assert F.realism_labels(day, 1.0)["delay_scale"] == "days-weeks"
    assert F.realism_labels(month, 1.0)["delay_scale"] == "months-years"
    assert F.realism_labels(week, 50.0)["alan_range"] and not F.realism_labels(0.9 * week, 2.0)["alan_range"]
    assert not F.realism_labels(1.0, 51.0)["alan_range"]


def test_crossed_bank_limit_keeps_whole_bases():
    full = crossed_bases()
    records = F.build_crossed_prompts(limit=8)
    assert 4 <= len(records) <= 16
    counts = Counter(r["pair_id"] for r in records)
    assert all(n == len(full[p]) for p, n in counts.items())


def test_fill_bank_fills_the_old_banks_regime_gaps():
    old = Counter((r.get("duration_id") or r["horizon_text"], r["regime"]) for r in F.old_bank_records())
    plan = F.fill_plan()
    assert len(plan) == 72 * F.FILL_PER_CELL
    assert all(old[(str(h), name)] < F.FILL_MIN_OLD for h, name in plan)
    fill = [r for r in F.build_fill_prompts() if not r["anchor"]]
    bases = Counter((r["duration_id"], r["regime"]) for r in fill if r["natural"])
    # pooled, every feasible (duration, regime) cell has 3 old prompts or 2 fill bases
    for h in F.HORIZONS:
        for name in F.feasible_regimes(h.log10_years):
            assert old[(str(h), name)] >= F.FILL_MIN_OLD or bases[(str(h), name)] == F.FILL_PER_CELL


def test_fill_bank_decorrelates_delays_from_the_horizon_only_when_pooled():
    def corr(records):
        return np.corrcoef(np.log10([r["horizon"] for r in records]),
                           np.log10([r["near_delay_years"] for r in records]))[0, 1]
    old = F.old_bank_records()
    fill = [r for r in F.build_fill_prompts() if not r["anchor"]]
    assert corr(fill) > 0.8 and abs(corr(old)) < 0.2 and corr(old + fill) < 0.4


def test_fill_anchors_are_old_prompts_unchanged():
    records = F.build_fill_prompts()
    anchors = [r for r in records if r["anchor"]]
    assert len({r["pair_id"] for r in anchors}) == F.ANCHOR_FULL + F.ANCHOR_PAIRS
    old = {r["sample_uid"]: r for r in F.build_full_prompts() + F.build_units_prompts()}
    for r in anchors:
        assert r["prompt"] == old[r["sample_uid"]]["prompt"] and r["regime"] == old[r["sample_uid"]]["regime"]
        assert sum(a["natural"] for a in anchors if a["pair_id"] == r["pair_id"]) == 1
    fill_uids = [r["sample_uid"] for r in records if not r["anchor"]]
    assert len(set(fill_uids)) == len(fill_uids) and not set(fill_uids) & set(old)


def test_fill_bank_routes_through_settings_with_an_anchor_in_the_smoke():
    from spar_horizon.banks import build_records, variant_builder
    from spar_horizon.config import Settings
    cfg = Settings(bank="fill", prefill="", choice_prefix="I choose:", limit=8)
    records = build_records(cfg)
    assert 4 <= len(records) <= 16 and any(r["anchor"] for r in records) and not all(r["anchor"] for r in records)
    keys = ("pair_id", "duration_id", "rendered_unit", "natural", "delay_scale", "alan_range", "anchor")
    assert all(k in r for r in records for k in keys)
    assert all(variant_builder(cfg)(r) == r["prompt"] for r in records)
    assert cfg.tag == "Qwen3-8B_think-on_fill"
