"""The full bank: a faithful copy of Alan's generator, plus our record columns."""

import json
import math
from collections import Counter
from pathlib import Path

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
