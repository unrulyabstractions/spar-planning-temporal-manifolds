"""Tier T0: the template against the report's sample, and the composition of both prompt sets."""

from collections import Counter

import pytest

from pbank.durations import HORIZON_GRID, Duration, unit_rewrites
from pbank.records import Option, regime
from pbank.t0 import (CONSTRAINT_PHRASINGS, ExtractionConfig, VerificationConfig, build_extraction_bank,
                      build_verification_grid, make_record, ratio_grid, render)

# The sample prompt of the report's Appendix A.1, with the LaTeX layout (indentation) removed.
REPORT_SAMPLE = """\
SITUATION: Plan for the future of the household.
TASK: You, the head of the household, are tasked to choose the best investment:
a) 10,000 dollars in 1 year.
b) 50,000 dollars in 10 years.
OBJECTIVE: Think deeply about which option is preferable.
CONSTRAINT: You must select the option that provides the greatest benefit for this time horizon: 8 years.
ACTION: Select one of the two options. Provide reasoning.
FORMAT: I choose: <a) or b)>. My reasoning: <1-3 sentences>"""


@pytest.fixture(scope="module")
def bank():
    return build_extraction_bank()


@pytest.fixture(scope="module")
def grid():
    return build_verification_grid()


def test_report_sample_renders_exactly():
    rec = make_record(uid="t", split="extraction", near_delay=Duration.parse("1 year"),
                      far_delay=Duration.parse("10 years"), near_reward=10_000, multiplier=5, near_first=True,
                      labels=("a)", "b)"), horizon=Duration.parse("8 years"), grid_value=Duration.parse("8 years"),
                      phrasing_id=0)
    print(rec.text)
    assert rec.text == REPORT_SAMPLE
    # The report: H/D = 0.8, correct answer a.
    assert rec.regime == "near_only" and rec.correct_label == "a)" and rec.h_over_far_delay == pytest.approx(0.8)


def test_phrasings_labels_and_null_line():
    near, far = Option(1000, "1 week", 0.02), Option(5000, "2 years", 2.0)
    for pid, phrasing in enumerate(CONSTRAINT_PHRASINGS):
        text = render(near=near, far=far, near_first=False, labels=("1)", "2)"), horizon=Duration.parse("3 days"),
                      phrasing_id=pid)
        assert "CONSTRAINT: " + phrasing.format(H="3 days") in text
        assert text.splitlines()[2] == "1) 5,000 dollars in 2 years."        # far first
        assert text.endswith("FORMAT: I choose: <1) or 2)>. My reasoning: <1-3 sentences>")
    null = render(near=near, far=far, near_first=True, labels=("a)", "b)"), horizon=None, phrasing_id=None)
    assert "CONSTRAINT" not in null and len(null.splitlines()) == 7


def test_extraction_composition(bank):
    with_h = [r for r in bank if r.horizon is not None]
    nulls = [r for r in bank if r.horizon is None]
    print("regimes:", Counter(r.regime for r in bank))
    print("units:", Counter(r.horizon_unit for r in with_h))
    print(f"rewritten {sum(r.rewritten for r in with_h)}/{len(with_h)}; "
          f"ties {sum(r.horizon_equals_a_delay for r in with_h)}")
    print("phrasings:", Counter(r.phrasing_id for r in with_h), "labels:", Counter(r.labels for r in bank))
    assert len(bank) == 500 and len(with_h) == 471 and len(nulls) == 29
    assert all(r.split == "extraction" and r.tier == "T0" for r in bank)
    assert len({r.uid for r in bank}) == 500 and len({r.text for r in bank}) == 500
    # Balanced horizons: each of the 48 grid values 9 or 10 times (471 = 48 * 9 + 39).
    per_value = Counter(r.horizon_grid_value for r in with_h)
    assert set(per_value) == {str(d) for d in HORIZON_GRID}
    assert sorted(Counter(per_value.values()).items()) == [(9, 9), (10, 39)]
    # Uniform draws over 4 phrasings and 2 label styles: loose bounds (each expected ~118 and ~250).
    assert all(90 <= n <= 150 for n in Counter(r.phrasing_id for r in with_h).values())
    assert all(210 <= n <= 290 for n in Counter(r.labels for r in bank).values())


def test_regime_and_answer_follow_from_written_values(bank):
    """Recompute every prompt's regime from the text fields, independently of how it was built."""
    for r in bank:
        h = None if r.horizon is None else Duration.parse(r.horizon)
        assert r.regime == regime(h, Duration.parse(r.near.delay), Duration.parse(r.far.delay)), r.uid
        if r.regime == "near_only":
            assert r.correct_label == r.near_label and r.correct_label in r.text
        elif r.regime == "both":
            assert r.correct_label == r.far_label
        else:
            assert r.correct_label is None
        assert r.far.reward == r.near.reward * r.reward_multiplier > r.near.reward


def test_rewrites_are_valid_and_never_change_the_regime(bank):
    for r in bank:
        if not r.rewritten:
            continue
        grid_value, written = Duration.parse(r.horizon_grid_value), Duration.parse(r.horizon)
        assert written in unit_rewrites(grid_value)
        near, far = Duration.parse(r.near.delay), Duration.parse(r.far.delay)
        assert regime(written, near, far) == regime(grid_value, near, far), r.uid


def test_null_twins_are_their_source_minus_the_constraint_line(bank):
    by_uid = {r.uid: r for r in bank}
    for r in bank:
        if r.horizon is not None:
            continue
        src = by_uid[r.uid.removesuffix("-null")]
        assert r.pair_id == src.pair_id == f"{src.uid}+null"
        assert r.text.splitlines() == [line for line in src.text.splitlines() if not line.startswith("CONSTRAINT:")]
        assert r.regime == "no_horizon" and r.correct_label is None


def test_same_seed_same_bank_other_seed_different(bank):
    again = build_extraction_bank(ExtractionConfig(seed=0))
    assert [r.text for r in again] == [r.text for r in bank]
    other = build_extraction_bank(ExtractionConfig(seed=1))
    assert sum(a.text == b.text for a, b in zip(other, bank)) < 5


def test_ratio_grid_avoids_the_boundary():
    ratios = ratio_grid(100, 2.8)
    nearest = sorted(ratios, key=lambda x: abs(x - 1))[:2]
    print(f"H/D from {ratios[0]:.4f} to {ratios[-1]:.4f}; nearest to 1: {nearest[0]:.4f}, {nearest[1]:.4f}")
    assert ratios[0] == pytest.approx(1 / 2.8) and ratios[-1] == pytest.approx(2.8)
    assert min(abs(x - 1) for x in ratios) > 0.009
    with pytest.raises(ValueError):
        ratio_grid(99, 2.8)


def test_verification_grid(grid, bank):
    print("regimes:", Counter(r.regime for r in grid), " far delays:", Counter(r.far.delay for r in grid))
    assert len(grid) == 200 and all(r.split == "verification" for r in grid)
    assert Counter(r.regime for r in grid) == {"near_only": 100, "both": 100}
    for r in grid:
        h, near, far = Duration.parse(r.horizon), Duration.parse(r.near.delay), Duration.parse(r.far.delay)
        assert h.unit == far.unit == "years"                      # matched units
        assert 0.35 <= r.h_over_far_delay <= 2.8
        assert near.days < h.days and h.days != far.days          # near always pays; no tie with the far delay
        assert not r.horizon_equals_a_delay and r.correct_label is not None
        # Rounding H to two decimals moved it by at most 0.005 years from its target.
        assert abs(h.years - r.notes["target_h_over_d"] * far.years) <= 0.005 + 1e-12
    # Each item appears in both orders, otherwise identical.
    items = {}
    for r in grid:
        items.setdefault(r.pair_id, []).append(r)
    assert len(items) == 100
    for pair in items.values():
        a, b = pair
        assert a.near_first and not b.near_first
        assert (a.horizon, a.near, a.far, a.labels, a.phrasing_id) == (b.horizon, b.near, b.far, b.labels, b.phrasing_id)
        assert a.correct_option == b.correct_option and a.correct_label != b.correct_label
    # Far delays balanced: 100 items over 6 delays -> 16 or 17 each.
    assert set(Counter(r.far.delay for r in grid[::2]).values()) <= {16, 17}
    # No overlap with the extraction bank.
    assert not {r.text for r in grid} & {r.text for r in bank}


def test_verification_grid_is_deterministic_per_seed(grid):
    assert [r.text for r in build_verification_grid(VerificationConfig(seed=0))] == [r.text for r in grid]
    other = build_verification_grid(VerificationConfig(seed=1))
    assert [r.horizon for r in other][::2] != [r.horizon for r in grid][::2]   # different far delays per ratio
