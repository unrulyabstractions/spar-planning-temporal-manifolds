"""The expanded domains banks (100 base items): no duplicate base prompts, and the entries added for
the expansion are each used by at most 5 items. The 20 original items keep their own entries."""

from collections import Counter

import pytest

from spar_horizon.eval_types import build
from spar_horizon.eval_types import domains
from spar_horizon.eval_types.common import EXPANDED_N_BASE, PLAN

SEEDS = (0, 1, 2, 3, 4)
MAX_USES = 5
DOMAIN_TYPES = ("implicit_concrete", "implicit_vague", "city_no_values", "city_values", "public_health",
                "coding_agent")


def base_records(type_id, seed):
    return [r for r in build(type_id, seed=seed, n_base=EXPANDED_N_BASE) if r["variant"] == "base"]


def uses_of(type_id, record):
    """The id of the list entry a base record was built from."""
    if type_id.startswith("implicit"):
        return record["anchor_id"]
    texts = " ".join(o["text"] for o in record["options"])
    if type_id == "city_no_values":
        pairs = domains.CITY_PROJECT_PAIRS + domains.CITY_PROJECT_PAIRS_ADDED
        return next(small for small, big in pairs if small in texts and big in texts)
    pairs = domains.CITY_VALUE_PAIRS + domains.CITY_VALUE_PAIRS_ADDED
    return next(small for _, small, big, _ in pairs if f"A {small} that" in texts and f"A {big} that" in texts)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("type_id", DOMAIN_TYPES)
def test_no_duplicate_base_prompts(type_id, seed):
    prompts = [r["prompt"] for r in base_records(type_id, seed)]
    assert len(prompts) == EXPANDED_N_BASE
    assert len(set(prompts)) == len(prompts)


ENTRY_TYPES = {
    "implicit_concrete": domains.CONCRETE_ANCHORS_ADDED,
    "implicit_vague": domains.VAGUE_ANCHORS_ADDED,
    "city_no_values": domains.CITY_PROJECT_PAIRS_ADDED,
    "city_values": domains.CITY_VALUE_PAIRS_ADDED,
}


def entry_id(type_id, entry):
    if type_id.startswith("implicit"):
        return entry.anchor_id
    return entry[0] if type_id == "city_no_values" else entry[1]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("type_id", ENTRY_TYPES)
def test_added_entries_are_used_at_most_five_times_and_originals_only_early(type_id, seed):
    added = {entry_id(type_id, e) for e in ENTRY_TYPES[type_id]}
    records = base_records(type_id, seed)
    counts = Counter(uses_of(type_id, r) for r in records[len(PLAN):])
    assert set(counts) <= added
    assert max(counts.values()) <= MAX_USES
    assert {uses_of(type_id, r) for r in records[:len(PLAN)]}.isdisjoint(added)


@pytest.mark.parametrize("type_id", ("implicit_concrete", "implicit_vague"))
@pytest.mark.parametrize("regime", ("near", "both", "none"))
def test_every_added_anchor_has_delays_in_every_regime(type_id, regime):
    for anchor in ENTRY_TYPES[type_id]:
        assert domains.delay_pairs(regime, anchor.lo_years, anchor.hi_years, domains.IMPLICIT_MENU), anchor.anchor_id


@pytest.mark.parametrize("type_id", ("implicit_concrete", "implicit_vague"))
def test_added_anchors_are_distinct_and_numerous_enough(type_id):
    added = ENTRY_TYPES[type_id]
    assert len({a.anchor_id for a in added}) == len(added) >= 16
