"""The expanded banks (100 base items) of the framing, robustness, investment and wording types: no
repeated prompt, the hand-written pools extended for the expansion are each used by at most 5 items,
original items keep the original wordings, and types that share an item_seed stay aligned."""

from collections import Counter

import pytest

from spar_horizon.eval_types import build
from spar_horizon.eval_types.common import EXPANDED_N_BASE, PLAN
from spar_horizon.eval_types.robustness import TEMPORAL_DISTRACTORS
from spar_horizon.horizons import Horizon

TYPE_IDS = ("system_prompt", "system_prompt_clear", "sycophancy_lean", "sycophancy_lean_clear", "role_assistant",
            "role_fund_manager", "prose_natural", "multiturn_analyst", "distractor_temporal", "unit_mismatch",
            "horizon_revised", "horizon_revised_clear", "t0_w0", "t0_w1_cash", "t0_w3_clear")
MAX_USES = 5
FIRST_NEW = len(PLAN)

# type -> record fields that index a pool extended for the expansion (absent on the original items)
EXTENDED_POOLS = {
    "sycophancy_lean": ("lean_phrasing_id",),
    "sycophancy_lean_clear": ("lean_phrasing_id",),
    "prose_natural": ("intro_id", "closing_id"),
    "multiturn_analyst": ("script_id",),
    "unit_mismatch": ("unit_pair",),
    "horizon_revised": ("update_phrasing_id",),
    "horizon_revised_clear": ("update_phrasing_id",),
}
# pairs that must decide the same problems prompt by prompt (item_seed shared)
ALIGNED = (("system_prompt", "system_prompt_clear"), ("sycophancy_lean", "sycophancy_lean_clear"),
           ("horizon_revised", "horizon_revised_clear"), ("t0_w0", "t0_w1_cash"), ("t0_w0", "t0_w3_clear"))


def bases(type_id):
    return [r for r in build(type_id, n_base=EXPANDED_N_BASE) if r["variant"] == "base"]


def text(record):
    p = record["prompt"]
    return p if isinstance(p, str) else "\n".join(m["content"] for m in p)


def decision(record):
    return record["variant"], record["regime"], record["labels"], record["options"], record["truth_labels"]


@pytest.mark.parametrize("type_id", TYPE_IDS)
def test_no_duplicate_base_prompts(type_id):
    rows = bases(type_id)
    assert len(rows) == EXPANDED_N_BASE
    assert len({text(r) for r in rows}) == EXPANDED_N_BASE


@pytest.mark.parametrize("type_id,fields", EXTENDED_POOLS.items())
def test_extended_pools_have_a_reuse_cap_and_leave_originals_alone(type_id, fields):
    rows = bases(type_id)
    for field in fields:
        assert all(field not in r for r in rows[:FIRST_NEW])
        counts = Counter(r[field] for r in rows[FIRST_NEW:])
        assert sum(counts.values()) == EXPANDED_N_BASE - FIRST_NEW
        assert max(counts.values()) <= MAX_USES, (field, counts)


def test_prose_horizon_sentences_are_capped():
    rows = bases("prose_natural")
    assert max(Counter(r["phrasing_id"] for r in rows[FIRST_NEW:]).values()) <= MAX_USES
    assert all(r["phrasing_id"] < 3 for r in rows[:FIRST_NEW])
    assert all(r["phrasing_id"] >= 3 for r in rows[FIRST_NEW:])


def test_distractor_sentences_are_capped_and_originals_keep_theirs():
    rows = bases("distractor_temporal")
    original_sentences = {t.format(d=d) for t, ds in TEMPORAL_DISTRACTORS for d in ds}
    assert {r["distractor_text"] for r in rows[:FIRST_NEW]} <= original_sentences
    new = Counter(r["distractor_text"] for r in rows[FIRST_NEW:])
    assert not set(new) & original_sentences
    assert max(new.values()) <= MAX_USES, new
    assert all(r["distractor_log_gap"] >= 1 for r in rows)


def test_unit_mismatch_new_items_state_the_horizon_in_another_unit():
    for r in bases("unit_mismatch")[FIRST_NEW:]:
        delay_unit, horizon_unit = r["unit_pair"].split("/")
        assert delay_unit != horizon_unit
        assert Horizon.parse(r["rendered_horizon_text"]).unit == horizon_unit
        assert all(Horizon.parse(o["delay_text"]).unit == delay_unit for o in r["options"])


@pytest.mark.parametrize("first,second", ALIGNED)
def test_item_seed_sharing_types_stay_aligned(first, second):
    a, b = build(first, n_base=EXPANDED_N_BASE), build(second, n_base=EXPANDED_N_BASE)
    assert [decision(r) for r in a] == [decision(r) for r in b]
    for r, s in zip(a, b):
        if r["variant"] != "twin":
            assert r["prompt"] != s["prompt"]


@pytest.mark.parametrize("type_id", ("horizon_revised", "horizon_revised_clear"))
def test_revision_line_states_the_new_horizon_after_the_stale_one(type_id):
    for r in bases(type_id):
        t = text(r)
        assert f"{r['stale_horizon_text']}." in t.split("UPDATE:")[0]
        update = t[t.index("UPDATE:"):].split("\n")[0]
        assert r["horizon_text"] in update
        assert t.index("CONSTRAINT") < t.index("UPDATE") < t.index("ACTION")
