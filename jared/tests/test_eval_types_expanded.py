"""The expanded banks (100 base items per type): sizes, and that every record of the original bank
is reproduced exactly, so E0-E21 results and cached generations stay valid on those rows."""

import pytest

from spar_horizon.eval_types import TYPES, build
from spar_horizon.eval_types.common import EXPANDED_N_BASE

# The types that passed E19's gate with thinking on (EvalsPromptsV2.md §8).
GATED = ("system_prompt", "city_no_values", "distractor_temporal", "t0_w0", "t0_w3_clear", "sycophancy_lean",
         "t0_w1_cash", "implicit_vague", "city_values", "implicit_concrete", "role_assistant", "prose_natural",
         "multiturn_analyst", "system_prompt_clear", "unit_mismatch", "role_fund_manager", "public_health",
         "horizon_revised_clear", "coding_agent")


@pytest.mark.parametrize("type_id", GATED)
def test_expanded_bank_keeps_every_original_record(type_id):
    original = build(type_id)
    expanded = {r["sample_uid"]: r for r in build(type_id, n_base=EXPANDED_N_BASE)}
    for r in original:
        assert expanded.get(r["sample_uid"]) == r, r["sample_uid"]


@pytest.mark.parametrize("type_id", GATED)
def test_expanded_bank_sizes(type_id):
    records = build(type_id, n_base=EXPANDED_N_BASE)
    variants = [r["variant"] for r in records]
    assert variants.count("base") == EXPANDED_N_BASE
    assert variants.count("swap") == 40
    assert variants.count("twin") == (20 if TYPES[type_id].twins else 0)
    assert len({r["sample_uid"] for r in records}) == len(records)
