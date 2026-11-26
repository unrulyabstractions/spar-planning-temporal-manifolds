"""Tests for confound_splits.py. Run (from evaluation/): python -m pytest -q test_confound_splits.py"""

import json
from collections import Counter
from pathlib import Path

import pytest

import confound_splits as cs
from pbank.durations import HORIZON_GRID
from pbank.records import prompts_sha256, read_jsonl

BANK = Path(__file__).resolve().parents[1] / "prompt_bank" / "data" / "t0_s0"
MP_BANK = BANK.parent / "t0_mp0"
DEBATABLE = {"x0004", "x0311", "x0042", "x0224", "x0234", "x0238", "x0272", "x0290", "x0340"}


@pytest.fixture(scope="module")
def records():
    return read_jsonl(BANK / "prompts.jsonl")


@pytest.fixture(scope="module")
def splits(records):
    return cs.build_splits(records)


def by_uid(records):
    return {r["uid"]: r for r in records}


def roles(rows, role):
    return {r["uid"] for r in rows if r["role"] == role}


# --- index files ------------------------------------------------------------------------------------------------

def test_index_rows_name_existing_uids_once(records, splits):
    uids = {r["uid"] for r in records}
    for name, rows in splits.items():
        assert {r["uid"] for r in rows} <= uids, name
        assert len(rows) == len({r["uid"] for r in rows}), f"{name} lists a uid twice"
        assert all(set(r) >= {"uid", "role"} for r in rows)


def test_splits_use_extraction_records_except_debatable(records, splits):
    split_of = {r["uid"]: r["split"] for r in records}
    for name, rows in splits.items():
        if name != "debatable":
            assert {split_of[r["uid"]] for r in rows} == {"extraction"}, name


def test_files_on_disk_are_what_the_code_builds(records, tmp_path):
    cs.write_splits(BANK, tmp_path)
    for name in cs.SPLIT_RULES:
        assert (tmp_path / f"{name}.jsonl").read_text() == (BANK / "splits" / f"{name}.jsonl").read_text()
    manifest = json.loads((tmp_path / "c5_surface.manifest.json").read_text())
    assert set(manifest) == {"name", "rule", "seed", "bank_prompts_sha256", "counts"}
    assert manifest["seed"] == 0 and manifest["counts"]["rows"] == len(read_jsonl(tmp_path / "c5_surface.jsonl"))


def test_build_is_deterministic_and_leaves_records_unchanged(records):
    before = json.dumps(records, sort_keys=True)
    assert cs.build_splits(records) == cs.build_splits(records)
    assert json.dumps(records, sort_keys=True) == before


# --- reserved horizons ------------------------------------------------------------------------------------------

def test_reserved_horizons_hold_out_eight_grid_values_keeping_both_ends(records, splits):
    rows, recs = splits["reserved_horizons"], by_uid(records)
    held_out = {r["horizon_grid_value"] for r in rows if r["role"] == "test"}
    assert len(held_out) == 8
    assert {str(HORIZON_GRID[0]), str(HORIZON_GRID[-1])}.isdisjoint(held_out)
    assert held_out == {str(HORIZON_GRID[i]) for i in (2, 8, 14, 20, 26, 32, 38, 44)}
    assert {r["uid"] for r in rows} == {r["uid"] for r in records if r["split"] == "extraction"}
    fit_values = {r["horizon_grid_value"] for r in rows if r["role"] == "fit"}
    assert fit_values.isdisjoint(held_out)
    for r in rows:
        assert (recs[r["uid"]]["horizon_grid_value"] in held_out) == (r["role"] == "test") or r["uid"].endswith("-null")


def test_null_twin_follows_its_source(splits):
    role = {r["uid"]: r["role"] for r in splits["reserved_horizons"]}
    twins = [u for u in role if u.endswith("-null")]
    assert twins and all(role[u] == role[u.removesuffix("-null")] for u in twins)


# --- C5 ---------------------------------------------------------------------------------------------------------

def test_c5_each_test_set_differs_from_fit_in_one_feature(records, splits):
    recs = by_uid(records)
    rows = splits["c5_surface"]
    features = lambda r: (r["phrasing_id"] in cs.FIT_PHRASINGS, not r["rewritten"], r["labels"][0] == "a)")
    by_set = {}
    for row in rows:
        by_set.setdefault(row["test_set"], set()).add(features(recs[row["uid"]]))
    assert by_set == {"fit": {(True, True, True)}, "test_phrasing": {(False, True, True)},
                      "test_rewrite": {(True, False, True)}, "test_labels": {(True, True, False)}}
    assert {r["role"] for r in rows if r["test_set"] == "fit"} == {"fit"}
    assert all(recs[r["uid"]]["horizon"] is not None for r in rows)


# --- C1 ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, with_none", [("c1_balanced", False), ("c1_balanced_with_none", True)])
def test_c1_cells_are_equal_in_size_and_classes_are_right(records, splits, name, with_none):
    recs = by_uid(records)
    cells = Counter()
    for row in splits[name]:
        r = recs[row["uid"]]
        ratio = r["h_over_far_delay"]
        assert row["caa_class"] == ("long" if ratio >= 4 else "short") and (ratio >= 4 or ratio <= 0.25)
        assert with_none or r["regime"] != "none"
        far_label_is_first = not r["near_first"]
        assert (r["labels"][0] if far_label_is_first else r["labels"][1]) == row["far_label"]
        cells[(row["caa_class"], r["near_first"], r["labels"][0])] += 1
    assert len(cells) == 8 and len(set(cells.values())) == 1


def test_c1_with_none_can_draw_none_prompts_and_without_cannot(records, splits):
    recs = by_uid(records)
    assert "none" not in {recs[r["uid"]]["regime"] for r in splits["c1_balanced"]}
    assert "none" in {recs[r["uid"]]["regime"] for r in splits["c1_balanced_with_none"]}


def test_c1_seed_changes_the_subsample(records):
    a = {r["uid"] for r in cs.c1_balanced(records, include_none=True, seed=0)}
    b = {r["uid"] for r in cs.c1_balanced(records, include_none=True, seed=1)}
    assert a != b


# --- debatable --------------------------------------------------------------------------------------------------

def test_debatable_is_the_ties_and_near_delay_rewrites(splits):
    assert roles(splits["debatable"], "drop") == DEBATABLE


# --- balance table ----------------------------------------------------------------------------------------------

def synthetic(n_per_regime=40, skew=False):
    """near_first is independent of regime unless `skew`, where `both` prompts are mostly near-first."""
    rows = []
    for regime, correct in (("near_only", "near"), ("both", "far")):
        for i in range(n_per_regime):
            near_first = (i < 0.9 * n_per_regime) if skew and regime == "both" else i % 2 == 0
            rows.append({"regime": regime, "correct_option": correct, "near_first": near_first,
                         "labels": ["a)", "b)"] if i % 2 else ["1)", "2)"], "phrasing_id": i % 4,
                         "rewritten": False, "reward_multiplier": 2})
    return rows


def test_balance_table_quiet_on_balanced_rows():
    tab = cs.balance_table(synthetic())
    assert all(not cell["flags"] for per_by in tab.values() for cell in per_by.values())
    assert tab["near_first"]["regime"]["counts"]["both"] == {"True": 20, "False": 20}


def test_balance_table_flags_a_skewed_factor():
    tab = cs.balance_table(synthetic(skew=True))
    flags = tab["near_first"]["regime"]["flags"]
    assert {(f["by_level"], f["factor_level"]) for f in flags} == {
        ("both", "True"), ("both", "False"), ("near_only", "True"), ("near_only", "False")}
    both_true = next(f for f in flags if (f["by_level"], f["factor_level"]) == ("both", "True"))
    assert both_true["share"] == 0.9 and both_true["overall_share"] == pytest.approx(0.7)
    assert tab["near_first"]["correct_option"]["flags"]
    assert not tab["labels"]["regime"]["flags"]


def test_balance_table_ignores_groups_below_ten_records():
    rows = synthetic(skew=True, n_per_regime=5)
    assert not cs.balance_table(rows)["near_first"]["regime"]["flags"]


def test_balance_table_flags_known_confounds_of_t0(records):
    extraction = [r for r in records if r["split"] == "extraction"]
    flags = cs.balance_table(extraction)["rewritten"]["regime"]["flags"]
    assert {f["by_level"] for f in flags} >= {"both", "near_only"}


def test_balance_markdown_marks_flags(records):
    text = cs.balance_markdown({"t": (len(records), cs.balance_table(records))})
    assert "*" in text and "## t (n = 700)" in text


# --- minimal pairs ----------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pairs():
    return cs.build_minimal_pairs()


def test_minimal_pair_bank_shape(pairs):
    assert len(pairs) == 480 and {r.split for r in pairs} == {"minimal_pair"}
    assert all(r.uid.startswith("m") for r in pairs)
    items = Counter(r.pair_id for r in pairs)
    assert len(items) == 120 and set(items.values()) == {4}
    assert len({r.text for r in pairs}) == 480


def test_minimal_pair_regimes_straddle_the_far_delay(pairs):
    for r in pairs:
        short = "-short-" in r.uid
        assert r.regime == ("near_only" if short else "both")
        assert r.correct_option == ("near" if short else "far")
        lo, hi = cs.MP_RATIO_RANGE
        assert (1 / hi - 0.01 <= r.h_over_far_delay <= 1 / lo + 0.01) if short else (lo - 0.01 <= r.h_over_far_delay <= hi + 0.01)
        assert not r.horizon_equals_a_delay and not r.rewritten and r.horizon_unit == "years"
        assert r.horizon == r.horizon_grid_value


def test_minimal_pair_partners_differ_only_in_the_horizon_text(pairs):
    recs = {r.uid: r for r in pairs}
    for item in {r.pair_id for r in pairs}:
        for order in ("near-first", "far-first"):
            short, long = recs[f"{item}-short-{order}"], recs[f"{item}-long-{order}"]
            assert short.text.replace(short.horizon, "H") == long.text.replace(long.horizon, "H")
            assert short.horizon != long.horizon
        a, b = recs[f"{item}-short-near-first"], recs[f"{item}-short-far-first"]
        assert (a.near, a.far, a.phrasing_id, a.labels, a.horizon) == (b.near, b.far, b.phrasing_id, b.labels, b.horizon)
        assert a.near_first and not b.near_first


def test_minimal_pair_horizons_have_the_same_number_format(pairs):
    """Both horizons of every item: exactly two decimals, no trailing zero, so format cannot carry the regime."""
    for r in pairs:
        number = r.horizon.split()[0]
        assert len(number.split(".")[1]) == 2 and not number.endswith("0"), r.uid
    recs = {r.uid: r for r in pairs}
    for item in {r.pair_id for r in pairs}:
        short, long = recs[f"{item}-short-near-first"], recs[f"{item}-long-near-first"]
        assert short.text.replace(short.horizon, "H") == long.text.replace(long.horizon, "H")


def test_minimal_pair_near_delay_is_below_the_short_horizon_and_items_are_balanced(pairs):
    for r in pairs:
        if "-short-" in r.uid:
            assert r.near.delay_years < r.horizon_years
    items = [r for r in pairs if r.uid.endswith("-short-near-first")]
    assert len(items) == 120
    for counts in (Counter(r.far.delay for r in items), Counter(r.reward_multiplier for r in items),
                   Counter(r.phrasing_id for r in items), Counter("/".join(r.labels) for r in items)):
        assert len(set(counts.values())) == 1, counts
    assert set(Counter(r.far.delay for r in items).values()) == {20}
    assert set(Counter(r.reward_multiplier for r in items).values()) == {24}


def test_minimal_pairs_distinct_from_t0_and_deterministic(pairs, records):
    assert {r.text for r in pairs}.isdisjoint(r["text"] for r in records)
    assert prompts_sha256(pairs) == prompts_sha256(cs.build_minimal_pairs())
    assert prompts_sha256(pairs) != prompts_sha256(cs.build_minimal_pairs(seed=1))


def test_minimal_pair_item_count_must_keep_the_balance():
    with pytest.raises(ValueError):
        cs.build_minimal_pairs(n_items=100)


def test_committed_minimal_pair_bank_matches_the_builder(pairs):
    manifest = json.loads((MP_BANK / "manifest.json").read_text())
    assert manifest["prompts_sha256"] == prompts_sha256(pairs)
    assert [r["uid"] for r in read_jsonl(MP_BANK / "prompts.jsonl")] == [r.uid for r in pairs]
