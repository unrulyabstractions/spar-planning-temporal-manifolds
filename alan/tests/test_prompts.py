import math
from ptm.prompts import CONSTRAINT_PHRASINGS, UNIT_REWRITES, DatasetConfig, PromptFormat, generate, to_frame, from_frame
from ptm.horizons import STANDARD_HORIZONS, Horizon


def test_generation_is_deterministic_and_complete():
    a = generate(DatasetConfig(n=50, seed=3))
    b = generate(DatasetConfig(n=50, seed=3))
    assert [s.text for s in a] == [s.text for s in b]
    assert len({s.sample_uid for s in a}) == 50
    nulls = [s for s in a if s.horizon_text is None]
    assert all(math.isnan(s.horizon_years) for s in nulls)
    assert all(s.short_delay_years < s.long_delay_years and s.short_reward < s.long_reward for s in a)


def test_render_has_markers_and_horizon():
    s = generate(DatasetConfig(n=200, seed=1))
    with_h = next(x for x in s if x.horizon_text is not None)
    no_h = next(x for x in s if x.horizon_text is None)
    for m in ["SITUATION:", "TASK:", "OBJECTIVE:", "ACTION:", "FORMAT:", "I choose:", "My reasoning:"]:
        assert m in with_h.text and m in no_h.text
    assert "CONSTRAINT:" in with_h.text and with_h.horizon_text in with_h.text
    assert "CONSTRAINT:" not in no_h.text
    print("\n--- example prompt ---\n" + with_h.text)


def test_order_and_labels_consistent():
    for s in generate(DatasetConfig(n=100, seed=2)):
        lines = s.text.split("\n")
        first_opt = lines[2]
        assert first_opt.startswith(s.label_a)
        if s.short_first:
            assert s.short_delay_text in first_opt and s.short_label == s.label_a
        else:
            assert s.long_delay_text in first_opt and s.long_label == s.label_a


def test_frame_roundtrip():
    s = generate(DatasetConfig(n=20, seed=5))
    back = from_frame(to_frame(s))
    assert [x.text for x in back] == [x.text for x in s]
    assert [x.horizon_text for x in back] == [x.horizon_text for x in s]


def test_unit_rewrites_preserve_magnitude_within_10pct():
    canon = {str(h): h for h in STANDARD_HORIZONS}
    assert set(UNIT_REWRITES) == set(canon)
    for label, alts in UNIT_REWRITES.items():
        for alt in alts:
            h = Horizon.parse(alt)
            assert float(h.value).is_integer(), alt
            ratio = h.years / canon[label].years
            assert 0.9 <= ratio <= 1.1, (label, alt, ratio)
            assert h.unit != canon[label].unit, (label, alt)
    print("\nrewrite counts:", {k: len(v) + 1 for k, v in UNIT_REWRITES.items()})


def test_unit_rewrite_dataset_columns_and_render():
    s = generate(DatasetConfig(n=400, seed=7, unit_rewrite=True))
    rewritten = [x for x in s if x.horizon_text and x.horizon_text != x.horizon_canonical]
    assert len(rewritten) > 100
    for x in rewritten:
        constraint = next(l for l in x.text.split("\n") if l.startswith("CONSTRAINT:"))
        assert x.horizon_text in constraint and x.horizon_canonical not in constraint
        assert Horizon.parse(x.horizon_text).unit == x.horizon_unit
        assert abs(x.horizon_years / Horizon.parse(x.horizon_canonical).years - 1) <= 0.1
    units = {x.horizon_unit for x in rewritten}
    assert {"hours", "days", "weeks", "months", "decades", "centuries"} <= units
    # canonical-only config never rewrites
    for x in generate(DatasetConfig(n=200, seed=7)):
        assert x.horizon_text is None or x.horizon_text == x.horizon_canonical


def test_phrasing_variants_render():
    s = generate(DatasetConfig(n=300, seed=8, phrasing_ids=(0, 1, 2, 3)))
    seen = set()
    for x in s:
        if x.horizon_text is None:
            assert "CONSTRAINT:" not in x.text
            continue
        seen.add(x.phrasing_id)
        expected = CONSTRAINT_PHRASINGS[x.phrasing_id].format(horizon=x.horizon_text)
        assert ("CONSTRAINT: " + expected) in x.text
    assert seen == {0, 1, 2, 3}
    # phrasing 0 equals the original template exactly
    a = generate(DatasetConfig(n=50, seed=9)); b = generate(DatasetConfig(n=50, seed=9, phrasing_ids=(0,)))
    assert [x.text for x in a] == [x.text for x in b]


def test_frame_roundtrip_with_new_columns():
    s = generate(DatasetConfig(n=60, seed=10, unit_rewrite=True, phrasing_ids=(0, 2)))
    back = from_frame(to_frame(s))
    assert [(x.text, x.phrasing_id, x.horizon_unit) for x in back] == [(x.text, x.phrasing_id, x.horizon_unit) for x in s]


def test_paired_unit_rewrite_shares_everything_but_the_horizon_text():
    s = generate(DatasetConfig(n=300, seed=11, unit_rewrite=True, paired=True))
    df = to_frame(s)
    groups = df[df.pair_id.notna()].groupby("pair_id")
    assert groups.ngroups > 60
    for pid, g in groups:
        if g.horizon_text.isna().all():
            assert len(g) == 1
            continue
        assert g.horizon_canonical.nunique() == 1
        assert len(g) == 1 + len(UNIT_REWRITES[g.horizon_canonical.iloc[0]]) or pid == df.pair_id.dropna().iloc[-1]
        for col in ["short_reward", "long_reward", "short_delay_text", "long_delay_text", "short_first", "label_a", "domain"]:
            assert g[col].nunique() == 1, (pid, col)
        assert g.horizon_text.nunique() == len(g)
        assert g.sample_uid.nunique() == len(g)
    nulls = df[df.horizon_text.isna()]
    assert nulls.pair_id.notna().all() and nulls.groupby("pair_id").size().max() == 1


def test_main_dataset_regenerates_identically():
    """Guards the RNG stream: data/prompts/investment_n2000_s0.parquet must be reproducible."""
    import os
    import pandas as pd
    path = "data/prompts/investment_n2000_s0.parquet"
    if not os.path.exists(path):
        import pytest; pytest.skip("main dataset not present")
    old = pd.read_parquet(path)
    new = to_frame(generate(DatasetConfig(n=2000, seed=0)))
    assert (old.text.values == new.text.values).all()
