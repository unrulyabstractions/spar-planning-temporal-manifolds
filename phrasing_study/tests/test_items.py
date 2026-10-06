import math

import pytest

from phr.durations import has_duration, parse_years
from phr.items import build_choice, build_state, check_config, choice_variants, load_config
from phr.multiturn import build_specs, messages, user_message


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def choice(cfg):
    return build_choice(cfg)


def test_config_is_valid(cfg):
    assert check_config(cfg) == []


@pytest.mark.parametrize("text,years", [
    ("7 days", 7 / 365.25), ("a quarter of a year", 0.25), ("half a decade", 5.0), ("half a century", 50.0),
    ("a couple of years", 2.0), ("18,250 days", 18250 / 365.25), ("6-12 months", 1.0), ("two decades", 20.0),
    ("Time horizon: 9 months", 0.75), ("no idea", None)])
def test_parse_years(text, years):
    got = parse_years(text)
    assert (got is None and years is None) or math.isclose(got, years, rel_tol=1e-6)


def test_ids_unique_and_refs_exist(choice):
    assert choice.item_id.is_unique
    assert not choice.duplicated(["variant", "cell"]).any()
    keys = set(zip(choice.variant, choice.cell))
    has_ref = choice[choice.ref_variant.notna()]
    assert all((r, c) in keys for r, c in zip(has_ref.ref_variant, has_ref.cell))


def test_variant_differs_from_reference_only_in_its_change(choice):
    """Every variant's text differs from its reference on the same cell, and only in a few lines."""
    texts = dict(zip(zip(choice.variant, choice.cell), choice.text))
    for r in choice[choice.ref_variant.notna()].itertuples():
        ref = texts[(r.ref_variant, r.cell)]
        assert r.text != ref, r.variant
        if r.variant not in ("prose", "constraint_first"):
            changed = sum(a != b for a, b in zip(r.text.split("\n"), ref.split("\n")))
            changed += abs(len(r.text.split("\n")) - len(ref.split("\n")))
            assert changed <= 3, (r.variant, changed)


def test_horizon_only_where_expected(choice):
    nh = choice[choice.family == "no_horizon"]
    assert not nh.text.str.contains("CONSTRAINT").any()
    imp = choice[(choice.family == "implicit") & ~choice.with_number.astype(bool)]
    assert not any(has_duration(t.split("CONSTRAINT:")[1].split("\n")[0]) for t in imp.text)


def test_order_and_labels(choice):
    for r in choice.sample(200, random_state=0).itertuples():
        lines = r.text.split("\n")
        first = next(l for l in lines if l.startswith(("a)", "b)", "A)", "B)", "1)", "2)")))
        is_short_first = first.startswith(r.label_short)
        assert is_short_first == r.short_first or not r.short_first and first.startswith(r.label_long)


def test_unit_forms_cover_levels(cfg, choice):
    u = choice[choice.family == "unit"]
    for v in cfg["choice"]["families"]["unit"]["forms"]:
        assert u[u.variant == v].level.nunique() >= 3, v


def test_state_items(cfg):
    s = build_state(cfg)
    assert s.item_id.is_unique and s.text.str.contains("how long is the time horizon").all()


def test_multiturn_specs(cfg):
    specs = build_specs(cfg)
    by = {s.conv_id: s for s in specs}
    for s in specs:
        if s.cue_where == "step":
            assert s.parent in by and by[s.parent].cue is None and by[s.parent].greedy == s.greedy
            assert s.branch_turn == cfg["multiturn"]["cue_step"] + 1
            assert user_message(s, cfg, s.branch_turn).endswith(cfg["multiturn"]["cues"][s.cue])
            assert all(user_message(s, cfg, k) == user_message(by[s.parent], cfg, k) for k in range(1, s.branch_turn))
        if s.cue_where == "first":
            assert s.first_user.endswith(cfg["multiturn"]["cues"][s.cue])
        if s.condition == "free":
            assert not has_duration(s.first_user)
    m = messages(specs[0], cfg, ["outline", "step 1"])
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user", "assistant", "user"]
