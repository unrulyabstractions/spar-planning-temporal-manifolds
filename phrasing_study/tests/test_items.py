import math

import pytest

from phr.durations import has_duration, parse_years
from phr.items import build_choice, build_state, check_config, cue_table, fill_template, load_config
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
        if r.family != "layout":
            changed = sum(a != b for a, b in zip(r.text.split("\n"), ref.split("\n")))
            changed += abs(len(r.text.split("\n")) - len(ref.split("\n")))
            assert changed <= 3, (r.variant, changed)


def test_horizon_only_where_expected(choice):
    nh = choice[choice.family == "no_horizon"]
    assert not nh.text.str.contains("CONSTRAINT").any()
    imp = choice[(choice.family == "implicit") & ~choice.with_number.astype(bool)]
    assert not any(has_duration(t) or any(c.isdigit() for c in t) for t in imp.h_text)


def test_order_and_labels(choice):
    for r in choice.sample(200, random_state=0).itertuples():
        lines = r.text.split("\n")
        first = next(l.lstrip("- ") for l in lines if l.lstrip("- ").startswith(("a)", "b)", "A)", "B)", "1)", "2)")))
        short_line_first = first.startswith(r.label_short)
        assert short_line_first == r.short_first


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
            assert user_message(s, cfg, s.branch_turn).endswith(cue_table(cfg)[s.cue]["text"])
            assert all(user_message(s, cfg, k) == user_message(by[s.parent], cfg, k) for k in range(1, s.branch_turn))
        if s.cue_where == "first":
            assert s.first_user.endswith(cue_table(cfg)[s.cue]["text"])
        if s.condition == "free":
            assert not has_duration(s.first_user)
    m = messages(specs[0], cfg, ["outline", "step 1"])
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user", "assistant", "user"]


def test_fill_template_optional_parts():
    t = "A {x}\n[B: {y}]\n{z}[ and {y}] end"
    assert fill_template(t, {"x": 1, "y": None, "z": "z"}) == "A 1\nz end"
    assert fill_template(t, {"x": 1, "y": "y", "z": "z"}) == "A 1\nB: y\nz and y end"


def test_every_layout_renders_options_format_and_horizon(cfg, choice):
    can = choice[choice.family.isin(["canonical", "layout"])]
    assert set(can.layout) == set(cfg["choice"]["layouts"])
    for r in can.itertuples():
        assert r.h_text in r.text and "I choose: <" in r.text and r.text.count(" dollars in ") + r.text.count(" prevented in ") == 2


def test_no_horizon_and_cues_in_every_layout(choice):
    nh = choice[choice.family == "no_horizon"]
    for r in nh.itertuples():
        assert "CONSTRAINT" not in r.text and "time horizon" not in r.text
    cross = choice[choice.family == "layout_cross"]
    assert cross.layout.nunique() == choice[choice.family == "layout"].layout.nunique()
    cues = cue_table(load_config())
    for r in cross[cross.cue.notna()].itertuples():
        assert r.text.count(cues[r.cue]["text"]) == 1


def test_implicit_items_have_determinacy(choice):
    imp = choice[choice.family == "implicit"]
    assert set(imp.determinacy) == {"concrete", "vague"} and imp.layout.nunique() >= 2
