import math

import pytest

from spar_horizon import prompts as P

STARTER_FIRST = (
    "You must choose the best investment:\n"
    "a) 1,000 dollars in 6 months.\n"
    "b) 50,000 dollars in 10 years.\n"
    "Select the option with the greatest benefit for this time horizon: 30 seconds.\n"
    "Answer with a) or b)."
)


def test_bank_matches_starter():
    recs = P.build_prompts()
    assert len(recs) == 116
    assert recs[0]["prompt"] == STARTER_FIRST
    assert sum(r["horizon"] is None for r in recs) == 20
    assert len({r["prompt"] for r in recs if r["horizon"] is not None}) == 96


def test_every_horizon_prompt_reparses():
    for r in P.build_prompts():
        if r["horizon"] is None:
            assert "time horizon" not in r["prompt"]
            continue
        got = P.parse_horizon_years(r["prompt"].split("\n")[3])
        assert got is not None
        assert math.isclose(got, r["horizon"], rel_tol=0.05), (r["prompt"], got)


def test_subset_keeps_spread():
    recs = P.build_prompts(limit=6)
    assert len(recs) == 6
    horizons = [r["horizon"] for r in recs]
    assert len(set(horizons)) >= 4


def test_swap_and_relabel():
    base = P.choice_prompt((1_000, 50_000), ("6 months", "10 years"), 1)
    swapped = P.choice_prompt((1_000, 50_000), ("6 months", "10 years"), 1, swap=True)
    assert base.split("\n")[1][3:] == swapped.split("\n")[2][3:]
    assert swapped.split("\n")[1] == "a) 50,000 dollars in 10 years."
    relabeled = P.choice_prompt((1_000, 50_000), ("6 months", "10 years"), 1, labels=("1", "2"))
    assert relabeled.split("\n")[1].startswith("1) ")
    assert relabeled.endswith("Answer with 1) or 2).")


@pytest.mark.parametrize("name", list(P.CONSTRAINT_TEMPLATES))
def test_constraint_templates_carry_horizon(name):
    p = P.choice_prompt((1_000, 50_000), ("6 months", "10 years"), 5, template=name)
    assert P.parse_horizon_years(p.split("\n")[3]) == 5
    assert p.split("\n")[:3] == STARTER_FIRST.split("\n")[:3]


def test_constraint_templates_are_distinct():
    sentences = {t.format(h="5 years") for t in P.CONSTRAINT_TEMPLATES.values()}
    assert len(sentences) == len(P.CONSTRAINT_TEMPLATES)


@pytest.mark.parametrize("name", list(P.PERTURBATIONS))
def test_perturbations_keep_horizon(name):
    base = P.choice_prompt((5_000, 200_000), ("1 month", "5 years"), 2)
    out = P.PERTURBATIONS[name](base)
    assert out != base
    constraint = P.CONSTRAINT_TEMPLATES["starter"].format(h="2 years")
    assert constraint in out
    assert out.endswith("Answer with a) or b).")
    assert "a) 5,000 dollars in 1 month." in out and "b) 200,000 dollars in 5 years." in out


def test_horizon_first_moves_constraint_above_options():
    base = P.choice_prompt((5_000, 200_000), ("1 month", "5 years"), 2)
    lines = P.perturb_horizon_first(base).split("\n")
    assert lines[1].startswith("Select the option") and lines[2].startswith("a) ")


def test_implicit_bank():
    recs = P.build_implicit_prompts()
    assert len(recs) == len(P.IMPLICIT_HORIZONS) * 6
    for r in recs:
        assert "time horizon" not in r["prompt"]
        assert P.IMPLICIT_HORIZONS[r["horizon"]] in r["prompt"]


def test_unit_bank_reparses_to_same_duration():
    recs = P.build_unit_prompts()
    assert len(recs) == 6 * 3 * 6
    for r in recs:
        got = P.parse_horizon_years(r["spelling"])
        assert got is not None, r["spelling"]
        assert math.isclose(got, r["horizon"], rel_tol=0.1), (r["spelling"], got, r["horizon"])
        assert r["spelling"] in r["prompt"]


def test_horizon_text_units():
    assert P.horizon_text(30 / P.SECONDS_PER_YEAR) == "30 seconds"
    assert P.horizon_text(1 / 365) == "1 days"
    assert P.horizon_text(0.5) == "6 months"
    assert P.horizon_text(500) == "500 years"
