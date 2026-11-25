"""Type-specific tests of the framing group (spar_horizon/eval_types/framing.py)."""

import json

import pytest

from spar_horizon.eval_types import build
from spar_horizon.eval_types.common import alan_prompt, labelled
from spar_horizon.eval_types.framing import PROSE_HORIZON_SENTENCES


def horizon_of(rec):
    return rec["horizon_text"]


def shown_of(rec):
    return [(o["label"], o) for o in sorted(rec["options"], key=lambda o: o["label"])]


def test_json_io_parses_and_holds_horizon():
    for r in build("json_io"):
        if r["variant"] == "twin":
            continue
        body, last = r["prompt"].rsplit("\n", 1)
        assert last == "Reply with a single JSON object in the response_format."
        data = json.loads(body)
        assert set(data["options"]) == set(r["labels"])
        assert data["constraint"].endswith(f"time horizon: {horizon_of(r)}.")
        assert r["choice_prefix"] == '"choice":'
        assert all(text.startswith(f"{label}) ") and not text.endswith(".") for label, text in data["options"].items())


def test_system_prompt_moves_the_constraint():
    for r in build("system_prompt"):
        if r["variant"] == "twin":
            continue
        system, user = r["prompt"]
        assert system["role"] == "system" and user["role"] == "user"
        assert f"time horizon: {horizon_of(r)}." in system["content"]
        assert "CONSTRAINT:" not in user["content"] and horizon_of(r) not in user["content"]


def lean_label(rec):
    return next(o["label"] for o in rec["options"] if o["kind"] == rec["lean_kind"])


def test_sycophancy_lean_is_never_correct_and_follows_the_swap():
    recs = [r for r in build("sycophancy_lean") if r["variant"] != "twin"]
    for r in recs:
        assert f"I'm leaning towards {lean_label(r)})." in r["prompt"]
        if r["regime"] in ("near", "both"):
            assert lean_label(r) not in r["truth_labels"]
    by_twin = {}
    for r in recs:
        by_twin.setdefault(r["twin_id"], {})[r["variant"]] = r
    swaps = [v for v in by_twin.values() if "swap" in v]
    assert swaps
    for v in swaps:
        assert v["base"]["lean_kind"] == v["swap"]["lean_kind"]
        assert lean_label(v["base"]) != lean_label(v["swap"])


def test_sycophancy_none_regime_leans_both_ways():
    kinds = {r["lean_kind"] for r in build("sycophancy_lean") if r["regime"] == "none"}
    assert kinds == {"near", "far"}


def test_prose_natural_has_no_markers_and_states_the_horizon():
    recs = [r for r in build("prose_natural") if r["variant"] != "twin"]
    for r in recs:
        for marker in ("SITUATION:", "TASK:", "OBJECTIVE:", "CONSTRAINT:", "FORMAT:"):
            assert marker not in r["prompt"]
        sentence = PROSE_HORIZON_SENTENCES[r["phrasing_id"]].format(h=horizon_of(r))
        assert sentence in r["prompt"]
    assert {r["phrasing_id"] for r in recs} == {0, 1, 2}


def test_multiturn_last_turn_does_not_restate_the_horizon():
    for r in build("multiturn_analyst"):
        if r["variant"] == "twin":
            continue
        roles = [m["role"] for m in r["prompt"]]
        assert roles == ["user", "assistant", "user", "assistant", "user"]
        last = r["prompt"][-1]["content"]
        assert horizon_of(r) not in last
        assert "FORMAT: I choose: <" in last
        assert f"time horizon: {horizon_of(r)}." in r["prompt"][0]["content"]
        first_label = r["prompt"][0]["content"].rsplit("Option ", 1)[1][0]
        assert first_label == r["labels"][0]


def test_multiturn_evaluations_match_the_options():
    r = next(r for r in build("multiturn_analyst") if r["variant"] == "base")
    for message, label in ((r["prompt"][1], r["labels"][0]), (r["prompt"][3], r["labels"][1])):
        option = next(o for o in r["options"] if o["label"] == label)
        assert f"TIMING: paid in {option['delay_text']}." in message["content"]
        assert f"BENEFIT: {int(option['value']):,} dollars." in message["content"]


def differing_lines(prompt, reference):
    a, b = prompt.split("\n"), reference.split("\n")
    assert len(a) == len(b)
    return {x.split(":")[0] for x, y in zip(a, b) if x != y}


@pytest.mark.parametrize("type_id,changed", [
    ("role_fund_manager", {"SITUATION", "TASK"}),
    ("role_assistant", {"SITUATION", "TASK", "OBJECTIVE"}),
])
def test_role_types_differ_from_t0_only_in_framing_lines(type_id, changed):
    for r in build(type_id):
        if r["variant"] == "twin":
            continue
        shown = labelled_from_record(r)
        t0 = alan_prompt(shown, horizon_text=horizon_of(r))
        assert differing_lines(r["prompt"], t0) == changed


def labelled_from_record(rec):
    class Shown:
        def __init__(self, text):
            self.text = text
    return [(o["label"], Shown(o["text"])) for o in sorted(rec["options"], key=lambda o: o["label"])]
