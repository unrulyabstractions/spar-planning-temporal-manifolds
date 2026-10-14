from spar_horizon.behavior import near_index, only_near_delivers, parse_choice

# Strings the Qwen3.5-0.8B model actually emitted (2026-09-21).
QWEN_A = "a)<|im_end|>\n<|endoftext|>"
QWEN_2 = "2)<|im_end|>\n<|endoftext|>"


def test_parse_real_outputs():
    assert parse_choice(QWEN_A) == 0
    assert parse_choice(QWEN_2, labels=("1", "2")) == 1
    assert parse_choice(QWEN_A, labels=("1", "2")) is None


def test_parse_first_label_wins():
    assert parse_choice("I choose: b) rather than a)") == 1
    assert parse_choice("Both a) and b) are fine") == 0


def test_parse_empty_and_missing():
    assert parse_choice("") is None
    assert parse_choice(None) is None
    assert parse_choice("I cannot decide.") is None


def test_only_near_delivers():
    rec = {"horizon": 1.0, "near_delay_years": 0.5, "far_delay_years": 10.0}
    assert only_near_delivers(rec)
    assert not only_near_delivers({**rec, "horizon": 0.25})
    assert not only_near_delivers({**rec, "horizon": 10.0})
    assert not only_near_delivers({**rec, "horizon": None})


def test_near_index_defaults_to_first_option():
    assert near_index({"horizon": 1.0}) == 0
    assert near_index({"near_label": "a"}) == 0 and near_index({"near_label": "b"}) == 1


def test_eligible_count_is_27():
    from spar_horizon.prompts import build_prompts
    assert sum(only_near_delivers(r) for r in build_prompts()) == 27


def _rec(horizon, near_label, regime=None, truth=None):
    return {"horizon": horizon, "near_label": near_label, "regime": regime, "truth": truth}


def test_base_behavior_position_bias_and_accuracy():
    from spar_horizon.behavior import base_behavior
    # a model that always answers a): picks near when near is a), far when near is b)
    recs = [_rec(1.0, "a", "near", "near"), _rec(1.0, "b", "near", "near"),
            _rec(50.0, "a", "both", "far"), _rec(50.0, "b", "both", "far"), _rec(None, "a", "null")]
    out = base_behavior(recs, [" a) x"] * 5)
    assert out["position_bias"] == 1.0
    assert out["accuracy"] == {"near": [1, 2], "both": [1, 2]}
    by_h = {row["horizon"]: row for row in out["by_horizon"]}
    assert list(by_h) == [1.0, 50.0, None]
    assert by_h[1.0]["near_rate_near_a"] == 1.0 and by_h[1.0]["near_rate_near_b"] == 0.0
    assert by_h[None]["near_rate_near_b"] is None


def test_base_behavior_unbiased_and_unparsed():
    from spar_horizon.behavior import base_behavior
    # a model that always picks the near option, whatever its label
    recs = [_rec(1.0, "a", "near", "near"), _rec(1.0, "b", "near", "near"), _rec(2.0, "a", "near", "near")]
    out = base_behavior(recs, [" a) x", " b) y", None])
    assert out["position_bias"] == 0.0 and out["accuracy"] == {"near": [2, 2]}
    assert [row["horizon"] for row in out["by_horizon"]] == [1.0]


def test_base_behavior_starter_bank_has_no_bias_estimate():
    from spar_horizon.behavior import base_behavior
    from spar_horizon.prompts import build_prompts
    recs = build_prompts(limit=6)
    out = base_behavior(recs, [" a) x"] * len(recs))
    assert out["position_bias"] is None and out["accuracy"] == {}
