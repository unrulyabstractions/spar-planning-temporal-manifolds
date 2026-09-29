from spar_horizon.behavior import only_near_delivers, parse_choice

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
    rec = {"horizon": 1.0, "delays": ("6 months", "10 years")}
    assert only_near_delivers(rec)
    assert not only_near_delivers({**rec, "horizon": 0.25})
    assert not only_near_delivers({**rec, "horizon": 10.0})
    assert not only_near_delivers({**rec, "horizon": None})


def test_eligible_count_is_27():
    from spar_horizon.prompts import build_prompts
    assert sum(only_near_delivers(r) for r in build_prompts()) == 27
