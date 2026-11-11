"""Bank names, run tag, record path of answers.json and label parsing for the choice banks."""

import sys

import numpy as np
import pytest

from spar_horizon.banks import build_records, variant_builder
from spar_horizon.behavior import parse_choice
from spar_horizon.config import Settings
from spar_horizon.full_bank import CHOICE_BANKS, choice_variant, render_variant
from pipeline.s1_data_vectors.experiments import common
from pipeline.s1_data_vectors.experiments import exp0_baseline

MODEL = "Qwen/Qwen3.5-0.8B"


@pytest.mark.parametrize("bank,k,roi", [("choice4", 4, False), ("choice8", 8, False),
                                        ("choice4_roi", 4, True), ("choice8_roi", 8, True)])
def test_routing_and_tag(bank, k, roi):
    cfg = Settings(bank=bank, prefill="", choice_prefix="I choose:", limit=6)
    records = build_records(cfg)
    assert 6 <= len(records) <= 12
    assert all(r["n_options"] == k and r["roi_explicit"] is roi for r in records)
    assert variant_builder(cfg) is choice_variant
    assert variant_builder(cfg)(records[0]) == records[0]["prompt"]
    assert cfg.tag == f"Qwen3-8B_think-on_{bank}"


def test_other_banks_keep_their_variant_builder():
    assert variant_builder(Settings(bank="full", prefill="")) is render_variant


def test_unknown_bank_is_rejected():
    with pytest.raises(ValueError):
        Settings(bank="choice5")


@pytest.mark.parametrize("bank", list(CHOICE_BANKS))
def test_cli_treats_choice_banks_as_generated(bank, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--bank", bank])
    _, cfg = common.parse_args("t")
    assert (cfg.bank, cfg.prefill, cfg.choice_prefix) == (bank, "", "I choose:")


def test_parse_choice_with_eight_labels():
    labels = list("abcdefgh")
    assert parse_choice("e) 12,000 dollars", labels) == 4
    assert parse_choice(" **h)** is best", labels) == 7
    assert parse_choice("I prefer c) over f)", labels) == 2
    assert parse_choice("no label", labels) is None
    numbers = [str(i) for i in range(1, 9)]
    assert parse_choice("7) wins", numbers) == 6 and parse_choice("e) x", numbers) is None


def test_letters_a_to_h_are_single_tokens_with_and_without_a_space():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except OSError:
        pytest.skip(f"{MODEL} tokenizer is not in the local HuggingFace cache")
    from spar_horizon.model_io import label_ids
    labels = list("abcdefgh")
    plain = label_ids(tok, labels)
    spaced = label_ids(tok, labels, Settings(model=MODEL, prefill="", choice_prefix="I choose:"))
    assert len(set(plain)) == len(set(spaced)) == 8 and set(plain).isdisjoint(spaced)


def test_answers_json_rows_carry_the_choice_fields():
    cfg = Settings(bank="choice4", prefill="", choice_prefix="I choose:", limit=6)
    records = build_records(cfg)
    answers = [f"{r['labels'][1]}) answer" for r in records]
    rows = exp0_baseline.answer_rows(records, answers, [True] * len(records),
                                     [(False, 100 + i, "think") for i in range(len(records))])
    for row, r in zip(rows, records):
        for key in ("options", "labels", "near_labels", "far_labels", "truth_label", "n_options", "principal",
                    "roi_explicit", "gap_in_half", "pair_id", "duration_id", "rendered_unit", "natural",
                    "delay_scale", "alan_range", "regime", "truth"):
            assert row[key] == r[key]
        assert "near_label" not in row and row["label"] == r["labels"][1] and row["think_len"] >= 100


def test_label_probe_target_is_chose_a_near_half_option():
    records = build_records(Settings(bank="choice4", prefill="", choice_prefix="I choose:", limit=6))
    near_pick = [f"{r['near_labels'][0]}) x" for r in records]
    far_pick = [f"{r['far_labels'][0]}) x" for r in records]
    assert exp0_baseline.label_probe_target(records, near_pick).tolist() == [1] * len(records)
    assert exp0_baseline.label_probe_target(records, far_pick).tolist() == [0] * len(records)
    assert exp0_baseline.label_probe_target(records, [None] + near_pick[1:]).tolist() == [1] * (len(records) - 1)
    ab = [{"regime": "near"}, {"regime": "near"}]
    assert exp0_baseline.label_probe_target(ab, ["a) x", " b) y"]).tolist() == [0, 1]
