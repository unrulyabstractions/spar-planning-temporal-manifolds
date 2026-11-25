"""E21 thinking-off runner, no real model: position resolution under left padding (0.8B tokenizer),
phrase_end on cues in system, assistant and message-end positions, E19's prefill text, the stated-horizon
parser, and an end-to-end run on a tiny random Qwen3 whose files the boundary analysis reads."""

import dataclasses
import json
import warnings

import numpy as np
import pytest
import torch

from pipeline.s1_data_vectors.analysis.evaltypes_steering_prepare import off_sequence
from pipeline.s1_data_vectors.experiments import exp21_boundary_off as runner
from pipeline.s3_analysis.eval_types_boundary import analyse_type, load_run, position_index, type_activations
from spar_horizon.config import Settings
from spar_horizon.eval_types import build
from test_exp14_states import MODEL, tiny, tok  # noqa: F401

OFF = Settings(thinking="off", prefill="I choose:")


def first_record(type_id, variant="base"):
    return next(r for r in build(type_id) if r["variant"] == variant)


def resolve(tok, record, prefill="I choose:"):
    ids, offsets, text = runner.encode(tok, record, prefill)
    return ids, runner.record_positions(tok, record, ids, offsets, text)


def decode_at(tok, ids, positions):
    return [tok.decode([ids[p]]) if p >= 0 else None for p in positions]


@pytest.mark.parametrize("type_id", ["t0_w0", "mt_first", "persona_model"])
def test_header_and_prelabel_positions(tok, type_id):
    record = first_record(type_id)
    ids, positions = resolve(tok, record)
    assert decode_at(tok, ids, positions)[:6] == [*runner.HEADER, ":"]
    assert positions[5] == len(ids) - 1
    assert positions[:5] == list(range(positions[0], positions[0] + 5))
    tokens = list(runner.TOKEN_NAMES)
    assert position_index(tokens, "suffix_end") == 4
    assert position_index(tokens, "prelabel") == 5
    assert position_index(tokens, "phrase_end") == 6


def test_thinking_off_text_is_e19s(tok):
    record = first_record("mt_first")
    ids, _, text = runner.encode(tok, record, "I choose:")
    assert ids == off_sequence(tok, record, OFF)
    assert text.endswith("<think>\n\n</think>\n\nI choose:")


def cue_token(tok, record):
    ids, positions = resolve(tok, record)
    return tok.decode([ids[positions[6]]]), positions[6], ids


@pytest.mark.parametrize("type_id", ["persona_model", "persona_user", "mt_assistant", "mt_first", "mt_long", "no_horizon"])
def test_phrase_end_is_the_cues_last_token(tok, type_id):
    for record in build(type_id)[:6]:
        if not record.get("cue_text"):
            assert resolve(tok, record)[1][6] == -1
            continue
        token, _, _ = cue_token(tok, record)
        assert record["cue_text"].rstrip().endswith(token.strip()), (record["sample_uid"], token)


def test_cue_in_system_message_and_assistant_turn_are_in_those_messages(tok):
    system = first_record("persona_model")
    assert system["cue_message"] == 0 and system["prompt"][0]["role"] == "system"
    assistant = first_record("mt_assistant")
    assert assistant["prompt"][assistant["cue_message"]]["role"] == "assistant"
    for record in (system, assistant):
        ids, positions = resolve(tok, record)
        before = tok.decode(ids[:positions[6] + 1])
        assert before.rstrip().endswith(record["cue_text"].rstrip()[-3:])


def test_phrase_end_at_a_message_end(tok):
    record = {"sample_uid": "x", "prompt": [{"role": "user", "content": "Plan for 3 weeks."},
                                            {"role": "assistant", "content": "Understood."},
                                            {"role": "user", "content": "Pick one."}],
              "cue_text": "Plan for 3 weeks.", "cue_message": 0}
    ids, offsets, text = runner.encode(tok, record, "I choose:")
    positions = runner.record_positions(tok, record, ids, offsets, text)
    assert tok.decode([ids[positions[6]]]) == "."
    assert tok.decode(ids[positions[6] - 1:positions[6] + 2]).endswith(".<|im_end|>")


def test_cue_in_a_repeated_message_uses_its_own_message(tok):
    record = {"sample_uid": "x", "prompt": [{"role": "user", "content": "Pick one."},
                                            {"role": "assistant", "content": "Which?"},
                                            {"role": "user", "content": "Pick one."}],
              "cue_text": "Pick one.", "cue_message": 2}
    ids, offsets, text = runner.encode(tok, record, "I choose:")
    last = runner.cue_last_char(text, record)
    assert last == text.rindex("Pick one.") + len("Pick one.") - 1


@pytest.mark.parametrize("text, years", [
    (" about 5 years.", 5.0), (" three months", 0.25), (" a year", 1.0), (" two decades or so", 20.0),
    (" 6 to 12 months", 1.0), (" the next fifty-nine years", 59.0), (" unclear", None), (" several years", None),
    (" 10 days (roughly)", 10 / 365.25)])
def test_parse_years(text, years):
    got = runner.parse_years(text)
    assert (got is None) == (years is None)
    if years is not None:
        assert got == pytest.approx(years, rel=1e-3)


def chunk_item(tok, record, labels_ids):
    ids, positions = resolve(tok, record)
    return ids, positions, labels_ids


def test_left_padded_batch_equals_single_rows(tok, tiny):
    records = [first_record("t0_w0"), first_record("mt_long"), first_record("persona_model"), first_record("no_horizon")]
    label_ids = (tok.encode(" a", add_special_tokens=False)[0], tok.encode(" b", add_special_tokens=False)[0])
    items = [chunk_item(tok, r, label_ids) for r in records]
    assert len({len(ids) for ids, _, _ in items}) == len(items)   # four different lengths: real padding
    layers = (10, 14)
    batched = runner.forward_chunk(tiny, "cpu", tok.pad_token_id, items, layers)
    for item, (acts, logits) in zip(items, batched):
        (single_acts, single_logits), = runner.forward_chunk(tiny, "cpu", tok.pad_token_id, [item], layers)
        np.testing.assert_allclose(acts, single_acts, atol=2e-2, equal_nan=True)
        np.testing.assert_allclose(logits, single_logits, atol=1e-3)
    assert np.isnan(batched[3][0][:, 6]).all() and np.isfinite(batched[3][0][:, :6]).all()
    assert np.isfinite(batched[2][0]).all()


def test_batch_steps_down_after_out_of_memory():
    calls = []

    def run(chunk):
        calls.append(len(chunk))
        if len(chunk) > 2:
            raise torch.OutOfMemoryError("x")
        return list(chunk)

    assert runner.in_batches(list(range(7)), 6, "cpu", run) == list(range(7))
    assert calls[0] == 6 and max(calls[1:]) <= 4


@pytest.fixture
def setup(tok, tiny, monkeypatch):
    monkeypatch.setattr(runner, "load_model", lambda cfg: (tok, tiny, "cpu"))
    return dataclasses.replace(Settings(model=MODEL, thinking="off", prefill="I choose:", bank="full"), batch_size=3)


def test_end_to_end_on_a_tiny_model(setup, tmp_path):
    type_ids = ["persona_model", "t0_w0", "free_form", "roi_unequal"]
    runner.run(setup, type_ids, (10, 14, 99), tmp_path, limit=4)
    settings = json.loads((tmp_path / "settings.json").read_text())
    assert settings["thinking"] == "off" and settings["layers"] == [10, 14] and settings["code_hash"]
    assert not (tmp_path / "free_form").exists() and not (tmp_path / "roi_unequal").exists()

    types, tokens, mode = load_run(tmp_path)
    assert mode == "off" and set(types) == {"persona_model", "t0_w0"}
    assert [position_index(tokens, p) for p in ("suffix_end", "prelabel", "phrase_end")] == [4, 5, 6]
    persona = types["persona_model"]
    assert len(persona) == 4
    assert all(r["stated_horizon_text"] is not None and "stated_horizon_years" in r for r in persona)
    assert not any("stated_horizon_text" in r for r in types["t0_w0"])
    assert all(r["phrase_end_token"] is not None for r in persona)

    for type_id, rows in types.items():
        X = type_activations(tmp_path / type_id, rows, 14)
        assert X.shape == (4, 7, 32)
        assert np.isfinite(X[:, :6]).all()
        assert np.isfinite(X[:, 6]).all() == (type_id == "persona_model")   # t0_w0 has no cue
        z = np.load(tmp_path / type_id / "acts_0.npz")
        assert z["L10"].dtype == np.float16 and z["label_logits"].shape == (4, 2)
        np.testing.assert_allclose(z["label_logits"][1], rows[1]["label_logits"], rtol=1e-6)

    rows = types["t0_w0"]
    X = type_activations(tmp_path / "t0_w0", rows, 14)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record, _ = analyse_type(rows, X[:, 5], None, {}, 5, 5, np.random.default_rng(0))
    assert record["n_rows"] == 4

    again = (tmp_path / "persona_model" / "rows.json").stat().st_mtime_ns
    runner.run(setup, ["persona_model"], (10, 14), tmp_path, limit=4)
    assert (tmp_path / "persona_model" / "rows.json").stat().st_mtime_ns == again
