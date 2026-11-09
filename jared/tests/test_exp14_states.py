"""E14 phrase states: phrase roles on real bank prompts (the cached Qwen3.5-0.8B tokenizer),
row order, keys and file handling on a tiny random-weight Qwen3. No model weights."""

import argparse
import json

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from pipeline.s1_data_vectors.experiments import exp14_phrase_states as states
from spar_horizon import steering as S
from spar_horizon.config import Settings
from spar_horizon.full_bank import build_full_prompts

MODEL = "Qwen/Qwen3.5-0.8B"
GENERATION = {"model": MODEL, "thinking": "on", "n_response": 6, "max_think_tokens": 3, "prefill": "",
              "choice_prefix": "I choose:", "answer_window": 8}
BANK_SETTINGS = {"full": {"n_prompts": 60, "seed": 0, "limit": None}}


@pytest.fixture(scope="module")
def tok():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "left"
    return tokenizer


@pytest.fixture(scope="module")
def tiny(tok):
    torch.manual_seed(0)
    config = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=19, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=len(tok), head_dim=8)
    return Qwen3ForCausalLM(config).eval()


@pytest.fixture(scope="module")
def cfg():
    return Settings(**GENERATION)


def bank_records():
    return build_full_prompts(n=60)


def write_prompts_file(path, n_rows):
    """A phrase prompts file (interfaces §1) over the first `n_rows` bank records."""
    records = bank_records()[:n_rows]
    meta = {"model": MODEL, "generation": GENERATION, "bank_settings": BANK_SETTINGS,
            "selected": list(range(n_rows)), "selected_banks": ["full"] * n_rows,
            "sample_uids": [r["sample_uid"] for r in records]}
    path.write_text(json.dumps(meta))
    return records


def test_phrase_roles_are_number_tokens_then_the_unit_token(tok, cfg):
    for record in bank_records()[:25]:
        positions = states.record_phrase(tok, cfg, record)
        ids = S.encode(tok, record["prompt"], cfg)
        number_text = "".join(tok.decode([ids[p]]) for p in positions[:-1])
        unit_text = tok.decode([ids[positions[-1]]])
        assert number_text.strip() == record["horizon_text"].split()[0]
        assert unit_text.strip() == record["horizon_text"].split()[1]


def test_a_one_token_phrase_is_an_error(tok, cfg):
    record = {"prompt": "Choose for this time horizon: years.", "horizon_text": "years", "sample_uid": "x"}
    with pytest.raises(ValueError, match="at least a number and a unit"):
        states.record_phrase(tok, cfg, record)


def test_states_follow_record_order_with_the_expected_keys(tok, tiny, cfg):
    records = bank_records()[:5]
    batched = states.phrase_states(tiny, "cpu", tok, cfg, records, batch_size=3)
    keys = {f"L{layer}|{role}" for layer in (12, 14, 16, 18) for role in ("number", "unit")} | {"n_number"}
    assert set(batched) == keys
    assert all(batched[k].shape == (5, 32) and batched[k].dtype == np.float32 for k in keys - {"n_number"})
    positions = [states.record_phrase(tok, cfg, r) for r in records]
    assert batched["n_number"].tolist() == [len(p) - 1 for p in positions]
    # batching and the length sort must not change any row
    for i, record in enumerate(records):
        alone = states.phrase_states(tiny, "cpu", tok, cfg, [record], batch_size=1)
        for key in keys:
            np.testing.assert_allclose(batched[key][i], alone[key][0], atol=1e-4)
    # the unit row is the model's layer-16 state at the last phrase token; the number row the mean of the others
    ids = S.encode(tok, records[2]["prompt"], cfg)
    _, hidden, _ = S.forward(tiny, "cpu", [ids], tok.pad_token_id, hidden=True)
    state = hidden[16][0]
    np.testing.assert_allclose(batched["L16|unit"][2], state[positions[2][-1]].numpy(), atol=1e-4)
    np.testing.assert_allclose(batched["L16|number"][2], state[positions[2][:-1]].mean(0).numpy(), atol=1e-4)


def test_a_model_without_layer_18_is_rejected(tok, cfg):
    shallow = Qwen3ForCausalLM(Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                           num_attention_heads=4, num_key_value_heads=2, vocab_size=len(tok),
                                           head_dim=8)).eval()
    with pytest.raises(ValueError, match="need layer 18"):
        states.phrase_states(shallow, "cpu", tok, cfg, bank_records()[:2], batch_size=2)


def test_main_writes_states_under_the_model_folder_and_skips_when_present(tok, tiny, tmp_path, monkeypatch):
    prompts_file = tmp_path / "phrase_prompts.json"
    records = write_prompts_file(prompts_file, 6)
    monkeypatch.setattr(states, "load_model", lambda cfg: (tok, tiny, "cpu"))
    parser_args = argparse.Namespace(prompts=str(prompts_file), model=None, layers=None, limit=4, batch_size=2,
                                     out=str(tmp_path / "out"))
    states.main(parser_args)
    out_dir = tmp_path / "out" / "exp14_Qwen3.5-0.8B_phrase"
    meta = json.loads((out_dir / "states.json").read_text())
    assert meta["model"] == MODEL and meta["layers"] == [12, 14, 16, 18] and meta["n"] == 4
    z = np.load(out_dir / "states.npz")
    assert z["sample_uids"].tolist() == [r["sample_uid"] for r in records[:4]]
    assert z["L14|number"].shape == (4, 32)

    monkeypatch.setattr(states, "load_model", lambda cfg: pytest.fail("loaded the model although states.npz exists"))
    states.main(parser_args)


def test_model_override_names_the_output_folder(tok, tiny, tmp_path, monkeypatch):
    prompts_file = tmp_path / "phrase_prompts.json"
    write_prompts_file(prompts_file, 2)
    meta = json.loads(prompts_file.read_text())
    meta["model"] = "Qwen/Qwen3-8B"
    prompts_file.write_text(json.dumps(meta))
    seen = []
    monkeypatch.setattr(states, "load_model", lambda cfg: seen.append(cfg.model) or (tok, tiny, "cpu"))
    parser_args = argparse.Namespace(prompts=str(prompts_file), model=MODEL, layers=None, limit=None, batch_size=2,
                                     out=str(tmp_path / "out"))
    states.main(parser_args)
    assert seen == [MODEL]
    assert json.loads((tmp_path / "out" / "exp14_Qwen3.5-0.8B_phrase" / "states.json").read_text())["model"] == MODEL


def test_other_layers_get_their_own_file_with_the_same_keys_and_their_own_skip_rule(tok, tiny, tmp_path, monkeypatch):
    prompts_file = tmp_path / "phrase_prompts.json"
    write_prompts_file(prompts_file, 4)
    monkeypatch.setattr(states, "load_model", lambda cfg: (tok, tiny, "cpu"))
    out_dir = tmp_path / "out" / "exp14_Qwen3.5-0.8B_phrase"

    def run(layers):
        states.main(argparse.Namespace(prompts=str(prompts_file), model=None, layers=layers, limit=None,
                                       batch_size=2, out=str(tmp_path / "out")))

    run(None)
    run("10,14")
    assert states.states_name((10, 14)) == "states_L10-14" and states.states_name(states.LAYERS) == "states"
    assert json.loads((out_dir / "states_L10-14.json").read_text())["layers"] == [10, 14]
    default, other = np.load(out_dir / "states.npz"), np.load(out_dir / "states_L10-14.npz")
    assert {k for k in other.files if "|" in k} == {f"L{layer}|{role}" for layer in (10, 14) for role in ("number", "unit")}
    assert default["sample_uids"].tolist() == other["sample_uids"].tolist()
    # the layer both files hold is the same state, whichever run extracted it
    np.testing.assert_allclose(default["L14|unit"], other["L14|unit"], atol=1e-4)
    assert not np.allclose(other["L10|unit"], other["L14|unit"])

    monkeypatch.setattr(states, "load_model", lambda cfg: pytest.fail("loaded the model although the file exists"))
    run("10,14")
    run(None)
