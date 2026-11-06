"""Chunked choice-bank E0: prompt-side positions on the real 0.8B tokenizer, the extra `prompt`
region and the number-token mean on a tiny random-weight Qwen3, chunk write / skip / atomicity,
and assembling from partial chunks. No model weights."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import exp0_choice as choice  # noqa: E402
from spar_horizon import extract as extract_module  # noqa: E402
from spar_horizon.config import Settings  # noqa: E402
from spar_horizon.extract import extract, prompt_side_positions  # noqa: E402
from spar_horizon.full_bank import build_choice_prompts  # noqa: E402
from spar_horizon.model_io import chat_text  # noqa: E402
from test_exp14_states import MODEL, tiny, tok  # noqa: E402,F401

CHUNK = 3


@pytest.fixture(scope="module")
def cfg():
    return Settings(model=MODEL, thinking="on", prefill="", choice_prefix="I choose:", bank="choice4",
                    max_think_tokens=3, n_response=6, answer_window=8)


@pytest.fixture(scope="module")
def records():
    return build_choice_prompts(4, limit=7)[:7]


def decoded(tok, cfg, rec, indices):
    ids = tok.encode(chat_text(tok, rec["prompt"], cfg), add_special_tokens=False)
    return [tok.decode([ids[i]]) for i in indices], ids


@pytest.mark.parametrize("k", [4, 8])
def test_prompt_positions_decode_to_the_expected_characters_inside_their_own_line(tok, cfg, k):
    rec = build_choice_prompts(k, limit=3)[2]
    side = prompt_side_positions(tok, rec["prompt"], rec, cfg)
    assert len(side["labels"]) == len(side["ends"]) == k and side["order"] == list(range(k))
    assert decoded(tok, cfg, rec, side["labels"])[0] == list(rec["labels"])
    assert decoded(tok, cfg, rec, side["ends"])[0] == ["."] * k
    assert "".join(decoded(tok, cfg, rec, side["number"] + [side["unit"]])[0]).strip() == rec["horizon_text"]
    assert decoded(tok, cfg, rec, [side["unit"]])[0][0].strip().isalpha()
    spans = tok(chat_text(tok, rec["prompt"], cfg), add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    text = chat_text(tok, rec["prompt"], cfg)
    for j, (lab, end) in enumerate(zip(side["labels"], side["ends"])):
        line = text[spans[lab][0]:text.index("\n", spans[lab][0])]
        assert line.startswith(f"{rec['labels'][j]}) ") and spans[lab][1] <= spans[end][0] < spans[end][1] <= len(line) + spans[lab][0]
        assert lab < end < (side["labels"][j + 1] if j + 1 < k else len(spans))


def test_option_order_follows_the_lines_when_the_record_lists_options_in_another_order(tok, cfg):
    rec = build_choice_prompts(4, limit=3)[0]
    shuffled = {**rec, "options": rec["options"][::-1]}
    assert prompt_side_positions(tok, rec["prompt"], shuffled, cfg)["order"] == [3, 2, 1, 0]


@pytest.fixture(scope="module")
def extracted(tok, tiny, cfg, records):
    return extract(tok, tiny, "cpu", [r["prompt"] for r in records[:3]], cfg, layers=(10, 14), records=records[:3],
                   prompt_positions=True)


def test_prompt_region_is_appended_after_the_answer_and_holds_two_plus_two_k_positions(extracted):
    ext = extracted
    start, end = ext.regions["prompt"]
    assert start == ext.regions["answer"][1] and end - start == 2 + 2 * 4
    assert ext.activations[0].shape == (ext.kept.sum(), end, 32) and ext.option_order.shape == (ext.kept.sum(), 4)
    assert ext.tokens[start:end][:4] == ["phrase number (mean)", "phrase unit", "option 1 label", "option 1 end"]
    assert ext.keep_idx.shape[1] == end


def test_number_slot_is_the_mean_of_the_number_tokens_and_other_slots_are_single_tokens(extracted, tok, tiny, cfg, records):
    ext = extracted
    start = ext.regions["prompt"][0]
    g, rec = ext.generations[0], records[0]
    side = prompt_side_positions(tok, rec["prompt"], rec, cfg)
    with torch.no_grad():
        h = tiny(input_ids=g.ids[None], output_hidden_states=True).hidden_states[10][0]
    np.testing.assert_allclose(ext.activations[0][0, start], h[side["number"]].mean(0).numpy(), atol=1e-5)
    np.testing.assert_allclose(ext.activations[0][0, start + 1], h[side["unit"]].numpy(), atol=1e-5)
    np.testing.assert_allclose(ext.activations[0][0, start + 2 + 2 * 3 + 1], h[side["ends"][3]].numpy(), atol=1e-5)
    assert ext.keep_idx[0, start + 1] == side["unit"] and ext.keep_idx[0, start] == side["number"][-1]


def test_default_extraction_has_no_prompt_region(tok, tiny, cfg, records):
    ext = extract(tok, tiny, "cpu", [r["prompt"] for r in records[:2]], cfg, layers=(10,), records=records[:2])
    assert "prompt" not in ext.regions and ext.option_order is None
    assert ext.activations[0].shape[1] == ext.regions["answer"][1]


@pytest.fixture
def tiny_model(tok, tiny, monkeypatch):
    monkeypatch.setattr(choice, "load_model", lambda cfg: (tok, tiny, "cpu"))


def test_chunks_are_written_with_both_files_and_layers_beyond_the_model_are_dropped(tiny_model, cfg, records, tmp_path):
    choice.run_chunks(cfg, tmp_path, records, CHUNK, (10, 14, 30))
    chunk_dir = tmp_path / "chunks"
    assert choice.finished_chunks(chunk_dir) == [0, 1, 2]
    z = np.load(chunk_dir / "chunk_000.npz")
    assert z["layers"].tolist() == [10, 14] and {"L10", "L14"} <= set(z.files) and "L30" not in z.files
    assert z["option_order"].shape[1] == 4 and z["label_logits"].shape[1] == 4
    assert set(z.files) >= {"region_prompt", "keep_idx", "keep_ids", "forced_mask", "think_len", "lead", "kept", "tokens"}
    meta = json.loads((chunk_dir / "chunk_000.json").read_text())
    assert meta["sample_uids"] == [r["sample_uid"] for r in records[:CHUNK]]
    kept_rows = [r for r in meta["rows"] if "label" in r]
    assert len(kept_rows) == int(z["kept"].sum()) == len(z["label_logits"])
    assert {"options", "labels", "truth_label", "regime", "pair_id", "duration_id", "rendered_unit",
            "think_text", "forced"} <= set(kept_rows[0])
    assert not list(chunk_dir.glob("*.tmp"))


def test_a_rerun_skips_finished_chunks_and_a_killed_run_loses_only_the_chunk_in_flight(tiny_model, cfg, records, tmp_path,
                                                                                    monkeypatch):
    real, calls = extract_module.extract, []

    def dies_on_second(*args, **kwargs):
        calls.append(len(args[3]))
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(choice, "extract", dies_on_second)
    with pytest.raises(KeyboardInterrupt):
        choice.run_chunks(cfg, tmp_path, records, CHUNK, (10,))
    assert choice.finished_chunks(tmp_path / "chunks") == [0]
    calls.clear()
    monkeypatch.setattr(choice, "extract", lambda *a, **k: calls.append(len(a[3])) or real(*a, **k))
    choice.run_chunks(cfg, tmp_path, records, CHUNK, (10,))
    assert calls == [CHUNK, 1] and choice.finished_chunks(tmp_path / "chunks") == [0, 1, 2]
    monkeypatch.setattr(choice, "extract", lambda *a, **k: pytest.fail("extract ran for a finished chunk"))
    choice.run_chunks(cfg, tmp_path, records, CHUNK, (10,))


def test_a_write_that_dies_midway_leaves_no_chunk_behind(tiny_model, cfg, records, tmp_path, monkeypatch):
    def dies(f, **arrays):
        f.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(choice.np, "savez", dies)
    with pytest.raises(OSError):
        choice.run_chunks(cfg, tmp_path, records[:CHUNK], CHUNK, (10,))
    chunk_dir = tmp_path / "chunks"
    assert choice.finished_chunks(chunk_dir) == [] and not list(chunk_dir.glob("chunk_*.npz"))


def test_chunks_written_for_other_prompts_are_refused(tiny_model, cfg, records, tmp_path):
    choice.run_chunks(cfg, tmp_path, records[:CHUNK], CHUNK, (10,))
    with pytest.raises(ValueError, match="other prompts"):
        choice.run_chunks(cfg, tmp_path, records[1:], CHUNK, (10,))


def test_assemble_joins_whatever_chunks_exist(tiny_model, cfg, records, tmp_path, monkeypatch):
    real = extract_module.extract
    calls = []

    def stops_after_first(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(choice, "extract", stops_after_first)
    with pytest.raises(KeyboardInterrupt):
        choice.run_chunks(cfg, tmp_path, records, CHUNK, (10,))
    choice.assemble(cfg, tmp_path, len(records), CHUNK)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    rows = json.loads((tmp_path / "answers.json").read_text())
    assert manifest["n_chunks_done"] == 1 and manifest["n_chunks_total"] == 3 and len(rows) == CHUNK
    assert manifest["n_kept"] == sum("label" in r for r in rows) and 0 <= manifest["forced_rate"] <= 1
    assert manifest["multichoice_behavior"]["all"]["n"] <= manifest["n_kept"]


def test_assemble_with_no_chunks_writes_an_empty_answers_file(cfg, tmp_path):
    choice.assemble(cfg, tmp_path, 7, CHUNK)
    assert json.loads((tmp_path / "answers.json").read_text()) == []
    assert json.loads((tmp_path / "manifest.json").read_text())["n_chunks_done"] == 0
