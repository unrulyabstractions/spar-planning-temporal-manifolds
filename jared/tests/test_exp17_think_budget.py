"""E17 driver on a tiny random-weight Qwen3 and the cached Qwen3.5-0.8B tokenizer: the answer pass
equals E0's, a trace that closes inside the extension is not forced and keeps its head, chunks are
saved once and skipped on a rerun, unit 3 resumes from unit 2's saved ids. Inputs are built here in
the format analysis/think_budget_stage0.py writes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import exp17_think_budget as exp17  # noqa: E402
from spar_horizon import model_io as M  # noqa: E402
from spar_horizon.config import Settings  # noqa: E402
from test_exp14_states import GENERATION, MODEL, cfg, tiny, tok  # noqa: E402,F401

BASE = 5            # think tokens in the synthetic stage 0 head
N = 5
SETTINGS = {**GENERATION, "bank": "choice4", "cache_dir": None}


def prompt_text(i):
    return f"Pick one of the options for plan {i}. Options: a) 10 now, b) 20 later."


def write_inputs(directory, tok, cfg, n=N, regimes=None):
    """stage0_choice4.{json,npz}: n force-closed traces with BASE random think ids each."""
    regimes = regimes or ["near" if i % 2 else "both" for i in range(n)]
    rng = np.random.default_rng(0)
    prompts = [tok.encode(M.chat_text(tok, prompt_text(i), cfg), add_special_tokens=False) for i in range(n)]
    thinks = [rng.integers(1000, 5000, BASE).tolist() for _ in range(n)]
    rows = [{"sample_uid": f"u{i}", "regime": regimes[i], "truth_label": "a", "prompt": prompt_text(i)} for i in range(n)]
    record = {"bank": "choice4", "model": MODEL, "settings": SETTINGS, "think_cap": BASE, "rows": rows}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stage0_choice4.json").write_text(json.dumps(record))
    flat = lambda lists: (np.concatenate(lists).astype(np.int32), np.r_[0, np.cumsum([len(x) for x in lists])])  # noqa: E731
    (p, po), (t, to) = flat(prompts), flat(thinks)
    np.savez(directory / "stage0_choice4.npz", prompt_ids=p, prompt_offsets=po, think_ids=t, think_offsets=to)
    return prompts, thinks


def make_args(directory, out, unit, **kw):
    defaults = dict(bank="choice4", unit=unit, model=MODEL, inputs=str(directory), out=str(out), batch_size=2,
                    chunk_size=2, extra_tokens=3, limit=None, max_chunks=None)
    return argparse.Namespace(**{**defaults, **kw})


@pytest.fixture
def tiny_model(tok, tiny, monkeypatch):
    monkeypatch.setattr(exp17, "load_model", lambda cfg: (tok, tiny, "cpu"))


def test_balanced_order_is_a_permutation_with_balanced_prefixes():
    regimes = ["both"] * 75 + ["near"] * 30
    order = exp17.balanced_order(regimes)
    assert sorted(order) == list(range(105)) and order == exp17.balanced_order(regimes)
    for prefix in (21, 42, 63):
        near = sum(regimes[i] == "near" for i in order[:prefix])
        assert abs(near - prefix * 30 / 105) <= 2


def test_the_answer_pass_equals_e0s_second_stage(tok, tiny, cfg):
    """A forced trace goes through close_and_answer exactly as model_io._generate_rows built it."""
    cfg = Settings(**{**SETTINGS, "max_think_tokens": 6})
    ids = tok.encode(M.chat_text(tok, prompt_text(0), cfg), add_special_tokens=False)
    results = {}
    M._generate_rows(tok, tiny, "cpu", [ids], [0], cfg, results)
    expected = results[0]
    assert expected.forced
    trace = expected.ids[:expected.prompt_len + expected.think_len]
    [got] = exp17.close_and_answer(tok, tiny, "cpu", [len(ids)], [trace], cfg)
    assert got.forced and got.think_len == expected.think_len
    assert (got.prompt_len, got.answer_start, got.pre_len, got.lead) == (
        expected.prompt_len, expected.answer_start, expected.pre_len, expected.lead)
    assert got.ids.tolist() == expected.ids.tolist()


def test_a_trace_that_closes_inside_the_extension_is_natural_and_keeps_its_head(tok, tiny, cfg, monkeypatch):
    tid = M.think_end_id(tok)
    prompt, think = tok.encode("Pick a plan.", add_special_tokens=False), [1001, 1002, 1003]
    head = prompt + think
    answer = tok.encode("I choose: b", add_special_tokens=False)
    calls = []

    def fake_greedy(model, tokenizer, device, id_lists, max_new, min_new):
        calls.append((id_lists, max_new))
        if len(calls) == 1:
            closes = id_lists[0] + [2001, tid] + tok.encode("\n\n", add_special_tokens=False)
            return [torch.tensor(closes), torch.tensor(id_lists[1] + [2001, 2002, 2003])]
        return [torch.tensor(ids + answer) for ids in id_lists]

    monkeypatch.setattr(exp17, "_greedy", fake_greedy)
    traces = exp17.extend_traces(tok, None, "cpu", [head, head], 3)
    gens = exp17.close_and_answer(tok, None, "cpu", [len(prompt)] * 2, traces, cfg)
    natural, forced = gens
    assert calls[0][1] == 3
    assert not natural.forced and natural.think_len == len(think) + 1
    assert forced.forced and forced.think_len == len(think) + 3
    second = calls[1][0]
    assert second[0][:len(head) + 1] == head + [2001]            # the head is unchanged, then the new tokens
    assert second[0][len(head) + 1] == tid
    assert second[1][:len(head) + 3] == head + [2001, 2002, 2003]
    assert second[1][len(head) + 3:len(head) + 3 + len(M.close_ids(tok))] == M.close_ids(tok)
    assert [g.answer_text(tok).strip()[:1] for g in gens] == ["b", "b"]


def test_unit_one_rows_and_chunks_are_saved_and_a_rerun_skips_them(tiny_model, tok, cfg, tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    write_inputs(inputs, tok, cfg)
    args = make_args(inputs, tmp_path / "out", 1)
    run_dir = exp17.run(args)
    chunks = sorted((run_dir / "unit1").glob("chunk_*.json"))
    assert [c.name for c in chunks] == ["chunk_000.json", "chunk_001.json", "chunk_002.json"]
    rows = [r for c in chunks for r in json.loads(c.read_text())["rows"]]
    assert sorted(r["sample_uid"] for r in rows) == [f"u{i}" for i in range(N)]
    assert all(r["forced"] and r["think_len"] == BASE and r["new_think_len"] == 0 and r["unit"] == 1 for r in rows)
    assert all(set(r) == {"unit", "sample_uid", "label", "answer", "correct", "forced", "think_len", "new_think_len"} for r in rows)
    assert not list((run_dir / "unit1").glob("*.npz"))
    monkeypatch.setattr(exp17, "run_chunk", lambda *a: pytest.fail("a finished chunk was rerun"))
    exp17.run(args)


def test_a_stopped_run_resumes_at_its_first_missing_chunk(tiny_model, tok, cfg, tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    write_inputs(inputs, tok, cfg)
    run_dir = exp17.run(make_args(inputs, tmp_path / "out", 2, max_chunks=1))
    assert [p.name for p in (run_dir / "unit2").glob("chunk_*.json")] == ["chunk_000.json"]
    first = (run_dir / "unit2" / "chunk_000.json").read_text()
    ran = []
    original = exp17.run_chunk
    monkeypatch.setattr(exp17, "run_chunk", lambda unit, ctx, n, idx: ran.append(n) or original(unit, ctx, n, idx))
    exp17.run(make_args(inputs, tmp_path / "out", 2))
    assert ran == [1, 2]
    assert (run_dir / "unit2" / "chunk_000.json").read_text() == first


def test_a_chunk_written_for_other_traces_is_an_error(tiny_model, tok, cfg, tmp_path):
    inputs = tmp_path / "inputs"
    write_inputs(inputs, tok, cfg)
    exp17.run(make_args(inputs, tmp_path / "out", 1, max_chunks=1))
    with pytest.raises(ValueError, match="other traces"):
        exp17.run(make_args(inputs, tmp_path / "out", 1, chunk_size=3))


def test_unit_two_saves_head_plus_new_think_ids_and_unit_three_resumes_from_them(tiny_model, tok, cfg, tmp_path):
    inputs = tmp_path / "inputs"
    _, thinks = write_inputs(inputs, tok, cfg)
    run_dir = exp17.run(make_args(inputs, tmp_path / "out", 2, extra_tokens=3))
    order = exp17.balanced_order([r["regime"] for r in json.loads((inputs / "stage0_choice4.json").read_text())["rows"]])
    meta, saved = exp17.read_chunk(run_dir / "unit2", 0)
    first = order[0]
    assert saved[f"u{first}"][:BASE] == thinks[first]             # the head is untouched
    assert len(saved[f"u{first}"]) == BASE + 3
    assert all(r["forced"] and r["new_think_len"] == 3 for r in meta["rows"])

    run_dir = exp17.run(make_args(inputs, tmp_path / "out", 3, extra_tokens=2, batch_size=1))
    meta3, saved3 = exp17.read_chunk(run_dir / "unit3", 0)
    assert saved3[f"u{first}"][:BASE + 3] == saved[f"u{first}"]
    assert all(len(t) == BASE + 3 + 2 for t in saved3.values())
    assert all(r["think_len"] == BASE + 5 and r["new_think_len"] == 5 for r in meta3["rows"])


def test_unit_three_takes_only_the_traces_unit_two_left_open(tmp_path):
    rows = [{"sample_uid": f"u{i}"} for i in range(4)]
    unit2 = tmp_path / "unit2"
    exp17.write_chunk(unit2, 0, 2, ["u0", "u1"], [{"sample_uid": "u0", "forced": False}, {"sample_uid": "u1", "forced": True}],
                      [[1], [2]], 0.0)
    plan = exp17.plan_chunks([0, 1, 2, 3], rows, 3, tmp_path, 2)
    assert plan == [(0, [1])]                                      # chunk 1 of unit 2 is not finished
