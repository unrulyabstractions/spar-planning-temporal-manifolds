"""Part B scan driver on a tiny random-weight Qwen3 and the cached Qwen3.5-0.8B tokenizer: unit
order and skipping, the prefill edit at each position, the answer-slot cells re-scoring the
baseline's reasoning without generating, claim and resume. Inputs are built here in the format
analysis/scan_prepare.py writes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import exp15_scan as scan  # noqa: E402
from exp14_phrase_states import record_phrase  # noqa: E402
from spar_horizon import steering as S  # noqa: E402
from spar_horizon.model_io import label_ids  # noqa: E402
from test_exp14_states import BANK_SETTINGS, GENERATION, MODEL, bank_records, cfg, tiny, tok  # noqa: E402,F401

D = 32
N = 4
LAYERS = (18, 14, 10)           # the tiny model has 19 layers; L14 carries zero steps
POSITIONS = ("number", "unit", "suffix", "answer")
DIRECTIONS = ("manifold", "random")


def write_inputs(out_dir):
    """inputs.json and inputs.npz for four bank prompts. Steps are random except at L14, where
    every step is zero (a zero step must reproduce the unedited run)."""
    records = bank_records()[:N]
    prompts = [{"transition": ("none-near", "both-near")[i % 2], "start_regime": ("none", "both")[i % 2],
                "target_regime": "near", "log_h": float(np.log10(r["horizon"])), "target_log_h": 0.5}
               for i, r in enumerate(records)]
    meta = {"model": MODEL, "generation": GENERATION, "bank_settings": BANK_SETTINGS,
            "selected": list(range(N)), "selected_banks": ["full"] * N,
            "sample_uids": [r["sample_uid"] for r in records], "prompts": prompts,
            "transitions": ["none-near", "both-near"], "layers": list(LAYERS), "positions": list(POSITIONS),
            "directions": list(DIRECTIONS)}
    rng = np.random.default_rng(0)
    steps = {f"step|L{layer}|{direction}|{position}": (0 if layer == 14 else 3.0) * rng.normal(size=(N, D)).astype(np.float32)
             for layer in LAYERS for direction in DIRECTIONS for position in POSITIONS}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inputs.json").write_text(json.dumps(meta))
    np.savez(out_dir / "inputs.npz", **steps)
    return records, steps


def run_args(out_dir, only=None, limit=None):
    return argparse.Namespace(dir=str(out_dir), limit=limit, max_think_tokens=2, batch_size=3, score_batch_size=2,
                              only=only)


def load(out_dir, name):
    return json.loads((out_dir / "conditions" / f"{name.replace(' ', '_')}.json").read_text())


@pytest.fixture(scope="module")
def full_run(tok, tiny, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("scan") / "exp15_scan"
    write_inputs(out_dir)
    order = []
    original = scan.run_claimed
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(scan, "load_model", lambda cfg: (tok, tiny, "cpu"))
        patcher.setattr(scan, "run_claimed", lambda d, name, run: order.append(name) or original(d, name, run))
        scan.main(run_args(out_dir))
    return out_dir, order


@pytest.fixture
def tiny_model(tok, tiny, monkeypatch):
    monkeypatch.setattr(scan, "load_model", lambda cfg: (tok, tiny, "cpu"))


def test_units_run_baseline_then_reruns_then_each_layers_pairs_back_to_back():
    layers, positions, directions = (18, 14, 22, 10, 26), POSITIONS, DIRECTIONS
    units = scan.planned_units(layers, positions, directions)
    names = [u[0] for u in units]
    assert names[:5] == ["baseline", "noop 1", "noop 2", "L18 number manifold", "L18 number random"]
    assert len(units) == 3 + 40 and len(set(names)) == len(names)
    cells = names[3:]
    assert cells[:8] == [f"L18 {p} {d}" for p in positions for d in directions]
    assert [c.split()[0] for c in cells[::8]] == ["L18", "L14", "L22", "L10", "L26"]
    assert all(cells[i].rsplit(" ", 1)[1] == "manifold" and cells[i + 1].rsplit(" ", 1)[1] == "random"
               and cells[i].rsplit(" ", 1)[0] == cells[i + 1].rsplit(" ", 1)[0] for i in range(0, 40, 2))
    assert units[0][1:] == ("baseline", None, None, None) and units[3][1:] == ("cell", 18, "number", "manifold")


def test_reruns_use_another_prompt_order_and_smaller_batches_than_the_baseline():
    for name, expected_batch in (("noop 1", 5), ("noop 2", 4)):
        order, batch = scan.rerun_batching(name, 10)
        assert sorted(order.tolist()) == list(range(10)) and order.tolist() != list(range(10))
        assert batch == expected_batch
    assert scan.rerun_batching("noop 1", 10)[0].tolist() != scan.rerun_batching("noop 2", 10)[0].tolist()


def test_every_unit_is_written_in_run_order_with_json_and_npz(full_run):
    out_dir, order = full_run
    assert order == [u[0] for u in scan.planned_units(LAYERS, POSITIONS, DIRECTIONS)]
    for name in order:
        assert (out_dir / "conditions" / f"{name.replace(' ', '_')}.json").exists()
        assert (out_dir / "conditions" / f"{name.replace(' ', '_')}.npz").exists()
    assert not [p for p in (out_dir / "conditions").iterdir() if p.suffix == ".lock" and p.is_file()]


def test_row_formats(full_run):
    out_dir, _ = full_run
    keys = {"sample_uid", "bank", "transition", "near_label", "log_h", "target_log_h", "near_lo", "label"}
    generated = load(out_dir, "L18 suffix manifold")
    assert {"name", "kind", "layer", "position", "direction", "rows", "seconds"} <= set(generated)
    assert generated["layer"] == 18 and generated["position"] == "suffix" and generated["direction"] == "manifold"
    assert all(set(r) == keys | {"forced", "think_len", "think_text"} for r in generated["rows"])
    answer = load(out_dir, "L18 answer manifold")
    assert all(set(r) == keys | {"ref_near_lo", "ref_label"} for r in answer["rows"])
    assert load(out_dir, "baseline")["layer"] is None
    assert {"ids", "offsets", "answer_start", "prompt_len"} <= set(np.load(out_dir / "conditions" / "L18_number_random.npz").files)


def test_a_zero_step_reproduces_the_baseline_at_every_position(full_run):
    out_dir, _ = full_run
    base = load(out_dir, "baseline")["rows"]
    for position in ("number", "unit", "suffix"):
        for direction in DIRECTIONS:
            rows = load(out_dir, f"L14 {position} {direction}")["rows"]
            assert [r["think_text"] for r in rows] == [r["think_text"] for r in base]
            np.testing.assert_allclose([r["near_lo"] for r in rows], [r["near_lo"] for r in base], atol=1e-4)
    for direction in DIRECTIONS:
        rows = load(out_dir, f"L14 answer {direction}")["rows"]
        assert [r["near_lo"] for r in rows] == [r["ref_near_lo"] for r in rows]
        assert [r["label"] for r in rows] == [r["ref_label"] for r in rows]


def test_a_nonzero_step_moves_the_scores_at_every_position(full_run):
    out_dir, _ = full_run
    base = np.array([r["near_lo"] for r in load(out_dir, "baseline")["rows"]])
    for position in ("number", "unit", "suffix"):
        lo = np.array([r["near_lo"] for r in load(out_dir, f"L18 {position} manifold")["rows"]])
        assert not np.allclose(lo, base, atol=1e-5), position
    answer = load(out_dir, "L18 answer manifold")["rows"]
    assert not np.allclose([r["near_lo"] for r in answer], [r["ref_near_lo"] for r in answer], atol=1e-5)


def test_the_reruns_follow_the_baseline_in_prompt_order(full_run):
    out_dir, _ = full_run
    base = load(out_dir, "baseline")["rows"]
    for name in ("noop 1", "noop 2"):
        rows = load(out_dir, name)["rows"]
        assert [r["sample_uid"] for r in rows] == [r["sample_uid"] for r in base]
        np.testing.assert_allclose([r["near_lo"] for r in rows], [r["near_lo"] for r in base], atol=1e-3)


def test_answer_cells_rescore_the_baseline_reasoning_at_the_last_position(full_run, tok, tiny, cfg):
    out_dir, _ = full_run
    z = np.load(out_dir / "conditions" / "baseline.npz")
    inputs = np.load(out_dir / "inputs.npz")
    records = bank_records()[:N]
    seqs = [z["ids"][a:b][:s].tolist() for a, b, s in zip(z["offsets"][:-1], z["offsets"][1:], z["answer_start"])]
    delta = inputs["step|L18|manifold|answer"]
    written = [tok.decode([int(z["ids"][a:b][s])]) for a, b, s in zip(z["offsets"][:-1], z["offsets"][1:], z["answer_start"])]
    labels = [label_ids(tok, cfg=cfg) if w.startswith(" ") else label_ids(tok) for w in written]
    expected = S.score(tiny, "cpu", seqs, [r["near_label"] == "a" for r in records], labels, tok.pad_token_id, 2,
                       layer=18, positions=[[len(s) - 1] for s in seqs],
                       make_edit=lambda rows: S.Add(torch.tensor(delta[rows])))
    got = [r["near_lo"] for r in load(out_dir, "L18 answer manifold")["rows"]]
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_answer_cells_never_generate(tiny_model, tmp_path, monkeypatch):
    write_inputs(tmp_path)
    scan.main(run_args(tmp_path, only="baseline"))
    monkeypatch.setattr(scan, "generate", lambda *a, **k: pytest.fail("an answer-slot cell generated"))
    scan.main(run_args(tmp_path, only="L18 answer manifold;L18 answer random"))
    assert len(load(tmp_path, "L18 answer random")["rows"]) == N


def test_answer_cells_wait_for_the_baseline(tiny_model, tmp_path, capsys):
    write_inputs(tmp_path)
    scan.main(run_args(tmp_path, only="L18 answer manifold;L18 unit manifold"))
    done = sorted(p.name for p in (tmp_path / "conditions").glob("*.json"))
    assert done == ["L18_unit_manifold.json"]
    assert "baseline unit is not finished" in capsys.readouterr().out


def test_only_and_limit_select_units_and_prompts_and_a_rerun_skips_finished_units(tiny_model, tmp_path, monkeypatch, capsys):
    write_inputs(tmp_path)
    args = run_args(tmp_path, only="baseline;noop 1;L18 number random", limit=2)
    scan.main(args)
    assert sorted(p.name for p in (tmp_path / "conditions").glob("*.json")) == [
        "L18_number_random.json", "baseline.json", "noop_1.json"]
    assert all(len(load(tmp_path, n)["rows"]) == 2 for n in ("baseline", "noop 1", "L18 number random"))
    lines = [line for line in capsys.readouterr().out.splitlines() if "near log-odds" in line]
    assert len(lines) == 3 and "flips" in lines[1] and "flips" not in lines[0]    # the baseline has nothing to flip from

    monkeypatch.setattr(scan, "run_unit", lambda *a: pytest.fail("a finished unit was rerun"))
    scan.main(args)


def test_a_claimed_unfinished_unit_is_left_to_its_worker(tiny_model, tmp_path):
    write_inputs(tmp_path)
    (tmp_path / "conditions").mkdir()
    (tmp_path / "conditions" / "baseline.lock").mkdir()
    scan.main(run_args(tmp_path, only="baseline;noop 1", limit=1))
    assert sorted(p.name for p in (tmp_path / "conditions").glob("*.json")) == ["noop_1.json"]


def test_edit_positions_per_position(tok, cfg):
    records = bank_records()[:2]
    ctx = {"phrase": [record_phrase(tok, cfg, r) for r in records], "newline": [[7], [9]]}
    phrase = ctx["phrase"][0]
    assert scan.edit_positions("number", ctx, 0) == phrase[:-1]
    assert scan.edit_positions("unit", ctx, 0) == phrase[-1:]
    assert scan.edit_positions("suffix", ctx, 1) == [9]
    assert len(phrase) >= 2
