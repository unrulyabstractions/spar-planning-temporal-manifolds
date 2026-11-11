"""E18 runner: unit packing, the saved states and rows (tiny random-weight Qwen3 and the cached 0.8B
tokenizer), skipping finished units, losing only the unit in flight, OOM step-down. No model weights."""

import json
from itertools import groupby

import numpy as np
import pytest
import torch

import pipeline.s2_steering.experiments.units_implied.exp18_implied as e18
from spar_horizon import extract as extract_module
from spar_horizon.config import Settings
from spar_horizon.implied_bank import build_implied_prompts
from test_exp14_states import MODEL, tiny, tok  # noqa: F401

SIZE = 5
LIMIT = 8   # records per module


@pytest.fixture(scope="module")
def cfg():
    return Settings(model=MODEL, thinking="on", prefill="", choice_prefix="I choose:", bank="implied",
                    max_think_tokens=3, n_response=6, answer_window=8, think_fractions=(1.0,), batch_size=SIZE)


@pytest.fixture(scope="module")
def records():
    return build_implied_prompts()


@pytest.fixture
def units(records):
    return e18.planned_units(records, ("M1", "M4"), SIZE, LIMIT)


@pytest.fixture
def tiny_model(tok, tiny, monkeypatch):
    monkeypatch.setattr(e18, "load_model", lambda cfg: (tok, tiny, "cpu"))


def test_units_never_split_a_base_and_cover_the_module(records):
    planned = e18.planned_units(records, ("M1", "M2", "M3", "M4"), 12)
    assert [len(u) for _, _, u in planned if len(u) > 12] == []
    for name, module, unit in planned:
        assert name.startswith(module) and {r["module"] for r in unit} == {module}
        for twin_id, group in groupby(unit, key=lambda r: r["twin_id"]):
            assert len(list(group)) == len([r for r in records if r["twin_id"] == twin_id])
    for module, expected in {"M1": 144, "M2": 72, "M3": 48, "M4": 24}.items():
        assert sum(len(u) for _, m, u in planned if m == module) == expected
    assert [n for n, m, _ in planned if m == "M1"][:2] == ["M1_000", "M1_001"]


def test_limit_keeps_whole_bases_per_module(units):
    for module in ("M1", "M4"):
        mine = [r for _, m, u in units if m == module for r in u]
        assert 4 <= len(mine) <= 14


def test_unknown_module_is_rejected(cfg, tmp_path):
    with pytest.raises(ValueError, match="unknown modules"):
        e18.run(cfg, tmp_path, ("M9",))


def test_a_unit_is_written_with_rows_states_and_no_temp_files(tiny_model, cfg, units, tmp_path):
    e18.run_units(cfg, tmp_path, units[:1], (12, 16, 24))
    name, _, unit = units[0]
    z = np.load(tmp_path / "units" / f"{name}.npz")
    assert z["layers"].tolist() == [12, 16] and "L24" not in z.files
    n_suffix = len([c for c in z["cols"] if str(c).startswith("suffix")])
    assert z["cols"].tolist() == [f"suffix_{i}" for i in range(n_suffix)] + ["think@1", "prelabel"]
    assert z["col_tokens"][1] == "\n" and z["L12"].dtype == np.float32
    n_kept = int(z["kept"].sum())
    assert z["L12"].shape == (n_kept, n_suffix + 2, 32) and z["label_logits"].shape == (n_kept, 2)
    assert len(z["sample_uids"]) == n_kept and z["forced_mask"].shape == (n_kept,)
    meta = json.loads((tmp_path / "units" / f"{name}.json").read_text())
    assert meta["sample_uids"] == [r["sample_uid"] for r in unit] and len(meta["rows"]) == len(unit)
    kept_rows = [r for r in meta["rows"] if "label" in r]
    assert len(kept_rows) == n_kept
    assert {"family", "anchor_id", "rendering", "twin_id", "regime", "truth", "context", "horizon_text",
            "us_convention", "think_text", "forced", "label_logits", "near_label"} <= set(kept_rows[0])
    assert "prompt" not in kept_rows[0]
    assert not list((tmp_path / "units").glob("*.tmp"))


def test_a_rerun_skips_finished_units_and_a_killed_run_loses_only_the_unit_in_flight(tiny_model, cfg, units, tmp_path,
                                                                                    monkeypatch):
    real, calls = extract_module.extract, []

    def dies_on_second(*args, **kwargs):
        calls.append(len(args[3]))
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(e18, "extract", dies_on_second)
    with pytest.raises(KeyboardInterrupt):
        e18.run_units(cfg, tmp_path, units, (12,))
    unit_dir = tmp_path / "units"
    assert e18.finished_units(unit_dir) == [units[0][0]]
    calls.clear()
    monkeypatch.setattr(e18, "extract", lambda *a, **k: calls.append(len(a[3])) or real(*a, **k))
    e18.run_units(cfg, tmp_path, units, (12,))
    assert calls == [len(u) for _, _, u in units[1:]]
    assert e18.finished_units(unit_dir) == sorted(n for n, _, _ in units)
    monkeypatch.setattr(e18, "extract", lambda *a, **k: pytest.fail("extract ran for a finished unit"))
    e18.run_units(cfg, tmp_path, units, (12,))


def test_a_write_that_dies_midway_leaves_no_unit_behind(tiny_model, cfg, units, tmp_path, monkeypatch):
    def dies(f, **arrays):
        f.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(e18.np, "savez", dies)
    with pytest.raises(OSError):
        e18.run_units(cfg, tmp_path, units[:1], (12,))
    assert e18.finished_units(tmp_path / "units") == [] and not list((tmp_path / "units").glob("*.npz"))


def test_units_written_for_other_prompts_are_refused(tiny_model, cfg, records, units, tmp_path):
    e18.run_units(cfg, tmp_path, units[:1], (12,))
    other = e18.planned_units(records, ("M1", "M4"), SIZE + 3, LIMIT)
    with pytest.raises(ValueError, match="other prompts"):
        e18.run_units(cfg, tmp_path, other, (12,))


def test_an_out_of_memory_error_lowers_the_batch_for_the_rest_of_the_run(tiny_model, cfg, units, tmp_path, monkeypatch):
    real, state = extract_module.extract, {"failed": False, "sizes": []}

    def ooms_once(*args, **kwargs):
        state["sizes"].append(args[4].batch_size)
        if not state["failed"]:
            state["failed"] = True
            raise torch.OutOfMemoryError("simulated")
        return real(*args, **kwargs)

    local = Settings(**{**cfg.to_dict(), "think_fractions": (1.0,)})
    monkeypatch.setattr(e18, "extract", ooms_once)
    e18.run_units(local, tmp_path, units[:2], (12,))
    assert state["sizes"] == [SIZE, SIZE * 2 // 3, SIZE * 2 // 3] and local.batch_size == SIZE * 2 // 3
    assert len(e18.finished_units(tmp_path / "units")) == 2


def test_run_writes_a_manifest_with_the_units_done(tiny_model, cfg, tmp_path):
    out = e18.run(cfg, tmp_path, ("M4",), 4, (12,))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["modules"] == ["M4"] and manifest["layers"] == [12]
    assert len(manifest["units_done"]) == manifest["n_units_planned"] >= 1
