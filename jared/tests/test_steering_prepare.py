"""Run-folder selection in analysis/steering_prepare.py: one results folder holds
the 8B and 14B runs side by side (10_spec_scale_14b.md section 3)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from steering_prepare import main, matches_e0, run_dir  # noqa: E402
from spar_horizon.model_io import Generation  # noqa: E402


def finished_run(root, name):
    run = root / name
    run.mkdir()
    (run / "manifest.json").write_text("{}")
    return run


def test_run_dir_picks_the_requested_model(tmp_path):
    finished_run(tmp_path, "exp0_Qwen3-8B_think-on_full")
    run_14b = finished_run(tmp_path, "exp0_Qwen3-14B_think-on_full")
    assert run_dir(tmp_path, "Qwen/Qwen3-14B", "on") == run_14b


def test_run_dir_rejects_an_unfinished_run(tmp_path):
    (tmp_path / "exp0_Qwen3-14B_think-on_full").mkdir()   # generation still running: no manifest yet
    with pytest.raises(FileNotFoundError):
        run_dir(tmp_path, "Qwen/Qwen3-14B", "on")


def test_no_reasoning_refuses_thinking_on_cells(tmp_path):
    with pytest.raises(ValueError, match="thinking-on"):
        main(tmp_path, tmp_path, "off:pre-label:27,on:pre-label:27", "patch", "Qwen/Qwen3-14B", reasoning=False)


def test_cached_reasoning_must_equal_the_e0_trace():
    ids = torch.tensor([5, 6, 7, 8, 9, 10])
    g = Generation(ids, 2, 5, 2, False)
    z = {"kept": np.array([True, True]), "think_len": np.array([0, 2]), "forced_mask": np.array([False, False]),
         "keep_idx": np.array([[0, 1], [1, 3]]), "keep_ids": np.array([[0, 0], [6, 8]])}
    assert matches_e0(g, z, 1)
    z["keep_ids"][1, 1] = 99                          # the cache holds a different trace than E0 saw
    assert not matches_e0(g, z, 1)
    z["keep_ids"][1, 1] = 8
    z["think_len"][1] = 3
    assert not matches_e0(g, z, 1)
