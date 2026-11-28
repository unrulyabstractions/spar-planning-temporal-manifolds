"""Tests for screen.py (no model): the sweep's stop rule, the per-config record, resuming, the TF32 check, the selection
rule, and that replay_eval accepts the calibration.json that select writes. Run (from evaluation/):
python -m pytest -q test_screen.py"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import replay_eval as rv
import screen


def row(x, odd, healthy=True):
    return {"x": x, "mean_odd_logit": odd, "healthy": healthy, "even_label_mass": 0.9}


def points(odds, healthy=None):
    """A fake point(s): the sweep row at each grid strength, and the strengths asked for."""
    grid = [0.1, 0.2, 0.4, 0.8]
    healthy = healthy or [True] * len(odds)
    table = {x: row(x, o, h) for x, o, h in zip(grid, odds, healthy)}
    asked = []

    def point(s):
        asked.append(s)
        return dict(table[s], odd_near_only=0.1, odd_both=0.2, odd_far_first=0.3, odd_near_first=0.1, letter=0.1)

    return point, grid, asked


def test_sweep_stops_at_the_first_unhealthy_strength_or_sign_change():
    point, grid, asked = points([0.2, 0.5, 0.9, 1.2])
    assert [r["x"] for r in screen.sweep_until_stop(point, grid)] == grid           # all healthy, one sign: whole grid
    point, grid, asked = points([0.2, 0.5, 0.9, 1.2], [True, True, False, True])
    assert [r["x"] for r in screen.sweep_until_stop(point, grid)] == [0.1, 0.2, 0.4] and asked == [0.1, 0.2, 0.4]
    point, grid, asked = points([-0.2, -0.5, 0.3, -1.0])
    assert [r["x"] for r in screen.sweep_until_stop(point, grid)] == [0.1, 0.2, 0.4]
    point, grid, asked = points([0.2, 0.5, 0.9, 1.2], [False] * 4)
    assert [r["x"] for r in screen.sweep_until_stop(point, grid)] == [0.1] and asked == [0.1]


def test_screen_config_scores_the_peak_rule():
    point, grid, _ = points([0.2, 0.7, 0.5, -0.3])
    rec = screen.screen_config(point, grid, mass0=0.95)
    assert rec["strength"] == 0.2 and rec["strength_rank"] == 1 and rec["S"] == 0.7 and rec["healthy"]
    assert (rec["strength"], rec["rule"]) == rv.choose_strength(rec["sweep"], "peak", None)
    assert rec["mass_ratio"] == pytest.approx(0.9 / 0.95) and rec["letter"] == 0.1 and rec["odd_both"] == 0.2
    assert [r["x"] for r in rec["sweep"]] == grid                                    # stopped at the sign change
    point, grid, _ = points([0.2, 0.7, 0.5, 0.3], [False] * 4)
    rec = screen.screen_config(point, grid, mass0=0.95)
    assert not rec["healthy"] and len(rec["sweep"]) == 1                             # never a candidate


class OddReplayer:
    """y(±x) = y0 ± x × effect[uid]; label mass 0.9 at every strength."""

    def __init__(self, effect):
        self.effect = effect

    def run(self, items, edit, x, positions="all"):
        return (np.array([x * self.effect[it["uid"]] for it in items], np.float32),
                np.full(len(items), 0.9, np.float32))


def test_screen_row_splits_the_odd_effect_by_regime_and_order():
    items = [{"uid": "a", "regime": "near_only", "near_first": True}, {"uid": "b", "regime": "near_only", "near_first": False},
             {"uid": "c", "regime": "both", "near_first": True}, {"uid": "d", "regime": "both", "near_first": False}]
    rep = OddReplayer({"a": 1.0, "b": 3.0, "c": 2.0, "d": 2.0})
    r = screen.screen_row(rep, items, None, 0.5, "all", mass0=0.9)
    o = {"a": 0.5, "b": 1.5, "c": 1.0, "d": 1.0}
    assert r["mean_odd_logit"] == pytest.approx(((o["a"] + o["c"]) / 2 + (o["b"] + o["d"]) / 2) / 2)
    assert r["odd_near_only"] == pytest.approx(1.0) and r["odd_both"] == pytest.approx(1.0)
    assert r["odd_far_first"] == pytest.approx(1.25) and r["odd_near_first"] == pytest.approx(0.75)
    assert r["letter"] == pytest.approx(0.25) and r["healthy"]


def test_resume_skips_configs_already_screened(tmp_path):
    path = tmp_path / "screen.jsonl"
    bundles = [tmp_path / "caa__L20", tmp_path / "trend__L20"]
    configs = [(b, s) for b in bundles for s in ("all", "prompt")]
    calls = []

    def run_config(bundle, span):
        calls.append((bundle.name, span))
        return {"S": 0.1, "strength": 0.2, "healthy": True}

    screen.run_screen(configs[:1], run_config, path, deadline=None)
    done = screen.done_configs(path)
    assert done == {("caa__L20", "all")}
    todo = [c for c in configs if (c[0].name, c[1]) not in done]
    screen.run_screen(todo, run_config, path, deadline=None)
    assert calls == [("caa__L20", "all"), ("caa__L20", "prompt"), ("trend__L20", "all"), ("trend__L20", "prompt")]
    assert len(rv.read_jsonl(path)) == 4 and screen.done_configs(path) == {(b.name, s) for b, s in configs}
    # cmd_screen with everything done returns before loading any model
    screen.cmd_screen(SimpleNamespace(out=str(tmp_path), bundles=[str(b) for b in bundles], write_span=["all", "prompt"]))


def test_no_config_starts_past_the_deadline(tmp_path):
    calls = []
    screen.run_screen([(tmp_path / "caa", "all")], lambda b, s: calls.append(b) or {}, tmp_path / "s.jsonl",
                      deadline=time.time() - 1)
    assert calls == [] and not (tmp_path / "s.jsonl").exists()


def test_tf32_check_turns_tf32_off_when_y_moves_too_much():
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

    class Rep:
        def __init__(self, gap):
            self.gap = gap

        def unsteered(self, items):
            y = np.array([1.0, 2.0]) + (self.gap if torch.backends.cuda.matmul.allow_tf32 else 0.0)
            return y, np.full(2, 0.9)

    try:
        check, (y, _) = screen.tf32_check(Rep(0.2), [{}, {}])
        assert not check["used"] and check["max_abs_diff_logits"] == pytest.approx(0.2)
        assert not torch.backends.cuda.matmul.allow_tf32 and np.allclose(y, [1.0, 2.0])   # the fp32 values
        check, (y, _) = screen.tf32_check(Rep(0.01), [{}, {}])
        assert check["used"] and torch.backends.cuda.matmul.allow_tf32 and np.allclose(y, [1.01, 2.01])
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before


def record(bundle, method, S, rank, healthy=True, span="all", kind="push"):
    grid = [0.1, 0.2, 0.4, 0.8] if kind == "push" else [0.5, 1.0, 2.0, 4.0]
    return {"bundle": bundle, "path": f"/m/{bundle}", "method": method, "layer": 20, "site": "prompt_end",
            "span": span, "kind": kind, "grid": grid, "sweep": [row(grid[rank], S, healthy)], "strength": grid[rank],
            "strength_rank": rank, "rule": "peak", "S": S, "healthy": healthy, "mass_ratio": 0.97,
            "unsteered_label_mass": 0.95, "odd_near_only": S, "odd_both": S, "odd_far_first": S, "odd_near_first": S,
            "letter": 0.0, "n_items": 30, "seconds": 1.0}


def test_select_excludes_controls_and_unhealthy_and_breaks_ties_by_strength():
    records = [record("choice__L24", "choice", 5.0, 0), record("reward__L24", "reward", 4.0, 0),
               record("caa__L16", "caa", 4.5, 0, healthy=False),
               record("caa__L20", "caa", 2.0, 3), record("trend__L20", "trend", 1.8, 1, span="prompt"),
               record("probe__L20", "probe", 1.7, 0)]
    winner, candidates = screen.select(records)
    # 1.8 is within 0.25 of the best candidate's 2.0 and at a smaller strength; 1.7 is not within the tie
    assert winner["bundle"] == "trend__L20"
    assert [r["bundle"] for r in candidates] == ["caa__L20", "trend__L20", "probe__L20"]
    winner, _ = screen.select([record("caa__L20", "caa", 2.0, 2), record("trend__L20", "trend", 1.9, 2)])
    assert winner["bundle"] == "caa__L20"                                            # same strength: the higher S
    assert screen.select([record("choice__L24", "choice", 5.0, 0)]) == (None, [])


def test_select_writes_a_calibration_that_replay_eval_accepts(tmp_path):
    out, confirm = tmp_path / "screen", tmp_path / "confirm"
    out.mkdir()
    records = [record("caa__L20", "caa", 0.6, 2), record("choice__L24", "choice", 3.0, 0),
               record("curve-displace__L20", "curve-displace", 0.9, 2, span="prompt", kind="target")]
    rv.write_jsonl_atomic(out / "screen.jsonl", records)
    screen.main(["select", "--out", str(out), "--confirm-out", str(confirm)])
    win = json.loads((out / "winner.json").read_text())
    assert win["winner"]["bundle"] == "curve-displace__L20" and len(win["candidates"]) == 2
    md = (out / "screen.md").read_text()
    assert md.index("| 20 | prompt_end | choice |") < md.index("| 20 | prompt_end | curve-displace |") and "| control |" in md
    cal = json.loads((confirm / "calibration.json").read_text())
    assert cal["x_shift"] == 2.0 and cal["x_push"] is None and cal["write_span"] == "prompt"
    assert cal["rule"].startswith("fixed by screen.py select")
    args = SimpleNamespace(out=str(confirm), write_span="prompt")
    assert rv.stage_calibrate(args, None, None, None) == cal                          # returned as is, no model
    screen.main(["select", "--out", str(out), "--confirm-out", str(confirm)])        # rerun: same file, accepted
    rv.write_jsonl_atomic(out / "screen.jsonl", records + [record("trend__L16", "trend", 5.0, 0)])
    with pytest.raises(SystemExit, match="exists with another"):
        screen.main(["select", "--out", str(out), "--confirm-out", str(confirm)])


def test_patch_baselines_are_keyed_by_name_and_never_selected(tmp_path):
    configs = screen.patch_configs(["/m/trend__L16_horizon_last"], [16, 20])
    assert {p.name for p in configs} == {"patch:text", "patch:full@L16", "patch:full@L20",
                                         "patch:trend__L16_horizon_last"}
    records = [record("patch:full@L16", "patch:full", 9.0, 0, kind="patch"),
               record("patch:trend__L16", "patch:trend", 5.0, 0, kind="patch"), record("caa__L20", "caa", 1.0, 0)]
    winner, candidates = screen.select(records)
    assert winner["bundle"] == "caa__L20" and [r["bundle"] for r in candidates] == ["caa__L20"]
    assert "patch baseline" in screen.screen_md(records, winner)


def test_patch_baselines_need_a_horizon_span(tmp_path):
    args = SimpleNamespace(out=str(tmp_path), bundles=[], patch_bundles=[], patch_layers=[16], write_span=["all"])
    with pytest.raises(SystemExit, match="horizon"):
        screen.cmd_screen(args)


def test_patch_select_takes_the_best_full_patch_and_its_best_basis(tmp_path):
    def patch(bundle, method, S, layer, span):
        return {**record(bundle, method, S, 0, span=span, kind="patch"), "layer": layer}
    records = [patch("patch:full@L16", "patch:full", 3.0, 16, "horizon"),
               patch("patch:full@L20", "patch:full", 5.0, 20, "from_horizon"),
               patch("patch:trend__L20", "patch:trend", 2.0, 20, "from_horizon"),
               patch("patch:caa__L20", "patch:caa", 2.5, 20, "from_horizon"),
               patch("patch:caa__L20", "patch:caa", 4.0, 20, "horizon"),
               patch("patch:text", "patch:text", 6.0, None, "text")]
    choice = screen.best_patch(records)
    assert (choice["layer"], choice["span"], choice["basis"]["bundle"]) == (20, "from_horizon", "patch:caa__L20")
    assert choice["basis"]["S"] == 2.5
    out = tmp_path / "screen"
    out.mkdir()
    rv.write_jsonl_atomic(out / "screen.jsonl", records)
    screen.main(["patch-select", "--out", str(out), "--confirm-out", str(tmp_path / "confirm")])
    cal = json.loads((tmp_path / "confirm" / "calibration.json").read_text())
    assert cal["write_span"] == "from_horizon" and cal["x_push"] is None
