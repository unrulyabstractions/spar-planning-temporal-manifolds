"""Tests for layer_sweep.py on a synthetic activation set (CPU, no model). Run (from evaluation/): python -m pytest -q test_layer_sweep.py"""

import json

import numpy as np

import layer_sweep as LS
from test_curve_dim import HORIZONS, planted


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_run(tmp_path):
    """Layer 1 carries the horizon curve, layer 2 is noise; every third prompt is `none`, a few are no-horizon or held out."""
    X, t, groups = planted(nuisance=30)
    rng = np.random.default_rng(0)
    regimes = [("none", "near_only", "both")[i % 3] for i in range(len(groups))]
    prompts = [{"uid": f"u{i}", "split": "extraction", "horizon": g, "horizon_grid_value": g, "regime": regimes[i],
                "h_over_far_delay": float(10 ** (t[i] - 1.5))} for i, g in enumerate(groups)]
    prompts[0]["horizon"] = None
    bank = tmp_path / "bank"
    acts = tmp_path / "acts"
    bank.mkdir(), acts.mkdir()
    write_jsonl(bank / "prompts.jsonl", prompts)
    write_jsonl(acts / "index.jsonl", [{"row": i, "uid": p["uid"], "chosen_option": "far" if p["h_over_far_delay"] >= 1 else "near"}
                                       for i, p in enumerate(prompts)])
    np.save(acts / "L1_prompt_end.npy", X.astype(np.float32))
    np.save(acts / "L2_prompt_end.npy", rng.normal(size=X.shape).astype(np.float32))
    write_jsonl(tmp_path / "roles.jsonl", [{"uid": p["uid"], "role": "fit" if i % 4 else "test"} for i, p in enumerate(prompts)])
    return acts, bank, tmp_path / "roles.jsonl"


def test_rows_exclude_none_no_horizon_and_other_roles(tmp_path):
    acts, bank, roles = make_run(tmp_path)
    everything = LS.select_rows(acts / "index.jsonl", bank / "prompts.jsonl")
    fit_only = LS.select_rows(acts / "index.jsonl", bank / "prompts.jsonl", roles, "fit")
    assert all(r["regime"] != "none" and r["horizon"] for r in everything)
    assert 0 < len(fit_only) < len(everything)
    assert len(everything) > 100


def test_sweep_ranks_the_informative_layer_above_noise_and_writes_both_files(tmp_path):
    acts, bank, roles = make_run(tmp_path)
    cells = LS.main([str(acts), "--bank", str(bank), "--uids", str(roles), "--out", str(tmp_path / "out")])
    assert cells[(1, "prompt_end")]["ridge_rho"] > 0.8
    assert abs(cells[(2, "prompt_end")]["ridge_rho"]) < 0.5
    assert cells[(1, "prompt_end")]["caa_reliability"] > cells[(2, "prompt_end")]["caa_reliability"]
    assert cells[(1, "prompt_end")]["class_gap"] > 0.9                      # classes are far / near by construction
    report = json.loads((tmp_path / "out" / "layer_sweep.json").read_text())
    assert {(c["layer"], c["site"]) for c in report["cells"]} == {(1, "prompt_end"), (2, "prompt_end")}
    page = (tmp_path / "out" / "layer_sweep.html").read_text()
    assert "L1" in page and "L2" in page and "http" not in page
