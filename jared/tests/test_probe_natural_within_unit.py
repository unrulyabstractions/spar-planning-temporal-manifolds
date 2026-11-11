"""The natural-closes / within-unit driver: run lookup by model, and its table and figure on a planted cell."""

import numpy as np

from pipeline.s1_data_vectors.analysis import probe_natural_within_unit as D
from pipeline.s1_data_vectors.analysis.probe_dimensionality import run_file
from spar_horizon.probes import analyze_cell


def test_run_file_picks_the_requested_model(tmp_path):
    for name in ("exp0_Qwen3-8B_think-on_full", "exp0_Qwen3-14B_think-on_full"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "activations.npz").touch()
    assert run_file(tmp_path, "on").parent.name == "exp0_Qwen3-8B_think-on_full"
    assert run_file(tmp_path, "on", "Qwen/Qwen3-14B").parent.name == "exp0_Qwen3-14B_think-on_full"
    assert run_file(tmp_path, "off") is None


def test_summary_and_figure_on_a_planted_cell(tmp_path):
    log_h = np.repeat(np.linspace(-6, 2.7, 10), 6)
    rng = np.random.default_rng(0)
    X = np.outer(log_h, rng.normal(size=100)) + rng.normal(scale=0.1, size=(len(log_h), 100))
    # the unit words are not on the full-bank grid, so pass them explicitly
    units = np.where(log_h < np.median(log_h), "short", "long")
    cell = analyze_cell(X, log_h, units)
    cell["n"], cell["n_horizons"] = len(log_h), 10
    entry = {"layer": 3, "sweep_best": None, "all": cell, "natural": cell}
    results = {m: {f"{mode}:{label}": entry for mode, label in D.CELLS} for m in D.MODELS}
    text = D.summarize(results)
    assert "within" in text and "Qwen/Qwen3-14B" in text
    D.plot(results, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").exists()
