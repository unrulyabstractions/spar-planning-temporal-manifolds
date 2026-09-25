import json
import numpy as np
import pandas as pd
import torch
from ptm.analysis import cell_metrics, sweep, behavior_by_horizon, plot_cell, plot_sweep, plot_behavior
from ptm.capture import SampleResult
from ptm.prompts import DatasetConfig, generate
from ptm.store import RunData, RunWriter


def _synthetic_run(tmp_path, n=300, d=40, L=2, seed=0):
    """Activations whose PC1 is log horizon along a fixed direction, plus isotropic noise."""
    rng = np.random.default_rng(seed)
    samples = generate(DatasetConfig(n=n, seed=seed))
    direction = rng.normal(size=d); direction /= np.linalg.norm(direction)
    labels = ["T0", "R0"]
    w = RunWriter(tmp_path / "run", meta={"n_layers": L, "d_model": d, "position_labels": labels, "model_name": "synthetic"}, shard_rows=50)
    w.register_samples(samples)
    res = []
    for s in samples:
        y = 0.0 if s.horizon_text is None else np.log10(s.horizon_years)
        acts = np.zeros((L + 1, len(labels), d), dtype=np.float32)
        acts[:, 0] = 5 * y * direction + 0.5 * rng.normal(size=d)   # T0 carries horizon strongly
        acts[:, 1] = rng.normal(size=d)                              # R0 is pure noise
        # behavior: choose short when horizon is short
        chose_short_prob = 1 / (1 + np.exp(2 * (y - 0.3)))
        chose_short = rng.random() < chose_short_prob
        choice = ("a" if chose_short else "b") if s.short_first else ("b" if chose_short else "a")
        p_short = float(chose_short_prob)
        pa, pb = (p_short, 1 - p_short) if s.short_first else (1 - p_short, p_short)
        lo = float(np.log(p_short / (1 - p_short)))
        la, lb = (lo, 0.0) if s.short_first else (0.0, lo)
        res.append(SampleResult(sample_uid=s.sample_uid, gen_text=f"I choose: {choice}).", choice=choice, p_a=pa, p_b=pb, logit_a=la, logit_b=lb,
                                acts=torch.tensor(acts).to(torch.bfloat16), pos_valid=[True, True]))
    w(res); w.close()
    return RunData(tmp_path / "run")


def test_ordinality_recovers_planted_structure(tmp_path):
    run = _synthetic_run(tmp_path)
    m, _ = cell_metrics(run, layer=1, position="T0", supervised=True)
    print(f"planted cell: rho_pc1={m.rho_pc1_horizon:.3f} ridge_r2={m.ridge_r2_horizon:.3f} reward_rho={m.rho_pc1_reward_ratio:.3f} sil={m.silhouette_choice_pc3:.3f}")
    assert abs(m.rho_pc1_horizon) > 0.95
    assert m.ridge_r2_horizon > 0.9
    assert abs(m.rho_pc1_reward_ratio) < 0.3          # reward control stays low
    assert abs(m.rho_pc1_p_short) > 0.8               # behavior aligns with PC1 by construction
    m0, _ = cell_metrics(run, layer=1, position="R0")
    print(f"noise cell:   rho_pc1={m0.rho_pc1_horizon:.3f}")
    assert abs(m0.rho_pc1_horizon) < 0.3


def test_sweep_and_plots(tmp_path):
    run = _synthetic_run(tmp_path, n=120, d=12)
    t = sweep(run)
    assert set(t["position"]) == {"T0", "R0"} and len(t) == 2 * 3
    b = behavior_by_horizon(run)
    assert "none" in set(b["horizon_bin"]) and b["n"].sum() == 120
    assert plot_cell(run, 1, "T0", tmp_path / "f" / "cell.png").exists()
    assert plot_sweep(t, run, tmp_path / "f" / "sweep.png").exists()
    assert plot_behavior(run, tmp_path / "f" / "beh.png").exists()
