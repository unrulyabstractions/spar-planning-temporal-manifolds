"""Whole pipeline on a synthetic run: smoke prompts -> fake activations written with Alan's RunWriter
-> scripts/evaluate.py. The fake residual stream is built so the answer is known:

  dim 0 : log H                              (the horizon)
  dim 1 : log H + 0.8 (log D - log H)        ("durations in context": 80% of the way to D; = log H on clean prompts)
  dim 2 : log D                              (the distractor on its own; 0 on clean prompts)

On clean prompts dims 0 and 1 are identical, so a decoder trained only on clean prompts splits its
weight between them and is pulled toward D (baseline pull ~0.5 x 0.8 = 0.4). With distractors in
training it can put all weight on dim 0 and ignore D, so tier A/B/C pulls should be ~0.
Behavior depends on H only.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ptm.capture import SampleResult
from ptm.chat import position_labels
from ptm.prompts import from_frame
from ptm.store import RunWriter
from rad.build import DistractorConfig, build_all, smoke_subset

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_on_synthetic_run(tmp_path):
    df = smoke_subset(build_all(DistractorConfig()), n_configs_per_split=2)
    prompts = tmp_path / "prompts.parquet"; df.to_parquet(prompts, index=False)
    labels = position_labels(9, 8); L, d = 2, 16
    w = RunWriter(tmp_path / "run", meta=dict(n_layers=L, d_model=d, position_labels=labels, model_name="synthetic"), shard_rows=50)
    samples = from_frame(df); w.register_samples(samples)
    rng = np.random.default_rng(0)
    res = []
    for s in samples:
        lh = np.log10(s.horizon_years)
        ld = np.log10(s.distractor_years) if s.distractor_text else 0.0
        v = rng.normal(0, 0.02, (L + 1, len(labels), d))
        v[..., 0] += lh; v[..., 1] += lh + (0.8 * (ld - lh) if s.distractor_text else 0.0); v[..., 2] += ld
        lo = -2.0 * lh + rng.normal(0, 0.3)
        la, lb = (lo, 0.0) if s.short_first else (0.0, lo)
        res.append(SampleResult(sample_uid=s.sample_uid, gen_text="I choose: a).", choice="a" if la > lb else "b",
                                logit_a=la, logit_b=lb, p_a=0.5, p_b=0.5, acts=torch.tensor(v, dtype=torch.bfloat16),
                                pos_valid=[True] * len(labels)))
    w(res); w.close()

    out = tmp_path / "out"
    cmd = [sys.executable, str(ROOT / "scripts" / "evaluate.py"), str(tmp_path / "run"), str(prompts), str(out),
           "--layers", "1,2", "--positions", "T1,T3", "--n-boot", "50", "--alphas", "0.01,0.1,1", "--cells-full", "L1:T1,auto"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    for f in ["behavior_gate.csv", "sweep.csv", "headline_cells.csv", "text_baselines.csv", "summary.md", "summary.json", "predictions.npz"]:
        assert (out / f).exists(), f
    beh = pd.read_csv(out / "behavior_gate.csv")
    assert beh.irrelevant.all()
    h = pd.read_csv(out / "headline_cells.csv"); h = h[h.cell == "L1:T1"]
    pre = h[h.tier == "pre-C"].pull; assert (pre > 0.3).all(), pre.tolist()
    for tier in ("A", "B", "C"):
        p = h[h.tier == tier].pull; assert (p.abs() < 0.1).all(), (tier, p.tolist())
    assert "SEPARABLE" in (out / "summary.md").read_text()
    assert json.loads((out / "summary.json").read_text())["cells"][0] == "L1:T1"
