import json
import numpy as np
import torch
from ptm.capture import SampleResult
from ptm.prompts import DatasetConfig, generate
from ptm.store import RunData, RunWriter


def test_writer_reader_roundtrip(tmp_path):
    samples = generate(DatasetConfig(n=7, seed=0))
    L, n_pos, d = 3, 4, 5
    labels = ["T0", "T1", "R0", "R1"]
    w = RunWriter(tmp_path / "run", meta={"n_layers": L, "d_model": d, "position_labels": labels, "model_name": "fake"}, shard_rows=3)
    w.register_samples(samples)
    results = []
    for i, s in enumerate(samples):
        acts = torch.arange(L + 1, dtype=torch.float32)[:, None, None] * 100 + torch.arange(n_pos)[None, :, None] * 10 + i
        acts = acts.expand(L + 1, n_pos, d).clone().to(torch.bfloat16)
        results.append(SampleResult(sample_uid=s.sample_uid, gen_text="I choose: a).", choice="a" if i % 2 else None,
                                    logit_a=2.0, logit_b=0.0, p_a=0.7, p_b=0.2, acts=acts, pos_valid=[True, True, True, i % 3 != 0]))
    w(results[:4]); w(results[4:]); w.close()

    r = RunData(tmp_path / "run")
    assert len(r.index) == 7 and sorted(r.index["shard"].unique()) == [0, 1, 2]
    X = r.get(layer=2, label="R1")
    assert X.shape == (7, d)
    np.testing.assert_allclose(X[:, 0], [2 * 100 + 3 * 10 + i for i in range(7)])
    assert r.valid_mask("R1").tolist() == [i % 3 != 0 for i in range(7)]
    # behavior mapping respects label order
    for i, s in enumerate(samples):
        row = r.index.iloc[i]
        if i % 2:
            assert row["chose_short"] == s.short_first
            assert row["p_short_vocab"] == (0.7 if s.short_first else 0.2)
            assert abs(row["p_short"] - (1 / (1 + np.exp(-2.0)) if s.short_first else 1 / (1 + np.exp(2.0)))) < 1e-9
        else:
            assert row["chose_short"] is None or (isinstance(row["chose_short"], float) and np.isnan(row["chose_short"]))
    sub = r.index[r.index["shard"] == 1]
    Xs = r.get(layer=0, label="T0", rows=sub)
    np.testing.assert_allclose(Xs[:, 0], [3, 4, 5])
