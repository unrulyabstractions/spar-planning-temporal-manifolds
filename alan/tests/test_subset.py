import numpy as np, torch
from ptm.capture import SampleResult
from ptm.prompts import DatasetConfig, generate
from ptm.store import RunData, RunWriter
from ptm.subset import export_subset, load_subset


def test_subset_roundtrip_matches_full_store(tmp_path):
    samples = generate(DatasetConfig(n=11, seed=1)); L, d = 3, 6; labels = ["T0", "T1", "R0"]
    w = RunWriter(tmp_path / "run", meta={"n_layers": L, "d_model": d, "position_labels": labels, "model_name": "fake"}, shard_rows=4)
    w.register_samples(samples)
    rng = np.random.default_rng(0)
    w([SampleResult(sample_uid=s.sample_uid, gen_text="", acts=torch.tensor(rng.normal(size=(L + 1, 3, d)), dtype=torch.bfloat16), pos_valid=[True] * 3) for s in samples])
    w.close(); run = RunData(tmp_path / "run")
    meta = export_subset(run, layers=[1, 3], positions=["T1", "R0"], max_mb=0.0001)   # tiny cap -> several chunks
    assert len(meta["chunks"]) > 1 and meta["chunks"][-1][1] == 11
    for l in [1, 3]:
        for p in ["T1", "R0"]:
            np.testing.assert_array_equal(load_subset(tmp_path / "run", l, p), run.get(l, p))
    try:
        load_subset(tmp_path / "run", 2, "T1"); assert False
    except KeyError:
        pass
