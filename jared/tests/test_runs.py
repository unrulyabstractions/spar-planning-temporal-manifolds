import json

import numpy as np

from spar_horizon.config import Settings
from spar_horizon.extract import Extraction
from spar_horizon.runs import Manifest, load_activations, resolve_out_dir, save_activations


def test_out_dir_uses_workspace_on_the_box(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    cfg = Settings(model="Qwen/Qwen3-8B", thinking="off")
    d = resolve_out_dir("exp0", cfg)
    assert d == tmp_path / "results" / "horizon" / "exp0_Qwen3-8B_think-off"
    assert d.is_dir()


def test_out_dir_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE", "/nonexistent")
    d = resolve_out_dir("exp1", Settings(thinking="on"), out=tmp_path)
    assert d.parent == tmp_path


def test_manifest_roundtrip(tmp_path):
    cfg = Settings(thinking="off", limit=6)
    m = Manifest("exp0", cfg, tmp_path)
    m.add(n_prompts=6, skipped=np.int64(0), peak=np.float32(0.9))
    path = m.write()
    data = json.loads(path.read_text())
    assert data["experiment"] == "exp0" and data["settings"]["limit"] == 6
    assert data["skipped"] == 0 and abs(data["peak"] - 0.9) < 1e-6
    assert "wall_seconds" in data


def test_activations_roundtrip(tmp_path):
    acts = [np.zeros((4, 3, 8)), np.ones((4, 3, 8))]
    ext = Extraction(acts, ["x", "y", "z"], ["a)"] * 4, np.ones(4, bool), 2, [0] * 4)
    p = tmp_path / "a.npz"
    save_activations(p, ext, [1.0, None, 2.0, 5.0], rho=np.zeros((2, 3)))
    z = load_activations(p)
    assert len(z["activations"]) == 2 and z["activations"][1].sum() == 4 * 3 * 8
    assert z["tokens"] == ["x", "y", "z"] and z["n_suffix"] == 2
    assert np.isnan(z["horizons"][1]) and z["horizons"][3] == 5.0
