"""Tests for shared_draws.py. Run (from evaluation/): python -m pytest -q test_shared_draws.py"""

import json

import numpy as np
import pytest

import shared_draws as sd

MODEL, LAYER, SITE = "org/test-model", 38, "label"


def fake_run(tmp_path, n_ext=300, n_ver=40, n_null=20, d=64, seed=0, shuffle=False, name="act"):
    """A bank (extraction horizon prompts + no-horizon twins + verification) and an activations dir that also holds
    verification and no-horizon rows, which must never be used. Activations: strong 4-dim subspace + noise."""
    rng = np.random.default_rng(seed)
    recs = ([{"uid": f"x{i:04d}", "split": "extraction", "regime": "none" if i % 3 else "both"} for i in range(n_ext)]
            + [{"uid": f"x{i:04d}-null", "split": "extraction", "regime": "no_horizon"} for i in range(n_null)]
            + [{"uid": f"v{i:03d}-far-first", "split": "verification", "regime": "both"} for i in range(n_ver)])
    bank = tmp_path / "prompts.jsonl"
    if not bank.exists():
        bank.write_text("".join(json.dumps(r) + "\n" for r in recs))
    U = np.linalg.qr(np.random.default_rng(99).standard_normal((d, 4)))[0].T
    X = (rng.standard_normal((len(recs), 4)) * 10) @ U + rng.standard_normal((len(recs), d))
    uids = [r["uid"] for r in recs]
    order = rng.permutation(len(recs)) if shuffle else np.arange(len(recs))
    act = tmp_path / name
    act.mkdir()
    np.save(act / f"L{LAYER}_{SITE}.npy", X[order].astype(np.float32))
    (act / "index.jsonl").write_text("".join(json.dumps({"uid": uids[n], "row": int(k)}) + "\n"
                                             for k, n in enumerate(order)))
    return act, bank, U


def make_args(act, bank, out, **kw):
    a = ["make", "--activations", str(act), "--bank", str(bank), "--model", MODEL, "--layer", str(LAYER),
         "--site", SITE, "--out", str(out)]
    for k, v in kw.items():
        a += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    return a


def outdir(out):
    return out / "test-model" / f"L{LAYER}_{SITE}"


# ------------------------------------------------------------ randomness

def test_normals_moments():
    z = sd.normals(sd.stream("moments"), (200_000,))
    assert abs(z.mean()) < 0.01 and abs(z.var() - 1) < 0.01
    assert abs((np.abs(z) > 1.96).mean() - 0.05) < 0.003


def test_uniforms_open_interval():
    u = sd.uniforms(sd.stream("u"), 100_000)
    assert u.min() > 0 and u.max() < 1


def test_golden_values():
    """Pins the stream: if a numpy upgrade (or an edit) changes any draw, this fails. Values computed 2026-11-18."""
    z = sd.normals(sd.stream(sd.seed_key("google/gemma-4-31B-it", 38, "label", "isotropic")), (4,))
    assert np.allclose(z, GOLDEN, rtol=0, atol=1e-12), z.tolist()


GOLDEN = [0.07725265483087551, -0.4544450316832624, -1.1960370267056903, -0.38229818471070864]


def test_seed_keys_independent():
    V = [sd.draw_isotropic(64, 50, sd.seed_key(MODEL, LAYER, s, f)) for s in ("label", "prelabel")
         for f in ("isotropic", "covariance")]
    for a in range(len(V)):
        for b in range(a + 1, len(V)):
            assert np.abs(V[a] @ V[b].T).max() < 0.7      # unrelated streams: no near-duplicate rows


# ------------------------------------------------------------ make

def test_make_and_reproducible(tmp_path):
    act, bank, _ = fake_run(tmp_path)
    sd.main(make_args(act, bank, tmp_path / "o1"))
    sd.main(make_args(act, bank, tmp_path / "o2"))
    a, b = np.load(outdir(tmp_path / "o1") / "draws.npz"), np.load(outdir(tmp_path / "o2") / "draws.npz")
    for f in (*sd.FAMILIES, "real_diff_pairs"):
        assert np.array_equal(a[f], b[f])
    assert a["isotropic"].shape == (99, 64)
    assert np.allclose(np.linalg.norm(a["covariance"], axis=1), 1, atol=1e-6)


def test_row_order_does_not_matter(tmp_path):
    act1, bank, _ = fake_run(tmp_path, shuffle=False, name="a1")
    act2, _, _ = fake_run(tmp_path, shuffle=True, name="a2")
    sd.main(make_args(act1, bank, tmp_path / "o1"))
    sd.main(make_args(act2, bank, tmp_path / "o2"))
    a, b = np.load(outdir(tmp_path / "o1") / "draws.npz"), np.load(outdir(tmp_path / "o2") / "draws.npz")
    for f in (*sd.FAMILIES, "real_diff_pairs"):
        assert np.array_equal(a[f], b[f])


def test_iso_only_matches_full(tmp_path):
    act, bank, _ = fake_run(tmp_path)
    sd.main(make_args(act, bank, tmp_path / "full"))
    sd.main(["make", "--iso-only", "--d", "64", "--model", MODEL, "--layer", str(LAYER), "--site", SITE,
             "--out", str(tmp_path / "iso")])
    full, iso = np.load(outdir(tmp_path / "full") / "draws.npz"), np.load(outdir(tmp_path / "iso") / "draws.npz")
    assert np.array_equal(full["isotropic"], iso["isotropic"]) and "covariance" not in iso


def test_only_extraction_horizon_rows_used(tmp_path):
    act, bank, _ = fake_run(tmp_path, n_ext=300)
    sd.main(make_args(act, bank, tmp_path / "o"))
    uids = (outdir(tmp_path / "o") / "uids.txt").read_text().split()
    assert len(uids) == 300 and uids == sorted(uids)
    assert not any(u.endswith("-null") or u.startswith("v") for u in uids)
    pairs = np.load(outdir(tmp_path / "o") / "draws.npz")["real_diff_pairs"]
    assert (pairs[:, 0] != pairs[:, 1]).all() and pairs.max() < 300


def test_covariance_draws_follow_the_activations(tmp_path):
    """Isotropic draws put ~4/64 of their squared norm in the strong subspace; covariance and real_diff almost all."""
    act, bank, U = fake_run(tmp_path)
    sd.main(make_args(act, bank, tmp_path / "o"))
    z = np.load(outdir(tmp_path / "o") / "draws.npz")
    share = {f: float(((z[f].astype(np.float64) @ U.T) ** 2).sum(1).mean()) for f in sd.FAMILIES}
    assert share["isotropic"] < 0.15 and share["covariance"] > 0.8 and share["real_diff"] > 0.8


def test_refuses_small_and_overwrite(tmp_path):
    act, bank, _ = fake_run(tmp_path, n_ext=13)
    with pytest.raises(SystemExit, match="only 13 eligible rows"):
        sd.main(make_args(act, bank, tmp_path / "o"))
    sd.main(make_args(act, bank, tmp_path / "o", min_rows=10))
    with pytest.raises(SystemExit, match="never overwritten"):
        sd.main(make_args(act, bank, tmp_path / "o", min_rows=10))


def test_misaligned_index_refused(tmp_path):
    act, bank, _ = fake_run(tmp_path)
    lines = (act / "index.jsonl").read_text().splitlines()
    (act / "index.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(SystemExit, match="cannot align"):
        sd.main(make_args(act, bank, tmp_path / "o"))


# ------------------------------------------------------------ verify

def verify_args(out, act=None, bank=None):
    a = ["verify", "--draws", str(outdir(out))]
    return a + (["--activations", str(act), "--bank", str(bank)] if act else [])


def test_verify_passes_then_catches_tampering(tmp_path):
    act, bank, _ = fake_run(tmp_path)
    sd.main(make_args(act, bank, tmp_path / "o"))
    with pytest.raises(SystemExit) as e:
        sd.main(verify_args(tmp_path / "o", act, bank))
    assert e.value.code == 0
    z = dict(np.load(outdir(tmp_path / "o") / "draws.npz"))
    z["covariance"][0, 0] += 1e-3
    np.savez(outdir(tmp_path / "o") / "draws.npz", **z)
    with pytest.raises(SystemExit) as e:
        sd.main(verify_args(tmp_path / "o", act, bank))
    assert e.value.code == 1


def test_verify_catches_other_activations(tmp_path):
    act, bank, _ = fake_run(tmp_path, name="a1")
    other, _, _ = fake_run(tmp_path, seed=1, name="a2")
    sd.main(make_args(act, bank, tmp_path / "o"))
    with pytest.raises(SystemExit) as e:
        sd.main(verify_args(tmp_path / "o", other, bank))
    assert e.value.code == 1
