"""Tests for curve_dim.py and diagnose.py's curve-dimensionality section on synthetic data (CPU, no model).
Run (from evaluation/): python -m pytest -q test_curve_dim.py"""

import json

import numpy as np

import curve_dim as CD
import diagnose as D
from test_diagnose import make_artifacts, run_diagnose

HORIZONS = [f"{d} days" for d in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)]
DIM = 48


def planted(seed=0, per_horizon=20, nuisance=100):
    """Activations = a 1-D curve in log10 horizon (bent, so one linear direction does not fit it) + a large nuisance
    direction that does not depend on the horizon + noise."""
    rng = np.random.default_rng(seed)
    groups = [h for h in HORIZONS for _ in range(per_horizon)]
    t = np.array([np.log10(float(g.split()[0])) for g in groups])
    axes = np.linalg.qr(rng.normal(size=(DIM, 3)))[0]
    curve = 20 * np.stack([t, 0.3 * (t - t.mean()) ** 2], 1) @ axes[:, :2].T
    nuisance_part = nuisance * rng.normal(size=(len(t), 1)) * axes[:, 2]
    return curve + nuisance_part + 0.3 * rng.normal(size=(len(t), DIM)), t, groups


def test_horizon_folds_hold_out_whole_horizons_and_partition_the_prompts():
    _, _, groups = planted()
    folds = CD.horizon_folds(groups)
    assert np.array_equal(np.sum(folds, axis=0), np.ones(len(groups)))
    for test in folds:
        held_out = {g for g, te in zip(groups, test) if te}
        assert held_out and held_out.isdisjoint(g for g, te in zip(groups, test) if not te)


def test_null_rejects_the_nuisance_dimension_and_keeps_the_curve():
    """The nuisance is the largest dimension of the means, but it is the same in the shuffled data: ratio ~1."""
    X, _, groups = planted()
    lam, null95 = CD.eigenvalues_against_null(X, groups, n_perm=30)
    ratio = lam / null95
    assert ratio[0] < 1.5 and lam[0] > lam[1]                # the nuisance leads the spectrum and is not horizon structure
    assert (ratio[1:3] > 3).all()                            # the two curve dimensions
    assert (ratio[3:6] < 1.5).all()


def test_null_finds_nothing_in_pure_noise():
    rng = np.random.default_rng(1)
    groups = [h for h in HORIZONS for _ in range(20)]
    lam, null95 = CD.eigenvalues_against_null(rng.normal(size=(len(groups), DIM)), groups, n_perm=30)
    assert 0.8 < np.median(lam / null95) < 1.2             # each component sits at its null; one may pass it by chance


def test_curve_and_ridge_recover_held_out_horizons_despite_a_per_prompt_nuisance():
    """A nuisance that varies from prompt to prompt (sd 30 against the curve's ~20 per decade) but averages out
    within each horizon's 20 prompts stays out of the curve's two components."""
    X, t, groups = planted(nuisance=30)
    res = CD.analyse(X, t, groups, ks=(1, 2), n_perm=10)
    assert res["curve"][2]["spearman"] > 0.9 and res["curve"][2]["median_abs_err"] < 0.4
    assert res["ridge"]["spearman"] > 0.9
    assert "curve at k=2" in CD.verdict(res, 2, k_more=1)


def test_ridge_matches_the_closed_form_on_a_noiseless_line():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 5))
    y = X @ np.array([1.0, -2.0, 0.5, 0, 3.0]) + 4
    pred = CD.ridge_predict(X[:40], y[:40], X[40:], alphas=np.array([1e-6]))
    assert np.allclose(pred, y[40:], atol=1e-3)


def test_section_renders_without_a_curve_bundle(tmp_path):
    """Fit rows of a single-direction bundle with horizons, activations present, no curve bundle."""
    art = make_artifacts(tmp_path / "art", letter_signal=0.0)
    bundle = art / "methods" / "caa__test"
    rows = [json.loads(l) for l in (bundle / "fit_uids.jsonl").read_text().splitlines()]
    X, t, groups = planted(per_horizon=len(rows) // len(HORIZONS))
    for r, g in zip(rows, groups):
        r["horizon_grid_value"] = g
    rows = rows[:len(groups)]
    (bundle / "fit_uids.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (art / "activations" / "index.jsonl").write_text("".join(json.dumps({"row": i, "uid": r["uid"]}) + "\n" for i, r in enumerate(rows)))
    np.save(art / "activations" / "L1_label.npy", X.astype(np.float32))
    for r in rows:                                   # near_only / both / none alternate, so `without none` has 2/3 of the rows
        assert r["regime"] in ("none", "near_only", "both")
    _, report = run_diagnose(tmp_path, "--artifacts", str(art))
    section = report["sections"]["Cd"]
    assert section["skipped"] is None and section["checks"] == []
    assert section["data"]["has_curve_bundle"] is False and section["data"]["curve_k_used"] == 4
    result = section["data"]["sites"]["label"]
    assert set(result) == {"all prompts", "without none"}
    assert result["all prompts"]["n_prompts"] == len(rows)
    assert result["without none"]["n_prompts"] < len(rows)
    assert any("curve at k=4" in line for line in section["data"]["verdicts"])


def test_section_is_skipped_when_fit_rows_have_no_horizons(tmp_path):
    art = make_artifacts(tmp_path / "art", letter_signal=0.0)
    _, report = run_diagnose(tmp_path, "--artifacts", str(art))
    assert report["sections"]["Cd"]["skipped"].startswith("skipped:")
    assert D.absent(tmp_path / "nowhere") is not None
