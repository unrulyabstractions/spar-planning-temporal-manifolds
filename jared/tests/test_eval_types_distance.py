"""Distance from Alan's prompt: pairing of rows with their originals, and the scaled cosine distance
on synthetic hidden states."""

import numpy as np

from pipeline.s3_analysis.eval_types_distance import (cell_distances, cosine_distance, match_originals,
                                                      original_prompts, originals_scale)
from spar_horizon.eval_types import build


def test_cosine_distance():
    a = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    b = np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]])
    assert np.allclose(cosine_distance(a, b), [0.0, 1.0, 2.0])


def test_match_originals():
    rows = {t: build(t) for t in ("t0_w0", "t0_w3_clear", "city_values", "free_form")}
    match = match_originals(rows)
    # wording ladder: same item and variant in t0_w0
    for r in rows["t0_w3_clear"]:
        orig, how = match[r["sample_uid"]]
        assert orig == r["sample_uid"].replace("t0_w3_clear", "t0_w0") and how == "same item in t0_w0"
    assert all(match[r["sample_uid"]] == (r["sample_uid"], "is original (t0_w0)") for r in rows["t0_w0"])
    # twin types: only base rows of twin items pair, with their own item's twin
    city = rows["city_values"]
    paired = {u: o for u, (o, how) in match.items() if how == "twin"}
    twin_items = {r["twin_id"] for r in city if r["variant"] == "twin"}
    assert len(twin_items) == 4 and len(paired) == 4
    assert all(o == u.replace("_base", "_twin") for u, o in paired.items())
    assert all(match[r["sample_uid"]][1] == "unpaired" for r in city
               if r["variant"] == "swap" or r["twin_id"] not in twin_items)
    assert all(match[r["sample_uid"]] == (None, "unpaired") for r in rows["free_form"])


def test_original_prompts_dedupe():
    # system_prompt_clear shares items with system_prompt, so their twins are the same four prompts
    rows = {t: build(t) for t in ("system_prompt", "system_prompt_clear", "t0_w0")}
    originals = original_prompts(rows)
    assert len(originals) == 4 + len(rows["t0_w0"])
    assert len({r["prompt"] for r in originals}) == len(originals)


def test_cell_distances_scale():
    rng = np.random.default_rng(0)
    d = 64
    # four originals, two per regime; a row equal to its original; a row equal to another original
    O = rng.normal(size=(4, d))
    X = np.vstack([O, O[0], O[1]])
    uids = ["o0", "o1", "o2", "o3", "same", "other"]
    regimes = ["near", "near", "both", "both"]
    matched = {"o0": None, "o1": None, "o2": None, "o3": None, "same": "o0", "other": "o0"}
    paired, unpaired, scale = cell_distances(X, uids, uids[:4], regimes, matched)
    Xc = X - X.mean(axis=0)
    expected_scale = np.median([cosine_distance(Xc[0], Xc[1]), cosine_distance(Xc[2], Xc[3])])
    assert np.isclose(scale, expected_scale)
    assert paired["same"] == 0.0
    # "other" sits on o1, so its distance to o0 is exactly the (o0, o1) original pair's distance
    assert np.isclose(paired["other"], cosine_distance(Xc[0], Xc[1]) / scale)
    assert set(paired) == {"same", "other"}
    assert np.isclose(np.median([unpaired[u] for u in uids[:4]]), 1.0)


def test_originals_scale_same_regime_only():
    X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    # identical within each regime, orthogonal across: only within-regime pairs count
    assert originals_scale(X, ["near", "near", "both", "both"]) == 0.0


def test_project_spaces():
    from pipeline.s3_analysis.eval_types_distance import fit_bank_spaces, project
    rng = np.random.default_rng(1)
    d, n = 32, 200
    # a bank whose horizon lives along axis 0 and whose other variance spans axes 1-3 only
    log_h = rng.uniform(-2, 1.5, n)
    Xb = np.zeros((n, d))
    Xb[:, 0] = 3 * log_h
    Xb[:, 1:4] = rng.normal(size=(n, 3))
    bank = fit_bank_spaces(Xb, log_h)
    inside = np.zeros((1, d))
    inside[0, 0] = 3.0
    outside = np.zeros((1, d))
    outside[0, 10] = 5.0
    p = project(bank, np.vstack([inside, outside]) + bank["pca_mean"])
    assert p["bank_pca"].shape[1] == bank["pca_basis"].shape[0] <= 4
    assert np.isclose(p["outside"][0], 0.0, atol=1e-4) and np.isclose(p["outside"][1], 1.0, atol=1e-2)
    # the probe reads log horizon off axis 0, so a step of 3 along it is about one decade
    assert abs((p["readout"][0] - p["readout"][1]) - 1.0) < 0.1
    # the horizon subspace contains axis 0
    assert np.abs(bank["cell"]["basis"][:, 0]).max() > 0.99
