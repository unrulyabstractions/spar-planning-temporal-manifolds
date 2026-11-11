"""Inputs of the phrase number patch (analysis/patch_prepare.py, 27_spec_phrase_number_patch.md): the
same-unit crossing pairs, their balance and order, the steps and their random twins, on states
planted over real bank records. The 0.8B tokenizer is the only download."""

import json

import numpy as np
import pytest

from pipeline.s1_data_vectors.analysis import patch_prepare as prep
from pipeline.s2_steering.experiments.ablate_steer.exp8_steering import selected_records
from spar_horizon.config import Settings
from spar_horizon.full_bank import HORIZONS, regime, render_variant
from spar_horizon.steering import span_part
from pipeline.s1_data_vectors.analysis.steering_prepare import steerable
from test_phrase_prepare import GENERATION, prompts_meta, tok, write_states  # noqa: F401

PAIRS = 4


@pytest.fixture(scope="module")
def cfg():
    return Settings(**GENERATION)


@pytest.fixture(scope="module")
def run(tmp_path_factory, prompts_meta):
    directory = tmp_path_factory.mktemp("exp14")
    write_states(directory, prompts_meta)
    out_dir = tmp_path_factory.mktemp("exp16")
    prep.patch_main(directory, out_dir, pairs=PAIRS, seed=0)
    meta = json.loads((out_dir / "inputs.json").read_text())
    return {"out": out_dir, "states": directory / "states.npz", "meta": meta, "records": selected_records(meta), "arrays": np.load(out_dir / "inputs.npz")}


def test_pairs_are_balanced_and_interleaved(run):
    transitions = [p["transition"] for p in run["meta"]["prompts"]]
    assert transitions == ["none-near", "both-near"] * (PAIRS // 2)
    assert [p["start_regime"] for p in run["meta"]["prompts"]] == ["none", "both"] * (PAIRS // 2)


def test_every_pair_is_same_unit_equal_length_and_crosses_into_near(run, tok, cfg):
    for rec, p in zip(run["records"], run["meta"]["prompts"]):
        near, far = rec["near_delay_years"], rec["far_delay_years"]
        donor = next(h for h in HORIZONS if str(h) == p["donor_horizon_text"])
        assert steerable(rec) and rec["regime"] == p["start_regime"] != "near"
        assert regime(donor.years, near, far) == "near" == p["target_regime"]
        assert p["donor_horizon_text"].split()[-1] == rec["horizon_text"].split()[-1]
        assert prep.phrase_length(tok, cfg, rec["prompt"], rec["horizon_text"]) == prep.phrase_length(
            tok, cfg, render_variant({**rec, "horizon_text": str(donor)}), str(donor))
        assert p["donor_log_h"] == pytest.approx(donor.log10_years) == pytest.approx(p["target_log_h"])
        assert p["log_h"] == pytest.approx(np.log10(rec["horizon"]))


def test_the_donor_is_the_valid_horizon_nearest_the_midpoint_of_the_delays(run, tok, cfg):
    for rec, p in zip(run["records"], run["meta"]["prompts"]):
        valid = [h for h in HORIZONS if str(h).split()[-1] == rec["horizon_text"].split()[-1]
                 and regime(h.years, rec["near_delay_years"], rec["far_delay_years"]) == "near"
                 and prep.phrase_length(tok, cfg, render_variant({**rec, "horizon_text": str(h)}), str(h))
                 == prep.phrase_length(tok, cfg, rec["prompt"], rec["horizon_text"])]
        best = min(abs(h.log10_years - p["midpoint_log_h"]) for h in valid)
        assert abs(p["donor_log_h"] - p["midpoint_log_h"]) == pytest.approx(best)


def test_a_prompt_with_no_same_unit_near_horizon_has_no_donor(tok, cfg, run):
    rec = next(r for r in run["records"] if r["horizon_text"].endswith("years"))
    squeezed = {**rec, "near_delay_years": 1e-5, "far_delay_years": 2e-5}   # near is 5-10 minutes: no horizon in years
    assert prep.same_unit_donor(squeezed, -4.8, tok, cfg) is None


def test_too_few_pairs_or_an_odd_count_is_an_error(tmp_path, prompts_meta):
    write_states(tmp_path, prompts_meta)
    with pytest.raises(ValueError, match="same-unit donor"):
        prep.patch_main(tmp_path, tmp_path / "out", pairs=2000)
    with pytest.raises(ValueError, match="balanced"):
        prep.patch_main(tmp_path, tmp_path / "out", pairs=5)


def test_arrays_have_the_expected_keys_and_shapes(run):
    z = run["arrays"]
    assert set(z.files) == {"step|L16|manifold|number", "step|L16|manifold|unit", "step|L16|random|number",
                            "step|L16|inspan|number", "basis|L16"}
    assert z["basis|L16"].shape == (8, z["step|L16|manifold|number"].shape[1])
    assert all(z[k].shape[0] == PAIRS and z[k].dtype == np.float32 for k in z.files if k != "basis|L16")
    assert run["meta"]["patch_layer"] == 16 and run["meta"]["depth_layers"] == [12, 22]


def test_random_twins_are_as_long_as_the_manifold_step_and_the_inspan_one_lies_in_the_subspace(run):
    z, basis = run["arrays"], run["arrays"]["basis|L16"]
    manifold = z["step|L16|manifold|number"]
    length = np.linalg.norm(manifold, axis=1)
    assert length.min() > 0
    for kind in ("random", "inspan"):
        np.testing.assert_allclose(np.linalg.norm(z[f"step|L16|{kind}|number"], axis=1), length, rtol=1e-4)
    inspan = z["step|L16|inspan|number"]
    np.testing.assert_allclose(span_part(inspan, basis), inspan, atol=1e-4 * length.max())
    np.testing.assert_allclose(span_part(manifold, basis), manifold, atol=1e-4 * length.max())
    random = z["step|L16|random|number"]
    assert np.linalg.norm(span_part(random, basis)) < 0.5 * np.linalg.norm(random)    # 8 of 48 dims
    assert not np.allclose(random, inspan)


def test_the_manifold_step_follows_the_planted_curve_from_the_horizon_to_the_donor(run):
    """The planted duration curve runs along axes[0]: the step's component there has the sign of the move."""
    axis = np.load(run["states"])["axes"][0]
    move = np.array([p["donor_log_h"] - p["log_h"] for p in run["meta"]["prompts"]])
    along = run["arrays"]["step|L16|manifold|number"] @ axis
    assert np.all(np.abs(move) > 0)
    assert np.all(np.sign(along) == np.sign(move))
