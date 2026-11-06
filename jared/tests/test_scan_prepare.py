"""Part B scan inputs (analysis/scan_prepare.py) on planted states over real bank records: E0 row
alignment by sample_uid (and its failures), the prompts, the steps and their random twins, the
forced factor at the answer slot, and merging the phrase states files. No E0 files or model."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

import scan_prepare as prep  # noqa: E402
from exp8_steering import selected_records  # noqa: E402
from steering_prepare import phrase_columns  # noqa: E402
from test_phrase_prepare import D, MODEL, planted_curve, prompts_meta, tok  # noqa: E402,F401
from spar_horizon.steering import curve_delta  # noqa: E402

LAYERS = (18, 14, 22)    # L14 and L18 in states.npz, L22 in a layer-specific file


class FakeE0:
    """E0's pooled labels and activations, in an order of its own (reversed), planted from the
    records the same way as the phrase states."""

    def __init__(self, records, columns, axes_by_cell, seed=3):
        rng = np.random.default_rng(seed)
        self.order = np.arange(len(records))[::-1]
        self.forced = rng.random(len(records)) < 0.3
        self.labels = {"sample_uids": np.array([records[i]["sample_uid"] for i in self.order]),
                       "y": columns["log_h"][self.order], "duration": columns["duration"][self.order],
                       "units": columns["factors"]["unit"][self.order], "regime": columns["regime"][self.order],
                       "near_first": columns["factors"]["order"][self.order], "forced": self.forced[self.order]}
        self.columns, self.axes_by_cell = columns, axes_by_cell

    def matrix(self, position, layer):
        axes = self.axes_by_cell[{"suffix newline": "suffix", "pre-label": "answer"}[position], layer]
        X = planted_curve(self.columns["duration"], axes) + 0.5 * np.outer(self.forced, np.ones(D))
        return X[self.order]


def write_phrase_states(directory, records, columns, prompts_meta):
    rng = np.random.default_rng(1)
    uids = np.array([r["sample_uid"] for r in records])
    axes = {}
    files = {"states": (14, 18), "states_L22": (22,)}
    for stem, layers in files.items():
        arrays = {"sample_uids": uids}
        for layer in layers:
            for role in ("number", "unit"):
                axes[role, layer] = np.linalg.qr(rng.normal(size=(D, 2)))[0].T
                X = planted_curve(columns["duration"], axes[role, layer])
                arrays[f"L{layer}|{role}"] = (X + 0.01 * rng.normal(size=X.shape)).astype(np.float32)
        np.savez(directory / f"{stem}.npz", **arrays)
    (directory / "prompts.json").write_text(json.dumps(prompts_meta))
    (directory / "states.json").write_text(json.dumps({"model": MODEL, "layers": [14, 18], "n": len(records),
                                                       "prompts_file": str(directory / "prompts.json")}))
    for position in ("suffix", "answer"):
        for layer in LAYERS:
            axes[position, layer] = np.linalg.qr(rng.normal(size=(D, 2)))[0].T
    return axes


@pytest.fixture(scope="module")
def planted(tmp_path_factory, prompts_meta):
    directory = tmp_path_factory.mktemp("exp14")
    records = selected_records(prompts_meta)
    columns = phrase_columns(records)
    axes = write_phrase_states(directory, records, columns, prompts_meta)
    e0 = FakeE0(records, columns, axes)
    out_dir = directory / "scan"
    prep.scan_main(directory, e0, out_dir, layers=LAYERS, per_transition=5, seed=0)
    return {"records": records, "columns": columns, "e0": e0, "axes": axes, "dir": directory, "out": out_dir,
            "meta": json.loads((out_dir / "inputs.json").read_text()), "steps": np.load(out_dir / "inputs.npz")}


def test_e0_rows_are_matched_by_sample_uid_whatever_their_order(planted):
    order = prep.e0_order(planted["records"], planted["columns"], planted["e0"].labels)
    assert [str(planted["e0"].labels["sample_uids"][i]) for i in order] == [r["sample_uid"] for r in planted["records"]]
    assert order[0] == len(order) - 1 and order[-1] == 0


def test_a_prompt_without_an_e0_row_or_a_repeated_uid_stops_the_prepare(planted):
    labels = {k: v[1:] for k, v in planted["e0"].labels.items()}
    with pytest.raises(ValueError, match="have no E0 row"):
        prep.e0_order(planted["records"], planted["columns"], labels)
    repeated = {k: np.concatenate([v, v[:1]]) for k, v in planted["e0"].labels.items()}
    with pytest.raises(ValueError, match="repeat a sample_uid"):
        prep.e0_order(planted["records"], planted["columns"], repeated)


@pytest.mark.parametrize("key", ["y", "duration", "units", "regime", "near_first"])
def test_a_matched_row_that_disagrees_with_the_record_stops_the_prepare(planted, key):
    labels = {k: v.copy() for k, v in planted["e0"].labels.items()}
    labels[key][5] = {"y": 99.0, "duration": 99.0, "units": "zzz", "regime": "zzz",
                      "near_first": not labels["near_first"][5]}[key]
    with pytest.raises(ValueError, match=f"disagree with the phrase prompts on '{key}'"):
        prep.e0_order(planted["records"], planted["columns"], labels)


def test_prompts_are_five_per_transition_interleaved_and_the_same_for_every_cell(planted):
    prompts = planted["meta"]["prompts"]
    assert [p["transition"] for p in prompts] == ["none-near", "both-near"] * 5
    assert [p["start_regime"] for p in prompts] == ["none", "both"] * 5
    assert all(p["target_regime"] == "near" for p in prompts)
    records = selected_records(planted["meta"])
    assert [r["sample_uid"] for r in records] == planted["meta"]["sample_uids"]
    assert len(set(planted["meta"]["sample_uids"])) == 10
    for step in planted["steps"].values():
        assert step.shape == (10, D)


def test_every_cell_has_a_manifold_step_and_a_random_twin_of_the_same_length(planted):
    steps = planted["steps"]
    assert sorted(steps.files) == sorted(f"step|L{layer}|{kind}|{position}" for layer in LAYERS
                                         for kind in ("manifold", "random") for position in prep.POSITIONS)
    for layer in LAYERS:
        for position in prep.POSITIONS:
            manifold, random = steps[f"step|L{layer}|manifold|{position}"], steps[f"step|L{layer}|random|{position}"]
            np.testing.assert_allclose(np.linalg.norm(random, axis=1), np.linalg.norm(manifold, axis=1), rtol=1e-4)
            cos = (manifold * random).sum(1) / (np.linalg.norm(manifold, axis=1) * np.linalg.norm(random, axis=1))
            assert np.abs(cos).max() < 0.8
    assert not np.allclose(steps["step|L18|random|number"], steps["step|L14|random|number"])


def test_manifold_steps_follow_the_planted_curve_toward_each_prompts_target(planted):
    prompts, steps = planted["meta"]["prompts"], planted["steps"]
    log_h, target = np.array([p["log_h"] for p in prompts]), np.array([p["target_log_h"] for p in prompts])
    for position, axis in (("number", planted["axes"]["number", 18][0]), ("suffix", planted["axes"]["suffix", 18][0])):
        along = steps[f"step|L18|manifold|{position}"] @ axis
        assert (np.sign(along) == np.sign(target - log_h)).all()
        # the planted curve moves 1 unit of the first axis per decade
        np.testing.assert_allclose(along, target - log_h, rtol=0.3, atol=0.3)


def test_suffix_and_answer_cells_are_fitted_on_the_e0_rows_aligned_to_the_records(planted):
    # misaligned rows (E0's reversed order used as is) would fit a different curve
    e0, columns = planted["e0"], planted["columns"]
    order = prep.e0_order(planted["records"], columns, e0.labels)
    fit, _ = prep.cell_fit("suffix", 18, None, e0, order, columns, prep.varying_factors(columns))
    log_h = np.array([p["log_h"] for p in planted["meta"]["prompts"]])
    target = np.array([p["target_log_h"] for p in planted["meta"]["prompts"]])
    np.testing.assert_allclose(curve_delta(fit, log_h, target - log_h).astype(np.float32),
                               planted["steps"]["step|L18|manifold|suffix"], atol=1e-4)


def test_only_the_answer_slot_fit_has_the_forced_close_factor(planted):
    dims = {(c["layer"], c["position"]): c["factor_span_dims"] for c in planted["meta"]["cells"]}
    for layer in LAYERS:
        assert dims[layer, "answer"] == dims[layer, "suffix"] + 1 == dims[layer, "number"] + 1


def test_the_report_and_meta_hold_the_in_span_share_of_every_cell(planted):
    cells = planted["meta"]["cells"]
    assert len(cells) == 3 * 4 and all(0 <= c["in_span_share"] <= 1 for c in cells)
    text = (planted["out"] / "fit_report.txt").read_text()
    assert "no significance claim" in text and "in-span share of the manifold step" in text
    assert all(f"L{layer} " in text for layer in LAYERS)
    assert planted["meta"]["layers"] == list(LAYERS) and planted["meta"]["positions"] == list(prep.POSITIONS)
    assert planted["meta"]["directions"] == ["manifold", "random"]


def test_phrase_states_merge_across_files_and_a_missing_layer_is_named(planted):
    arrays, uids = prep.load_phrase_states(planted["dir"], (14, 22))
    assert set(arrays) == {f"L{layer}|{role}" for layer in (14, 22) for role in ("number", "unit")}
    assert uids == [r["sample_uid"] for r in planted["records"]]
    with pytest.raises(FileNotFoundError, match="L10"):
        prep.load_phrase_states(planted["dir"], (10, 14))
