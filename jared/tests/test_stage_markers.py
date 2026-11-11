"""Crossing logic of analysis/stage_markers.py on toy curves."""

import json

import numpy as np
import pytest

from pipeline.s1_data_vectors.analysis.stage_markers import first_crossing, last_crossing, verdict_s1
from pipeline.s2_steering.analysis.steering_analysis import saved_decision


def test_first_crossing_gives_layer_and_interpolated_position():
    curve = [np.nan, 0.0, 0.1, 0.3, 0.7, 0.9]
    assert first_crossing(curve, 0.5) == (4, pytest.approx(3.5))


def test_first_crossing_none_when_never_reached():
    assert first_crossing([np.nan, 0.1, 0.2], 0.5) is None


def test_last_crossing_is_the_last_layer_at_or_above_threshold():
    curve = [np.nan, 1.0, 0.8, 0.6, 0.4, 0.0, 0.0]
    assert last_crossing(curve, 0.5) == (3, pytest.approx(3.5))


def markers(n, off, on):
    return {"n_layers": n, "off": {k: (v, float(v)) for k, v in off.items()}, "on": {k: (v, float(v)) for k, v in on.items()}}


def test_s1_scales_by_relative_depth():
    off8 = dict(encode_end=18, spread_peak=19, handoff=24, bind=27)
    on8 = dict(handoff=24, bind=27)
    scaled = {
        "Qwen3-8B": markers(36, off8, on8),
        "Qwen3-14B": markers(40, {k: round(v * 40 / 36) for k, v in off8.items()}, {k: round(v * 40 / 36) for k, v in on8.items()}),
    }
    assert verdict_s1(scaled)["verdict"] == "PASS"


def test_s1_fails_when_the_order_inverts():
    off8 = dict(encode_end=18, spread_peak=19, handoff=24, bind=27)
    off14 = dict(encode_end=18, spread_peak=23, handoff=30, bind=27)
    s1 = verdict_s1({"Qwen3-8B": markers(36, off8, dict(handoff=24, bind=27)),
                     "Qwen3-14B": markers(40, off14, dict(handoff=27, bind=30))})
    assert s1["verdict"] == "FAIL" and s1["order_inverted_in"] == ["off"]


def test_saved_decision_drops_rows_whose_label_logits_are_the_wrong_pair(tmp_path):
    # the third prompt wrote `**b)**` (lead 4): its saved ' a'/' b' logits are not the a-vs-b choice
    answers = [{"near_label": "a", "label": "a"}, {"near_label": "b", "label": "a"}, {"near_label": "a", "label": "b"}]
    (tmp_path / "answers.json").write_text(json.dumps(answers))
    z = {"horizons": np.array([1.0, 2.0, 3.0]), "lead": np.array([3, 3, 4]),
         "label_logits": np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 5.0]])}
    ab, near, chose_near = saved_decision(tmp_path, z)
    assert ab[:2].tolist() == [2.0, 1.0] and np.isnan(ab[2]) and np.isnan(near[2])
    assert chose_near.tolist() == [True, False, False]
