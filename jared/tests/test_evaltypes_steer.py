"""E19: which eval rows are steered, the inputs round trip, and the analysis on hand-made results."""

import json

import numpy as np

from pipeline.s1_data_vectors.analysis.evaltypes_steering_prepare import eval_rows, pack, steerable
from pipeline.s2_steering.experiments.ablate_steer.exp19_evaltypes_steer import limited, load
from pipeline.s3_analysis.eval_types_steer import analyse, baseline, crossing
from spar_horizon.eval_types import build


def test_steerable_rows(tmp_path):
    for type_id in ("t0_w0", "json_io", "loss_framing", "roi_unequal", "free_form", "city_values"):
        (tmp_path / type_id).mkdir()
        rows = [dict(r, act_file="acts_0.npz", act_row=i) for i, r in enumerate(build(type_id))]
        (tmp_path / type_id / "rows.json").write_text(json.dumps(rows))
    rows, left_out = eval_rows(tmp_path)
    assert set(rows) == {"t0_w0", "city_values"}
    assert set(left_out) >= {"json_io", "loss_framing", "roi_unequal", "free_form"}
    assert all(r["regime"] in ("near", "both") for t in rows.values() for r in t)
    city = rows["city_values"]
    assert {r["variant"] for r in city} == {"base", "swap", "twin"}
    assert len(city) == 16 + 8 + 4
    assert not any(steerable(r) for r in build("json_io") if r["variant"] != "twin")


def test_pack():
    ids, offsets = pack([[1, 2, 3], [4], [5, 6]])
    assert ids.tolist() == [1, 2, 3, 4, 5, 6] and offsets.tolist() == [0, 3, 4, 6]


def write_inputs(tmp_path):
    d = 8
    cell = {"mean": np.zeros(d), "w": np.ones(d), "basis": np.eye(d)[:2], "curve_logh": np.array([0.0, 1.0]),
            "curve_coords": np.zeros((2, 2)), "caa": np.ones(d), "b": 0.0}
    arrays = {f"{m}|pre-label|24|{k}": np.asarray(v) for m in ("off", "on") for k, v in cell.items()}
    rows = [{"sample_uid": f"t_{i:03d}_base", "type_id": "t", "variant": "base", "horizon": 1.0, "near_label": "a",
             "regime": "near"} for i in range(3)] + [{"sample_uid": "t_000_twin", "type_id": "t", "variant": "twin",
                                                      "horizon": 1.0, "near_label": "a", "regime": "near"}]
    for mode, keep in (("off", [0, 1, 2, 3]), ("on", [0, 2])):
        arrays[f"{mode}|ids"], arrays[f"{mode}|offsets"] = pack([[10 + i, 20 + i] for i in keep])
        arrays[f"{mode}|row"] = np.array(keep)
        arrays[f"{mode}|labels"] = np.array([(64, 65)] * len(keep))
    np.savez(tmp_path / "inputs_x.npz", **arrays)
    meta = {"model": "m", "rows": rows,
            "cells": [{"mode": m, "label": "pre-label", "layer": 24, "k": 2} for m in ("off", "on")]}
    (tmp_path / "inputs_x.json").write_text(json.dumps(meta))
    return tmp_path / "inputs_x.npz"


def test_load_and_limit(tmp_path):
    meta, cells, seq = load(write_inputs(tmp_path))
    assert cells["off"]["basis"].shape == (2, 8) and cells["on"]["b"] == 0.0
    ids, labels, rows = seq["on"]
    assert ids == [[10, 20], [12, 22]] and rows == [0, 2] and labels == [(64, 65)] * 2
    small = limited(meta, seq, 1)
    assert small["off"][2] == [0, 3]     # first base row and the twin
    assert small["on"][2] == [0]


def row(uid, regime, variant, base, curve_minus, curve_plus, random_minus=None, random_plus=None):
    return {"sample_uid": uid, "type_id": "t", "variant": variant, "twin_id": uid.rsplit("_", 1)[0],
            "regime": regime, "near_label": "a", "base": base, "curve -2": curve_minus, "curve +2": curve_plus,
            "random 0 -2": base if random_minus is None else random_minus,
            "random 0 +2": base if random_plus is None else random_plus}


def test_crossing_and_baseline():
    rows = [row("t_000_base", "near", "base", 6.0, 6.0, -2.0),     # near, steered longer: flips to far
            row("t_000_swap", "near", "swap", 4.0, 4.0, 2.0),      # moves toward far, no flip
            row("t_001_base", "both", "base", -6.0, 1.0, -6.0),    # both, steered shorter: flips to near
            row("t_001_swap", "both", "swap", 2.0, 3.0, 2.0)]      # wrong already: chose near, no flip count
    b = baseline(rows)
    assert b["accuracy"] == 0.75 and b["order"] == 0.5 and b["n_order_items"] == 2
    assert b["contrast"] == (6 + 4) / 2 - (-6 + 2) / 2
    assert not b["gate"]
    c = crossing(rows, "curve")
    assert [t for t, _ in c] == [8.0, 2.0, 7.0, 1.0]
    assert [f for _, f in c] == [True, False, True, None]


def test_analyse_gate_and_specific():
    rng = np.random.default_rng(0)
    rows, meta_rows = [], []
    for i in range(12):
        regime = "near" if i % 2 == 0 else "both"
        base = 5.0 if regime == "near" else -5.0
        # the curve moves every crossing row 2 logits toward the new answer; random moves nothing
        r = row(f"t_{i:03d}_base", regime, "base", base + rng.normal(0, 0.1),
                base + (2.0 if regime == "both" else 0.0), base - (2.0 if regime == "near" else 0.0))
        rows.append({k: r[k] for k in ("sample_uid", "base", "curve -2", "curve +2", "random 0 -2", "random 0 +2")})
        meta_rows.append({k: r[k] for k in ("sample_uid", "type_id", "variant", "twin_id", "regime", "near_label")})
    results = {"modes": {"off": {"rows": rows + [dict(rows[0], sample_uid="t_000_twin")]}}}
    meta = {"rows": meta_rows + [dict(meta_rows[0], sample_uid="t_000_twin", variant="twin")]}
    out = analyse(results, meta, {"types": {}})["modes"]["off"]
    t = out["types"]["t"]
    assert t["baseline"]["gate"] is False          # order agreement undefined: no swap rows
    contrast = t["baseline"]["contrast"]
    assert abs(t["steer"]["toward_curve"]["mean"] - 2.0 / contrast) < 0.01
    assert t["steer"]["toward_random"]["mean"] == 0.0
    assert abs(t["steer"]["specific"]["mean"] - 2.0 / contrast) < 0.01
    assert t["steer"]["flips"] == {"n": 12, "k": 0, "rate": 0.0}
    assert out["twins_pooled"]["baseline"]["n"] == 1
