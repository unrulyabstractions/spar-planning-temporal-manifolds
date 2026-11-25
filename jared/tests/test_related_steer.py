"""E23 (EvalsRelatedQualities.md §3): the prepare step's row rules and behaviour sign, exp23 on the tiny
random Qwen3, and the report's arithmetic and rank test on hand-made results."""

import json

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pipeline.s1_data_vectors.analysis import related_steering_prepare as prep
from pipeline.s2_steering.analysis import related_report as report
from pipeline.s2_steering.experiments.ablate_steer import exp23_related_steer as e23
from spar_horizon import steering as S
from spar_horizon.config import Settings

D = 32
LABELS = (40, 41)
PAD = 0


def planted_cell(seed=0, n=60):
    """fit_cell on random states with four horizons, so the curve has a few dimensions."""
    rng = np.random.default_rng(seed)
    log_h = np.arange(n) % 4 - 1.0
    X = rng.normal(size=(n, D)) + np.outer(log_h, rng.normal(size=D))
    return S.fit_cell(X, log_h)


# --- prepare ----------------------------------------------------------------------------------------

def row(**over):
    base = {"sample_uid": "mwe_power_000_base", "set_id": "mwe_power", "variant": "base", "twin_id": "mwe_power_000",
            "split": "fit", "labels": ["a", "b"], "choice_prefix": "I choose:", "matching_label": "a",
            "act_file": "acts_0.npz", "act_row": 0, "answer": "a) because", "label_logits": [3.0, 1.0]}
    return {**base, **over}


@pytest.mark.parametrize("over, reason", [
    ({}, None),
    ({"labels": ["a", "b", "c"]}, "not an a/b row after 'I choose:'"),
    ({"act_file": None, "answer": None}, "generation too short to keep"),
    ({"answer": "Neither option"}, "answer does not start with a or b"),
    ({"answer": "b) since"}, None),
])
def test_unreadable_reason(over, reason):
    assert prep.unreadable_reason(row(**over)) == reason


def test_behaviour_log_odds_is_matching_minus_other():
    assert prep.behaviour_log_odds(row(matching_label="a")) == 2.0
    assert prep.behaviour_log_odds(row(matching_label="b")) == -2.0


def test_behaviour_caa_needs_ten_fit_rows_per_class_and_ignores_eval_rows():
    rng = np.random.default_rng(1)
    rows = ([row(set_id="s", split="fit")] * 24 + [row(set_id="s", split="eval")] * 10
            + [row(set_id="t", split="fit")] * 15)
    lo = np.r_[np.full(12, 1.0), np.full(12, -1.0), np.full(10, 5.0), np.full(10, 1.0), np.full(5, -1.0)]
    X = rng.normal(size=(len(rows), D))
    vectors, info = prep.behaviour_caa(rows, X, lo)
    assert info["s"] == {"available": True, "n_matching": 12, "n_other": 12}
    assert info["t"] == {"available": False, "n_matching": 10, "n_other": 5}
    assert set(vectors) == {"s"}
    np.testing.assert_allclose(vectors["s"], X[:12].mean(0) - X[12:24].mean(0))


def write_run(root, cell):
    """A related run with two sets: four rows each (one item in both orders, fit and eval), one of them
    unreadable, plus the cell file related_cells writes."""
    run = root / "related_m_think-on"
    rng = np.random.default_rng(2)
    for set_id in ("mwe_power", "sorh"):
        (run / set_id).mkdir(parents=True)
        rows = []
        for k, split in enumerate(("fit", "eval")):
            for variant, matching in (("base", "a"), ("swap", "b")):
                rows.append(row(sample_uid=f"{set_id}_{k:03d}_{variant}", set_id=set_id, variant=variant,
                                twin_id=f"{set_id}_{k:03d}", split=split, matching_label=matching, act_row=len(rows)))
        rows[-1]["answer"] = "I cannot choose"
        (run / set_id / "rows.json").write_text(json.dumps(rows))
        acts = rng.normal(size=(4, 3, D))
        np.savez(run / set_id / "acts_0.npz", L24=acts, region_prelabel=np.array([1, 2]))
    np.savez(root / "cell.npz", **{k: np.asarray(v) for k, v in cell.items()})
    (root / "cell.json").write_text(json.dumps({"layer": 24}))
    return run


def test_prepare_main_round_trips_through_load_inputs(tmp_path, monkeypatch):
    cell = planted_cell()
    run = write_run(tmp_path, cell)
    monkeypatch.setattr(prep, "eval_settings", lambda _run: Settings())
    monkeypatch.setattr(prep.AutoTokenizer, "from_pretrained", lambda _name: None)
    # the cached generation of row 2 of sorh is not E0's: dropped with that reason
    monkeypatch.setattr(prep, "on_sequence", lambda _tok, r, _cfg, _acts: None if r["sample_uid"] == "sorh_001_base"
                        else ([1, 2, 3, len(r["sample_uid"])], 2, LABELS))
    prep.main(tmp_path, tmp_path / "out", cell_path=tmp_path / "cell.npz", model="x/m")
    inputs = prep.load_inputs(tmp_path / "out" / "inputs_m_related.npz")
    uids = [r["sample_uid"] for r in inputs["meta"]["rows"]]
    assert uids == ["mwe_power_000_base", "mwe_power_000_swap", "mwe_power_001_base", "sorh_000_base",
                    "sorh_000_swap"]
    reasons = {u["sample_uid"]: u["reason"] for u in inputs["meta"]["unreadable"]}
    assert reasons == {"mwe_power_001_swap": "answer does not start with a or b",
                       "sorh_001_base": "cached generation is not the one the activations were taken on",
                       "sorh_001_swap": "answer does not start with a or b"}
    assert inputs["near_first"] == [True, False, True, True, False]
    assert inputs["ids"][0] == [1, 2, 3, len("mwe_power_000_base")]
    # the start horizon is the curve readout of the row's own pre-label state (region_prelabel[0] = 1)
    acts = np.load(run / "mwe_power" / "acts_0.npz")
    np.testing.assert_allclose(inputs["log_h"][:3], S.curve_readout(cell, acts["L24"][:3, 1]))
    assert inputs["meta"]["caa"]["mwe_power"]["available"] is False and inputs["caa"] == {}


# --- exp23 ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=D, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=64, head_dim=8)
    return Qwen3ForCausalLM(cfg).eval()


def tiny_inputs():
    seqs = [[3, 5, 7, 9, 11, 13], [4, 6, 8, 10], [7, 7, 9, 12, 15], [5, 9, 2, 8]]
    rows = [{"sample_uid": f"s_{i // 2:03d}_{v}", "set_id": "s", "variant": v, "twin_id": f"s_{i // 2:03d}",
             "split": "eval"} for i, v in enumerate(("base", "swap") * 2)]
    return {"cell": planted_cell(), "log_h": np.array([0.0, 0.5, -0.5, 1.0]), "ids": seqs,
            "near_first": [True, False, True, False], "labels": [LABELS] * 4, "meta": {"rows": rows}}


def test_exp23_conditions_base_and_probes_on_the_tiny_model(tiny):
    inputs = tiny_inputs()
    idx = [0, 1, 2, 3]
    caa = np.random.default_rng(5).normal(size=D)
    rng = np.random.default_rng(6)
    probes = [(3, rng.normal(size=D), rng.normal(size=D), 0.5)]
    rows, names = e23.run_set(tiny, "cpu", PAD, inputs, idx, caa, layer=2, k=3, probes=probes, batch_size=2)
    assert names == ["curve -2", "curve +2", *(f"random {s} {a:+d}" for s in range(3) for a in (-2, 2)),
                     "caa -2", "caa +2"]
    unedited = S.score_last_edits(tiny, "cpu", inputs["ids"], inputs["near_first"], inputs["labels"], PAD, 2, [None])
    np.testing.assert_allclose([r["base"] for r in rows], unedited[0], atol=1e-5)
    for r in rows:
        assert set(r["probe"]["L3"]) == {"base", *names}
        assert all(np.isfinite(r[name]) for name in names)
    # the edit moves the outcome and the probe read after the edited layer
    assert max(abs(r["curve +2"] - r["base"]) for r in rows) > 1e-4
    assert max(abs(r["probe"]["L3"]["curve +2"] - r["probe"]["L3"]["base"]) for r in rows) > 1e-4


def test_exp23_without_caa_or_probes(tiny):
    rows, names = e23.run_set(tiny, "cpu", PAD, tiny_inputs(), [0, 1], None, layer=2, k=1, probes=None,
                              batch_size=2)
    assert names == ["curve -2", "curve +2", "random 0 -2", "random 0 +2"]
    assert "probe" not in rows[0]


def test_limited_keeps_the_first_rows_per_set_and_variant():
    rows = [{"set_id": s, "variant": v} for s in ("a", "b") for _ in range(3) for v in ("base", "swap")]
    assert e23.limited(rows, 2) == [0, 1, 2, 3, 6, 7, 8, 9]
    assert e23.limited(rows, None) == list(range(12))


# --- report -----------------------------------------------------------------------------------------

def test_rank_test_counts_ties_against_the_statistic():
    assert report.rank_test(1.0, [0.5, -1.0, 2.0])["p"] == 0.75    # |-1| and |2| reach 1
    assert report.rank_test(-3.0, [0.5, 1.0, 2.0]) == {"stat": -3.0, "sign": -1, "p": 0.25, "random_abs_median": 1.0}


def test_parts_split_a_response_into_odd_and_even():
    values = {"base": np.array([1.0]), "x +2": np.array([4.0]), "x -2": np.array([0.0])}
    odd, even = report.parts(values, "x")
    assert odd.tolist() == [2.0] and even.tolist() == [1.0]


def hand_results(n_items=20, k=19, curve_odd=0.0, seed=0, flip_order=0):
    """exp23-style rows for one set (eval split, both orders) with a planted curve odd part, and the
    matching inputs meta."""
    rng = np.random.default_rng(seed)
    conditions = [f"{d} {a:+d}" for d in ["curve", *(f"random {s}" for s in range(k)), "caa"] for a in (-2, 2)]
    random_odd = rng.normal(scale=0.1, size=k)
    rows = []
    for i in range(n_items):
        base = 1.0 + rng.normal(scale=0.1)
        for variant in ("base", "swap"):
            r = {"sample_uid": f"s_{i:03d}_{variant}", "set_id": "s", "variant": variant, "twin_id": f"s_{i:03d}",
                 "split": "eval", "log_h": 0.0,
                 "base": -base if (variant == "swap" and i < flip_order) else base}
            for c in conditions:
                direction, alpha = c.rsplit(" ", 1)
                sign = 1 if alpha == "+2" else -1
                odd = (curve_odd if direction == "curve" else 1.0 if direction == "caa"
                       else random_odd[int(direction.split()[1])])
                r[c] = r["base"] + sign * odd + rng.normal(scale=0.01)
            rows.append(r)
    results = {"inputs": "x", "layer": 24, "k": k, "limit": None, "probe": None,
               "sets": {"s": {"conditions": conditions, "rows": rows}}}
    meta = {"rows": [{k_: r[k_] for k_ in ("set_id", "twin_id", "variant", "split")} for r in rows],
            "unreadable": [], "caa": {"s": {"available": True, "n_matching": 10, "n_other": 10}}}
    inputs = {"meta": meta, "cell": planted_cell(), "caa": {"s": np.ones(D)}}
    return results, inputs


def test_report_detects_a_planted_curve_effect_and_not_a_null():
    results, inputs = hand_results(curve_odd=1.0)
    s = report.analyse(results, inputs)["sets"]["s"]
    assert s["gate"] and s["readable"] == 1.0 and s["order"] == 1.0 and s["behaviour_rate"] == 1.0
    assert s["tests"]["curve"]["odd"]["p"] == 0.05 and s["tests"]["curve"]["odd"]["sign"] == 1
    assert s["has_power"] is True
    null = report.analyse(*hand_results(curve_odd=0.0))["sets"]["s"]
    assert null["tests"]["curve"]["odd"]["p"] > 0.5


def test_report_gate_fails_on_order_disagreement_and_skips_the_tests():
    s = report.analyse(*hand_results(curve_odd=1.0, flip_order=8))["sets"]["s"]
    assert s["order"] == 0.6 and s["gate"] is False and s["tests"] is None
    md = report.markdown(report.analyse(*hand_results(curve_odd=1.0, flip_order=8)))
    assert "**FAIL**" in md and "not tested" in md
