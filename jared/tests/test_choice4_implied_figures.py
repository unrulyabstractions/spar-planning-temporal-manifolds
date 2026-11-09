import json

import pytest

from pipeline.s3_analysis.figures import choice4_implied_figures as figs


def test_wilson_matches_known_value():
    lo, hi = figs.wilson_interval(1, 7)
    assert lo == pytest.approx(0.0257, abs=1e-3)
    assert hi == pytest.approx(0.5131, abs=1e-3)


def test_wilson_stays_inside_unit_interval_at_extremes():
    assert figs.wilson_interval(0, 40)[0] == 0.0
    assert figs.wilson_interval(40, 40)[1] == 1.0
    assert figs.wilson_interval(0, 40)[1] > 0.0


def test_wilson_empty_sample_is_uninformative():
    assert figs.wilson_interval(0, 0) == (0.0, 1.0)


def test_significance_text_follows_p():
    assert "n.s." in figs.significance_text(0.34)
    assert "n.s." not in figs.significance_text(0.01)


def position(n):
    return {"n": n, "pick_rate": [0.25] * 4, "truth_rate": [0.25] * 4, "late_pick": 0.5, "excess_late": 0.0}


def accuracy(exact, n):
    return {"n": n, "exact": exact, "half": exact, "within_half": n}


def fake_c4():
    return {
        "reference": {"natural": {"accuracy": accuracy(9, 10), "position": position(10)},
                      "forced": {"accuracy": accuracy(6, 12), "position": position(12)}},
        "unit2": {"n": 12, "closed": [8, 12], "closed_by_regime": {"both": [5, 8], "near": [3, 4]},
                  "closed_by_loop": {"loop": [2, 6], "no loop": [6, 6]},
                  "accuracy_closed": accuracy(7, 8), "original_accuracy_closed": accuracy(5, 8),
                  "wrong_to_right": 3, "right_to_wrong": 1, "still_open_accuracy": accuracy(2, 4),
                  "position_closed": position(8), "position_original_same_prompts": position(8)},
        "unit3": {"attempted": 4, "closed_in_unit3": 0},
    }


def fake_implied():
    cell = {"n": 16, "correct": 12}
    family = {f"{arm} {regime}": cell for arm in ("implied", "explicit") for regime in ("near", "both")}
    family["mcnemar"] = {"pairs": 32, "p": 0.2}
    site_cell = {"n": 20, "mae": 0.5, "rho": 0.7}
    sites = ["think@1 L16", "prelabel L16", "suffix_1 L16"]
    return {
        "behaviour": {f: dict(family) for f in "ABC"},
        "conversion": {f: {"n": 40, "converted": 38, "flips_regime": 2} for f in "ABC"},
        "mediation": {"flips": 6, "follows_stated": 1, "follows_share": 1 / 6},
        "readout": {f"{site} {f}": {"implied": site_cell, "explicit": site_cell} for site in sites for f in "ABC"},
    }


def test_main_writes_all_six_figures_from_json(tmp_path):
    results = tmp_path / "horizon"
    (results / "exp17_Qwen3-8B_think_budget").mkdir(parents=True)
    (results / "exp18_Qwen3-8B_think-on_implied").mkdir(parents=True)
    (results / "exp17_Qwen3-8B_think_budget" / "think_budget_summary.json").write_text(json.dumps({"choice4": fake_c4()}))
    (results / "exp18_Qwen3-8B_think-on_implied" / "implied_summary.json").write_text(json.dumps(fake_implied()))
    out = tmp_path / "out"

    figs.main([str(results), str(out)])

    assert sorted(p.name for p in out.glob("*.png")) == [
        "choice4_1_errors_by_close.png", "choice4_2_position_bias.png", "choice4_3_more_room.png",
        "implied_1_behaviour.png", "implied_2_conversion.png", "implied_3_readout.png"]
