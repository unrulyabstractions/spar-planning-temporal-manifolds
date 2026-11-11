"""Summary of the phrase number patch (pipeline/s2_steering/analysis/patch_summary.py): what a trace states against the
donor, the per-unit counts, the criteria C0-C3 and their pending states, on synthetic rows."""

import json

import numpy as np
import pytest

import pipeline.s2_steering.analysis.patch_summary as ps
from spar_horizon.horizons import Horizon

LAYER, DEPTH = 16, [12, 22]
OWN, DONOR = Horizon(3, "weeks"), Horizon(8, "weeks")
STATES_DONOR = "So the time horizon is 8 weeks, which is short."
STATES_OWN = "So the time horizon is 3 weeks, which is short."
STATES_OTHER_UNIT = "So the time horizon is 8 months, which is long."
STATES_NOTHING = "Let me compare the two options."


def row(i, text=STATES_OWN, label="a", near_lo=0.0, forced=False, near_label="b"):
    return {"sample_uid": f"u{i}", "transition": "none-near", "start_regime": "none", "target_regime": "near",
            "horizon_text": str(OWN), "donor_horizon_text": str(DONOR), "log_h": OWN.log10_years,
            "target_log_h": DONOR.log10_years, "donor_log_h": DONOR.log10_years, "near_label": near_label,
            "near_delay_years": 0.0833, "far_delay_years": 30.0, "near_lo": near_lo, "label": label,
            "forced": forced, "think_len": 10, "think_text": text}


def write_unit(directory, name, rows):
    path = directory / "conditions" / f"{name.replace(' ', '_')}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"name": name, "rows": rows}))


def write_run(directory, units):
    """inputs.json plus the units {name: rows} the test wants finished."""
    (directory / "inputs.json").write_text(json.dumps({"patch_layer": LAYER, "depth_layers": DEPTH}))
    for name, rows in units.items():
        write_unit(directory, name, rows)


def rows_of(texts, **kwargs):
    return [row(i, text, **kwargs) for i, text in enumerate(texts)]


def test_stated_measures_compare_the_first_stated_horizon_with_the_donor_the_prompt_and_the_target():
    donor = ps.stated_measures(row(0, STATES_DONOR))
    assert donor["states_donor"] and donor["lands_target"] and donor["moved"] and donor["stated"]
    assert donor["shift"] == pytest.approx(DONOR.log10_years - OWN.log10_years)     # toward the donor is positive
    own = ps.stated_measures(row(0, STATES_OWN))
    assert not own["states_donor"] and not own["lands_target"] and not own["moved"] and own["shift"] == pytest.approx(0)
    other_unit = ps.stated_measures(row(0, STATES_OTHER_UNIT))
    assert not other_unit["states_donor"] and other_unit["moved"]
    nothing = ps.stated_measures(row(0, STATES_NOTHING))
    assert nothing["stated"] is False and np.isnan(nothing["shift"])


def test_a_shift_away_from_the_donor_is_negative():
    smaller = row(0, "So the time horizon is 1 week, which is short.")
    assert ps.stated_measures(smaller)["shift"] < 0


def test_unit_measures_count_what_is_stated_and_the_flips_against_the_baseline():
    base = [row(0, label="a", near_lo=-1.0), row(1, label="a", near_lo=-1.0), row(2, label="b", near_lo=1.0, near_label="b"),
            row(3, label="a", near_lo=-2.0, forced=True)]
    unit = [row(0, STATES_DONOR, "b", near_lo=1.0), row(1, STATES_OWN, "a", near_lo=-1.0),
            row(2, STATES_NOTHING, "b", near_lo=2.0, near_label="b"), row(3, STATES_DONOR, "b", near_lo=2.0)]
    m = ps.unit_measures(unit, base)
    assert (m["n"], m["n_stated"], m["states_donor"], m["lands_target"], m["moved"]) == (4, 3, 2, 2, 2)
    assert m["flips"] == 2 and m["flips_toward_near"] == 2     # both flips are to the near label "b"
    assert m["near_lo_shift"] == pytest.approx((2.0 + 0.0 + 1.0 + 4.0) / 4)
    assert m["n_natural"] == 3 and m["near_lo_shift_natural"] == pytest.approx(1.0)    # the forced pair is left out
    assert m["forced"] == 0
    assert m["shift_toward_donor"] == pytest.approx((2 * (DONOR.log10_years - OWN.log10_years) + 0) / 3)


def test_flips_to_the_near_label_are_counted():
    base = rows_of([STATES_OWN] * 3, label="a", near_label="b")
    unit = [row(0, label="b", near_label="b"), row(1, label="a", near_label="b"), row(2, label="b", near_label="b")]
    m = ps.unit_measures(unit, base)
    assert m["flips"] == 2 and m["flips_toward_near"] == 2


def name(source, positions, layer=LAYER):
    return ps.unit_name(source, positions, layer)


def measures(n, states_donor=0, lands_target=0, moved=0, flips=0):
    return {"n": n, "states_donor": states_donor, "lands_target": lands_target, "moved": moved, "flips": flips}


def test_c0_needs_zero_flips_of_the_self_patch():
    assert ps.evaluate_criteria({name("self", "phrase"): measures(24)}, LAYER)["C0"]["status"] == "pass"
    assert ps.evaluate_criteria({name("self", "phrase"): measures(24, flips=1)}, LAYER)["C0"]["status"] == "fail"


@pytest.mark.parametrize("lands,random_lands,status", [(12, 2, "pass"), (11, 2, "fail"), (12, 3, "fail"), (24, 0, "pass")])
def test_c1_thresholds_are_half_for_the_donor_patch_and_a_tenth_for_the_random(lands, random_lands, status):
    m = {name("donor", "number"): measures(24, lands_target=lands), name("random", "number"): measures(24, lands_target=random_lands)}
    assert ps.evaluate_criteria(m, LAYER)["C1"]["status"] == status


@pytest.mark.parametrize("manifold,status", [(0, "pass"), (3, "pass"), (4, "fail"), (10, "fail")])
def test_c2_compares_the_manifold_rate_with_a_quarter_of_the_donor_rate(manifold, status):
    m = {name("donor", "number"): measures(24, states_donor=16), name("manifold", "number"): measures(24, states_donor=manifold)}
    assert ps.evaluate_criteria(m, LAYER)["C2"]["status"] == status       # a quarter of 16 is 4: need < 4


def test_c2_is_undefined_when_the_donor_patch_never_states_its_number():
    m = {name("donor", "number"): measures(24), name("manifold", "number"): measures(24)}
    assert ps.evaluate_criteria(m, LAYER)["C2"]["status"] == "pending"


@pytest.mark.parametrize("moved,status", [(0, "pass"), (1, "pass"), (2, "fail")])
def test_c3_allows_one_pair_to_move_at_the_unit_token(moved, status):
    assert ps.evaluate_criteria({name("donor", "unit"): measures(24, moved=moved)}, LAYER)["C3"]["status"] == status


def test_criteria_are_pending_until_their_units_exist():
    criteria = ps.evaluate_criteria({}, LAYER)
    assert {c["status"] for c in criteria.values()} == {"pending"}
    assert "donor number L16" in criteria["C1"]["detail"] and "random number L16" in criteria["C1"]["detail"]


def test_a_summary_with_no_baseline_reports_nothing_to_compare(tmp_path):
    write_run(tmp_path, {})
    summary = ps.summarize(tmp_path)
    assert summary["baseline_done"] is False and summary["units"] == [] and summary["n_units_total"] == 11
    assert "no baseline unit yet" in ps.format_table(summary)


def test_a_partial_run_is_summarised_with_the_noise_floor_and_pending_criteria(tmp_path):
    base = rows_of([STATES_OWN] * 4, label="a")
    donor = rows_of([STATES_DONOR, STATES_DONOR, STATES_OWN, STATES_NOTHING], label="a")
    donor[0]["label"] = "b"
    rerun = rows_of([STATES_OWN] * 4, label="a")
    rerun[1]["label"] = "b"
    write_run(tmp_path, {"baseline": base, "rerun": rerun, name("donor", "number"): donor,
                         name("random", "number"): rows_of([STATES_OWN] * 4, label="a")})
    summary = ps.summarize(tmp_path)
    assert summary["n_units_done"] == 2 and summary["rerun_done"]
    assert summary["noise"]["flips"] == 1
    by_name = {u["name"]: u for u in summary["units"]}
    assert set(by_name) == {"donor number L16", "random number L16"} and by_name["donor number L16"]["condition"] == 3
    assert by_name["donor number L16"]["states_donor"] == 2 and by_name["donor number L16"]["flips"] == 1
    assert summary["criteria"]["C1"]["status"] == "pass" and summary["criteria"]["C0"]["status"] == "pending"
    text = ps.format_table(summary)
    assert "noise floor (rerun vs baseline): 1 of 4" in text and "C3: PENDING" in text


def test_main_writes_json_and_text(tmp_path, capsys):
    write_run(tmp_path, {"baseline": rows_of([STATES_OWN] * 2), name("self", "phrase"): rows_of([STATES_OWN] * 2)})
    ps.main(tmp_path)
    assert json.loads((tmp_path / "patch_summary.json").read_text())["criteria"]["C0"]["status"] == "pass"
    assert "criteria (fixed in spec" in (tmp_path / "patch_summary.txt").read_text()
    assert "C0: PASS" in capsys.readouterr().out
