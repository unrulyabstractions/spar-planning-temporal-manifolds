import json
import math

import numpy as np
import pytest

import pipeline.s3_analysis.by_label.exp14_analysis as ea

NEAR_DELAY, FAR_DELAY = 1.0, 10.0
PER_TRANSITION = 40
START_REGIME = {"both-near": "both", "none-near": "none", "near-both": "near", "near-none": "near"}
TARGET_REGIME = {"both-near": "near", "none-near": "near", "near-both": "both", "near-none": "none"}
# a duration in the target regime that is not near either delay
TARGET_TEXT = {"near": "the horizon is 3 years", "both": "the horizon is 50 years", "none": "the horizon is 0.3 years"}
NO_DURATION_TEXT = "the horizon is far off, so the choice is clear"


def base_log_odds(i):
    return 0.5 * i + 0.1 * ((i % 5) - 2)


# the prompt's own horizon per start regime, and the target per transition (log10 years), as the driver writes them
START_HORIZON = {"both": ("30 years", math.log10(30)), "none": ("6 months", math.log10(0.5)),
                 "near": ("5 years", math.log10(5))}
TARGET_LOG_H = {"both-near": 0.5, "none-near": 0.5, "near-both": 2.0, "near-none": -1.0}


def make_row(transition, condition, i, text, near_lo, label=0, forced=False):
    horizon_text, log_h = START_HORIZON[START_REGIME[transition]]
    return {"sample_uid": f"{transition}-{i}", "transition": transition, "condition": condition,
            "target_regime": TARGET_REGIME[transition], "start_regime": START_REGIME[transition],
            "horizon_text": horizon_text, "log_h": log_h, "target_log_h": TARGET_LOG_H[transition],
            "near_delay_years": NEAR_DELAY, "far_delay_years": FAR_DELAY, "near_lo": near_lo, "near_label": "a",
            "label": label, "forced": forced, "think_text": text, "names_own": False, "names_target": False}


def write_cell(directory, transition, condition, rows):
    name = f"{transition}__{condition.replace(' ', '_')}"
    (directory / "conditions").mkdir(parents=True, exist_ok=True)
    (directory / "conditions" / f"{name}.json").write_text(
        json.dumps({"name": name, "transition": transition, "condition": condition, "rows": rows}))


def write_inputs(directory, clamped=()):
    directory.mkdir(exist_ok=True)
    uids, prompts = [], []
    for transition in ea.TRANSITIONS:
        for i in range(PER_TRANSITION):
            uids.append(f"{transition}-{i}")
            prompts.append({"transition": transition, "opposite_clamped": f"{transition}-{i}" in clamped})
    (directory / "inputs.json").write_text(json.dumps({
        "sample_uids": uids, "prompts": prompts,
        "cells": [{"layer": 16, "role": "unit", "n": 500, "k": 12, "factor_span_dims": 9, "in_span_share": 0.44}]}))


def plant(directory, shares, shifts, transitions=ea.TRANSITIONS, clamped=()):
    """Write inputs.json and the condition files. shares[condition] is the share of traces that write
    a target-regime duration; shifts[condition] is the ORIENTED change in near log-odds from noop
    (so the raw change has the transition's sign). Conditions not in `shares` are not written."""
    write_inputs(directory, clamped)
    for transition in transitions:
        write_cell(directory, transition, "noop", [
            make_row(transition, "noop", i, NO_DURATION_TEXT, base_log_odds(i)) for i in range(PER_TRANSITION)])
        for condition, share in shares.items():
            if condition == "opposite L16" and transition not in ea.FROM_NONE_OR_BOTH:
                continue
            shift = shifts[condition]
            n_writing = round(share * PER_TRANSITION)
            write_cell(directory, transition, condition, [
                make_row(transition, condition, i,
                         TARGET_TEXT[TARGET_REGIME[transition]] if i < n_writing else NO_DURATION_TEXT,
                         base_log_odds(i) + ea.SIGN[transition] * shift)
                for i in range(PER_TRANSITION)])


ALL_CONDITIONS = ["effects L16", "random L16", "opposite L16", "donor L16", "invariant L16",
                  "effects L14", "effects L18", "random L14", "random L18"]


def passing_world(**overrides):
    shares = {c: 0.0 for c in ALL_CONDITIONS}
    shifts = {c: 0.0 for c in ALL_CONDITIONS}
    shares.update({"effects L16": 0.6, "donor L16": 0.8, "invariant L16": 0.5, "effects L14": 0.4, "effects L18": 0.1})
    shifts.update({"effects L16": 6.0, "donor L16": 12.0, "invariant L16": 4.0, "opposite L16": -6.0,
                   "effects L14": 3.5, "effects L18": 0.0})
    shares.update(overrides.get("shares", {}))
    shifts.update(overrides.get("shifts", {}))
    return shares, shifts


def run(tmp_path, world, **kwargs):
    plant(tmp_path, *world, **kwargs)
    return ea.analyse(tmp_path)


def test_a_planted_pass_passes_h1_h2_h3(tmp_path):
    summary = run(tmp_path, passing_world())
    assert summary["verdicts"] == {"H1": "pass", "H2": "pass", "H3": "pass"}
    assert summary["h1"]["share"]["mean"] == pytest.approx(0.6)
    assert summary["h2"]["change"]["mean"] == pytest.approx(6.0)
    assert summary["h3"]["against_opposite"]["difference"]["mean"] == pytest.approx(12.0)
    assert summary["pending"] == []


def test_h1_fails_when_the_share_is_low_when_random_is_high_and_when_the_ci_overlaps_random(tmp_path):
    low = run(tmp_path / "low", passing_world(shares={"effects L16": 0.2}))
    assert low["verdicts"]["H1"] == "fail"
    high_random = run(tmp_path / "random", passing_world(shares={"random L16": 0.2}))
    assert high_random["verdicts"]["H1"] == "fail"       # random above 0.10 although the difference is positive
    assert high_random["h1"]["difference"]["lo"] > 0
    overlap = run(tmp_path / "overlap", passing_world(shares={"effects L16": 0.35, "random L16": 0.3}))
    assert overlap["verdicts"]["H1"] == "fail"           # share >= 0.30, but not above random
    assert overlap["h1"]["difference"]["lo"] < 0.1


def test_h2_fails_when_the_oriented_change_is_below_015_c(tmp_path):
    summary = run(tmp_path, passing_world(shifts={"effects L16": 2.0}))
    assert summary["verdicts"]["H2"] == "fail"
    assert summary["verdicts"]["H1"] == "pass"


def test_h3_fails_when_the_opposite_step_does_the_same(tmp_path):
    summary = run(tmp_path, passing_world(shifts={"opposite L16": 6.0}))
    assert summary["h3"]["against_opposite"]["pass"] is False
    assert summary["verdicts"]["H3"] == "fail"


def test_h3_fails_when_a_near_reverse_raises_the_near_log_odds(tmp_path):
    plant(tmp_path, *passing_world())
    # near -> none rises: a monotone push has opposite signs on the two reverses
    rows = [make_row("near-none", "effects L16", i, TARGET_TEXT["none"], base_log_odds(i) + 6.0)
            for i in range(PER_TRANSITION)]
    write_cell(tmp_path, "near-none", "effects L16", rows)
    summary = ea.analyse(tmp_path)
    assert summary["h3"]["near_starts"]["near-none"]["pass"] is False
    assert summary["h3"]["near_starts"]["near-both"]["pass"] is True
    assert summary["verdicts"]["H3"] == "fail"


def test_orientation_signs(tmp_path):
    summary = run(tmp_path, passing_world())
    cells = {(c["transition"], c["condition"]): c for c in summary["cells"]}
    # a raw rise of 6 on none/both starts and a raw fall of 6 on near starts are both +6 oriented
    for transition in ea.TRANSITIONS:
        assert cells[(transition, "effects L16")]["oriented_near_log_odds_change"] == pytest.approx(6.0)
    # the same raw rise on a near start is oriented negative
    plant(tmp_path / "raw", *passing_world(shifts={"effects L16": -6.0}))
    wrong_way = ea.analyse(tmp_path / "raw")
    assert wrong_way["h2"]["change"]["mean"] == pytest.approx(-6.0)
    assert wrong_way["verdicts"]["H2"] == "fail"
    near_both = wrong_way["h3"]["near_starts"]["near-both"]["change"]["mean"]
    assert near_both == pytest.approx(6.0)               # unoriented: raw


def test_the_change_is_paired_by_sample_uid_not_by_row_order(tmp_path):
    plant(tmp_path, *passing_world())
    rows = [make_row("both-near", "effects L16", i, NO_DURATION_TEXT, base_log_odds(i) + 4.0)
            for i in range(PER_TRANSITION)]
    write_cell(tmp_path, "both-near", "effects L16", rows[::-1])
    _, data = ea.load(tmp_path)
    changes = ea.near_log_odds_change(data, "effects L16", ("both-near",))
    assert list(changes.values()) == pytest.approx([4.0] * PER_TRANSITION)
    assert len(changes) == PER_TRANSITION


def test_label_flips_are_counted_against_noop(tmp_path):
    plant(tmp_path, *passing_world())
    rows = [make_row("both-near", "effects L16", i, NO_DURATION_TEXT, 0.0, label=int(i < 10))
            for i in range(PER_TRANSITION)]
    write_cell(tmp_path, "both-near", "effects L16", rows)
    cell = next(c for c in ea.analyse(tmp_path)["cells"]
                if (c["transition"], c["condition"]) == ("both-near", "effects L16"))
    assert cell["label_flips_vs_noop"] == pytest.approx(10 / PER_TRANSITION)


def test_a_duration_within_ten_percent_of_a_delay_is_not_counted():
    near_target = make_row("both-near", "effects L16", 0, "so the delay of 1.05 years is short", 0.0)
    assert ea.writes_target_regime(near_target) is False        # 1.05 y is in the near regime, but is the near delay
    assert ea.trace_measures(near_target)["no_horizon"] is True
    outside = make_row("both-near", "effects L16", 0, "so the delay of 1.2 years is short", 0.0)
    assert ea.writes_target_regime(outside) is True             # 20% from the delay
    edge = make_row("both-near", "effects L16", 0, "so the delay of 1.09 years is short", 0.0)
    assert ea.writes_target_regime(edge) is False               # 9% from the delay
    none_target = make_row("near-none", "effects L16", 0, "maybe 0.95 years then", 0.0)
    assert ea.writes_target_regime(none_target) is False
    mixed = make_row("both-near", "effects L16", 0, "the near option pays in 1 year, the far in 10 years; "
                     "the horizon is 36 months", 0.0)
    assert ea.writes_target_regime(mixed) is True               # 3 years, written in months
    assert ea.trace_measures(mixed)["no_horizon"] is False


def test_only_the_target_regime_counts():
    row = make_row("both-near", "effects L16", 0, "the horizon is 50 years", 0.0)
    assert ea.writes_target_regime(row) is False                # a both-regime horizon, target is near


def test_side_effect_measures():
    conflict = make_row("both-near", "effects L16", 0, "Wait, the horizon is 3 years? Maybe a typo.", 0.0, forced=True)
    measures = ea.trace_measures(conflict)
    assert measures["conflict"] and measures["forced"] and measures["writes_target"]
    quiet = ea.trace_measures(make_row("both-near", "effects L16", 0, "the horizon is 3 years", 0.0))
    assert not quiet["conflict"] and not quiet["forced"]


def test_the_clamped_opposite_prompts_can_be_left_out(tmp_path):
    clamped = {f"both-near-{i}" for i in range(10)}
    plant(tmp_path, *passing_world(), clamped=clamped)
    # on the clamped prompts the opposite step did nothing, which inflates effects - opposite
    rows = [make_row("both-near", "opposite L16", i, NO_DURATION_TEXT, base_log_odds(i) + (0.0 if i < 10 else -6.0))
            for i in range(PER_TRANSITION)]
    write_cell(tmp_path, "both-near", "opposite L16", rows)
    summary = ea.analyse(tmp_path)
    with_clamped = summary["h3"]["against_opposite"]["difference"]
    without = summary["h3"]["against_opposite_without_clamped"]["difference"]
    assert without["n"] == with_clamped["n"] - 10
    assert without["mean"] == pytest.approx(12.0)
    assert with_clamped["mean"] < without["mean"]


def test_donor_ceiling_ratio_and_layer_profile(tmp_path):
    summary = run(tmp_path, passing_world())
    assert summary["reported"]["donor_ceiling"]["h1_share_ratio"] == pytest.approx(0.6 / 0.8)
    assert summary["reported"]["donor_ceiling"]["h2_change_ratio"] == pytest.approx(0.5)
    profile = summary["reported"]["layer_profile"]
    assert profile["L14"]["h1"]["pass"] is True and profile["L14"]["h2"]["pass"] is True
    assert profile["L18"]["h1"]["pass"] is False and profile["L18"]["h2"]["pass"] is False
    assert summary["reported"]["invariant_L16"]["h1"]["pass"] is True
    assert summary["reported"]["in_span_share"][0]["in_span_share"] == 0.44


def test_missing_conditions_are_pending_not_pass_or_fail(tmp_path):
    plant(tmp_path, {"effects L16": 0.6}, {"effects L16": 6.0}, transitions=("both-near",))
    summary = ea.analyse(tmp_path)
    assert summary["verdicts"] == {"H1": "pending", "H2": "pending", "H3": "pending"}
    assert "near-both__random_L16" in summary["pending"]
    assert summary["reported"]["layer_profile"]["L14"]["h1"]["pass"] is None
    ea.main(tmp_path)
    text = (tmp_path / "summary.txt").read_text()
    assert text.splitlines()[0].startswith("H1 pending")
    assert (tmp_path / "summary.json").exists()


def test_a_complete_failing_criterion_stays_fail_when_another_is_pending(tmp_path):
    # H1 only needs effects and random at L16, so it can be judged without the opposite step
    shares, shifts = passing_world(shares={"effects L16": 0.1})
    for condition in ("opposite L16", "donor L16", "invariant L16"):
        del shares[condition]
    plant(tmp_path, shares, shifts)
    verdicts = ea.analyse(tmp_path)["verdicts"]
    assert verdicts["H1"] == "fail" and verdicts["H3"] == "pending"


def test_the_text_leads_with_the_three_verdicts(tmp_path):
    plant(tmp_path, *passing_world())
    ea.main(tmp_path)
    lines = (tmp_path / "summary.txt").read_text().splitlines()
    assert [line.split(":")[0] for line in lines if line[:2] in ("H1", "H2", "H3")] == ["H1 pass", "H2 pass", "H3 pass"]
    assert lines[0].startswith("H1")
    assert json.loads((tmp_path / "summary.json").read_text())["verdicts"]["H3"] == "pass"


# ---- post hoc: the first stated horizon ----

def test_the_first_stated_horizon_skips_other_durations_and_reads_each_phrasing():
    def years(text):
        stated = ea.first_stated_horizon(text)
        return None if stated is None else round(stated[0], 4)
    assert years("That leaves 3 years to grow. Over a 13-year time horizon, b) wins.") == 13
    assert years("Wait, the horizon is 2.5 years, not 3 years.") == 2.5
    assert years("so for a 30-second horizon nothing pays") == round(30 / (365.25 * 86400), 4)
    assert years("a time horizon of fifty-nine years") == 59
    assert years("The time horizon given is 33 hours, about a week.") == round(33 / (365.25 * 24), 4)
    assert years(NO_DURATION_TEXT) is None


def test_stated_measures_place_the_stated_horizon_against_the_own_and_the_target():
    # near-none: own 5 years, target 0.1 years; delays 1 and 10 years
    def measures(text):
        return ea.stated_measures(make_row("near-none", "effects L16", 0, text, 0.0))
    landed = measures("the horizon is 5 months")
    assert landed["stated_regime"] == "none" and landed["states_target"]
    assert landed["stated_toward"] and landed["stated_unit_changed"]
    fewer_digits = measures("a 2-year time horizon")
    assert fewer_digits["stated_toward"] and not fewer_digits["stated_unit_changed"]
    assert not fewer_digits["states_target"]
    away = measures("the horizon is 50 years")
    assert away["stated_moved"] and not away["stated_toward"]
    own = measures("over a 5-year time horizon")
    assert not own["stated_moved"] and own["stated_regime"] == "near"
    none = measures(NO_DURATION_TEXT)
    assert none["stated_regime"] is None and not none["stated_moved"]


def test_the_choice_is_split_by_where_the_stated_horizon_lands(tmp_path):
    # both-near (sign +): noop states its own 30 years and chooses far; two steered traces state
    # 3 years (target regime) and choose near, two keep the own horizon and choose far
    write_inputs(tmp_path)
    own, landed = "the horizon is 30 years", "the horizon is 3 years"
    write_cell(tmp_path, "both-near", "noop", [make_row("both-near", "noop", i, own, -5.0, "b") for i in range(4)])
    write_cell(tmp_path, "both-near", "effects L16", [
        make_row("both-near", "effects L16", i, landed if i < 2 else own, 5.0 if i < 2 else -5.0, "a" if i < 2 else "b")
        for i in range(4)])
    write_cell(tmp_path, "both-near", "donor L16", [make_row("both-near", "donor L16", i, landed, 5.0, "a")
                                                    for i in range(4)])
    _, data = ea.load(tmp_path)
    by_landing = ea.choice_by_landing(data)
    groups = by_landing["groups"]
    assert groups["lands in target regime"]["p_near"]["mean"] == 1.0
    assert groups["lands in target regime"]["near_log_odds"]["mean"] == 10.0
    assert groups["lands in target regime"]["prompts"] == 2
    assert groups["own or none stated"]["p_near"] == {"n": 2, "mean": 0.0, "lo": 0.0, "hi": 0.0}
    assert groups["steered, not landed"]["p_near"]["n"] == 2
    assert groups["moved, not into target"]["p_near"]["n"] == 0
    assert groups["donor L16"]["p_near"]["mean"] == 1.0 and groups["donor L16"]["p_near"]["n"] == 4
    # the landed traces (prompts 0 and 1) against the donor patch of those two prompts only
    same = by_landing["landed_vs_donor_same_prompts"]
    assert same["landed"]["prompts"] == same["donor"]["prompts"] == 2
    assert same["donor"]["near_log_odds"]["mean"] == 10.0
    assert by_landing["by_transition"]["both-near"]["landed, same prompts"]["p_near"]["mean"] == 1.0
    assert by_landing["by_transition"]["both-near"]["random, same prompts as landed"]["prompts"] == 0   # no random run
    by_regime = ea.choice_by_stated_regime(data)
    assert by_regime["own, both"]["n"] == 6 and by_regime["own, both"]["mean"] == 0.0
    assert by_regime["moved, near"]["n"] == 6 and by_regime["moved, near"]["mean"] == 1.0


def test_main_writes_the_summary_and_the_figure_on_a_partial_run(tmp_path):
    plant(tmp_path, *passing_world(), transitions=("both-near",))
    ea.main(tmp_path)
    assert (tmp_path / "summary.txt").read_text().startswith("H1 pending")
    assert (tmp_path / "summary.png").stat().st_size > 0


def test_the_clustered_interval_resamples_prompts_not_traces():
    # two prompts, each seen twice: resampling prompts can draw one prompt twice, so the interval
    # reaches both prompt means; resampling the four traces would rarely draw all four the same
    values, prompts = [1.0, 1.0, 0.0, 0.0], ["a", "a", "b", "b"]
    stat = ea.clustered_mean(values, prompts)
    assert stat["mean"] == 0.5 and stat["lo"] == 0.0 and stat["hi"] == 1.0
    assert ea.clustered_mean([], [])["n"] == 0


def test_a_digit_edit_adds_or_drops_digits_of_the_prompt_number():
    assert ea.is_digit_edit("66", "6") and ea.is_digit_edit("13", "213") and ea.is_digit_edit("2.5", "25")
    assert not ea.is_digit_edit("3", "91") and not ea.is_digit_edit("39", "39")
    assert not ea.is_digit_edit("fifty", "5")
    row = make_row("both-near", "effects L16", 0, "over a 3-year time horizon", 0.0)   # own 30 years
    measures = ea.stated_measures(row)
    assert measures["digit_edit"] and np.isclose(measures["stated_shift"], -1.0)
    unit_change = ea.stated_measures(make_row("both-near", "effects L16", 0, "a 3-month time horizon", 0.0))
    assert unit_change["stated_unit_changed"] and not unit_change["digit_edit"]


FLAT_FIT = {f"{role} {term}": [value] for role in ("near", "far")
            for term, value in (("slope", 0.0), ("slope change above 0", 0.0), ("step", 0.0))}


def test_the_hinge_map_jumps_where_a_delay_falls_inside_the_horizon():
    fit = {**FLAT_FIT, "far step": [-10.0], "far slope change above 0": [2.0]}
    # far delay 10 years: a 30-year horizon has it inside (step), a 1-year horizon has it 1 decade outside (slope)
    inside = ea.hinge_near_log_odds(fit, np.log10(30), 0.0, 1.0)
    outside = ea.hinge_near_log_odds(fit, 0.0, 0.0, 1.0)
    assert np.isclose(inside, -10.0) and np.isclose(outside, 2.0)


def test_within_prompt_landing_and_the_h2_split_and_the_behaviour_prediction(tmp_path):
    # both-near: on each of 4 prompts one steered trace lands (states 3 years, +10) and one keeps 30 years (0)
    write_inputs(tmp_path)
    own, landed = "the horizon is 30 years", "the horizon is 3 years"
    write_cell(tmp_path, "both-near", "noop", [make_row("both-near", "noop", i, own, 0.0, "b") for i in range(4)])
    write_cell(tmp_path, "both-near", "effects L14", [make_row("both-near", "effects L14", i, landed, 10.0, "a")
                                                      for i in range(4)])
    write_cell(tmp_path, "both-near", "effects L16", [make_row("both-near", "effects L16", i, own, 0.0, "b")
                                                      for i in range(4)])
    _, data = ea.load(tmp_path)
    within = ea.within_prompt_landing(data)["both-near"]
    assert within["n"] == 4 and within["mean"] == 10.0
    split = ea.h2_by_landing(data, "effects L14", ("both-near",))
    assert split["landed"]["n"] == 4 and split["landed"]["share_of_total"] == 1.0
    fit = {**FLAT_FIT, "far step": [-10.0]}   # a horizon past the far delay (10 years) lowers near by 10
    prediction = ea.behaviour_prediction(data, fit)["both-near"]
    assert np.isclose(prediction["lands in target regime"]["predicted"], 10.0)   # 30 -> 3 years crosses the far delay
    assert prediction["lands in target regime"]["observed"] == 10.0
    assert np.isclose(prediction["own or none stated"]["predicted"], 0.0)


def test_natural_closes_drop_forced_traces_everywhere_and_landed_effects_split_by_condition(tmp_path):
    write_inputs(tmp_path)
    own, landed = "the horizon is 30 years", "the horizon is 3 years"
    write_cell(tmp_path, "both-near", "noop", [make_row("both-near", "noop", i, own, 0.0, "b", forced=i == 0)
                                               for i in range(3)])
    write_cell(tmp_path, "both-near", "effects L16", [make_row("both-near", "effects L16", i, landed, 6.0, "a",
                                                               forced=i == 1) for i in range(3)])
    _, data = ea.load(tmp_path)
    natural = ea.natural_closes(data)
    assert sorted(natural[("both-near", "noop")]) == ["both-near-1", "both-near-2"]
    assert sorted(natural[("both-near", "effects L16")]) == ["both-near-0", "both-near-2"]
    fit = {**FLAT_FIT, "far step": [-10.0]}
    split = ea.landed_by_condition(natural, fit)
    assert split["both-near | effects L16"]["n"] == 1          # only prompt 2 is natural in both runs
    assert split["both-near | effects L16"]["observed"] == 6.0
    assert np.isclose(split["both-near | effects L16"]["predicted"], 10.0)
