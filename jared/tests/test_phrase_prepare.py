"""E14's inputs (analysis/steering_prepare.py --phrase, 21_spec_phrase_steering.md): targets,
eligibility, the stratified draw, opposite steps, the step kinds and donors, on states
planted over real bank records. The 0.8B tokenizer is the only download."""

import json

import numpy as np
import pytest
from transformers import AutoTokenizer

from pipeline.s1_data_vectors.analysis.steering_prepare import (
    PHRASE_LAYERS, TRANSITIONS, allocate, eligible_rows, opposite_targets, phrase_columns, phrase_main,
    steerable, stratified_draw, transition_target)
from pipeline.s2_steering.experiments.ablate_steer.exp8_steering import selected_records
from spar_horizon.banks import build_records
from spar_horizon.config import Settings
from spar_horizon.full_bank import regime
from spar_horizon.horizons import Horizon
from spar_horizon.model_io import chat_text
from spar_horizon.steering import phrase_positions, span_part

MODEL = "Qwen/Qwen3.5-0.8B"
D = 48
PER_TRANSITION = 4
GENERATION = {"model": MODEL, "thinking": "on", "n_response": 6, "max_think_tokens": 64, "prefill": "",
              "choice_prefix": "I choose:", "answer_window": 48}
BANK_SETTINGS = {"full": {"n_prompts": 500, "seed": 0, "limit": None},
                 "units": {"n_prompts": 500, "seed": 0, "limit": 140}}
RANGE = (Horizon(30, "seconds").log10_years, np.log10(500.0))   # the curve: 30 s to 500 y


def planted_curve(duration, axes):
    return np.outer(duration, axes[0]) + 0.05 * np.outer(duration ** 2, axes[1])


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def prompts_meta():
    banks, selected_banks, selected, uids = list(BANK_SETTINGS), [], [], []
    for bank in banks:
        records = build_records(Settings(bank=bank, **BANK_SETTINGS[bank]))
        for i, rec in enumerate(records):
            if rec["horizon"] is None:
                continue
            selected_banks.append(bank)
            selected.append(i)
            uids.append(rec["sample_uid"])
    return {"model": MODEL, "generation": GENERATION, "bank_settings": BANK_SETTINGS, "selected": selected,
            "selected_banks": selected_banks, "sample_uids": uids}


def write_states(directory, prompts_meta, limit=None):
    """States = planted duration curve + unit, regime, order and delay offsets + a little noise."""
    records = selected_records(prompts_meta)[:limit]
    columns = phrase_columns(records)
    rng = np.random.default_rng(1)
    arrays = {"sample_uids": np.array([r["sample_uid"] for r in records]), "n_number": np.full(len(records), 2)}
    for layer in (12, *PHRASE_LAYERS):
        for role in ("number", "unit"):
            axes = np.linalg.qr(rng.normal(size=(D, 2)))[0].T
            offsets = {name: {level: rng.normal(size=D) for level in np.unique(values)}
                       for name, values in columns["factors"].items()}
            X = planted_curve(columns["duration"], axes)
            for name, values in columns["factors"].items():
                X += np.array([offsets[name][v] for v in values])
            X += np.outer(columns["log_near"], rng.normal(size=D)) + np.outer(columns["log_far"], rng.normal(size=D))
            arrays[f"L{layer}|{role}"] = (X + 0.01 * rng.normal(size=X.shape)).astype(np.float32)
            if layer == 16 and role == "number":
                arrays["axes"] = axes
    np.savez(directory / "states.npz", **arrays)
    (directory / "prompts.json").write_text(json.dumps(prompts_meta))
    (directory / "states.json").write_text(json.dumps({"model": MODEL, "layers": [12, 14, 16, 18],
                                                       "prompts_file": str(directory / "prompts.json"),
                                                       "n": len(records), "code_hash": "test"}))
    return records


@pytest.fixture(scope="module")
def run(tmp_path_factory, prompts_meta):
    directory = tmp_path_factory.mktemp("exp14")
    records = write_states(directory, prompts_meta)
    phrase_main(directory, per_transition=PER_TRANSITION, seed=0)
    return {"dir": directory, "records": records, "meta": json.loads((directory / "inputs.json").read_text()),
            "steps": np.load(directory / "inputs.npz"), "axes": np.load(directory / "states.npz")["axes"]}


def drawn_records(run):
    return selected_records(run["meta"])


def test_targets_per_transition():
    log_near, log_far = np.array([-1.0, 0.0]), np.array([1.0, 2.0])
    assert transition_target("near", log_near, log_far) == pytest.approx([0.0, 1.0])
    assert transition_target("both", log_near, log_far) == pytest.approx([2.0, 3.0])
    assert transition_target("none", log_near, log_far) == pytest.approx([-2.0, -1.0])
    assert [(name, start, target) for name, start, target in TRANSITIONS] == [
        ("both-near", "both", "near"), ("none-near", "none", "near"), ("near-both", "near", "both"),
        ("near-none", "near", "none")]


def test_eligibility_needs_start_regime_steerable_range_and_a_delay_gap():
    columns = {"regime": np.array(["none"] * 6 + ["near"] * 4),
               "log_near": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.5, 1.0, 1.0, 0.0]),
               "log_far": np.array([0.59, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.6])}
    steerable_rows = np.array([True, True, True, False, True, True, True, True, True, True])
    # row 0: gap 0.59; row 1: gap 0.6, eligible; row 3: not steerable; row 4: fine
    assert eligible_rows(columns, steerable_rows, "none", "near", RANGE).tolist() == [1, 2, 4, 5]
    # near-none targets log near - 1: row 6 is at -6.5 under the curve; near-both: row 8's target 3 is above it
    assert eligible_rows(columns, steerable_rows, "near", "none", RANGE).tolist() == [7, 8, 9]
    assert eligible_rows(columns, steerable_rows, "near", "both", RANGE).tolist() == [6, 7, 9]
    assert eligible_rows(columns, steerable_rows, "near", "both", (-6.0, 3.5)).tolist() == [6, 7, 8, 9]


def test_drawn_prompts_are_eligible(run):
    records, meta = drawn_records(run), run["meta"]
    assert len(records) == 4 * PER_TRANSITION
    for rec, prompt in zip(records, meta["prompts"]):
        name = prompt["transition"]
        _, start, target_regime = next(t for t in TRANSITIONS if t[0] == name)
        log_near, log_far = np.log10(rec["near_delay_years"]), np.log10(rec["far_delay_years"])
        assert steerable(rec) and rec["regime"] == start == prompt["start_regime"]
        assert prompt["target_log_h"] == pytest.approx(transition_target(target_regime, log_near, log_far))
        assert RANGE[0] <= prompt["target_log_h"] <= RANGE[1]
        assert prompt["log_h"] == pytest.approx(np.log10(rec["horizon"]))
        if target_regime == "near":
            assert log_far - log_near >= 0.6 - 1e-9


def test_eligible_counts_match_the_records(run):
    columns = phrase_columns(run["records"])
    steerable_rows = np.array([steerable(r) for r in run["records"]])
    grid = (columns["duration"].min(), columns["duration"].max())
    expected = {name: len(eligible_rows(columns, steerable_rows, start, target, grid)) for name, start, target in TRANSITIONS}
    assert run["meta"]["eligible"] == expected
    assert all(count > PER_TRANSITION for count in expected.values())


def test_allocation_is_proportional_has_a_minimum_of_one_and_sums_to_the_total():
    assert allocate([300, 100, 1], 32).tolist() == [23, 8, 1]
    assert allocate([100, 100], 32).tolist() == [16, 16]
    assert allocate([0, 50, 2], 8).tolist() == [0, 7, 1]
    assert allocate([40, 2, 2], 3).tolist() == [1, 1, 1]
    assert allocate([40, 2, 2], 2).sum() == 2
    assert allocate([3, 2], 10).tolist() == [3, 2]   # a short transition takes everything


def test_stratified_draw_is_seed_stable_proportional_and_without_replacement():
    banks = np.array(["full"] * 60 + ["units"] * 30 + ["fill"] * 3)
    rows = np.arange(len(banks))
    first = stratified_draw(rows, banks, 12, np.random.default_rng(0))
    again = stratified_draw(rows, banks, 12, np.random.default_rng(0))
    other = stratified_draw(rows, banks, 12, np.random.default_rng(1))
    assert first.tolist() == again.tolist() and first.tolist() != other.tolist()
    assert len(set(first.tolist())) == 12
    assert {b: int((banks[first] == b).sum()) for b in ("full", "units", "fill")} == {"full": 7, "units": 4, "fill": 1}


def test_the_drawn_prompts_are_seed_stable(run, prompts_meta, tmp_path):
    for name, _, _ in TRANSITIONS:
        assert sum(p["transition"] == name for p in run["meta"]["prompts"]) == PER_TRANSITION
    write_states(tmp_path, prompts_meta)
    phrase_main(tmp_path, per_transition=PER_TRANSITION, seed=0)
    assert json.loads((tmp_path / "inputs.json").read_text())["sample_uids"] == run["meta"]["sample_uids"]
    phrase_main(tmp_path, per_transition=PER_TRANSITION, seed=1)
    assert json.loads((tmp_path / "inputs.json").read_text())["sample_uids"] != run["meta"]["sample_uids"]


def test_a_near_prompt_serves_one_transition_only(run):
    by_transition = {name: {u for u, p in zip(run["meta"]["sample_uids"], run["meta"]["prompts"])
                            if p["transition"] == name} for name, _, _ in TRANSITIONS}
    assert not by_transition["near-both"] & by_transition["near-none"]


def test_opposite_targets_mirror_the_step_and_clamp_at_the_curve_ends():
    log_h = np.array([-3.0, -5.0, 2.0, 2.5])
    target = np.array([-1.0, -1.0, 0.0, 0.5])
    opposite, clamped = opposite_targets(log_h, target, RANGE)
    assert opposite.tolist() == pytest.approx([-5.0, RANGE[0], RANGE[1], RANGE[1]])
    assert clamped.tolist() == [False, True, True, True]


def test_opposite_steps_exist_for_none_and_both_starts_only(run):
    prompts, steps = run["meta"]["prompts"], run["steps"]["step|L16|opposite|number"]
    for prompt, row in zip(prompts, steps):
        if prompt["start_regime"] == "near":
            assert prompt["opposite_log_h"] is None and not prompt["opposite_clamped"]
            assert not row.any()
        else:
            assert RANGE[0] - 1e-6 <= prompt["opposite_log_h"] <= RANGE[1] + 1e-6
            assert row.any() == (not np.isclose(prompt["opposite_log_h"], prompt["log_h"], atol=1e-6))   # a prompt at the curve's end stays put
            mirrored = 2 * prompt["log_h"] - prompt["target_log_h"]
            assert prompt["opposite_clamped"] == (not RANGE[0] <= mirrored <= RANGE[1])
            if row.any():
                assert np.sign(prompt["opposite_log_h"] - prompt["log_h"]) == -np.sign(
                    prompt["target_log_h"] - prompt["log_h"])


def test_effects_steps_follow_the_planted_curve(run):
    prompts = run["meta"]["prompts"]
    log_h = np.array([p["log_h"] for p in prompts])
    target = np.array([p["target_log_h"] for p in prompts])
    # each (layer, role) plants its own axes: the L16 number cell's were saved with the states
    expected = planted_curve(target, run["axes"]) - planted_curve(log_h, run["axes"])
    assert run["steps"]["step|L16|effects|number"] == pytest.approx(expected, abs=0.15)


def test_random_steps_are_as_long_as_the_effects_steps_per_row(run):
    steps = run["steps"]
    for layer in PHRASE_LAYERS:
        for role in ("number", "unit"):
            effects, random = steps[f"step|L{layer}|effects|{role}"], steps[f"step|L{layer}|random|{role}"]
            assert np.linalg.norm(random, axis=1) == pytest.approx(np.linalg.norm(effects, axis=1), rel=1e-4)
    assert not np.allclose(steps["step|L14|random|number"], steps["step|L16|random|number"])
    assert not np.allclose(steps["step|L16|random|number"], steps["step|L16|random|unit"])


def test_invariant_plus_in_span_is_effects(run):
    """The factor span is rebuilt from the same fit by refitting the cell on the saved states."""
    from spar_horizon.steering import fit_duration_effects
    records, directory = run["records"], run["dir"]
    columns = phrase_columns(records)
    X = np.load(directory / "states.npz")["L16|unit"].astype(np.float64)
    cell = fit_duration_effects(X, columns["duration"], columns["factors"], orthogonal=False,
                                covariates={"log near delay": columns["log_near"], "log far delay": columns["log_far"]})
    effects, invariant = (run["steps"][f"step|L16|{kind}|unit"].astype(np.float64) for kind in ("effects", "invariant"))
    in_span = span_part(effects, cell["factor_span"])
    assert in_span.any()
    assert invariant + in_span == pytest.approx(effects, abs=1e-4)
    assert np.abs(span_part(invariant, cell["factor_span"])).max() < 1e-4


def test_only_layer_16_has_invariant_and_opposite_steps(run):
    keys = set(run["steps"].files)
    for layer in PHRASE_LAYERS:
        for role in ("number", "unit"):
            assert {f"step|L{layer}|{kind}|{role}" for kind in ("effects", "random")} <= keys
            assert (f"step|L{layer}|invariant|{role}" in keys) == (f"step|L{layer}|opposite|{role}" in keys) == (layer == 16)


def test_donors_have_the_recipients_phrase_length_and_the_target_regime(run, tok):
    cfg = Settings(**{**GENERATION, "model": MODEL})
    found = 0
    for rec, prompt in zip(drawn_records(run), run["meta"]["prompts"]):
        text = prompt["donor_horizon_text"]
        if text is None:
            assert prompt["donor_log_h"] is None
            continue
        found += 1
        donor_prompt = rec["prompt"].replace(rec["horizon_text"], text)
        assert len(phrase_positions(tok, chat_text(tok, donor_prompt, cfg), text)) == len(
            phrase_positions(tok, chat_text(tok, rec["prompt"], cfg), rec["horizon_text"]))
        assert regime(10 ** prompt["donor_log_h"], rec["near_delay_years"], rec["far_delay_years"]) == prompt["target_regime"]
    assert found >= len(run["meta"]["prompts"]) // 2


def test_inputs_round_trip_through_selected_records(run):
    meta = run["meta"]
    assert [r["sample_uid"] for r in selected_records(meta)] == meta["sample_uids"]
    assert meta["transitions"] == [name for name, _, _ in TRANSITIONS] and meta["layers"] == [14, 16, 18]
    assert meta["generation"]["model"] == MODEL
    assert {c["layer"] for c in meta["cells"]} == set(PHRASE_LAYERS) and len(meta["cells"]) == 6
    assert (run["dir"] / "fit_report.txt").read_text().startswith("E14")


def test_a_short_transition_is_an_error_unless_allowed(tmp_path, prompts_meta):
    write_states(tmp_path, prompts_meta, limit=200)
    with pytest.raises(ValueError, match="--allow-short"):
        phrase_main(tmp_path, per_transition=50)
    phrase_main(tmp_path, per_transition=50, allow_short=True)
    meta = json.loads((tmp_path / "inputs.json").read_text())
    counts = {name: sum(p["transition"] == name for p in meta["prompts"]) for name, _, _ in TRANSITIONS}
    assert min(meta["eligible"].values()) < 50 and counts["both-near"] == meta["eligible"]["both-near"]
    assert counts["none-near"] == 50 and counts["near-both"] == meta["eligible"]["near-both"]
    assert counts["near-none"] < meta["eligible"]["near-none"]   # near-both took some of its prompts
    report = (tmp_path / "fit_report.txt").read_text()
    assert "short" in report and "['unit']" in report and meta["left_out_factors"] == ["unit"]


def test_a_smoke_run_of_two_prompts_per_transition(tmp_path, prompts_meta):
    write_states(tmp_path, prompts_meta, limit=300)
    phrase_main(tmp_path, per_transition=2)
    meta = json.loads((tmp_path / "inputs.json").read_text())
    assert len(meta["prompts"]) == 8 and [r["sample_uid"] for r in selected_records(meta)] == meta["sample_uids"]
