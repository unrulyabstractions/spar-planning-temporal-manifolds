"""Steering with the effects curve (16_spec_pristine_manifold.md §5) on planted data:
the invariant / in-span split, U, the stage A and B prompt selections, pooled inputs
in E8, the additive prompt patch, and stage B's summary. No model download."""

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pipeline.s2_steering.experiments.ablate_steer.exp8_steering import (PRISTINE_DIRECTIONS, delta_rows, selected_records,
                                                                          toward_by_direction)
from pipeline.s2_steering.experiments.manifold_steer.exp13_suffix_steer import chose_near, names_duration, summarize
from pipeline.s1_data_vectors.analysis.steering_prepare import (
    concatenate_reasoning, pristine_cell, split_summary, stage_a_rows, stage_b_rows, steerable)
from spar_horizon import steering as S
from spar_horizon.banks import build_records
from spar_horizon.config import Settings

THETA = np.pi / 3   # angle between the planted duration axis and the unit direction


def planted_bank(unit_sign=1.0, noise=0.001, seed=0):
    """Durations crossed with overlapping units, as in the units bank. The duration
    axis lies at THETA to the unit direction u (part of it is "number line", 17 §6);
    regime, order and a delay covariate have their own directions. d is the 8B's width:
    the span of the estimated factor effects also holds their noise directions (e.g. a
    unit level's estimate minus twice the first's), each at full weight whatever the
    noise level, and they take about m / d of any other direction (0.002 here, 0.13 at d = 40).
    The noise per coordinate is small so that a row's noise norm stays ~0.06."""
    rng = np.random.default_rng(seed)
    d = 4096
    q = np.linalg.qr(rng.normal(size=(d, 5)))[0].T
    u, v = q[0], q[1]
    axis = np.cos(THETA) * u + np.sin(THETA) * v
    rows = []
    for i, h in enumerate(np.linspace(-2, 2, 15)):
        for unit in (i // 3, i // 3 + 1):             # each unit spans ~6 durations, overlapping its neighbours
            for _ in range(4):
                rows.append((h, unit, rng.integers(3), rng.random() < 0.5, rng.uniform(-2, 2)))
    log_h, unit, regime, order, delay = (np.array(c) for c in zip(*rows))
    X = (np.outer(log_h, axis) + unit_sign * np.outer(unit, u) + np.outer(regime, q[2]) + np.outer(order, q[3])
         + np.outer(delay, q[4]) + noise * rng.normal(size=(len(rows), d)))
    factors = {"unit": unit, "regime": regime, "order": order}
    return X, log_h, factors, {"log near delay": delay}, u, v


@pytest.fixture(scope="module")
def cell():
    X, log_h, factors, covariates, _, _ = planted_bank()
    return pristine_cell(X, log_h, log_h, factors, covariates)[0]


def test_factor_span_is_orthonormal_and_holds_every_factor_and_covariate_effect():
    X, log_h, factors, covariates, _, _ = planted_bank()
    fit = S.fit_duration_effects(X, log_h, factors, orthogonal=False, covariates=covariates)
    span = fit["factor_span"]
    assert span.shape == (5 + 2 + 1 + 1, X.shape[1])     # unit 6 levels, regime 3, order 2, one covariate
    assert np.allclose(span @ span.T, np.eye(len(span)), atol=1e-10)
    effects = np.vstack([e for _, e in fit["factors"].values()] + list(fit["covariates"].values()))
    assert np.allclose(S.span_part(effects, span), effects, atol=1e-10)


def test_invariant_and_in_span_parts_add_up_to_the_effects_step(cell):
    log_h = np.linspace(-1.5, 1.5, 7)
    for alpha in (-2, -1, 1, 2):
        step = delta_rows(cell, "effects curve", alpha, log_h)
        invariant = delta_rows(cell, "invariant part", alpha, log_h)
        in_span = delta_rows(cell, "in-span part", alpha, log_h)
        assert np.allclose(invariant + in_span, step, atol=1e-12)
        # the in-span part lies in the factor span, the invariant part is orthogonal to it
        assert np.allclose(S.span_part(in_span, cell["factor_span"]), in_span, atol=1e-10)
        assert np.abs(invariant @ cell["factor_span"].T).max() < 1e-10


def test_unit_and_random_directions_have_the_effects_steps_length(cell):
    log_h = np.linspace(-1.5, 1.5, 7)
    for alpha in (-1, 2):
        length = np.linalg.norm(delta_rows(cell, "effects curve", alpha, log_h), axis=1)
        for direction in ("unit U", "random 0", "random 1"):
            rows = delta_rows(cell, direction, alpha, log_h)
            assert np.allclose(np.linalg.norm(rows, axis=1), length)
        unit_rows = delta_rows(cell, "unit U", alpha, log_h)
        assert np.allclose(unit_rows / length[:, None], np.sign(alpha) * cell["unit_direction"])
    assert set(PRISTINE_DIRECTIONS) == {"effects curve", "invariant part", "in-span part", "unit U",
                                        "random 0", "random 1"}


def test_planted_split_is_recovered():
    # the duration axis is cos(THETA) u + sin(THETA) v: the in-span share of a step is
    # cos²(THETA), and the invariant part points along v
    X, log_h, factors, covariates, u, v = planted_bank()
    cell, old = pristine_cell(X, log_h, log_h, factors, covariates)
    steps = S.curve_delta(cell, np.array([-1.0, 0.0]), 1)
    in_span = S.span_part(steps, cell["factor_span"])
    share = (in_span ** 2).sum(1) / (steps ** 2).sum(1)
    assert np.allclose(share, np.cos(THETA) ** 2, atol=0.02)
    invariant = steps - in_span
    assert np.all(invariant @ v / np.linalg.norm(invariant, axis=1) > 0.99)
    summary = split_summary(cell, old, np.array([-1.0, 0.0, 0.5]))
    assert abs(summary["in-span share"] - np.cos(THETA) ** 2) < 0.02
    assert np.isclose(summary["invariant length"] ** 2 + summary["in-span length"] ** 2,
                      summary["effects length"] ** 2, rtol=0.02)


@pytest.mark.parametrize("unit_sign", [1.0, -1.0])
def test_unit_direction_is_the_planted_unit_axis_oriented_along_the_curve(unit_sign):
    # whichever way the unit levels run, U points the way the curve goes from short to long
    X, log_h, factors, covariates, u, _ = planted_bank(unit_sign)
    fit = S.fit_duration_effects(X, log_h, factors, orthogonal=False, covariates=covariates)
    assert S.unit_direction(fit) @ u > 0.99


def test_stage_a_steers_natural_near_and_both_renderings_without_anchors():
    records = [{"horizon": 1.0, "regime": "near"},
               {"horizon": 1.0, "regime": "both", "natural": True, "anchor": False},
               {"horizon": 1.0, "regime": "none"},
               {"horizon": 1.0, "regime": "near", "natural": False},
               {"horizon": 1.0, "regime": "both", "natural": True, "anchor": True},
               {"horizon": None, "regime": "none"}]
    assert stage_a_rows(records) == [0, 1]


def test_only_delays_within_alans_range_are_steered():
    # fill prompts carry the alan_range label; old-bank prompts are in range by construction
    assert steerable({"horizon": 1.0, "natural": True, "anchor": False, "alan_range": True})
    assert not steerable({"horizon": 1.0, "natural": True, "anchor": False, "alan_range": False})
    assert steerable({"horizon": 1.0})
    assert stage_a_rows([{"horizon": 1.0, "regime": "near", "natural": True, "alan_range": False}]) == []


def test_stage_b_takes_two_per_duration_in_distinct_regimes_when_it_can():
    durations = np.array(["1 day"] * 5 + ["3 years"] * 3 + ["9 hours"])
    regimes = np.array(["none", "none", "none", "near", "none", "both", "both", "both", "none"])
    rows = stage_b_rows(durations, regimes)
    assert len(rows) == 5
    by_duration = {d: sorted(regimes[rows[durations[rows] == d]]) for d in np.unique(durations)}
    assert by_duration == {"1 day": ["near", "none"], "3 years": ["both", "both"], "9 hours": ["none"]}
    assert np.array_equal(rows, stage_b_rows(durations, regimes))   # deterministic


def test_reasoning_blocks_concatenate_with_records_shifted_into_the_pooled_selection():
    def block(seqs, records):
        return {"ids": np.concatenate(seqs).astype(np.int32), "offsets": np.r_[0, np.cumsum([len(s) for s in seqs])],
                "close": np.array([len(s) - 2 for s in seqs]), "think_len": np.array([1] * len(seqs)),
                "forced": np.array([False] * len(seqs)), "labels": np.array([[1, 2]] * len(seqs)),
                "record": np.array(records)}

    post = concatenate_reasoning([(block([[1, 2, 3], [4, 5]], [0, 2]), 0), (None, 3),
                                  (block([[6, 7, 8, 9]], [1]), 5)])
    assert post["record"].tolist() == [0, 2, 6]
    seqs = [post["ids"][a:b].tolist() for a, b in zip(post["offsets"][:-1], post["offsets"][1:])]
    assert seqs == [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    assert concatenate_reasoning([(None, 0)]) is None


def test_pooled_inputs_rebuild_each_banks_records_in_order():
    settings = {"full": {"n_prompts": 500, "seed": 0, "limit": 6}, "units": {"n_prompts": 500, "seed": 0, "limit": 8}}
    full, units = (build_records(Settings(bank=b, **settings[b])) for b in ("full", "units"))
    meta = {"selected": [3, 0, 5], "selected_banks": ["units", "full", "full"], "bank_settings": settings,
            "sample_uids": [units[3]["sample_uid"], full[0]["sample_uid"], full[5]["sample_uid"]]}
    assert [r["sample_uid"] for r in selected_records(meta)] == meta["sample_uids"]
    meta["sample_uids"][0] = full[3]["sample_uid"]
    with pytest.raises(ValueError, match="differs"):
        selected_records(meta)


def test_thinking_on_is_judged_on_rendering_averages():
    # five prompts of one rendering pull +1, one prompt of another pulls -1: per prompt
    # the mean is +0.67, on rendering averages 0
    toward = [1.0] * 5 + [-1.0]
    steer = {"effects curve": {a: {"toward": toward if a > 0 else [np.nan] * 6} for a in (-2, -1, 1, 2)}}
    renderings = ["3 years"] * 5 + ["9 hours"]
    assert np.isclose(toward_by_direction(steer)["effects curve"][1][0], 4 / 6)
    assert np.isclose(toward_by_direction(steer, renderings)["effects curve"][1][0], 0.0)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=64, head_dim=8)
    return Qwen3ForCausalLM(cfg).eval()


def test_additive_prompt_patch_equals_adding_the_step_in_the_pass(tiny):
    seqs, layer, positions = [[3, 5, 7, 9, 11, 13], [4, 6, 8, 10]], 2, [[2], [1]]
    steps = [torch.randn(1, 32), torch.randn(1, 32)]
    with S.prompt_patch(tiny, layer, positions, steps, add=True):
        patched, _, _ = S.forward(tiny, "cpu", seqs, 0)
    added, _, _ = S.forward(tiny, "cpu", seqs, 0, layer=layer, positions=positions,
                            edit=S.Add(torch.cat(steps)))
    base, _, _ = S.forward(tiny, "cpu", seqs, 0)
    assert torch.allclose(patched, added, atol=1e-5)
    assert not torch.allclose(patched, base, atol=1e-3)


def test_steered_to_horizon_mentions_ignore_the_delays():
    think = "The horizon is 30 years. The second payout comes in 3 years."
    assert names_duration(think, 30.0, delays=(1.0, 3.0))
    assert not names_duration(think, 3.0, delays=(1.0, 3.0))       # 3 years is the far delay
    assert not names_duration(think, 300.0, delays=(1.0, 3.0))
    assert chose_near("a", "a") == 1.0 and chose_near("b", "a") == 0.0 and np.isnan(chose_near(None, "a"))


def test_stage_b_summary_signs_shifts_toward_the_rational_answer():
    def row(regime, near_lo, label, e0_near_lo, e0_label):
        return {"regime": regime, "near_label": "a", "near_lo": near_lo, "label": label, "e0_near_lo": e0_near_lo,
                "e0_label": e0_label, "names_own": True, "e0_names_own": True, "names_target": False,
                "e0_names_target": False, "forced": False, "e0_forced": False}

    # longer horizon (alpha +1): near-regime prompts should move to far (lower near log-odds)
    rows = [row("near", -1.0, "b", 2.0, "a"),        # flipped toward far: +3 toward
            row("near", 3.0, "a", 2.0, "a"),         # moved away by 1, label unchanged
            row("both", 0.0, "a", 1.0, "b"),         # not judged at alpha +1
            row("none", 0.5, "a", 0.0, "a")]
    s = summarize({"invariant part +1": {"direction": "invariant part", "alpha": 1, "rows": rows}})["invariant part +1"]
    assert s["n_crossing"] == 2
    assert np.isclose(s["toward_log_odds"][0], 1.0)
    assert s["flipped_toward"] == 0.5 and s["flipped_away"] == 0.0
    assert np.isclose(s["near_shift_by_regime"]["none"][0], 0.5)
