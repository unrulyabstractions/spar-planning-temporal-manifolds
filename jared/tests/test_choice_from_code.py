import numpy as np

import pipeline.s3_analysis.by_label.choice_from_code as cc
from synthetic_pooled import synthetic_pooled


def test_alpha_coordinate_moves_by_alpha_along_the_direction():
    rng = np.random.default_rng(0)
    centered, direction = rng.normal(size=(5, 7)), rng.normal(size=(5, 7))
    direction[2] = 0
    before = cc.alpha_coordinate(centered, direction)
    after = cc.alpha_coordinate(centered + 1.5 * direction, direction)
    assert np.allclose(np.delete(after - before, 2), 1.5)
    assert np.isnan(before[2])


def test_choice_effect_recovers_a_planted_slope_and_ignores_noise():
    X, data, _ = synthetic_pooled(seed=1)
    frame = cc.choice_frame(data)
    rng = np.random.default_rng(0)
    signal = rng.normal(size=len(X))
    data_rows = np.ones(len(X), bool)
    frame["chose_near"] = (rng.random(len(X)) < np.clip(0.5 + 0.2 * signal, 0, 1)).astype(float)
    planted = cc.choice_effect(frame, {"signal": signal}, data_rows, rng)["signal"]["effect"]
    assert planted[1] < 0.2 < planted[2] and planted[1] > 0.1
    noise = cc.choice_effect(frame, {"noise": rng.normal(size=len(X))}, data_rows, rng)["noise"]["effect"]
    assert noise[1] < 0 < noise[2]


def test_a_choice_planted_on_the_invariant_direction_is_found_there_and_not_by_random_directions():
    X, data, _ = synthetic_pooled(latent_to_choice=0.25)
    frame = cc.choice_frame(data)
    rng = np.random.default_rng(0)
    coords = cc.direction_coordinates(X, data)
    q2 = cc.direction_results(frame, coords, "all", rng)
    invariant = q2["alone"]["all regimes"]["invariant"]["invariant"]
    assert invariant["effect"][2] < 0   # longer-reading code, fewer near choices
    assert abs(cc.per_sd(invariant)[0]) > np.percentile(np.abs(q2["random per SD"]), 95)
    # the readout residual of Q1 carries the same planted dependence
    duration, factors, covariates = cc.fit_inputs(data)
    readout = cc.cross_fit("effects ⟂", X, duration, factors, covariates, cc.base_folds(data["pairs"]), 4)[0]
    residual = cc.choice_effect(frame, {"residual": readout - data["y"]}, np.ones(len(X), bool), rng)
    assert residual["residual"]["effect"][2] < 0
