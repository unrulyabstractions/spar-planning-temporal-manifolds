import numpy as np

import pipeline.s3_analysis.by_label.delay_step as ds


def fill_like(rng, durations=40, per_duration=12):
    """Margins drawn as the fill bank draws them (full_bank.delays_for): every delay at
    least 0.2 decades from the horizon, regimes rotating within each duration."""
    rows = []
    for d in range(durations):
        log_h = -6 + 9 * d / durations
        for b in range(per_duration):
            margin, other, gap = rng.uniform(0.2, 1.5), rng.uniform(0.2, 1.5), rng.uniform(0.3, 2.0)
            near, far = {0: (margin, margin + gap), 1: (-margin, other), 2: (-margin - gap, -margin)}[b % 3]
            rows.append((d, f"{d}-{b}", log_h, near, far))
    groups, bases, log_h, near, far = (np.array(c) for c in zip(*rows))
    return groups, bases, {"near": near, "far": far}, {"near": near + log_h, "far": far + log_h}


def fit(readout, groups, bases, margins, log_delays):
    return ds.step_fits(readout, 1.0, margins, log_delays, None, groups, bases, np.random.default_rng(1))


def test_a_planted_step_and_a_planted_slope_are_recovered_separately():
    rng = np.random.default_rng(0)
    groups, bases, margins, log_delays = fill_like(rng)
    readout = (2.0 * groups + 0.5 * (margins["far"] < 0) + 0.3 * margins["far"] - 0.1 * margins["near"]
               + 0.05 * rng.normal(size=len(groups)))
    hinge = fit(readout, groups, bases, margins, log_delays)["hinge"]
    assert abs(hinge["far step"][0] - 0.5) < 0.05 and hinge["far step"][1] > 0.4
    assert abs(hinge["far slope"][0] - 0.3) < 0.05
    assert abs(hinge["near step"][0]) < 0.05 and hinge["near step"][1] < 0 < hinge["near step"][2]
    assert abs(hinge["near slope"][0] + 0.1) < 0.05


def test_a_pure_slope_reads_as_no_step_once_the_smooth_terms_are_in():
    rng = np.random.default_rng(2)
    groups, bases, margins, log_delays = fill_like(rng)
    readout = 1.5 * groups + 0.4 * margins["far"] + 0.05 * rng.normal(size=len(groups))
    fits = fit(readout, groups, bases, margins, log_delays)
    for spec in ("hinge", "spline"):
        step = fits[spec]["far step"]
        assert abs(step[0]) < 0.06 and step[1] < 0 < step[2], spec
    # without smooth terms the slope masquerades as a step (19 §4's regime coefficient)
    assert fits["step only"]["far step"][0] < -0.3


def test_spline_terms_are_linear_beyond_the_outer_knots():
    knots = [0.0, 1.0, 2.0, 3.0]
    x = np.array([3.0, 4.0, 5.0, 6.0, -3.0, -2.0, -1.0])
    f = ds.spline_terms(x, knots)
    assert f.shape == (7, 2)
    assert np.allclose(np.diff(f[:4], 2, axis=0), 0)   # straight above the last knot
    assert np.allclose(f[4:], 0)                        # zero below the first


def test_curve_direction_is_the_slope_of_the_curve_points_on_log_horizon():
    rng = np.random.default_rng(3)
    direction = rng.normal(size=20)
    basis = np.linalg.qr(np.column_stack([direction, rng.normal(size=(20, 2))]))[0].T   # [3, 20] orthonormal
    logh = np.array([-2.0, -1.0, 0.5, 1.0, 3.0])
    points = np.outer(logh, direction)
    cell = {"curve_logh": logh, "curve_coords": points @ basis.T, "basis": basis, "mean": np.zeros(20)}
    assert np.allclose(ds.curve_direction(cell), direction)
    X = np.outer([0.0, 2.0], direction)
    assert np.allclose(ds.project(cell, X), [0.0, 2.0])
