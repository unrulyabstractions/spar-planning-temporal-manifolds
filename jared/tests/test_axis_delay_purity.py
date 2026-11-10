import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import axis_delay_purity as adp  # noqa: E402
from confound_sweep import FEATURES  # noqa: E402


def synthetic(rng, n_durations=30, per_duration=20, d=40):
    """Activations = horizon·h + near·u + far·v + noise, with h, u, v orthonormal, and
    delays drawn independently of the horizon."""
    h, u, v, *rest = np.linalg.qr(rng.normal(size=(d, 6)))[0].T
    logh = np.repeat(np.linspace(-6, 2, n_durations), per_duration)
    near, far = rng.uniform(-3, 1, len(logh)), rng.uniform(-1, 2, len(logh))
    X = np.outer(logh, h) + np.outer(near, u) + np.outer(far, v) + 0.05 * rng.normal(size=(len(logh), d))
    features = np.zeros((len(logh), len(FEATURES)))
    features[:, 0], features[:, 1] = near, far
    durations = np.array([f"full|{x:.3f}" for x in logh])
    data = {"y": logh, "features": features, "durations": durations, "pairs": np.arange(len(logh)).astype(str)}
    return X, data, (h, u, v, np.array(rest))


def curve_along(direction, others):
    """A k = 4 curve whose points move along `direction` (one per decade)."""
    basis = np.linalg.qr(np.column_stack([direction, *others[:3]]))[0].T
    logh = np.linspace(-6, 2, 9)
    return {"mean": np.zeros(len(direction)), "basis": basis, "curve_logh": logh,
            "curve_coords": np.outer(logh, direction) @ basis.T}


def test_a_contaminated_axis_is_caught_and_clean_axes_read_zero():
    rng = np.random.default_rng(0)
    X, data, (h, u, v, rest) = synthetic(rng)
    # (a) leans toward the near delay: a near decade moves 0.5 / 1.25 = 0.4 decades along it
    cells = {name: curve_along(h, rest) for name in adp.CANDIDATES}
    cells["(a) old effects"] = curve_along(h + 0.5 * u, [rest[0], rest[1], v])
    folds = np.arange(len(data["y"])) % 2
    rows = np.ones(len(folds), bool)
    out, differences = adp.purity(X, data, folds, [cells, cells], rows, rng)
    near = out["(a) old effects"]["log near delay"]
    assert abs(near[0] - 0.4) < 0.02 and near[1] > 0.35
    # the steering view: a one-decade step along h + 0.5u holds half a decade of near delay
    assert abs(out["(a) old effects"]["log near delay per horizon step"] - 0.5) < 0.03
    assert abs(out["(b) pooled effects"]["log near delay per horizon step"]) < 0.02
    for name in list(adp.CANDIDATES)[1:]:
        for delay in adp.DELAYS:
            assert abs(out[name][delay][0]) < 0.02, (name, delay)
            assert out[name][f"{delay} in span"] < 0.05, (name, delay)
    # (a) has v in its basis but not along its direction: in span (1 / |h + 0.5u| = 0.89), not along
    assert out["(a) old effects"]["log far delay in span"] > 0.85 and abs(out["(a) old effects"]["log far delay"][0]) < 0.02
    a_minus_b = differences["(a) old effects - (b) pooled effects"]
    assert a_minus_b[1] > 0.3


def test_pull_separates_the_horizon_from_the_delays():
    rng = np.random.default_rng(1)
    _, data, _ = synthetic(rng)
    readout = 0.9 * data["y"] - 0.3 * data["features"][:, 0] + 0.01 * rng.normal(size=len(data["y"]))
    fits = adp.pull(readout, data, np.ones(len(readout), bool), rng)
    assert abs(fits["across"]["log horizon"][0] - 0.9) < 0.01
    assert abs(fits["across"]["log near delay"][0] + 0.3) < 0.01
    assert abs(fits["within"]["log near delay"][0] + 0.3) < 0.01
    assert abs(fits["within"]["log far delay"][0]) < 0.01
