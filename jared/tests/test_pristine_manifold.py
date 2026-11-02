import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from pristine_manifold import median_ci, percent, unit_lookup  # noqa: E402


def test_percent_is_symmetric_in_over_and_under_reading():
    assert np.allclose(percent(np.log10([2.0, 0.5, 1.0])), [100, 100, 0])


def test_unit_lookup_never_uses_the_held_out_duration():
    y = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
    duration = y.copy()
    units = np.array(["days", "days", "days", "weeks", "weeks"])
    pred = unit_lookup(y, duration, units)
    # duration 0 (days): only duration 1's days row is left; duration 2 (weeks): duration 1's weeks row
    assert pred[0] == 1.0 and pred[4] == 1.0
    assert pred[2] == 0.0 and pred[3] == 2.0


def test_median_ci_resamples_whole_groups():
    values = np.array([1.0, 1.0, 5.0, 5.0])
    groups = np.array([0, 0, 1, 1])
    median, low, high = median_ci(values, groups, np.random.default_rng(0))
    assert median == 3.0 and low >= 1.0 and high <= 5.0


def test_systematic_shift_averages_prompt_noise_within_a_rendering():
    from pristine_manifold import systematic_shift
    # one rendering, four bases: shifts +0.3 and -0.3 decades cancel to no systematic shift
    shift = np.array([0.3, -0.3, 0.3, -0.3])
    labels = {"durations": np.array(["2 years"] * 4), "units": np.array(["months"] * 4),
              "pairs": np.array(["p0", "p1", "p2", "p3"])}
    median, _, _ = systematic_shift(shift, np.arange(4), labels, np.random.default_rng(0))
    assert median == 0.0
    # a constant +0.3 decade shift is a 100% systematic error
    median, _, _ = systematic_shift(np.full(4, np.log10(2)), np.arange(4), labels, np.random.default_rng(0))
    assert np.isclose(median, 100.0)


def test_systematic_shift_is_not_inflated_by_reasoning_noise():
    from pristine_manifold import systematic_shift
    rng = np.random.default_rng(0)
    n_render, n_base, noise = 400, 4, 0.3

    def labels_for(n):
        return {"durations": np.repeat([f"d{i}" for i in range(n_render)], n_base), "units": np.array(["months"] * n),
                "pairs": np.array([f"p{i}" for i in range(n)])}

    # noise only: the median |mean| of a rendering over 4 bases is ~0.1 decades (~26%), with no systematic shift
    shift = noise * rng.normal(size=n_render * n_base)
    median, low, _ = systematic_shift(shift, np.arange(shift.size), labels_for(shift.size), rng)
    assert median < 10 and low == 0.0
    # a 0.1-decade shift (26%) of random sign per rendering, under the same noise
    signs = np.repeat(rng.choice([-1, 1], size=n_render), n_base)
    shift = 0.1 * signs + noise * rng.normal(size=n_render * n_base)
    median, low, high = systematic_shift(shift, np.arange(shift.size), labels_for(shift.size), rng)
    assert abs(median - percent(0.1)) < 6 and low <= percent(0.1) <= high


def fake_part(bank, n_rows, kept, anchors, layer_value):
    """A run as load_parts returns it: activations with 2 positions per row at layer 16,
    whose value encodes (bank, row), and labels for the kept rows."""
    from spar_horizon.runs import REGIONS
    regions = {"suffix": (0, 2), "think": (2, 2), "pre": (2, 2), "prelabel": (1, 2), "answer": (2, 2)}
    z = {f"region_{r}": np.array(regions[r]) for r in REGIONS}
    z["tokens"] = np.array(["a", "b"])
    z["L16"] = np.stack([np.full((2, 3), layer_value + i) for i in range(n_rows)])
    labels = {"y": np.arange(len(kept), dtype=float), "anchor": np.array(anchors), "bank": np.full(len(kept), bank),
              "rows": np.array(kept)}
    return z, labels


def test_pooled_rows_line_up_with_their_labels_and_drop_anchors():
    from pristine_manifold import cell_matrix, pooled_labels
    # old run: rows 0 and 2 kept (row 1 had no horizon); fill run: row 1 is an anchor
    parts = [fake_part("full", 3, [0, 2], [False, False], 100), fake_part("fill", 3, [0, 1, 2], [False, True, False], 200)]
    labels = pooled_labels(parts)
    X = cell_matrix(parts, "pre-label", 16)
    assert list(labels["bank"]) == ["full", "full", "fill", "fill"] and "rows" not in labels
    assert list(X[:, 0]) == [100, 102, 200, 202]
    assert list(labels["y"]) == [0, 1, 0, 2]


def test_gated_jobs_pool_a_mode_whose_anchors_match_before_the_trace():
    from pristine_manifold import gated_jobs
    jobs = [("off", "pre-label", 16), ("on", "suffix newline", 16), ("on", "pre-label", 16)]
    # thinking on: the suffix matches, the cell after the trace does not (traces diverged)
    anchors = {"off": {"cells": {"pre-label L16": {"median": 0.0}}},
               "on": {"cells": {"suffix newline L16": {"median": 0.05}, "pre-label L16": {"median": 0.7}}}}
    assert gated_jobs(jobs, anchors) == jobs
    # a suffix mismatch means a different run setup: none of that mode's cells pool
    anchors["on"]["cells"]["suffix newline L16"]["median"] = 0.5
    assert gated_jobs(jobs, anchors) == jobs[:1]
    assert gated_jobs(jobs, None) == jobs
    assert gated_jobs(jobs, {"off": None, "on": None}) == jobs
