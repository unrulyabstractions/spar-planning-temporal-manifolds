"""Pure helpers of E6 (think-block PCA)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from exp6_think_pca import (N_COMP, bin_means, components_for, interleave_by_horizon,  # noqa: E402
                            probe_position, project, resolve_layers, rho_by_bin)
from spar_horizon.prompts import build_prompts  # noqa: E402


def test_interleave_is_a_permutation_spanning_horizons_early():
    records = build_prompts()
    order = interleave_by_horizon(records)
    assert sorted(order) == list(range(len(records)))
    n_h = len({r["horizon"] for r in records if r["horizon"] is not None})
    first = {records[i]["horizon"] for i in order[: n_h + 1]}
    assert len(first) == n_h + 1          # every horizon plus the no-horizon control
    assert records[order[-1]]["horizon"] is None


def test_components_alternate_three_six():
    assert [components_for(r) for r in range(5)] == [3, 6, 3, 6, 3]
    assert N_COMP == 6


def test_resolve_layers_fractions_and_ints():
    assert resolve_layers("0.25,0.5,0.75", 33) == [8, 16, 24]
    assert resolve_layers("8, 16,8", 33) == [8, 16]
    assert resolve_layers("0,40", 33) == [1, 32]        # clamped into [1, n-1]


def test_probe_position():
    assert probe_position(9) == 1
    assert probe_position(1) == 0


def test_bin_means_and_rho():
    v = np.arange(20, dtype=float)
    b = bin_means(v, 4)
    assert np.allclose(b, [2, 7, 12, 17])
    assert np.isnan(bin_means(v, 40)).any()
    log_h = np.linspace(-6, 2, 12)
    binned = np.outer(log_h, np.ones(4)) + 0.01 * np.random.default_rng(0).normal(size=(12, 4))
    assert np.allclose(rho_by_bin(binned, log_h), 1.0)


def test_project_shapes_and_values():
    rng = np.random.default_rng(0)
    d, T = 16, 7
    comp = rng.normal(size=(N_COMP, d)).astype(np.float32)
    base = {"think_comp": comp, "think_mean": np.zeros(d, np.float32),
            "suffix_comp": comp, "suffix_mean": np.ones(d, np.float32),
            "probe_coef": np.ones(d, np.float32), "probe_intercept": 2.0}
    h = rng.normal(size=(T, d)).astype(np.float32)
    zt, zs, pr = project(h, base, 3)
    assert zt.shape == (T, 3) and zs.shape == (T, 3) and pr.shape == (T,)
    assert np.allclose(zt, h @ comp[:3].T, atol=1e-5)
    assert np.allclose(zs, (h - 1) @ comp[:3].T, atol=1e-5)
    assert np.allclose(pr, h.sum(1) + 2, atol=1e-4)
    assert project(h, base, 6)[0].shape == (T, 6)


def test_generate_within_budget_stops_with_a_prefix(monkeypatch):
    import exp6_think_pca as e6

    class Cfg:
        batch_size = 2

    calls = []
    monkeypatch.setattr(e6, "generate_batch", lambda *a: calls.append(a[3]) or list(a[3]))
    budget = e6.Budget(0)                 # already exhausted
    out = e6.generate_within_budget(None, None, None, list("abcdef"), Cfg(), budget, 1.0)
    assert out == ["a", "b"]              # first batch always runs, then it stops
    assert len(calls) == 1
    out = e6.generate_within_budget(None, None, None, list("abcdef"), Cfg(), e6.Budget(10), 1.0)
    assert out == list("abcdef")
