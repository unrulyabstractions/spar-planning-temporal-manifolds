"""trend and probe (steer/horizon_decoders.py): planted-structure checks, then the fit.py/bundle.py/refit plumbing."""

import json

import numpy as np

import bundle
import fit
from fixtures import BANK, make_activations
from steer.horizon_decoders import horizon_decoders, horizon_folds
from steer.methods import load

D = 30


def planted(sign=1.0, seed=0):
    """Two regimes with very different horizons and a huge regime offset along `offset`; inside each regime the
    activation moves along `pattern` with log-horizon. A pooled long-minus-short mean difference sees only the offset."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((D, 2)))
    pattern, offset = basis[:, 0], basis[:, 1]
    grid = np.linspace(1.0, 6.0, 20)
    t = np.concatenate([np.tile(grid[:12], 6), np.tile(grid[8:], 6)])
    regime = np.array(["near_only"] * 72 + ["both"] * 72, dtype=object)
    X = (sign * t[:, None] * pattern + 50.0 * (regime == "both")[:, None] * offset
         + 0.1 * rng.standard_normal((len(t), D)))
    return X, t, regime, pattern, offset


def test_trend_and_probe_recover_the_within_regime_pattern_despite_a_regime_offset():
    X, t, regime, pattern, offset = planted()
    trend, probe, diag = horizon_decoders(X, t, regime)
    pooled_mean_difference = X[t >= 4].mean(axis=0) - X[t <= 3].mean(axis=0)
    caa_like = pooled_mean_difference / np.linalg.norm(pooled_mean_difference)
    print(f"cos with planted pattern: trend {trend.vector @ pattern:.3f}, probe {probe.vector @ pattern:.3f}, "
          f"CAA-like {caa_like @ pattern:.3f} (with the offset {caa_like @ offset:.3f}); {diag}")
    assert trend.vector @ pattern > 0.99 and probe.vector @ pattern > 0.95
    assert abs(caa_like @ pattern) < 0.2 < caa_like @ offset
    assert diag["trend"]["held_out_horizon_spearman"] > 0.9 and diag["probe"]["held_out_horizon_spearman"] > 0.9
    assert np.isclose(np.linalg.norm(trend.vector), 1) and np.isclose(np.linalg.norm(probe.vector), 1)


def test_both_point_toward_longer_horizons():
    for sign in (1.0, -1.0):
        X, t, regime, pattern, _ = planted(sign)
        trend, probe, _ = horizon_decoders(X, t, regime)
        assert np.sign(trend.vector @ pattern) == np.sign(probe.vector @ pattern) == sign


def test_horizon_folds_hold_out_whole_horizons_interleaved_by_sorted_horizon():
    t = np.array([3.0, 1.0, 2.0, 2.0, 5.0, 4.0, 1.0, 6.0])
    folds = horizon_folds(t, 3)
    assert [sorted(set(t[f])) for f in folds] == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert sum(f.sum() for f in folds) == len(t)


def fit_dir(tmp_path, *extra):
    acts = make_activations(tmp_path / "acts", splits=("extraction", "verification"))
    fit.main(["--activations", str(acts), "--bank", str(BANK), "--layer", "15", *extra], fits_root=tmp_path / "fits")
    return acts, tmp_path / "fits" / "acts" / "default" / "L15_label"


def test_fit_stores_both_directions_and_their_diagnostics(tmp_path):
    _, out = fit_dir(tmp_path)
    z = np.load(out / "directions.npz")
    hd = json.loads((out / "fit.json").read_text())["diagnostics"]["horizon_decoders"]
    print(hd)
    for key in ("trend", "probe"):
        assert np.isclose(np.linalg.norm(z[key]), 1) and f"{key}_raw_norm" in z.files
        assert {"held_out_horizon_spearman", "cos_caa", "cos_choice", "cos_reward"} <= hd[key].keys()
    assert hd["classes"]["n_prompts"] == 471 and hd["trend"]["held_out_horizon_spearman"] > 0.5


def test_bundle_makes_trend_and_probe_and_refit_reproduces_them(tmp_path):
    acts, out = fit_dir(tmp_path)
    outs = bundle.main(["--fit", str(out), "--activations", str(acts)], methods_root=tmp_path / "methods")
    by_method = {o.name.split("__")[0]: o for o in outs}
    assert {"trend", "probe"} <= by_method.keys()
    for name in ("trend", "probe"):
        m = load(by_method[name])
        assert np.array_equal(m.structure["direction"], np.load(out / "directions.npz")[name])
        again = m.refit(acts, BANK)
        assert np.array_equal(again.structure["direction"], m.structure["direction"])
