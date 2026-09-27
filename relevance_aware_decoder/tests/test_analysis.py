import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from rad.analysis import RidgePath, behavior_pull, pull_stats, regex_horizon


def test_ridge_path_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 200)); y = X[:, 0] - 2 * X[:, 3] + rng.normal(size=80)
    rp = RidgePath(X, y, device="numpy")
    for al in (0.1, 10.0, 1000.0):
        ref = Ridge(alpha=al).fit(X, y)
        np.testing.assert_allclose(rp.predict(X, al), ref.predict(X), rtol=1e-6, atol=1e-8)


def test_paired_pull_recovers_slope_and_unpaired_metric_is_confounded():
    rng = np.random.default_rng(1)
    n = 600
    log_h = rng.uniform(-2.5, 2, n); log_d = rng.uniform(-2.5, 2, n)
    clusters = np.repeat(np.arange(12), n // 12)
    shrink = 0.7 * log_h                                   # an ordinary decoder that regresses to the mean
    twin = shrink + rng.normal(0, 0.1, n)
    for true_pull in (0.0, 0.4):
        pred = twin + true_pull * (log_d - log_h) + rng.normal(0, 0.05, n)
        st = pull_stats(pred, twin, log_h, log_d, clusters, n_boot=200)
        assert abs(st["pull"] - true_pull) < 0.03 and st["pull_lo"] < true_pull < st["pull_hi"]
    # with zero true pull, Alan's unpaired metric is still clearly positive: gap and -H are correlated
    st0 = pull_stats(twin + rng.normal(0, 0.05, n), twin, log_h, log_d, clusters, n_boot=0)
    assert st0["rho_resid_unpaired"] > 0.3 and abs(st0["pull"]) < 0.05


def test_behavior_pull_scale():
    rng = np.random.default_rng(2)
    n = 240
    log_h = rng.uniform(-2.5, 2, n); log_d = rng.uniform(-2.5, 2, n); beta = -2.0
    base = pd.DataFrame(dict(short_first=rng.random(n) < 0.5, choice="a", scenario_id=np.repeat([f"s{i}" for i in range(12)], n // 12)))

    def frame(lo):                       # encode short log-odds into label logits respecting label order
        f = base.copy(); f["logit_a"] = np.where(f.short_first, lo, 0.0); f["logit_b"] = np.where(f.short_first, 0.0, lo)
        f["chose_short"] = lo > 0; return f

    twins = frame(beta * log_h)
    dist = frame(beta * (log_h + 0.25 * (log_d - log_h))); dist["log_gap"] = log_d - log_h
    b = behavior_pull(dist, twins, beta_h=beta, n_boot=100)
    assert abs(b["behavior_pull"] - 0.25) < 1e-6


def test_regex_horizon_is_fooled_by_other_horizon_sentence():
    t = ("You are X. For an unrelated decision last year, the household used a time horizon of 10 years. "
         "Your time horizon for this decision is 1 day: choose the option.")
    assert np.isclose(regex_horizon(t), 1.0)            # reads the distractor, by design of that family
