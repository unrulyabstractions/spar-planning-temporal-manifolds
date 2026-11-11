"""A small synthetic stand-in for confound_sweep.dataset's pooled data, shared by the
choice_from_code and direction_stability tests (not a test module itself)."""

import numpy as np

from spar_horizon.horizons import UNIT_YEARS, Horizon

DURATIONS = ("1 day", "3 days", "1 week", "2 weeks", "1 month", "3 months", "6 months", "1 year", "3 years",
             "10 years")
UNITS = ("days", "weeks", "months", "years")


def synthetic_pooled(n_bases=240, d=300, seed=0, latent_to_choice=0.0, unit_share_of_horizon=0.0,
                     latent_shift=0.3):
    """Activations X and a confound_sweep-shaped `data` dict. Each base prompt has two
    renderings in different units; the bank is "fill" for every third base.

    X = log h × horizon + unit effect + regime effect + order effect + delay slopes
        + latent × horizon + noise
    with the horizon direction orthogonal to every planted factor effect, except that
    `unit_share_of_horizon` of its squared length lies along the unit effect. The latent
    (one per base, standard normal) shifts the code by `latent_shift` along the horizon
    direction within a duration, and with `latent_to_choice` > 0 it lowers P(near) by
    that much per SD: a longer-reading code chooses far more often.

    Returns X, data, and the planted unit-length directions {"horizon", "unit"}."""
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.normal(size=(d, 8)))[0].T
    unit = q[1]
    horizon = np.sqrt(1 - unit_share_of_horizon) * q[0] + np.sqrt(unit_share_of_horizon) * unit
    unit_effect = {u: 0.4 * i * unit for i, u in enumerate(UNITS)}
    regime_effect = {"none": 0 * q[2], "near": 0.3 * q[2], "both": 0.5 * q[3]}
    rows = []
    for base in range(n_bases):
        duration = DURATIONS[rng.integers(len(DURATIONS))]
        bank = "fill" if base % 3 == 0 else "units"
        regime = ("none", "near", "both")[rng.integers(3)]
        latent = rng.normal()
        near_first = rng.random() < 0.5
        log_near, log_far = rng.uniform(-2, 1), rng.uniform(-1, 1.5)
        p_near = np.clip(0.5 - latent_to_choice * latent, 0.02, 0.98)
        for rendered in rng.choice(UNITS, size=2, replace=False):
            log_h = Horizon.parse(duration).log10_years
            x = (log_h * horizon + unit_effect[rendered] + regime_effect[regime] + 0.2 * near_first * q[4]
                 + 0.1 * log_near * q[5] + 0.1 * log_far * q[6] + latent_shift * latent * horizon
                 + 0.05 * rng.normal(size=d))
            features = [log_near, log_far, rng.uniform(1, 3), rng.uniform(0.1, 1), near_first,
                        np.log10(UNIT_YEARS[rendered]), regime == "near", regime == "both", False, 0.0]
            rows.append({"X": x, "y": log_h, "duration_id": duration, "durations": f"{bank}|{duration}",
                         "pairs": f"{bank}|{base}", "regime": regime, "chose_near": float(rng.random() < p_near),
                         "features": np.array(features, dtype=float)})
    X = np.stack([r["X"] for r in rows])
    data = {key: np.array([r[key] for r in rows]) for key in rows[0] if key != "X"}
    return X, data, {"horizon": horizon, "unit": unit}
