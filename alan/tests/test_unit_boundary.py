import math
from collections import defaultdict

from ptm.horizons import Horizon
from ptm.prompts import CONSTRAINT_PHRASINGS
from ptm.unit_boundary import UNITS, delay_in, horizon_in, ratio_grid, unit_boundary_set


def test_delay_and_horizon_renderings():
    got = {u: (str(delay_in(1, u)), str(delay_in(10, u))) for u in UNITS}
    print("\ndelays 1 y / 10 y:", got)
    assert got == {"weeks": ("52 weeks", "522 weeks"), "months": ("12 months", "120 months"),
                   "quarters": ("4 quarters", "40 quarters"), "years": ("1 year", "10 years")}
    assert horizon_in(9.3, "quarters") == Horizon(37.2, "quarters") and horizon_in(9.3, "months") == Horizon(112, "months")
    g = ratio_grid(24, 1.5)
    assert math.isclose(g[0], 1 / 1.5) and math.isclose(g[-1], 1.5) and all(abs(r - 1) > 1e-3 for r in g)


def test_set_is_consistent_and_only_units_and_horizon_vary():
    samples, dropped = unit_boundary_set(n=12, long_delays=(2, 10), phrasings=(0, 2))
    print(f"\n{len(samples)} prompts, dropped {dropped}")
    by_pair = defaultdict(list)
    for s in samples:
        # parameters agree with the rendered text
        assert math.isclose(s.horizon_years, Horizon.parse(s.horizon_text.replace(",", "")).years, rel_tol=1e-12)
        assert math.isclose(s.long_delay_years, Horizon.parse(s.long_delay_text).years)
        assert math.isclose(s.short_delay_years, Horizon.parse(s.short_delay_text).years)
        assert f" in {s.short_delay_text}." in s.text and f" in {s.long_delay_text}." in s.text
        assert CONSTRAINT_PHRASINGS[s.phrasing_id].format(horizon=s.horizon_text) in s.text
        # horizon inside the sweep and never equal to the long delay
        r = s.horizon_years / s.long_delay_years
        assert 1 / 1.5 * 0.95 < r < 1.5 * 1.05 and abs(r - 1) > 1e-9
        assert s.long_reward / s.short_reward in (5, 20)
        by_pair[s.pair_id].append(s)
    worst = 0.0
    for pid, ss in by_pair.items():
        base = ss[0].text.split("\n")
        y = [s.horizon_years for s in ss]
        worst = max(worst, max(y) / min(y) - 1)
        for s in ss[1:]:
            diff = {i for i, (l0, l1) in enumerate(zip(base, s.text.split("\n"))) if l0 != l1}
            assert diff <= {2, 3, 5}, (pid, diff)       # the two option lines and the CONSTRAINT line only
    print(f"{len(by_pair)} pairs; largest duration spread within a pair {worst:.4f}")
    assert worst < 0.07                                   # integer months at 16 months: 0.5/16 each way
    assert len({s.sample_uid for s in samples}) == len(samples) and len({s.text for s in samples}) == len(samples)


def test_switch_point_recovers_known_boundaries():
    import numpy as np
    from ptm.unit_boundary import switch_point
    rng = np.random.default_rng(0)
    x = np.tile(np.log(ratio_grid(24, 1.5)), 16)
    for mu, s in [(0.0, 0.03), (np.log(0.85), 0.03), (np.log(0.85), 0.10)]:
        y = (rng.random(x.size) < 1 / (1 + np.exp((x - mu) / s))).astype(int)
        est = switch_point(x, y)
        print(f"\ntrue switch H/D {np.exp(mu):.3f} (width {s}) -> estimated {np.exp(est):.3f}")
        assert abs(est - mu) < 0.03
    step = (x < np.log(0.9)).astype(int)                 # a clean step: the estimate lies between the grid points around it
    est = switch_point(x, step)
    lo, hi = x[x < np.log(0.9)].max(), x[x >= np.log(0.9)].min()
    print(f"clean step between {np.exp(lo):.3f} and {np.exp(hi):.3f} -> {np.exp(est):.3f}")
    assert lo <= est <= hi
    assert np.isnan(switch_point(x, np.ones_like(x, dtype=int)))
    assert np.isnan(switch_point(x, 1 - step))            # rising curve: not a switch from short to long
