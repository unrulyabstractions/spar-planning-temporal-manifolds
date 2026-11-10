"""Unit-compatibility boundary design: does writing the horizon and the option delays in different units move the
point where the choice switches?

One option pair per condition: short = SHORT_REWARD in 1 year, long = ratio x SHORT_REWARD in D years, with both
delays written in the delay unit ("52 weeks" / "12 months" / "4 quarters" / "1 year"). The horizon is swept densely
around the long delay, H/D log-uniform in [1/span, span], and written in the horizon unit. Under the constraint
("greatest benefit for this time horizon"), the short option is correct iff H < the long delay, so the switch point
is known. Crossed: horizon unit x delay unit x D x reward ratio x option order x constraint phrasing.

Renderings: weeks and months as integers, quarters and years with two decimals ("9.25 years", as in the 2026-10-03
years/months cells). The horizon's value is the rendered one, so horizon_years always matches the text; durations of
one pair_id agree across units up to that rounding. Weeks delays are rounded to whole weeks (52 weeks = 0.9966 y),
so analyses must use long_delay_years, not the nominal D.
"""

from __future__ import annotations

from .horizons import UNIT_YEARS, Horizon, canonical_unit
from .prompts import PromptSample, fixed_scenario

UNITS = ("weeks", "months", "quarters", "years")
LONG_DELAYS_Y = (2, 5, 10)
REWARD_RATIOS = (5, 20)
SHORT_REWARD = 10_000
PHRASINGS = (0, 1, 2, 3)
RENDER = {"weeks": "plain", "months": "plain", "quarters": "dec2", "years": "dec2"}


def delay_in(years: float, unit: str) -> Horizon:
    """A delay of `years` written in `unit` as a whole count (52 weeks for 1 year)."""
    u = canonical_unit(unit)
    return Horizon(max(1, round(years / UNIT_YEARS[u])), u)


def horizon_in(years: float, unit: str) -> Horizon:
    """A horizon of `years` written in `unit` at that unit's rendering precision (integer or two decimals)."""
    u = canonical_unit(unit)
    v = years / UNIT_YEARS[u]
    return Horizon(int(round(v)), u) if RENDER[u] == "plain" else Horizon(round(v, int(RENDER[u][3:])), u)


def ratio_grid(n: int, span: float) -> list[float]:
    """n ratios H/D log-uniform in [1/span, span]; n even keeps the nominal grid off H = D."""
    return [span ** (2 * i / (n - 1) - 1) for i in range(n)]


def unit_boundary_set(n: int = 24, span: float = 1.5, horizon_units=UNITS, delay_units=UNITS, long_delays=LONG_DELAYS_Y,
                      ratios=REWARD_RATIOS, orders=(True, False), phrasings=PHRASINGS) -> tuple[list[PromptSample], dict]:
    """All crossed prompts, and counts of what was dropped. Within a condition, horizons that round to the same
    rendered value are kept once; a horizon exactly equal to the long delay (ambiguous under the constraint) is dropped."""
    out: list[PromptSample] = []
    dropped = {"duplicate": 0, "equals_delay": 0}
    grid = ratio_grid(n, span)
    for D in long_delays:
        for du in delay_units:
            sd, ld = delay_in(1.0, du), delay_in(float(D), du)
            for hu in horizon_units:
                hs, idx, seen = [], [], set()
                for i, r in enumerate(grid):
                    h = horizon_in(r * D, hu)
                    if h.value in seen:
                        dropped["duplicate"] += len(ratios) * len(orders) * len(phrasings); continue
                    if abs(h.years - ld.years) < 1e-12:
                        dropped["equals_delay"] += len(ratios) * len(orders) * len(phrasings); continue
                    seen.add(h.value); hs.append(h); idx.append(i)
                for x in ratios:
                    for sf in orders:
                        for p in phrasings:
                            tag = f"D{D}_x{x}_{'sf' if sf else 'lf'}_p{p}"
                            ss = fixed_scenario(hs, short_reward=SHORT_REWARD, short_delay=sd, long_reward=SHORT_REWARD * x,
                                                long_delay=ld, short_first=sf, horizon_render=RENDER[hu], phrasing_id=p,
                                                uid_prefix=f"ub_{tag}_H{hu}_d{du}")
                            for s, i in zip(ss, idx):
                                s.sample_uid = f"ub_{tag}_H{hu}_d{du}_{i:02d}"
                                s.pair_id = f"ub_{tag}_{i:02d}"
                                s.scenario_id = f"D{D}_x{x}"
                            out += ss
    return out, dropped


def switch_point(x, y) -> float:
    """x at P(y = 1) = 0.5 from a logistic fit P = sigmoid(b0 + b1 x) of binary y (1 = short) on x = ln(H/D); NaN when y
    never changes or the fitted curve does not fall with x. C = 1e4: a near-separable cell (a clean step) gets a steep
    but finite slope, and the switch point stays inside the step."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    x, y = np.asarray(x, float), np.asarray(y, int)
    if y.size < 2 or y.min() == y.max():
        return float("nan")
    m = LogisticRegression(C=1e4, max_iter=1000).fit(x[:, None], y)
    b1 = m.coef_[0, 0]
    return float(-m.intercept_[0] / b1) if b1 < 0 else float("nan")
