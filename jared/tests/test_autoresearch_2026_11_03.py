"""Tests for the 2026-11-03 overnight analyses: unit_number_rsa, breakeven_rate, caa_confound,
trace_audit, decision_emergence. Synthetic data only."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import breakeven_rate as br  # noqa: E402
import caa_confound as caa  # noqa: E402
import decision_emergence as de  # noqa: E402
import trace_audit as ta  # noqa: E402
import unit_number_rsa as rsa  # noqa: E402

UNIT_YEARS = {"minutes": 1 / 525960, "hours": 1 / 8766, "days": 1 / 365.25, "weeks": 7 / 365.25}
RENDERINGS = [("60", "minutes", "1 hour"), ("1", "hours", "1 hour"), ("120", "minutes", "2 hours"),
              ("2", "hours", "2 hours"), ("24", "hours", "1 day"), ("1", "days", "1 day"),
              ("7", "days", "1 week"), ("1", "weeks", "1 week"), ("2", "weeks", "2 weeks"),
              ("14", "days", "2 weeks"), ("60", "days", "60 days"), ("3", "days", "3 days"),
              ("72", "hours", "3 days"), ("12", "hours", "12 hours")]


def table():
    rows = []
    for numeral, unit, duration in RENDERINGS:
        years = float(numeral) * UNIT_YEARS[unit]
        rows.append({"text": f"{numeral} {unit}", "duration": duration, "log_years": float(np.log10(years)),
                     "unit": unit, "numeral": numeral, "log_numeral": float(np.log10(float(numeral)))})
    return rows


def synthetic_cells(code, n_contexts=24, noise=0.5, seed=0):
    """(contexts, cells, dims): a cell code plus independent noise per context."""
    rng = np.random.default_rng(seed)
    return code[None] + noise * rng.normal(size=(n_contexts, *code.shape))


def test_pure_duration_code_reads_as_integrated():
    rows = table()
    rng = np.random.default_rng(1)
    log_years = np.array([r["log_years"] for r in rows])
    code = np.column_stack([log_years, np.cos(log_years), np.sin(log_years)]) @ rng.normal(size=(3, 64)) * 3
    pairs, _, facts = rsa.pair_design(rows)
    distance = rsa.cv_distances(synthetic_cells(code), pairs, rng)
    out = rsa.contrasts(distance, facts)
    assert out["same duration, other unit"] < 0.05
    assert out["integration ratio, other-unit reference"] < 0.2


def test_pure_unit_code_reads_as_text():
    rows = table()
    rng = np.random.default_rng(2)
    units = sorted({r["unit"] for r in rows})
    code = np.eye(len(units))[[units.index(r["unit"]) for r in rows]] @ rng.normal(size=(len(units), 64)) * 3
    pairs, _, facts = rsa.pair_design(rows)
    out = rsa.contrasts(rsa.cv_distances(synthetic_cells(code), pairs, rng), facts)
    assert out["integration ratio"] > 10          # same-unit pairs sit at ~0
    assert abs(out["numeral ratio"] - 1) < 0.25   # no numeral code planted


def test_cv_distances_are_unbiased_under_pure_noise():
    rows = table()
    rng = np.random.default_rng(3)
    pairs, _, _ = rsa.pair_design(rows)
    M = rng.normal(size=(24, len(rows), 64))
    raw = np.zeros(len(pairs))
    for _ in range(20):
        order = rng.permutation(24)
        a, b = M[order[:12]].mean(0), M[order[12:]].mean(0)
        raw += ((a[pairs[:, 0]] - a[pairs[:, 1]]) * (b[pairs[:, 0]] - b[pairs[:, 1]])).sum(1)
    # the naive squared distance would be ~2 * 64 / 12 ≈ 10.7 per pair; the split product centres on 0
    assert abs(raw.mean() / 20) < 1.0


def test_resampled_halves_stay_disjoint():
    rows = table()
    pairs, _, _ = rsa.pair_design(rows)
    M = np.zeros((24, len(rows), 4))
    M[:, :, 0] = np.arange(24)[:, None]            # each context carries its own offset in dim 0
    seen = []

    class Recorder:
        def __init__(self):
            self.rng = np.random.default_rng(4)

        def permutation(self, n):
            return self.rng.permutation(n)

        def choice(self, half, size):
            draw = self.rng.choice(half, size)
            seen.append(set(draw.tolist()))
            return draw

    rsa.cv_distances(M, pairs, Recorder(), resample=True)
    for first, second in zip(seen[::2], seen[1::2]):
        assert not first & second


def test_breakeven_rate_matches_compound_growth_and_caps():
    assert np.isclose(br.log10_breakeven(1.07 ** 10, 10.0), np.log10(0.07))
    assert br.log10_breakeven(25.0, 1e-6) == br.RATE_CAP
    assert br.log10_breakeven(2.0, 1e-3) == br.RATE_CAP


def test_caa_directions_recover_planted_forced_and_near_axes():
    rng = np.random.default_rng(5)
    n, d = 400, 50
    y = rng.uniform(-3, 2, n)
    durations = np.round(y, 1).astype(str)
    forced = rng.random(n) < 0.3
    near = rng.random(n) < 0.5
    regime = np.where(y < 0, "none", "both")
    axis_forced, axis_near = np.eye(d)[0], np.eye(d)[1]
    X = 0.1 * rng.normal(size=(n, d)) + 2 * forced[:, None] * axis_forced + 2 * near[:, None] * axis_near
    labels = {"y": y, "durations": durations, "forced": forced, "chose_near": near, "regime": regime}
    dirs, _ = caa.directions(X, labels)
    assert caa.unit(dirs["forced"]) @ axis_forced > 0.95
    assert caa.unit(dirs["near"]) @ axis_near > 0.95


def test_trace_themes_match_their_phrases():
    text = "The far payout arrives within the 39-year time horizon, but there is a risk of inflation."
    flags = {name: bool(p.search(text)) for name, p in ta.THEME_RE.items()}
    assert flags["within horizon"] and flags["risk"] and flags["inflation"]
    assert not flags["beyond horizon"] and not flags["reinvest"]


def test_decision_probe_finds_a_planted_signal_and_not_noise():
    rng = np.random.default_rng(6)
    n = 300
    groups = np.repeat(np.arange(150), 2)
    y = (rng.random(n) < 0.5).astype(int)
    # the reader z-scores each dimension, then keeps the top-64 PCs: a signal is found when it is spread
    # over many dimensions with variance above the noise eigenvalues (an axis-aligned one is z-scored
    # down to unit variance and can be missed; a lower bound, by design)
    direction = rng.choice([-1.0, 1.0], 200)
    X_signal = rng.normal(size=(n, 200)) + 0.5 * y[:, None] * direction
    X_noise = rng.normal(size=(n, 200))
    auc_signal = de.roc_auc_score(y, de.out_of_fold(de.pca_features(X_signal), y, groups))
    auc_noise = de.roc_auc_score(y, de.out_of_fold(de.pca_features(X_noise), y, groups))
    assert auc_signal > 0.75
    assert abs(auc_noise - 0.5) < 0.12


def test_wilson_interval_brackets_the_share_and_handles_extremes():
    import trace_coding_summary as tcs
    lo, hi = tcs.wilson(0, 45)
    assert lo == 0 or abs(lo) < 1e-12
    assert 0.05 < hi < 0.1                      # the rule-of-three bound is ~3/45 = 0.067
    lo, hi = tcs.wilson(30, 60)
    assert lo < 0.5 < hi


def test_digit_design_flags_shared_digit_count_and_digits():
    import numeral_digits as nd
    rows = [{"numeral": "60"}, {"numeral": "6"}, {"numeral": "90"}]
    D = nd.digit_design(rows)                   # pairs (60, 6), (60, 90), (6, 90)
    assert D.tolist() == [[0, 1, 0], [1, 0, 1], [0, 0, 0]]
