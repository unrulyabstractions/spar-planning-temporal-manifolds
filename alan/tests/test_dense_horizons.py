"""Dense single-unit horizon set (ptm.prompts.dense_integer_grid, fixed_scenario): only the horizon varies."""
import numpy as np

from ptm.horizons import Horizon
from ptm.prompts import dense_integer_grid, fixed_scenario


def test_dense_grid_distinct_sorted_and_spans_range():
    v = dense_integer_grid(1500, 36525)
    steps = np.diff(np.log10(v))
    print(f"\n  {len(v)} values, {v[:5]} ... {v[-3:]}; every integer up to {next(i for i, (a, b) in enumerate(zip(v, v[1:]), 1) if b - a > 1)}; "
          f"log10 step median {np.median(steps):.4f}, max {steps.max():.3f}")
    assert 1500 <= len(v) <= 1502 and v[0] == 1 and v[-1] == 36525
    assert all(b > a for a, b in zip(v, v[1:]))


def test_fixed_scenario_differs_only_in_constraint_line():
    hs = [Horizon(d, "days") for d in dense_integer_grid(200, 36525)]
    ss = fixed_scenario(hs, short_reward=10_000, short_delay=Horizon(1, "years"), long_reward=50_000,
                        long_delay=Horizon(10, "years"))
    lines = [s.text.split("\n") for s in ss]
    k = next(i for i, l in enumerate(lines[0]) if l.startswith("CONSTRAINT:"))
    assert all(len(l) == len(lines[0]) for l in lines)
    for i in range(len(lines[0])):
        same = len({l[i] for l in lines}) == 1
        assert same == (i != k), f"line {i} varies={not same}"
    assert len({s.text for s in ss}) == len(ss)
    assert all(s.horizon_text in s.text and s.horizon_unit == "days" for s in ss)
    np.testing.assert_allclose([s.horizon_years for s in ss], [h.value / 365.25 for h in hs])
    print("\n  " + ss[0].text.replace("\n", "\n  ") + f"\n  ... line {k} varies: {lines[1][k]!r} / {lines[-1][k]!r}")


def test_short_reward_control_differs_only_in_option_line_and_matches_number_strings():
    vals = dense_integer_grid(200, 36525)
    hs = fixed_scenario([Horizon(v, "days") for v in vals], short_reward=10_000, short_delay=Horizon(1, "years"),
                        long_reward=50_000, long_delay=Horizon(10, "years"), reward_commas=False)
    rs = fixed_scenario(Horizon(5, "years"), short_reward=[float(v) for v in vals], short_delay=Horizon(1, "years"),
                        long_reward=50_000, long_delay=Horizon(10, "years"), reward_commas=False)
    lines = [s.text.split("\n") for s in rs]
    for i in range(len(lines[0])):
        assert (len({l[i] for l in lines}) == 1) == (i != 2), f"line {i}"
    assert len({s.text for s in rs}) == len(rs) and {s.horizon_text for s in rs} == {"5 years"}
    # the varying number string is identical to the horizon set's, token-for-token as text
    for h, r, v in zip(hs, rs, vals):
        assert f"horizon: {v} day" in h.text and f"a) {v} dollars in 1 year." in r.text   # "1 day" is singular
    assert "b) 50000 dollars in 10 years." in rs[0].text
    print("\n  " + rs[-1].text.replace("\n", "\n  "))
    import pytest
    with pytest.raises(ValueError):
        fixed_scenario([Horizon(1, "days")], short_reward=[1.0], short_delay=Horizon(1, "years"), long_reward=5.0,
                       long_delay=Horizon(10, "years"))


def test_reward_commas_default_keeps_existing_rendering():
    from ptm.prompts import DatasetConfig, generate
    s = generate(DatasetConfig(n=20, seed=0))
    assert all(x.reward_commas for x in s) and any("," in x.short_line for x in s if x.short_reward >= 1000)


def test_constraint_first_moves_only_the_constraint_line():
    from ptm.prompts import PromptFormat
    vals = dense_integer_grid(50, 36525)
    kw = dict(short_reward=10_000, short_delay=Horizon(1, "years"), long_reward=50_000, long_delay=Horizon(10, "years"))
    base = fixed_scenario([Horizon(v, "days") for v in vals], **kw)
    moved = fixed_scenario([Horizon(v, "days") for v in vals], fmt=PromptFormat(name="cf", constraint_first=True), **kw)
    for b, m in zip(base, moved):
        bl, ml = b.text.split("\n"), m.text.split("\n")
        assert ml == bl[:2] + [bl[5]] + bl[2:5] + bl[6:]
        assert ml[2].startswith("CONSTRAINT:")
    print("\n  " + moved[0].text.replace("\n", "\n  "))


def test_long_reward_varies_only_the_long_option_line():
    vals = dense_integer_grid(100, 36525)
    ss = fixed_scenario(Horizon(20, "years"), short_reward=1000, long_reward=[float(v) for v in vals],
                        short_delay=Horizon(1, "years"), long_delay=Horizon(10, "years"), reward_commas=False)
    lines = [s.text.split("\n") for s in ss]
    for i in range(len(lines[0])):
        assert (len({l[i] for l in lines}) == 1) == (i != 3), f"line {i}"
    assert all(f"b) {v} dollars in 10 years." in s.text for s, v in zip(ss, vals))
    assert min(s.long_reward for s in ss) < 1000 < max(s.long_reward for s in ss)   # crosses the short reward on purpose


def test_pooled_units_generator(tmp_path):
    import subprocess, sys, pandas as pd
    out = tmp_path / "pooled.parquet"
    subprocess.run([sys.executable, "scripts/gen_dense_horizons.py", str(out), "--units", "days:40,weeks:40,months:40,years:20"],
                   check=True, capture_output=True)
    df = pd.read_parquet(out)
    print("\n  " + str(df.groupby("horizon_unit").horizon_value.agg(["size", "min", "max"]).to_dict()))
    assert df.horizon_unit.value_counts().to_dict() == {"days": 40, "weeks": 40, "months": 40, "years": 20}
    assert df.text.nunique() == len(df) and df.horizon_years.max() <= 100.0 + 1e-9
    assert df.groupby("horizon_unit").horizon_value.max().to_dict() == {"days": 36525, "weeks": 5217, "months": 1200, "years": 100}


def test_matched_duration_units(tmp_path):
    import subprocess, sys, pandas as pd
    out = tmp_path / "matched.parquet"
    r = subprocess.run([sys.executable, "scripts/gen_dense_horizons.py", str(out), "--matched-base", "years", "--n", "100",
                        "--max", "100", "--units", "days,weeks,months,years"], check=True, capture_output=True, text=True)
    df = pd.read_parquet(out)
    print("\n  " + "\n  ".join(l for l in r.stdout.splitlines() if "max relative" in l))
    assert df.horizon_unit.value_counts().to_dict() == {"days": 100, "weeks": 100, "months": 100, "years": 100}
    w = df.pivot(index="pair_id", columns="horizon_unit", values="horizon_years")
    assert (abs(w["days"] / w["years"] - 1) < 0.002).all() and (abs(w["weeks"] / w["years"] - 1) < 0.01).all()
    assert (w["months"] == w["years"]).all()                               # 12 y months = y years exactly
    assert df.text.nunique() == len(df)


def test_horizon_renderings_keep_value_and_unit():
    vals = [1, 7, 37, 120, 1200]
    kw = dict(short_reward=10_000, long_reward=50_000, short_delay=Horizon(1, "years"), long_delay=Horizon(10, "years"))
    hs = [Horizon(v, "years") for v in vals]
    plain, pad, dec = (fixed_scenario(hs, horizon_render=r, **kw) for r in ("plain", "pad4", "dec2"))
    assert [s.horizon_text for s in plain] == ["1 year", "7 years", "37 years", "120 years", "1200 years"]
    assert [s.horizon_text for s in pad] == ["0001 years", "0007 years", "0037 years", "0120 years", "1200 years"]
    assert [s.horizon_text for s in dec] == ["1.00 years", "7.00 years", "37.00 years", "120.00 years", "1200.00 years"]
    for a, b in zip(plain, pad):
        assert a.horizon_years == b.horizon_years and a.text.split("\n")[:5] == b.text.split("\n")[:5]
    assert {s.rendering for s in plain} == {None} and {s.rendering for s in pad} == {"pad4"}


def test_matched_decimal_years(tmp_path):
    import subprocess, sys, pandas as pd
    out = tmp_path / "dec.parquet"
    subprocess.run([sys.executable, "scripts/gen_dense_horizons.py", str(out), "--matched-base", "months", "--n", "1200",
                    "--max", "1200", "--units", "years", "--render", "dec2"], check=True, capture_output=True)
    df = pd.read_parquet(out)
    assert len(df) == 1200 and df.text.nunique() == 1200 and set(df.rendering) == {"dec2"}
    assert df.horizon_text.iloc[0] == "0.08 years" and df.horizon_text.iloc[36] == "3.08 years" and df.horizon_text.iloc[-1] == "100.00 years"
    base = df.pair_id.str.split("_").str[1].astype(int) / 12
    err = (df.horizon_years / base - 1).abs()
    assert err.max() < 0.041 and err[base >= 1].max() < 0.005                # 0.08 for 1/12; <= 0.5 % from 1 year
