import numpy as np
from ptm.horizons import Horizon
from ptm.matrix import MATRIX_HORIZONS, HELDOUT_HORIZONS, TRAIN_HORIZONS, MatrixConfig, build_matrix, build_controls, scale_horizon
from ptm.prompts import to_frame, from_frame


def test_grid():
    assert len(MATRIX_HORIZONS) == 12 and len(HELDOUT_HORIZONS) == 4 and len(TRAIN_HORIZONS) == 8
    ys = [h.years for h in MATRIX_HORIZONS]; assert ys == sorted(ys)
    assert MATRIX_HORIZONS[0] == TRAIN_HORIZONS[0] and MATRIX_HORIZONS[-1] == TRAIN_HORIZONS[-1]   # endpoints are trained


def test_matrix_is_a_full_crossing_with_config_level_splits():
    cfg = MatrixConfig(n_configs=6, n_test_configs=2, n_dev_configs=1, seed=3)
    df = to_frame(build_matrix(cfg))
    assert len(df) == 6 * 2 * 12 * 3
    assert df.groupby(["config_id", "domain", "horizon_text", "rendering"]).size().max() == 1
    assert df.groupby("config_id").split.nunique().max() == 1 and df.groupby("split").config_id.nunique().to_dict() == {"dev": 1, "test": 2, "train": 3}
    # option parameters fixed within a config across domains, horizons and renderings
    for col in ["short_reward", "long_reward", "short_delay_text", "long_delay_text", "short_first"]:
        assert df.groupby("config_id")[col].nunique().max() == 1
    assert df.prompt_id.nunique() == len(df) and df.sample_uid.nunique() == len(df)


def test_renderings_preserve_facts():
    df = to_frame(build_matrix(MatrixConfig(n_configs=4, n_test_configs=1, n_dev_configs=1, seed=4)))
    for r in df.itertuples():
        for fact in [f"{int(r.short_reward):,}", f"{int(r.long_reward):,}", r.short_delay_text, r.long_delay_text, r.horizon_text, "I choose:", "My reasoning:"]:
            assert fact in r.text, (r.rendering, fact)
        if r.rendering == "structured":
            assert "SITUATION:" in r.text and "CONSTRAINT:" in r.text
        else:
            assert "SITUATION:" not in r.text and "CONSTRAINT:" not in r.text
        # label a) always precedes b), and the option listed under a) matches short_first (search within the options span)
        ia, ib = r.text.index("a)"), r.text.index("b)")
        assert ia < ib
        span_a, span_b = r.text[ia:ib], r.text[ib:ib + 80]
        first_delay, second_delay = (r.short_delay_text, r.long_delay_text) if r.short_first else (r.long_delay_text, r.short_delay_text)
        assert first_delay in span_a and second_delay in span_b, (r.rendering, span_a, span_b)
    assert df.prompt_id.nunique() == len(df)   # varied never coincides with plain
    varied = df[df.rendering == "varied"]
    assert varied.variant.nunique() == 8 and varied.groupby(["config_id", "domain", "horizon_text"]).variant.nunique().max() == 1


def test_controls():
    cfg = MatrixConfig(n_configs=8, n_test_configs=2, n_dev_configs=1, seed=5, n_control_configs=3)
    k = to_frame(build_controls(cfg))
    assert set(k.condition) == {"mention", "role", "factual", "scaling"} and (k.split == "control").all()
    m = k[k.condition == "mention"]
    assert (m.distractor_years != m.horizon_years).all() and all(f"established {d} ago" in t for d, t in zip(m.distractor_text, m.text))
    r = k[k.condition == "role"]; assert all(f"You have {d} to prepare" in t for d, t in zip(r.distractor_text, r.text))
    f = k[k.condition == "factual"]; assert all(("I choose" not in t) and (h in t) for t, h in zip(f.text, f.horizon_text))
    s = k[k.condition == "scaling"]
    for (cid, dom, base), g in s.groupby(["config_id", "domain", "scale_base_horizon"]):
        assert sorted(g.scale) == [0.1, 1.0, 10.0]
        ratio = (g.horizon_years / g.short_delay_years).to_numpy(); assert np.all(np.abs(ratio / ratio[0] - 1) < 0.25), ratio   # ratios preserved to rounding
        one = g[g.scale == 1.0].iloc[0]; assert one.horizon_text == base
    # control configs come from the training split only
    train_ids = set(to_frame(build_matrix(cfg)).query("split == 'train'").config_id)
    assert set(k.config_id) <= train_ids


def test_scale_horizon():
    assert str(scale_horizon(Horizon.parse("1 month"), 10)) == "10 months"
    assert str(scale_horizon(Horizon.parse("10 years"), 10)) == "100 years" and str(scale_horizon(Horizon.parse("30 years"), 10)) == "300 years"
    for t in ["1 week", "1 month", "6 months", "2 years", "10 years", "30 years"]:
        assert str(scale_horizon(Horizon.parse(t), 1.0)) == t     # scale 1 reproduces the grid rendering
    assert abs(scale_horizon(Horizon.parse("1 year"), 0.1).years / 0.1 - 1) < 0.05


def test_frame_roundtrip_keeps_matrix_columns():
    s = build_matrix(MatrixConfig(n_configs=2, n_test_configs=1, n_dev_configs=0, seed=6))
    back = from_frame(to_frame(s))
    assert [(x.text, x.scenario_id, x.rendering, x.split, x.horizon_heldout) for x in back] == [(x.text, x.scenario_id, x.rendering, x.split, x.horizon_heldout) for x in s]
