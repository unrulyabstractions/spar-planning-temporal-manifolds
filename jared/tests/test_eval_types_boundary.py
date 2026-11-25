"""Boundary analysis on small synthetic arrays: planted direction, shuffled labels, item-grouped permutation,
targets, unit words, tier rules."""

import numpy as np

from pipeline.s3_analysis.eval_types_boundary import (BankCell, analyse_rows, permute_items, position_index,
                                                      row_target, tier, unit_word)

DIM = 60
UNITS = ["day", "week", "month", "year"]


def planted(rng, levels, rows_per_level, noise=0.05):
    """Activations = log horizon along a fixed direction plus noise."""
    direction = np.random.default_rng(1).normal(size=DIM)
    y = np.repeat(levels, rows_per_level)
    return y[:, None] * direction + noise * rng.normal(size=(len(y), DIM)), y


def eval_set(rng, shuffle=False):
    """30 items x 2 rows (base, swap), one horizon per item."""
    X, y = planted(rng, np.repeat(np.linspace(-2, 2, 30), 2), 1)
    items = np.repeat(np.arange(30), 2)
    units = np.array([UNITS[i % 4] for i in items], dtype=object)
    if shuffle:
        y = permute_items(y, items, 1, rng)[:, 0]
    return X, y, items, units


def bank(rng):
    return BankCell(*planted(rng, np.linspace(-2, 2, 40), 3))


def test_planted_direction_is_strong():
    rng = np.random.default_rng(0)
    X, y, items, units = eval_set(rng)
    report, _ = analyse_rows(X, y, items, units, bank(rng), {u: 0.0 for u in UNITS}, 100, 50, rng)
    assert report["fixed"]["tier"] == "strong" and report["refit"]["tier"] == "strong"
    assert report["fixed"]["spearman"] > 0.95 and abs(report["fixed"]["bias"]) < 0.5
    assert report["pca"]["pc1_spearman"] > 0.9 and not report["moved"]


def test_shuffled_labels_fail():
    rng = np.random.default_rng(0)
    X, y, items, units = eval_set(rng, shuffle=True)
    report, _ = analyse_rows(X, y, items, units, bank(rng), {u: 0.0 for u in UNITS}, 100, 50, rng)
    assert report["fixed"]["tier"] == "fail" and report["refit"]["tier"] == "fail"


def test_refit_cannot_score_by_memorising_horizons():
    """Few horizons shared by several items, each horizon with its own unordered code: a refit whose folds
    keep test horizons in training would read them back; holding out whole horizons must not."""
    rng = np.random.default_rng(0)
    levels = np.linspace(-2, 2, 10)
    codes = rng.normal(size=(len(levels), DIM))           # one random vector per horizon, no order
    level_of_item = np.repeat(np.arange(len(levels)), 4)   # 40 items, 4 per horizon
    items = np.repeat(np.arange(len(level_of_item)), 2)    # base + swap
    y = levels[level_of_item][items]
    X = codes[level_of_item][items] + 0.05 * rng.normal(size=(len(items), DIM))
    units = np.array([UNITS[i % 4] for i in items], dtype=object)
    report, _ = analyse_rows(X, y, items, units, None, {u: 0.0 for u in UNITS}, 100, 50, rng)
    assert report["refit"]["tier"] in ("fail", "above_random")
    assert report["refit"]["spearman"] < 0.5


def test_unit_lookup_beating_probe_blocks_strong():
    rng = np.random.default_rng(0)
    X, y, items, units = eval_set(rng)
    units = np.array([f"u{round(v, 3)}" for v in y], dtype=object)   # each target has its own unit word
    table = {u: float(u[1:]) for u in units}
    report, _ = analyse_rows(X, y, items, units, bank(rng), table, 100, 50, rng)
    assert report["unit_bank"]["spearman"] == 1.0
    assert report["fixed"]["tier"] == "weak"


def test_permute_items_moves_items_whole():
    y = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0])
    items = np.array([0, 0, 0, 1, 1, 2, 3, 3])
    draws = permute_items(y, items, 20, np.random.default_rng(0))
    assert draws.shape == (8, 20)
    for col in draws.T:
        assert all(len(set(col[items == i])) == 1 for i in range(4))
        assert sorted({col[0], col[3], col[5], col[6]}) == [1.0, 2.0, 3.0, 4.0]
    assert any(not np.array_equal(col, y) for col in draws.T)


def test_row_target():
    assert row_target({"horizon": 10.0}) == 1.0
    assert np.isclose(row_target({"horizon": None, "interval_years": [0.1, 10.0]}), 0.0)
    assert np.isclose(row_target({"horizon": float("nan"), "interval_years": [1, 100]}), 1.0)
    assert row_target({"horizon": None}) is None
    assert row_target({"horizon": 2.0, "interval_years": [1, 100]}) == np.log10(2.0)


def test_unit_word_normalizes_plural():
    assert unit_word("3 weeks") == unit_word("1 week") == "week"
    assert unit_word("4.22 years") == "year" and unit_word("Hours") == "hour"


def test_position_index():
    on = ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "think@0.25*", "think@0.5*", "think@1*",
          "</think>", "\n\n", ":", " a*"]
    off = ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n", ":"]
    assert [position_index(on, p) for p in ("suffix_end", "think_end", "prelabel", "phrase_end")] == [4, 7, 10, None]
    assert [position_index(off, p) for p in ("suffix_end", "think_end", "prelabel")] == [4, None, 9]


def test_tier_edge_cases():
    nan = float("nan")
    assert tier(0.95, 0.85, 0.3, 0.2, nan, True) == "strong"
    assert tier(0.95, 0.80, 0.3, 0.2, nan, True) == "strong"      # boundary is inclusive
    assert tier(0.95, 0.85, 0.3, 1.5, nan, True) == "weak"        # fixed probe: bias over a decade
    assert tier(0.95, 0.85, 0.3, 1.5, nan, False) == "strong"     # refit: bias not checked
    assert tier(0.95, 0.85, 0.3, 0.2, 0.97, True) == "weak"       # unit lookup does better
    assert tier(0.95, 0.85, 0.3, 0.2, 0.95, True) == "weak"       # a tie is not above
    assert tier(0.6, 0.5, 0.3, 0.0, nan, True) == "weak"
    assert tier(0.4, 0.35, 0.3, 0.0, nan, True) == "above_random"
    assert tier(0.4, 0.30, 0.3, 0.0, nan, True) == "fail"         # must exceed the null p95
    assert tier(0.1, float("nan"), 0.3, 0.0, nan, True) == "fail"


def test_unit_lookup_only_uses_a_unit_word_in_the_prompt():
    """An implied row carries its twin's horizon_text as the label; the lookup must not read it."""
    from pipeline.s3_analysis.eval_types_boundary import analyse_type
    rng = np.random.default_rng(0)
    X, y, items, _ = eval_set(rng)
    texts = [f"{1 + i % 9} {UNITS[i % 4]}s" for i in items]
    rows = [{"sample_uid": f"r{k}", "twin_id": f"i{i}", "horizon": float(10 ** v), "horizon_text": t,
             "prompt": "until the lease ends" if k % 2 else [{"role": "user", "content": f"horizon: {t}."}]}
            for k, (i, v, t) in enumerate(zip(items, y, texts))]
    record, _ = analyse_type(rows, X, None, {u: float(j) for j, u in enumerate(UNITS)}, 50, 20, rng)
    unit_rows = np.isfinite(np.asarray(record["predictions"]["unit_bank"], dtype=float))
    assert unit_rows.tolist() == [k % 2 == 0 for k in range(len(rows))]
