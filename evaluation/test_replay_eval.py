"""Tests for the runner's own logic (no GPU): label location, the other label's token, strata shuffles, and the
meaning/letter and odd/even arithmetic of analyze.py. Run (from evaluation/): python -m pytest -q test_replay_eval.py"""

import numpy as np
import pytest

import analyze
import replay_eval as re_


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen3-1.7B tokenizer not cached")


@pytest.mark.parametrize("answer,char", [("I choose: b). My reasoning: x", "b"), ("I choose: **a) 5,000 dollars**.", "a"),
                                         ("I choose: 2). My reasoning", "2"), ("I choose: 1) My reasoning", "1")])
def test_locate_label_and_pair(tok, answer, char):
    end = tok.convert_tokens_to_ids("</think>")
    ids = tok.encode("some thinking, I choose: a) maybe", add_special_tokens=False) + [end] + \
        tok.encode("\n\n" + answer, add_special_tokens=False)
    idx, got = re_.locate_label(tok, ids, end, thinking=True)
    assert got == char and char in tok.decode([ids[idx]])          # the label token, after the marker only
    other = {"a": "b", "b": "a", "1": "2", "2": "1"}[char]
    oid = re_.label_pair(tok, ids[idx], char, other)
    assert tok.decode([oid]).replace(other, char) == tok.decode([ids[idx]])


def test_locate_label_needs_closed_thinking(tok):
    ids = tok.encode("I choose: a) but still thinking", add_special_tokens=False)
    assert re_.locate_label(tok, ids, tok.convert_tokens_to_ids("</think>"), thinking=True)[0] is None


def test_shuffle_within_strata():
    bank = {f"x{i}": {"near_first": i % 2 == 0, "regime": ["none", "both"][i % 4 // 2],
                      "horizon_grid_value": f"{i} days"} for i in range(40)}
    rows = [{"uid": u} for u in bank]
    hz = re_.shuffled_horizons(rows, bank, seed=3)
    assert sorted(hz.values()) == sorted(b["horizon_grid_value"] for b in bank.values())
    for u, v in hz.items():
        src = f"x{int(v.split()[0])}"
        assert (bank[src]["near_first"], bank[src]["regime"]) == (bank[u]["near_first"], bank[u]["regime"])
    assert hz != {u: b["horizon_grid_value"] for u, b in bank.items()}


def _items(n_pairs=30):
    r = np.logspace(np.log10(1 / 2.8), np.log10(2.8), n_pairs)
    return [{"pair_id": i, "near_first": nf, "h_over_far_delay": float(r[i])} for i in range(n_pairs) for nf in (False, True)]


def test_meaning_letter_split():
    items = _items()
    y0 = np.array([4 * np.log10(it["h_over_far_delay"]) for it in items])
    farfirst = np.array([not it["near_first"] for it in items])
    # a pure letter push: +1 toward the first label = +1 toward far when far is first, -1 when near is first
    letter = np.where(farfirst, 1.0, -1.0)
    st = analyze.arm_stats(items, y0, {"y_plus": y0 + letter, "y_minus": y0 - letter}, 0.1)
    assert abs(st["meaning"]) < 1e-9 and abs(st["letter"] - 1) < 1e-9
    # a pure meaning push with some disruption (even part)
    st = analyze.arm_stats(items, y0, {"y_plus": y0 + 2 - 0.5, "y_minus": y0 - 2 - 0.5}, 0.1)
    assert abs(st["meaning"] - 2) < 1e-9 and abs(st["letter"]) < 1e-9 and abs(st["even_mean"] + 0.5) < 1e-9


def test_switch_point_shift_in_decades():
    items = _items(60)
    logr = np.log10([it["h_over_far_delay"] for it in items])
    s = 8.0
    mk = lambda loc: s * (logr - loc)
    st = analyze.arm_stats(items, mk(0.0), {"y_plus": mk(-0.1), "y_minus": mk(0.1)}, 0.1)
    assert abs(st["delta_log10_H"] - 0.1) < 1e-3      # +x acts as a 10^0.1 longer horizon
