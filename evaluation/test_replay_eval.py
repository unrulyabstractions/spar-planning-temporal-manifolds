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


def test_check_loops_detects_repeats_only():
    import check_loops
    block = list(range(64))
    assert check_loops.is_looping(list(range(5000, 5500)) + block * 4)       # stuck: one block over and over
    assert not check_loops.is_looping(list(range(2000)))                     # progressing: nothing repeats
    assert not check_loops.is_looping(block * 2)                             # too short to call


def test_batched_replay_matches_each_row_alone():
    """Right-padded batches read every row as a batch-1 pass would, with and without a steering edit (tiny random
    Qwen3 on CPU in float32, so any gap is padding leaking in, not bf16 arithmetic)."""
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steering"))
    from steer.edits import LinearEdit

    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    rep = object.__new__(re_.Replayer)
    from steer.hooks import steering
    rep.M, rep.layer, rep.batch_size, rep.steering = SimpleNamespace(model=model), 1, 3, steering
    lengths = [5, 9, 7, 12, 6]                                       # unequal, and 5 rows over batches of 3
    rep._cache = {f"u{i}": (list(range(1, n)), [n % 60 + 2]) for i, n in enumerate(lengths)}
    items = [{"uid": f"u{i}", "far_id": 3, "near_id": 4} for i in range(len(lengths))]
    edit = LinearEdit(np.eye(32)[0], typical_norm=5.0)
    for x in (0.0, 0.7):
        y, mass = rep.run(items, edit, x)
        for n, it in enumerate(items):
            y1, m1 = rep.run([it], edit, x)
            assert y[n] == pytest.approx(y1[0], abs=1e-5) and mass[n] == pytest.approx(m1[0], abs=1e-6)
    assert not np.allclose(rep.run(items, edit, 0.7)[0], rep.run(items, edit, 0.0)[0])   # the edit does act


def test_batched_replay_refuses_split_positions():
    rep = object.__new__(re_.Replayer)
    with pytest.raises(ValueError, match="positions='all' only"):
        rep.run([], None, 0.1, positions="generated")


def test_weights_in_float32_restores_the_exact_bf16_weights():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = transformers.Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with re_.weights_in(model, "float32"):
        assert model.dtype == torch.float32
    assert model.dtype == torch.bfloat16
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_health_uses_the_even_part_of_the_label_mass():
    # one-sided: +x lowers the mass as much as -x raises it (steering), so the even part is unchanged -> healthy
    even, ok = re_.label_mass_health(0.845, 1.000, 0.923, 0.95)
    assert abs(even - 0.9225) < 1e-12 and ok
    # both signs lose mass (disruption) -> unhealthy
    even, ok = re_.label_mass_health(0.85, 0.85, 0.923, 0.95)
    assert abs(even - 0.85) < 1e-12 and not ok


def test_gen_decodings_follow_the_baseline_run():
    pytest.importorskip("pbank.local_model")
    one = re_.gen_decodings("greedy", 5)                                   # greedy: one pass whatever is asked
    assert len(one) == 1 and one[0].mode == "greedy"
    assert re_.gen_decodings(None, 3)[0].mode == "greedy"                  # runs from before PR 19 record nothing
    run = {"mode": "sample", "temperature": 1.0, "top_p": 0.95, "top_k": 64, "seed": 7, "source": {}}
    ds = re_.gen_decodings(run, 3)
    assert [d.mode for d in ds] == ["sample"] * 3 and [d.seed for d in ds] == [7, 8, 9]
    assert all((d.temperature, d.top_p, d.top_k) == (1.0, 0.95, 64) for d in ds)
    with pytest.raises(ValueError):
        re_.gen_decodings({"mode": "beam"}, 1)


def test_run_settings_reads_decoding(tmp_path):
    import json
    base = {"model": "m", "thinking": True, "max_new_tokens": 4096, "batch_size": 8}
    (tmp_path / "run.json").write_text(json.dumps({"settings": base}))
    assert re_.run_settings(tmp_path)["decoding"] == "greedy"
    dec = {"mode": "sample", "temperature": 1.0, "top_p": 0.95, "top_k": 64, "seed": 0}
    (tmp_path / "run.json").write_text(json.dumps({"settings": base | {"decoding": dec}}))
    assert re_.run_settings(tmp_path)["decoding"] == dec
