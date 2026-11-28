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


def test_match_strength_finds_the_matching_scale():
    f = lambda x: 0.3 * x ** 2                                             # damage grows with strength
    x, fx, ok, steps = re_.match_strength(f, target=1.2, x0=0.01)
    assert ok and abs(fx / 1.2 - 1) <= re_.N6_TOLERANCE and abs(x - 2.0) < 0.15 and steps < 30
    x, fx, ok, _ = re_.match_strength(f, target=1e-6, x0=5.0)             # from above, by halving
    assert ok and abs(fx / 1e-6 - 1) <= re_.N6_TOLERANCE
    assert not re_.match_strength(lambda x: 0.0, target=1.0, x0=1.0)[2]   # no damage at any strength: not matched


def test_method_dirs_site_filter_and_duplicates(tmp_path):
    import json
    for name, site in (("caa", "label"), ("caa", "prelabel"), ("reward", "label")):
        d = tmp_path / f"{name}__{site}"
        d.mkdir()
        (d / "method.json").write_text(json.dumps({"method": name, "site": site}))
    assert set(re_.method_dirs(tmp_path, "label")) == {"caa", "reward"}
    assert re_.method_dirs(tmp_path, "prelabel")["caa"].name == "caa__prelabel"
    with pytest.raises(SystemExit):
        re_.method_dirs(tmp_path)                                          # caa twice, no site given


def sweep_rows(odds, healthy=None):
    xs = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    healthy = healthy or [True] * len(odds)
    return [{"x": x, "mean_odd_logit": o, "healthy": h} for x, o, h in zip(xs, odds, healthy)]


def test_choose_strength_target_rule_is_unchanged():
    from replay_eval import choose_strength
    assert choose_strength(sweep_rows([0.1, 0.5, 1.2, 2.0, 3.0, 4.0]), "target", 1.0) == (
        0.1, "smallest healthy x with |mean odd| >= target")
    # Qwen3-8B 2026-11-25: the reversed effect at x = 0.8 (|-1.16| >= 1) counts as reaching the target.
    qwen = sweep_rows([0.084, 0.183, 0.403, 0.652, 0.582, -1.160])
    assert choose_strength(qwen, "target", 1.0) == (0.8, "smallest healthy x with |mean odd| >= target")
    assert choose_strength(sweep_rows([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), "target", 1.0) == (
        0.8, "target not reached while healthy: largest healthy x")
    assert choose_strength(sweep_rows([0.1] * 6, [False] * 6), "target", 1.0)[0] == 0.02


def test_choose_strength_peak_rule_stops_at_the_first_sign_change():
    from replay_eval import choose_strength
    qwen = sweep_rows([0.084, 0.183, 0.403, 0.652, 0.582, -1.160])
    assert choose_strength(qwen, "peak", 1.0) == (
        0.2, "peak: largest |mean odd| among healthy x before the effect changes sign")
    # A method that steers backwards from the start is kept (sign is a result); unhealthy x are never chosen.
    assert choose_strength(sweep_rows([-0.1, -0.3, -0.6, -0.4, -2.0, 0.1], [True] * 4 + [False, True]), "peak", 1.0)[0] == 0.1
    assert choose_strength(sweep_rows([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), "peak", 1.0)[0] == 0.8


def test_prompt_write_span_edits_each_rows_prompt_tokens_only():
    """Write span "prompt" in a batched replay: per row, every prompt state is edited and no reply or padding state
    is, with rows of different prompt lengths in one pass; each row reads as hooks.replay_logits(positions="prompt")
    alone. Span "all" is what the replay did before the span existed: one steering block over every pass."""
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steering"))
    from steer.edits import LinearEdit
    from steer.hooks import replay_logits, steering
    from steer.layers import decoder_layers

    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    rep = object.__new__(re_.Replayer)
    rep.M, rep.layer, rep.batch_size, rep.steering = SimpleNamespace(model=model), 1, 2, steering
    # (prompt, reply) lengths; by total length the rows run 2, 0, 1, which is not their prompt-length order
    shapes = [(6, 2), (3, 7), (4, 1)]
    rep._cache = {f"u{i}": (list(range(1, 1 + p)), list(range(20, 20 + a))) for i, (p, a) in enumerate(shapes)}
    items = [{"uid": f"u{i}", "far_id": 3, "near_id": 4} for i in range(len(shapes))]
    edit = LinearEdit(np.eye(32)[0], typical_norm=5.0)
    order = sorted(range(len(shapes)), key=lambda i: sum(shapes[i]))
    passes = [order[s: s + rep.batch_size] for s in range(0, len(order), rep.batch_size)]

    def states_after_block_1(x):
        """Per pass, the residual stream entering block 2 (block 1's output, after the edit)."""
        seen = []
        handle = decoder_layers(model)[1].register_forward_pre_hook(
            lambda m, args, kwargs: seen.append((args[0] if args else kwargs["hidden_states"]).clone()),
            with_kwargs=True)
        try:
            y, _ = rep.run(items, edit, x)
        finally:
            handle.remove()
        return seen, y

    # the horizon spans need a bank and a tokenizer: test_patching.py
    for span in ("all", "prompt"):
        rep.write_span = span
        unedited, _ = states_after_block_1(0.0)
        edited, y = states_after_block_1(0.7)
        assert len(edited) == len(passes)
        for rows, h0, h1 in zip(passes, unedited, edited):
            changed = (h1 - h0).abs().amax(-1) > 0                          # [rows, tokens]
            for r, i in enumerate(rows):
                p, a = shapes[i]
                want = torch.zeros(changed.shape[1], dtype=torch.bool)
                want[: p if span == "prompt" else p + a] = True              # never the padding after p + a
                assert torch.equal(changed[r], want), (span, i)
        if span == "prompt":
            for n, it in enumerate(items):
                p_ids, a_ids = rep._cache[it["uid"]]
                lg = replay_logits(model, p_ids, a_ids, edit, 0.7, 1, positions="prompt")
                assert y[n] == pytest.approx(float(lg[3] - lg[4]), abs=1e-5)
        else:
            with torch.no_grad(), steering(model, edit, 0.7, 1, "all"):
                lg = rep._last_logits([p + a for p, a in map(rep.seqs, items)])
            assert np.array_equal(y, re_.Replayer._readout(items, lg)[0])


def write_bundle(d, method, kind, arrays, layer=3, site="prompt_end", typical_norm=2.0):
    """A bundle that steer.methods.load accepts (format and file hashes), with the fields the runner reads."""
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steering"))
    from steer.activations import file_sha256
    d.mkdir(parents=True)
    np.savez(d / "arrays.npz", **arrays)
    (d / "fit.json").write_text(json.dumps({"diagnostics": {"typical_norm": typical_norm}}))
    (d / "fit_uids.jsonl").write_text("")
    (d / "search.jsonl").write_text("")
    request = {"typical_norm": typical_norm} if kind == "push" else {"name": "shift"}
    meta = {"format": "steer-method/1", "method_id": d.name, "method": method, "kind": kind, "layer": layer,
            "site": site, "default_positions": "all", "request": request,
            "files": {n: file_sha256(d / n) for n in ("arrays.npz", "fit.json", "fit_uids.jsonl", "search.jsonl")}}
    (d / "method.json").write_text(json.dumps(meta))
    return d


class AddReplayer:
    """y = 1 + x × (an effect per edit object), label mass 0.9; plain() reads the unsteered y."""

    batch_size = 4

    def __init__(self):
        self.calls = []

    def run(self, items, edit, x, positions="all"):
        self.calls.append((edit, x))
        return np.full(len(items), 1.0 + x * 0.5, np.float32), np.full(len(items), 0.9, np.float32)

    def plain(self, it):
        torch = pytest.importorskip("torch")
        lg = torch.zeros(8)
        lg[it["far_id"]] = 1.0
        return lg


@pytest.mark.parametrize("only", ["trend", "curve-displace"])
def test_methods_dir_with_only_a_non_caa_bundle_replays_from_a_fixed_calibration(tmp_path, only):
    """The confirm stage: a methods directory with the screen's winner alone and screen.py's calibration.json."""
    import json
    from types import SimpleNamespace
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompt_bank"))   # pbank, when not installed
    d = 6
    methods = tmp_path / "methods"
    if only == "trend":
        write_bundle(methods / "trend__L3_prompt_end", "trend", "push", {"direction": np.eye(d)[0], "typical_norm": 2.0})
    else:
        t = np.linspace(0, 3, 5)
        write_bundle(methods / "curve-displace__L3_prompt_end", "curve-displace", "target",
                     {"mu": np.zeros(d), "U": np.eye(d)[:, :2], "t_grid": t, "curve": np.stack([t, t ** 2], 1),
                      "t_means": t, "z_means": np.stack([t, t ** 2], 1), "counts": np.ones(5), "bandwidth": 0.5,
                      "explained": np.array([0.9, 0.1])})
    draws = tmp_path / "draws"
    draws.mkdir()
    rng = np.random.default_rng(0)
    np.savez(draws / "draws.npz", **{f: rng.normal(size=(3, d)) for f in ("isotropic", "covariance", "real_diff")})
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "prompts.jsonl").write_text("")
    out = tmp_path / "out"
    out.mkdir()
    re_.write_jsonl_atomic(out / "items.jsonl", [{"uid": f"v{i}", "split": "verification", "far_id": 1, "near_id": 2,
                                                  "chosen": "far"} for i in range(3)])
    cal = {"x_push": 0.4 if only == "trend" else None, "x_shift": 1.0 if only == "trend" else 2.0, "fold": None,
           "rule": "fixed by screen.py select", "write_span": "prompt"}
    (out / "calibration.json").write_text(json.dumps(cal))
    args = SimpleNamespace(bank=str(bank), activations="unused", methods=str(methods), draws=str(draws), k=2,
                           n0_check=2, out=str(out), write_span="prompt", site=None,
                           arms=[only, "covariance"] if only == "trend" else [only])

    assert re_.stage_calibrate(args, None, None, None) == cal       # returns before looking for a caa bundle
    assert re_.methods_layer(re_.method_dirs(methods)) == 3
    arms = {a["arm"]: a for a in re_.build_arms(args, cal)}
    assert arms[only]["family"] == "method" and arms[only]["kind"] == ("push" if only == "trend" else "target")
    rep = AddReplayer()
    re_.stage_replay(args, None, None, rep, cal, keep_arm=lambda a: re_.arm_selected(a, args.arms))
    x = cal["x_push"] if only == "trend" else cal["x_shift"]
    replayed = {xx for _, xx in rep.calls}
    assert {x, -x} <= replayed
    if only == "trend":                                               # the push nulls at the winner's strength
        assert {"covariance[0]", "covariance[1]"} <= set(arms)
        assert sum(1 for _, xx in rep.calls if xx == 0.4) == 1 + 2   # trend and two covariance draws
    with pytest.raises(SystemExit, match="write span"):
        re_.stage_calibrate(SimpleNamespace(**{**vars(args), "write_span": "all"}), None, None, None)


def test_trend_and_probe_are_ranked_methods():
    from types import SimpleNamespace
    import json
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("caa", "trend", "probe", "reward"):
            (Path(tmp) / name).mkdir()
            (Path(tmp) / name / "method.json").write_text(json.dumps({"method": name, "site": "label"}))
        args = SimpleNamespace(methods=tmp, k=1, out=tmp)
        arms = {a["arm"]: a for a in re_.build_arms(args, {"x_push": 0.2, "x_shift": 1.0})}
    assert all(arms[m]["family"] == "method" and arms[m]["kind"] == "push" and arms[m]["x"] == 0.2
               for m in ("caa", "trend", "probe"))
    assert arms["reward"]["family"] == "control"
    assert re_.arm_selected(arms["trend"], ["trend", "covariance"]) and re_.arm_selected(arms["covariance[0]"], ["trend", "covariance"])
