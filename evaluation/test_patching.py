"""Horizon write spans and activation patching (patching.py, replay_eval's Replayer), on a tiny random Qwen3 with the
real Qwen3-8B tokenizer and chat template, so the horizon's tokens are found as on the box."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steering"))

import analyze
import patching
import replay_eval as re_

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

# an option states the same duration as the horizon: only the CONSTRAINT line's may change
PROMPT = ("SITUATION: Plan for the future of the household.\n"
          "TASK: You, the head of the household, are tasked to choose the best investment:\n"
          "a) 200,000 dollars in {h}.\n"
          "b) 100,000 dollars in 6 months.\n"
          "OBJECTIVE: Think deeply about which option is preferable.\n"
          "CONSTRAINT: You must select the option that provides the greatest benefit for this time horizon: {h}.\n"
          "ACTION: Select one of the two options. Provide reasoning.\n"
          "FORMAT: I choose: <a) or b)>. My reasoning: <1-3 sentences>")


@pytest.mark.parametrize("horizon, decades, want", [
    ("7 years", 1, "70 years"), ("7 years", -1, "0.7 years"), ("10.32 years", 1, "103.2 years"),
    ("10.32 years", -1, "1.032 years"), ("100 weeks", 1, "1000 weeks"), ("110 seconds", -1, "11 seconds"),
    ("0.88 years", -1, "0.088 years")])
def test_scaled_horizon(horizon, decades, want):
    assert patching.scaled_horizon(horizon, decades) == want


def test_with_horizon_changes_only_the_constraint_line():
    text = PROMPT.format(h="7 years")
    new = patching.with_horizon(text, "7 years", "70 years")
    assert "a) 200,000 dollars in 7 years." in new
    assert "time horizon: 70 years." in new and new.count("70 years") == 1


def test_span_indices():
    site = SimpleNamespace(token_indices=[10, 11, 12])
    assert patching.span_indices("horizon", site, 20) == [10, 11, 12]
    assert patching.span_indices("from_horizon", site, 20) == list(range(10, 20))
    assert patching.span_indices("horizon_last", site, 20) == [12]
    assert patching.span_indices("from_horizon_last", site, 20) == list(range(12, 20))


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen3-8B tokenizer not in the local cache")


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    return transformers.Qwen3ForCausalLM(cfg).eval()


def make_replayer(tok, model, horizons, write_span, reply_lens=(5, 9)):
    """A Replayer over prompts stating `horizons`, with made-up replies (token ids, no generation)."""
    from steer.hooks import steering

    def chat_text(text, thinking):
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=thinking)

    M = SimpleNamespace(model=model, tok=tok, lcm=SimpleNamespace(chat_text=chat_text))
    M.prompt_ids = lambda text, thinking: tok(chat_text(text, thinking), add_special_tokens=False)["input_ids"]
    rep = object.__new__(re_.Replayer)
    rep.M, rep.layer, rep.batch_size, rep.steering, rep.write_span = M, 1, 2, steering, write_span
    rep.thinking, rep._horizon = True, {}
    rep.bank = {f"u{i}": {"text": PROMPT.format(h=h), "horizon": h} for i, h in enumerate(horizons)}
    rep._cache = {u: (M.prompt_ids(b["text"], True), list(range(1000, 1000 + n)))
                  for (u, b), n in zip(rep.bank.items(), reply_lens)}
    items = [{"uid": u, "far_id": 3, "near_id": 4} for u in rep.bank]
    return rep, items


def block_outputs(model, run):
    """The residual stream after block 1 (what the hook at layer 1 writes), per forward pass inside run()."""
    from steer.layers import decoder_layers
    seen = []
    handle = decoder_layers(model)[1].register_forward_pre_hook(
        lambda m, args, kwargs: seen.append((args[0] if args else kwargs["hidden_states"]).clone()), with_kwargs=True)
    try:
        y, _ = run()
    finally:
        handle.remove()
    return seen, y


def rows_of_passes(rep, items):
    order = sorted(range(len(items)), key=lambda i: sum(map(len, rep.seqs(items[i]))))
    return [order[s: s + rep.batch_size] for s in range(0, len(order), rep.batch_size)]


@pytest.mark.parametrize("span", ["horizon", "from_horizon"])
def test_horizon_spans_edit_only_their_tokens(tok, model, span):
    from steer.edits import LinearEdit
    rep, items = make_replayer(tok, model, ["7 years", "110 seconds"], span)
    edit = LinearEdit(np.eye(32)[0], typical_norm=5.0)
    h0, _ = block_outputs(model, lambda: rep.run(items, edit, 0.0))
    h1, _ = block_outputs(model, lambda: rep.run(items, edit, 0.7))
    for rows, a, b in zip(rows_of_passes(rep, items), h0, h1):
        changed = (b - a).abs().amax(-1) > 0
        for r, i in enumerate(rows):
            site = rep.horizon_site(items[i])
            assert tok.decode([rep.seqs(items[i])[0][k] for k in site.token_indices]).strip() == \
                rep.bank[items[i]["uid"]]["horizon"]
            want = torch.zeros(changed.shape[1], dtype=torch.bool)
            want[patching.span_indices(span, site, len(rep.seqs(items[i])[0]))] = True
            assert torch.equal(changed[r], want), (span, i)


@pytest.mark.parametrize("span", ["horizon", "from_horizon"])
def test_full_patch_writes_the_source_states_at_aligned_tokens(tok, model, span):
    rep, items = make_replayer(tok, model, ["7 years", "110 seconds"], span)
    edit = patching.PatchEdit()
    h0, _ = block_outputs(model, lambda: rep.run(items, edit, 0.0))
    h1, _ = block_outputs(model, lambda: rep.run(items, edit, 1.0))
    patch_span = patching.PATCH_SPANS[span]
    src = rep.source_states(items, 1, patch_span, 1)
    passes = rows_of_passes(rep, items)
    # the first run also captured the source prompts' passes (PatchEdit.prepare): keep the replay passes, which are last
    for rows, a, b in zip(passes, h0[-len(passes):], h1[-len(passes):]):
        for r, i in enumerate(rows):
            pos = patching.span_indices(patch_span, rep.horizon_site(items[i]), len(rep.seqs(items[i])[0]))
            assert torch.allclose(b[r, pos].float(), src[i], atol=1e-5)
            other = torch.ones(b.shape[1], dtype=torch.bool)
            other[pos] = False
            assert torch.equal(b[r, other], a[r, other])


def test_patch_from_the_same_prompt_changes_nothing(tok, model):
    """decades = 0: the source is the target's own prompt, so the patch must give the unsteered readout back."""
    rep, items = make_replayer(tok, model, ["7 years", "15 minutes"], "from_horizon")
    y0, _ = rep.unsteered(items)
    y, _ = rep.run(items, patching.PatchEdit(decades=0), 1.0)
    assert np.allclose(y, y0, atol=1e-4)


def test_subspace_patch_moves_only_the_subspace():
    d, U = 6, patching.orthonormal_rows(np.random.default_rng(0).normal(size=(2, 6)))
    edit = patching.PatchEdit(U)
    h = torch.randn(1, 5, d)
    src = torch.randn(2, d)
    edit.sources = {("u", 1): src, ("u", -1): -src}
    mask = torch.zeros(1, 5, dtype=torch.bool)
    mask[0, [1, 3]] = True
    ctx = SimpleNamespace(mask=mask, rows=[{"uid": "u"}])
    delta = edit.delta(h, 1.0, ctx)
    Ut = torch.as_tensor(U, dtype=torch.float32)
    new = h + delta
    assert torch.allclose(new[0, [1, 3]] @ Ut.T, src @ Ut.T, atol=1e-5)          # the subspace part is the source's
    assert torch.allclose(delta - (delta @ Ut.T) @ Ut, torch.zeros_like(delta), atol=1e-5)   # nothing outside it
    assert torch.equal(delta[0, [0, 2, 4]], torch.zeros(3, d))
    assert torch.equal(edit.delta(h, 0.0, ctx), torch.zeros_like(h))
    with pytest.raises(ValueError):
        edit.delta(h, 0.5, ctx)


def test_text_edit_rewrites_the_horizon_and_keeps_the_reasoning(tok, model):
    rep, items = make_replayer(tok, model, ["7 years", "110 seconds"], "all")
    y, _ = rep.run(items, patching.TextEdit(), -1.0)
    seqs = [rep.M.prompt_ids(patching.with_horizon(rep.bank[it["uid"]]["text"], rep.bank[it["uid"]]["horizon"],
                                                   patching.scaled_horizon(rep.bank[it["uid"]]["horizon"], -1)), True)
            + rep.seqs(it)[1] for it in items]
    with torch.no_grad():
        lg = rep._last_logits(seqs)
    assert np.array_equal(y, re_.Replayer._readout(items, lg)[0])
    assert "11 seconds" in tok.decode(seqs[1])


def test_patch_needs_a_horizon_span(tok, model):
    rep, items = make_replayer(tok, model, ["7 years", "110 seconds"], "all")
    with pytest.raises(ValueError, match="horizon"):
        rep.run(items, patching.PatchEdit(), 1.0)


def test_patch_arms_are_ranked_only_against_their_own_null_bases():
    def arm(name, family, kind="push", k=None, meaning=1.0):
        return {"arm": name, "family": family, "kind": kind, "k": k, "meaning": meaning, "delta_log10_H": 0.1}
    res = {a["arm"]: a for a in [
        arm("trend", "method"), arm("covariance[0]", "covariance", meaning=0.5),
        arm("patch:full", "patch", "patch"), arm("patch:trend", "patch", "patch", k=1),
        arm("patch:curve-displace", "patch", "patch", k=4),
        arm("patch-null:k1[0]", "patch-null:k1", "patch", k=1, meaning=0.2),
        arm("patch-null:k4[0]", "patch-null:k4", "patch", k=4, meaning=0.3)]}
    comps = analyze.compare_to_nulls(res)
    assert set(comps["trend"]) == {"covariance"}
    assert set(comps["patch:trend"]) == {"patch-null:k1"}
    assert set(comps["patch:curve-displace"]) == {"patch-null:k4"}
    assert comps["patch:full"] == {}
