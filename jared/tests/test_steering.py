"""E8 steering machinery: hooks and scoring on a tiny random Qwen3 (no download),
direction maths on planted data, crossing pairs with the real 0.8B tokenizer."""

import random

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from spar_horizon import steering as S
from spar_horizon.config import Settings
from spar_horizon.full_bank import build_full_prompts, regime

LABELS = (1, 2)
ROWS = [LABELS] * 3
PAD = 0


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=64, head_dim=8)
    return Qwen3ForCausalLM(cfg).eval()


SEQS = [[3, 5, 7, 9, 11, 13], [4, 6, 8, 10], [7, 7, 9, 12, 15]]


def test_batched_left_padded_scoring_equals_single_rows(tiny):
    near_first = [True, False, True]
    batched = S.score(tiny, "cpu", SEQS, near_first, ROWS, PAD, batch_size=3)
    single = [S.score(tiny, "cpu", [s], [n], [LABELS], PAD, batch_size=1)[0] for s, n in zip(SEQS, near_first)]
    assert np.allclose(batched, single, atol=1e-4)
    flipped = S.score(tiny, "cpu", SEQS, [not n for n in near_first], ROWS, PAD, batch_size=3)
    assert np.allclose(flipped, -batched, atol=1e-6)


def test_zero_edit_changes_nothing(tiny):
    base = S.score(tiny, "cpu", SEQS, [True] * 3, ROWS, PAD, batch_size=3)
    zero = S.score(tiny, "cpu", SEQS, [True] * 3, ROWS, PAD, batch_size=3, layer=2,
                   positions=[[len(s) - 1] for s in SEQS], make_edit=lambda rows: S.Add(torch.zeros(len(rows), 32)))
    assert np.allclose(base, zero, atol=1e-6)


def test_edit_acts_from_its_position_on_and_not_before(tiny):
    # transformers records hidden_states[layer] before forward hooks run, so the edit
    # is checked where it must show: the next layer, from the edited position on
    seq, layer, pos = SEQS[0], 2, 3
    _, clean, _ = S.forward(tiny, "cpu", [seq], PAD, hidden=True)
    delta = torch.zeros(1, 32)
    delta[0, 5] = 1.0
    _, edited, _ = S.forward(tiny, "cpu", [seq], PAD, hidden=True, layer=layer, positions=[[pos]], edit=S.Add(delta))
    after = edited[layer + 1][0] - clean[layer + 1][0]
    assert torch.allclose(after[:pos], torch.zeros_like(after[:pos]), atol=1e-6)   # causal: earlier positions untouched
    assert after[pos:].abs().max() > 1e-3
    assert torch.allclose(edited[layer - 1], clean[layer - 1])                       # earlier layers untouched


def test_replacing_every_position_reproduces_the_donor(tiny):
    recipient, donor = [3, 5, 7, 9, 11], [3, 5, 8, 9, 11]
    donor_logits, donor_hidden, _ = S.forward(tiny, "cpu", [donor], PAD, hidden=True)
    layer, positions = 1, list(range(len(donor)))
    values = S.gather(donor_hidden[layer], [positions], [0])
    patched, _, _ = S.forward(tiny, "cpu", [recipient], PAD, layer=layer, positions=[positions], edit=S.Replace(values))
    assert torch.allclose(patched, donor_logits, atol=1e-5)


@pytest.mark.parametrize("layer", [0, 3])
def test_embeddings_and_post_norm_indices_are_rejected(tiny, layer):
    # index 3 of a 3-layer model is recorded after the final norm: patching it would
    # apply the norm twice (verified: it does not reproduce the donor's logits)
    with pytest.raises(ValueError):
        S.forward(tiny, "cpu", [SEQS[0]], PAD, layer=layer, positions=[[1]], edit=S.Add(torch.zeros(1, 32)))


def test_mean_ablate_removes_the_subspace_component():
    rng = np.random.default_rng(0)
    basis = torch.tensor(S.random_orthonormal(16, 3, seed=1), dtype=torch.float32)
    mean = torch.tensor(rng.normal(size=16), dtype=torch.float32)
    x = torch.tensor(rng.normal(size=(5, 16)), dtype=torch.float32)
    out = S.MeanAblate(mean, basis)(x, None, None)
    assert torch.allclose((out - mean) @ basis.T, torch.zeros(5, 3), atol=1e-5)
    assert torch.allclose(S.MeanAblate(mean, basis)(out, None, None), out, atol=1e-5)   # idempotent
    orth = x - ((x - mean) @ basis.T) @ basis
    assert torch.allclose(out, orth)


def planted(n_per=8, horizons=12, d=40, seed=0):
    rng = np.random.default_rng(seed)
    log_h = np.repeat(np.linspace(-5, 2.5, horizons), n_per)
    direction = rng.normal(size=d)
    return np.outer(log_h, direction) + rng.normal(scale=0.05, size=(len(log_h), d)), log_h, direction


def test_fit_cell_recovers_a_planted_direction_and_curve():
    X, log_h, direction = planted()
    cell = S.fit_cell(X, log_h)
    readout = (X - cell["mean"]) @ cell["w"] + cell["b"]
    assert np.corrcoef(readout, log_h)[0, 1] > 0.99
    assert abs(np.dot(cell["basis"][0], direction) / np.linalg.norm(direction)) > 0.99
    assert cell["basis"].shape[0] == 8 and cell["curve_coords"].shape == (12, 8)
    assert np.dot(cell["caa"], direction) > 0


def test_matched_deltas_have_the_curve_length_and_the_given_direction():
    X, log_h, _ = planted()
    cell = S.fit_cell(X, log_h)
    vector = np.random.default_rng(3).normal(size=X.shape[1])
    for alpha in (-2, -1, 1, 2):
        curve = S.curve_delta(cell, log_h, alpha)
        moved = S.matched_deltas(cell, vector, alpha, log_h)
        assert np.allclose(np.linalg.norm(moved, axis=1), np.linalg.norm(curve, axis=1))
        nonzero = np.linalg.norm(moved, axis=1) > 0
        cosine = moved[nonzero] @ vector / (np.linalg.norm(moved[nonzero], axis=1) * np.linalg.norm(vector))
        assert np.allclose(cosine, np.sign(alpha))


def test_curve_delta_is_zero_at_zero_exact_on_grid_and_clamped():
    X, log_h, _ = planted()
    cell = S.fit_cell(X, log_h)
    grid = cell["curve_logh"]
    assert np.allclose(S.curve_delta(cell, grid[:3], 0.0), 0)
    step = grid[1] - grid[0]
    expected = (cell["curve_coords"][1] - cell["curve_coords"][0]) @ cell["basis"]
    assert np.allclose(S.curve_delta(cell, grid[:1], step)[0], expected)
    far = S.curve_delta(cell, grid[-1:], 5.0)                 # beyond the last horizon: clamped, no motion
    assert np.allclose(far, 0)


def test_bootstrap_and_contrast():
    mean, lo, hi = S.bootstrap_mean([1.0, 2.0, 3.0, np.nan])
    assert mean == 2.0 and lo <= 2.0 <= hi
    assert S.contrast(np.array([5.0, 3.0, -1.0]), ["near", "near", "both"]) == 5.0


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")


def test_crossing_pairs_are_equal_length_horizon_only_and_cross_regimes(tok):
    cfg = Settings(model="Qwen/Qwen3.5-0.8B", thinking="off", prefill="I choose:", bank="full")
    records = build_full_prompts(n=120)
    pairs = S.crossing_pairs(records, tok, cfg, random.Random(0), limit=20)
    assert len(pairs) >= 10
    for p in pairs:
        rec = records[p["record"]]
        assert len(p["recipient"]) == len(p["donor"])
        start, end = p["span"]
        assert p["recipient"][:start] == p["donor"][:start] and p["recipient"][end:] == p["donor"][end:]
        assert p["donor_horizon"] in tok.decode(p["donor"][start - 2:end + 2]) or start < end
        here = regime(rec["horizon"], rec["near_delay_years"], rec["far_delay_years"])
        assert here in ("near", "both")
        assert tok.decode(p["recipient"][p["prompt_end"]:]).startswith("<|im_end|>")
        groups = S.patch_groups(p, tok)
        assert groups[0][0] == "horizon" and groups[-1][1] == [len(p["recipient"]) - 1]


def test_horizon_span_covers_the_whole_phrase_even_when_the_unit_is_shared(tok):
    cfg = Settings(model="Qwen/Qwen3.5-0.8B", thinking="off", prefill="I choose:", bank="full")
    records = build_full_prompts(n=120)
    pairs = S.crossing_pairs(records, tok, cfg, random.Random(0), limit=40)
    shared_unit = 0
    for p in pairs:
        start, end = p["span"]
        recipient_horizon = records[p["record"]]["horizon_text"]
        assert recipient_horizon in tok.decode(p["recipient"][start:end])
        assert p["donor_horizon"] in tok.decode(p["donor"][start:end])
        groups = dict(S.patch_groups(p, tok))
        assert groups["horizon"] == list(range(start, end))
        shared_unit += recipient_horizon.split()[-1] == p["donor_horizon"].split()[-1]
    assert shared_unit > 0                               # "91 years" -> "16 years" shares " years"


def test_per_row_labels_are_used(tiny):
    a = S.score(tiny, "cpu", SEQS[:2], [True, True], [(1, 2), (3, 4)], PAD, batch_size=2)
    b = [S.score(tiny, "cpu", [SEQS[0]], [True], [(1, 2)], PAD, 1)[0], S.score(tiny, "cpu", [SEQS[1]], [True], [(3, 4)], PAD, 1)[0]]
    assert np.allclose(a, b, atol=1e-4)


def test_toward_rational_and_removed_fraction():
    effect = np.array([-0.4, -0.2, 0.3, 0.5])
    regimes = ["near", "near", "both", "both"]
    assert np.allclose(S.toward_rational(effect, regimes, 1)[:2], [0.4, 0.2])
    assert np.isnan(S.toward_rational(effect, regimes, 1)[2:]).all()
    assert np.allclose(S.toward_rational(effect, regimes, -1)[2:], [0.3, 0.5])
    base = np.array([10.0, 12.0, 0.0, 2.0])          # contrast 10
    edited = np.array([6.0, 8.0, 1.0, 3.0])          # contrast 5
    mean, lo, hi = S.removed_fraction(base, edited, regimes)
    assert np.isclose(mean, 0.5) and lo <= 0.5 <= hi


def test_contrast_and_removed_fraction_need_both_regimes():
    lo = np.array([1.0, 2.0])
    assert np.isnan(S.contrast(lo, ["both", "both"]))
    assert all(np.isnan(S.removed_fraction(lo, lo, ["both", "both"])))


def test_capture_last_matches_recorded_hidden_states(tiny):
    logits, captured = S.forward_capture_last(tiny, "cpu", SEQS, PAD)
    ref_logits, hidden, _ = S.forward(tiny, "cpu", SEQS, PAD, hidden=True)
    assert torch.allclose(logits, ref_logits, atol=1e-5)
    assert len(captured) == 2                                  # editable indices 1 and 2 of a 3-layer model
    for layer in (1, 2):
        assert torch.allclose(captured[layer - 1], hidden[layer][:, -1].float(), atol=1e-5)


def test_answer_slot_patch_with_captured_values_moves_the_recipient(tiny):
    # the captured donor state, written into the recipient's last position, must change its logits
    recipient, donor = [SEQS[0]], [SEQS[2]]
    _, captured = S.forward_capture_last(tiny, "cpu", donor, PAD)
    base, _, _ = S.forward(tiny, "cpu", recipient, PAD)
    patched, _, _ = S.forward(tiny, "cpu", recipient, PAD, layer=2, positions=[[len(SEQS[0]) - 1]],
                              edit=S.Replace(captured[1]))
    assert not torch.allclose(patched, base)


def test_answer_pairs_cross_regimes_and_keep_the_near_label():
    regimes = ["near", "both", "near", "both", "near"]
    near_labels = ["a", "a", "b", "b", "a"]
    pairs = S.answer_pairs(regimes, near_labels, random.Random(0), limit=10)
    assert len(pairs) == 5
    for r, d in pairs:
        assert regimes[r] != regimes[d] and near_labels[r] == near_labels[d]


def test_decision_direction_recovers_a_planted_decision_inside_the_basis():
    rng = np.random.default_rng(0)
    basis = S.random_orthonormal(40, 4, seed=3)
    coords = rng.normal(size=(300, 4))
    X = coords @ basis + rng.normal(scale=0.01, size=(300, 40))
    near_lo = 3 * coords[:, 1] - coords[:, 2]
    cell = {"mean": np.zeros(40), "basis": basis}
    u, r2 = S.decision_direction(X, near_lo, cell)
    planted = (3 * basis[1] - basis[2]) / np.linalg.norm(3 * basis[1] - basis[2])
    assert r2 > 0.99 and u @ planted > 0.999          # right direction and sign (raises the near log-odds)
    assert np.allclose(np.linalg.norm(u), 1)


def test_curve_splits_into_decision_part_and_the_rest():
    rng = np.random.default_rng(1)
    u = rng.normal(size=12)
    u /= np.linalg.norm(u)
    deltas = rng.normal(size=(5, 12))
    rest = S.without_direction(deltas, u)
    assert np.allclose(rest @ u, 0)
    assert np.allclose(rest + np.outer(deltas @ u, u), deltas)


def test_basis_without_drops_one_dimension_orthogonal_to_the_decision():
    basis = S.random_orthonormal(30, 4, seed=2)
    u = np.random.default_rng(3).normal(size=30)
    u /= np.linalg.norm(u)
    reduced = S.basis_without(basis, u)
    assert reduced.shape == (3, 30)
    assert np.allclose(reduced @ reduced.T, np.eye(3), atol=1e-10)    # orthonormal
    assert np.allclose(reduced @ (basis.T @ (basis @ u)), 0, atol=1e-10)   # orthogonal to u's part in the span
    assert np.allclose(reduced @ basis.T @ basis, reduced, atol=1e-10)   # still inside the span


# --- E10: prompt_patch during generation ----------------------------------------

def hidden_at(model, seq, layer):
    """hidden_states[layer] of one unpadded sequence, [T, d]."""
    _, hidden, _ = S.forward(model, "cpu", [seq], PAD, hidden=True)
    return hidden[layer][0]


def test_prompt_patch_with_own_states_changes_nothing(tiny):
    layer, positions = 2, [[1, 2], [0], [3]]
    own = [hidden_at(tiny, s, layer)[p] for s, p in zip(SEQS, positions)]
    base, _, _ = S.forward(tiny, "cpu", SEQS, PAD)
    with S.prompt_patch(tiny, layer, positions, own):
        patched, _, _ = S.forward(tiny, "cpu", SEQS, PAD)
    assert torch.allclose(base, patched, atol=1e-5)


def test_prompt_patch_of_every_position_reproduces_the_donor_in_a_padded_batch(tiny):
    # rows of different lengths, so the left padding differs per row
    layer = 2
    donors = [[s[0]] + [(t + 1) % 64 for t in s[1:]] for s in SEQS]
    values = [hidden_at(tiny, d, layer) for d in donors]
    with S.prompt_patch(tiny, layer, [list(range(len(s))) for s in SEQS], values):
        patched, _, _ = S.forward(tiny, "cpu", SEQS, PAD)
    donor_logits, _, _ = S.forward(tiny, "cpu", donors, PAD)
    assert torch.allclose(patched, donor_logits, atol=1e-4)


def test_prompt_patch_applies_to_the_prefill_and_not_to_decode_steps(tiny):
    layer, seq = 2, SEQS[0]
    donor = [seq[0]] + [(t + 1) % 64 for t in seq[1:]]
    values = [hidden_at(tiny, donor, layer)]
    calls = []
    with S.prompt_patch(tiny, layer, [list(range(len(seq)))], values):
        handle = tiny.model.layers[layer - 1].register_forward_hook(lambda m, i, o: calls.append(
            (o[0] if isinstance(o, tuple) else o).shape[1]))
        out = tiny.generate(input_ids=torch.tensor([seq]), attention_mask=torch.ones(1, len(seq), dtype=torch.long),
                            max_new_tokens=3, do_sample=False, pad_token_id=PAD)
        handle.remove()
    donor_logits, _, _ = S.forward(tiny, "cpu", [donor], PAD)
    assert calls == [len(seq), 1, 1]
    assert out[0, len(seq)].item() == donor_logits[0].argmax().item()   # first token read from the patched prefill


def test_prompt_patch_of_one_row_needs_no_mask_and_a_batch_does(tiny):
    layer, seq = 2, SEQS[0]
    donor = [seq[0]] + [(t + 1) % 64 for t in seq[1:]]
    values = [hidden_at(tiny, donor, layer)]
    with S.prompt_patch(tiny, layer, [list(range(len(seq)))], values):
        patched = tiny(input_ids=torch.tensor([seq])).logits[0, -1]   # no attention mask, as on the box
    donor_logits, _, _ = S.forward(tiny, "cpu", [donor], PAD)
    assert torch.allclose(patched, donor_logits[0], atol=1e-4)
    with pytest.raises(ValueError, match="2-D attention mask"):
        with S.prompt_patch(tiny, layer, [[0], [0]], values * 2):
            tiny(input_ids=torch.tensor([seq, seq]))


def test_subspace_patch_swaps_only_the_span_of_the_basis(tiny):
    layer, seq = 2, SEQS[0]
    donor = [seq[0]] + [(t + 1) % 64 for t in seq[1:]]
    values = [hidden_at(tiny, donor, layer)]
    everything = list(range(len(seq)))
    with S.prompt_patch(tiny, layer, [everything], values, basis=torch.eye(32)):
        full, _, _ = S.forward(tiny, "cpu", [seq], PAD)
    donor_logits, _, _ = S.forward(tiny, "cpu", [donor], PAD)
    assert torch.allclose(full, donor_logits, atol=1e-4)
    # a basis orthogonal to every donor-recipient difference swaps nothing
    diff = (values[0] - hidden_at(tiny, seq, layer)).numpy()
    _, _, vt = np.linalg.svd(diff)
    orthogonal = torch.tensor(vt[len(seq):len(seq) + 4], dtype=torch.float32)
    base, _, _ = S.forward(tiny, "cpu", [seq], PAD)
    with S.prompt_patch(tiny, layer, [everything], values, basis=orthogonal):
        none, _, _ = S.forward(tiny, "cpu", [seq], PAD)
    assert torch.allclose(none, base, atol=1e-4)


def test_phrase_positions_cover_exactly_the_horizon_text(tok):
    cfg = Settings(model="Qwen/Qwen3.5-0.8B", thinking="on", prefill="", choice_prefix="I choose:", bank="full")
    from spar_horizon.model_io import chat_text
    for rec in [r for r in build_full_prompts(n=60) if r["horizon"] is not None][:20]:
        text = chat_text(tok, rec["prompt"], cfg)
        ids = tok.encode(text, add_special_tokens=False)
        positions = S.phrase_positions(tok, text, rec["horizon_text"])
        assert positions == list(range(positions[0], positions[-1] + 1))
        assert tok.decode(ids[positions[0]:positions[-1] + 1]).strip() == rec["horizon_text"]


def test_last_position_edits_through_the_prefix_cache_equal_full_passes(tiny):
    """score_last_edits reuses one prefix cache for every edit; each result must equal
    a full forward pass with the same edit, in any order, with left padding."""
    near_first = [True, False, True]
    last = [[len(s) - 1] for s in SEQS]
    gen = torch.Generator().manual_seed(1)
    delta = torch.randn(3, 32, generator=gen)
    basis = torch.linalg.qr(torch.randn(32, 2, generator=gen))[0].T
    mean = torch.randn(32, generator=gen)
    donor = torch.randn(3, 32, generator=gen)
    edits = [None, (1, lambda rows: S.Add(delta[rows])), (2, lambda rows: S.MeanAblate(mean, basis)),
             (2, lambda rows: S.Replace(donor[rows])), None]
    cached = S.score_last_edits(tiny, "cpu", SEQS, near_first, ROWS, PAD, batch_size=2, edits=edits)
    for e, spec in enumerate(edits):
        if spec is None:
            full = S.score(tiny, "cpu", SEQS, near_first, ROWS, PAD, batch_size=2)
        else:
            full = S.score(tiny, "cpu", SEQS, near_first, ROWS, PAD, batch_size=2, layer=spec[0],
                           positions=last, make_edit=spec[1])
        np.testing.assert_allclose(cached[e], full, atol=1e-5)
    reversed_order = S.score_last_edits(tiny, "cpu", SEQS, near_first, ROWS, PAD, batch_size=2, edits=edits[::-1])
    np.testing.assert_allclose(reversed_order[::-1], cached, atol=1e-5)


def test_models_with_linear_attention_fall_back_to_full_passes(tiny, monkeypatch):
    """A recurrent layer's cache cannot be cropped, so score_last_edits must not use it."""
    monkeypatch.setattr(tiny.config, "layer_types", ["full_attention", "linear_attention", "full_attention"])
    monkeypatch.setattr(tiny, "forward", lambda *a, **k: (_ for _ in ()).throw(AssertionError("prefix path used")))
    calls = []
    monkeypatch.setattr(S, "score", lambda *a, **k: calls.append(a) or np.zeros(len(a[2])))
    out = S.score_last_edits(tiny, "cpu", SEQS, [True] * 3, ROWS, PAD, batch_size=2,
                             edits=[None, (1, lambda rows: S.Add(torch.zeros(len(rows), 32)))])
    assert out.shape == (2, 3) and len(calls) == 2


def confounded_bank(seed=0, d=40, noise=0.05):
    """Activations with a planted duration curve (2 dims), a unit effect that leans into
    the curve's plane, and regime and forced effects. Units track the duration but
    overlap, as on the units bank: each duration comes in its two neighbouring units."""
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.normal(size=(d, 6)))[0].T   # orthonormal directions
    durations = np.linspace(0, 3, 13)                 # log10 duration grid
    unit_dir = 0.6 * q[0] + 0.8 * q[2]                # half inside the curve plane (q0, q1)
    rows = []
    for i, h in enumerate(durations):
        for unit in (i // 3, i // 3 + 1):             # 5 units, each spanning ~4 durations
            for _ in range(6):
                rows.append((h, unit, rng.integers(3), rng.random() < 0.3 * h / 3))
    log_h = np.array([r[0] for r in rows])
    unit = np.array([r[1] for r in rows])
    regime_ = np.array([r[2] for r in rows])
    forced = np.array([r[3] for r in rows])
    X = (np.outer(log_h, q[0]) + np.outer(np.sin(log_h), q[1]) + np.outer(unit, unit_dir)
         + np.outer(regime_, q[3]) + np.outer(forced, 2 * q[4]) + noise * rng.normal(size=(len(rows), d)))
    return X, log_h, {"unit": unit, "regime": regime_, "forced": forced}, q, unit_dir


def test_duration_effects_read_the_duration_and_not_the_unit():
    X, log_h, factors, q, unit_dir = confounded_bank()
    clean = S.fit_duration_effects(X, log_h, factors, k_max=2)
    naive = S.fit_cell(X, log_h, k_max=2)
    # same duration, next unit up: a pristine readout does not move
    same_duration = log_h == log_h[0]
    low, high = same_duration & (factors["unit"] == 0), same_duration & (factors["unit"] == 1)
    for cell, ok in ((clean, True), (naive, False)):
        readout = S.curve_readout(cell, X)
        gap = abs(readout[high].mean() - readout[low].mean())
        assert (gap < 0.05) == ok, gap
    # the estimated factor effects are projected out exactly; the planted ones up to noise
    for name in factors:
        assert np.abs(clean["basis"] @ clean["factors"][name][1].T).max() < 1e-8
    assert np.abs(clean["basis"] @ unit_dir).max() < 0.02
    assert np.abs(clean["basis"] @ q[4]).max() < 0.02
    # the duration curve survives: the readout tracks log duration
    assert np.corrcoef(S.curve_readout(clean, X), log_h)[0, 1] > 0.95


def test_duration_effects_report_factor_levels_and_effects():
    # low noise: units far apart are linked only through a chain of overlaps, so their
    # estimates collect the noise of every link
    X, log_h, factors, q, unit_dir = confounded_bank(noise=0.01)
    clean = S.fit_duration_effects(X, log_h, factors)
    levels, effects = clean["factors"]["unit"]
    assert list(levels) == [0, 1, 2, 3, 4, 5] and effects.shape == (5, X.shape[1])
    # unit k relative to unit 0 is k * unit_dir
    assert np.allclose(effects @ unit_dir, [1, 2, 3, 4, 5], atol=0.05)


def test_duration_effects_refuse_a_factor_collinear_with_duration():
    X, log_h, factors, _, _ = confounded_bank()
    with pytest.raises(ValueError, match="not separable"):
        S.fit_duration_effects(X, log_h, {"copy of duration": np.round(log_h, 3)})


def test_curve_readout_on_the_curve_returns_its_horizon():
    X, log_h, _ = planted()
    cell = S.fit_cell(X, log_h)
    on_curve = cell["mean"] + cell["curve_coords"] @ cell["basis"]
    assert np.allclose(S.curve_readout(cell, on_curve), cell["curve_logh"], atol=0.01)


def test_duration_effects_separate_a_delay_that_tracks_the_horizon_in_part_of_the_bank():
    # pooled banks: in one half the delay is independent of the horizon (old banks), in the
    # other it sits about a decade past it (fill bank); a planted delay direction must be
    # recovered and kept out of the curve, and a fit without the covariate leans into it
    rng = np.random.default_rng(0)
    d = 30
    q = np.linalg.qr(rng.normal(size=(d, 3)))[0].T
    log_h = np.repeat(np.linspace(-2, 2, 9), 20)
    delay = np.where(np.arange(len(log_h)) % 2, rng.uniform(-2, 2, len(log_h)), log_h + 1 + 0.2 * rng.normal(size=len(log_h)))
    X = np.outer(log_h, q[0]) + np.outer(np.cos(log_h), q[1]) + np.outer(delay, q[2]) + 0.05 * rng.normal(size=(len(log_h), d))
    cell = S.fit_duration_effects(X, log_h, {}, k_max=2, covariates={"delay": delay})
    assert np.allclose(cell["covariates"]["delay"], q[2], atol=0.02)
    assert np.abs(cell["basis"] @ cell["covariates"]["delay"]).max() < 1e-8
    without = S.fit_duration_effects(X, log_h, {}, k_max=2, orthogonal=False)
    assert np.abs(without["basis"] @ q[2]).max() > 0.2
