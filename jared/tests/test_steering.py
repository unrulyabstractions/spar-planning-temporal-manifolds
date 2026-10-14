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
