"""The steering hook during real generate() calls (left padding, key-value cache) and in replay.

A tiny random Qwen3 on the CPU (always runs), and real Qwen3.5-0.8B through pbank's LocalChatModel on a GPU (bf16,
batched chat generation; skipped without a GPU or the cached model).
"""

import numpy as np
import pytest
import torch

from steer.directions import random_directions
from steer.edits import ROLE_GENERATED, ROLE_PAD, ROLE_PROMPT, LinearEdit
from steer.hooks import at_generated, at_prompt, replay_logits, steering
from test_layers import tiny_qwen3

LAYER = 2
NEW = 6


@pytest.fixture(scope="module")
def tiny():
    model = tiny_qwen3()
    ids = torch.tensor([[0, 0, 0, 5, 9, 2, 7, 11], [3, 8, 1, 4, 6, 10, 12, 13]])   # row 0 left-padded by 3
    mask = (torch.arange(8)[None, :] >= torch.tensor([[3], [0]])).long()
    return model, ids, mask


def generate(model, ids, mask):
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=NEW, min_new_tokens=NEW,
                             do_sample=False, pad_token_id=0, output_scores=True, return_dict_in_generate=True)
    return out.sequences, torch.stack(out.scores)          # scores: [steps, batch, vocab]


def unit(d, seed=1):
    return random_directions(d, 1, seed)[0]


def test_roles_and_masks_follow_prefill_decode_and_padding(tiny):
    model, ids, mask = tiny
    for positions in ("all", "prompt", "generated"):
        record = []
        with steering(model, LinearEdit(unit(32), 1.0), 0.0, LAYER, positions, record=record):
            generate(model, ids, mask)
        prefill, decode = record[0]["ctx"], [r["ctx"] for r in record[1:]]
        print(f"{positions:9s}: {len(record)} passes; prefill roles {prefill.role.tolist()}, mask {prefill.mask.int().tolist()}; "
              f"decode roles {[c.role.flatten().tolist() for c in decode]}, positions {[c.position[0, 0].item() for c in decode]}")
        assert len(record) == NEW                     # one prefill + NEW - 1 decode steps (the last token is not fed back)
        assert prefill.role.tolist() == [[ROLE_PAD] * 3 + [ROLE_PROMPT] * 5, [ROLE_PROMPT] * 8]
        assert all(c.role.tolist() == [[ROLE_GENERATED], [ROLE_GENERATED]] for c in decode)
        assert [c.position[0, 0].item() for c in decode] == list(range(8, 8 + NEW - 1))
        assert not prefill.mask[0, :3].any()          # padding is never edited
        expect_prefill = positions in ("all", "prompt")
        expect_decode = positions in ("all", "generated")
        assert bool(prefill.mask[0, 3:].all()) == expect_prefill and bool(prefill.mask[1].all()) == expect_prefill
        assert all(bool(c.mask.all()) == expect_decode and bool(c.mask.any()) == expect_decode for c in decode)


def test_zero_edit_reproduces_the_unhooked_model_exactly(tiny):
    model, ids, mask = tiny
    seq, scores = generate(model, ids, mask)
    with steering(model, LinearEdit(unit(32), 3.0), 0.0, LAYER, "all"):
        seq0, scores0 = generate(model, ids, mask)
    print(f"zero edit: sequences equal {torch.equal(seq, seq0)}, max |score diff| {(scores - scores0).abs().max():.1e}")
    assert torch.equal(seq, seq0) and torch.equal(scores, scores0)


def test_generated_only_leaves_the_prompt_and_first_token_alone(tiny):
    model, ids, mask = tiny
    seq, scores = generate(model, ids, mask)
    with steering(model, LinearEdit(unit(32), 3.0), 2.0, LAYER, "generated"):
        seq_g, scores_g = generate(model, ids, mask)
    with steering(model, LinearEdit(unit(32), 3.0), 2.0, LAYER, "prompt"):
        seq_p, scores_p = generate(model, ids, mask)
    print(f"generated-only: step-0 scores diff {(scores_g[0] - scores[0]).abs().max():.1e}, later steps "
          f"{(scores_g[1:] - scores[1:]).abs().max():.2f}; prompt-only: step-0 diff {(scores_p[0] - scores[0]).abs().max():.2f}")
    assert torch.equal(scores_g[0], scores[0])         # the first token comes from the unedited prefill
    assert (scores_g[1:] - scores[1:]).abs().max() > 1e-2
    assert (scores_p[0] - scores[0]).abs().max() > 1e-2


def test_padding_rows_do_not_change_what_real_tokens_receive(tiny):
    # The same prompt alone (no padding) and left-padded in a batch must get the same edit at its real tokens.
    model, ids, mask = tiny
    rec_batch, rec_alone = [], []
    with steering(model, LinearEdit(unit(32), 3.0), 0.5, LAYER, "all", record=rec_batch):
        generate(model, ids, mask)
    with steering(model, LinearEdit(unit(32), 3.0), 0.5, LAYER, "all", record=rec_alone):
        generate(model, ids[:1, 3:], mask[:1, 3:])
    a, b = rec_batch[0]["delta_norm"][0], rec_alone[0]["delta_norm"][0]
    print(f"displacement norms, padded row {a.tolist()} vs alone {b.tolist()}")
    assert torch.equal(a[:3], torch.zeros(3)) and torch.allclose(a[3:], b)


def test_callable_positions_and_per_row_strengths(tiny):
    model, ids, mask = tiny
    only_last_prompt_token = lambda ctx: ctx.position == 7
    record = []
    with steering(model, LinearEdit(unit(32), 3.0), torch.tensor([0.5, -0.25]), LAYER, only_last_prompt_token,
                  record=record):
        generate(model, ids, mask)
    sizes = record[0]["delta_norm"]
    print(f"callable positions: edited at {record[0]['ctx'].mask.nonzero().tolist()}, sizes {sizes[:, 7].tolist()}")
    assert record[0]["ctx"].mask.sum() == 2 and all(not r["ctx"].mask.any() for r in record[1:])
    assert torch.allclose(sizes[:, 7], torch.tensor([1.5, 0.75]))


def test_at_generated_edits_one_token_per_row(tiny):
    # Row 0 (left-padded by 3): generated token 2, absolute position 8 + 2 = 10, fed in on decode pass 3.
    # Row 1: index -1, the last prompt token (position 7, in the prefill), whose state predicts generated token 0.
    model, ids, mask = tiny
    seq, scores = generate(model, ids, mask)
    record = []
    with steering(model, LinearEdit(unit(32), 3.0), 2.0, LAYER, at_generated([2, -1]), record=record):
        seq_e, scores_e = generate(model, ids, mask)
    edited = [(p, r, int(c.position[r, t])) for p, rec in enumerate(record)
              for c in [rec["ctx"]] for r, t in c.mask.nonzero().tolist()]
    diff = (scores_e - scores).abs().amax(-1)                         # [steps, batch]
    print(f"at_generated([2, -1]): edited (pass, row, position) {edited}; per-step max |score diff| row 0 "
          f"{[f'{v:.1e}' for v in diff[:, 0].tolist()]}, row 1 {[f'{v:.1e}' for v in diff[:, 1].tolist()]}")
    assert edited == [(0, 1, 7), (3, 0, 10)]
    assert torch.equal(scores_e[:3, 0], scores[:3, 0]) and diff[3, 0] > 1e-2   # row 0: unchanged until after token 2
    assert diff[0, 1] > 1e-2                                                     # row 1: the first token's scores move
    # The same callable reused for a second run re-reads the prompt length from that run's first pass.
    cols, reused, rec_short = slice(1, None), at_generated([2, 0]), []
    with steering(model, LinearEdit(unit(32), 3.0), 2.0, LAYER, reused):
        generate(model, ids, mask)                                              # prompt length 8
    with steering(model, LinearEdit(unit(32), 3.0), 2.0, LAYER, reused, record=rec_short):
        generate(model, ids[:, cols], mask[:, cols])                            # prompt length 7
    short = [(p, r, int(rec["ctx"].position[r, t])) for p, rec in enumerate(rec_short)
             for r, t in rec["ctx"].mask.nonzero().tolist()]
    print(f"reused on a 7-token prompt batch: edited (pass, row, position) {short}")
    assert short == [(1, 1, 7), (3, 0, 9)]
    # Replay: one pass over prompt (5) + answer (3); generated index 1 is position 6.
    prompt, answer, rec_replay = [5, 9, 2, 7, 11], [4, 6, 10], []
    with torch.no_grad(), steering(model, LinearEdit(unit(32), 3.0), 2.0, LAYER, at_generated([1]),
                                   prompt_len=len(prompt), record=rec_replay):
        model(input_ids=torch.tensor([prompt + answer]))
    print(f"replay: edited positions {rec_replay[0]['ctx'].mask.nonzero()[:, 1].tolist()}")
    assert rec_replay[0]["ctx"].mask.nonzero()[:, 1].tolist() == [6]


def test_at_prompt_edits_chosen_prompt_tokens_per_row(tiny):
    # Row 0 is left-padded by 3, so its tokens 1 and 2 sit at positions 4 and 5; row 1 has none; no decode pass is edited.
    model, ids, mask = tiny
    seq, scores = generate(model, ids, mask)
    record = []
    with steering(model, LinearEdit(unit(32), 3.0), 2.0, LAYER, at_prompt([[1, 2], []]), record=record):
        seq_e, scores_e = generate(model, ids, mask)
    edited = [(p, r, t) for p, rec in enumerate(record) for r, t in rec["ctx"].mask.nonzero().tolist()]
    diff = (scores_e - scores).abs().amax(-1)
    print(f"at_prompt([[1, 2], []]): edited (pass, row, position) {edited}; step-0 max |score diff| per row {diff[0].tolist()}")
    assert edited == [(0, 0, 4), (0, 0, 5)]
    assert diff[0, 0] > 1e-2 and torch.equal(scores_e[:, 1], scores[:, 1])       # row 1 untouched, bit for bit
    # Replay (one pass, no padding): the indices are the positions themselves.
    rec_replay = []
    with torch.no_grad(), steering(model, LinearEdit(unit(32), 3.0), 2.0, LAYER, at_prompt([[0, 3]]), prompt_len=5,
                                   record=rec_replay):
        model(input_ids=torch.tensor([[5, 9, 2, 7, 11, 4, 6, 10]]))
    assert rec_replay[0]["ctx"].mask.nonzero()[:, 1].tolist() == [0, 3]
    # An index beyond the row's prompt is refused, not wrapped into another token.
    with pytest.raises(ValueError), steering(model, LinearEdit(unit(32), 3.0), 2.0, LAYER, at_prompt([[5], []])):
        generate(model, ids, mask)                                              # row 0 has 5 prompt tokens: 0..4


def test_roles_without_padding(tiny):
    # transformers 5.22.2 passes no cache_position and drops an all-ones attention_mask at batch 1; the positions
    # then come from position_ids. Before 2026-11-19 they fell back to arange(seq_len), so every decode pass looked
    # like a new one-token prompt and "generated" edited nothing.
    model, ids, mask = tiny
    record = []
    with steering(model, LinearEdit(unit(32), 1.0), 0.0, LAYER, "generated", record=record):
        generate(model, ids[:1, 3:], mask[:1, 3:])                              # one row, no padding, prompt length 5
    decode = [r["ctx"] for r in record[1:]]
    print(f"batch 1, no padding: decode positions {[int(c.position[0, 0]) for c in decode]}, "
          f"roles {[int(c.role[0, 0]) for c in decode]}, edited {[int(c.mask.sum()) for c in decode]}")
    assert [int(c.position[0, 0]) for c in decode] == list(range(5, 5 + NEW - 1))
    assert all(bool((c.role == ROLE_GENERATED).all()) and bool(c.mask.all()) for c in decode)
    assert record[0]["ctx"].role.tolist() == [[ROLE_PROMPT] * 5] and not record[0]["ctx"].mask.any()


def test_replay_logits(tiny):
    model, _, _ = tiny
    prompt, answer = [5, 9, 2, 7, 11], [4, 6, 10]
    with torch.no_grad():
        plain = model(input_ids=torch.tensor([prompt + answer])).logits[0, -1].float()
    zero = replay_logits(model, prompt, answer, LinearEdit(unit(32), 3.0), 0.0, LAYER)
    gen_only = replay_logits(model, prompt, answer, LinearEdit(unit(32), 3.0), 2.0, LAYER, "generated")
    with torch.no_grad(), steering(model, LinearEdit(unit(32), 3.0), 2.0, LAYER,
                                   lambda ctx: ctx.position >= len(prompt), prompt_len=len(prompt)):
        manual = model(input_ids=torch.tensor([prompt + answer])).logits[0, -1].float()
    print(f"replay: zero edit vs plain max |diff| {(zero - plain).abs().max():.1e}; generated-only edit moved logits by "
          f"{(gen_only - plain).abs().max():.2f}, equal to a manual position mask: {torch.equal(gen_only, manual)}")
    assert torch.equal(zero, plain) and torch.equal(gen_only, manual) and (gen_only - plain).abs().max() > 1e-2


# ---- real model, bf16, through pbank's batched chat generation -----------------------------------------------------

@pytest.fixture(scope="module")
def qwen35():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from pbank.local_model import LocalChatModel
    try:
        return LocalChatModel("Qwen/Qwen3.5-0.8B")
    except OSError:
        pytest.skip("Qwen3.5-0.8B not cached")


def bank_texts(n):
    from fixtures import BANK
    from pbank.records import load_prompts
    return [r.text for r in load_prompts(BANK)[:n]]


def test_real_model_zero_edit_is_exact_and_a_strong_edit_is_not(qwen35):
    lcm, texts = qwen35, bank_texts(6)
    d = lcm.model.config.get_text_config().hidden_size
    base = lcm.generate(texts, thinking=False, max_new_tokens=40)
    record = []
    with steering(lcm.model, LinearEdit(unit(d), 50.0), 0.0, 12, "all", record=record):
        zero = lcm.generate(texts, thinking=False, max_new_tokens=40)
    with steering(lcm.model, LinearEdit(unit(d), 50.0), 4.0, 12, "all"):
        strong = lcm.generate(texts, thinking=False, max_new_tokens=40)
    same = [a.generated_ids == b.generated_ids for a, b in zip(base, zero)]
    changed = sum(a.generated_ids != b.generated_ids for a, b in zip(base, strong))
    pads = (record[0]["ctx"].role == ROLE_PAD).sum(1).tolist()
    print(f"Qwen3.5-0.8B bf16, 6 prompts batched (left padding per row {pads}): zero edit identical ids {same}; "
          f"strength 4.0 changed {changed}/6. First reply: {base[0].answer!r} -> {strong[0].answer!r}")
    assert all(same) and changed >= 1 and record[0]["ctx"].role.eq(ROLE_PROMPT).any()
    assert len(record) > 1 and record[1]["ctx"].role.eq(ROLE_GENERATED).all()
