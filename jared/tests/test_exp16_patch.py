"""Phrase number-patch driver on a tiny random-weight Qwen3 and the cached Qwen3.5-0.8B tokenizer:
unit order, which token positions each patch touches (and nothing else), the subspace and step
arithmetic, the rerun's order, claim and resume. Inputs are built here in the format
pipeline/s1_data_vectors/analysis/patch_prepare.py writes."""

import argparse
import json
from contextlib import nullcontext

import numpy as np
import pytest
import torch

import pipeline.s2_steering.experiments.patch.exp16_phrase_patch as patch
from pipeline.s1_data_vectors.experiments.exp14_phrase_states import record_phrase
from pipeline.s2_steering.experiments.patch.exp10_patch_reasoning import states_at
from pipeline.s2_steering.analysis.patch_summary import planned_units
from spar_horizon import steering as S
from test_exp14_states import BANK_SETTINGS, GENERATION, MODEL, bank_records, cfg, tiny, tok  # noqa: F401

D = 32
N = 4
LAYER = 16
DEPTH = [12, 18]                  # the tiny model has 19 layers
NAMES = ["baseline", "rerun", "self phrase L16", "donor phrase L16", "donor number L16", "manifold number L16",
         "manifold phrase L16", "random number L16", "donor unit L16", "subspace number L16", "inspan number L16",
         "donor number L12", "donor number L18"]


def rotate_digits(text):
    """Same token count as `text` (digits are single tokens), another number."""
    return text.translate(str.maketrans("0123456789", "1234567890"))


def write_inputs(out_dir):
    records = bank_records()[:N]
    prompts = [{"transition": ("none-near", "both-near")[i % 2], "start_regime": ("none", "both")[i % 2],
                "target_regime": "near", "log_h": float(np.log10(r["horizon"])), "target_log_h": 0.5,
                "donor_horizon_text": rotate_digits(r["horizon_text"]), "donor_log_h": 0.5}
               for i, r in enumerate(records)]
    meta = {"model": MODEL, "generation": GENERATION, "bank_settings": BANK_SETTINGS,
            "selected": list(range(N)), "selected_banks": ["full"] * N,
            "sample_uids": [r["sample_uid"] for r in records], "prompts": prompts,
            "transitions": ["none-near", "both-near"], "patch_layer": LAYER, "depth_layers": DEPTH}
    rng = np.random.default_rng(0)
    arrays = {f"step|L{LAYER}|{kind}|{role}": (3.0 * rng.normal(size=(N, D))).astype(np.float32)
              for kind, role in (("manifold", "number"), ("manifold", "unit"), ("random", "number"), ("inspan", "number"))}
    arrays[f"basis|L{LAYER}"] = np.linalg.qr(rng.normal(size=(D, 8)))[0].T.astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inputs.json").write_text(json.dumps(meta))
    np.savez(out_dir / "inputs.npz", **arrays)
    return records, prompts, arrays


def run_args(out_dir, only=None, limit=None):
    return argparse.Namespace(dir=str(out_dir), limit=limit, max_think_tokens=2, batch_size=3, score_batch_size=2,
                              only=only)


def load(out_dir, name):
    return json.loads((out_dir / "conditions" / f"{name.replace(' ', '_')}.json").read_text())


def make_ctx(tok, tiny, cfg, out_dir):
    records, prompts, _ = write_inputs(out_dir)
    donor_ids, donor_phrase = patch.donor_inputs(tok, cfg, records, prompts)
    phrase = [record_phrase(tok, cfg, r) for r in records]
    assert donor_phrase == phrase
    return {"model": tiny, "device": "cpu", "tokenizer": tok, "cfg": cfg, "steps": np.load(out_dir / "inputs.npz"),
            "records": records, "prompts": prompts, "banks": ["full"] * N,
            "ids": [S.encode(tok, r["prompt"], cfg) for r in records], "donor_ids": donor_ids, "phrase": phrase,
            "batch_size": 3, "score_batch_size": 2}


@pytest.fixture(scope="module")
def full_run(tok, tiny, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("patch") / "exp16_patch"
    write_inputs(out_dir)
    order = []
    original = patch.run_claimed
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(patch, "load_model", lambda cfg: (tok, tiny, "cpu"))
        patcher.setattr(patch, "run_claimed", lambda d, name, run: order.append(name) or original(d, name, run))
        patch.main(run_args(out_dir))
    return out_dir, order


@pytest.fixture
def tiny_model(tok, tiny, monkeypatch):
    monkeypatch.setattr(patch, "load_model", lambda cfg: (tok, tiny, "cpu"))


def hidden_at_layer(ctx, patch_ctx=None):
    """Per pair, the [len(ids), d] output of decoder layer LAYER - 1 (hidden_states[LAYER]) of the recipient
    prompt, read by a hook registered after the patch's, so that it sees the patched values."""
    out = []
    for i in range(N):
        seen = []

        def capture(_module, _inputs, output):
            seen.append((output[0] if isinstance(output, tuple) else output)[0].float())

        with patch_ctx([i]) if patch_ctx else nullcontext():
            handle = ctx["model"].model.layers[LAYER - 1].register_forward_hook(capture)
            try:
                with torch.no_grad():
                    S.forward(ctx["model"], "cpu", [ctx["ids"][i]], ctx["tokenizer"].pad_token_id)
            finally:
                handle.remove()
        out.append(seen[0])
    return out


def test_units_run_baseline_rerun_then_the_conditions_of_the_spec():
    units = planned_units(16, (12, 22))
    assert [u[0] for u in units] == NAMES[:11] + ["donor number L12", "donor number L22"]
    assert [u[1] for u in units] == [None, None, 1, 2, 3, 6, 7, 8, 4, 5, 9, 10, 10]


def test_every_unit_is_written_in_run_order_with_json_and_npz(full_run):
    out_dir, order = full_run
    assert order == NAMES
    for name in order:
        assert (out_dir / "conditions" / f"{name.replace(' ', '_')}.json").exists()
        assert (out_dir / "conditions" / f"{name.replace(' ', '_')}.npz").exists()
    assert not [p for p in (out_dir / "conditions").iterdir() if p.suffix == ".lock" and p.is_file()]


def test_row_format(full_run):
    out_dir, _ = full_run
    unit = load(out_dir, "donor number L16")
    assert {"name", "condition", "source", "positions", "layer", "rows", "seconds"} <= set(unit)
    assert (unit["condition"], unit["source"], unit["positions"], unit["layer"]) == (3, "donor", "number", 16)
    assert len(unit["rows"]) == N
    assert {"sample_uid", "transition", "horizon_text", "donor_horizon_text", "donor_log_h", "target_regime", "near_lo",
            "label", "forced", "think_len", "think_text", "near_delay_years", "far_delay_years"} <= set(unit["rows"][0])
    assert load(out_dir, "baseline")["layer"] is None
    assert {"ids", "offsets", "answer_start", "prompt_len"} <= set(np.load(out_dir / "conditions" / "donor_number_L16.npz").files)


def test_the_self_patch_reproduces_the_baseline(full_run):
    out_dir, _ = full_run
    base, own = load(out_dir, "baseline")["rows"], load(out_dir, "self phrase L16")["rows"]
    assert [r["label"] for r in own] == [r["label"] for r in base]
    assert [r["think_text"] for r in own] == [r["think_text"] for r in base]
    np.testing.assert_allclose([r["near_lo"] for r in own], [r["near_lo"] for r in base], atol=1e-3)


def test_real_patches_move_the_scores(full_run):
    out_dir, _ = full_run
    base = np.array([r["near_lo"] for r in load(out_dir, "baseline")["rows"]])
    for name in NAMES[3:]:
        lo = np.array([r["near_lo"] for r in load(out_dir, name)["rows"]])
        assert not np.allclose(lo, base, atol=1e-5), name


def test_the_rerun_has_the_baselines_pairs_in_the_baselines_order(full_run):
    out_dir, _ = full_run
    base, rerun = load(out_dir, "baseline")["rows"], load(out_dir, "rerun")["rows"]
    assert [r["sample_uid"] for r in rerun] == [r["sample_uid"] for r in base]
    np.testing.assert_allclose([r["near_lo"] for r in rerun], [r["near_lo"] for r in base], atol=1e-3)


def test_the_rerun_uses_another_order_with_the_same_pairs():
    order, batch = patch.rerun_batching(24)
    assert sorted(order.tolist()) == list(range(24)) and order.tolist() != list(range(24))
    assert batch == 12


def test_token_positions_split_the_phrase_into_number_tokens_and_the_unit_token():
    assert patch.token_positions([5, 6, 7, 8], "phrase") == [5, 6, 7, 8]
    assert patch.token_positions([5, 6, 7, 8], "number") == [5, 6, 7]
    assert patch.token_positions([5, 6, 7, 8], "unit") == [8]
    states = torch.arange(8).reshape(4, 2)
    assert patch.phrase_values(states, "number").tolist() == states[:3].tolist()
    assert patch.phrase_values(states, "unit").tolist() == states[3:].tolist()


@pytest.mark.parametrize("where", ["phrase", "number", "unit"])
def test_a_donor_patch_sets_exactly_its_tokens_to_the_donors_states(tok, tiny, cfg, tmp_path, where):
    ctx = make_ctx(tok, tiny, cfg, tmp_path)
    plain = hidden_at_layer(ctx)
    donor = states_at(tiny, "cpu", tok, ctx["donor_ids"], ctx["phrase"], [LAYER], 2)[LAYER]
    patched = hidden_at_layer(ctx, patch.unit_patch(ctx, "donor", where, LAYER))
    for i, phrase in enumerate(ctx["phrase"]):
        touched = patch.token_positions(phrase, where)
        untouched = [p for p in phrase if p not in touched]
        torch.testing.assert_close(patched[i][touched], patch.phrase_values(donor[i], where), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(patched[i][untouched], plain[i][untouched], atol=1e-4, rtol=1e-4)
        assert not torch.allclose(patched[i][touched], plain[i][touched], atol=1e-3)
        before = torch.arange(len(plain[i])) < phrase[0]
        torch.testing.assert_close(patched[i][before], plain[i][before], atol=1e-4, rtol=1e-4)


def test_the_subspace_patch_swaps_only_the_donors_top_eight_component(tok, tiny, cfg, tmp_path):
    ctx = make_ctx(tok, tiny, cfg, tmp_path)
    basis = torch.tensor(ctx["steps"][f"basis|L{LAYER}"])
    plain = hidden_at_layer(ctx)
    donor = states_at(tiny, "cpu", tok, ctx["donor_ids"], ctx["phrase"], [LAYER], 2)[LAYER]
    patched = hidden_at_layer(ctx, patch.unit_patch(ctx, "subspace", "number", LAYER))
    for i, phrase in enumerate(ctx["phrase"]):
        number = phrase[:-1]
        expected = plain[i][number] + ((donor[i][:-1] - plain[i][number]) @ basis.T) @ basis
        torch.testing.assert_close(patched[i][number], expected, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(patched[i][phrase[-1:]], plain[i][phrase[-1:]], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("source,where", [("manifold", "number"), ("manifold", "phrase"), ("random", "number"),
                                          ("inspan", "number")])
def test_a_step_is_added_to_the_number_tokens_and_for_the_whole_phrase_the_unit_step_to_the_unit_token(
        tok, tiny, cfg, tmp_path, source, where):
    ctx = make_ctx(tok, tiny, cfg, tmp_path)
    steps = ctx["steps"]
    plain = hidden_at_layer(ctx)
    patched = hidden_at_layer(ctx, patch.unit_patch(ctx, source, where, LAYER))
    for i, phrase in enumerate(ctx["phrase"]):
        number_step = torch.tensor(steps[f"step|L{LAYER}|{source}|number"][i])
        torch.testing.assert_close(patched[i][phrase[:-1]], plain[i][phrase[:-1]] + number_step, atol=1e-4, rtol=1e-4)
        unit_step = torch.tensor(steps[f"step|L{LAYER}|manifold|unit"][i]) if where == "phrase" else 0
        torch.testing.assert_close(patched[i][phrase[-1:]], plain[i][phrase[-1:]] + unit_step, atol=1e-4, rtol=1e-4)


def test_only_and_limit_select_units_and_pairs_and_a_rerun_skips_finished_units(tiny_model, tmp_path, monkeypatch, capsys):
    write_inputs(tmp_path)
    args = run_args(tmp_path, only="baseline;rerun;donor number L16", limit=2)
    patch.main(args)
    assert sorted(p.name for p in (tmp_path / "conditions").glob("*.json")) == [
        "baseline.json", "donor_number_L16.json", "rerun.json"]
    assert all(len(load(tmp_path, n)["rows"]) == 2 for n in ("baseline", "rerun", "donor number L16"))
    lines = [line for line in capsys.readouterr().out.splitlines() if "near log-odds" in line]
    assert len(lines) == 3 and "flips" in lines[1] and "flips" not in lines[0]

    monkeypatch.setattr(patch, "run_unit", lambda *a: pytest.fail("a finished unit was rerun"))
    patch.main(args)


def test_a_claimed_unfinished_unit_is_left_to_its_worker(tiny_model, tmp_path):
    write_inputs(tmp_path)
    (tmp_path / "conditions").mkdir()
    (tmp_path / "conditions" / "baseline.lock").mkdir()
    patch.main(run_args(tmp_path, only="baseline;rerun", limit=1))
    assert sorted(p.name for p in (tmp_path / "conditions").glob("*.json")) == ["rerun.json"]


def test_a_donor_phrase_at_other_positions_than_the_recipients_is_an_error(tiny_model, tmp_path):
    records, prompts, _ = write_inputs(tmp_path)
    meta = json.loads((tmp_path / "inputs.json").read_text())
    meta["prompts"][0]["donor_horizon_text"] = records[0]["horizon_text"].split()[0] + "0 " + records[0]["horizon_text"].split()[1]
    (tmp_path / "inputs.json").write_text(json.dumps(meta))
    with pytest.raises(AssertionError, match="donor phrase at"):
        patch.main(run_args(tmp_path, only="baseline"))
