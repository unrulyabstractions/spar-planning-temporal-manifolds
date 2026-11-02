"""E14 phrase steering driver on a tiny random-weight Qwen3 and the cached Qwen3.5-0.8B
tokenizer: step layout, prefill locality, zero step = noop, unit order and files, claim and
resume, donor positions. Inputs are built here, following the interfaces doc."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import exp14_phrase_steer as steer  # noqa: E402
from exp10_patch_reasoning import states_at  # noqa: E402
from exp14_phrase_states import record_phrase  # noqa: E402
from spar_horizon import steering as S  # noqa: E402
from test_exp14_states import BANK_SETTINGS, GENERATION, MODEL, bank_records, cfg, tiny, tok  # noqa: E402,F401

D = 32
TRANSITIONS = ("both-near", "near-both")
STARTS = {"both-near": "both", "near-both": "near"}
UNITS_BOTH_START = ["noop", "effects L16", "random L16", "donor L16", "invariant L16", "opposite L16",
                    "effects L14", "effects L18", "random L14", "random L18"]


def rotate_digits(text):
    """Same token count as `text` (digits are single tokens), another number."""
    return text.translate(str.maketrans("0123456789", "1234567890"))


def write_inputs(out_dir, zero_effects_l16=False):
    """inputs.json and inputs.npz (interfaces §3) for six bank prompts: three both-near
    (the last has no donor) then three near-both."""
    records = bank_records()[:6]
    prompts = []
    for i, record in enumerate(records):
        transition = TRANSITIONS[i // 3]
        has_donor = i != 2
        prompts.append({"transition": transition, "start_regime": STARTS[transition],
                        "target_regime": "near" if i < 3 else "both",
                        "log_h": float(np.log10(record["horizon"])), "target_log_h": 0.5,
                        "opposite_log_h": None, "opposite_clamped": False,
                        "donor_horizon_text": rotate_digits(record["horizon_text"]) if has_donor else None,
                        "donor_log_h": 1.0 if has_donor else None})
    meta = {"model": MODEL, "generation": GENERATION, "bank_settings": BANK_SETTINGS,
            "selected": list(range(6)), "selected_banks": ["full"] * 6,
            "sample_uids": [r["sample_uid"] for r in records], "prompts": prompts,
            "transitions": list(TRANSITIONS), "layers": [14, 16, 18]}
    rng = np.random.default_rng(0)
    steps = {}
    for layer in (14, 16, 18):
        for kind in ("effects", "random") + (("invariant", "opposite") if layer == 16 else ()):
            for role in ("number", "unit"):
                steps[f"step|L{layer}|{kind}|{role}"] = (0.3 * rng.normal(size=(6, D))).astype(np.float32)
    for role in ("number", "unit"):
        steps[f"step|L16|opposite|{role}"][3:] = 0     # near starts have no opposite step
        if zero_effects_l16:
            steps[f"step|L16|effects|{role}"][:] = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inputs.json").write_text(json.dumps(meta))
    np.savez(out_dir / "inputs.npz", **steps)
    return records, meta, steps


def run_args(out_dir, only=None, limit=2):
    return argparse.Namespace(dir=str(out_dir), limit=limit, max_think_tokens=2, batch_size=2, score_batch_size=2,
                              only=only)


@pytest.fixture
def tiny_model(tok, tiny, monkeypatch):
    monkeypatch.setattr(steer, "load_model", lambda cfg: (tok, tiny, "cpu"))


@pytest.fixture(scope="module")
def full_run(tok, tiny, tmp_path_factory):
    """Every unit of both transitions, two prompts each, effects L16 step zero; the order
    in which the driver claimed the units."""
    out_dir = tmp_path_factory.mktemp("full") / "exp14_phrase"
    write_inputs(out_dir, zero_effects_l16=True)
    order = []
    original = steer.run_claimed
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(steer, "load_model", lambda cfg: (tok, tiny, "cpu"))
        patcher.setattr(steer, "run_claimed", lambda d, name, run: order.append(name) or original(d, name, run))
        steer.main(run_args(out_dir))
    return out_dir, order


def load_unit(out_dir, name):
    return json.loads((out_dir / "conditions" / f"{name.replace(' ', '_')}.json").read_text())


def test_deltas_put_the_number_step_on_number_tokens_and_the_unit_step_on_the_last(tmp_path):
    _, _, steps = write_inputs(tmp_path)
    deltas = steer.step_deltas(np.load(tmp_path / "inputs.npz"), 16, "effects", [4, 1], [3, 2])
    assert [d.shape for d in deltas] == [(3, D), (2, D)]
    for delta, row, n in zip(deltas, [4, 1], [3, 2]):
        for k in range(n - 1):
            np.testing.assert_array_equal(delta[k].numpy(), steps["step|L16|effects|number"][row])
        np.testing.assert_array_equal(delta[-1].numpy(), steps["step|L16|effects|unit"][row])


def test_a_phrase_step_changes_the_prefill_only_at_and_after_the_phrase(tok, tiny, cfg, tmp_path):
    records, _, _ = write_inputs(tmp_path)
    record = records[0]
    ids = S.encode(tok, record["prompt"], cfg)
    positions = record_phrase(tok, cfg, record)
    deltas = steer.step_deltas(np.load(tmp_path / "inputs.npz"), 16, "effects", [0], [len(positions)])
    _, clean, _ = S.forward(tiny, "cpu", [ids], tok.pad_token_id, hidden=True)
    with S.prompt_patch(tiny, 16, [positions], deltas, add=True):
        _, edited, _ = S.forward(tiny, "cpu", [ids], tok.pad_token_id, hidden=True)
    first = positions[0]
    for layer in (17, 18, 19):
        diff = (edited[layer] - clean[layer])[0]
        assert diff[:first].abs().max() < 1e-6                 # causal: the prompt before the phrase is untouched
        assert diff[first].abs().max() > 1e-4                  # the edited token itself
        assert diff[positions[-1]:].abs().max() > 1e-4         # and what reads it
    assert torch.allclose(edited[15], clean[15])


def test_the_unit_plan_follows_the_run_order_and_drops_opposite_for_near_starts():
    both = steer.planned_units("both-near", "both")
    assert [name for name, _, _ in both] == [f"both-near__{c}" for c in UNITS_BOTH_START]
    near = steer.planned_units("near-both", "near")
    assert [name for name, _, _ in near] == [f"near-both__{c}" for c in UNITS_BOTH_START if c != "opposite L16"]
    assert [name for name, _, _ in steer.planned_units("none-near", "none")] == \
        [f"none-near__{c}" for c in UNITS_BOTH_START]


def test_units_run_in_order_and_write_json_and_npz_files(full_run):
    out_dir, order = full_run
    expected = [f"{t}__{c}" for t in TRANSITIONS for c in UNITS_BOTH_START
                if not (t == "near-both" and c == "opposite L16")]
    assert order == expected
    files = sorted(p.name for p in (out_dir / "conditions").iterdir())
    assert "both-near__effects_L16.json" in files and "both-near__effects_L16.npz" in files
    assert "near-both__random_L18.json" in files
    assert not any("near-both__opposite" in f for f in files)
    assert len([f for f in files if f.endswith(".json")]) == len(expected)
    assert not [f for f in files if f.endswith(".lock") and (out_dir / "conditions" / f).is_file()]


def test_unit_files_follow_the_row_format(full_run):
    out_dir, _ = full_run
    unit = load_unit(out_dir, "both-near__effects L16")
    assert {"name", "transition", "condition", "layer", "seconds", "rows"} <= set(unit)
    assert unit["name"] == "both-near__effects L16" and unit["layer"] == 16
    keys = {"sample_uid", "bank", "transition", "condition", "layer", "start_regime", "target_regime",
            "horizon_text", "log_h", "target_log_h", "near_label", "near_delay_years", "far_delay_years",
            "near_lo", "label", "forced", "think_len", "think_text", "names_own", "names_target"}
    assert len(unit["rows"]) == 2 and all(set(r) == keys for r in unit["rows"])
    assert load_unit(out_dir, "both-near__noop")["layer"] is None
    arrays = np.load(out_dir / "conditions" / "both-near__effects_L16.npz")
    assert {"ids", "offsets", "answer_start", "prompt_len"} <= set(arrays.files)


def test_a_zero_step_reproduces_the_noop_rerun(full_run):
    out_dir, _ = full_run
    noop = load_unit(out_dir, "both-near__noop")["rows"]
    zero = load_unit(out_dir, "both-near__effects L16")["rows"]
    assert [r["label"] for r in noop] == [r["label"] for r in zero]
    np.testing.assert_allclose([r["near_lo"] for r in noop], [r["near_lo"] for r in zero], atol=1e-6)
    assert [r["think_text"] for r in noop] == [r["think_text"] for r in zero]


def test_a_nonzero_step_changes_the_scores(full_run):
    out_dir, _ = full_run
    noop = [r["near_lo"] for r in load_unit(out_dir, "both-near__noop")["rows"]]
    steered = [r["near_lo"] for r in load_unit(out_dir, "both-near__effects L14")["rows"]]
    assert not np.allclose(noop, steered, atol=1e-7)


def test_donor_unit_leaves_out_prompts_without_a_donor_and_counts_them(full_run):
    out_dir, _ = full_run
    donor = load_unit(out_dir, "both-near__donor L16")
    assert len(donor["rows"]) == 2 and donor["n_without_donor"] == 0     # limit 2 keeps rows 0 and 1
    assert load_unit(out_dir, "both-near__noop")["n_without_donor"] == 0


def test_the_donor_count_is_recorded_when_a_prompt_has_none(tiny_model, tmp_path):
    write_inputs(tmp_path)
    steer.main(run_args(tmp_path, only="both-near__donor L16", limit=3))
    donor = load_unit(tmp_path, "both-near__donor L16")
    assert len(donor["rows"]) == 2 and donor["n_without_donor"] == 1


def test_only_runs_the_named_units_and_a_rerun_skips_finished_ones(tiny_model, tmp_path, monkeypatch, capsys):
    write_inputs(tmp_path)
    args = run_args(tmp_path, only="both-near__noop;near-both__random L18", limit=1)
    steer.main(args)
    done = sorted(p.name for p in (tmp_path / "conditions").glob("*.json"))
    assert done == ["both-near__noop.json", "near-both__random_L18.json"]
    lines = [line for line in capsys.readouterr().out.splitlines() if "near log-odds" in line]
    assert len(lines) == 2 and "names target" in lines[0] and "forced" in lines[0]

    monkeypatch.setattr(steer, "run_unit", lambda *a: pytest.fail("a finished unit was rerun"))
    steer.main(args)


def test_a_claimed_unfinished_unit_is_left_to_its_worker(tiny_model, tmp_path, monkeypatch):
    write_inputs(tmp_path)
    (tmp_path / "conditions").mkdir()
    (tmp_path / "conditions" / "both-near__noop.lock").mkdir()
    steer.main(run_args(tmp_path, only="both-near__noop;both-near__effects L16", limit=1))
    assert sorted(p.name for p in (tmp_path / "conditions").glob("*.json")) == ["both-near__effects_L16.json"]


def donor_context(tok, tiny, cfg, tmp_path, donor_text):
    records, meta, _ = write_inputs(tmp_path)
    prompts = [dict(p) for p in meta["prompts"]]
    prompts[0]["donor_horizon_text"] = donor_text(records[0]["horizon_text"])
    return {"model": tiny, "device": "cpu", "tokenizer": tok, "cfg": cfg, "prompts": prompts,
            "records": {0: records[0]}, "score_batch_size": 2}, records[0]


def test_donor_states_come_from_the_donor_prompt_at_the_recipients_phrase_positions(tok, tiny, cfg, tmp_path):
    ctx, record = donor_context(tok, tiny, cfg, tmp_path, rotate_digits)
    positions = record_phrase(tok, cfg, record)
    (values,) = steer.donor_states(ctx, [0], [positions])
    assert values.shape == (len(positions), D)
    own = states_at(tiny, "cpu", tok, [S.encode(tok, record["prompt"], cfg)], [positions], [16], 1)[16][0]
    assert not torch.allclose(values, own)

    # a donor with the recipient's own horizon text rebuilds the recipient's prompt exactly
    ctx, record = donor_context(tok, tiny, cfg, tmp_path, lambda text: text)
    (same,) = steer.donor_states(ctx, [0], [positions])
    assert torch.allclose(same, own, atol=1e-5)


def test_donor_phrase_with_another_token_count_is_an_error(tok, tiny, cfg, tmp_path):
    ctx, record = donor_context(tok, tiny, cfg, tmp_path, lambda text: "1" + text)
    with pytest.raises(AssertionError, match="donor phrase at"):
        steer.donor_states(ctx, [0], [record_phrase(tok, cfg, record)])


def test_transition_rows_take_the_first_n_per_transition_and_skip_empty_ones():
    prompts = [{"transition": t} for t in ("a", "a", "a", "b", "b")]
    assert steer.transition_rows(prompts, ["a", "b", "c"], 2) == {"a": [0, 1], "b": [3, 4]}
    assert steer.transition_rows(prompts, ["a", "b"], None) == {"a": [0, 1, 2], "b": [3, 4]}
