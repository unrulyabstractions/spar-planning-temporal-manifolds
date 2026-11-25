"""E20 transfer on eval types, no model: the rows as steering records, the donor prompt built by replacing
the horizon phrase, its phrase tokens on the Qwen tokenizer, the row draw and the summary."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer

import pipeline.s2_steering.experiments.manifold_steer.exp20_evaltype_transfer as exp20
from pipeline.s2_steering.experiments.manifold_steer.exp20_evaltype_transfer import (
    plan_rows, select_rows, summarize)
from spar_horizon.config import Settings
from spar_horizon.eval_types import build
from spar_horizon.eval_types.records import eval_records, find_donor, with_horizon_text
from spar_horizon.full_bank import regime
from spar_horizon.model_io import chat_text
from spar_horizon.steering import phrase_positions
from test_exp14_states import tiny  # noqa: F401

DOMAIN_TYPES = ("city_no_values", "city_values", "public_health", "coding_agent")
EXTRA_TYPES = ("distractor_temporal", "multiturn_analyst", "role_assistant", "role_fund_manager", "sycophancy_lean",
               "system_prompt", "system_prompt_clear", "t0_w0", "t0_w1_cash", "t0_w3_clear")
REAL_RUN = Path.home() / "SPAR/results/horizon/evaltypes_Qwen3-8B_think-on"
CFG = Settings(model="Qwen/Qwen3-8B", thinking="on", prefill="", choice_prefix="I choose:", answer_window=200)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(CFG.model)


def written_rows(type_id, tmp_path):
    """The type's rows as E0 saved them: the built records plus an answer and the label logits
    (the first label wins by 10 logits when the row's index is even)."""
    rows = []
    for i, row in enumerate(build(type_id)):
        first_wins = i % 2 == 0
        rows.append({**row, "answer": f" {row['labels'][0 if first_wins else 1]}) My reasoning: ...",
                     "label_logits": [30.0, 20.0] if first_wins else [20.0, 30.0]})
    path = tmp_path / f"{type_id}.json"
    path.write_text(json.dumps(rows))
    return path


@pytest.mark.parametrize("type_id", ["city_values", "multiturn_analyst"])
def test_eval_records_fields(type_id, tmp_path):
    rows = json.loads(written_rows(type_id, tmp_path).read_text())
    records = eval_records(written_rows(type_id, tmp_path))
    assert len(records) == len(rows)
    for row, record in zip(rows, records):
        by_kind = {o["kind"]: o["delay_years"] for o in row["options"]}
        assert (record["near_delay_years"], record["far_delay_years"]) == (by_kind["near"], by_kind["far"])
        assert record["near_delay_years"] < record["far_delay_years"]
        assert record["label"] == row["answer"].strip()[0]
        first_wins = row["label_logits"][0] > row["label_logits"][1]
        near_is_first = row["near_label"] == row["labels"][0]
        assert record["near_lo"] == (10.0 if first_wins == near_is_first else -10.0)
        assert {k: v for k, v in record.items() if k in row} == row


def test_eval_records_list_prompt_stays_a_message_list(tmp_path):
    records = eval_records(written_rows("multiturn_analyst", tmp_path))
    own = [r for r in records if r["variant"] != "twin"]   # twins are Alan's single-message template
    assert own and all(isinstance(r["prompt"], list) and r["prompt"][-1]["role"] == "user" for r in own)


def test_eval_records_without_near_label_raises(tmp_path):
    row = json.loads(written_rows("city_values", tmp_path).read_text())[0]
    del row["near_label"]
    path = tmp_path / "no_near_label.json"
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="near_label"):
        eval_records(path)


@pytest.mark.skipif(not REAL_RUN.exists(), reason="the eval-type run's results are not on this machine")
@pytest.mark.parametrize("type_id", ["city_values", "multiturn_analyst"])
def test_eval_records_on_the_real_run(type_id):
    records = eval_records(REAL_RUN / type_id / "rows.json")
    assert len(records) == 32
    assert all(r["label"] in ("a", "b") and np.isfinite(r["near_lo"]) for r in records)
    # E0's label is the side its near log-odds favour
    assert all((r["near_lo"] > 0) == (r["label"] == r["near_label"]) for r in records if not r["forced"])


def test_with_horizon_text_changes_only_the_phrase():
    prompt = "TASK: choose.\nCONSTRAINT: ... for this time horizon: 2 years.\nACTION: go. 2 years later."
    assert with_horizon_text(prompt, "2 years", "15 months") == \
        "TASK: choose.\nCONSTRAINT: ... for this time horizon: 15 months.\nACTION: go. 2 years later."
    messages = [{"role": "system", "content": "horizon: 2 years."}, {"role": "user", "content": "choose a) or b)"}]
    replaced = with_horizon_text(messages, "2 years", "3 days")
    assert replaced == [{"role": "system", "content": "horizon: 3 days."}, messages[1]]
    assert messages[0]["content"] == "horizon: 2 years."


@pytest.mark.parametrize("prompt", ["no phrase here", "horizon: 2 years and horizon: 2 years"])
def test_with_horizon_text_needs_exactly_one_phrase(prompt):
    with pytest.raises(ValueError, match="expected once"):
        with_horizon_text(prompt, "2 years", "3 days")


@pytest.mark.parametrize("type_id", DOMAIN_TYPES + EXTRA_TYPES)
def test_phrase_and_donor_of_every_row(type_id, tok, tmp_path):
    records = eval_records(written_rows(type_id, tmp_path))
    assert len(records) >= 32
    for record in records:
        positions = phrase_positions(tok, chat_text(tok, record["prompt"], CFG), record["horizon_text"])
        assert len(positions) >= 2, record["sample_uid"]
    rows, n_skipped = plan_rows(records, tok, CFG)
    assert len(rows) + n_skipped == len(records)
    assert len(rows) > n_skipped
    for row in rows:
        donor_text = row["donor_horizon_text"]
        own = phrase_positions(tok, chat_text(tok, row["prompt"], CFG), row["horizon_text"])
        donor = phrase_positions(tok, chat_text(tok, row["donor_prompt"], CFG), donor_text)
        assert donor == own
        assert row["donor_prompt"] != row["prompt"]
        assert with_horizon_text(row["donor_prompt"], donor_text, row["horizon_text"]) == row["prompt"]
        assert row["target_regime"] == {"near": "both", "both": "near", "none": "near"}[row["start_regime"]]
        assert row["target_label"] in row["labels"]


def test_donor_is_in_the_target_regime_nearest_the_target(tok, tmp_path):
    record = next(r for r in eval_records(written_rows("city_values", tmp_path)) if r["regime"] == "both")
    target_log_h = np.log10(np.sqrt(record["near_delay_years"] * record["far_delay_years"]))
    text, log_h, prompt = find_donor(record, target_log_h, "near", tok, CFG)
    assert regime(10 ** log_h, record["near_delay_years"], record["far_delay_years"]) == "near"
    assert prompt == with_horizon_text(record["prompt"], record["horizon_text"], text)
    assert abs(log_h - target_log_h) < 0.15   # the grid is 48 points over 4.2 decades


@pytest.mark.parametrize("type_id", ["city_values", "t0_w0"])
def test_select_rows_draws_half_near_half_both_deterministically(type_id, tmp_path):
    records = eval_records(written_rows(type_id, tmp_path))
    chosen = select_rows(records, 16)
    assert [r["regime"] for r in chosen].count("near") == 8 and [r["regime"] for r in chosen].count("both") == 8
    assert len(chosen) == 16
    assert [r["sample_uid"] for r in chosen] == [r["sample_uid"] for r in select_rows(records, 16)]
    assert select_rows(records, None) is records


def test_select_rows_takes_every_row_of_a_short_regime(tmp_path):
    records = eval_records(written_rows("city_values", tmp_path))
    few_near = [r for r in records if r["regime"] == "both"] + [r for r in records if r["regime"] == "near"][:3]
    chosen = select_rows(few_near, 16)
    assert [r["regime"] for r in chosen].count("near") == 3 and [r["regime"] for r in chosen].count("both") == 8


def summary_row(uid, target_regime, e0_label, label, near_lo, forced=False):
    target_label = "a" if target_regime == "near" else "b"
    return {"sample_uid": uid, "target_regime": target_regime, "target_label": target_label, "e0_label": e0_label,
            "label": label, "near_lo": near_lo, "forced": forced, "names_target": False}


def test_summarize_flips_and_shift_against_noop():
    noop = [summary_row("u0", "near", "b", "b", -2.0), summary_row("u1", "near", "b", "a", 1.0),
            summary_row("u2", "both", "a", "a", 3.0), summary_row("u3", "near", "a", "a", 4.0)]
    donor = [summary_row("u0", "near", "b", "a", 2.0), summary_row("u1", "near", "b", "a", 3.0),
             summary_row("u2", "both", "a", "b", -1.0), summary_row("u3", "near", "a", "a", 4.0, forced=True)]
    results = {"noop": {"rows": noop, "n_skipped_no_donor": 1}, "donor": {"rows": donor, "n_skipped_no_donor": 1}}
    out = summarize(results)
    assert out["noop"]["flipped_toward"] == pytest.approx(1 / 3)   # u0, u1 and u2 could flip; u1 did
    assert out["donor"]["flipped_toward"] == 1.0
    assert out["donor"]["flipped_toward_minus_noop"] == pytest.approx(2 / 3)
    # near log-odds up by 4, 2, 0 for near targets; for the both target down by 4, which is toward it
    assert out["donor"]["toward_log_odds_vs_noop"][0] == pytest.approx((4 + 2 + 4 + 0) / 4)
    assert out["donor"]["forced"] == 0.25
    assert "toward_log_odds_vs_noop" not in out["noop"]
    assert out["donor"]["n"] == 4 and out["donor"]["n_skipped_no_donor"] == 1


def fake_bank_cell(d):
    """A curve of three horizons in a 2-dimensional basis of the tiny model's hidden space."""
    rng = np.random.default_rng(0)
    return {"curve_logh": np.array([-2.0, 0.0, 3.0]), "curve_coords": rng.normal(size=(3, 2)),
            "basis": rng.normal(size=(2, d))}


def test_the_four_arms_run_end_to_end_on_a_tiny_model(tok, tiny, tmp_path, monkeypatch):  # noqa: F811
    run_dir = tmp_path / "evaltypes"
    written_rows("city_values", tmp_path)
    (run_dir / "city_values").mkdir(parents=True)
    (run_dir / "city_values" / "rows.json").write_text((tmp_path / "city_values.json").read_text())
    (run_dir / "settings.json").write_text(json.dumps({
        "model": CFG.model, "thinking": "on", "n_response": 6, "max_think_tokens": 3072, "prefill": "",
        "choice_prefix": "I choose:", "answer_window": 8, "layers": [10, 16]}))
    tok.padding_side = "left"
    monkeypatch.setattr(exp20, "load_model", lambda cfg: (tok, tiny, "cpu"))
    monkeypatch.setattr(exp20, "bank_cells", lambda *args: {"number": fake_bank_cell(32), "unit": fake_bank_cell(32)})
    exp20.main(argparse.Namespace(
        types="city_values", arms="random,effects,noop,donor", evaltypes_dir=str(run_dir), exp14_dir="unused",
        bank_prompts=None, rows_per_type=None, limit=3, model=None, max_think_tokens=2, batch_size=2,
        score_batch_size=2, out=str(tmp_path / "out")))
    type_dir = tmp_path / "out" / "exp20_Qwen3-8B_evaltype_transfer" / "city_values"
    results = {arm: json.loads((type_dir / "conditions" / f"{arm}.json").read_text()) for arm in exp20.ARMS}
    for result in results.values():
        assert len(result["rows"]) == 3
        assert all(np.isfinite(r["near_lo"]) for r in result["rows"])
    assert [r["sample_uid"] for r in results["donor"]["rows"]] == [r["sample_uid"] for r in results["noop"]["rows"]]
    summary = json.loads((type_dir / "summary.json").read_text())
    assert set(summary) == set(exp20.ARMS)
    assert "toward_log_odds_vs_noop" in summary["donor"] and "toward_log_odds_vs_noop" not in summary["noop"]
    # a finished (type, arm) is not rerun
    assert not exp20.run_claimed(type_dir, "noop", lambda: pytest.fail("rerun"))
