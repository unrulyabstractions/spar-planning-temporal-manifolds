"""E22 prompting baseline, no model: where the instruction lands in the rendered prompt, the rows each mode
and arm runs on, and the analysis on synthetic results with planted effects."""

import dataclasses
import json

import numpy as np
import pytest
from transformers import AutoTokenizer

from pipeline.s1_data_vectors.analysis.evaltypes_steering_prepare import PREFIX, off_sequence
from pipeline.s2_steering.experiments.prompting.exp22_prompting_baseline import (
    ARMS, CROSSING_ARMS, INSTRUCTIONS, plan, with_instruction)
from pipeline.s3_analysis.eval_types_prompting import analyse, load_prompting, markdown
from spar_horizon.config import Settings
from spar_horizon.eval_types import build
from spar_horizon.model_io import chat_text

TYPES = ("system_prompt", "prose_natural", "coding_agent")
CFG_ON = Settings(model="Qwen/Qwen3-8B", thinking="on", prefill="", choice_prefix=PREFIX, answer_window=200)
CFG_OFF = dataclasses.replace(CFG_ON, thinking="off", prefill=PREFIX, choice_prefix="")


@pytest.fixture(scope="module")
def tok():
    try:
        return AutoTokenizer.from_pretrained(CFG_ON.model, local_files_only=True)
    except OSError:
        pytest.skip("Qwen3-8B tokenizer is not cached locally")


def test_instruction_is_the_last_line_of_a_single_message():
    prompt = build("prose_natural")[0]["prompt"]
    edited = with_instruction(prompt, "long")
    assert edited == f"{prompt}\n{INSTRUCTIONS['long']}"
    assert edited.splitlines()[-1] == INSTRUCTIONS["long"]
    assert with_instruction(prompt, "none") == prompt


def test_instruction_is_the_last_line_of_the_last_user_turn():
    prompt = build("system_prompt")[0]["prompt"]
    edited = with_instruction(prompt, "shift_short")
    assert [m["role"] for m in edited] == [m["role"] for m in prompt] == ["system", "user"]
    assert edited[0] == prompt[0]
    assert edited[-1]["content"] == f"{prompt[-1]['content']}\n{INSTRUCTIONS['shift_short']}"
    assert prompt[-1]["content"].splitlines()[-1].startswith("FORMAT:")      # the input is not modified


@pytest.mark.parametrize("type_id", TYPES)
def test_rendered_prompt_ends_with_the_sentence_before_the_assistant_turn(type_id, tok):
    record = build(type_id)[0]
    format_line = record["prompt"][-1]["content"] if isinstance(record["prompt"], list) else record["prompt"]
    format_line = format_line.splitlines()[-1]
    for cfg, tail in ((CFG_ON, "<|im_end|>\n<|im_start|>assistant\n"), (CFG_OFF, "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" + PREFIX)):
        text = chat_text(tok, with_instruction(record["prompt"], "neutral"), cfg)
        assert text.endswith(f"{format_line}\n{INSTRUCTIONS['neutral']}{tail}")
        assert text.count(INSTRUCTIONS["neutral"]) == 1
    ids = off_sequence(tok, {**record, "prompt": with_instruction(record["prompt"], "neutral")}, CFG_OFF)
    assert tok.decode([ids[-1]]) == ":"


@pytest.mark.parametrize("type_id", TYPES)
def test_plans(type_id):
    rows = build(type_id)
    off = plan(rows, "off")
    assert list(off) == ["none", *ARMS] and all(len(v) == 32 for v in off.values())
    on = plan(rows, "on")
    assert len(on["neutral"]) == 24 and sum(map(len, on.values())) == 72
    for r in on["neutral"]:
        assert r["regime"] in ("near", "both") and r["variant"] in ("base", "swap")
        assert [a for a in ARMS if r in on[a]] == ["neutral", *CROSSING_ARMS[r["regime"]]]


LO = {"near": 4.0, "both": -4.0, "none": 0.0}      # neutral near log-odds: contrast 8


def write_arms(root, mode, type_id, values):
    """Results as the runner writes them; `values(row, arm)` is a row's near log-odds under the arm."""
    for arm, rows in plan(build(type_id), mode).items():
        folder = root / type_id / mode / "conditions"
        folder.mkdir(parents=True, exist_ok=True)
        out = [{"sample_uid": r["sample_uid"], "type_id": type_id, "arm": arm, "regime": r["regime"],
                "variant": r["variant"], "twin_id": r["twin_id"], "near_label": r["near_label"],
                "near_lo": values(r, arm)} for r in rows]
        (folder / f"{arm}.json").write_text(json.dumps({"arm": arm, "rows": out}))


def planted(row, arm):
    """Disposition arms move crossing rows by 6 logits toward the instructed side and every other row by
    1 (2 on none rows); shift arms move crossing rows to the other option and leave the rest alone."""
    base, regime = LO[row["regime"]], row["regime"]
    sign = {"long": -1, "shift_long": -1, "short": 1, "shift_short": 1}.get(arm)
    if sign is None:
        return base
    crosses = arm in CROSSING_ARMS.get(regime, ())
    if arm in ("long", "short"):
        return base + sign * (6.0 if crosses else 2.0 if regime == "none" else 1.0)
    return base + sign * 8.0 if crosses else base


def steering_results(type_id):
    """E21-style rows: the curve pushes crossing rows by 4 logits and leaves the others."""
    rows = []
    for r in build(type_id):
        if r["regime"] == "none":
            continue
        base = LO[r["regime"]]
        edit = {"near": 4.0, "both": -4.0}[r["regime"]]
        rows.append({"sample_uid": r["sample_uid"], "base": base,
                     "L24 curve +2": base - edit if r["regime"] == "near" else base,
                     "L24 curve -2": base - edit if r["regime"] == "both" else base,
                     "L22+24+26+28 curve +2": base, "L22+24+26+28 curve -2": base})
    return rows


def test_analysis_recovers_planted_effects(tmp_path):
    type_id = "system_prompt"
    for mode in ("off", "on"):
        write_arms(tmp_path, mode, type_id, lambda r, arm: planted(r, arm) if arm != "none" else LO[r["regime"]] + 1)
    e21 = {"modes": {"off": {"rows": steering_results(type_id)}, "on": {"rows": steering_results(type_id)}}}
    e20 = tmp_path / "e20"
    (e20 / type_id).mkdir(parents=True)
    (e20 / type_id / "summary.json").write_text(json.dumps(
        {"donor": {"n": 15, "flipped_toward": 0.9, "toward_log_odds_vs_noop": [4.0, 2.0, 6.0]}}))

    out = analyse(load_prompting(tmp_path), e21, e20)

    off, on = out["modes"]["off"][type_id], out["modes"]["on"][type_id]
    assert off["baseline"]["contrast"] == 8.0 and off["baseline"]["gate"]
    disposition, shift = off["methods"]["prompt disposition"], off["methods"]["prompt shift"]
    assert disposition["toward"]["mean"] == pytest.approx(6 / 8, abs=1e-3)
    assert shift["toward"]["mean"] == pytest.approx(1.0)
    assert disposition["flips"] == {"n": 24, "k": 24, "rate": 1.0} == shift["flips"]
    # 12 near rows (short, 1), 12 both rows (long, 1), 4 none rows x 2 arms x 2 logits, in units of 8
    assert disposition["spillover"]["mean"] == pytest.approx((12 + 12 + 16) / 32 / 8, abs=1e-3)
    assert shift["spillover"]["mean"] == 0.0
    curve = off["methods"]["steering L24 curve ±2"]
    assert curve["toward"]["mean"] == pytest.approx(0.5) and curve["spillover"]["mean"] == 0.0
    assert curve["flips"]["n"] == 24 and off["methods"]["steering L22+24+26+28 curve ±2"]["toward"]["mean"] == 0.0

    assert on["methods"]["prompt disposition"]["toward"]["mean"] == pytest.approx(6 / 8, abs=1e-3)
    assert "spillover" not in on["methods"]["prompt shift"] and "spillover" not in on["methods"]["steering L24 curve ±2"]
    assert on["donor"] == {"n": 15, "flipped_toward": 0.9, "toward": [0.5, 0.25, 0.75]}
    assert "prompt disposition" in markdown(out)


def test_analysis_without_steering_or_donor(tmp_path):
    write_arms(tmp_path, "off", "coding_agent", planted)
    out = analyse(load_prompting(tmp_path))
    assert list(out["modes"]["off"]["coding_agent"]["methods"]) == ["prompt disposition", "prompt shift"]
    assert out["modes"]["on"] == {}
    assert np.isfinite(out["modes"]["off"]["coding_agent"]["methods"]["prompt shift"]["toward"]["mean"])
