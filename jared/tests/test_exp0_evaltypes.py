"""Eval-type E0 runner on a tiny random-weight Qwen3 (no weights downloaded): every record gets a
row in record order, rows point at the right npz row, message prompts run, and done types are skipped."""

import json

import numpy as np
import pytest

from pipeline.s1_data_vectors.experiments import exp0_evaltypes as runner
from spar_horizon.config import Settings
from spar_horizon.eval_types.common import EvalType, alan_prompt, build_records, standard_item
from test_exp14_states import MODEL, tiny, tok  # noqa: F401


def plain_render(item, shown):
    return alan_prompt(shown, horizon_text=item.horizon_text)


def system_render(item, shown):
    return [{"role": "system", "content": f"The horizon is {item.horizon_text}."},
            {"role": "user", "content": alan_prompt(shown)}]


TEST_TYPES = {
    "t_plain": EvalType("t_plain", lambda rng, i, regime: standard_item(rng, regime), plain_render,
                        plan=("near", "both", "none")),
    "t_messages": EvalType("t_messages", lambda rng, i, regime: standard_item(rng, regime), system_render,
                           choice_prefix="Choice:", plan=("near", "both")),
}


@pytest.fixture
def setup(tok, tiny, monkeypatch):
    monkeypatch.setattr(runner, "load_model", lambda cfg: (tok, tiny, "cpu"))
    monkeypatch.setattr(runner, "build", lambda type_id, seed: build_records(TEST_TYPES[type_id], seed))
    return Settings(model=MODEL, thinking="on", bank="full", prefill="", choice_prefix="I choose:",
                    max_think_tokens=3, n_response=6, answer_window=8)


def test_rows_follow_records_and_point_at_their_npz_rows(setup, tmp_path):
    run_dir = runner.run(["t_plain", "t_messages"], setup, tmp_path, (10, 14, 99))
    for type_id in TEST_TYPES:
        records = build_records(TEST_TYPES[type_id], 0)
        rows = json.loads((run_dir / type_id / "rows.json").read_text())
        assert [r["sample_uid"] for r in rows] == [r["sample_uid"] for r in records]
        for row in rows:
            assert row["answer"] is not None
            z = np.load(run_dir / type_id / row["act_file"])
            assert z["layers"].tolist() == [10, 14]
            assert len(row["label_logits"]) == len(row["labels"])
            np.testing.assert_allclose(z["label_logits"][row["act_row"]], row["label_logits"], rtol=1e-6)
            assert row["response_text"].lstrip().startswith("</think>")   # a forced close adds "\n" first
    messages = json.loads((run_dir / "t_messages" / "rows.json").read_text())
    # base/swap records use the type's prefix, twins Alan's: two npz groups
    assert {r["act_file"] for r in messages} == {"acts_0.npz", "acts_1.npz"}
    assert {r["choice_prefix"] for r in messages if r["act_file"] == "acts_1.npz"} == {"I choose:"}


def test_done_types_are_skipped(setup, tmp_path, monkeypatch):
    runner.run(["t_plain"], setup, tmp_path, (10,))
    monkeypatch.setattr(runner, "extract", lambda *a, **k: pytest.fail("extract ran for a finished type"))
    runner.run(["t_plain"], setup, tmp_path, (10,))
