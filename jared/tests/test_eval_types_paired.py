"""Paired comparison on hand-answered ladder rows: discordant counts, regimes, rate bands."""

import json

from pipeline.s3_analysis.eval_types_paired import compare_types
from spar_horizon.eval_types import build


def answered(rec, label, forced=False):
    return {**rec, "answer": f"{label})", "forced": forced, "think_len": 100, "think_text": "",
            "response_text": f"</think>\n\nI choose: {label}). My reasoning: x"}


def write(tmp_path, type_id, rows):
    (tmp_path / type_id).mkdir()
    (tmp_path / type_id / "rows.json").write_text(json.dumps(rows))


def test_compare_types(tmp_path):
    w0, w3 = build("t0_w0"), build("t0_w3_clear")
    wrong = lambda r: next(l for l in r["labels"] if l not in r["truth_labels"])
    # A: always the near option; B: always right
    write(tmp_path, "t0_w0", [answered(r, r["near_label"]) for r in w0])
    write(tmp_path, "t0_w3_clear", [answered(r, (r["truth_labels"] or [wrong(r)])[0], forced=True) for r in w3])
    res = compare_types(tmp_path, "t0_w0", "t0_w3_clear")
    near, both = res["regimes"]["near"], res["regimes"]["both"]
    assert (near["n"], near["acc_a"], near["acc_b"], near["only_a"], near["only_b"]) == (16, 16, 16, 0, 0)
    assert (both["n"], both["acc_a"], both["acc_b"], both["only_b"]) == (28, 0, 28, 28)
    assert both["p"] < 1e-6 and near["p"] == 1.0
    assert "none" not in res["regimes"] or res["regimes"]["none"]["n"] == 0
    assert sum(c["n"] for c in res["both_by_rate"].values()) == 28
    assert res["forced"] == (0, 48, 48)      # over all 40 base + 8 swap rows, `none` included
    assert compare_types(tmp_path, "t0_w0", "t0_w1_cash") is None
