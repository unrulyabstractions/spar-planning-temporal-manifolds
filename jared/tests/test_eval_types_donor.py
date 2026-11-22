"""E20 per-type table on hand-made summaries: distance lookup, ordering, arms, correlations."""

import json

from pipeline.s3_analysis.eval_types_donor import distance_correlations, markdown, table_rows


def arm(flipped_over_noop, log_odds, forced=0.0):
    return {"n": 10, "n_skipped_no_donor": 2, "flipped_toward": 0.5, "flipped_toward_minus_noop": flipped_over_noop,
            "toward_log_odds_vs_noop": log_odds, "forced": forced}


def distance_file(medians):
    return {"rank_cell": "L18_suffix_end",
            "types": {t: {"distance": {"L18_suffix_end": {"horizon": {"paired": {"median": m}}}}} for t, m in medians.items()}}


def write_summary(root, type_id, summary):
    (root / type_id).mkdir()
    (root / type_id / "summary.json").write_text(json.dumps(summary))


def test_rows_are_ordered_by_distance_and_skip_types_without_noop(tmp_path):
    noop = {"n": 10, "n_skipped_no_donor": 2, "flipped_toward": 0.1, "forced": 0.2}
    write_summary(tmp_path, "far", {"noop": noop, "donor": arm(0.2, [5.0, 1.0, 9.0])})
    write_summary(tmp_path, "near", {"noop": noop, "donor": arm(0.8, [20.0, 15.0, 25.0]),
                                     "effects": arm(0.0, [1.0, -1.0, 3.0])})
    write_summary(tmp_path, "unfinished", {"donor": arm(0.5, [9.0, 5.0, 13.0])})
    rows = table_rows(tmp_path, distance_file({"far": 9.0, "near": 1.0, "unfinished": 4.0}))
    assert [r["type_id"] for r in rows] == ["near", "far"]
    assert rows[0]["effects"]["flipped_over_noop"] == 0.0 and "effects" not in rows[1]
    assert rows[0]["noop_flipped"] == 0.1 and rows[0]["no_donor"] == 2
    correlations = distance_correlations(rows)
    assert abs(correlations["flipped_over_noop"]["rho"] + 1.0) < 1e-9 and correlations["log_odds"]["n"] == 2
    text = markdown(rows, correlations)
    assert "| near | 1.00 | 10 (2) | 0.10 | +0.80 | +20.0 [+15.0, +25.0] | +0.00 |" in text
    assert "| far |" in text and "- | -" in text
