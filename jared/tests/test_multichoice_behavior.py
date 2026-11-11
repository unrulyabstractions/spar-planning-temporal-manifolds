"""multichoice_behavior on synthetic answers and logits with known values."""

import numpy as np
import pytest
from scipy.special import logsumexp

from spar_horizon.behavior import multichoice_behavior
from spar_horizon.full_bank import build_choice_prompts

RECORDS = build_choice_prompts(4, False, 0, 2, None)


def pick(rec, which):
    key = {"truth": rec["truth_label"],
           "wrong_half": next(o["label"] for o in rec["options"] if o["half"] != rec["truth"] and o["rank_in_half"] == 1),
           "second_in_half": next(o["label"] for o in rec["options"] if o["half"] == rec["truth"] and o["rank_in_half"] == 2),
           "worst_wrong_half": next(o["label"] for o in rec["options"] if o["half"] != rec["truth"] and o["rank_in_half"] == 2)}
    return f"{key[which]}) because"


def test_accuracy_per_regime_and_overall():
    recs = RECORDS[:60]
    answers = [pick(r, "truth" if i % 3 == 0 else "second_in_half" if i % 3 == 1 else "wrong_half")
               for i, r in enumerate(recs)]
    out = multichoice_behavior(recs, answers)
    n = len(recs)
    assert out["all"]["n"] == n
    assert out["all"]["accuracy"]["exact"] == [20, n]
    assert out["all"]["accuracy"]["half"] == [40, n]
    assert out["all"]["accuracy"]["within_half"] == [40, n]   # truth and wrong_half are both the best of the half chosen
    assert out["near"]["n"] + out["both"]["n"] == n
    assert out["near"]["accuracy"]["exact"][0] + out["both"]["accuracy"]["exact"][0] == 20


def test_unparsed_answers_are_left_out():
    recs = RECORDS[:4]
    out = multichoice_behavior(recs, [pick(recs[0], "truth"), None, "no label", ""])
    assert out["all"]["n"] == 1 and out["all"]["accuracy"]["exact"] == [1, 1]


def test_half_and_within_half_logodds_and_contrast():
    recs = RECORDS[:40]
    logits = np.zeros((len(recs), 4))
    for i, r in enumerate(recs):
        for lab in r["near_labels"]:
            logits[i, r["labels"].index(lab)] = 2.0
        logits[i, r["labels"].index(r["truth_label"])] += 1.0
    out = multichoice_behavior(recs, [pick(r, "truth") for r in recs], logits)
    expected_half = [logsumexp(logits[i][[r["labels"].index(l) for l in r["near_labels"]]])
                     - logsumexp(logits[i][[r["labels"].index(l) for l in r["far_labels"]]]) for i, r in enumerate(recs)]
    near_rows = [i for i, r in enumerate(recs) if r["regime"] == "near"]
    both_rows = [i for i, r in enumerate(recs) if r["regime"] == "both"]
    assert out["all"]["half_logodds"] == pytest.approx(np.mean(expected_half))
    assert out["C"] == pytest.approx(np.mean([expected_half[i] for i in near_rows]) - np.mean([expected_half[i] for i in both_rows]))
    assert out["C"] > 0


def test_truth_logodds_value():
    rec = RECORDS[0]
    logits = np.zeros((1, 4))
    logits[0, rec["labels"].index(rec["truth_label"])] = 3.0
    out = multichoice_behavior([rec], [pick(rec, "truth")], logits)
    assert out["all"]["truth_logodds"] == pytest.approx(3.0)


def test_pick_rate_per_position():
    recs = RECORDS[:20]
    answers = [f"{r['labels'][2]}) x" for r in recs]
    assert multichoice_behavior(recs, answers)["all"]["pick_rate"] == [0, 0, 1, 0]


def test_rendering_consistency():
    by_base = {}
    for r in RECORDS:
        by_base.setdefault(r["pair_id"], []).append(r)
    multi = [b for b in by_base.values() if len(b) >= 3][:10]
    recs = [r for b in multi for r in b]
    answers, logits = [], []
    for j, b in enumerate(multi):
        for m, r in enumerate(b):
            answers.append(pick(r, "truth" if (j < 6 or m > 0) else "wrong_half"))
            row = np.zeros(4)
            row[r["labels"].index(r["truth_label"])] = float(m) if j == 0 else 0.0
            logits.append(row)
    out = multichoice_behavior(recs, answers, np.array(logits))["all"]["consistency"]
    assert out["same_choice"] == [6, 10]    # bases 6-9 pick the wrong half in their first rendering only
    assert out["half_logodds_std"] > 0


def test_baselines():
    recs = RECORDS[:80]
    out = multichoice_behavior(recs, [pick(r, "truth") for r in recs])
    b = {name: out[name]["baselines"] for name in ("near", "both")}
    assert b["both"]["largest_payout"] == {"exact": 1.0, "half": 1.0, "within_half": 1.0}
    assert b["near"]["largest_payout"]["half"] == 0.0 and b["near"]["largest_payout"]["exact"] == 0.0
    assert b["near"]["earliest_then_largest"] == {"exact": 1.0, "half": 1.0, "within_half": 1.0}
    assert b["both"]["earliest_then_largest"]["half"] == 0.0
    assert b["near"]["uniform"] == {"exact": 0.25, "half": 0.5, "within_half": 0.5}


def test_accuracy_against_gap_bins():
    recs = RECORDS[:30]
    answers = [pick(r, "truth" if r["gap_in_half"] > 1.3 else "wrong_half") for r in recs]
    bins = multichoice_behavior(recs, answers)["all"]["gap_bins"]
    assert [b["exact"][1] for b in bins] == [10, 10, 10]
    assert bins[0]["gap_mean"] < bins[1]["gap_mean"] < bins[2]["gap_mean"]
    assert bins[0]["exact"][0] <= bins[1]["exact"][0] <= bins[2]["exact"][0]
