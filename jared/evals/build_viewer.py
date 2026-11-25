"""Embed every samples/*.jsonl record into viewer.html (one self-contained page for review).

Usage: python3 build_viewer.py   (from jared/evals/; reads viewer_template.html, writes viewer.html)

The page also shows three automatic flags per record. They implement proposals that are not yet
approved, so they are warnings for review, not checks that fail a bank.
"""

import json
from pathlib import Path

from review_samples import best_within

HERE = Path(__file__).parent
RISK_RANK = {"low": 0, "medium": 1, "high": 2}
LOCK_WORDS = ("cannot", "locked", "lock-up", "not be withdrawn", "not be redeemed", "no early")

FAMILY_GROUPS = [
    ("Explicit horizon (T1)", [
        ("inv_t1_2opt", "Free-text prompt, horizon stated, 2 options, risk crossed with duration."),
        ("inv_t1_k_equal10", "4 or 8 options, all compounding at 10%/yr from one principal (structured template)."),
        ("inv_t1_k_equal5", "Same at 5%/yr."),
        ("inv_t1_k_unequal", "4 or 8 options, unequal returns, no rate stated (structured template)."),
    ]),
    ("Implicit horizon (T2, T3)", [
        ("inv_t2_implicit_2opt", "Horizon implied by a life event from the anchor table, 2 options."),
        ("inv_t2_implicit_k", "Horizon implied, 4 or 8 options."),
        ("inv_t3_open", "Open-ended allocation or bond duration; readout is a number from the answer."),
    ]),
    ("Role, format and domain", [
        ("inv_role_hedgefund", "System prompt makes the model a fund manager or finance professional."),
        ("inv_json", "The question and the answer are JSON."),
        ("city_valued", "City projects with costs, benefits and completion times."),
        ("city_unvalued", "Quick project vs long-term project, no values; no correct answer, E[log h] only."),
    ]),
    ("Robustness", [
        ("distractor", "An irrelevant duration in the prompt that would change the answer if used as the horizon."),
        ("multiturn_cba", "An earlier assistant turn holds a cost-benefit table and a recommendation; then choose."),
        ("multiturn_recheck", "The model's earlier answer (wrong in one twin, right in the other), then \"Are you sure?\"."),
    ]),
]


def horizon_not_decisive(rec):
    """True when no other horizon would make a different option correct.

    The best option only changes when H crosses an option's duration, so trying each duration
    covers every horizon. An answer of None (nothing pays in time) does not count as a change.
    """
    if rec["truth_rule"] != "max_value_within_H" or not rec["options"]:
        return False
    answers = {best_within(rec["options"], option["h_years"]) for option in rec["options"]}
    return not (answers - {rec["truth_label"], None})


def risk_favors_truth(rec):
    """True when the correct option is strictly the safest, so a risk-averse model passes without reading H."""
    options = rec["options"]
    if not rec["truth_label"] or not options or any(o.get("risk") not in RISK_RANK for o in options):
        return False
    truth = next(o for o in options if o["label"] == rec["truth_label"])
    others = [o for o in options if o is not truth]
    return all(RISK_RANK[truth["risk"]] < RISK_RANK[o["risk"]] for o in others)


def unlocked_late_options(rec):
    """Labels of investment options paying after H whose text never says they cannot be cashed out early.

    The structured template ("X dollars in N years") is exempt: there the payout date is the only way to get the money.
    """
    horizon = rec["horizon_years"]
    if rec["domain"] != "investment" or rec["surface"] == "structured" or horizon is None:
        return []
    return [o["label"] for o in rec["options"]
            if o["h_years"] > horizon and not any(word in o["text"].lower() for word in LOCK_WORDS)]


def main():
    families = []
    for group, members in FAMILY_GROUPS:
        for family, description in members:
            path = HERE / "samples" / f"{family}.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for rec in records:
                rec["flags"] = {
                    "horizon_not_decisive": horizon_not_decisive(rec),
                    "risk_favors_truth": risk_favors_truth(rec),
                    "unlocked_late_options": unlocked_late_options(rec),
                }
            families.append({"family": family, "group": group, "description": description, "records": records})
    listed = {f["family"] for f in families}
    unlisted = sorted(p.stem for p in (HERE / "samples").glob("*.jsonl") if p.stem not in listed)
    assert not unlisted, f"add these families to FAMILY_GROUPS: {unlisted}"
    # Escape "</" so prompt text can never close the <script> tag that holds the data.
    data = json.dumps(families).replace("</", "<\\/")
    page = (HERE / "viewer_template.html").read_text().replace("/*SAMPLES_JSON*/[]", data)
    (HERE / "viewer.html").write_text(page)
    count = sum(len(f["records"]) for f in families)
    print(f"wrote viewer.html: {len(families)} families, {count} records")


if __name__ == "__main__":
    main()
