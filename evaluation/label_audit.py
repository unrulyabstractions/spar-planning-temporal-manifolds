"""Label audit of prompt banks: which confound labels each record carries, whether the labels agree with the rule,
and which records are debatable.

    python label_audit.py ../prompt_bank/data/t0_s0 ~/SPAR/results/horizon/evaltype_banks/* --out reports/label_audit

Writes `label_audit.json` and `label_audit.md` (a summary table, then one section per bank) into --out.
`audit_bank(records)` takes a bank's rows (dicts as in prompts.jsonl) and returns one bank's result, so the
diagnostics page can call it. Stdlib only apart from pbank, so it also runs on a box.

Three checks per record:

1. Presence. Each label is "present", "null" or "missing" (key absent). A null is only a gap where the record
   should carry the label: `phrasing_id` is expected when the text has a CONSTRAINT line in one of
   pbank.t0.CONSTRAINT_PHRASINGS; the eval-type keys (`type_id`, `variant`, `twin_id`) are expected in every
   eval-type record.
2. Recheck. Regime and correct option are recomputed from the record's own delays and horizon with pbank's exact
   arithmetic, and `correct_label` from labels / near_first / correct option.
3. Debatable. Exact ties (a payout arriving exactly at the horizon counts, but a model may read it either way),
   horizons within 1% of a delay, and unit rewrites.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompt_bank"))
from pbank.durations import Duration  # noqa: E402
from pbank.records import CORRECT_OPTION, read_jsonl, regime  # noqa: E402
from pbank.t0 import CONSTRAINT_PHRASINGS  # noqa: E402

LABELS = ("regime", "correct_option", "correct_label", "h_over_far_delay", "labels", "near_first", "phrasing_id",
          "rewritten", "reward_multiplier", "horizon_equals_a_delay", "pair_id")
EVALTYPE_KEYS = ("type_id", "variant", "twin_id")      # in `notes`; the family, the order/twin variant, the item
DOMAIN_KEYS = ("domain", "city_domain")
# Notes that identify one surface rendering of an eval-type item (a wording, a script, a unit pair, ...).
SURFACE_KEYS = ("phrasing_id", "update_phrasing_id", "lean_phrasing_id", "intro_id", "closing_id", "script_id",
                "unit_pair", "anchor_id", "rendered_horizon_text", "distractor_text", "stale_horizon_text")
NEAR_BOUNDARY = Fraction(1, 100)


def phrasing_regexes() -> list[re.Pattern]:
    """One pattern per CONSTRAINT phrasing, {H} as a wildcard. Not anchored to a line: a system message carries
    the phrasing without the "CONSTRAINT: " prefix, and some eval types put it mid-line."""
    out = []
    for template in CONSTRAINT_PHRASINGS:
        head, tail = template.split("{H}")
        out.append(re.compile(re.escape(head) + r"[^\n]+?" + re.escape(tail)))
    return out


PHRASING_REGEXES = phrasing_regexes()


def phrasing_in_text(text: str):
    """The id of the CONSTRAINT phrasing the text contains, or None."""
    for phrasing_id, pattern in enumerate(PHRASING_REGEXES):
        if pattern.search(text):
            return phrasing_id
    return None


def status(row: dict, key: str) -> str:
    if key not in row:
        return "missing"
    return "null" if row[key] is None else "present"


YEARS_PER_COARSE_UNIT = {"decade": 10, "century": 100, "centuries": 100}


def parse_duration(text: str) -> Duration:
    """pbank's Duration.parse, plus decades and centuries (eval type `unit_mismatch` writes them; the T0
    units stop at years)."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?) (decade|century|centuries)s?", text)
    if match:
        return Duration(Fraction(match.group(1)) * YEARS_PER_COARSE_UNIT[match.group(2)], "years")
    return Duration.parse(text)


def parse_durations(row: dict):
    """(horizon, near delay, far delay) as Durations, or None where a record has no horizon or a delay that
    is not written as a duration."""
    try:
        return tuple(parse_duration(text) for text in (row["horizon"], row["near"]["delay"], row["far"]["delay"]))
    except (TypeError, ValueError, KeyError):
        return None


def rule_regime_has_answer(row: dict) -> bool:
    """Regimes near_only and both name a correct option, except "both" without reward values to compare."""
    if row.get("regime") == "both" and None in (row["near"]["reward"], row["far"]["reward"]):
        return False
    return row.get("regime") in CORRECT_OPTION


def rule_labels(row: dict, durations):
    """(regime, correct option, correct label) from the decision rule alone. `durations` is None for the null
    control (no horizon), where the rule gives no_horizon."""
    if row.get("horizon") is None:
        return "no_horizon", None, None
    horizon, near_delay, far_delay = durations
    try:
        rule_regime = regime(horizon, near_delay, far_delay)
    except ValueError:
        return "far_pays_without_near", None, None
    correct = CORRECT_OPTION.get(rule_regime) if rule_regime_has_answer({**row, "regime": rule_regime}) else None
    if correct is None:
        return rule_regime, None, None
    if row.get("near_first") is None or row.get("labels") is None:
        return rule_regime, correct, None
    near_label = row["labels"][0] if row["near_first"] else row["labels"][1]
    far_label = row["labels"][1] if row["near_first"] else row["labels"][0]
    return rule_regime, correct, near_label if correct == "near" else far_label


def debatable_kinds(row: dict, durations) -> list[str]:
    """Why a record may be read either way: "tie", "near_boundary" (horizon within 1% of a delay, not a tie),
    "rewrite" (horizon written in a unit other than its natural one)."""
    kinds = []
    if durations is not None:
        horizon, near_delay, far_delay = durations
        for delay in (near_delay, far_delay):
            if horizon.days == delay.days:
                kinds.append("tie")
                break
        else:
            if any(abs(horizon.days / d.days - 1) <= NEAR_BOUNDARY for d in (near_delay, far_delay)):
                kinds.append("near_boundary")
    if row.get("rewritten"):
        kinds.append("rewrite")
    return kinds


def expected_labels(row: dict) -> dict[str, bool]:
    """Which labels the record should carry (see the module docstring). Gaps are reported only for these."""
    near_reward, far_reward = (row.get("near") or {}).get("reward"), (row.get("far") or {}).get("reward")
    return {
        "regime": True, "labels": True, "near_first": True, "rewritten": True, "horizon_equals_a_delay": True,
        "correct_option": rule_regime_has_answer(row),
        "correct_label": row.get("correct_option") is not None,
        "h_over_far_delay": row.get("horizon_years") is not None,
        "reward_multiplier": near_reward is not None and far_reward is not None,
        "phrasing_id": phrasing_in_text(row["text"]) is not None,
        "pair_id": row.get("tier") == "evaltype" or row.get("split") == "verification",
    }


def audit_bank(records: list[dict]) -> dict:
    """One bank's audit. Keys:
    n; tier (set of tiers); labels {label: {present, null, missing}}; evaltype_keys {key: counts} over the
    eval-type records, with domain/surface counted as present when any DOMAIN_KEYS / SURFACE_KEYS note exists;
    unset {label: [uids]} labels that are null or missing where expected; mismatches {check: [uids]};
    unparsed [uids] whose durations could not be parsed (no recheck, no tie test); debatable [{uid, kinds, ...}]."""
    counts = {label: {"present": 0, "null": 0, "missing": 0} for label in LABELS}
    notes_counts = {key: {"present": 0, "null": 0, "missing": 0} for key in EVALTYPE_KEYS + ("domain", "surface")}
    unset: dict[str, list] = {}
    mismatches = {name: [] for name in ("regime", "correct_option", "correct_label", "phrasing_id",
                                        "horizon_equals_a_delay")}
    unparsed, debatable = [], []

    for row in records:
        uid = row["uid"]
        for label in LABELS:
            counts[label][status(row, label)] += 1
        for label, expected in expected_labels(row).items():
            if expected and status(row, label) != "present":
                unset.setdefault(label, []).append(uid)

        if row.get("tier") == "evaltype":
            notes = row.get("notes") or {}
            for key in EVALTYPE_KEYS:
                notes_counts[key][status(notes, key)] += 1
                if status(notes, key) != "present":
                    unset.setdefault(f"notes.{key}", []).append(uid)
            notes_counts["domain"]["present" if any(k in notes for k in DOMAIN_KEYS) else "missing"] += 1
            notes_counts["surface"]["present" if any(k in notes for k in SURFACE_KEYS) else "missing"] += 1

        detected = phrasing_in_text(row["text"])
        if row.get("phrasing_id") is not None and row["phrasing_id"] != detected:
            mismatches["phrasing_id"].append(uid)

        durations = parse_durations(row)
        if row.get("horizon") is not None and durations is None:
            unparsed.append(uid)
            continue
        rule_regime, rule_correct, rule_label = rule_labels(row, durations)
        for name, expected in (("regime", rule_regime), ("correct_option", rule_correct),
                               ("correct_label", rule_label)):
            if row.get(name) != expected:
                mismatches[name].append(uid)
        if durations is not None:
            horizon, near_delay, far_delay = durations
            is_tie = horizon.days in (near_delay.days, far_delay.days)
            if row.get("horizon_equals_a_delay") != is_tie:
                mismatches["horizon_equals_a_delay"].append(uid)
        kinds = debatable_kinds(row, durations)
        if kinds:
            debatable.append({"uid": uid, "kinds": kinds, "horizon": row.get("horizon"),
                              "near_delay": row["near"]["delay"], "far_delay": row["far"]["delay"],
                              "regime": row.get("regime")})

    return {"n": len(records), "tier": sorted({row.get("tier") for row in records}), "labels": counts,
            "evaltype_keys": notes_counts if any(row.get("tier") == "evaltype" for row in records) else {},
            "unset": unset, "mismatches": mismatches, "unparsed": unparsed, "debatable": debatable}


# ---------------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------------

def table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def kind_count(result: dict, kind: str) -> int:
    return sum(kind in d["kinds"] for d in result["debatable"])


def render_markdown(audits: dict[str, dict]) -> str:
    lines = ["# Label audit", "",
             "`unset`: labels null or missing where the record should carry them. `mismatch`: records whose regime, "
             "correct option, correct label, phrasing id or tie flag disagree with the rule recomputed from their own "
             "delays and horizon. `debatable`: ties / horizon within 1% of a delay / unit rewrites.", ""]
    summary = [[name, a["n"], sum(len(u) for u in a["unset"].values()),
                sum(len(m) for m in a["mismatches"].values()), len(a["unparsed"]),
                kind_count(a, "tie"), kind_count(a, "near_boundary"), kind_count(a, "rewrite")]
               for name, a in audits.items()]
    lines += table(["bank", "n", "unset", "mismatch", "unparsed", "ties", "near boundary", "rewrites"], summary)
    for name, a in audits.items():
        lines += ["", f"## {name}", "", f"n = {a['n']}, tier {', '.join(map(str, a['tier']))}", ""]
        rows = [[label, *(c[s] for s in ("present", "null", "missing")), len(a["unset"].get(label, []))]
                for label, c in a["labels"].items()]
        rows += [[f"notes.{key}", *(c[s] for s in ("present", "null", "missing")), len(a["unset"].get(f"notes.{key}", []))]
                 for key, c in a["evaltype_keys"].items()]
        lines += table(["label", "present", "null", "missing", "unset"], rows)
        lines += ["", "Recheck mismatches: " + ", ".join(f"{k} {len(v)}" for k, v in a["mismatches"].items())]
        for check, uids in a["mismatches"].items():
            if uids:
                lines.append(f"- {check}: {', '.join(uids[:10])}{' ...' if len(uids) > 10 else ''}")
        if a["unparsed"]:
            lines.append(f"- durations not parseable, not rechecked: {len(a['unparsed'])} "
                         f"(first {', '.join(a['unparsed'][:5])})")
        listed = [d for d in a["debatable"] if d["kinds"] != ["rewrite"]]
        lines += ["", f"Debatable: {kind_count(a, 'tie')} ties, {kind_count(a, 'near_boundary')} near boundary, "
                      f"{kind_count(a, 'rewrite')} rewrites (all in the json; listed here: ties and near boundary)."]
        if listed:
            lines += [""] + table(["uid", "kinds", "horizon", "near delay", "far delay", "regime"],
                                  [[d["uid"], "+".join(d["kinds"]), d["horizon"], d["near_delay"], d["far_delay"],
                                    d["regime"]] for d in listed[:30]])
            if len(listed) > 30:
                lines.append(f"... and {len(listed) - 30} more")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("banks", nargs="+", type=Path, help="bank directories holding prompts.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    audits = {bank.name: audit_bank(read_jsonl(bank / "prompts.jsonl")) for bank in args.banks}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "label_audit.json").write_text(json.dumps(audits, indent=1) + "\n")
    (args.out / "label_audit.md").write_text(render_markdown(audits))
    print(render_markdown(audits).split("\n## ")[0])


if __name__ == "__main__":
    main()
