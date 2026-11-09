"""Which unit does the reasoning write the horizon in? (16_spec_pristine_manifold.md §3, step 0)

On the units bank the prompt states the horizon in one rendering ("36 months"), but
the reasoning may restate it in another ("36 months, which is three years") and then
reason in that unit. After the think block the answer-slot code may carry the written
unit rather than the prompt's. This is a side view for P1 (the fit does not use it).

A horizon mention is any "<number> <unit>" in the think text whose duration is
within 10% of the prompt's horizon (the units bank's renderings are within 10% of
each other), with numbers as digits or words ("three-year", "fifty-nine years",
"a year"). Prompts with a delay within 10% of the horizon are marked ambiguous: a
mention there cannot be told apart from the delay.

Per prompt: first, last and most frequent written unit, and whether any mention uses
a unit other than the rendered one ("converts").

    python -m pipeline.s3_analysis.by_datatype.written_unit /home/Jared/SPAR/results/horizon
    -> <root>/analysis/written_unit.json, a table on stdout

Reads saved answers only; runs no model.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from spar_horizon.durations import TOLERANCE, close_to, durations, horizon_units, parse_number  # noqa: F401
from spar_horizon.e0_runs import read_answers, run_records
from spar_horizon.horizons import UNIT_YEARS, canonical_unit

RUN = "exp0_Qwen3-8B_think-on_units"


def summarize(units, rendered_unit):
    """First, last and most frequent written unit, and whether any differs from the rendered one."""
    if not units:
        return {"n_mentions": 0, "first": None, "last": None, "main": None, "converts": False}
    counts = Counter(units)
    main = max(counts, key=lambda u: (counts[u], -units.index(u)))   # ties: the earlier one
    return {"n_mentions": len(units), "first": units[0], "last": units[-1], "main": main,
            "converts": any(u != rendered_unit for u in units)}


def rate(rows, key):
    return sum(bool(r[key]) for r in rows) / len(rows) if rows else float("nan")


def main(root):
    root = Path(root)
    answers = [a for a in read_answers(root / RUN) if "think_text" in a]
    records = run_records(root, RUN)
    rows = []
    for a in answers:
        rec = records[a["sample_uid"]]
        h = rec["horizon"]
        ambiguous = close_to(rec["near_delay_years"], h) or close_to(rec["far_delay_years"], h)
        summary = summarize(horizon_units(a["think_text"], h), canonical_unit(a["rendered_unit"]))
        rows.append({"sample_uid": a["sample_uid"], "pair_id": a["pair_id"], "duration_id": a["duration_id"],
                     "rendered_unit": canonical_unit(a["rendered_unit"]), "natural": a["natural"],
                     "forced": a["forced"], "ambiguous": ambiguous, **summary})

    clear = [r for r in rows if not r["ambiguous"]]
    found = [r for r in clear if r["n_mentions"]]
    print(f"{len(rows)} thinking-on prompts; {len(rows) - len(clear)} ambiguous (a delay within 10% of the horizon); "
          f"{len(found)} of {len(clear)} mention the horizon")
    print(f"{'rendering':10s} {'unit':8s} {'n':>4s} {'converts':>9s} {'last = rendered':>16s} {'main = rendered':>16s}")
    table = {}
    for natural in (True, False):
        for unit in sorted({r["rendered_unit"] for r in found}, key=lambda u: UNIT_YEARS[u]):
            group = [r for r in found if r["natural"] == natural and r["rendered_unit"] == unit]
            if not group:
                continue
            same_last = sum(r["last"] == unit for r in group) / len(group)
            same_main = sum(r["main"] == unit for r in group) / len(group)
            table[f"{'natural' if natural else 'other'}|{unit}"] = {
                "n": len(group), "converts": rate(group, "converts"), "last_same": same_last, "main_same": same_main,
                "main_units": dict(Counter(r["main"] for r in group))}
            print(f"{'natural' if natural else 'other':10s} {unit:8s} {len(group):4d} {rate(group, 'converts'):9.2f} "
                  f"{same_last:16.2f} {same_main:16.2f}   main: {dict(Counter(r['main'] for r in group))}")
    pairs = {}
    for r in found:
        pairs.setdefault(r["pair_id"], []).append(r["main"])
    same_main_in_pair = [len(set(units)) == 1 for units in pairs.values() if len(units) > 1]
    print(f"pairs whose renderings are reasoned about in the same main unit: "
          f"{sum(same_main_in_pair)} of {len(same_main_in_pair)}")

    out = root / "analysis" / "written_unit.json"
    out.write_text(json.dumps({"tolerance": TOLERANCE, "summary": table,
                               "pairs_same_main": [sum(same_main_in_pair), len(same_main_in_pair)],
                               "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
