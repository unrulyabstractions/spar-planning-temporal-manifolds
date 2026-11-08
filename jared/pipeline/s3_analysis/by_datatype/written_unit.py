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

import dataclasses
import json
import re
import sys
from collections import Counter
from pathlib import Path

from spar_horizon.banks import build_records
from spar_horizon.config import Settings
from spar_horizon.horizons import UNIT_YEARS, canonical_unit

TOLERANCE = 0.10
RUN = "exp0_Qwen3-8B_think-on_units"

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen").split()
TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
WORD_VALUES = {**{w: i for i, w in enumerate(ONES)}, **{w: 20 + 10 * i for i, w in enumerate(TENS)},
               "a": 1, "an": 1, "hundred": 100, "thousand": 1000}

# a number (digits, a word, or a hyphenated compound like "fifty-nine"), then a time unit,
# joined by spaces or a hyphen ("three-year")
DURATION = re.compile(r"\b(?P<number>\d[\d,]*(?:\.\d+)?|[a-z]+(?:-[a-z]+)?)[\s-]+"
                      r"(?P<unit>seconds?|minutes?|hours?|days?|weeks?|months?|years?|decades?|century|centuries)\b",
                      re.IGNORECASE)


def parse_number(text):
    """The value of a digit string or an English number word ("fifty-nine"), or None."""
    text = text.lower().replace(",", "")
    if text[0].isdigit():
        return float(text)
    if text in WORD_VALUES:
        return float(WORD_VALUES[text])
    tens, _, ones = text.partition("-")
    if tens in TENS and ones in ONES[1:10]:
        return float(WORD_VALUES[tens] + WORD_VALUES[ones])
    return None


def durations(text):
    """Every "<number> <unit>" in `text` as (years, canonical unit), in order."""
    out = []
    for match in DURATION.finditer(text):
        value = parse_number(match["number"])
        if value is not None:
            unit = canonical_unit(match["unit"])
            out.append((value * UNIT_YEARS[unit], unit))
    return out


def close_to(years, target, tolerance=TOLERANCE):
    return abs(years / target - 1) <= tolerance


def horizon_units(text, horizon_years):
    """The units of the mentions in `text` whose duration matches the horizon, in order."""
    return [unit for years, unit in durations(text) if close_to(years, horizon_years)]


def summarize(units, rendered_unit):
    """First, last and most frequent written unit, and whether any differs from the rendered one."""
    if not units:
        return {"n_mentions": 0, "first": None, "last": None, "main": None, "converts": False}
    counts = Counter(units)
    main = max(counts, key=lambda u: (counts[u], -units.index(u)))   # ties: the earlier one
    return {"n_mentions": len(units), "first": units[0], "last": units[-1], "main": main,
            "converts": any(u != rendered_unit for u in units)}


def run_records(root, run=RUN):
    """The records of an E0 run (the units-bank thinking-on run by default), by sample_uid."""
    manifest = json.loads((root / run / "manifest.json").read_text())
    known = {f.name for f in dataclasses.fields(Settings)}
    cfg = Settings(**{k: v for k, v in manifest["settings"].items() if k in known})
    return {r["sample_uid"]: r for r in build_records(cfg)}


def rate(rows, key):
    return sum(bool(r[key]) for r in rows) / len(rows) if rows else float("nan")


def main(root):
    root = Path(root)
    answers = [a for a in json.loads((root / RUN / "answers.json").read_text()) if "think_text" in a]
    records = run_records(root)
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
