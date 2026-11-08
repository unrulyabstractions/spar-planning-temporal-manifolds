"""Arithmetic in the both-regime reasoning and the choice (post hoc, 2026-11-03)

Reads the agent-coded traces in OUT_DIR (both_b8a/b_coded.json: every natural-close
both-regime trace of the 8B on the full and units banks, coded by two Sonnet agents with
one codebook; the 14B batch was stopped by a safeguard and is not coded). Each trace has
`computes` in {correct, error, overridden, none}: a correct numeric comparison of the two
options, one with an error that flips it, a correct one the trace then contradicts, or
no numbers. Codes are an agent's judgement of free text, checked here only by agreement
with an independent earlier coding of 32 of the same traces (trace_coding.json).

Reports: computes × final choice with Wilson 95% CIs on the near share; the same split by
the break-even rate r* (breakeven_rate.py); and agreement with the earlier coding.

    python -m pipeline.s3_analysis.by_label.trace_coding_summary /home/Jared/SPAR/results/horizon OUT_DIR
    -> OUT_DIR/trace_coding_summary.txt
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline.s3_analysis.by_label.breakeven_rate import log10_breakeven
from pipeline.s3_analysis.by_datatype.written_unit import run_records

BATCHES = ("both_b8a", "both_b8b")
CODES = ("correct", "error", "overridden", "none")
# the earlier open coding (trace_coding.json) -> did it judge a correct worked comparison as deciding?
EARLIER_CORRECT = {"REINVEST_CALC_CORRECT"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def load(root, out_dir):
    records = {}
    for bank in ("full", "units"):
        records.update({f"8B|{bank}|{uid}": r for uid, r in run_records(root, f"exp0_Qwen3-8B_think-on_{bank}").items()})
    rows = []
    for batch in BATCHES:
        for t in json.loads((out_dir / f"{batch}_coded.json").read_text())["traces"]:
            r = records[t["key"]]
            ratio = r["rewards"][1] / r["rewards"][0]
            t["log_rate"] = log10_breakeven(ratio, r["far_delay_years"] - r["near_delay_years"])
            rows.append(t)
    return rows


def table(rows, label):
    lines = [f"{label}: n {len(rows)}"]
    for code in CODES:
        group = [t for t in rows if t["computes"] == code]
        near = sum(t["final_choice"] == "near" for t in group)
        lo, hi = wilson(near, len(group))
        lines.append(f"  {code:10s} n {len(group):3d}  near {near:3d}  far {len(group) - near:3d}  "
                     f"near share {near / max(len(group), 1):.2f} [{lo:.2f}, {hi:.2f}]")
    return lines


def agreement(rows, out_dir):
    """Earlier coding (sample of 32 both-regime traces) vs this one, on 'a correct comparison decides'."""
    earlier = json.loads((out_dir / "trace_coding.json").read_text())["traces"]
    sample = {json.loads(line)["id"]: json.loads(line) for line in (out_dir / "trace_sample.jsonl").read_text().splitlines()}
    now = {t["key"]: t for t in rows}
    pairs = []
    for t in earlier:
        if t["regime"] != "both":
            continue
        s = sample[t["id"]]
        key = f"8B|{s['bank']}|{s['sample_uid']}"
        if key in now:
            pairs.append((t["primary_reason"] in EARLIER_CORRECT, now[key]["computes"] == "correct"))
    counts = Counter(pairs)
    agree = counts[(True, True)] + counts[(False, False)]
    return [f"agreement with the earlier coding on 'correct comparison' (earlier, now): {dict(counts)}; "
            f"{agree} of {len(pairs)} agree"]


def main(root, out_dir):
    root, out_dir = Path(root), Path(out_dir)
    rows = load(root, out_dir)
    lines = table(rows, "Qwen3-8B both regime, natural closes, full + units")
    lines.append("")
    for name, keep in (("r* < 10%/yr", lambda t: t["log_rate"] < -1), ("r* ≥ 10%/yr", lambda t: t["log_rate"] >= -1)):
        lines += table([t for t in rows if keep(t)], f"  by break-even rate, {name}")
    lines.append("")
    correct_near = [t for t in rows if t["computes"] == "correct" and t["final_choice"] == "near"]
    lines.append("correct comparison but near: favoured_when_recomputed = "
                 + str(Counter(t["favoured_when_recomputed"] for t in correct_near))
                 + ", log10 r* " + ", ".join(f"{t['log_rate']:.2f}" for t in correct_near))
    mismatch = sum(t["favoured_when_recomputed"] not in (None, t["final_choice"]) for t in rows)
    lines.append(f"final choice ≠ the trace's own comparison redone correctly: {mismatch} of "
                 f"{sum(t['favoured_when_recomputed'] is not None for t in rows)} traces with a comparison")
    lines += agreement(rows, out_dir)
    text = "\n".join(lines)
    (out_dir / "trace_coding_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:3])
