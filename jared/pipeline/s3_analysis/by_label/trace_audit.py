"""What the unsteered reasoning says, and how it lines up with the choice (post hoc, 2026-11-03)

Saved traces only (answers.json of the thinking-on E0 runs; no model). For each prompt
with a horizon:
  stated horizon   the first duration the trace states as its horizon
                   (exp14_analysis.first_stated_horizon), its shift from the prompt's
                   horizon in decades, and the regime it puts the delays in.
  converts         the trace restates the horizon in another unit (written_unit).
  themes           regex flags for the arguments the trace makes (THEMES below). They
                   are written by the reasoning, so they are post-treatment: they
                   describe which arguments come with which choice, not what causes it.
Reported by model, bank, regime and choice, natural closes and all; in the regimes
where the choice varies, each theme's share among near and far choosers with a Fisher
exact p (two-sided, uncorrected: read as descriptive).

    python -m pipeline.s3_analysis.by_label.trace_audit /home/Jared/SPAR/results/horizon OUT_DIR
    -> OUT_DIR/trace_audit.{json,txt}
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

from pipeline.s3_analysis.by_label.exp14_analysis import CONFLICT
from spar_horizon.durations import MOVED, first_stated_horizon, horizon_units
from spar_horizon.e0_runs import run_records
from spar_horizon.full_bank import regime as regime_of
from spar_horizon.horizons import canonical_unit

RUNS = (("Qwen3-8B", "full"), ("Qwen3-8B", "units"), ("Qwen3-8B", "fill"), ("Qwen3-14B", "full"))
HORIZON = r"(?:\S+[- ]\w+ )?(?:time[- ])?horizon"
THEMES = {
    "within horizon": rf"\b(?:within|inside|before the end of) (?:the|this|that|our|a) {HORIZON}",
    "beyond horizon": rf"\b(?:beyond|outside|exceeds?|exceeding|after|longer than|past) (?:the|this|that|our|a) {HORIZON}",
    "end of horizon": r"\b(?:end|edge|close|limit) of the (?:\S+[- ]\w+ )?(?:time[- ])?(?:horizon|period)\b",
    "time left to use": r"\b(?:time (?:left )?to (?:use|enjoy|spend|invest|benefit)|remaining time|no time (?:left )?to|"
                        r"(?:use|enjoy|spend) (?:it|the money|the funds) (?:within|during))",
    "risk": r"\b(?:risk|risky|uncertain|uncertainty|guarantee[ds]?)\b",
    "inflation": r"\binflation",
    "time value": r"\b(?:time value|present value|discount(?:ing|ed)?)\b",
    "reinvest": r"\b(?:reinvest\w*|compound\w*|interest rate|rate of return|annual return)",
    "immediacy": r"\b(?:immediate(?:ly)?|liquidity|right away|access to (?:the )?(?:money|funds|cash))\b",
    "bigger amount": r"\b(?:larger|bigger|greater|higher|much more|significantly more) (?:amount|sum|payout|reward|return|value)",
    "conflict": CONFLICT.pattern,
}
THEME_RE = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in THEMES.items()}
VARYING = ("none", "both")
# none regime: the trace questions the horizon (broader than the "conflict" theme)
QUESTIONS_HORIZON = re.compile(r"typo|mistake|inconsistent|misprint|doesn't make sense|does not make sense", re.IGNORECASE)
H_NEAR_EDGES = (-3, -2, -1)            # log10(horizon / near delay) bins for the none-regime gradient


def trace_rows(root, model, bank):
    run = f"exp0_{model}_think-on_{bank}"
    records = run_records(root, run)
    rows = []
    for a in json.loads((root / run / "answers.json").read_text()):
        if a.get("horizon") is None or "think_len" not in a or a.get("anchor"):
            continue
        record = records[a["sample_uid"]]
        text = a["think_text"]
        stated = first_stated_horizon(text)
        row = {"model": model, "bank": bank, "regime": a["regime"], "natural": not a["forced"],
               "chose": "near" if a["label"] == a["near_label"] else "far", "think_len": a["think_len"],
               "converts": any(u != canonical_unit(record["horizon_text"].split()[1])
                               for u in horizon_units(text, a["horizon"])),
               "stated": stated is not None}
        if stated is not None:
            years = stated[0]
            row["stated_shift"] = float(np.log10(years / a["horizon"]))
            row["stated_regime"] = regime_of(years, record["near_delay_years"], record["far_delay_years"])
        row.update({f"theme {name}": bool(pattern.search(text)) for name, pattern in THEME_RE.items()})
        row["questions horizon"] = bool(QUESTIONS_HORIZON.search(text))
        row["log h/near"] = float(np.log10(a["horizon"] / record["near_delay_years"]))
        rows.append(row)
    return rows


def share(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def stated_summary(rows):
    stated = [r for r in rows if r["stated"]]
    moved = [r for r in stated if abs(r["stated_shift"]) > MOVED]
    wrong_regime = [r for r in stated if r["stated_regime"] != r["regime"]]
    return {"n": len(rows), "states a horizon": len(stated) / max(len(rows), 1),
            "moved > 0.05 dec": len(moved), "stated regime ≠ prompt regime": len(wrong_regime),
            "examples": [(r["regime"], r["stated_regime"], round(r["stated_shift"], 2), r["chose"]) for r in wrong_regime[:8]]}


def theme_table(rows):
    """Per theme: share among near choosers, among far choosers, Fisher p."""
    near = [r for r in rows if r["chose"] == "near"]
    far = [r for r in rows if r["chose"] == "far"]
    out = {}
    for name in THEMES:
        key = f"theme {name}"
        a, b = sum(r[key] for r in near), sum(r[key] for r in far)
        p = fisher_exact([[a, len(near) - a], [b, len(far) - b]])[1] if near and far else float("nan")
        out[name] = {"near": a / max(len(near), 1), "far": b / max(len(far), 1), "p": float(p)}
    return {"n near": len(near), "n far": len(far), "themes": out}


def none_gradient(rows):
    """None regime, natural closes: by log10(horizon / near delay) bin, the share of traces that
    question the horizon, and the near share with and without that flag."""
    rows = [r for r in rows if r["regime"] == "none" and r["natural"]]
    bins = np.digitize([r["log h/near"] for r in rows], H_NEAR_EDGES)
    out = {}
    for b in np.unique(bins):
        group = [r for r, k in zip(rows, bins) if k == b]
        flagged = [r for r in group if r["questions horizon"]]
        unflagged = [r for r in group if not r["questions horizon"]]
        near = lambda rs: float(np.mean([r["chose"] == "near" for r in rs])) if rs else float("nan")  # noqa: E731
        out[int(b)] = {"n": len(group), "questions horizon": len(flagged) / len(group), "near": near(group),
                       "near | questions": near(flagged), "near | not": near(unflagged)}
    return out


def analyse(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], "old banks" if r["bank"] != "fill" else "fill")].append(r)
    result = {}
    for (model, banks), group in groups.items():
        natural = [r for r in group if r["natural"]]
        entry = {"stated, natural closes": stated_summary(natural), "stated, all": stated_summary(group),
                 "converts (natural)": share(natural, "converts"), "by regime": {}}
        for regime in ("none", "near", "both"):
            in_regime = [r for r in natural if r["regime"] == regime]
            if not in_regime:
                continue
            entry["by regime"][regime] = {
                "n": len(in_regime), "near share": share([{"x": r["chose"] == "near"} for r in in_regime], "x"),
                "median think_len near/far": [float(np.median([r["think_len"] for r in in_regime if r["chose"] == c] or [np.nan]))
                                              for c in ("near", "far")],
                "themes": theme_table(in_regime) if regime in VARYING else
                {name: share(in_regime, f"theme {name}") for name in THEMES}}
        if banks == "old banks":
            entry["none gradient"] = none_gradient(natural)
        result[f"{model} {banks}"] = entry
    return result


def summary_lines(result):
    lines = ["Trace audit (thinking on, saved traces). Natural closes unless marked.", ""]
    for name, entry in result.items():
        s = entry["stated, natural closes"]
        lines.append(f"{name}: n {s['n']}; states a horizon {s['states a horizon']:.2f}; stated moved > 0.05 decades "
                     f"{s['moved > 0.05 dec']}; stated regime ≠ prompt regime {s['stated regime ≠ prompt regime']}; "
                     f"restates the horizon in another unit {entry['converts (natural)']:.2f}")
        if s["examples"]:
            lines.append(f"  wrong-regime examples (prompt, stated, shift, chose): {s['examples']}")
        sa = entry["stated, all"]
        lines.append(f"  all closes: n {sa['n']}, stated regime ≠ prompt regime {sa['stated regime ≠ prompt regime']}")
        for regime, r in entry["by regime"].items():
            lines.append(f"  regime {regime}: n {r['n']}, near share {r['near share']:.2f}, "
                         f"median think length near/far {r['median think_len near/far'][0]:.0f}/{r['median think_len near/far'][1]:.0f}")
            if regime in VARYING:
                t = r["themes"]
                lines.append(f"    theme share among near ({t['n near']}) vs far ({t['n far']}) choosers, Fisher p:")
                for theme, v in t["themes"].items():
                    flag = " *" if v["p"] < 0.01 else ""
                    lines.append(f"      {theme:18s} {v['near']:.2f} vs {v['far']:.2f}  p {v['p']:.3g}{flag}")
        if "none gradient" in entry:
            lines.append(f"  none regime by log10(horizon / near delay), bins at {H_NEAR_EDGES}:")
            for b, g in entry["none gradient"].items():
                lines.append(f"    bin {b}: n {g['n']:3d}  questions the horizon {g['questions horizon']:.2f}  near {g['near']:.2f}"
                             f"  (near | questions {g['near | questions']:.2f}, near | not {g['near | not']:.2f})")
        lines.append("")
    return lines


def main(root, out_dir):
    root, out_dir = Path(root), Path(out_dir)
    rows = [row for model, bank in RUNS for row in trace_rows(root, model, bank)]
    result = analyse(rows)
    (out_dir / "trace_audit.json").write_text(json.dumps(result, indent=1))
    (out_dir / "trace_audit_rows.json").write_text(json.dumps(rows))
    text = "\n".join(summary_lines(result))
    (out_dir / "trace_audit.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:3])
