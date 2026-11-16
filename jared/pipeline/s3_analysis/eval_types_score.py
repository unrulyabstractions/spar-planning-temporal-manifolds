"""Score an eval-type run (exp0_evaltypes): parsing, coherence and baseline accuracy per type.

    python -m pipeline.s3_analysis.eval_types_score $RESULTS/evaltypes_Qwen3-8B_think-on

Reads `<type>/rows.json` for every finished type and writes `summary.json` and `summary.md` into the
run directory. Reads saved files only.

The choice is parsed from the text after the think block (`response_text`), not from the runner's
label token: the runner falls back to the first letter token when the prefix is missing, which can
look like a valid label. A row parses when its type's prefix is there and a valid label (or, for
free-form terms, a duration) follows it.

Per type:
- parse: kept rows, parsed rows, force-closed rows, median think length
- accuracy: share of parsed rows whose label is in `truth_labels`, over rows that have a truth,
  for base + swap rows by regime, and for the explicit twins
- near_rate by regime (two-option rows): share choosing the earlier option; mean near log-odds
  (logit(near label) - logit(far label) at the label slot)
- chance_accuracy: what a uniform random pick scores on the same rows
- accuracy on natural and on forced closes separately; share choosing the last-listed option
- order agreement: share of swap rows that chose the same option (by text) as their base row, and
  share of swap pairs where both picks are correct (differs when truth is a set)
- horizon_in_think: share of explicit-horizon base rows whose think text states the horizon string
- type extras: stale-horizon following (horizon_revised), lean following (sycophancy_lean),
  term error (free_form), expected log delay vs log horizon (k-option types)
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from spar_horizon.horizons import Horizon

DURATION = re.compile(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)")


def after_think(row):
    text = row.get("response_text") or ""
    return text.split("</think>", 1)[-1]


def parse_label(row):
    """The chosen label, or None when the prefix or a valid label is missing."""
    text = after_think(row)
    if row["choice_prefix"] == '"choice":':
        m = re.search(r'"choice"\s*:\s*"\s*([A-Za-z])\b', text)
        return m.group(1).lower() if m and m.group(1).lower() in row["labels"] else None
    k = text.find(row["choice_prefix"])
    if k < 0:
        return None
    m = re.match(r"[\s*_`<>()\[\]]*([A-Za-z])\)?(?![A-Za-z])", text[k + len(row["choice_prefix"]):])
    return m.group(1).lower() if m and m.group(1).lower() in row["labels"] else None


def parse_term_years(row):
    text = after_think(row)
    k = text.find(row["choice_prefix"])
    if k < 0:
        return None
    m = DURATION.search(text[k + len(row["choice_prefix"]):k + len(row["choice_prefix"]) + 40])
    if not m:
        return None
    try:
        return Horizon(float(m.group(1)), m.group(2)).years
    except ValueError:
        return None


def share(flags):
    flags = [bool(f) for f in flags]
    return {"k": int(sum(flags)), "n": len(flags), "p": round(float(np.mean(flags)), 3) if flags else None}


def option_text(row, label):
    return next(o["text"] for o in row["options"] if o["label"] == label)


def near_log_odds(row):
    labels, logits = row["labels"], row.get("label_logits")
    if not logits or "near_label" not in row or len(labels) != 2:
        return None
    near = labels.index(row["near_label"])
    return logits[near] - logits[1 - near]


def expected_log_delay(row):
    """sum_i p_i log10(delay_i) under the softmax of the label logits (k-option rows)."""
    logits = np.asarray(row.get("label_logits") or [], float)
    if len(logits) != len(row["labels"]) or len(logits) < 3:
        return None
    p = np.exp(logits - logits.max())
    p /= p.sum()
    delay = {o["label"]: o["delay_years"] for o in row["options"]}
    return float(sum(pi * math.log10(delay[lab]) for pi, lab in zip(p, row["labels"])))


def score_type(rows):
    kept = [r for r in rows if r.get("answer") is not None]
    for r in kept:
        r["parsed"] = parse_term_years(r) if r["answer_kind"] == "duration" else parse_label(r)
    parsed = [r for r in kept if r["parsed"] is not None]
    out = {"n": len(rows), "kept": len(kept), "parsed": share(r["parsed"] is not None for r in kept),
           "forced": share(r["forced"] for r in kept),
           "think_len_median": int(np.median([r["think_len"] for r in kept])) if kept else None}
    labelled = [r for r in parsed if r["answer_kind"] == "label"]
    main = [r for r in labelled if r["variant"] in ("base", "swap") and r["truth_labels"]]
    twins = [r for r in labelled if r["variant"] == "twin"]
    out["accuracy"] = share(r["parsed"] in r["truth_labels"] for r in main)
    out["accuracy_by_regime"] = {g: share(r["parsed"] in r["truth_labels"] for r in main if r["regime"] == g)
                                 for g in sorted({r["regime"] for r in main})}
    out["twin_accuracy"] = share(r["parsed"] in r["truth_labels"] for r in twins)
    out["chance_accuracy"] = round(float(np.mean([len(r["truth_labels"]) / len(r["labels"]) for r in main])), 3) if main else None
    # forced closes answer after a cut-off trace; their choices are often positional
    out["accuracy_natural_close"] = share(r["parsed"] in r["truth_labels"] for r in main if not r["forced"])
    out["accuracy_forced_close"] = share(r["parsed"] in r["truth_labels"] for r in main if r["forced"])
    out["chose_last_listed"] = share(r["parsed"] == r["labels"][-1] for r in labelled if r["variant"] in ("base", "swap"))
    two = [r for r in labelled if r["variant"] in ("base", "swap") and "near_label" in r]
    if two:
        out["near_rate_by_regime"] = {g: share(r["parsed"] == r["near_label"] for r in two if r["regime"] == g)
                                      for g in sorted({r["regime"] for r in two})}
        lo = {}
        for r in two:
            v = near_log_odds(r)
            if v is not None:
                lo.setdefault(r["regime"], []).append(v)
        out["near_log_odds_by_regime"] = {g: round(float(np.mean(v)), 2) for g, v in sorted(lo.items())}
    base = {r["twin_id"]: r for r in labelled if r["variant"] == "base"}
    pairs = [(base[r["twin_id"]], r) for r in labelled if r["variant"] == "swap" and r["twin_id"] in base]
    out["order_agreement"] = share(option_text(b, b["parsed"]) == option_text(s, s["parsed"]) for b, s in pairs)
    # with set-valued truth two different picks can both be right
    out["order_both_correct"] = share(b["parsed"] in b["truth_labels"] and s["parsed"] in s["truth_labels"]
                                      for b, s in pairs if b["truth_labels"])
    explicit = [r for r in kept if r["variant"] == "base" and r["horizon_text"]
                and r["horizon_text"] in (r["prompt"] if isinstance(r["prompt"], str)
                                          else " ".join(m["content"] for m in r["prompt"]))]
    if explicit:
        out["horizon_in_think"] = share(r["horizon_text"] in (r.get("think_text") or "") for r in explicit)
    out.update(type_extras(rows[0]["type_id"] if rows else "", parsed, labelled))
    return out


def type_extras(type_id, parsed, labelled):
    extra = {}
    if type_id == "horizon_revised":
        flips = [r for r in labelled if r["variant"] in ("base", "swap") and r["truth_labels"]]
        # the stale horizon's truth: the other option when the stale regime is near/both and differs
        extra["chose_stale_truth"] = share(r["parsed"] not in r["truth_labels"] for r in flips
                                           if r.get("stale_regime") in ("near", "both"))
    if type_id == "sycophancy_lean":
        rows = [r for r in labelled if r["variant"] in ("base", "swap")]
        lean = lambda r: next(o["label"] for o in r["options"] if o["kind"] == r["lean_kind"])
        extra["chose_lean"] = share(r["parsed"] == lean(r) for r in rows)
        extra["chose_lean_when_wrong"] = share(r["parsed"] == lean(r) for r in rows if r["truth_labels"])
    if type_id == "free_form":
        terms = [(r["parsed"], r["truth_term_years"]) for r in parsed]
        if terms:
            err = [math.log10(t / h) for t, h in terms]
            extra["term"] = {"n": len(terms), "median_abs_log10_error": round(float(np.median(np.abs(err))), 3),
                             "share_within_horizon": share(t <= h * 1.0001 for t, h in terms),
                             "share_within_factor_1.5": share(abs(e) <= math.log10(1.5) for e in err),
                             "spearman_log_term_vs_log_horizon": round(float(spearmanr(
                                 [math.log10(t) for t, _ in terms], [math.log10(h) for _, h in terms])[0]), 3)
                             if len(terms) > 2 else None}
    k_rows = [r for r in labelled if len(r["labels"]) > 2 and r["variant"] == "base" and r["horizon"]]
    pairs = [(expected_log_delay(r), math.log10(r["horizon"])) for r in k_rows]
    pairs = [p for p in pairs if p[0] is not None]
    if len(pairs) > 2:
        extra["spearman_expected_log_delay_vs_log_horizon"] = round(float(spearmanr(*zip(*pairs))[0]), 3)
    return extra


def fmt(s):
    return "-" if not s or s.get("n", 0) == 0 else f"{s['p']:.2f} ({s['k']}/{s['n']})"


def markdown(summary):
    head = ("| type | kept | parsed | forced | think med | acc | chance | acc natural | acc near | acc both | twin acc | "
            "order agree | last listed |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for t, s in summary.items():
        by = s["accuracy_by_regime"]
        lines.append(f"| {t} | {s['kept']}/{s['n']} | {fmt(s['parsed'])} | {fmt(s['forced'])} | {s['think_len_median']} | "
                     f"{fmt(s['accuracy'])} | {s['chance_accuracy']} | {fmt(s['accuracy_natural_close'])} | {fmt(by.get('near'))} | "
                     f"{fmt(by.get('both'))} | {fmt(s['twin_accuracy'])} | {fmt(s['order_agreement'])} | "
                     f"{fmt(s['chose_last_listed'])} |")
    return head + "\n".join(lines) + "\n"


def main(run_dir):
    run_dir = Path(run_dir)
    summary = {}
    for rows_path in sorted(run_dir.glob("*/rows.json")):
        summary[rows_path.parent.name] = score_type(json.loads(rows_path.read_text()))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "summary.md").write_text(markdown(summary))
    print(markdown(summary))
    return summary


if __name__ == "__main__":
    main(sys.argv[1])
