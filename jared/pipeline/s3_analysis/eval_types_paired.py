"""Paired comparison of eval types that share items (`EvalType.item_seed`): the v2 rewordings
against their originals, and the T0 wording ladder.

    python -m pipeline.s3_analysis.eval_types_paired $RESULTS/evaltypes_Qwen3-8B_think-on

Rows pair by (item index, variant): the same decision, options and order, in two wordings. Per pair
of types and regime: accuracy of each, the discordant counts (right only in A, right only in B) and
an exact two-sided McNemar p; force-closed share and median think length of each. For the ladder,
`both` accuracy is also split by the far option's implied yearly rate (`implied_rate`), since below
a plausible reinvestment rate the original wording's truth is disputable.
Writes `paired.md` into the run directory.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from pipeline.s3_analysis.eval_types_score import parse_label

PAIRS = [("t0_w0", "t0_w1_cash"), ("t0_w0", "t0_w3_clear"), ("t0_w1_cash", "t0_w3_clear"),
         ("roi_unequal", "roi_unequal_clear"), ("system_prompt", "system_prompt_clear"),
         ("sycophancy_lean", "sycophancy_lean_clear"), ("horizon_revised", "horizon_revised_clear")]
RATE_BANDS = ((0, 0.25), (0.25, 1.0), (1.0, float("inf")))


def load(run_dir, type_id):
    path = Path(run_dir) / type_id / "rows.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    return {(r["twin_id"].rsplit("_", 1)[-1], r["variant"]): r for r in rows}


def correct(r):
    return parse_label(r) in r["truth_labels"]


def mcnemar(only_a, only_b):
    n = only_a + only_b
    return 1.0 if n == 0 else binomtest(only_a, n, 0.5).pvalue


def compare(a, b, keys):
    keys = [k for k in keys if a[k]["truth_labels"]]
    ra = [correct(a[k]) for k in keys]
    rb = [correct(b[k]) for k in keys]
    only_a = sum(x and not y for x, y in zip(ra, rb))
    only_b = sum(y and not x for x, y in zip(ra, rb))
    return {"n": len(keys), "acc_a": sum(ra), "acc_b": sum(rb), "only_a": only_a, "only_b": only_b,
            "p": mcnemar(only_a, only_b)}


def compare_types(run_dir, type_a, type_b):
    a, b = load(run_dir, type_a), load(run_dir, type_b)
    if a is None or b is None:
        return None
    keys = [k for k in a if k in b and k[1] in ("base", "swap")]
    out = {"regimes": {g: compare(a, b, [k for k in keys if a[k]["regime"] == g])
                       for g in sorted({a[k]["regime"] for k in keys})},
           "all": compare(a, b, keys),
           "forced": (sum(a[k]["forced"] for k in keys), sum(b[k]["forced"] for k in keys), len(keys)),
           "think_median": (int(np.median([a[k]["think_len"] for k in keys])),
                            int(np.median([b[k]["think_len"] for k in keys])))}
    if "implied_rate" in a[keys[0]]:
        both = [k for k in keys if a[k]["regime"] == "both"]
        out["both_by_rate"] = {f"{lo:g}-{hi:g}": compare(a, b, [k for k in both if lo <= a[k]["implied_rate"] < hi])
                               for lo, hi in RATE_BANDS}
    return out


def cell(c):
    if not c or c["n"] == 0:
        return "-"
    return f"{c['acc_a']}/{c['n']} -> {c['acc_b']}/{c['n']} (+{c['only_b']} -{c['only_a']}, p={c['p']:.2g})"


def markdown(results):
    lines = ["| A -> B | all | near | both | forced A / B | think median A / B |", "|---|---|---|---|---|---|"]
    for (ta, tb), r in results.items():
        lines.append(f"| {ta} -> {tb} | {cell(r['all'])} | {cell(r['regimes'].get('near'))} | "
                     f"{cell(r['regimes'].get('both'))} | {r['forced'][0]} / {r['forced'][1]} of {r['forced'][2]} | "
                     f"{r['think_median'][0]} / {r['think_median'][1]} |")
    rate_lines = [f"| {ta} -> {tb} | " + " | ".join(cell(c) for c in r["both_by_rate"].values()) + " |"
                  for (ta, tb), r in results.items() if "both_by_rate" in r]
    if rate_lines:
        lines += ["", "`both` rows by the far option's implied yearly rate:", "",
                  "| A -> B | " + " | ".join(f"rate {lo:g}-{hi:g}" for lo, hi in RATE_BANDS) + " |",
                  "|---|" + "---|" * len(RATE_BANDS)] + rate_lines
    return "\n".join(lines) + "\n"


def main(run_dir):
    results = {}
    for ta, tb in PAIRS:
        r = compare_types(run_dir, ta, tb)
        if r is not None:
            results[(ta, tb)] = r
    text = ("Paired accuracy, A -> B: correct in A -> correct in B (+right only in B, -right only in A, exact "
            "McNemar p).\n\n" + markdown(results))
    (Path(run_dir) / "paired.md").write_text(text)
    print(text)
    return results


if __name__ == "__main__":
    main(sys.argv[1])
