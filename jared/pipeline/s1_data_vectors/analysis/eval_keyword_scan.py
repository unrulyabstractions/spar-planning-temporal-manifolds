"""Q2 keyword scan (EvalsRelatedQualities.md §4): how often does the model's own reasoning talk about
being tested? Runs the broad KEYWORDS regex and the strict TEST_AWARENESS regex over the cached think text of the T0 full bank, every eval type
and, when present, a related-qualities run.

    python -m pipeline.s1_data_vectors.analysis.eval_keyword_scan /workspace/results/horizon --out <dir>

Reads, under the results/horizon directory given:
  exp0_<model>_think-on_full/answers.json     T0 full bank (think_text per prompt)
  evaltypes_<model>_think-on/<type>/rows.json eval types
  related_<model>_think-on/<type>/rows.json   related-qualities types (same layout)
Writes eval_keyword_scan.{json,md} into --out. The md lists random TEST_AWARENESS hit snippets with an
empty "true positive?" column to fill by hand, so the strict share can be corrected for precision.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from spar_horizon.eval_awareness import TEST_AWARENESS, keyword_hits

N_SNIPPETS = 40
T0_MODEL = "Qwen3-8B"


def trace_sources(results_dir):
    """{source name: [think texts]} for the T0 full bank and every type under evaltypes_ / related_ runs."""
    root = Path(results_dir)
    sources = {}
    for path in sorted(root.glob(f"exp0_{T0_MODEL}_think-on_full/answers.json")):
        sources["t0_full"] = [row["think_text"] for row in json.loads(path.read_text()) if row.get("think_text")]
    for prefix in ("evaltypes", "related"):
        for run_dir in sorted(root.glob(f"{prefix}_*_think-on")):
            for rows_path in sorted(run_dir.glob("*/rows.json")):
                rows = json.loads(rows_path.read_text())
                sources[f"{prefix}/{rows_path.parent.name}"] = [r["think_text"] for r in rows if r.get("think_text")]
    return sources


def scan_source(texts):
    """Counts and every hit of one source's traces."""
    traces_with_hit = traces_with_strict_hit = 0
    words, strict_hits = Counter(), []
    for index, text in enumerate(texts):
        found = keyword_hits(text)
        strict = keyword_hits(text, pattern=TEST_AWARENESS)
        traces_with_hit += bool(found)
        traces_with_strict_hit += bool(strict)
        words.update(match.lower() for match, _ in found)
        strict_hits += [{"trace": index, "match": match, "snippet": snippet} for match, snippet in strict]
    n = len(texts)
    return {"n_traces": n, "n_with_hit": traces_with_hit, "share_with_hit": traces_with_hit / n if n else 0.0,
            "n_with_strict_hit": traces_with_strict_hit,
            "share_with_strict_hit": traces_with_strict_hit / n if n else 0.0,
            "top_words": words.most_common(8), "hits": strict_hits}


def scan(results_dir, seed=0):
    sources = {name: scan_source(texts) for name, texts in trace_sources(results_dir).items()}
    pool = [(name, hit) for name, result in sources.items() for hit in result["hits"]]
    sampled = random.Random(seed).sample(pool, min(N_SNIPPETS, len(pool)))
    return {"sources": {name: {k: v for k, v in result.items() if k != "hits"} for name, result in sources.items()},
            "snippets": [{"source": name, **hit} for name, hit in sampled]}


def markdown(report):
    lines = ["# Eval-awareness keyword scan", "",
             "Share of reasoning traces with at least one match: `strict` = TEST_AWARENESS (the prompt "
             "itself called a test, quiz or trick question), `broad` = KEYWORDS, mostly \"evaluate\" in the sense "
             "of weighing options (`spar_horizon/eval_awareness.py`).", "",
             "| source | traces | strict | broad | top broad words |", "|---|---|---|---|---|"]
    for name, r in report["sources"].items():
        words = ", ".join(f"{w} ({n})" for w, n in r["top_words"][:5])
        lines.append(f"| {name} | {r['n_traces']} | {r['share_with_strict_hit']:.1%} | {r['share_with_hit']:.1%} "
                     f"| {words} |")
    lines += ["", f"## {len(report['snippets'])} random strict hits to check by hand", "",
              "| source | match | snippet | true positive? |", "|---|---|---|---|"]
    for s in report["snippets"]:
        lines.append(f"| {s['source']} | {s['match']} | {s['snippet'].replace('|', '/')} |  |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results_dir", help="the results/horizon directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = scan(args.results_dir, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_keyword_scan.json").write_text(json.dumps(report, indent=1))
    (out / "eval_keyword_scan.md").write_text(markdown(report))
    print(markdown(report).split("## ")[0])
    print(f"wrote {out / 'eval_keyword_scan.json'} and .md")


if __name__ == "__main__":
    main()
