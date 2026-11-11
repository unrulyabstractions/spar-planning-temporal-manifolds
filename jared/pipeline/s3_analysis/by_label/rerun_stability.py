"""How stable is the thinking-on choice when the same prompt is run again? (post hoc, 2026-11-03)

E10's recipient condition re-ran 96 full/units-bank prompts unpatched, in a different
batch; bf16 batched greedy decoding is not bit-stable across batch composition (`01` §7),
so the rerun is a small perturbation of the same greedy trace. `08` §11 noted that 12 of
96 labels changed. Here: by regime, how many change, in which direction, against the E0
cache. Saved answers only.

    python -m pipeline.s3_analysis.by_label.rerun_stability /home/Jared/SPAR/results/horizon OUT_DIR
    -> OUT_DIR/rerun_stability.txt
"""
import json
import sys
from collections import Counter
from pathlib import Path

from scipy.stats import fisher_exact


def main(root, out_dir):
    root, out_dir = Path(root), Path(out_dir)
    original = {}
    for bank in ("full", "units"):
        for a in json.loads((root / f"exp0_Qwen3-8B_think-on_{bank}" / "answers.json").read_text()):
            original[a["sample_uid"]] = a
    rows = json.loads((root / "exp10_Qwen3-8B_reasoning" / "conditions" / "recipient.json").read_text())["rows"]
    counts, directions, seen = Counter(), Counter(), set()
    for r in rows:
        if r["sample_uid"] in seen:
            continue
        seen.add(r["sample_uid"])
        o = original[r["sample_uid"]]
        flipped = r["label"] != o["label"]
        close = "forced" if o["forced"] else "natural"
        counts[(r["regime"], close, flipped)] += 1
        if flipped:
            before = "near" if o["label"] == o["near_label"] else "far"
            after = "near" if r["label"] == r["near_label"] else "far"
            directions[(r["regime"], close, f"{before}→{after}")] += 1
    lines = ["E10 recipient rerun vs the E0 cache, 8B thinking on (same prompts, another batch)", ""]
    for regime in sorted({k[0] for k in counts}):
        for close in ("natural", "forced"):
            n_flip, n_same = counts[(regime, close, True)], counts[(regime, close, False)]
            if n_flip + n_same:
                lines.append(f"  {regime:5s} {close:8s} flipped {n_flip:2d} of {n_flip + n_same:2d}")
    both = [sum(counts[("both", c, f)] for c in ("natural", "forced")) for f in (True, False)]
    near = [sum(counts[("near", c, f)] for c in ("natural", "forced")) for f in (True, False)]
    p = fisher_exact([both, near])[1]
    lines.append(f"  both vs near regime, all closes: {both[0]}/{sum(both)} vs {near[0]}/{sum(near)}, Fisher p {p:.2g}")
    lines.append("  directions: " + ", ".join(f"{k[0]} {k[1]} {k[2]}: {v}" for k, v in sorted(directions.items())))
    text = "\n".join(lines)
    (out_dir / "rerun_stability.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:3])
