"""Is the arithmetic shortcut set before reasoning? (post hoc, 2026-11-03)

trace_coding_summary.py: on the 8B's both-regime prompts where far wins at any normal
rate (break-even r* ≥ 10%/yr), the choice marks the reasoning almost exactly: a correct
comparison went far 45 of 45 times, an erroneous, overridden or absent one went near 30
of 32 times. So on these rows "chose near" ≈ "the trace will not carry the arithmetic
through". decision_emergence.py pooled the fill bank, where the surface separates near
from far by itself; here the rows are only the old-bank both prompts, so the surface has
little to go on.

Readers (decision_emergence's): surface metadata, prompt TF-IDF, activations (top-64 PCs),
activations + surface; out of fold, AUC with base-prompt bootstrap CIs. Two row sets:
r* ≥ 10%/yr (n ≈ 77, the clean one) and all old-bank both prompts (n 105).

    python analysis/heuristic_disposition.py /home/Jared/SPAR/results/horizon OUT_DIR
    -> OUT_DIR/heuristic_disposition.{json,txt}
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import decision_emergence as de  # noqa: E402
from breakeven_rate import log10_breakeven  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
CELLS = [(p, layer) for p in ("suffix newline", "suffix last") for layer in (8, 12, 16, 20, 24, 28, 36)]


def main(root, out_dir):
    root, out_dir = Path(root), Path(out_dir)
    rng = np.random.default_rng(0)
    de.RUNS["Qwen3-8B"] = ("full", "units")          # old banks only: the fill would let the surface win alone
    parts, data = de.load(root, "Qwen3-8B")
    rates = []
    for prompt, ratio_log, near_log, far_log in zip(data["prompt"], *(data["features"][:, de.FEATURES.index(f)].T
                                                                     for f in ("log ratio", "log near delay", "log far delay"))):
        rates.append(log10_breakeven(10 ** ratio_log, 10 ** far_log - 10 ** near_log))
    rates = np.array(rates)
    base = (data["regime"] == "both") & data["natural_close"]
    row_sets = {"r* ≥ 10%/yr": np.flatnonzero(base & (rates >= -1)), "all old-bank both": np.flatnonzero(base)}
    result = {}
    for name, rows in row_sets.items():
        y, groups = data["chose_near"][rows].astype(int), data["pairs"][rows]
        surface = de.surface_matrix(data, rows)
        baseline = {"surface": de.out_of_fold(de.fixed_features(surface), y, groups),
                    "text": de.out_of_fold(de.text_features(data["prompt"][rows]), y, groups)}
        best = max(baseline, key=lambda k: de.roc_auc_score(y, baseline[k]))
        entry = {"n": int(len(rows)), "near share": float(y.mean()),
                 "surface": de.auc_ci(y, baseline["surface"], groups, rng), "text": de.auc_ci(y, baseline["text"], groups, rng),
                 "cells": {}}
        for position, layer in CELLS:
            X = de.cell(parts, position, layer)[rows]
            acts = de.out_of_fold(de.pca_features(X), y, groups)
            entry["cells"][f"{position} L{layer}"] = {"acts": de.auc_ci(y, acts, groups, rng),
                                                      "acts - best surface": de.paired_gap(y, acts, baseline[best], groups, rng)}
        result[name] = entry
    (out_dir / "heuristic_disposition.json").write_text(json.dumps(result, indent=1))
    lines = ["Chose near within the 8B's old-bank both prompts (natural closes): before reasoning, beyond the surface?", ""]
    for name, e in result.items():
        lines.append(f"{name}: n {e['n']}, near share {e['near share']:.2f}; surface {de.fmt(e['surface'])}, text {de.fmt(e['text'])}")
        for cell_name, c in e["cells"].items():
            lines.append(f"  {cell_name:20s} acts {de.fmt(c['acts'])}   acts - best surface {de.fmt(c['acts - best surface'])}")
        lines.append("")
    text = "\n".join(lines)
    (out_dir / "heuristic_disposition.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:3])
