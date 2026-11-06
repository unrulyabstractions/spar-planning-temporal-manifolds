"""Summary of the Part B steering scan (26_spec_overnight_choice_scan.md §2) over whatever cell
files exist; a partial scan is fine and the output says how many cells are done.

    python analysis/scan_summary.py results/horizon/exp15_Qwen3-8B_scan
    -> <dir>/scan_summary.json and scan_summary.txt, and the table on stdout

THIS IS A SCREEN, NOT A TEST. Ten prompts per cell, no confidence intervals, no p-values and no
significance claim: the rule below only picks the cells worth a closer look.

A flip is a label different from the baseline label of the same prompt. Toward-target flips are
the flips to the near label (every scan transition, none -> near and both -> near, aims at near).
Excess = the cell's flips minus the mean of the two no-op reruns' flips; excess toward is the
cell's toward-target flips minus the mean of the no-ops' toward-target flips. A cell is reported
when excess toward >= REPORT_EXCESS, always next to its random twin (the other direction at the
same layer and position) and the no-op flips.

The reference of a regenerating cell is the baseline unit's label and near log-odds. The
answer-slot cells do not regenerate: their rows carry `ref_label` and `ref_near_lo`, the unedited
re-score of the same fixed reasoning on the same pass, and a flip is a change of that label.
Those flips are of a different kind from the regenerating cells' and not comparable with them.
"""

import json
import sys
from pathlib import Path

import numpy as np

BASELINE = "baseline"
NOOPS = ("noop 1", "noop 2")
REPORT_EXCESS = 4


def cell_name(layer, position, direction):
    return f"L{layer} {position} {direction}"


def unit_file(directory, name):
    return Path(directory) / "conditions" / f"{name.replace(' ', '_')}.json"


def load_unit(directory, name):
    """The finished unit `name` (its json), or None."""
    path = unit_file(directory, name)
    return json.loads(path.read_text()) if path.exists() else None


def flip_measures(rows, baseline_rows):
    """Flips of `rows` against their reference: the row's own `ref_label`/`ref_near_lo` where it
    has them (answer-slot cells), else the baseline unit's row of the same prompt."""
    baseline = {r["sample_uid"]: r for r in baseline_rows}
    flips = toward = 0
    shifts = []
    for row in rows:
        if "ref_label" in row:
            label, near_lo = row["ref_label"], row["ref_near_lo"]
        else:
            label, near_lo = baseline[row["sample_uid"]]["label"], baseline[row["sample_uid"]]["near_lo"]
        flipped = row["label"] != label
        flips += flipped
        toward += flipped and row["label"] == row["near_label"]
        shifts.append(row["near_lo"] - near_lo)
    return {"n": len(rows), "flips": int(flips), "toward": int(toward), "away": int(flips - toward),
            "near_lo_shift": float(np.mean(shifts))}


def noise_measures(noops):
    """{noop name: flip measures} and the means over the reruns: {"flips", "toward"}."""
    mean = {key: float(np.mean([m[key] for m in noops.values()])) for key in ("flips", "toward")}
    return {"reruns": noops, "mean": mean}


def judge(measures, noise):
    """Excess flips of a cell over the no-op mean, and whether it is reported."""
    excess_total = measures["flips"] - noise["mean"]["flips"]
    excess_toward = measures["toward"] - noise["mean"]["toward"]
    return {"excess_total": excess_total, "excess_toward": excess_toward,
            "reported": bool(excess_toward >= REPORT_EXCESS)}


def twin_name(layer, position, direction, directions):
    other = next(d for d in directions if d != direction)
    return cell_name(layer, position, other)


def summarize(directory):
    """The summary dict over the finished units of the scan in `directory`."""
    meta = json.loads((Path(directory) / "inputs.json").read_text())
    directions = meta["directions"]
    names = [cell_name(layer, position, direction) for layer in meta["layers"] for position in meta["positions"]
             for direction in directions]
    units = {name: load_unit(directory, name) for name in (BASELINE, *NOOPS, *names)}
    baseline = units[BASELINE]
    out = {"n_cells_done": sum(units[n] is not None for n in names), "n_cells_total": len(names),
           "baseline_done": baseline is not None,
           "noops_done": sum(units[n] is not None for n in NOOPS), "noise": None, "cells": [], "reported": []}
    if baseline is None:
        return out
    noops = {n: flip_measures(units[n]["rows"], baseline["rows"]) for n in NOOPS if units[n] is not None}
    noise = noise_measures(noops) if noops else None
    out["noise"] = noise
    for name in names:
        unit = units[name]
        if unit is None:
            continue
        measures = flip_measures(unit["rows"], baseline["rows"])
        entry = {"name": name, "layer": unit["layer"], "position": unit["position"], "direction": unit["direction"],
                 **measures, **(judge(measures, noise) if noise else {"excess_total": None, "excess_toward": None,
                                                                    "reported": False})}
        entry["twin"] = twin_name(unit["layer"], unit["position"], unit["direction"], directions)
        out["cells"].append(entry)
    by_name = {c["name"]: c for c in out["cells"]}
    out["reported"] = [{"cell": c, "twin": by_name.get(c["twin"]), "noise": noise}
                       for c in out["cells"] if c["reported"]]
    return out


def format_table(summary):
    lines = ["Part B steering scan: a SCREEN on 10 prompts per cell; no significance claim.",
             f"cells done: {summary['n_cells_done']} of {summary['n_cells_total']}; baseline "
             f"{'done' if summary['baseline_done'] else 'MISSING'}; no-op reruns {summary['noops_done']} of {len(NOOPS)}"]
    noise = summary["noise"]
    if noise is None:
        return "\n".join(lines + ["no no-op rerun yet: no excess can be computed"]) + "\n"
    for name, m in noise["reruns"].items():
        lines.append(f"  {name}: {m['flips']} flips ({m['toward']} toward near) of {m['n']}")
    lines += [f"  mean no-op flips {noise['mean']['flips']:.1f}, toward near {noise['mean']['toward']:.1f}", "",
              f"{'cell':26s} {'flips':>5s} {'toward':>6s} {'away':>4s} {'excess':>6s} {'ex.tow':>6s} {'d near lo':>9s}  note"]
    for c in summary["cells"]:
        note = "REPORTED" if c["reported"] else ""
        lines.append(f"{c['name']:26s} {c['flips']:5d} {c['toward']:6d} {c['away']:4d} {c['excess_total']:+6.1f} "
                     f"{c['excess_toward']:+6.1f} {c['near_lo_shift']:+9.2f}  {note}")
    lines += ["", f"reported: excess toward-target flips >= {REPORT_EXCESS}"]
    if not summary["reported"]:
        lines.append("  none")
    for item in summary["reported"]:
        c, twin = item["cell"], item["twin"]
        twin_text = (f"twin {twin['name']}: {twin['flips']} flips, {twin['toward']} toward, excess toward "
                     f"{twin['excess_toward']:+.1f}" if twin else f"twin {c['twin']}: not run yet")
        lines.append(f"  {c['name']}: {c['flips']} flips, {c['toward']} toward, excess toward "
                     f"{c['excess_toward']:+.1f}, d near lo {c['near_lo_shift']:+.2f}")
        lines.append(f"      {twin_text}; no-op flips "
                     + ", ".join(f"{n} {m['flips']} ({m['toward']} toward)" for n, m in item["noise"]["reruns"].items()))
    return "\n".join(lines) + "\n"


def main(directory):
    summary = summarize(directory)
    (Path(directory) / "scan_summary.json").write_text(json.dumps(summary, indent=1))
    text = format_table(summary)
    (Path(directory) / "scan_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
