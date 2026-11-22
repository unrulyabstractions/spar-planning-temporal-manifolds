"""E20 per-type table: the phrase donor patch (and the bank-fit step) against each eval type's
distance from Alan's prompt (EvalsSteeringSpec.md §4).

    python -m pipeline.s3_analysis.eval_types_donor [E20_DIR] [EVALTYPES_DIR]

Reads <E20_DIR>/<type>/summary.json and the distance file's rank cell (horizon subspace, last prompt
token; `eval_distance.json`), writes <E20_DIR>/donor_table.md and .json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RESULTS = Path.home() / "SPAR/results/horizon"
ARMS = ("donor", "effects", "random")


def type_distance(distance, type_id):
    """The type's median horizon-subspace distance (paired) at the distance file's rank cell."""
    cell = distance["types"][type_id]["distance"][distance["rank_cell"]]
    return cell["horizon"]["paired"]["median"]


def table_rows(e20_dir, distance):
    """One row per type that has a summary: distance, noop's flip rate, and per arm the flip rate
    over noop and the log-odds shift toward the target versus noop."""
    rows = []
    for summary_file in sorted(Path(e20_dir).glob("*/summary.json")):
        type_id = summary_file.parent.name
        summary = json.loads(summary_file.read_text())
        if "noop" not in summary or "donor" not in summary:
            continue
        row = {"type_id": type_id, "distance": type_distance(distance, type_id), "n": summary["donor"]["n"],
               "no_donor": summary["donor"]["n_skipped_no_donor"], "noop_flipped": summary["noop"]["flipped_toward"],
               "noop_forced": summary["noop"]["forced"]}
        for arm in ARMS:
            if arm in summary:
                row[arm] = {"flipped_over_noop": summary[arm]["flipped_toward_minus_noop"],
                            "log_odds": summary[arm]["toward_log_odds_vs_noop"], "forced": summary[arm]["forced"]}
        rows.append(row)
    return sorted(rows, key=lambda r: r["distance"])


def distance_correlations(rows):
    """Spearman of the donor effect (flip rate over noop, log-odds vs noop) against distance."""
    distance = [r["distance"] for r in rows]
    out = {}
    for name, values in (("flipped_over_noop", [r["donor"]["flipped_over_noop"] for r in rows]),
                         ("log_odds", [r["donor"]["log_odds"][0] for r in rows])):
        rho, p = spearmanr(distance, values)
        out[name] = {"rho": float(rho), "p": float(p), "n": len(rows)}
    return out


def markdown(rows, correlations):
    def arm_cells(row, arm):
        if arm not in row:
            return "- | -"
        a = row[arm]
        mean, low, high = a["log_odds"]
        return f"{a['flipped_over_noop']:+.2f} | {mean:+.1f} [{low:+.1f}, {high:+.1f}]"

    lines = ["| type | dist | n (no donor) | noop flipped | donor: flipped − noop | donor: log-odds vs noop "
             "| effects: flipped − noop | effects: log-odds | random: flipped − noop | random: log-odds | forced noop / donor |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['type_id']} | {r['distance']:.2f} | {r['n']} ({r['no_donor']}) | {r['noop_flipped']:.2f} | "
                     + " | ".join(arm_cells(r, arm) for arm in ARMS)
                     + f" | {r['noop_forced']:.2f} / {r['donor']['forced']:.2f} |")
    lines.append("")
    for name, c in correlations.items():
        lines.append(f"- Spearman, distance vs donor {name}: {c['rho']:+.2f} (n = {c['n']}, p = {c['p']:.3f})")
    return "\n".join(lines) + "\n"


def main(e20_dir, evaltypes_dir):
    distance = json.loads((Path(evaltypes_dir) / "eval_distance.json").read_text())
    rows = table_rows(e20_dir, distance)
    correlations = distance_correlations(rows)
    (Path(e20_dir) / "donor_table.json").write_text(json.dumps({"rows": rows, "correlations": correlations}, indent=1))
    text = markdown(rows, correlations)
    (Path(e20_dir) / "donor_table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else RESULTS / "exp20_Qwen3-8B_evaltype_transfer",
         sys.argv[2] if len(sys.argv) > 2 else RESULTS / "evaltypes_Qwen3-8B_think-on")
