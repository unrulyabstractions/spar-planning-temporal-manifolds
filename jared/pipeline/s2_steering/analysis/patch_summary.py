"""Summary of the phrase number-patch experiment (27_spec_phrase_number_patch.md §2-§3) over
whatever unit files exist; a partial run is fine and the output says which units are missing.

    python -m pipeline.s2_steering.analysis.patch_summary steering-or-results/exp16_Qwen3-8B_patch
    -> <dir>/patch_summary.json and patch_summary.txt, and the table on stdout

A SCREEN on 24 pairs: counts and rates, no confidence intervals and no significance claim. Every
patched unit is compared with the baseline unit of the same pair (the unpatched run in the inputs'
order); the noise floor is the rerun (another order) against the baseline.

Per unit:
  states donor   the trace's first stated horizon (exp14_analysis.first_stated_horizon) is the donor's
                 number, in the donor's unit
  lands target   that horizon is in the target (near) regime for the pair's delays
  moved          that horizon is further than MOVED decades from the prompt's own
  shift          decades the stated horizon moved toward the donor, among traces that state one
  d near lo      change in near log-odds against the baseline, over natural closes (neither trace
                 forced) and over all pairs; positive is toward the near label, the donor's regime
  flips          labels different from the baseline's, and how many are to the near label

Criteria (spec §3), each "pass", "fail" or "pending" (a unit it needs is missing):
  C0  the self patch changes 0 labels
  C1  donor patch at the number tokens: >= 50% land in the target regime, random number <= 10%
  C2  manifold number step states the donor's number in < 25% of the donor patch's rate
  C3  donor patch at the unit token moves the stated number in <= 1 pair
"""

import json
import sys
from pathlib import Path

import numpy as np

from pipeline.s3_analysis.by_label.exp14_analysis import MOVED, first_stated_horizon
from pipeline.s2_steering.analysis.scan_summary import flip_measures, load_unit
from spar_horizon.full_bank import regime
from pipeline.s3_analysis.by_datatype.written_unit import close_to, durations

BASELINE = "baseline"
RERUN = "rerun"
C1_LANDS_MIN = 0.5
C1_RANDOM_MAX = 0.10
C2_RATIO_MAX = 0.25
C3_MAX_PAIRS = 1

# (condition number of spec §2, source, positions) in run order: positions are the whole phrase
# (number and unit tokens), the number tokens, or the unit token
CONDITIONS = ((1, "self", "phrase"), (2, "donor", "phrase"), (3, "donor", "number"), (6, "manifold", "number"),
              (7, "manifold", "phrase"), (8, "random", "number"), (4, "donor", "unit"), (5, "subspace", "number"),
              (9, "inspan", "number"))


def unit_name(source, positions, layer):
    return f"{source} {positions} L{layer}"


def planned_units(layer, depth_layers):
    """[(name, condition number, source, positions, layer)] in run order: the baseline and the rerun
    (number None, source None), the conditions at `layer`, then condition 10, the number-token donor
    patch at each of `depth_layers`."""
    units = [(BASELINE, None, None, None, None), (RERUN, None, None, None, None)]
    units += [(unit_name(source, positions, layer), number, source, positions, layer)
              for number, source, positions in CONDITIONS]
    units += [(unit_name("donor", "number", depth), 10, "donor", "number", depth) for depth in depth_layers]
    return units


def stated_measures(row):
    """Where the trace's first stated horizon sits against the pair's donor, own horizon and target."""
    stated = first_stated_horizon(row["think_text"])
    if stated is None:
        return {"stated": False, "states_donor": False, "lands_target": False, "moved": False, "shift": np.nan}
    years, unit, _ = stated
    donor_years = 10 ** row["donor_log_h"]
    donor_unit = durations(row["donor_horizon_text"])[0][1]
    shift = np.log10(years) - row["log_h"]
    toward = np.sign(row["donor_log_h"] - row["log_h"])
    return {"stated": True, "states_donor": bool(unit == donor_unit and close_to(years, donor_years)),
            "lands_target": regime(years, row["near_delay_years"], row["far_delay_years"]) == row["target_regime"],
            "moved": bool(abs(shift) > MOVED), "shift": float(shift * toward)}


def unit_measures(rows, baseline_rows):
    """The per-unit measures of the module doc over `rows`, against the baseline rows of the same pairs."""
    base = {r["sample_uid"]: r for r in baseline_rows}
    stated = [stated_measures(r) for r in rows]
    shifts = [m["shift"] for m in stated if m["stated"]]
    natural = [r["near_lo"] - base[r["sample_uid"]]["near_lo"] for r in rows
               if not r["forced"] and not base[r["sample_uid"]]["forced"]]
    n = len(rows)
    flips = flip_measures(rows, baseline_rows)
    return {"n": n, "n_stated": len(shifts),
            "states_donor": sum(m["states_donor"] for m in stated), "lands_target": sum(m["lands_target"] for m in stated),
            "moved": sum(m["moved"] for m in stated), "shift_toward_donor": float(np.mean(shifts)) if shifts else None,
            "near_lo_shift_natural": float(np.mean(natural)) if natural else None, "n_natural": len(natural),
            "near_lo_shift": flips["near_lo_shift"], "flips": flips["flips"], "flips_toward_near": flips["toward"],
            "forced": sum(bool(r["forced"]) for r in rows)}


def rate(measures, key):
    return measures[key] / measures["n"]


def criterion(status, detail):
    return {"status": status, "detail": detail}


def pending(*missing):
    return criterion("pending", "needs " + ", ".join(missing))


def evaluate_criteria(m, layer):
    """C0-C3 from the measures `m` {unit name: unit_measures} of the units that exist."""
    self_u, donor_u = m.get(unit_name("self", "phrase", layer)), m.get(unit_name("donor", "number", layer))
    random_u, manifold_u = m.get(unit_name("random", "number", layer)), m.get(unit_name("manifold", "number", layer))
    unit_u = m.get(unit_name("donor", "unit", layer))
    out = {}
    out["C0"] = (criterion("pass" if self_u["flips"] == 0 else "fail", f"self patch flips {self_u['flips']} of {self_u['n']} labels")
                 if self_u else pending(unit_name("self", "phrase", layer)))
    if donor_u and random_u:
        lands, random_lands = rate(donor_u, "lands_target"), rate(random_u, "lands_target")
        passed = lands >= C1_LANDS_MIN and random_lands <= C1_RANDOM_MAX
        out["C1"] = criterion("pass" if passed else "fail",
                              f"donor number tokens land in the target regime in {donor_u['lands_target']} of {donor_u['n']} "
                              f"({lands:.2f}, need >= {C1_LANDS_MIN}); random {random_u['lands_target']} of {random_u['n']} "
                              f"({random_lands:.2f}, need <= {C1_RANDOM_MAX})")
    else:
        out["C1"] = pending(unit_name("donor", "number", layer), unit_name("random", "number", layer))
    if donor_u and manifold_u:
        donor_rate, manifold_rate = rate(donor_u, "states_donor"), rate(manifold_u, "states_donor")
        detail = (f"manifold number step states the donor's number in {manifold_u['states_donor']} of {manifold_u['n']} "
                  f"({manifold_rate:.2f}), the donor patch in {donor_u['states_donor']} of {donor_u['n']} ({donor_rate:.2f})")
        if donor_rate == 0:
            out["C2"] = criterion("pending", detail + "; the donor patch never states its number, so the ratio is undefined")
        else:
            out["C2"] = criterion("pass" if manifold_rate < C2_RATIO_MAX * donor_rate else "fail",
                                  detail + f"; ratio {manifold_rate / donor_rate:.2f}, need < {C2_RATIO_MAX}")
    else:
        out["C2"] = pending(unit_name("donor", "number", layer), unit_name("manifold", "number", layer))
    out["C3"] = (criterion("pass" if unit_u["moved"] <= C3_MAX_PAIRS else "fail",
                           f"unit-token donor patch moves the stated number in {unit_u['moved']} of {unit_u['n']} pairs "
                           f"(need <= {C3_MAX_PAIRS})")
                 if unit_u else pending(unit_name("donor", "unit", layer)))
    return out


def summarize(directory):
    """The summary dict over the finished units of the run in `directory`."""
    meta = json.loads((Path(directory) / "inputs.json").read_text())
    layer = meta["patch_layer"]
    planned = planned_units(layer, meta["depth_layers"])
    units = {name: load_unit(directory, name) for name, *_ in planned}
    out = {"layer": layer, "n_units_done": sum(u is not None for n, u in units.items() if n not in (BASELINE, RERUN)),
           "n_units_total": len(planned) - 2, "baseline_done": units[BASELINE] is not None,
           "rerun_done": units[RERUN] is not None, "noise": None, "units": [], "criteria": None}
    if units[BASELINE] is None:
        return out
    base = units[BASELINE]["rows"]
    measures = {}
    for name, number, source, positions, unit_layer in planned[1:]:
        if units[name] is None:
            continue
        measures[name] = unit_measures(units[name]["rows"], base)
        if name != RERUN:
            out["units"].append({"name": name, "condition": number, "source": source, "positions": positions,
                                 "layer": unit_layer, **measures[name]})
    out["noise"] = measures.get(RERUN)
    out["criteria"] = evaluate_criteria({n: m for n, m in measures.items() if n != RERUN}, layer)
    return out


def format_table(summary):
    lines = ["Phrase number patch: a SCREEN on same-unit crossing pairs; no significance claim.",
             f"units done: {summary['n_units_done']} of {summary['n_units_total']}; baseline "
             f"{'done' if summary['baseline_done'] else 'MISSING'}; rerun {'done' if summary['rerun_done'] else 'MISSING'}"]
    if not summary["baseline_done"]:
        return "\n".join(lines + ["no baseline unit yet: nothing to compare with"]) + "\n"
    noise = summary["noise"]
    if noise:
        lines.append(f"noise floor (rerun vs baseline): {noise['flips']} of {noise['n']} labels flip "
                     f"({noise['flips_toward_near']} to near), d near log-odds {noise['near_lo_shift']:+.2f}, "
                     f"states donor {noise['states_donor']}, lands target {noise['lands_target']}")
    lines += ["", f"{'#':>2s} {'unit':26s} {'n':>3s} {'donor':>5s} {'target':>6s} {'moved':>5s} {'shift':>6s} "
                  f"{'d near lo':>9s} {'natural':>8s} {'flips':>5s} {'to near':>7s} {'forced':>6s}"]
    for u in summary["units"]:
        shift = "   n/a" if u["shift_toward_donor"] is None else f"{u['shift_toward_donor']:+6.2f}"
        natural = "     n/a" if u["near_lo_shift_natural"] is None else f"{u['near_lo_shift_natural']:+8.2f}"
        lines.append(f"{u['condition']:>2d} {u['name']:26s} {u['n']:3d} {u['states_donor']:5d} {u['lands_target']:6d} "
                     f"{u['moved']:5d} {shift} {u['near_lo_shift']:+9.2f} {natural} {u['flips']:5d} "
                     f"{u['flips_toward_near']:7d} {u['forced']:6d}")
    lines += ["", "donor / target / moved: traces whose first stated horizon is the donor's number, lands in the target "
                  "regime, or moved from the prompt's own; shift: decades toward the donor among traces that state one;",
              "natural: d near log-odds over pairs where neither trace was cut off; to near: flips to the near label", "",
              "criteria (fixed in spec §3):"]
    lines += [f"  {name}: {c['status'].upper():8s} {c['detail']}" for name, c in summary["criteria"].items()]
    return "\n".join(lines) + "\n"


def main(directory):
    summary = summarize(directory)
    (Path(directory) / "patch_summary.json").write_text(json.dumps(summary, indent=1))
    text = format_table(summary)
    (Path(directory) / "patch_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
