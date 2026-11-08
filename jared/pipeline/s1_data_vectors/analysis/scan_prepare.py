"""Inputs of the Part B steering scan (26_spec_overnight_choice_scan.md §2): the pooled duration
manifold per (layer, position), and per-prompt steps for the scan's 10 prompts. A screen, not a
test: nothing here or in the driver makes a significance claim.

    python analysis/scan_prepare.py --phrase-dir results/horizon/exp14_Qwen3-8B_phrase \\
        --e0-root results/horizon
    -> results/horizon/exp15_Qwen3-8B_scan/{inputs.json, inputs.npz, fit_report.txt}

Four positions, five layers (10, 14, 18, 22, 26):
  number, unit   the horizon phrase; fitted on the phrase states of exp14_phrase_states.py
                 (states.npz holds L14 and L18, states_L10-22-26.npz the other layers)
  suffix         the suffix newline of the thinking-on prompt
  answer         the pre-label position (the `:` before the label)
                 both fitted on the saved thinking-on E0 activations (exp0_*_think-on_{full,units,fill}),
                 read one layer at a time (~150 MB per bank and layer)

Each cell is the 16 §2 fit as steering_prepare.py --phrase does it (`fit_phrase_cell`: duration
effects, the factors unit, regime and option order, the log near and far delays as covariates)
over the 1,386 pooled prompts. The answer slot sits after the reasoning, so its fit also has the
forced-close factor of 16 §4; the three earlier positions cannot see a forced close.

Alignment: the phrase prompts file lists each pooled bank's records that have a horizon (anchors
left out), and E0's pooled labels (pristine_manifold.pooled_labels) list the same prompts. Rows are
matched by sample_uid, and the prepare stops if a prompt has no E0 row, a uid repeats, or a
matched row disagrees with the record on log horizon, duration, rendered unit, regime or option
order. Before any suffix-newline fit it also runs the fill-bank anchor check of stage B
(pristine_manifold.gated_jobs) and stops if the fill run cannot be pooled.

Prompts: `per_transition` (5) of the E14 eligible none-near and both-near prompts, drawn with
steering_prepare.draw_prompts (seed 0, stratified by bank, natural renderings), interleaved so
that any prefix holds both transitions. The same prompts serve every cell.

Steps (steering_prepare's, per prompt): `manifold`, curve_delta from the prompt's horizon to its
target (the log-midpoint of its delays); `random`, a Gaussian direction as long as that step, per
cell. Written as step|L{layer}|{kind}|{position} [n, d] in inputs.npz.
"""

import argparse
import json
import sys
from itertools import zip_longest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from exp8_steering import selected_records  # noqa: E402
from steering_prepare import (OUT, TRANSITIONS, draw_prompts, fit_phrase_cell, phrase_cell_report,  # noqa: E402
                              phrase_columns, random_like, steerable, transition_target, varying_factors)
from spar_horizon.steering import curve_delta  # noqa: E402

SCAN_LAYERS = (18, 14, 22, 10, 26)           # run order: the most motivated layers first
POSITIONS = ("number", "unit", "suffix", "answer")
DIRECTIONS = ("manifold", "random")
E0_POSITION = {"suffix": "suffix newline", "answer": "pre-label"}
SCAN_TRANSITIONS = ("none-near", "both-near")
PER_TRANSITION = 5
SCAN_NAME = "exp15_{}_scan"


class E0Source:
    """The pooled thinking-on E0 runs: `labels` (pristine_manifold.pooled_labels) and, per
    position name of the activation file and layer, the stacked activations in the labels' order."""

    def __init__(self, parts):
        from pristine_manifold import pooled_labels   # imported here: pristine_manifold imports steering_prepare

        self.parts = parts
        self.labels = pooled_labels(parts)

    def matrix(self, position, layer):
        from pristine_manifold import cell_matrix

        return cell_matrix(self.parts, position, layer)


def load_e0(root, model, layers):
    """The pooled E0 runs of `model` ("Qwen3-8B") under `root`, after the fill anchor check
    at the suffix newline of each layer (as steering_prepare.pooled_main). Returns (source, anchors)."""
    from pristine_manifold import anchor_check, gated_jobs, load_parts

    parts = load_parts(Path(root), "on", "pooled", model)
    gate = [("on", "suffix newline", layer) for layer in layers]
    anchors = anchor_check(Path(root), "on", parts, [(p, l) for _, p, l in gate])
    if any(job not in gated_jobs(gate, {"on": anchors}) for job in gate):
        raise ValueError("the fill run's anchors do not match the old runs at the suffix (19 §2): "
                         "its cells cannot be pooled")
    return E0Source(parts), anchors


def e0_order(records, columns, labels):
    """E0 pooled row of each record (matched by sample_uid), after checking that the matched
    rows describe the same prompts: log horizon, duration, rendered unit, regime and option
    order. Raises ValueError otherwise."""
    uids = [str(u) for u in labels["sample_uids"]]
    index = {uid: i for i, uid in enumerate(uids)}
    if len(index) != len(uids):
        raise ValueError("E0's pooled labels repeat a sample_uid")
    missing = [r["sample_uid"] for r in records if r["sample_uid"] not in index]
    if missing:
        raise ValueError(f"{len(missing)} of {len(records)} phrase prompts have no E0 row, e.g. {missing[:3]}")
    order = np.array([index[r["sample_uid"]] for r in records])
    expected = {"y": columns["log_h"], "duration": columns["duration"], "units": columns["factors"]["unit"],
                "regime": columns["regime"], "near_first": columns["factors"]["order"]}
    for key, values in expected.items():
        found = np.asarray(labels[key])[order]
        same = np.allclose(found, values, atol=1e-6) if found.dtype.kind == "f" else np.array_equal(found, values)
        if not same:
            raise ValueError(f"E0 rows matched by sample_uid disagree with the phrase prompts on {key!r}")
    return order


def load_phrase_states(directory, layers):
    """{"L{layer}|{role}": [n, d]} for `layers` from states.npz and the layer-specific states_L*.npz
    in `directory`, and the sample_uids they share (checked across files)."""
    wanted = {f"L{layer}|{role}" for layer in layers for role in ("number", "unit")}
    arrays, uids = {}, None
    for path in [directory / "states.npz", *sorted(directory.glob("states_L*.npz"))]:
        if not path.exists():
            continue
        z = np.load(path)
        these = [str(u) for u in z["sample_uids"]]
        if uids is not None and these != uids:
            raise ValueError(f"{path.name}: sample_uids differ from the other states files")
        uids = these
        for key in wanted & (set(z.files) - set(arrays)):
            arrays[key] = z[key]
    missing = sorted(wanted - set(arrays))
    if missing:
        raise FileNotFoundError(f"no phrase states for {missing} in {directory}: run exp14_phrase_states.py "
                                "--layers for them")
    return arrays, uids


def interleave(*groups):
    """Rows of the groups in turn (a, b, a, b, ...), so that a prefix holds every group."""
    return np.array([row for rows in zip_longest(*groups) for row in rows if row is not None], dtype=int)


def cell_fit(position, layer, phrase, e0, order, columns, factors):
    """(fit, X) of one cell: the phrase positions from the phrase states, the others from E0."""
    if position in ("number", "unit"):
        X = phrase[f"L{layer}|{position}"]
    else:
        X = e0.matrix(E0_POSITION[position], layer)[order]
    if position == "answer":
        factors = {**factors, "forced": np.asarray(e0.labels["forced"])[order]}
    return fit_phrase_cell(X, columns, factors), X.astype(np.float64)


def scan_report_text(cells, layers, eligible, drawn_banks, anchors):
    lines = ["Part B steering scan: pooled fit per (layer, position) "
             "(26_spec_overnight_choice_scan.md §2); a screen, no significance claim.",
             "In-span share = the share of a step's squared length inside the span of the fit's factor and "
             "covariate effects (the part that moves with unit, regime, order or delays rather than the duration).",
             ""]
    if anchors is not None:
        lines += [f"fill anchors vs old runs: {json.dumps(anchors)}", ""]
    lines += ["in-span share of the manifold step, mean over the scan's prompts:",
              "  layer " + " ".join(f"{p:>7s}" for p in POSITIONS)]
    share = {(c["layer"], c["position"]): c["in_span_share"] for c in cells}
    for layer in layers:
        lines.append(f"  L{layer:<4d} " + " ".join(f"{share[layer, p]:7.2f}" for p in POSITIONS))
    lines.append("")
    for c in cells:
        lines.append(f"L{c['layer']} {c['position']:6s}: {c['n']} rows, k={c['k']}, factor span "
                     f"{c['factor_span_dims']} dims, P4 {c['p4_regime_auc']:.2f}; step length {c['step_length']:.1f}, "
                     f"in-span share {c['in_span_share']:.2f}")
    lines += ["", "prompts (the same in every cell): " + ", ".join(f"{t} eligible {eligible[t]}, drawn "
              + "/".join(f"{b} {n}" for b, n in drawn_banks[t].items()) for t in SCAN_TRANSITIONS)]
    return "\n".join(lines) + "\n"


def scan_main(phrase_dir, e0, out_dir, layers=SCAN_LAYERS, per_transition=PER_TRANSITION, seed=0,
              prompts_file=None, anchors=None):
    """Write the scan's inputs.npz, inputs.json and fit_report.txt to `out_dir`. `e0` is an
    E0Source (or anything with .labels and .matrix(position, layer))."""
    phrase_dir, out_dir = Path(phrase_dir), Path(out_dir)
    states_meta = json.loads((phrase_dir / "states.json").read_text())
    phrase, state_uids = load_phrase_states(phrase_dir, layers)
    prompts_path = Path(prompts_file or states_meta["prompts_file"])
    if not prompts_path.exists():
        prompts_path = OUT / prompts_path.name
    prompts_meta = json.loads(prompts_path.read_text())
    records = selected_records(prompts_meta)[:len(state_uids)]
    if [r["sample_uid"] for r in records] != state_uids:
        raise ValueError("the phrase states are not those of the prompts file's first rows")
    banks = np.array(prompts_meta["selected_banks"][:len(records)])
    columns = phrase_columns(records)
    order = e0_order(records, columns, e0.labels)
    factors = varying_factors(columns)
    steerable_rows = np.array([steerable(r) for r in records])
    curve_range = (columns["duration"].min(), columns["duration"].max())
    drawn, eligible = draw_prompts(columns, steerable_rows, banks, curve_range, per_transition, seed, False)
    rows = interleave(*(drawn[t] for t in SCAN_TRANSITIONS))
    transition_of = np.array([next(t for t in SCAN_TRANSITIONS if r in drawn[t]) for r in rows])
    target_regime = {name: target for name, _, target in TRANSITIONS}
    log_h = columns["log_h"][rows]
    target = np.array([transition_target(target_regime[t], columns["log_near"][r], columns["log_far"][r])
                       for t, r in zip(transition_of, rows)])

    arrays, cells = {}, []
    for layer in layers:
        for position in POSITIONS:
            fit, X = cell_fit(position, layer, phrase, e0, order, columns, factors)
            manifold = curve_delta(fit, log_h, target - log_h)
            steps = {"manifold": manifold,
                     "random": random_like(manifold, np.random.default_rng([seed, layer, POSITIONS.index(position)]))}
            arrays.update({f"step|L{layer}|{kind}|{position}": step.astype(np.float32) for kind, step in steps.items()})
            report = phrase_cell_report(layer, position, fit, X, columns, manifold, transition_of)
            cells.append({**{k: v for k, v in report.items() if k != "role"}, "position": position})
            print(f"L{layer} {position}: k={report['k']}, in-span share {report['in_span_share']:.2f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "inputs.npz", **arrays)
    prompts = [{"transition": t, "start_regime": columns["regime"][r], "target_regime": target_regime[t],
                "log_h": float(h), "target_log_h": float(tg)}
               for t, r, h, tg in zip(transition_of, rows, log_h, target)]
    meta = {"model": states_meta["model"], "generation": {**prompts_meta["generation"], "model": states_meta["model"]},
            "bank_settings": prompts_meta["bank_settings"],
            "selected": [prompts_meta["selected"][r] for r in rows], "selected_banks": [str(banks[r]) for r in rows],
            "sample_uids": [records[r]["sample_uid"] for r in rows], "prompts": prompts,
            "transitions": list(SCAN_TRANSITIONS), "layers": list(layers), "positions": list(POSITIONS),
            "directions": list(DIRECTIONS), "cells": cells, "eligible": eligible, "per_transition": per_transition,
            "seed": seed, "phrase_dir": str(phrase_dir), "prompts_file": str(prompts_path),
            "anchor_check": anchors}
    (out_dir / "inputs.json").write_text(json.dumps(meta, indent=1))
    drawn_banks = {t: {b: int((banks[drawn[t]] == b).sum()) for b in dict.fromkeys(banks) if (banks[drawn[t]] == b).any()}
                   for t in SCAN_TRANSITIONS}
    report = scan_report_text(cells, layers, eligible, drawn_banks, anchors)
    (out_dir / "fit_report.txt").write_text(report)
    print(report)


def main(args):
    phrase_dir = Path(args.phrase_dir)
    model = json.loads((phrase_dir / "states.json").read_text())["model"].split("/")[-1]
    layers = tuple(int(x) for x in args.layers.split(","))
    e0, anchors = load_e0(args.e0_root, model, layers)
    out_dir = Path(args.out) if args.out else phrase_dir.parent / SCAN_NAME.format(model)
    scan_main(phrase_dir, e0, out_dir, layers, args.per_transition, args.seed, args.prompts, anchors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phrase-dir", required=True, help="exp14_<model>_phrase folder with the phrase states")
    parser.add_argument("--e0-root", required=True, help="results folder with the exp0_*_think-on_* runs")
    parser.add_argument("--out", default=None, help="output folder (default: exp15_<model>_scan beside the phrase folder)")
    parser.add_argument("--prompts", default=None, help="phrase prompts file (default: the states' own)")
    parser.add_argument("--layers", default=",".join(map(str, SCAN_LAYERS)), help="layers to prepare, in run order")
    parser.add_argument("--per-transition", type=int, default=PER_TRANSITION)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
