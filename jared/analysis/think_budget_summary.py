"""Criteria P0, C1, C2, C3 of 28_spec_choice4_side.md from whatever E17 chunks exist.

    python analysis/think_budget_summary.py /workspace/results/horizon/exp17_Qwen3-8B_think_budget
    -> <run root>/think_budget_summary.{txt,json}

The run root holds one folder per bank (choice4, choice8) with unit1/, unit2/, unit3/ chunk files
(experiments/exp17_think_budget.py). Stage 0 (steering/exp17_Qwen3-8B_think_budget/stage0_<bank>.json)
gives each trace's original forced label, loop flag, last committed label and the bank's truth. Any
prefix of chunks is usable: every figure carries its n, and the C1 net-fix threshold scales with n.

  P0  unit 1 reproduces at least 85% of the original forced labels
  C1  at least 60% of the traces close within unit 2's 3,072 extra tokens; on those, exact accuracy is
      at least 85% and wrong->right minus right->wrong (against the original forced label) is at least
      10 per 105 traces
  C2  excess-late (sum over the second-half positions of pick rate minus truth rate) of the closed
      continuations is at most +0.17 for choice4 (half of its forced +0.34), half of the bank's own
      forced excess-late for other banks
  C3  of the looping traces at most 30% close, of the others at least 70%
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from exp17_think_budget import INPUTS_DIR, finished_chunks, read_chunk  # noqa: E402
from think_budget_stage0 import accuracy, position_stats  # noqa: E402

P0_MIN = 0.85
C1_CLOSE_MIN, C1_ACCURACY_MIN, C1_NET_PER_105 = 0.60, 0.85, 10
C2_MAX_CHOICE4 = 0.17
C3_LOOP_MAX, C3_OTHER_MIN = 0.30, 0.70


def load_unit(bank_dir, unit):
    """{sample_uid: row} of every finished chunk of a unit, and the chunk numbers."""
    directory = Path(bank_dir) / f"unit{unit}"
    if not directory.exists():
        return {}, []
    chunks = finished_chunks(directory)
    return {r["sample_uid"]: r for n in chunks for r in read_chunk(directory, n)[0]["rows"]}, chunks


def rate(hits, n):
    return [int(hits), int(n)]


def share(pair):
    return pair[0] / pair[1] if pair[1] else None


def transitions(metas, original, new):
    """(wrong->right, right->wrong) between two label lists."""
    right = lambda m, lab: lab == m["truth_label"]  # noqa: E731
    fixed = sum(not right(m, o) and right(m, n) for m, o, n in zip(metas, original, new))
    broken = sum(right(m, o) and not right(m, n) for m, o, n in zip(metas, original, new))
    return int(fixed), int(broken)


def agreement(rows_with_labels):
    """[agree, n] of (label, tentative) pairs where the trace has a committed label."""
    pairs = [(lab, t) for lab, t in rows_with_labels if t]
    return rate(sum(lab == t for lab, t in pairs), len(pairs))


def close_rates(pairs):
    """{group: [closed, n]} for (group, closed) pairs."""
    out = {}
    for group, closed in pairs:
        hit, n = out.get(group, (0, 0))
        out[group] = (hit + bool(closed), n + 1)
    return {group: rate(*v) for group, v in sorted(out.items())}


def unit_one(metas, rows):
    """Batching noise floor: unit 1 labels against the original forced labels."""
    labels = [r["label"] for r in rows]
    original = [m["orig_label"] for m in metas]
    fixed, broken = transitions(metas, original, labels)
    return {"n": len(rows), "same_label": rate(sum(a == b for a, b in zip(labels, original)), len(rows)),
            "wrong_to_right": fixed, "right_to_wrong": broken,
            "accuracy": accuracy(metas, labels), "original_accuracy": accuracy(metas, original),
            "position": position_stats(metas, labels)}


def unit_two(metas, rows):
    """Continuation after 3,072 more tokens: close rates, and the closed ones against their forced labels."""
    closed = [(m, r) for m, r in zip(metas, rows) if not r["forced"]]
    cm, cr = [m for m, _ in closed], [r for _, r in closed]
    labels, original = [r["label"] for r in cr], [m["orig_label"] for m in cm]
    fixed, broken = transitions(cm, original, labels)
    new_lens = [r["new_think_len"] for r in cr]
    return {"n": len(rows), "closed": rate(len(closed), len(rows)),
            "closed_by_regime": close_rates((m["regime"], not r["forced"]) for m, r in zip(metas, rows)),
            "closed_by_loop": close_rates(("loop" if m["loop"] else "no loop", not r["forced"]) for m, r in zip(metas, rows)),
            "accuracy_closed": accuracy(cm, labels), "original_accuracy_closed": accuracy(cm, original),
            "wrong_to_right": fixed, "right_to_wrong": broken, "net_fixes": fixed - broken,
            "position_closed": position_stats(cm, labels), "position_original_same_prompts": position_stats(cm, original),
            "still_open_accuracy": accuracy([m for m, r in zip(metas, rows) if r["forced"]],
                                            [r["label"] for r in rows if r["forced"]]),
            "new_tokens_closed": {"mean": float(np.mean(new_lens)), "median": float(np.median(new_lens))} if new_lens else None,
            "agrees_with_tentative": {"forced_label": agreement((m["orig_label"], m["tentative"]) for m in cm),
                                      "continued_label": agreement((r["label"], m["tentative"]) for m, r in closed)}}


def unit_three(metas2, rows2, rows3):
    """Of the traces still open after unit 2, those that closed in the 2,048 more tokens; and the
    cumulative close within 8,192."""
    open_uids = {m["sample_uid"] for m, r in zip(metas2, rows2) if r["forced"]}
    attempted = [r for uid, r in rows3.items() if uid in open_uids]
    closed3 = sum(not r["forced"] for r in attempted)
    closed2 = sum(not r["forced"] for r in rows2)
    return {"open_after_unit2": len(open_uids), "attempted": len(attempted), "closed_in_unit3": closed3,
            "closed_within_8192_of_all_unit2": rate(closed2 + closed3, len(rows2)),
            "note": "unit 3 covers the chunks it was run on; the cumulative figure counts the unattempted open traces as open"}


def criteria(bank, stage0, u1, u2):
    """P0, C1, C2, C3 as {value(s), pass}; pass is None when the unit is missing or empty."""
    out = {}
    if u1 and u1["n"]:
        out["P0"] = {"reproduced": u1["same_label"], "share": share(u1["same_label"]), "threshold": P0_MIN,
                     "pass": share(u1["same_label"]) >= P0_MIN}
    else:
        out["P0"] = {"pass": None}
    if not (u2 and u2["n"]):
        return {**out, "C1": {"pass": None}, "C2": {"pass": None}, "C3": {"pass": None}}
    closed, acc = share(u2["closed"]), share(rate(u2["accuracy_closed"]["exact"], u2["accuracy_closed"]["n"]))
    net_threshold = C1_NET_PER_105 * u2["n"] / 105
    out["C1"] = {"closed_share": closed, "closed_min": C1_CLOSE_MIN, "exact_accuracy_closed": acc, "accuracy_min": C1_ACCURACY_MIN,
                 "net_fixes": u2["net_fixes"], "net_min": net_threshold,
                 "pass": bool(closed >= C1_CLOSE_MIN and acc is not None and acc >= C1_ACCURACY_MIN
                              and u2["net_fixes"] >= net_threshold)}
    excess = u2["position_closed"]["excess_late"]
    forced_excess = stage0["reference"]["forced"]["position"]["excess_late"]
    limit = C2_MAX_CHOICE4 if bank == "choice4" else forced_excess / 2
    out["C2"] = {"excess_late_closed": excess, "max": limit, "forced_excess_late_bank": forced_excess,
                 "pass": None if excess is None else bool(excess <= limit)}
    by_loop = u2["closed_by_loop"]
    loop, other = share(by_loop.get("loop", [0, 0])), share(by_loop.get("no loop", [0, 0]))
    out["C3"] = {"loop_closed": by_loop.get("loop"), "loop_closed_max": C3_LOOP_MAX,
                 "no_loop_closed": by_loop.get("no loop"), "no_loop_closed_min": C3_OTHER_MIN,
                 "pass": None if loop is None or other is None else bool(loop <= C3_LOOP_MAX and other >= C3_OTHER_MIN)}
    return out


def summarize_bank(bank, stage0, unit_rows):
    """The summary dict of one bank. `unit_rows` is {unit: {uid: row}}."""
    by_uid = {r["sample_uid"]: r for r in stage0["rows"]}
    pair = lambda rows: ([by_uid[uid] for uid in rows], list(rows.values()))  # noqa: E731
    out = {"bank": bank, "n_forced_traces": len(stage0["rows"]), "n_loop": sum(r["loop"] for r in stage0["rows"]),
           "reference": stage0["reference"]}
    u1 = unit_one(*pair(unit_rows[1])) if unit_rows.get(1) else None
    u2 = unit_two(*pair(unit_rows[2])) if unit_rows.get(2) else None
    out["unit1"], out["unit2"] = u1, u2
    if u2 and unit_rows.get(3):
        metas2, rows2 = pair(unit_rows[2])
        out["unit3"] = unit_three(metas2, rows2, unit_rows[3])
    out["criteria"] = criteria(bank, stage0, u1, u2)
    return out


def fmt_rates(rates):
    return " ".join(f"{r:.2f}" for r in rates) if rates else "-"


def fmt_pair(pair):
    return f"{pair[0]}/{pair[1]}" if pair else "-"


def fmt_position(label, pos):
    if not pos or not pos["n"]:
        return f"  {label}: no rows"
    return (f"  {label}: n {pos['n']}, pick {fmt_rates(pos['pick_rate'])}, truth {fmt_rates(pos['truth_rate'])}, "
            f"late pick {pos['late_pick']:.2f}, excess-late {pos['excess_late']:+.2f}")


def fmt_accuracy(label, acc):
    return f"  {label}: n {acc['n']}, exact {acc['exact']}, half {acc['half']}, within-half {acc['within_half']}"


def bank_text(s, chunks):
    lines = [f"== {s['bank']}: {s['n_forced_traces']} force-closed traces ({s['n_loop']} flagged looping); chunks done: "
             + ", ".join(f"unit {u}: {len(c)}" for u, c in sorted(chunks.items())) + " =="]
    ref = s["reference"]
    lines += ["reference from E0 (all rows of the bank):",
              fmt_accuracy("forced", ref["forced"]["accuracy"]), fmt_position("forced", ref["forced"]["position"]),
              fmt_accuracy("natural", ref["natural"]["accuracy"]), fmt_position("natural", ref["natural"]["position"])]
    u1, u2 = s["unit1"], s["unit2"]
    if u1:
        lines += ["", f"unit 1 (answer pass again, new batches), n {u1['n']}:",
                  f"  same label as the original: {fmt_pair(u1['same_label'])}   wrong->right {u1['wrong_to_right']}  "
                  f"right->wrong {u1['right_to_wrong']}",
                  fmt_accuracy("unit 1", u1["accuracy"]), fmt_accuracy("original", u1["original_accuracy"]),
                  fmt_position("unit 1", u1["position"])]
    if u2:
        by_regime = ", ".join(f"{k} {fmt_pair(v)}" for k, v in u2["closed_by_regime"].items())
        by_loop = ", ".join(f"{k} {fmt_pair(v)}" for k, v in u2["closed_by_loop"].items())
        tokens = u2["new_tokens_closed"]
        agree = u2["agrees_with_tentative"]
        lines += ["", f"unit 2 (up to 3,072 more think tokens), n {u2['n']}:",
                  f"  closed within the extension: {fmt_pair(u2['closed'])}   by regime: {by_regime}   by loop: {by_loop}",
                  fmt_accuracy("continued, closed", u2["accuracy_closed"]),
                  fmt_accuracy("their forced labels", u2["original_accuracy_closed"]),
                  f"  wrong->right {u2['wrong_to_right']}  right->wrong {u2['right_to_wrong']}  net fixes {u2['net_fixes']}",
                  fmt_position("continued, closed", u2["position_closed"]),
                  fmt_position("forced, same prompts", u2["position_original_same_prompts"]),
                  fmt_accuracy("still open (forced again)", u2["still_open_accuracy"]),
                  "  new think tokens of the closed: " + (f"mean {tokens['mean']:.0f}, median {tokens['median']:.0f}" if tokens else "-"),
                  f"  agreement with the committed label: forced {fmt_pair(agree['forced_label'])}, "
                  f"continued {fmt_pair(agree['continued_label'])}"]
    if "unit3" in s:
        u3 = s["unit3"]
        lines += ["", f"unit 3 (to 8,192): {u3['attempted']} of the {u3['open_after_unit2']} traces open after unit 2 attempted, "
                      f"{u3['closed_in_unit3']} closed; closed within 8,192 of all unit-2 traces: "
                      f"{fmt_pair(u3['closed_within_8192_of_all_unit2'])} (unattempted counted open)"]
    lines += ["", "criteria:"]
    for name, c in s["criteria"].items():
        verdict = {True: "PASS", False: "FAIL", None: "not evaluated"}[c["pass"]]
        detail = ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {fmt_pair(v) if isinstance(v, list) else v}"
                           for k, v in c.items() if k != "pass")
        lines.append(f"  {name}: {verdict}   {detail}")
    return "\n".join(lines)


def summarize_root(run_root, inputs_dir):
    """(summary dict, text) over every bank folder under `run_root`."""
    summary, texts = {}, []
    for bank in ("choice4", "choice8"):
        bank_dir = Path(run_root) / bank
        if not bank_dir.exists():
            continue
        stage0 = json.loads((Path(inputs_dir) / f"stage0_{bank}.json").read_text())
        loaded = {unit: load_unit(bank_dir, unit) for unit in (1, 2, 3)}
        summary[bank] = summarize_bank(bank, stage0, {u: rows for u, (rows, _) in loaded.items()})
        texts.append(bank_text(summary[bank], {u: chunks for u, (_, chunks) in loaded.items()}))
    return summary, "\n\n".join(texts) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_root", help="<out>/exp17_Qwen3-8B_think_budget")
    parser.add_argument("--inputs", default=str(INPUTS_DIR), help="folder with stage0_<bank>.json")
    args = parser.parse_args()
    result, text = summarize_root(args.run_root, args.inputs)
    (Path(args.run_root) / "think_budget_summary.json").write_text(json.dumps(result, indent=1))
    (Path(args.run_root) / "think_budget_summary.txt").write_text(text)
    print(text)
