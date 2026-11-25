"""E22 analysis: does a plain instruction move the choice like steering does, and does it change the
horizon or only lean toward later or sooner options? (EvalsPromptsV2.md §10)

    python -m pipeline.s3_analysis.eval_types_prompting $RESULTS/exp22_Qwen3-8B_prompting \\
        --e21 $RESULTS/exp21_Qwen3-8B_evaltypes/results.json --e20 $RESULTS/exp20_Qwen3-8B_evaltype_transfer

Per type and mode, on the type's own rows (twins apart, as E19), for four methods: the two prompting
families (disposition: long on near rows, short on both rows; shift: shift_long, shift_short) and E21's
steering (L24 curve +-2, and L22+24+26+28 curve +-2, from the rows already saved):
- toward: effect toward the answer the instruction (or the +-2 decades) makes rational, on the rows it
  crosses (`eval_types_steer.crossing`: near rows made longer, both rows shorter), in units of the
  type's contrast. Prompting's effect is the arm minus `neutral`;
- flips: share of the crossing rows `neutral` (or the unsteered row) answers correctly whose choice turns;
- spillover (thinking off): the same shift on the rows the instruction should not change (long on both
  rows, short on near rows, both on none rows; steering has no none rows), positive toward the option
  the instruction favours. A horizon change moves crossing rows only; a lean moves all.
The contrast and the gate statistics come from the `neutral` arm and serve prompting and steering alike.
Thinking on adds E20's donor-patch flip rate where present (it regenerates too, as prompting does).
Writes eval_prompting.json and eval_prompting.md into the prompting folder.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.s2_steering.experiments.prompting.exp22_prompting_baseline import MODES
from pipeline.s3_analysis.eval_types_steer import baseline, bootstrap, crossing

STEP = 2
# prompting arm -> (direction, sign of the step): long = +2 decades, short = -2, as in crossing()
PROMPT_KEYS = {"disposition": {"long": +STEP, "short": -STEP}, "shift": {"shift_long": +STEP, "shift_short": -STEP}}
STEERING = {"steering L24 curve ±2": "L24 curve", "steering L22+24+26+28 curve ±2": "L22+24+26+28 curve"}


def load_prompting(prompting_dir):
    """{mode: {type: {arm: {sample_uid: row}}}} of the arms finished so far."""
    out = {}
    for path in sorted(Path(prompting_dir).glob("*/*/conditions/*.json")):
        mode, type_id = path.parent.parent.name, path.parent.parent.parent.name
        result = json.loads(path.read_text())
        out.setdefault(mode, {}).setdefault(type_id, {})[result["arm"]] = {r["sample_uid"]: r for r in result["rows"]}
    return out


def prompting_rows(arms):
    """One row per item of the `neutral` arm, as `crossing` reads it: `base` is neutral's near log-odds
    and "<family> +2" / "<family> -2" the long and short arm's, where the arm ran on the row."""
    rows = []
    for uid, neutral in arms["neutral"].items():
        row = {k: neutral[k] for k in ("sample_uid", "regime", "variant", "twin_id")} | {"base": neutral["near_lo"]}
        for family, keys in PROMPT_KEYS.items():
            for arm, step in keys.items():
                if uid in arms.get(arm, {}):
                    row[f"{family} {step:+d}"] = arms[arm][uid]["near_lo"]
        rows.append(row)
    return rows


def steering_rows(e21_mode, own):
    """E21's rows of the same items, with the item fields of the prompting rows."""
    info = {r["sample_uid"]: r for r in own}
    return [{**info[x["sample_uid"]], **x} for x in e21_mode["rows"] if x["sample_uid"] in info]


def spillover(rows, direction, step=STEP):
    """Per row and instruction not meant to move it, the shift of the near log-odds toward the option
    the instruction favours (a longer horizon: far, near log-odds down). Rows without the key are skipped."""
    out = []
    for r in rows:
        keys = {"near": [-step], "both": [+step]}.get(r["regime"], [-step, +step])
        out += [(-1.0 if k > 0 else 1.0) * (r[f"{direction} {k:+d}"] - r["base"]) for k in keys if f"{direction} {k:+d}" in r]
    return out


def effect(rows, direction, contrast, rng, with_spillover):
    cross = crossing([r for r in rows if _has_crossing_key(r, direction)], direction, STEP)
    flips = [f for _, f in cross if f is not None]
    out = {"toward": bootstrap([t / contrast for t, _ in cross], rng),
           "flips": {"n": len(flips), "k": int(sum(flips)), "rate": round(float(np.mean(flips)), 3) if flips else None}}
    if with_spillover:
        out["spillover"] = bootstrap([s / contrast for s in spillover(rows, direction)], rng)
    return out


def _has_crossing_key(row, direction):
    """Whether the row is one the step crosses (near or both) and the condition ran on it."""
    return row["regime"] in ("near", "both") and f"{direction} {+STEP if row['regime'] == 'near' else -STEP:+d}" in row


def donor(e20_dir, type_id, contrast):
    """E20's donor arm for the type: its flip rate toward the target regime and its near log-odds shift
    against its noop arm in contrast units [mean, ci low, ci high]; None where not run."""
    path = Path(e20_dir) / type_id / "summary.json"
    if not path.exists():
        return None
    entry = json.loads(path.read_text()).get("donor")
    if entry is None:
        return None
    shift = entry.get("toward_log_odds_vs_noop")
    return {"n": entry["n"], "flipped_toward": round(entry["flipped_toward"], 3),
            "toward": None if shift is None else [round(x / contrast, 3) for x in shift]}


def analyse(prompting, e21=None, e20_dir=None):
    rng = np.random.default_rng(0)
    out = {"step": STEP, "modes": {}}
    for mode in MODES:
        out["modes"][mode] = types = {}
        for type_id, arms in prompting.get(mode, {}).items():
            if "neutral" not in arms:
                continue
            own = [r for r in prompting_rows(arms) if r["variant"] != "twin"]
            base = baseline([r for r in own if r["regime"] != "none"])
            methods = {f"prompt {family}": effect(own, family, base["contrast"], rng, mode == "off") for family in PROMPT_KEYS}
            if e21 is not None:
                steer = steering_rows(e21["modes"][mode], own)
                methods |= {name: effect(steer, key, base["contrast"], rng, mode == "off") for name, key in STEERING.items()}
            types[type_id] = {"baseline": base, "methods": methods}
            if mode == "on" and e20_dir is not None:
                types[type_id]["donor"] = donor(e20_dir, type_id, base["contrast"])
    return out


def _ci(v):
    return "-" if v is None else f"{v['mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"


def markdown(out):
    lines = ["# E22: prompting baseline against steering (Qwen3-8B)", "",
             "Toward = effect toward the answer the instruction makes rational, on the rows it crosses, in units of "
             "the type's contrast (neutral arm); prompting's effect is the arm minus `neutral`. Flips = crossing rows "
             "`neutral` answers correctly whose choice turns. Spillover = the same shift on the rows the instruction "
             "should not change, positive toward the favoured option (thinking off). Steering is E21's saved rows "
             "(thinking on: E0's reasoning held fixed).", ""]
    for mode, types in out["modes"].items():
        if not types:
            continue
        lines += [f"## Thinking {mode}", "", "| type | contrast | accuracy | method | toward | flips | spillover |",
                  "|---|---|---|---|---|---|---|"]
        for type_id, x in types.items():
            b = x["baseline"]
            for name, m in x["methods"].items():
                f = m["flips"]
                lines.append(f"| {type_id} | {b['contrast']:.1f} | {b['accuracy']:.2f} | {name} | {_ci(m['toward'])} "
                             f"| {f['k']}/{f['n']} | {_ci(m.get('spillover'))} |")
            d = x.get("donor")
            if d:
                toward = "-" if d["toward"] is None else f"{d['toward'][0]:+.2f}"
                lines.append(f"| {type_id} | | | E20 donor patch | {toward} "
                             f"| flip rate {d['flipped_toward']:.2f} (n {d['n']}) | |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(args):
    prompting_dir = Path(args.prompting_dir)
    e21 = json.loads(Path(args.e21).read_text()) if args.e21 else None
    out = analyse(load_prompting(prompting_dir), e21, args.e20)
    (prompting_dir / "eval_prompting.json").write_text(json.dumps(out, indent=1))
    md = markdown(out)
    (prompting_dir / "eval_prompting.md").write_text(md)
    print(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("prompting_dir", help="the exp22_<model>_prompting folder")
    parser.add_argument("--e21", default=None, help="E21's results.json, for the steering columns")
    parser.add_argument("--e20", default=None, help="E20's folder, for the donor flip rate (thinking on)")
    main(parser.parse_args())
