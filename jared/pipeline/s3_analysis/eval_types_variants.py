"""E21 analysis: dose, direction, layer, stacking and removal of the answer-slot edit on the eval
types (EvalsPromptsV2.md §9).

    python -m pipeline.s3_analysis.eval_types_variants $RESULTS/exp21_Qwen3-8B_evaltypes \\
        steering/inputs_Qwen3-8B_evaltypes.json

Per type and mode, from the type's own rows (twins apart), with E19's baseline and gate:
- toward: the effect toward the newly rational option on the rows the step crosses (near rows at
  +step, both rows at -step), in units of the type's contrast; flips as in E19;
- dose: the L24 curve at 1, 2, 4 decades, random 0 at 2 and 4, and the step the rows actually got
  (the curve ends at the bank's shortest and longest horizon);
- direction: curve, mean difference, line and two random directions at L24, 2 decades;
- layer: the curve and random 0 at each layer of the sweep, and the stacked L22-L28 step;
- removal: share of the near-minus-both contrast that mean-ablating the top-k horizon subspace
  removes, against the largest of three random subspaces at the same k.
Readouts fixed before the run, across gated types (bootstrap over types):
- dose: ratio of mean toward at 4 and at 2 decades (near 2 = linear); clean if random 0 at 4
  stays <= 0.1 in every gated type;
- direction: curve minus mean difference, per type then averaged; the shape matters if its CI
  excludes 0;
- removal: the horizon subspace is needed at the answer token if it removes more than the largest
  random subspace, per type and k.
Writes eval_variants.json and eval_variants.md into the results folder.
"""

import json
import sys
from pathlib import Path

import numpy as np

from pipeline.s3_analysis.eval_types_steer import GATE, baseline, bootstrap, crossing, joined_rows
from spar_horizon.steering import removed_fraction

LAYER = 24
SWEEP = (16, 20, 22, 24, 26, 28, 32)
STACK = "L22+24+26+28"
DOSES = (1, 2, 4)
DIRECTIONS = ("curve", "mean difference", "line", "random 0", "random 1")
ABLATE_KS = (1, 2, 4, 8)
N_RANDOM = 3
RANDOM_LIMIT = 0.1
BOOT = 2000


def toward(rows, condition, step, contrast, rng):
    c = crossing(rows, condition, step)
    flips = [f for _, f in c if f is not None]
    return {"toward": bootstrap([t / contrast for t, _ in c], rng), "flips_k": int(sum(flips)), "flips_n": len(flips)}


def effective_step(rows, step, curve_range):
    """Mean decades the crossing rows actually move: steps stop at the ends of the bank's curve."""
    lo, hi = curve_range
    moved = []
    for r in rows:
        log_h = np.log10(r["horizon"])
        moved.append(min(log_h + step, hi) - log_h if r["regime"] == "near" else log_h - max(log_h - step, lo))
    return round(float(np.mean(moved)), 2)


def removal(rows):
    base = np.array([r["base"] for r in rows])
    regimes = [r["regime"] for r in rows]
    out = {}
    for k in ABLATE_KS:
        horizon = removed_fraction(base, np.array([r[f"L{LAYER} ablate horizon k{k}"] for r in rows]), regimes)
        randoms = [removed_fraction(base, np.array([r[f"L{LAYER} ablate random {s} k{k}"] for r in rows]), regimes)[0]
                   for s in range(N_RANDOM)]
        out[k] = {"horizon": [round(v, 3) for v in horizon], "random_max": round(max(randoms), 3),
                  "beats_random": bool(horizon[1] > max(randoms))}
    return out


def type_summary(rows, curve_range, rng):
    base = baseline(rows)
    c = base["contrast"]
    return {"baseline": base,
            "dose": {**{f"curve {s}": toward(rows, f"L{LAYER} curve", s, c, rng) for s in DOSES},
                     **{f"random 0 {s}": toward(rows, f"L{LAYER} random 0", s, c, rng) for s in (2, 4)}},
            "effective_step": {s: effective_step(rows, s, curve_range) for s in DOSES},
            "direction": {d: toward(rows, f"L{LAYER} {d}", 2, c, rng) for d in DIRECTIONS},
            "layer": {layer: {d: toward(rows, f"L{layer} {d}", 2, c, rng) for d in ("curve", "random 0")}
                      for layer in SWEEP},
            "stacked": toward(rows, f"{STACK} curve", 2, c, rng),
            "removal": removal(rows)}


def mean_of(x):
    return x["toward"]["mean"] if x["toward"] else np.nan


def over_types(values, rng):
    """Mean over gated types, 95% CI by resampling types."""
    v = np.array([x for x in values if np.isfinite(x)])
    if len(v) < 2:
        return None
    means = v[rng.integers(0, len(v), size=(BOOT, len(v)))].mean(axis=1)
    return {"n_types": len(v), "mean": round(float(v.mean()), 3),
            "ci95": [round(float(q), 3) for q in np.percentile(means, [2.5, 97.5])]}


def readouts(types, rows_by_type, rng):
    gated = [t for t, x in types.items() if x["baseline"]["gate"]]
    t2 = over_types([mean_of(types[t]["dose"]["curve 2"]) for t in gated], rng)
    t4 = over_types([mean_of(types[t]["dose"]["curve 4"]) for t in gated], rng)
    random4 = {t: mean_of(types[t]["dose"]["random 0 4"]) for t in gated}
    # curve minus mean difference row by row, averaged per type, then over types
    shape = []
    for t in gated:
        rows, c = rows_by_type[t], types[t]["baseline"]["contrast"]
        curve = [x for x, _ in crossing(rows, f"L{LAYER} curve")]
        caa = [x for x, _ in crossing(rows, f"L{LAYER} mean difference")]
        shape.append(float(np.mean(np.subtract(curve, caa))) / c)
    best_layer = {t: max(SWEEP, key=lambda layer: mean_of(types[t]["layer"][layer]["curve"])) for t in gated}
    return {"gated": gated,
            "dose": {"toward 2": t2, "toward 4": t4,
                     "ratio 4/2": round(t4["mean"] / t2["mean"], 2) if t2 and t4 and t2["mean"] else None,
                     "random 0 at 4, max over types": round(max(random4.values()), 3) if random4 else None,
                     "clean": bool(random4) and max(random4.values()) <= RANDOM_LIMIT},
            "direction": {"curve minus mean difference": over_types(shape, rng),
                          **{d: over_types([mean_of(types[t]["direction"][d]) for t in gated], rng)
                             for d in DIRECTIONS}},
            "layer": {"curve": {layer: over_types([mean_of(types[t]["layer"][layer]["curve"]) for t in gated], rng)
                                for layer in SWEEP},
                      "best layer per type": best_layer,
                      "stacked": over_types([mean_of(types[t]["stacked"]) for t in gated], rng)},
            "removal": {k: {"types beating random": sum(types[t]["removal"][k]["beats_random"] for t in gated),
                            "of": len(gated)} for k in ABLATE_KS}}


def analyse(results, meta):
    out = {"gate": GATE, "modes": {}}
    for mode, m in results["modes"].items():
        rng = np.random.default_rng(0)
        rows = joined_rows(results, meta, mode)
        curve_range = m["curve_range"][str(LAYER)]
        rows_by_type = {}
        for r in rows:
            if r["variant"] != "twin":
                rows_by_type.setdefault(r["type_id"], []).append(r)
        types = {t: type_summary(own, curve_range, rng) for t, own in rows_by_type.items()}
        twins = [r for r in rows if r["variant"] == "twin"]
        out["modes"][mode] = {"types": types, "readouts": readouts(types, rows_by_type, rng),
                              "twins_pooled": type_summary(twins, curve_range, rng) if twins else None}
    return out


def _m(x):
    v = mean_of(x)
    return "-" if not np.isfinite(v) else f"{v:+.2f}"


def _ci(v):
    return "-" if v is None else f"{v['mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"


def markdown(out):
    lines = ["# E21: answer-slot variants on the eval types (Qwen3-8B)", "",
             f"Toward = effect toward the newly rational option on the rows the step crosses, in units of the "
             f"type's contrast (E19). Gate: accuracy >= {GATE['accuracy']}, order >= {GATE['order']}, contrast >= "
             f"{GATE['contrast']}. Types in alphabetical order; those that fail the gate are marked.", ""]
    for mode, m in out["modes"].items():
        R = m["readouts"]
        entries = ([("(twins pooled)", m["twins_pooled"])] if m["twins_pooled"] else []) + sorted(m["types"].items())
        name = lambda t, x: t if x["baseline"]["gate"] or t.startswith("(") else f"{t} (FAIL)"
        lines += [f"## Thinking {mode}", "", "### Readouts (gated types, CI over types)", "",
                  f"- dose: toward at 2 decades {_ci(R['dose']['toward 2'])}, at 4 {_ci(R['dose']['toward 4'])}, "
                  f"ratio {R['dose']['ratio 4/2']}; random 0 at 4, largest over types "
                  f"{R['dose']['random 0 at 4, max over types']} ({'clean' if R['dose']['clean'] else 'NOT clean'})",
                  f"- direction: curve minus mean difference {_ci(R['direction']['curve minus mean difference'])}; "
                  + ", ".join(f"{d} {_ci(R['direction'][d])}" for d in DIRECTIONS),
                  "- layer (curve): " + ", ".join(f"L{layer} {_ci(v)}" for layer, v in R["layer"]["curve"].items())
                  + f"; stacked {_ci(R['layer']['stacked'])}",
                  "- removal, types where the horizon subspace removes more than every random one: "
                  + ", ".join(f"k={k} {v['types beating random']}/{v['of']}" for k, v in R["removal"].items()), ""]
        lines += ["### Dose (L24 curve)", "",
                  "| type | curve 1 | curve 2 | curve 4 | random 2 | random 4 | flips 2 | flips 4 | step got at 4 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for t, x in entries:
            d = x["dose"]
            lines.append(f"| {name(t, x)} | " + " | ".join(_m(d[k]) for k in ("curve 1", "curve 2", "curve 4",
                                                                              "random 0 2", "random 0 4"))
                         + f" | {d['curve 2']['flips_k']}/{d['curve 2']['flips_n']} "
                         f"| {d['curve 4']['flips_k']}/{d['curve 4']['flips_n']} | {x['effective_step'][4]} |")
        lines += ["", "### Direction (L24, 2 decades)", "", "| type | " + " | ".join(DIRECTIONS) + " |",
                  "|---" * (len(DIRECTIONS) + 1) + "|"]
        lines += [f"| {name(t, x)} | " + " | ".join(_m(x["direction"][d]) for d in DIRECTIONS) + " |" for t, x in entries]
        lines += ["", "### Layer (curve at 2 decades; random 0 in brackets)", "",
                  "| type | " + " | ".join(f"L{layer}" for layer in SWEEP) + f" | {STACK} |",
                  "|---" * (len(SWEEP) + 2) + "|"]
        lines += [f"| {name(t, x)} | " + " | ".join(f"{_m(x['layer'][layer]['curve'])} ({_m(x['layer'][layer]['random 0'])})"
                                                    for layer in SWEEP) + f" | {_m(x['stacked'])} |" for t, x in entries]
        lines += ["", "### Removal at L24 (share of contrast removed: horizon subspace / largest random)", "",
                  "| type | " + " | ".join(f"k={k}" for k in ABLATE_KS) + " |", "|---" * (len(ABLATE_KS) + 1) + "|"]
        lines += [f"| {name(t, x)} | " + " | ".join(
            f"{x['removal'][k]['horizon'][0]:+.2f} / {x['removal'][k]['random_max']:+.2f}"
            + ("*" if x["removal"][k]["beats_random"] else "") for k in ABLATE_KS) + " |" for t, x in entries]
        lines += ["", "\\* the horizon subspace's lower 95% bound is above the largest random subspace.", ""]
    return "\n".join(lines) + "\n"


def main(results_dir, inputs_json):
    results_dir = Path(results_dir)
    results = json.loads((results_dir / "results.json").read_text())
    meta = json.loads(Path(inputs_json).read_text())
    out = analyse(results, meta)
    (results_dir / "eval_variants.json").write_text(json.dumps(out, indent=1, default=str))
    md = markdown(out)
    (results_dir / "eval_variants.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(*sys.argv[1:3])
