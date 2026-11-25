"""E19 analysis: how much of the full bank's answer-slot steering survives on each eval type, and
does it fall with the type's distance from Alan's prompt? (EvalsPromptsV2.md §8)

    python -m pipeline.s3_analysis.eval_types_steer $RESULTS/exp19_Qwen3-8B_evaltypes \\
        steering/inputs_Qwen3-8B_evaltypes.json $RESULTS/evaltypes_Qwen3-8B_think-on

Per type and mode, from the type's own rows (twins apart):
- baseline: accuracy of the unsteered choice (near log-odds > 0 means the near option), order
  agreement (base and swapped rows of an item choose the same option), and the contrast (mean near
  log-odds of near-regime rows minus both-regime rows);
- gate, fixed before the run: accuracy >= 0.65, order agreement >= 0.75, contrast >= 2 logits. A
  type that fails has no stable decision to steer; its effects are reported but not interpreted;
- toward: the effect, in units of the type's contrast, toward the answer the steered horizon makes
  rational, on the rows where it changes: near rows at +2 decades (toward far), both rows at -2
  (toward near). Curve, random, and their per-row difference (specific);
- flips: share of those rows whose choice turns to the newly rational option under the curve;
- ceiling: E8's T5 patch transfer at L22-L28 within the type, and the first layer reaching 0.5.
References: `t0_w0` (Alan's wording, in this run), the twins of all types pooled (Alan's template)
and E8 on the full bank (BANK). Across gated types: Spearman of specific toward and of toward over
the L24 ceiling with the distances in eval_distance.json. Thinking on: the unsteered log-odds are
checked against the label logits E0 saved for the same reasoning.
Writes eval_steer.json and eval_steer.md into the results folder.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

GATE = {"accuracy": 0.65, "order": 0.75, "contrast": 2.0}
PATCH_LAYERS = (22, 24, 26, 28)
# E8 on the full bank (08_results_steering.md §5): curve at the pre-label token, L24, toward the
# rational answer at the crossing conditions (both -2, near +2), in contrast units
BANK = {"off": {"both -2": 0.38, "near +2": 0.39}, "on": {"both -2": 0.15, "near +2": 0.14}}
DISTANCES = (("L18_suffix_end", "full"), ("L18_suffix_end", "horizon"), ("L18_suffix_end", "probe_shift"),
             ("L24_prelabel", "full"), ("L24_prelabel", "horizon"))
BOOT = 2000


def chose_near(lo):
    return np.asarray(lo) > 0


def bootstrap(values, rng):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) == 0:
        return None
    means = v[rng.integers(0, len(v), size=(BOOT, len(v)))].mean(axis=1)
    return {"n": len(v), "mean": round(float(v.mean()), 3),
            "ci95": [round(float(q), 3) for q in np.percentile(means, [2.5, 97.5])]}


def baseline(rows):
    """Accuracy, order agreement and contrast of the unsteered choice."""
    lo = np.array([r["base"] for r in rows])
    regime = np.array([r["regime"] for r in rows])
    correct = chose_near(lo) == (regime == "near")
    by_item = {}
    for r in rows:
        by_item.setdefault(r["twin_id"], {})[r["variant"]] = r["base"] > 0
    pairs = [v for v in by_item.values() if "base" in v and "swap" in v]
    order = float(np.mean([v["base"] == v["swap"] for v in pairs])) if pairs else float("nan")
    has_both_regimes = (regime == "near").any() and (regime == "both").any()
    contrast = float(lo[regime == "near"].mean() - lo[regime == "both"].mean()) if has_both_regimes else float("nan")
    out = {"n": len(rows), "accuracy": round(float(correct.mean()), 3), "order": round(order, 3),
           "n_order_items": len(pairs), "contrast": round(contrast, 2)}
    out["gate"] = bool(out["accuracy"] >= GATE["accuracy"] and order >= GATE["order"]
                       and contrast >= GATE["contrast"])
    return out


def crossing(rows, direction, step=2):
    """Per row, (toward, flipped) under `direction`, on the rows the step crosses: near rows at
    +step decades, both rows at -step. `toward` is in log-odds, positive toward the newly rational
    option; `flipped` is None when the row already chose that option unsteered."""
    out = []
    for r in rows:
        if r["regime"] == "near":
            edited, sign = r[f"{direction} {step:+d}"], -1.0    # longer horizon: toward far, near log-odds down
        else:
            edited, sign = r[f"{direction} {-step:+d}"], 1.0    # shorter horizon: toward near, near log-odds up
        toward = sign * (edited - r["base"])
        chose_old_rational = (r["base"] > 0) == (r["regime"] == "near")
        flipped = (edited > 0) != (r["base"] > 0) if chose_old_rational else None
        out.append((toward, flipped))
    return out


def steering(rows, contrast, rng):
    curve, random = crossing(rows, "curve"), crossing(rows, "random 0")
    toward_c = [t / contrast for t, _ in curve]
    toward_r = [t / contrast for t, _ in random]
    flips = [f for _, f in curve if f is not None]
    return {"toward_curve": bootstrap(toward_c, rng), "toward_random": bootstrap(toward_r, rng),
            "specific": bootstrap([c - r for c, r in zip(toward_c, toward_r)], rng),
            "toward_curve_logits": bootstrap([t for t, _ in curve], rng),
            "flips": {"n": len(flips), "k": int(sum(flips)), "rate": round(float(np.mean(flips)), 3) if flips else None}}


def ceiling(patch):
    if patch is None:
        return None
    transfer = [t[0] for t in patch["transfer"]]
    onset = next((layer for layer, t in enumerate(transfer) if layer and np.isfinite(t) and t >= 0.5), None)
    return {"n_used": patch["n_used"], "onset": onset,
            **{f"L{layer}": round(transfer[layer], 3) for layer in PATCH_LAYERS}}


def saved_check(rows):
    """|unsteered near log-odds - E0's saved one| for the same reasoning (thinking on)."""
    gaps = []
    for r in rows:
        a, b = r["label_logits"]
        saved = a - b if r["near_label"] == "a" else b - a
        gaps.append(abs(r["base"] - saved))
    return {"n": len(gaps), "median": round(float(np.median(gaps)), 3), "max": round(float(np.max(gaps)), 3)}


def joined_rows(results, meta, mode):
    info = {r["sample_uid"]: r for r in meta["rows"]}
    return [{**info[x["sample_uid"]], **x} for x in results["modes"][mode]["rows"]]


def analyse(results, meta, distance):
    rng = np.random.default_rng(0)
    out = {"gate": GATE, "bank": BANK, "modes": {}}
    for mode in results["modes"]:
        rows = joined_rows(results, meta, mode)
        patch = results["modes"][mode].get("patch", {})
        types = {}
        for type_id in dict.fromkeys(r["type_id"] for r in rows):
            own = [r for r in rows if r["type_id"] == type_id and r["variant"] != "twin"]
            base = baseline(own)
            types[type_id] = {"baseline": base, "steer": steering(own, base["contrast"], rng),
                              "ceiling": ceiling(patch.get(type_id)),
                              "distance": {f"{cell} {measure}": _distance(distance, type_id, cell, measure)
                                           for cell, measure in DISTANCES}}
        twins = [r for r in rows if r["variant"] == "twin"]
        twin_base = baseline(twins)
        mode_out = {"types": types,
                    "twins_pooled": {"baseline": twin_base, "steer": steering(twins, twin_base["contrast"], rng)},
                    "correlations": correlations(types)}
        if mode == "on":
            mode_out["saved_check"] = saved_check(rows)
        out["modes"][mode] = mode_out
    return out


def _distance(distance, type_id, cell, measure):
    d = distance["types"].get(type_id, {}).get("distance", {}).get(cell)
    if d is None:
        return None
    v = d[measure] if measure == "probe_shift" else d[measure]["paired"]
    return None if v is None else v["median"]


def correlations(types):
    """Across gated types: Spearman of specific toward, and of curve toward over the L24 ceiling,
    with each distance."""
    gated = [t for t, x in types.items() if x["baseline"]["gate"]]
    out = {}
    for key in (f"{c} {m}" for c, m in DISTANCES):
        for target in ("specific", "toward / L24 ceiling"):
            pairs = []
            for t in gated:
                d = types[t]["distance"][key]
                s = types[t]["steer"]
                if target == "specific":
                    y = s["specific"]["mean"] if s["specific"] else None
                else:
                    c = types[t]["ceiling"]
                    y = (s["toward_curve"]["mean"] / c["L24"]) if (c and s["toward_curve"] and c["L24"] > 0.1) else None
                if d is not None and y is not None:
                    pairs.append((d, y))
            if len(pairs) >= 5:
                rho, p = spearmanr(*zip(*pairs))
                out[f"{key} vs {target}"] = {"n": len(pairs), "spearman": round(float(rho), 3), "p": round(float(p), 3)}
    return out


def _ci(v):
    return "-" if v is None else f"{v['mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"


def markdown(out):
    lines = ["# E19: the full bank's answer-slot steering on the eval types (Qwen3-8B, L24)", "",
             f"Gate (fixed before the run): accuracy >= {GATE['accuracy']}, order agreement >= {GATE['order']}, "
             f"contrast >= {GATE['contrast']} logits. Toward = effect toward the newly rational option on the rows "
             "the ±2-decade step crosses, in units of the type's contrast. Bank (E8): "
             + "; ".join(f"thinking {m}: " + ", ".join(f"{k} {v:.2f}" for k, v in b.items()) for m, b in BANK.items())
             + ". Ceiling = within-type answer-slot patch transfer.", ""]
    for mode, m in out["modes"].items():
        lines += [f"## Thinking {mode}", ""]
        if "saved_check" in m:
            s = m["saved_check"]
            lines += [f"Unsteered vs E0's saved log-odds (same reasoning): median |gap| {s['median']}, max {s['max']} "
                      f"over {s['n']} rows.", ""]
        lines += ["| type | gate | acc | order | contrast | toward: curve | random | specific | flips | "
                  "ceiling L22 / L24 / L26 / L28 (onset) | L18 suffix horizon dist |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        tw = m["twins_pooled"]
        ordered = sorted(m["types"], key=lambda t: (m["types"][t]["distance"]["L18_suffix_end full"] or 0))
        for name, x in [("(twins pooled, Alan template)", tw)] + [(t, m["types"][t]) for t in ordered]:
            b, s, c = x["baseline"], x["steer"], x.get("ceiling")
            f = s["flips"]
            ceil = "-" if not c else (" / ".join(f"{c[f'L{l}']:.2f}" for l in PATCH_LAYERS) + f" ({c['onset']})")
            d = x.get("distance", {}).get("L18_suffix_end horizon")
            lines.append(f"| {name} | {'pass' if b['gate'] else 'FAIL'} | {b['accuracy']:.2f} | {b['order']:.2f} "
                         f"| {b['contrast']:.1f} | {_ci(s['toward_curve'])} | {_ci(s['toward_random'])} "
                         f"| {_ci(s['specific'])} | {f['k']}/{f['n']} | {ceil} | {'-' if d is None else f'{d:.2f}'} |")
        lines += ["", "Across gated types (Spearman):", ""]
        lines += [f"- {k}: {v['spearman']:+.2f} (n = {v['n']}, p = {v['p']:.3f})" for k, v in m["correlations"].items()]
        lines.append("")
    return "\n".join(lines) + "\n"


def main(results_dir, inputs_json, eval_run):
    results_dir = Path(results_dir)
    results = json.loads((results_dir / "results.json").read_text())
    meta = json.loads(Path(inputs_json).read_text())
    distance = json.loads((Path(eval_run) / "eval_distance.json").read_text())
    out = analyse(results, meta, distance)
    (results_dir / "eval_steer.json").write_text(json.dumps(out, indent=1))
    md = markdown(out)
    (results_dir / "eval_steer.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(*sys.argv[1:4])
