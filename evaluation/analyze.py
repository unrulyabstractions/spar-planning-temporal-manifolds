"""Report card from replay_eval.py's outputs (CPU only).

Per arm (method, control or null draw), on the verification grid, from y = logit(far) - logit(near) at the answer slot:
  odd    = (y(+x) - y(-x)) / 2           the steering part (P2)
  even   = (y(+x) + y(-x)) / 2 - y(0)    the disruption part
  meaning / letter (C1): per verification pair (same item, both option orders),
         meaning = mean of the two orders' odd parts (push toward far whatever its label),
         letter  = half their difference, far-first minus near-first (push toward the first-listed label)
  switch point (P1, C2): a logistic P(far) = sigmoid(s * (log10 H/D - loc)) fitted to the per-prompt replay
         probabilities; delta_log10_H = (loc(-x) - loc(+x)) / 2, in decades of horizon (positive = acts as if the
         horizon were longer). Plateaus: mean P(far) where H/D < 1/2 (near is clearly correct) and > 2.
  label mass: p(far) + p(near) at the answer slot, a health proxy (drops when the edit breaks the format).
Each method's meaning statistic is ranked against every null family: p = (1 + #{|null| >= stat}) / (K + 1).

Usage: python analyze.py --out <replay_eval out dir>   -> report.json, report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

import nulls


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_switch(logr, p):
    """Location and slope of P(far) vs log10(H/D); nan if the fit fails."""
    try:
        (loc, s), _ = curve_fit(lambda x, loc, s: sigmoid(s * (x - loc)), logr, p, p0=[0.0, 3.0],
                                bounds=([-3, 0.01], [3, 200]), maxfev=5000)
        return float(loc), float(s)
    except Exception:
        return float("nan"), float("nan")


def arm_stats(items, y0, rec, x):
    logr = np.log10([it["h_over_far_delay"] for it in items])
    out = {}
    yp = rec["y_plus"]
    if "y_minus" in rec:
        ym = rec["y_minus"]
        odd, even = nulls.odd_even(yp, ym)
        even = even - y0
    else:                                    # single-request arm (snap at 0): effect vs unsteered
        odd, even = yp - y0, np.zeros_like(yp)
        ym = None
    out["odd_mean"] = float(odd.mean())
    out["even_mean"] = float(even.mean())
    out["abs_change_mean"] = float(np.abs(yp - y0).mean())
    # C1 split by verification pair
    by = {}
    for n, it in enumerate(items):
        by.setdefault(it["pair_id"], {})["far" if not it["near_first"] else "near"] = n
    pairs = [(d["far"], d["near"]) for d in by.values() if len(d) == 2]
    if pairs:
        f, nn = np.array(pairs).T
        out["meaning"] = float(((odd[f] + odd[nn]) / 2).mean())
        out["letter"] = float(((odd[f] - odd[nn]) / 2).mean())
        out["n_pairs"] = len(pairs)
    # switch point
    loc0, s0 = fit_switch(logr, sigmoid(y0))
    locp, sp = fit_switch(logr, sigmoid(yp))
    out["switch_loc_plus"], out["slope_plus"] = locp, sp
    if ym is not None:
        locm, sm = fit_switch(logr, sigmoid(ym))
        out["switch_loc_minus"], out["slope_minus"] = locm, sm
        out["delta_log10_H"] = (locm - locp) / 2
    else:
        out["delta_log10_H"] = loc0 - locp
    low, high = logr < np.log10(0.5), logr > np.log10(2)
    for tag, y in (("plus", yp), ("minus", ym)):
        if y is not None:
            out[f"plateau_low_{tag}"] = float(sigmoid(y[low]).mean())
            out[f"plateau_high_{tag}"] = float(sigmoid(y[high]).mean())
    for tag in ("plus", "minus"):
        if f"mass_{tag}" in rec:
            out[f"label_mass_{tag}"] = float(rec[f"mass_{tag}"].mean())
    return out


def arm_results(out: Path, items, y0, keep=None) -> dict:
    """arm_stats of every arm in arms.jsonl with saved results, on the items where the boolean mask `keep` is True
    (all items when None)."""
    if keep is not None:
        items = [it for it, k in zip(items, keep) if k]
        y0 = y0[keep]
    res = {}
    for arm in read_jsonl(out / "arms.jsonl"):
        path = out / "arms" / (hashlib.sha1(arm["arm"].encode()).hexdigest()[:12] + ".npz")
        if path.exists():
            rec = dict(np.load(path))
            if keep is not None:
                rec = {k: v[keep] for k, v in rec.items()}
            res[arm["arm"]] = {**arm, **arm_stats(items, y0, rec, arm["x"])}
    return res


# Families reported as arms, never pooled as nulls. Prompting arms (replay_eval --prompting) have no matched null
# family; their floor is the prompt:neutral sentence, so they are reported but not ranked.
REPORTED = ("method", "control", "prompting")


def compare_to_nulls(res: dict, stats=("meaning", "delta_log10_H")) -> dict:
    """Each method and control (not curve-snap@0) ranked against every null family of its kind, per statistic."""
    fams = {}
    for r in res.values():
        if r["family"] not in REPORTED:
            fams.setdefault(r["family"], []).append(r)
    comps = {}
    for name, r in res.items():
        if r["family"] not in ("method", "control") or name == "curve-snap@0":
            continue
        comps[name] = {}
        for fam, rows in fams.items():
            if fam.startswith("shuffled:") and fam != f"shuffled:{name}":
                continue
            if (r["kind"] == "target") != (rows[0]["kind"] == "target"):
                continue                           # push methods vs push nulls; curves vs shuffled curves
            for stat in stats:
                null = np.array([q.get(stat, np.nan) for q in rows], float)
                null = null[np.isfinite(null)]
                if len(null) and np.isfinite(r.get(stat, np.nan)):
                    comps[name].setdefault(fam, {})[stat] = {
                        "stat": r[stat], "p": nulls.rank_p(r[stat], null), "K": len(null),
                        "null_p95_abs": float(np.percentile(np.abs(null), 95))}
    return comps


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    items = [it for it in read_jsonl(out / "items.jsonl") if it["split"] == "verification"]
    base = np.load(out / "arms" / "_none.npz")
    y0 = base["y"]
    logr = np.log10([it["h_over_far_delay"] for it in items])
    loc0, s0 = fit_switch(logr, sigmoid(y0))
    res = arm_results(out, items, y0)
    comps = compare_to_nulls(res)
    cal = json.loads((out / "calibration.json").read_text())
    n0 = json.loads((out / "n0_check.json").read_text()) if (out / "n0_check.json").exists() else {}
    skipped = read_jsonl(out / "skipped.jsonl")
    n_ver_skipped = sum(1 for s in skipped if s["split"] == "verification")
    gen = {}
    gdir = out / "gencheck"
    if gdir.exists():
        for f in sorted(gdir.glob("*.jsonl")):
            rows = read_jsonl(f)
            ok = [r for r in rows if r["parsed"]]
            gen[f.stem] = {"n": len(rows), "samples": len({r.get("sample", 0) for r in rows}), "parsed": len(ok), "p_far": float(np.mean([r["chose_far"] for r in ok])) if ok else None,
                           "hit_cap": sum(r["hit_cap"] for r in rows),
                           "mean_tokens": float(np.mean([r["n_generated"] for r in rows]))}
    report = {"n_items": len(items), "n_verification_unreadable": n_ver_skipped, "calibration": cal, "n0": n0,
              "unsteered": {"switch_loc": loc0, "slope": s0, "label_mass": float(base["mass"].mean())},
              "arms": res, "comparisons": comps, "gencheck": gen}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")

    L = ["# Report card (replay readout)", "",
         f"- Verification items readable: {len(items)} (unreadable: {n_ver_skipped}, listed in skipped.jsonl)",
         f"- Push strength x = {cal['x_push']:g} × typical norm (calibrated on held-out fold {cal['fold']}, "
         f"target {cal['target_logit']} logit, reached: {cal['reached_target']}); target shift = {cal['x_shift']:g} decades",
         f"- N0: zero edit vs hook-free max |Δy| = {n0.get('max_abs_diff_vs_hook_free')}; unsteered replay matches the "
         f"generated choice on {n0.get('replay_matches_generated_choice', float('nan')):.1%}",
         f"- Unsteered switch point log10(H/D) = {loc0:+.3f}, slope {s0:.2f}, label mass {base['mass'].mean():.3f}", "",
         "## Methods and controls", "",
         "| arm | x | meaning (logit) | letter | even | Δlog10 H* | slope ± | low/high plateau (+x) | label mass ± |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        if r["family"] in REPORTED:
            L.append(f"| {name} | {r['x']:g} | {r.get('meaning', float('nan')):+.3f} | {r.get('letter', float('nan')):+.3f} | "
                     f"{r['even_mean']:+.3f} | {r['delta_log10_H']:+.3f} | {r.get('slope_plus', float('nan')):.2f} / "
                     f"{r.get('slope_minus', float('nan')):.2f} | {r.get('plateau_low_plus', float('nan')):.2f} / "
                     f"{r.get('plateau_high_plus', float('nan')):.2f} | {r.get('label_mass_plus', float('nan')):.3f} / "
                     f"{r.get('label_mass_minus', float('nan')):.3f} |")
    L += ["", "## Against the nulls (rank p; |null| 95th percentile)", "",
          "| method | null family | K | meaning: stat / p95 / p | Δlog10 H*: stat / p95 / p |", "|---|---|---|---|---|"]
    for name, by in comps.items():
        for fam, st in by.items():
            m, d = st.get("meaning", {}), st.get("delta_log10_H", {})
            L.append(f"| {name} | {fam} | {m.get('K', d.get('K'))} | "
                     f"{m.get('stat', float('nan')):+.3f} / {m.get('null_p95_abs', float('nan')):.3f} / {m.get('p', float('nan')):.2f} | "
                     f"{d.get('stat', float('nan')):+.3f} / {d.get('null_p95_abs', float('nan')):.3f} / {d.get('p', float('nan')):.2f} |")
    if gen:
        L += ["", "## Generation check (subset)", "", "| arm | samples | parsed | P(far) | hit cap | mean tokens |",
              "|---|---|---|---|---|---|"]
        for k, g in gen.items():
            L.append(f"| {k} | {g['samples']} | {g['parsed']}/{g['n']} | {g['p_far'] if g['p_far'] is None else round(g['p_far'], 2)} | "
                     f"{g['hit_cap']} | {g['mean_tokens']:.0f} |")
    L += ["", "Plumbing run: the model is small and the numbers are not evidence about steering."]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
