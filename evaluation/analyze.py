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
  N6 (damage-matched) is a family per method, n6:<method>: isotropic draws rescaled to the method's damage (replay_eval
  stage n6); a method is ranked only against its own.
  Confidence intervals (methods, controls, comparisons): paired bootstrap over verification pairs (both option orders
  of a pair are resampled together), --boot resamples, the same resamples for every arm; percentile 95% intervals.
  Readout agreement (generation check arms): per arm, how often the generated choice equals the replay prediction
  (y > 0 means far) on the same items, and the ± effect on P(far) in generation vs replay.

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

REPORTED = ("method", "control", "comparison")      # arms with their own rows in the report (not null draws)


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


def pair_clusters(items) -> list[list[int]]:
    """Item indices grouped by verification pair (the bootstrap's resampling unit)."""
    by = {}
    for n, it in enumerate(items):
        by.setdefault(it["pair_id"], []).append(n)
    return list(by.values())


def bootstrap_ci(items, y0, rec, x, B: int, seed: int = 0, stats=("meaning", "letter", "even_mean", "delta_log10_H")):
    """Percentile 95% intervals of arm_stats over pair-cluster resamples. Same seed = same resamples for every arm."""
    clusters = pair_clusters(items)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in stats}
    for _ in range(B):
        pick = rng.integers(0, len(clusters), len(clusters))
        idx = np.concatenate([clusters[c] for c in pick])
        sub_items = [dict(items[i], pair_id=f"{items[i]['pair_id']}#{j}") for j, c in enumerate(pick) for i in clusters[c]]
        r = arm_stats(sub_items, y0[idx], {k: v[idx] for k, v in rec.items()}, x)
        for k in stats:
            draws[k].append(r.get(k, np.nan))
    out = {}
    for k, v in draws.items():
        v = np.array(v, float)
        v = v[np.isfinite(v)]
        out[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 0.9 * B else None
    return out


GEN_TO_REPLAY = {"none": (None, 0), "caa+": ("caa", +1), "caa-": ("caa", -1), "isotropic[0]+": ("isotropic[0]", +1),
                 "isotropic[0]-": ("isotropic[0]", -1), "curve-displace+": ("curve-displace", +1),
                 "curve-displace-": ("curve-displace", -1)}


def readout_agreement(out: Path, items, y0, arm_file, B: int = 1000, seed: int = 0) -> dict:
    """Generation check vs replay on the same items and requests. Per generation arm: the share of parsed generations
    whose choice equals the replay prediction (y > 0 = far), and both readouts' P(far). Per ± pair: the odd effect on
    P(far), (P(+) - P(-)) / 2, in generation (mean over samples) and in replay (mean sigmoid(y)), with a bootstrap
    interval for generation's over items."""
    gdir = out / "gencheck"
    if not gdir.exists():
        return {}
    pos = {it["uid"]: n for n, it in enumerate(items)}
    replay_y = {}
    for name, (arm, sign) in GEN_TO_REPLAY.items():
        if arm is None:
            replay_y[name] = y0
        elif arm_file(arm) is not None:
            r = arm_file(arm)
            replay_y[name] = r["y_plus"] if sign > 0 else r.get("y_minus")
    per_arm, gen_p = {}, {}
    for f in sorted(gdir.glob("*.jsonl")):
        name = f.stem
        if replay_y.get(name) is None:
            continue
        rows = [r for r in read_jsonl(f) if r["parsed"] and r["uid"] in pos]
        y = replay_y[name]
        agree = [r["chose_far"] == bool(y[pos[r["uid"]]] > 0) for r in rows]
        by_item = {}
        for r in rows:
            by_item.setdefault(r["uid"], []).append(float(r["chose_far"]))
        gen_p[name] = {u: float(np.mean(v)) for u, v in by_item.items()}
        uids = sorted(by_item)
        per_arm[name] = {"n_parsed": len(rows), "n_items": len(uids),
                         "choice_agreement": float(np.mean(agree)) if agree else None,
                         "p_far_generated": float(np.mean([gen_p[name][u] for u in uids])) if uids else None,
                         "p_far_replay": float(np.mean(sigmoid(y[[pos[u] for u in uids]]))) if uids else None}
    effects = {}
    rng = np.random.default_rng(seed)
    for base in ("caa", "isotropic[0]", "curve-displace"):
        a, b = f"{base}+", f"{base}-"
        if a not in gen_p or b not in gen_p:
            continue
        both = sorted(set(gen_p[a]) & set(gen_p[b]))
        if not both:
            continue
        g = np.array([(gen_p[a][u] - gen_p[b][u]) / 2 for u in both])
        r = np.array([(sigmoid(replay_y[a][pos[u]]) - sigmoid(replay_y[b][pos[u]])) / 2 for u in both])
        boots = [g[rng.integers(0, len(g), len(g))].mean() for _ in range(B)]
        effects[base] = {"n_items": len(both), "odd_p_far_generated": float(g.mean()),
                         "odd_p_far_generated_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                         "odd_p_far_replay": float(r.mean()),
                         "same_sign": bool(np.sign(g.mean()) == np.sign(r.mean())) if g.mean() and r.mean() else None}
    return {"arms": per_arm, "effects": effects}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--boot", type=int, default=1000, help="bootstrap resamples for the confidence intervals (0: none)")
    a = p.parse_args(argv)
    out = Path(a.out)
    items = [it for it in read_jsonl(out / "items.jsonl") if it["split"] == "verification"]
    arms = read_jsonl(out / "arms.jsonl")
    base = np.load(out / "arms" / "_none.npz")
    y0 = base["y"]
    logr = np.log10([it["h_over_far_delay"] for it in items])
    loc0, s0 = fit_switch(logr, sigmoid(y0))
    res = {}

    def arm_file(name):
        path = out / "arms" / (hashlib.sha1(name.encode()).hexdigest()[:12] + ".npz")
        return dict(np.load(path)) if path.exists() else None

    for arm in arms:
        rec = arm_file(arm["arm"])
        if rec is not None:
            res[arm["arm"]] = {**arm, **arm_stats(items, y0, rec, arm["x"])}
            if a.boot and arm["family"] in REPORTED:
                res[arm["arm"]]["ci"] = bootstrap_ci(items, y0, rec, arm["x"], a.boot)
    fams = {}
    for r in res.values():
        if r["family"] not in REPORTED:
            fams.setdefault(r["family"], []).append(r)
    comps = {}
    for name, r in res.items():
        if r["family"] not in REPORTED or name == "curve-snap@0":
            continue
        comps[name] = {}
        for fam, rows in fams.items():
            if fam.startswith("n6:"):
                if fam != f"n6:{name}":
                    continue                       # N6 is matched to one method's damage: only that method
            else:
                if fam.startswith("shuffled:") and fam != f"shuffled:{name}":
                    continue
                if (r["kind"] == "target") != (rows[0]["kind"] == "target"):
                    continue                       # push methods vs push nulls; curves vs shuffled curves
            for stat in ("meaning", "delta_log10_H"):
                null = np.array([q.get(stat, np.nan) for q in rows], float)
                null = null[np.isfinite(null)]
                if len(null) and np.isfinite(r.get(stat, np.nan)):
                    comps[name].setdefault(fam, {})[stat] = {
                        "stat": r[stat], "p": nulls.rank_p(r[stat], null), "K": len(null),
                        "null_p95_abs": float(np.percentile(np.abs(null), 95))}
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
    n6 = json.loads((out / "n6.json").read_text()) if (out / "n6.json").exists() else None
    agreement = readout_agreement(out, items, y0, arm_file, B=a.boot or 1000)
    report = {"n_items": len(items), "n_verification_unreadable": n_ver_skipped, "calibration": cal, "n0": n0,
              "unsteered": {"switch_loc": loc0, "slope": s0, "label_mass": float(base["mass"].mean())},
              "arms": res, "comparisons": comps, "gencheck": gen, "n6": n6, "agreement": agreement}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")

    L = ["# Report card (replay readout)", "",
         f"- Verification items readable: {len(items)} (unreadable: {n_ver_skipped}, listed in skipped.jsonl)",
         f"- Push strength x = {cal['x_push']:g} × typical norm (calibrated on held-out fold {cal['fold']}, "
         f"target {cal['target_logit']} logit, reached: {cal['reached_target']}); target shift = {cal['x_shift']:g} decades",
         f"- N0: zero edit vs hook-free max |Δy| = {n0.get('max_abs_diff_vs_hook_free')}; unsteered replay matches the "
         f"generated choice on {n0.get('replay_matches_generated_choice', float('nan')):.1%}",
         f"- Unsteered switch point log10(H/D) = {loc0:+.3f}, slope {s0:.2f}, label mass {base['mass'].mean():.3f}", "",
         "## Methods, controls and comparisons", "",
         f"95% intervals: paired bootstrap over verification pairs ({a.boot} resamples)." if a.boot else "No intervals (--boot 0).",
         "",
         "| arm | x | meaning (logit) | letter | even | Δlog10 H* | slope ± | low/high plateau (+x) | label mass ± |",
         "|---|---|---|---|---|---|---|---|---|"]

    def with_ci(r, key, fmt="+.3f"):
        v = r.get(key, float("nan"))
        ci = (r.get("ci") or {}).get(key)
        return f"{v:{fmt}}" + (f" [{ci[0]:{fmt}}, {ci[1]:{fmt}}]" if ci else "")

    for name, r in res.items():
        if r["family"] in REPORTED:
            L.append(f"| {name} | {r['x']:g} | {with_ci(r, 'meaning')} | {with_ci(r, 'letter')} | "
                     f"{with_ci(r, 'even_mean')} | {with_ci(r, 'delta_log10_H')} | {r.get('slope_plus', float('nan')):.2f} / "
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
    if n6:
        L += ["", "## N6: damage-matched draws", "",
              f"Damage = mean next-token KL on {len(n6['neutral_texts'])} neutral texts (mean of ±x); a draw is matched "
              f"within ±{n6['tolerance']:.0%}.", "",
              "| method | x | KL | matched draws | draw strength (median, × typical norm) |", "|---|---|---|---|---|"]
        for m, rec in n6["methods"].items():
            xs = [d["x"] for d in rec["draws"] if d["matched"]]
            L.append(f"| {m} | {rec['x_method']:g} | {rec['kl_method']:.4f} | {len(xs)}/{len(rec['draws'])} | "
                     f"{np.median(xs) if xs else float('nan'):.3g} |")
    if agreement:
        L += ["", "## Readout agreement: generation vs replay (generation-check items)", "",
              "| arm | parsed | items | choice agreement | P(far) generated | P(far) replay |", "|---|---|---|---|---|---|"]
        for k, g in agreement["arms"].items():
            f = lambda v, d=2: "–" if v is None else f"{v:.{d}f}"
            L.append(f"| {k} | {g['n_parsed']} | {g['n_items']} | {f(g['choice_agreement'])} | "
                     f"{f(g['p_far_generated'])} | {f(g['p_far_replay'])} |")
        if agreement["effects"]:
            L += ["", "| ± pair | items | odd ΔP(far) generated [95% CI] | odd ΔP(far) replay | same sign |", "|---|---|---|---|---|"]
            for k, e in agreement["effects"].items():
                ci = e["odd_p_far_generated_ci"]
                L.append(f"| {k} | {e['n_items']} | {e['odd_p_far_generated']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                         f"{e['odd_p_far_replay']:+.3f} | {e['same_sign']} |")
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
