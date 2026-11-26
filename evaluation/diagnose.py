"""Diagnostics report for a steering run (CPU only: numpy and json, no model, no plotting library).

Reads what the run already wrote and shows, before anyone reads an effect, whether the directions and the dose could
have worked: data and labels (A), baseline behaviour (B), each fitted direction (C), the push strength (D), outcomes
(E) and readout noise (F). It reports and never gates: every check shows its value, its threshold and a flag.
A section whose inputs are missing says "skipped: <file>", so the tool runs right after fitting and again at the end.

Usage:
  python diagnose.py <run dir> [--artifacts DIR] [--bank DIR] [--baseline DIR] [--out DIR]
    --artifacts has activations/ and methods/ (override with --activations / --methods);
    --bank has prompts.jsonl (and checks/); --baseline has responses.jsonl; --out defaults to the run dir.
Writes diagnostics.html (one self-contained page), diagnostics.json (every number) and diagnostics_items.csv.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from confound_splits import balance_table
from label_audit import audit_bank

# Thresholds, chosen from the 2026-11-25 Qwen3-8B numbers; to be revised.
RELIABILITY_MIN = 0.7        # median split-half cosine of a refitted direction
NUISANCE_COS_MIN = 0.3       # |cos| with letter / style / order / cap is flagged above its shuffled-label null's 95th
N_PERMUTATIONS = 100         # percentile, and never below this floor
CLASS_GAP_MIN = 0.3          # far-rate gap between a direction's + and - classes
SANITY_COS_MIN = 0.99        # saved bundle direction vs the recomputed one
NONE_FAR_RATE_MAX = 0.25     # far rate on `none` prompts (no option pays within the horizon)
FIRST_LISTED_BIAS_MAX = 0.15  # (P(far | far listed first) - P(far | far listed second)) / 2
CHECKER_AGREE_MIN = 0.95
ACC_DIFF_MIN = 0.15          # a phrasing / label style / order whose accuracy differs from the rest by more than this ...
ACC_FISHER_P = 0.05          # ... with Fisher exact p below this
PEAK_DROP = 0.9              # the chosen x is "past the peak" when its effect is below this share of the peak effect
POS_CONTROL_P_MAX = 0.05     # the positive control must beat every null family at this rank p
N0_FRACTION = 0.5            # batched-vs-alone difference allowed, as a share of the largest meaning effect
# 100 random halvings: with 30 the median moved by up to 0.15 between seeds for directions fitted on small classes.
# A half-size refit is noisier than the full fit, so the reliability is conservative (no Spearman-Brown correction).
N_SPLITS = 100
FIRST_LABELS, SECOND_LABELS = ("a)", "1)"), ("b)", "2)")


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def under(d, option, name) -> Path:
    return Path(d if d else f"<{option} not given>") / name


def absent(*paths) -> str | None:
    for p in paths:
        if not p.exists():
            return f"skipped: {p} not found"
    return None


def cosine(a, b):
    if a is None or b is None:
        return None
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else None


def check(name, value, threshold, look, meaning):
    """One row of a checks table. look=None: no threshold, shown for information."""
    return {"name": name, "value": value, "threshold": threshold, "flag": look, "meaning": meaning}


# ---- directions: signed classes on the fit rows --------------------------------------------------------------------

def by(key, pos, neg):
    return lambda r: 1 if r.get(key) == pos else -1 if r.get(key) == neg else 0


def letter_class(r):
    return 1 if r["choice_label"] in FIRST_LABELS else -1 if r["choice_label"] in SECOND_LABELS else 0


# name: (class of a row: +1 / -1 / 0 unused, regimes whose difference of means are averaged; None: one pool)
DIRECTIONS = {"caa": (by("caa_class", "long", "short"), None),
              "caa-within": (by("caa_within_class", "long", "short"), ("near_only", "both")),
              "choice": (by("chosen_option", "far", "near"), None),
              "regime": (by("regime", "both", "near_only"), None),
              "reward": (by("reward_class", "high", "low"), None)}
NUISANCES = {"letter": (letter_class, None), "style": (by("label_style", "a)/b)", "1)/2)"), None),
             "order": (by("near_first", True, False), None), "cap": (by("hit_cap", True, False), None)}
COMPARE = [*NUISANCES, "choice", "regime", "reward"]
NUISANCE_MEANING = {"letter": "the direction mostly encodes which label was answered",
                    "style": "the direction mostly encodes letters vs numbers", "order": "the direction mostly encodes which option was listed first",
                    "cap": "the direction mostly encodes a run that hit the token cap"}


def encode(rows, sign_fn, regimes):
    sign = np.array([sign_fn(r) for r in rows])
    if regimes is None:
        return sign, np.zeros(len(rows), int), ("all",)
    return sign, np.array([regimes.index(r["regime"]) if r["regime"] in regimes else -1 for r in rows]), regimes


def fit_direction(X, sign, group, idx):
    """Mean over groups of mean(X[+]) - mean(X[-]) on rows idx; a group missing a class is dropped; None if none left."""
    parts = []
    for g in np.unique(group[idx]):
        if g < 0:
            continue
        rows = idx[group[idx] == g]
        pos, neg = rows[sign[rows] > 0], rows[sign[rows] < 0]
        if len(pos) and len(neg):
            parts.append(X[pos].mean(0) - X[neg].mean(0))
    return np.mean(parts, 0) if parts else None


def permuted_nuisance_units(X, sign, rng):
    """Unit nuisance directions refitted with the nuisance labels shuffled among the rows that have one.

    Mean differences of raw residual activations share a few huge dimensions (Qwen3-8B L24: one dimension averages
    ~89 against a mean |x| of 1.5), so even a random split of the rows has |cos| ~0.5 with any direction. A nuisance
    cosine means something only above this null."""
    used = np.flatnonzero(sign != 0)
    units = []
    for _ in range(N_PERMUTATIONS):
        shuffled = sign.copy()
        shuffled[used] = rng.permutation(sign[used])
        v = fit_direction(X, shuffled, np.zeros(len(sign), int), np.arange(len(sign)))
        if v is not None and np.linalg.norm(v):
            units.append(v / np.linalg.norm(v))
    return np.array(units)


def nuisance_look(numbers, nuis):
    """A nuisance cosine is flagged above its shuffled-label null's 95th percentile and the NUISANCE_COS_MIN floor."""
    c, null95 = numbers["cos"].get(nuis), numbers.get("cos_null95", {}).get(nuis)
    return c is not None and null95 is not None and abs(c) > max(null95, NUISANCE_COS_MIN)


def site_numbers(X, coded, splits, typical_norm=None):
    """Per direction at one site: raw norm, split-half reliability, cosines with the other directions, and each
    nuisance cosine's permutation null (95th percentile of |cos|)."""
    everyone = np.arange(len(X))
    fit = lambda name, idx: fit_direction(X, *coded[name][:2], idx)
    vecs = {name: fit(name, everyone) for name in coded}
    rng = np.random.default_rng(1)
    null_units = {n: permuted_nuisance_units(X, coded[n][0], rng) for n in NUISANCES if vecs.get(n) is not None}
    out = {}
    for name in DIRECTIONS:
        if vecs[name] is None:
            continue
        halves = [cosine(fit(name, a), fit(name, b)) for a, b in splits]
        halves = [h for h in halves if h is not None]
        norm = float(np.linalg.norm(vecs[name]))
        out[name] = {"raw_norm": norm, "norm_over_typical": norm / typical_norm if typical_norm else None,
                     "reliability": float(np.median(halves)) if halves else None,
                     "reliability_p10_p90": [float(q) for q in np.percentile(halves, [10, 90])] if halves else None,
                     "cos": {o: cosine(vecs[name], vecs[o]) for o in COMPARE if o != name},
                     "cos_null95": {n: float(np.percentile(np.abs(units @ (vecs[name] / norm)), 95))
                                    for n, units in null_units.items() if len(units) and norm}}
    return out, vecs


def class_stats(rows, coded):
    """Class sizes by regime and the far-choice rate of the + and - class (per regime for caa-within)."""
    far = np.array([r.get("chosen_option") == "far" for r in rows])
    parsed = np.array([r.get("chosen_option") in ("far", "near") for r in rows])
    regime = np.array([r["regime"] for r in rows])

    def rate(mask):
        mask = mask & parsed
        return float(far[mask].mean()) if mask.any() else None

    out = {}
    for name in DIRECTIONS:
        sign, group, names = coded[name]
        sizes = {str(g): [int(((sign == s) & (regime == g)).sum()) for s in (1, -1)] for g in sorted(set(regime))}
        pools = []
        for i, g in enumerate(names):
            pos, neg = rate((sign == 1) & (group == i)), rate((sign == -1) & (group == i))
            pools.append({"group": g, "n_pos": int(((sign == 1) & (group == i)).sum()), "n_neg": int(((sign == -1) & (group == i)).sum()),
                          "far_pos": pos, "far_neg": neg, "gap": None if pos is None or neg is None else pos - neg})
        out[name] = {"sizes_by_regime": sizes, "pools": pools}
    return out


# ---- rendering helpers ---------------------------------------------------------------------------------------------

def esc(x):
    return html.escape(str(x))


def num(v, d=2):
    return "-" if v is None else v if isinstance(v, str) else f"{v:.{d}f}"


def table(headers, rows):
    """rows: lists of cells; a cell is text or (text, css class)."""
    def td(c):
        text, cls = c if isinstance(c, tuple) else (c, "")
        return f'<td class="{cls}">{esc(text)}</td>'
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(td(c) for c in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


def checks_table(checks):
    rows = []
    for c in checks:
        look = c["flag"]
        flag = "-" if look is None else ("look", "look") if look else "ok"
        value = num(c["value"])
        rows.append([c["name"], (value, "look") if look else value, c["threshold"], flag, c["meaning"]])
    return table(["check", "value", "threshold", "flag", 'what a "look" means'], rows)


def line_plot(label, xs, ys, xticks, xlabel, ylabel, marks=(), ylim=None):
    """Inline SVG line plot; marks are (x, text) vertical lines."""
    w, h, left, right, top, bottom = 480, 270, 52, 14, 16, 46
    x0, x1 = min(xs), max(xs)
    y0, y1 = ylim or (min([*ys, 0]), max([*ys, 0]))
    px = lambda x: left + (x - x0) / ((x1 - x0) or 1) * (w - left - right)
    py = lambda y: h - bottom - (y - y0) / ((y1 - y0) or 1) * (h - top - bottom)
    s = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">',
         f'<line class="axis" x1="{left}" y1="{py(y0):.1f}" x2="{w - right}" y2="{py(y0):.1f}"/>',
         f'<line class="axis" x1="{left}" y1="{py(y0):.1f}" x2="{left}" y2="{py(y1):.1f}"/>']
    if y0 < 0 < y1:
        s.append(f'<line class="zero" x1="{left}" y1="{py(0):.1f}" x2="{w - right}" y2="{py(0):.1f}"/>')
    for y in (y0, y1):
        s.append(f'<text class="t" x="{left - 6}" y="{py(y) + 4:.1f}" text-anchor="end">{y:.2g}</text>')
    for x, text in xticks:
        s.append(f'<text class="t" x="{px(x):.1f}" y="{h - bottom + 16}" text-anchor="middle">{esc(text)}</text>')
    for x, text in marks:
        s.append(f'<line class="mark" x1="{px(x):.1f}" y1="{top}" x2="{px(x):.1f}" y2="{py(y0):.1f}"/>'
                 f'<text class="t" x="{px(x):.1f}" y="{top - 3}" text-anchor="middle">{esc(text)}</text>')
    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    s.append(f'<polyline class="series" points="{pts}"/>')
    s += [f'<circle class="dot" cx="{px(x):.1f}" cy="{py(y):.1f}" r="3"/>' for x, y in zip(xs, ys)]
    s.append(f'<text class="t" x="{(left + w - right) / 2}" y="{h - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    s.append(f'<text class="t" x="12" y="{(top + h - bottom) / 2}" text-anchor="middle" transform="rotate(-90 12 {(top + h - bottom) / 2})">{esc(ylabel)}</text>')
    return '<div class="plot">' + "".join(s) + "</svg></div>"


def skipped_section(sid, title, why):
    return {"id": sid, "title": title, "skipped": why, "checks": [], "data": None, "body": f'<p class="skip">{esc(why)}</p>'}


# ---- sections --------------------------------------------------------------------------------------------------------

def section_a(prompts, checks_dir):
    title = "A. Data and labels"
    if prompts is None:
        return skipped_section("A", title, "skipped: prompts.jsonl not found (needs --bank)")
    splits = sorted({p["split"] for p in prompts.values()})
    features = {"regime": lambda p: p["regime"], "label style": lambda p: "/".join(p["labels"]),
                "near option listed first": lambda p: p["near_first"], "phrasing": lambda p: p["phrasing_id"],
                "unit rewrite": lambda p: p["rewritten"]}
    rows, counts = [], {}
    for fname, fn in features.items():
        for value in sorted({str(fn(p)) for p in prompts.values()}):
            n = [sum(1 for p in prompts.values() if p["split"] == s and str(fn(p)) == value) for s in splits]
            rows.append([fname, value, *map(str, n)])
            counts[f"{fname}={value}"] = dict(zip(splits, n))
    body = ["<h3>Counts by split</h3>", table(["feature", "value", *splits], rows)]
    checks, checkers = [], {}
    files = sorted(checks_dir.glob("*.jsonl")) if checks_dir.is_dir() else []
    if not files:
        body.append(f'<p class="skip">checker: skipped: {esc(checks_dir)} has no .jsonl</p>')
    for f in files:
        rs = read_jsonl(f)
        agree = sum(r["agrees"] for r in rs) / len(rs)
        problems = Counter(p["category"] for r in rs for p in r.get("problems", []))
        checkers[f.stem] = {"n": len(rs), "agree_rate": agree, "verdicts": dict(Counter(r["verdict"] for r in rs)), "problems": dict(problems)}
        checks.append(check(f"checker {f.stem}: agrees with the generator's answer", agree, f"< {CHECKER_AGREE_MIN}", agree < CHECKER_AGREE_MIN,
                            "the checker solves a share of prompts differently: the stored correct option may be wrong there"))
        body += [f"<h3>Checker {esc(f.stem)}: {sum(r['agrees'] for r in rs)}/{len(rs)} agree, {sum(problems.values())} problem flags</h3>",
                 table(["problem category", "flags"], [[c, str(n)] for c, n in problems.most_common()])]
    audits, balances = {}, {}
    for split in splits:
        records = [p for p in prompts.values() if p["split"] == split]
        audit = audits[split] = audit_bank(records)
        mismatched = sorted({uid for uids in audit["mismatches"].values() for uid in uids})
        unset = sorted({uid for uids in audit["unset"].values() for uid in uids})
        checks.append(check(f"labels [{split}]: rule-based recheck of regime / correct answer / label disagrees", len(mismatched),
                            "> 0", bool(mismatched), "a stored label differs from what the prompt's own options and horizon give"))
        checks.append(check(f"labels [{split}]: prompts with a confound label unset", len(unset), "> 0", bool(unset),
                            "an analysis cannot split these prompts by that factor"))
        kinds = Counter(k for d in audit["debatable"] for k in d["kinds"])
        body.append(f"<h3>Label audit [{esc(split)}]: {len(mismatched)} recheck mismatches, {len(unset)} with unset labels; "
                    f"debatable: {esc(', '.join(f'{k} {n}' for k, n in sorted(kinds.items())) or 'none')}</h3>")
        balance = balances[split] = balance_table([p for p in records if p["regime"] != "no_horizon"])
        flagged = [(factor, by_name, f) for factor, per_by in balance.items() for by_name, b in per_by.items() for f in b["flags"]]
        for factor in sorted({factor for factor, _, _ in flagged}):
            worst = max((f for fa, _, f in flagged if fa == factor), key=lambda f: abs(f["share"] - f["overall_share"]))
            checks.append(check(f"balance [{split}]: {factor} vs regime / correct option", worst["share"] - worst["overall_share"],
                                "a share > 0.10 from the overall share", True,
                                f"{factor} = {worst['factor_level']} is {worst['share']:.0%} of {worst['by_level']} prompts against "
                                f"{worst['overall_share']:.0%} overall: a contrast between those groups also contrasts {factor}"))
        rows = [[factor, by_name, f["by_level"], f["factor_level"], f"{f['share']:.2f}", f"{f['overall_share']:.2f}", str(f["n"])]
                for factor, by_name, f in flagged]
        body.append(table(["factor", "against", "group", "level", "share in group", "overall", "group n"], rows) if rows
                    else "<p>No factor more than 0.10 off its overall share.</p>")
    body += ["<h3>Per-prompt table</h3>", "<p>One row per prompt (split, fold, regime, option, H/far delay, label style, choice, correct, "
                                          "hit the cap, trace length, CAA classes) in <code>diagnostics_items.csv</code>.</p>"]
    data = {"counts": counts, "checkers": checkers, "label_audit": audits, "balance": balances}
    return {"id": "A", "title": title, "skipped": None, "checks": checks, "data": data, "body": "".join(body)}


def fisher_p(a, b, c, d):
    """Two-sided Fisher exact p of the 2x2 table [[a, b], [c, d]] (sum of tables no more likely than the observed one)."""
    r1, r2, c1 = a + b, c + d, a + c
    prob = lambda x: math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(r1 + r2, c1)
    p_obs = prob(a)
    return min(1.0, sum(prob(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if prob(x) <= p_obs * (1 + 1e-9)))


def accuracy_breakdown(parsed):
    """Accuracy of each level of a prompt feature within the scored regimes, against the pooled rest (Fisher exact)."""
    factors = {"CONSTRAINT phrasing": lambda r: r["phrasing_id"], "label style": lambda r: "/".join(r["labels"]),
               "option order (near first)": lambda r: r["near_first"]}
    rows, checks, data = [], [], {}
    for fname, fn in factors.items():
        for regime in ("near_only", "both"):
            g = [(fn(r), r["chosen_option"] == r["correct_option"]) for r in parsed if r["regime"] == regime]
            for level in sorted({lv for lv, _ in g}, key=str):
                hit = [ok for lv, ok in g if lv == level]
                rest = [ok for lv, ok in g if lv != level]
                if not rest:
                    continue
                acc, acc_rest = sum(hit) / len(hit), sum(rest) / len(rest)
                p = fisher_p(sum(hit), len(hit) - sum(hit), sum(rest), len(rest) - sum(rest))
                look = abs(acc - acc_rest) > ACC_DIFF_MIN and p < ACC_FISHER_P
                rows.append([fname, regime, str(level), str(len(hit)), (f"{acc:.2f}", "look" if look else ""), f"{acc_rest:.2f}", f"{p:.3f}"])
                data[f"{fname}/{regime}/{level}"] = {"n": len(hit), "accuracy": acc, "accuracy_rest": acc_rest, "fisher_p": p}
                checks.append(check(f"accuracy, {fname} = {level}, regime {regime}: vs the rest ({acc_rest:.2f})", acc,
                                    f"differs > {ACC_DIFF_MIN} and Fisher p < {ACC_FISHER_P} (p = {p:.3f})", look,
                                    "the answer's correctness depends on how the prompt is worded or listed, not only on the horizon"))
    return rows, checks, data


def section_b(prompts, baseline):
    title = "B. Baseline behaviour"
    if prompts is None or baseline is None:
        return skipped_section("B", title, "skipped: " + ("prompts.jsonl (needs --bank)" if prompts is None else "responses.jsonl (needs --baseline)") + " not found")
    rows = [{**prompts[u], **b} for u, b in baseline.items() if u in prompts]
    parsed = [r for r in rows if r["chosen_option"] in ("far", "near")]
    acc_rows, acc = [], {}
    for split in sorted({r["split"] for r in rows}):
        for regime in ("near_only", "both"):
            g = [r for r in rows if r["split"] == split and r["regime"] == regime]
            if g:
                acc[f"{split}/{regime}"] = {"n": len(g), "parsed": sum(r["chosen_option"] in ("far", "near") for r in g), "correct": sum(bool(r.get("correct")) for r in g)}
                a = acc[f"{split}/{regime}"]
                acc_rows.append([split, regime, str(a["n"]), str(a["parsed"]), f"{a['correct'] / a['n']:.2f}"])
    acc_table, acc_checks, acc_data = accuracy_breakdown(parsed)
    far_rate = lambda g: float(np.mean([r["chosen_option"] == "far" for r in g])) if g else None
    far_by_regime = {reg: far_rate([r for r in parsed if r["regime"] == reg]) for reg in sorted({r["regime"] for r in parsed})}
    none_rate = far_rate([r for r in parsed if r["regime"] == "none"])
    p_far_first = far_rate([r for r in parsed if not r["near_first"]])
    p_far_second = far_rate([r for r in parsed if r["near_first"]])
    bias = None if p_far_first is None or p_far_second is None else (p_far_first - p_far_second) / 2
    checks = [check("far rate on `none` prompts (no option pays within the horizon)", none_rate, f"> {NONE_FAR_RATE_MAX}",
                    none_rate is not None and none_rate > NONE_FAR_RATE_MAX, "the model answered like a long horizon where neither option pays out"),
              *acc_checks,
              check("first-listed bias: P(far | far first) vs P(far | far second)", bias, f"|.| > {FIRST_LISTED_BIAS_MAX}",
                    bias is not None and abs(bias) > FIRST_LISTED_BIAS_MAX, "answers follow the listing order more than the options' meaning")]
    ver = [r for r in parsed if r["split"] == "verification"]
    plot, bins = "", []
    if ver:
        logr = np.log10([r["h_over_far_delay"] for r in ver])
        far = np.array([r["chosen_option"] == "far" for r in ver], float)
        edges = np.linspace(logr.min(), logr.max(), 9)
        which = np.clip(np.digitize(logr, edges) - 1, 0, 7)
        bins = [{"log10_h_over_d": float(logr[which == i].mean()), "p_far": float(far[which == i].mean()), "n": int((which == i).sum())}
                for i in range(8) if (which == i).any()]
        ticks = [(t, str(t)) for t in range(int(np.ceil(logr.min())), int(np.floor(logr.max())) + 1)]
        plot = line_plot("P(far) against log10 H/D", [b["log10_h_over_d"] for b in bins], [b["p_far"] for b in bins], ticks,
                         "log10 H / far delay (verification grid)", "P(far)", marks=[(0, "H=D")], ylim=(0, 1))
    body = ["<h3>Accuracy (correct / all prompts of the group)</h3>", table(["split", "regime", "prompts", "parsed", "accuracy"], acc_rows),
            "<h3>Far rate by regime (parsed answers)</h3>", table(["regime", "P(far)"], [[k, num(v)] for k, v in far_by_regime.items()]),
            "<h3>Accuracy by phrasing, label style and order (parsed answers)</h3>",
            table(["feature", "regime", "level", "n", "accuracy", "rest", "Fisher p"], acc_table),
            checks_table(checks), "<h3>P(far) on the verification grid</h3>", plot]
    data = {"accuracy": acc, "far_rate_by_regime": far_by_regime, "p_far_far_first": p_far_first, "p_far_far_second": p_far_second,
            "first_listed_bias": bias, "accuracy_breakdown": acc_data, "grid_bins": bins}
    return {"id": "B", "title": title, "skipped": None, "checks": checks, "data": data, "body": "".join(body)}


def load_bundles(methods_dir):
    """Method bundles under methods_dir: meta, saved direction (single-direction methods), fit rows."""
    out = []
    for meta_path in sorted(methods_dir.glob("*/method.json")) if methods_dir.is_dir() else []:
        d = meta_path.parent
        meta = json.loads(meta_path.read_text())
        arrays = np.load(d / "arrays.npz")
        single = "direction" in arrays.files
        out.append({"method": meta["method"], "site": meta["site"], "layer": meta["layer"], "single": single,
                    "direction": arrays["direction"] if single else None,
                    "typical_norm": float(arrays["typical_norm"]) if single else None,
                    "fit_rows": read_jsonl(d / "fit_uids.jsonl") if (d / "fit_uids.jsonl").exists() else None})
    return out


def section_c(bundles, acts_dir, baseline):
    title = "C. Each fitted direction"
    index_path = under(acts_dir, "--activations", "index.jsonl")
    if absent(index_path):
        return skipped_section("C", title, absent(index_path))
    if not bundles:
        return skipped_section("C", title, "skipped: no <method>/method.json found (needs --methods or --artifacts)")
    singles = [b for b in bundles if b["single"]]
    withrows = next((b for b in sorted(bundles, key=lambda b: b["method"] != "caa") if b["fit_rows"]), None)
    if not singles or withrows is None:
        return skipped_section("C", title, "skipped: no single-direction bundle with fit_uids.jsonl")
    main = withrows
    main_site, layer = main["site"], main["layer"]
    row_of = {r["uid"]: r.get("row", i) for i, r in enumerate(read_jsonl(index_path))}
    rows = [dict(r, hit_cap=baseline[r["uid"]]["hit_token_cap"] if baseline and r["uid"] in baseline else None)
            for r in main["fit_rows"] if r["uid"] in row_of]
    idx = np.array([row_of[r["uid"]] for r in rows])
    coded = {name: encode(rows, *spec) for name, spec in {**DIRECTIONS, **NUISANCES}.items()}
    rng = np.random.default_rng(0)
    splits = []
    for _ in range(N_SPLITS):
        perm = rng.permutation(len(rows))
        splits.append((perm[:len(rows) // 2], perm[len(rows) // 2:]))
    stats = class_stats(rows, coded)
    sites, site_vecs = {}, {}
    for path in sorted(acts_dir.glob(f"L{layer}_*.npy")):
        site = path.stem.split("_", 1)[1]
        X = np.load(path, mmap_mode="r")[idx].astype(np.float64)
        sites[site], site_vecs[site] = site_numbers(X, coded, splits, main["typical_norm"] if site == main_site else None)
    if main_site not in sites:
        return skipped_section("C", title, f"skipped: {acts_dir / f'L{layer}_{main_site}.npy'} not found")

    checks, sanity, skipped = [], {}, []
    for name, now in sites[main_site].items():
        gaps = [] if name == "choice" else stats[name]["pools"]
        spread = now["reliability_p10_p90"]
        spread_text = f", 10-90% {spread[0]:+.2f} to {spread[1]:+.2f}" if spread else ""
        checks.append(check(f"{name}: split-half reliability (median of {N_SPLITS}{spread_text})", now["reliability"], f"< {RELIABILITY_MIN}",
                            now["reliability"] is None or now["reliability"] < RELIABILITY_MIN, "refitting on a random half gives a different direction: it is mostly noise"))
        for nuis in NUISANCES:
            c, null95 = now["cos"][nuis], now["cos_null95"].get(nuis)
            look = nuisance_look(now, nuis)
            threshold = f"|cos| > shuffled-label null 95% ({null95:.2f})" if null95 is not None else "no null"
            checks.append(check(f"{name}: cos with {nuis}", c, threshold, look, NUISANCE_MEANING[nuis]))
        for pool in gaps:
            g = pool["gap"]
            checks.append(check(f"{name} [{pool['group']}]: far-rate gap + minus - class", g, f"< {CLASS_GAP_MIN}", g is None or g < CLASS_GAP_MIN,
                                "the two classes barely differ in what the model chose, so the direction need not carry the choice"))
    for b in bundles:
        if not b["single"]:
            skipped.append(f"{b['method']}: not a single direction (a curve), no cosines or reliability")
        elif b["method"] in DIRECTIONS and b["site"] in sites:
            c = cosine(b["direction"], site_vecs[b["site"]][b["method"]])
            sanity[b["method"]] = c
            checks.append(check(f"{b['method']}: saved bundle vs recomputed direction", c, f"cos < {SANITY_COS_MIN}", c is None or c < SANITY_COS_MIN,
                                "the saved direction is not what these fit rows give: the bundle and the labels disagree"))
    body = [f"<p>{len(rows)} fit rows ({len(main['fit_rows']) - len(rows)} without activations); site <b>{esc(main_site)}</b>, layer {layer}. "
            "Directions are refitted from the fit rows (+ minus - class). <code>caa-within</code> averages near_only and both; "
            "<code>regime</code> is both minus near_only.</p>"]
    cls_rows = []
    for name, st in stats.items():
        sizes = ", ".join(f"{g} {a}/{b}" for g, (a, b) in st["sizes_by_regime"].items() if a or b)
        for pool in st["pools"]:
            gap = pool["gap"]
            look = name != "choice" and (gap is None or gap < CLASS_GAP_MIN)
            cls_rows.append([name, sizes, pool["group"], f"{pool['n_pos']}/{pool['n_neg']}", num(pool["far_pos"]), num(pool["far_neg"]), (num(gap), "look" if look else "")])
    body += ["<h3>Classes and their behaviour</h3>", table(["direction", "+/- sizes by regime", "pool", "n +/-", "far rate +", "far rate -", "gap"], cls_rows),
             "<h3>Checks</h3>", checks_table(checks)]
    cos_rows = []
    for name, now in sites[main_site].items():
        cells = [(num(now["cos"].get(o)), "look" if o in NUISANCES and nuisance_look(now, o) else "") if o in now["cos"] else "-" for o in COMPARE]
        cos_rows.append([name, num(now["norm_over_typical"], 3), num(sanity.get(name)), *cells])
    body += [f"<h3>Cosines at {esc(main_site)}</h3>", table(["direction", "norm / typical norm", "saved vs recomputed", *COMPARE], cos_rows),
             "<h3>The same at every extracted site</h3>"]
    site_rows = []
    for name in sites[main_site]:
        for site, numbers in sites.items():
            n = numbers.get(name)
            if n is None:
                continue
            rel = n["reliability"]
            cells = [(num(n["cos"][o]), "look" if o in NUISANCES and nuisance_look(n, o) else "") for o in [*NUISANCES, "choice"] if o in n["cos"]]
            site_rows.append([name, site + (" *" if site == main_site else ""), (num(rel), "look" if rel is None or rel < RELIABILITY_MIN else ""), *cells])
    body += [table(["direction", "site", "reliability", *NUISANCES, "choice"], site_rows), "<p>* the site the bundles were saved at.</p>"]
    if skipped:
        body.append("<h3>Skipped</h3><ul>" + "".join(f"<li>{esc(s)}</li>" for s in skipped) + "</ul>")
    data = {"site": main_site, "layer": layer, "n_fit_rows": len(rows), "classes": stats, "sites": sites, "saved_vs_recomputed": sanity, "skipped": skipped}
    return {"id": "C", "title": title, "skipped": None, "checks": checks, "data": data, "body": "".join(body)}


def section_d(run, report):
    title = "D. Strength"
    cal_path = run / "calibration.json"
    if absent(cal_path):
        return skipped_section("D", title, absent(cal_path))
    cal = json.loads(cal_path.read_text())
    sweep, x_push = cal["sweep"], cal["x_push"]
    xs, odd = [s["x"] for s in sweep], [s["mean_odd_logit"] for s in sweep]
    sign = 1 if odd[0] >= 0 else -1
    peak = int(np.argmax([sign * e for e in odd]))
    at_push = next((e for x, e in zip(xs, odd) if np.isclose(x, x_push)), None)
    past = x_push > xs[peak] and (at_push is None or sign * at_push < PEAK_DROP * sign * odd[peak])
    flips = any(sign * e < 0 for e in odd)
    checks = [check("chosen x is past the dose peak", x_push, f"x > {xs[peak]:g} (peak) and effect < {PEAK_DROP:.0%} of the peak's", past,
                    "a smaller push moved the held-out items more: the rule picked a strength beyond where the effect is largest"),
              check("odd effect changes sign along the grid (most opposed effect)", min(odd, key=lambda e: sign * e), "any effect opposite to the smallest x's", flips,
                    "past some x the push acts the other way: the curve is not a dose-response")]
    data = {"x": xs, "mean_odd_logit": odd, "label_mass": [s["even_label_mass"] for s in sweep], "x_push": x_push, "rule": cal.get("rule"),
            "peak_x": xs[peak], "effect_at_x_push": at_push}
    if (run / "report.json").exists():
        arm = json.loads((run / "report.json").read_text())["arms"].get("caa")
        if arm and at_push is not None:
            checks.append(check("caa: held-out effect at x vs verification-grid effect (sign)", f"{at_push:+.2f} vs {arm['odd_mean']:+.2f}", "opposite signs",
                                bool(at_push * arm["odd_mean"] < 0), "the strength was calibrated on one effect and the verification grid shows the opposite one"))
            data["verification_odd_mean"] = arm["odd_mean"]
    plot = line_plot("Calibration dose curve", [float(np.log10(x)) for x in xs], odd, [(float(np.log10(x)), f"{x:g}") for x in xs],
                     "push strength x (x typical norm, log scale)", "mean odd effect (logit)",
                     marks=[(float(np.log10(xs[peak])), "peak"), (float(np.log10(x_push)), "chosen")])
    rows = [[f"{s['x']:g}", f"{s['mean_odd_logit']:+.3f}", f"{s['even_label_mass']:.4f}", "yes" if s["healthy"] else ("no", "look")] for s in sweep]
    body = [f"<p>Rule: {esc(cal.get('rule'))}. Chosen x = {x_push:g}; peak at x = {xs[peak]:g}.</p>", plot,
            table(["x", "mean odd effect", "label mass", "healthy"], rows), checks_table(checks)]
    return {"id": "D", "title": title, "skipped": None, "checks": checks, "data": data, "body": "".join(body)}


def section_e(report):
    title = "E. Outcomes"
    if report is None:
        return skipped_section("E", title, "skipped: report.json not found")
    arms, comps = report["arms"], report["comparisons"]
    shown = {k: a for k, a in arms.items() if a["family"] in ("method", "control", "comparison", "prompting")}
    rows, checks = [], []
    for name, a in shown.items():
        ps = ", ".join(f"{fam} {st['meaning']['p']:.2f}" for fam, st in comps.get(name, {}).items() if "meaning" in st)
        g = lambda k, d=2: num(a.get(k), d)
        rows.append([name, a["family"], f"{a['x']:g}", g("meaning"), g("letter"), g("even_mean"), ps or "-", g("delta_log10_H"),
                     f"{g('plateau_low_plus')} / {g('plateau_high_plus')}", f"{g('slope_plus')} / {g('slope_minus')}"])
        if a["family"] == "method" and "meaning" in a:
            checks.append(check(f"{name}: letter vs meaning", f"letter {a['letter']:.2f}, meaning {a['meaning']:.2f}", "|letter| > |meaning|",
                                abs(a["letter"]) > abs(a["meaning"]), "the push moves the answer label more than the option's meaning"))
    choice = arms.get("choice")
    if choice is None:
        checks.append(check("choice (positive control)", "not run", f"must run and beat every null (p <= {POS_CONTROL_P_MAX})", True,
                            "the readout cannot show an effect; read nothing else"))
    else:
        worst = max([st["meaning"]["p"] for st in comps.get("choice", {}).values() if "meaning" in st], default=None)
        checks.append(check("choice (positive control): worst rank p against the nulls", worst, f"> {POS_CONTROL_P_MAX}", worst is None or worst > POS_CONTROL_P_MAX,
                            "the readout cannot show an effect; read nothing else"))
    body = [table(["arm", "family", "x", "meaning", "letter", "even", "rank p of meaning", "dlog10 H*", "plateau low/high (+x)", "slope +/-"], rows), checks_table(checks)]
    return {"id": "E", "title": title, "skipped": None, "checks": checks, "data": {"arms": shown}, "body": "".join(body)}


def section_f(run, report):
    title = "F. Readout noise"
    n0_path = run / "n0_check.json"
    if absent(n0_path):
        return skipped_section("F", title, absent(n0_path))
    n0 = json.loads(n0_path.read_text())
    settings = json.loads((run / "settings.json").read_text()) if (run / "settings.json").exists() else {}
    dtype = settings.get("args", {}).get("replay_dtype")
    effects = [abs(a["meaning"]) for a in (report or {}).get("arms", {}).values() if a["family"] == "method" and "meaning" in a]
    alone = n0["max_abs_diff_batched_vs_alone"]
    limit = N0_FRACTION * max(effects) if effects else None
    checks = [check("batched vs one-at-a-time: max |difference| (logit)", alone, f"> {N0_FRACTION} x largest meaning effect" + (f" ({limit:.2f})" if limit else " (needs report.json)"),
                    None if limit is None else alone > limit, "batching noise is as large as the effects being measured"),
              check("hook-free vs zero edit: max |difference| (logit)", n0["max_abs_diff_vs_hook_free"], "-", None, "-"),
              check("replay matches the generated choice", n0["replay_matches_generated_choice"], "-", None, "-"),
              check(f"replay dtype: {dtype or 'unknown'}", n0["replay_batch_size"], "-", None, "(value: replay batch size)")]
    return {"id": "F", "title": title, "skipped": None, "checks": checks, "data": {"n0": n0, "replay_dtype": dtype},
            "body": checks_table(checks)}


# ---- page ------------------------------------------------------------------------------------------------------------

LIGHT = "--bg:#ffffff;--fg:#1f2328;--muted:#59636e;--border:#d1d9e0;--panel:#f6f8fa;--accent:#0969da;--look:#9a3412;--look-bg:#fff1e6;--ok:#1a7f37"
DARK = "--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--border:#30363d;--panel:#161b22;--accent:#58a6ff;--look:#ffa657;--look-bg:#3a2411;--ok:#3fb950"
CSS = f"""
:root{{{LIGHT}}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{{DARK}}}}}
:root[data-theme="dark"]{{{DARK}}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--fg);margin:0;padding:16px;font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:980px;margin:0 auto}}
h1{{font-size:1.5rem;margin:0 0 .25rem}} h2{{font-size:1.25rem;margin:2rem 0 .5rem;border-bottom:1px solid var(--border)}} h3{{font-size:1.05rem;margin:1.25rem 0 .4rem}}
a{{color:var(--accent)}} code{{background:var(--panel);padding:0 .25em;border-radius:3px}} p,li{{overflow-wrap:anywhere}}
.scroll{{overflow-x:auto;max-width:100%;margin:.5rem 0}}
table{{border-collapse:collapse;font-size:14px}} th,td{{border:1px solid var(--border);padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:var(--panel)}} td.look{{background:var(--look-bg);color:var(--look);font-weight:600}}
.skip{{color:var(--muted);font-style:italic}} .ok{{color:var(--ok)}} .muted{{color:var(--muted)}}
.plot{{max-width:560px}} svg{{width:100%;height:auto;display:block}}
svg .t{{fill:var(--muted);font-size:11px}} svg .axis{{stroke:var(--muted)}} svg .zero{{stroke:var(--border)}} svg .mark{{stroke:var(--look);stroke-dasharray:4 3}}
svg .series{{fill:none;stroke:var(--accent);stroke-width:2}} svg .dot{{fill:var(--accent)}}
"""


def render_page(run, sections, flags, inputs):
    summary = "".join(f'<li><a href="#sec-{s}">{esc(s)}</a> {esc(n)}: {esc(v)} (threshold {esc(t)})</li>' for s, n, v, t in flags)
    if not flags:
        summary = "<li>no checks look</li>"
    skipped = "".join(f'<li><a href="#sec-{s["id"]}">{esc(s["id"])}</a> {esc(s["skipped"])}</li>' for s in sections if s["skipped"])
    parts = [f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
             f"<title>Steering run diagnostics</title><style>{CSS}</style></head><body><main>",
             f"<h1>Steering run diagnostics</h1><p class=\"muted\">{esc(run.name)}. Reports, never gates: a look is a place to look, not a verdict.</p>",
             f"<h2>Summary: {len(flags)} checks look</h2><ul>{summary}</ul>"]
    if skipped:
        parts.append(f"<h3>Skipped sections</h3><ul>{skipped}</ul>")
    parts.append("<p class=\"muted\">Inputs: " + esc("; ".join(f"{k} {v}" for k, v in inputs.items())) + "</p>")
    for s in sections:
        parts.append(f'<section id="sec-{s["id"]}"><h2>{esc(s["title"])}</h2>{s["body"]}</section>')
    parts.append("</main></body></html>")
    return "".join(parts)


def write_items_csv(path, prompts, baseline, fit_rows):
    fit = {r["uid"]: r for r in fit_rows or []}
    cols = ["uid", "split", "fold", "regime", "correct_option", "h_over_far_delay", "label_style", "far_label", "near_first", "chosen",
            "correct", "hit_cap", "trace_length", "caa_class", "caa_within_class"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for uid, p in prompts.items():
            b, r = (baseline or {}).get(uid, {}), fit.get(uid, {})
            far_label = p["labels"][1] if p["near_first"] else p["labels"][0]
            w.writerow([uid, p["split"], r.get("fold"), p["regime"], p["correct_option"], p["h_over_far_delay"], "/".join(p["labels"]), far_label,
                        p["near_first"], b.get("chosen_option"), b.get("correct"), b.get("hit_token_cap"), b.get("n_generated"),
                        r.get("caa_class"), r.get("caa_within_class")])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--artifacts"), ap.add_argument("--activations"), ap.add_argument("--methods")
    ap.add_argument("--bank"), ap.add_argument("--baseline"), ap.add_argument("--out")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    out = Path(a.out) if a.out else run
    out.mkdir(parents=True, exist_ok=True)
    acts = Path(a.activations) if a.activations else Path(a.artifacts) / "activations" if a.artifacts else None
    methods = Path(a.methods) if a.methods else Path(a.artifacts) / "methods" if a.artifacts else None

    prompts_path, resp_path = under(a.bank, "--bank", "prompts.jsonl"), under(a.baseline, "--baseline", "responses.jsonl")
    prompts = {r["uid"]: r for r in read_jsonl(prompts_path)} if prompts_path.exists() else None
    baseline = {r["uid"]: r for r in read_jsonl(resp_path)} if resp_path.exists() else None
    report = json.loads((run / "report.json").read_text()) if (run / "report.json").exists() else None
    bundles = load_bundles(methods) if methods else []

    sections = [section_a(prompts, under(a.bank, "--bank", "checks")), section_b(prompts, baseline),
                section_c(bundles, acts, baseline), section_d(run, report), section_e(report), section_f(run, report)]
    flags = [(s["id"], c["name"], num(c["value"]), c["threshold"]) for s in sections for c in s["checks"] if c["flag"]]
    inputs = {"run": run, "bank": prompts_path if prompts else "missing", "baseline": resp_path if baseline else "missing",
              "activations": acts or "missing", "methods": methods or "missing"}
    (out / "diagnostics.html").write_text(render_page(run, sections, flags, inputs))
    (out / "diagnostics.json").write_text(json.dumps(
        {"run": str(run), "flags": [{"section": s, "check": n, "value": v, "threshold": t} for s, n, v, t in flags],
         "sections": {s["id"]: {"title": s["title"], "skipped": s["skipped"], "checks": s["checks"], "data": s["data"]} for s in sections}},
        indent=2, default=float) + "\n")
    if prompts:
        write_items_csv(out / "diagnostics_items.csv", prompts, baseline, next((b["fit_rows"] for b in bundles if b["fit_rows"]), None))
    print(f"{len(flags)} checks look; wrote {out / 'diagnostics.html'}")
    for s, n, v, t in flags:
        print(f"  [{s}] {n}: {v} (threshold {t})")
    return flags


if __name__ == "__main__":
    main()
