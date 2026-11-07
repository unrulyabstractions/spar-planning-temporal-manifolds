"""E18 summary: implied horizons (29_spec_implied_horizon_side.md §4 measures, §5 criteria C0-C4).

    python analysis/implied_summary.py <dir> [--probe-runs <E0 run dir> ...]
    # <dir> = <out>/exp18_Qwen3-8B_think-on_implied
    -> <dir>/implied_summary.txt, implied_summary.json, implied_audit_sample.json

Reads <dir>/units/*.json and .npz (experiments/exp18_implied.py); runs no model. Works on whatever units
exist: a criterion without its rows reads "pending", never pass or fail. Measures:

  (a) behaviour   accuracy against the truth on near and both rows, implied vs explicit twin, per family
                  (McNemar exact test on pairs sharing twin_id); near share in none; near share of the null
                  twin; forced-close rate per arm. Rows from natural think closes are reported beside all rows.
  (c) conversion  the first duration the trace states as its horizon (`first_stated_implied`: the E14 reader
                  extended for anchor phrasing), its error against the anchor, and whether it flips the regime.
  mediation       where the stated regime differs from the true one, does the choice follow the stated one.
  (b) readout     a ridge probe trained on the explicit E0 runs (--probe-runs: the full and fill runs'
                  activations.npz) read on the saved states of implied and explicit prompts. Without
                  --probe-runs it reads "pending".

The blind audit (§4 (c)) is implied_audit_sample.json: 40 implied traces without any automatic coding;
the automatic coding of the same traces is `audit_key` in implied_summary.json.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp14_analysis import HORIZON_AFTER, HORIZON_BEFORE  # noqa: E402
from spar_horizon.full_bank import regime  # noqa: E402
from spar_horizon.horizons import UNIT_YEARS, canonical_unit  # noqa: E402
from spar_horizon.probes import fit_ridge  # noqa: E402
from written_unit import DURATION, close_to, parse_number  # noqa: E402

FAMILIES = ("A", "B", "C")
REGIMES = ("near", "both")
C0_MIN = 0.90
C1_MIN, C1_TWIN_GAP, C1_IGNORED = 0.85, 0.10, 0.15
C2_CONVERTS, C2_ERROR, C2_INSIDE = 0.80, 0.10, 0.85
C3_RATIO, C3_PRESENT, C3_ABSENT = 1.5, 0.8, 0.3
C4_MIN_FLIPS, C4_FOLLOWS = 10, 0.75
AUDIT_N = 40
CELLS = (("suffix_1", (16,)), ("think@1", (16, 20, 24)), ("prelabel", (16, 20, 24)))   # (state column, layers)
NATURAL_ONLY = ("think@1", "prelabel")

# anchor phrasing: a duration the trace states with a connective in front ("the lease ends in 18 months",
# "that is about 7 hours"); the E14 reader needs the word "horizon" next to it
ANCHOR_BEFORE = re.compile(
    r"(?:\b(?:is|are|be|equals?|means?|about|approximately|roughly|around|nearly|just|only|exactly|in|for|of|over|"
    r"within|next|until|that's|that is|which is|spans?|lasts?|runs?|remaining|ends?|expires?)\s+|[=→]\s*)+$",
    re.IGNORECASE)


# ---- loading ----

def load_rows(run_dir):
    """Every unit's rows, in unit order."""
    rows = []
    for path in sorted((Path(run_dir) / "units").glob("*.json")):
        rows += json.loads(path.read_text())["rows"]
    return rows


def labelled(rows):
    return [r for r in rows if r.get("label") in ("a", "b")]


def load_states(run_dir):
    """{(layer, column): (sample_uids, matrix)} over the finished units."""
    pieces = {}
    for path in sorted((Path(run_dir) / "units").glob("*.json")):
        with np.load(path.with_suffix(".npz")) as z:
            for layer in z["layers"]:
                for j, col in enumerate(z["cols"]):
                    pieces.setdefault((int(layer), str(col)), []).append((z["sample_uids"], z[f"L{layer}"][:, j]))
    return {key: (np.concatenate([u for u, _ in p]), np.concatenate([x for _, x in p])) for key, p in pieces.items()}


def load_probe_training(run_dirs, cells=CELLS):
    """{(layer, column): (X, y)} from explicit E0 runs: log10 horizon of every kept prompt with one, states at
    the suffix newline, think@1 and the pre-label token; the last two from natural closes only."""
    train = {}
    for run_dir in run_dirs:
        with np.load(Path(run_dir) / "activations.npz") as z:
            regions = {name[7:]: z[name] for name in z.files if name.startswith("region_")}
            fractions = list(np.round(z["think_fractions"], 6))
            columns = {"suffix_1": regions["suffix"][0] + 1,
                       "think@1": regions["think"][0] + fractions.index(1.0),
                       "prelabel": regions["prelabel"][0]}
            horizons, forced = z["horizons"], z["forced_mask"]
            has_horizon = ~np.isnan(horizons)
            for column, layers in cells:
                keep = has_horizon & ~forced if column in NATURAL_ONLY else has_horizon
                for layer in layers:
                    X = z[f"L{layer}"][:, columns[column]][keep]
                    X0, y0 = train.get((layer, column), (X[:0], np.zeros(0)))
                    train[(layer, column)] = (np.concatenate([X0, X]), np.concatenate([y0, np.log10(horizons[keep])]))
    return train


# ---- (a) behaviour ----

def chose_near(row):
    return row["label"] == row["near_label"]


def is_correct(row):
    return chose_near(row) == (row["truth"] == "near")


def tally(rows):
    n = len(rows)
    return {"n": n, "correct": sum(map(is_correct, rows)) if rows and rows[0]["truth"] else None,
            "near_share": float(np.mean([chose_near(r) for r in rows])) if n else None,
            "forced": float(np.mean([r["forced"] for r in rows])) if n else None}


def accuracy(rows):
    """Share of `rows` (near and both) choosing the rational option, None without rows."""
    return float(np.mean([is_correct(r) for r in rows])) if rows else None


def select(rows, family=None, rendering=None, regimes=None, natural=False):
    return [r for r in rows if (family is None or r["family"] == family)
            and (rendering is None or r["rendering"] == rendering)
            and (regimes is None or r["regime"] in regimes) and (not natural or not r["forced"])]


def mcnemar(rows, family):
    """Exact two-sided McNemar test over bases (twin_id) where both renderings answered a near/both prompt."""
    by_base = {}
    for r in select(labelled(rows), family, regimes=REGIMES):
        if r["rendering"] in ("implied", "explicit"):
            by_base.setdefault(r["twin_id"], {})[r["rendering"]] = is_correct(r)
    pairs = [v for v in by_base.values() if len(v) == 2]
    only_implied = sum(v["implied"] and not v["explicit"] for v in pairs)
    only_explicit = sum(v["explicit"] and not v["implied"] for v in pairs)
    discordant = only_implied + only_explicit
    p = binomtest(only_implied, discordant, 0.5).pvalue if discordant else None
    return {"pairs": len(pairs), "only_implied_correct": only_implied, "only_explicit_correct": only_explicit, "p": p}


def behaviour(rows):
    rows = labelled(rows)
    out = {}
    for family in FAMILIES:
        arms = {}
        for rendering in ("implied", "explicit"):
            for regime_name in (*REGIMES, "none"):
                mine = select(rows, family, rendering, (regime_name,))
                arms[f"{rendering} {regime_name}"] = tally(mine)
                arms[f"{rendering} {regime_name} natural"] = tally(select(rows, family, rendering, (regime_name,), True))
            arms[f"{rendering} near+both"] = tally(select(rows, family, rendering, REGIMES))
        arms["null"] = tally(select(rows, family, "null"))
        arms["mcnemar"] = mcnemar(rows, family)
        out[family] = arms
    return out


# ---- (c) conversion and mediation ----

def first_stated_implied(text, near_years, far_years):
    """(years, unit, number as written) of the first duration the trace states as its horizon, or None. As
    exp14_analysis.first_stated_horizon, plus durations introduced by a connective (ANCHOR_BEFORE), leaving out
    mentions of the prompt's own delays."""
    for match in DURATION.finditer(text):
        value = parse_number(match["number"])
        if value is None:
            continue
        unit = canonical_unit(match["unit"])
        years = value * UNIT_YEARS[unit]
        if close_to(years, near_years) or close_to(years, far_years):
            continue
        before = text[max(0, match.start() - 40):match.start()]
        if HORIZON_AFTER.match(text, match.end()) or HORIZON_BEFORE.search(before) or ANCHOR_BEFORE.search(before):
            return years, unit, match["number"]
    return None


def stated_regime(row, years):
    return regime(years, row["near_delay_years"], row["far_delay_years"])


def conversion(rows):
    out = {}
    for family in FAMILIES:
        mine = select(labelled(rows), family, "implied")
        stated = [(r, first_stated_implied(r["think_text"], r["near_delay_years"], r["far_delay_years"])) for r in mine]
        converted = [(r, s[0]) for r, s in stated if s is not None]
        entry = {"n": len(mine), "converted": len(converted),
                 "converted_share": len(converted) / len(mine) if mine else None,
                 "flips_regime": sum(stated_regime(r, y) != r["base_regime"] for r, y in converted)}
        if family == "B":
            inside = [r["anchor_lo"] <= y <= r["anchor_hi"] for r, y in converted]
            entry["inside_share"] = float(np.mean(inside)) if inside else None
        else:
            errors = [abs(np.log10(y) - r["log_h"]) for r, y in converted]
            entry["median_error_decades"] = float(np.median(errors)) if errors else None
        out[family] = entry
    return out


def mediation(rows):
    """Choice against the stated regime on implied near/both prompts whose trace states a duration in the other
    of near/both; stated regimes none are left out (they have no rational option)."""
    flips = []
    for r in select(labelled(rows), None, "implied", REGIMES):
        stated = first_stated_implied(r["think_text"], r["near_delay_years"], r["far_delay_years"])
        if stated is None:
            continue
        reg = stated_regime(r, stated[0])
        if reg in REGIMES and reg != r["regime"]:
            flips.append(chose_near(r) == (reg == "near"))
    return {"flips": len(flips), "follows_stated": int(sum(flips)),
            "follows_share": float(np.mean(flips)) if flips else None}


# ---- (b) readout ----

def readout_stats(predicted, true):
    rho = spearmanr(predicted, true)[0] if len(true) > 2 and np.ptp(true) > 0 and np.ptp(predicted) > 0 else float("nan")
    return {"n": len(true), "mae": float(np.mean(np.abs(predicted - true))) if len(true) else None, "rho": float(rho)}


def c3_verdict(implied, explicit):
    if not implied["n"] or not explicit["n"] or not explicit["mae"] or np.isnan(implied["rho"]):
        return "pending"
    if implied["mae"] <= C3_RATIO * explicit["mae"] and implied["rho"] >= C3_PRESENT:
        return "present"
    return "absent" if implied["rho"] < C3_ABSENT else "intermediate"


def readout(rows, states, train):
    """Per (column, layer) and family: the probe's MAE (decades) and Spearman rho with the true log duration
    on implied and explicit prompts, and the C3 verdict."""
    by_uid = {r["sample_uid"]: r for r in rows if r.get("label") and r["family"] in FAMILIES}
    out = {}
    for (layer, column), (X, y) in sorted(train.items()):
        if (layer, column) not in states:
            continue
        uids, Z = states[(layer, column)]
        probe = fit_ridge(X, y, np.round(y, 3))
        predicted = probe.predict(Z)
        for family in FAMILIES:
            arms = {}
            for rendering in ("implied", "explicit"):
                keep = [i for i, u in enumerate(uids) if u in by_uid and by_uid[u]["family"] == family
                        and by_uid[u]["rendering"] == rendering and not (column in NATURAL_ONLY and by_uid[u]["forced"])]
                true = np.array([by_uid[uids[i]]["log_h"] for i in keep])
                arms[rendering] = readout_stats(predicted[keep], true)
            out[f"{column} L{layer} {family}"] = {**arms, "verdict": c3_verdict(arms["implied"], arms["explicit"])}
    return out


# ---- criteria ----

def status(ok):
    return "pending" if ok is None else ("pass" if ok else "fail")


def criteria(behave, convert, mediate, readouts, probe_given):
    explicit = {r: [behave[f][f"explicit {r}"] for f in FAMILIES if behave[f][f"explicit {r}"]["n"]] for r in REGIMES}
    twin_acc = {r: (sum(x["correct"] for x in explicit[r]) / sum(x["n"] for x in explicit[r])) if explicit[r] else None
                for r in REGIMES}
    c0 = status(None if None in twin_acc.values() else all(a >= C0_MIN for a in twin_acc.values()))
    c1 = {}
    for family in FAMILIES:
        imp, twin = behave[family]["implied near+both"], behave[family]["explicit near+both"]
        null = behave[family]["null"]["near_share"]
        if not imp["n"]:
            c1[family] = "pending"
            continue
        ignored = null is not None and all(
            behave[family][f"implied {r}"]["n"] and abs(behave[family][f"implied {r}"]["near_share"] - null) <= C1_IGNORED
            for r in REGIMES)
        acc_i = imp["correct"] / imp["n"]
        acc_t = twin["correct"] / twin["n"] if twin["n"] else None
        ok = acc_i >= C1_MIN and (acc_t is None or acc_i >= acc_t - C1_TWIN_GAP)
        c1[family] = "pass" if ok else ("fail (ignored)" if ignored else "fail")
    n = sum(convert[f]["n"] for f in FAMILIES)
    share = sum(convert[f]["converted"] for f in FAMILIES) / n if n else None
    parts = {"converts": None if share is None else share >= C2_CONVERTS}
    for family in ("A", "C"):
        e = convert[family].get("median_error_decades")
        parts[family] = None if e is None else e <= C2_ERROR
    inside = convert["B"].get("inside_share")
    parts["B"] = None if inside is None else inside >= C2_INSIDE
    known = [v for v in parts.values() if v is not None]
    c2 = {"parts": {k: status(v) for k, v in parts.items()},
          "overall": "pending" if not known else status(all(known) and parts["converts"] is not None)}
    c4 = ("pending" if not mediate["flips"] else "underpowered" if mediate["flips"] < C4_MIN_FLIPS
          else status(mediate["follows_share"] >= C4_FOLLOWS))
    c3 = {k: v["verdict"] for k, v in readouts.items()} if readouts else ("pending" if not probe_given else "no states")
    return {"C0": c0, "C0 twin accuracy": twin_acc, "C1": c1, "C2": c2, "C3": c3, "C4": c4}


# ---- audit ----

def audit_sample(rows, n=AUDIT_N, seed=0):
    """`n` implied traces of A, B, C in equal shares, in random order: the sample for the blind audit (no
    automatic coding in it) and the key of the automatic coding."""
    rng = np.random.default_rng(seed)
    pool = {f: select(labelled(rows), f, "implied") for f in FAMILIES}
    picked = []
    for i, family in enumerate(FAMILIES):
        share = n // len(FAMILIES) + (i < n % len(FAMILIES))
        mine = pool[family]
        if not mine:
            continue
        picked += [mine[j] for j in sorted(rng.choice(len(mine), min(share, len(mine)), replace=False))]
    order = rng.permutation(len(picked))
    items, key = [], []
    for audit_id, j in enumerate(order):
        r = picked[j]
        items.append({"id": audit_id, "family": r["family"], "context": r["context"],
                      "constraint_phrase": r["horizon_text"], "forced": r["forced"], "think_text": r["think_text"]})
        stated = first_stated_implied(r["think_text"], r["near_delay_years"], r["far_delay_years"])
        key.append({"id": audit_id, "sample_uid": r["sample_uid"], "anchor_id": r["anchor_id"],
                    "true_years": 10 ** r["log_h"], "auto_stated": None if stated is None else list(stated)})
    return items, key


# ---- report ----

def pct(x):
    return "  n/a" if x is None else f"{x:5.0%}"


def format_report(summary):
    lines = [f"E18 implied horizons: {summary['n_rows']} rows ({summary['n_labelled']} labelled) in "
             f"{len(summary['units'])} units: {', '.join(summary['units']) or 'none'}", "", "(a) behaviour"]
    for family, arms in summary["behaviour"].items():
        lines.append(f"  family {family}   accuracy implied / explicit (n), natural closes only in brackets")
        for regime_name in (*REGIMES, "none"):
            cells = []
            for rendering in ("implied", "explicit"):
                t, nat = arms[f"{rendering} {regime_name}"], arms[f"{rendering} {regime_name} natural"]
                cells.append(f"{pct(headline(t, regime_name))} ({t['n']:3d}) [{pct(headline(nat, regime_name))}, "
                             f"{nat['n']:3d}]  forced {pct(t['forced'])}")
            label = "accuracy" if regime_name != "none" else "near share"
            lines.append(f"    {regime_name:5s} {label:10s} implied {cells[0]}   explicit {cells[1]}")
        m = arms["mcnemar"]
        lines.append(f"    null twin near share {pct(arms['null']['near_share'])} (n {arms['null']['n']})   McNemar: "
                     f"{m['pairs']} pairs, only implied right {m['only_implied_correct']}, only explicit right "
                     f"{m['only_explicit_correct']}, p {m['p'] if m['p'] is None else round(m['p'], 3)}")
    lines += ["", "(c) conversion (implied rows)"]
    for family, c in summary["conversion"].items():
        extra = (f"inside [lo, hi] {pct(c['inside_share'])}" if family == "B"
                 else f"median error {c['median_error_decades'] if c['median_error_decades'] is None else round(c['median_error_decades'], 3)} decades")
        lines.append(f"  {family}: states a duration in {c['converted']}/{c['n']} ({pct(c['converted_share'])}), {extra}, "
                     f"stated regime differs from the true one in {c['flips_regime']}")
    m = summary["mediation"]
    lines += ["", f"mediation: {m['flips']} regime-flipping misconversions; choice follows the stated regime in "
                  f"{m['follows_stated']} ({pct(m['follows_share'])})", "", "(b) readout (MAE in decades, Spearman rho)"]
    if not summary["readout"]:
        lines.append("  pending: " + ("no saved states" if summary["probe_given"] else "no --probe-runs given"))
    for name, r in summary["readout"].items():
        lines.append(f"  {name:22s} implied MAE {_num(r['implied']['mae'])} rho {_num(r['implied']['rho'])} (n {r['implied']['n']})"
                     f"   explicit MAE {_num(r['explicit']['mae'])} rho {_num(r['explicit']['rho'])} (n {r['explicit']['n']})"
                     f"   {r['verdict']}")
    c = summary["criteria"]
    lines += ["", "criteria", f"  C0 reference valid: {c['C0']}   twin accuracy {c['C0 twin accuracy']}",
              f"  C1 behaviour tracks: {c['C1']}", f"  C2 converts: {c['C2']['overall']}   {c['C2']['parts']}",
              f"  C3 same states: {c['C3'] if isinstance(c['C3'], str) else 'see (b), verdict per cell'}",
              f"  C4 mediation: {c['C4']}"]
    return "\n".join(lines) + "\n"


def headline(t, regime_name):
    """Accuracy of a near/both tally, near share of a none tally."""
    if regime_name == "none":
        return t["near_share"]
    return t["correct"] / t["n"] if t["n"] else None


def _num(x):
    return "  n/a" if x is None or np.isnan(x) else f"{x:5.2f}"


def summarize(run_dir, probe_runs=()):
    rows = load_rows(run_dir)
    units = sorted(p.stem for p in (Path(run_dir) / "units").glob("*.json"))
    behave, convert, mediate = behaviour(rows), conversion(rows), mediation(rows)
    readouts = readout(rows, load_states(run_dir), load_probe_training(probe_runs)) if probe_runs and units else {}
    items, key = audit_sample(rows)
    summary = {"n_rows": len(rows), "n_labelled": len(labelled(rows)), "units": units, "behaviour": behave,
               "conversion": convert, "mediation": mediate, "readout": readouts, "probe_given": bool(probe_runs),
               "criteria": criteria(behave, convert, mediate, readouts, bool(probe_runs)), "audit_key": key}
    return summary, items


def main(run_dir, probe_runs=()):
    run_dir = Path(run_dir)
    summary, items = summarize(run_dir, probe_runs)
    (run_dir / "implied_summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "implied_audit_sample.json").write_text(json.dumps({"items": items}, indent=1))
    report = format_report(summary)
    (run_dir / "implied_summary.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dir")
    parser.add_argument("--probe-runs", nargs="*", default=[], help="E0 run dirs whose activations.npz train the probe")
    args = parser.parse_args()
    main(args.dir, args.probe_runs)
