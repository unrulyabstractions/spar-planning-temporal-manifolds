"""E14 analysis: did steering the horizon phrase move what the reasoning writes and what it chooses?
(21_spec_phrase_steering.md §4 measures, §5 criteria, fixed before the run.)

    python -m pipeline.s3_analysis.by_label.exp14_analysis <dir>      # <dir> = <out>/exp14_Qwen3-8B_phrase
    -> <dir>/summary.json, <dir>/summary.txt, <dir>/summary.png (post hoc figure)

Reads <dir>/inputs.json and <dir>/conditions/*.json; runs no model. Works on partial runs:
a criterion whose conditions are not all there reads "pending", never pass or fail.

Everything is paired per prompt (sample_uid within a transition). Confidence intervals are
percentile intervals over N_BOOT resamples of prompts, seed 0; a pooled statistic resamples
the (transition, prompt) pairs.

Post hoc (2026-11-01, added after the run; 22_results_phrase_steering.md): the §4 share counts
any duration written anywhere in the trace ("that leaves 3 years to grow"), so the noop rerun
already scores 0.25-0.38 on it. The post hoc measures read the first duration the trace states
as its horizon instead, and ask whether the choice follows it. They gate nothing.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

from spar_horizon.full_bank import regime
from spar_horizon.horizons import UNIT_YEARS, canonical_unit
from pipeline.s3_analysis.by_datatype.written_unit import DURATION, close_to, durations, parse_number

N_BOOT = 2000
C = 19.2                    # near log-odds step of the thinking-on behaviour map (behaviour_step.py)
H1_SHARE = 0.30
H1_RANDOM_MAX = 0.10
H2_H3_MIN = 0.15 * C        # the oriented near log-odds change H2 and H3 must reach
LAYERS = (14, 16, 18)

# + where the transition should raise the near log-odds, - where it should lower them
SIGN = {"both-near": 1, "none-near": 1, "near-both": -1, "near-none": -1}
TRANSITIONS = tuple(SIGN)
FROM_NONE_OR_BOTH = ("both-near", "none-near")
FROM_NEAR = ("near-both", "near-none")

# the trace noticing that the stated horizon does not fit the prompt
CONFLICT = re.compile(r"typo|wait, the horizon|mistake|inconsistent", re.IGNORECASE)

# post hoc: a duration stated as the horizon ("a 13-year time horizon", "the horizon is 2.5 years")
HORIZON_AFTER = re.compile(r"[\s-]+(?:time[\s-]+)?horizon", re.IGNORECASE)
HORIZON_BEFORE = re.compile(r"horizon (?:given |here |stated )?(?:is|of) (?:about |approximately |roughly )?$",
                            re.IGNORECASE)
MOVED = 0.05                # decades (12%): further than any rendering of the same horizon
STEERED = ("effects L14", "effects L16", "effects L18", "invariant L16")


# ---- per-trace measures ----

def written_durations(row):
    """The durations the trace writes, as years, leaving out mentions of the prompt's own delays."""
    near, far = row["near_delay_years"], row["far_delay_years"]
    return [years for years, _ in durations(row["think_text"])
            if not close_to(years, near) and not close_to(years, far)]


def writes_target_regime(row):
    near, far = row["near_delay_years"], row["far_delay_years"]
    return any(regime(years, near, far) == row["target_regime"] for years in written_durations(row))


def first_stated_horizon(text):
    """(years, unit, number as written) of the first duration the trace states as its horizon, or None."""
    for match in DURATION.finditer(text):
        before = text[max(0, match.start() - 40):match.start()]
        if HORIZON_AFTER.match(text, match.end()) or HORIZON_BEFORE.search(before):
            value = parse_number(match["number"])
            if value is not None:
                unit = canonical_unit(match["unit"])
                return value * UNIT_YEARS[unit], unit, match["number"]
    return None


def digits_of(number_text):
    """The digit string of a written number ("1,250" -> "1250"; "2.5" -> "25"), or "" for a word ("fifty")."""
    return "".join(ch for ch in number_text if ch.isdigit())


def is_digit_edit(stated_number, own_number):
    """A different number whose digits contain the prompt's, or are contained in them (6 -> 66, 213 -> 13,
    25 -> 2.5)."""
    stated, own = digits_of(stated_number), digits_of(own_number)
    return bool(stated and own and stated_number != own_number and (stated in own or own in stated))


def stated_measures(row):
    """Post hoc: where the first stated horizon sits against the prompt's own horizon and the target."""
    stated = first_stated_horizon(row["think_text"])
    if stated is None:
        return {"stated_regime": None, "states_target": False, "stated_moved": False,
                "stated_toward": False, "stated_unit_changed": False, "stated_shift": np.nan, "digit_edit": False,
                "stated_is_delay": False}
    years, unit, number = stated
    shift = np.log10(years) - row["log_h"]
    moved = bool(abs(shift) > MOVED)
    stated_regime = regime(years, row["near_delay_years"], row["far_delay_years"])
    own_unit = durations(row["horizon_text"])[0][1]
    return {"stated_regime": stated_regime, "states_target": stated_regime == row["target_regime"],
            "stated_moved": moved,
            "stated_toward": moved and bool(np.sign(shift) == np.sign(row["target_log_h"] - row["log_h"])),
            "stated_unit_changed": moved and unit != own_unit,
            "stated_shift": float(shift),
            "digit_edit": moved and unit == own_unit and is_digit_edit(number, row["horizon_text"].split()[0]),
            # the H1 measure leaves such mentions out; this one keeps them and reports how many there are
            "stated_is_delay": close_to(years, row["near_delay_years"]) or close_to(years, row["far_delay_years"])}


def trace_measures(row):
    return {"writes_target": writes_target_regime(row),
            "names_target": bool(row["names_target"]), "names_own": bool(row["names_own"]),
            "conflict": bool(CONFLICT.search(row["think_text"])),
            "forced": bool(row["forced"]),
            "no_horizon": not written_durations(row),
            "chose_near": row["label"] == row["near_label"],
            **stated_measures(row)}


# ---- statistics ----

def bootstrap_mean(values, seed=0):
    """(mean, low, high) of `values`, the interval from N_BOOT resamples of its entries."""
    values = np.asarray(values, float)
    if len(values) == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    resamples = np.random.default_rng(seed).integers(0, len(values), (N_BOOT, len(values)))
    low, high = np.percentile(values[resamples].mean(axis=1), [2.5, 97.5])
    return {"n": len(values), "mean": float(values.mean()), "lo": float(low), "hi": float(high)}


def paired_difference(first, second):
    """bootstrap_mean of first - second over the keys both dicts have."""
    keys = sorted(set(first) & set(second))
    return bootstrap_mean([first[k] - second[k] for k in keys])


def ratio_of_means(numerator, denominator):
    keys = sorted(set(numerator) & set(denominator))
    if not keys:
        return None
    denominator_mean = np.mean([denominator[k] for k in keys])
    return float(np.mean([numerator[k] for k in keys]) / denominator_mean) if denominator_mean > 0 else None


def verdict(checks):
    """pass / fail / pending from a list of True, False, None (None = an input is missing)."""
    if any(check is None for check in checks):
        return "pending"
    return "pass" if all(checks) else "fail"


# ---- loading ----

def load(directory):
    """(inputs, data) with data[(transition, condition)] = {sample_uid: row + trace measures}."""
    inputs = json.loads((directory / "inputs.json").read_text())
    data = {}
    for path in sorted((directory / "conditions").glob("*.json")):
        cell = json.loads(path.read_text())
        data[(cell["transition"], cell["condition"])] = {
            row["sample_uid"]: {**row, **trace_measures(row)} for row in cell["rows"]}
    return inputs, data


def per_prompt(data, condition, transitions, value):
    """{(transition, uid): value(row)} pooled over `transitions`; None if any is missing."""
    if any((t, condition) not in data for t in transitions):
        return None
    return {(t, uid): value(row) for t in transitions for uid, row in data[(t, condition)].items()}


def share(data, condition, transitions, measure="writes_target"):
    return per_prompt(data, condition, transitions, lambda row: float(row[measure]))


def near_log_odds_change(data, condition, transitions, oriented=True):
    """{(transition, uid): change in near log-odds from the noop rerun, signed by SIGN if oriented}.
    Prompts without a finite log-odds in both runs are left out."""
    if any((t, condition) not in data or (t, "noop") not in data for t in transitions):
        return None
    changes = {}
    for t in transitions:
        noop = data[(t, "noop")]
        for uid, row in data[(t, condition)].items():
            if uid in noop and np.isfinite(row["near_lo"]) and np.isfinite(noop[uid]["near_lo"]):
                changes[(t, uid)] = (SIGN[t] if oriented else 1) * (row["near_lo"] - noop[uid]["near_lo"])
    return changes


# ---- criteria ----

def h1(data, effects, random, transitions):
    """The target-regime share of `effects` against `random` (same prompts); passes at >= 0.30,
    CI of the paired difference above 0, random <= 0.10."""
    steered, control = share(data, effects, transitions), share(data, random, transitions)
    if steered is None or control is None:
        return {"effects": effects, "random": random, "pass": None}
    result = {"effects": effects, "random": random, "share": bootstrap_mean(list(steered.values())),
              "random_share": bootstrap_mean(list(control.values())),
              "difference": paired_difference(steered, control)}
    result["pass"] = bool(result["share"]["mean"] >= H1_SHARE and result["difference"]["lo"] > 0
                          and result["random_share"]["mean"] <= H1_RANDOM_MAX)
    return result


def h2(data, condition, transitions):
    """Oriented change in near log-odds from the noop rerun; passes at >= 0.15 C with CI above 0."""
    changes = near_log_odds_change(data, condition, transitions)
    if changes is None:
        return {"condition": condition, "pass": None}
    result = {"condition": condition, "change": bootstrap_mean(list(changes.values()))}
    result["pass"] = bool(result["change"]["n"] and result["change"]["mean"] >= H2_H3_MIN
                          and result["change"]["lo"] > 0)
    return result


def h3_against_opposite(data, opposite_clamped, exclude_clamped=False):
    """Effects minus opposite step, oriented, on the none and both starts."""
    effects = per_prompt(data, "effects L16", FROM_NONE_OR_BOTH, lambda row: row["near_lo"])
    opposite = per_prompt(data, "opposite L16", FROM_NONE_OR_BOTH, lambda row: row["near_lo"])
    if effects is None or opposite is None:
        return {"pass": None}
    keys = [k for k in sorted(set(effects) & set(opposite))
            if np.isfinite(effects[k]) and np.isfinite(opposite[k])
            and not (exclude_clamped and opposite_clamped.get(k[1], False))]
    stat = bootstrap_mean([SIGN[k[0]] * (effects[k] - opposite[k]) for k in keys])
    return {"difference": stat, "pass": bool(stat["n"] and stat["mean"] >= H2_H3_MIN and stat["lo"] > 0)}


def h3_near_starts(data):
    """Both reverses from a near start lower the near log-odds: unoriented change, CI below 0."""
    result = {}
    for t in FROM_NEAR:
        changes = near_log_odds_change(data, "effects L16", (t,), oriented=False)
        if changes is None:
            result[t] = {"pass": None}
        else:
            stat = bootstrap_mean(list(changes.values()))
            result[t] = {"change": stat, "pass": bool(stat["n"] and stat["hi"] < 0)}
    return result


def donor_ceiling(data):
    """Effects over donor, for the H1 share and the H2 change, on the prompts that have a donor row."""
    shares = (share(data, "effects L16", TRANSITIONS), share(data, "donor L16", TRANSITIONS))
    changes = (near_log_odds_change(data, "effects L16", TRANSITIONS),
               near_log_odds_change(data, "donor L16", TRANSITIONS))
    return {"h1_share_ratio": None if None in shares else ratio_of_means(*shares),
            "h2_change_ratio": None if None in changes else ratio_of_means(*changes)}


# ---- post hoc: does the choice follow the stated horizon? ----

LANDING_GROUPS = ("lands in target regime", "landed, natural closes", "moved, not into target", "own or none stated",
                  "steered, not landed", "donor L16")
NOT_LANDED = ("moved, not into target", "own or none stated")
BEHAVIOUR_FIT = Path("analysis") / "behaviour_step.json"   # under the results root, the run folder's parent
RANDOM_CONDITIONS = ("random L14", "random L16", "random L18")
MIN_PROMPTS_SHOWN = 3         # the figure leaves out choice groups from fewer prompts


def clustered_mean(values, clusters, seed=0):
    """bootstrap_mean, resampling clusters (prompts) instead of entries: a prompt can appear once per
    steered condition, so its traces are not independent."""
    values = np.asarray(values, float)
    if len(values) == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    ids, index = np.unique(np.asarray(clusters), return_inverse=True)
    sums, counts = np.bincount(index, values, len(ids)), np.bincount(index, minlength=len(ids))
    draws = np.random.default_rng(seed).integers(0, len(ids), (N_BOOT, len(ids)))
    low, high = np.percentile(sums[draws].sum(1) / counts[draws].sum(1), [2.5, 97.5])
    return {"n": len(values), "mean": float(values.mean()), "lo": float(low), "hi": float(high)}


def landing_group(condition, row):
    if condition == "donor L16":
        return "donor L16"
    if condition in RANDOM_CONDITIONS:
        return "random"
    if row["states_target"]:
        return "lands in target regime"
    return "moved, not into target" if row["stated_moved"] else "own or none stated"


def choice_changes(data):
    """(group, transition, uid, P(near) change, near log-odds change) per steered, random and donor trace,
    oriented by SIGN and taken against the noop rerun of the same prompt."""
    entries = []
    for (t, condition), rows in data.items():
        if condition not in STEERED + RANDOM_CONDITIONS + ("donor L16",):
            continue
        noop = data.get((t, "noop"), {})
        for uid, row in rows.items():
            if uid in noop:
                group = landing_group(condition, row)
                entry = (SIGN[t] * (float(row["chose_near"]) - float(noop[uid]["chose_near"])),
                         SIGN[t] * (row["near_lo"] - noop[uid]["near_lo"]))
                entries.append((group, t, uid, *entry))
                if group == "lands in target regime" and not row["forced"]:
                    entries.append(("landed, natural closes", t, uid, *entry))
    return entries


def change_stats(entries):
    """Both choice changes over `entries`, CIs resampling prompts; with the number of prompts."""
    prompts = [f"{t}/{uid}" for _, t, uid, _, _ in entries]
    finite = [(prompt, lo) for prompt, (*_, lo) in zip(prompts, entries) if np.isfinite(lo)]
    return {"prompts": len(set(prompts)),
            "p_near": clustered_mean([e[3] for e in entries], prompts),
            "near_log_odds": clustered_mean([lo for _, lo in finite], [prompt for prompt, _ in finite])}


def choice_by_landing(data):
    """The choice change on the steered traces (STEERED) split by where the first stated horizon lands, with
    random (L14-18) and donor L16 alongside; per transition; and the landed traces against the donor patch of the
    same prompts. Landing happens after the steer, so the split is not randomized: the same-prompt donor
    comparison and the per-transition split are the checks on what selection could do."""
    entries = choice_changes(data)
    # the steered traces that did not land, both kinds together
    entries += [("steered, not landed", *e[1:]) for e in entries if e[0] in NOT_LANDED]
    donor = {(e[1], e[2]): e for e in entries if e[0] == "donor L16"}
    landed = [e for e in entries if e[0] == "lands in target regime"]
    by_transition = {}
    for t in TRANSITIONS:
        here = [e for e in entries if e[1] == t]
        landed_here = [e for e in landed if e[1] == t and (t, e[2]) in donor]
        # random on the prompts that landed: the baseline for regression to the mean under label churn
        landed_prompts = {e[2] for e in landed if e[1] == t}
        by_transition[t] = {**{name: change_stats([e for e in here if e[0] == name])
                               for name in LANDING_GROUPS + ("random",)},
                            "random, same prompts as landed": change_stats([e for e in here if e[0] == "random"
                                                                             and e[2] in landed_prompts]),
                            "landed, same prompts": change_stats(landed_here),
                            "donor, same prompts": change_stats([donor[(t, e[2])] for e in landed_here])}
    with_donor = [e for e in landed if (e[1], e[2]) in donor]
    return {"groups": {name: change_stats([e for e in entries if e[0] == name])
                       for name in LANDING_GROUPS + ("random",)},
            "landed_vs_donor_same_prompts": {"landed": change_stats(with_donor),
                                             "donor": change_stats([donor[(e[1], e[2])] for e in with_donor])},
            "by_transition": by_transition}


def within_prompt_landing(data):
    """Per transition, over prompts with both landed and not-landed steered traces: the mean over prompts of
    (landed minus not-landed oriented near log-odds change), CI resampling prompts. Removes what differs between
    prompts; not what differs between traces of one prompt."""
    entries = choice_changes(data)
    out = {}
    for t in TRANSITIONS:
        differences = []
        for uid in sorted({e[2] for e in entries if e[1] == t}):
            landed = [e[4] for e in entries if e[1] == t and e[2] == uid and e[0] == "lands in target regime"
                      and np.isfinite(e[4])]
            others = [e[4] for e in entries if e[1] == t and e[2] == uid and e[0] in NOT_LANDED and np.isfinite(e[4])]
            if landed and others:
                differences.append(np.mean(landed) - np.mean(others))
        out[t] = bootstrap_mean(differences)
    return out


def hinge_near_log_odds(fit, log_h, log_near, log_far):
    """The delay terms of the unsteered thinking-on behaviour map (behaviour_step.py, hinge) at a horizon:
    per delay, slope * m + slope change * max(m, 0) + step * 1(m < 0), m = log10(delay / horizon)."""
    total = 0.0
    for role, log_delay in (("near", log_near), ("far", log_far)):
        m = log_delay - log_h
        total += (fit[f"{role} slope"][0] * m + fit[f"{role} slope change above 0"][0] * max(m, 0.0)
                  + fit[f"{role} step"][0] * float(m < 0))
    return total


def behaviour_prediction(data, fit):
    """Per transition and landing group, over steered traces that state a horizon: the mean oriented change in near
    log-odds observed, beside what the behaviour map predicts from the move of the stated horizon alone."""
    out = {}
    for t in TRANSITIONS:
        noop = data.get((t, "noop"), {})
        groups = {}
        for c in STEERED:
            for uid, row in data.get((t, c), {}).items():
                if uid not in noop or not np.isfinite(row["stated_shift"]) or not np.isfinite(row["near_lo"]):
                    continue
                log_near, log_far = np.log10(row["near_delay_years"]), np.log10(row["far_delay_years"])
                predicted = (hinge_near_log_odds(fit, row["log_h"] + row["stated_shift"], log_near, log_far)
                             - hinge_near_log_odds(fit, row["log_h"], log_near, log_far))
                observed = row["near_lo"] - noop[uid]["near_lo"]
                groups.setdefault(landing_group(c, row), []).append((SIGN[t] * observed, SIGN[t] * predicted))
        out[t] = {name: {"n": len(pairs), "observed": float(np.mean([o for o, _ in pairs])),
                         "predicted": float(np.mean([q for _, q in pairs]))} for name, pairs in groups.items()}
    return out


def natural_closes(data):
    """`data` without force-closed traces (in every condition, the rerun included). A forced close pulls the near
    log-odds toward 0 (about 4 in size, against about 12 for natural closes), which reads as a large change on the
    near starts, whose reruns sit near +12."""
    return {key: {uid: row for uid, row in rows.items() if not row["forced"]} for key, rows in data.items()}


def landed_by_condition(data, fit):
    """Per transition and steered condition: the landed traces' mean oriented change in near log-odds, observed and
    predicted by the behaviour map (fit None: observed only)."""
    out = {}
    for t in TRANSITIONS:
        noop = data.get((t, "noop"), {})
        for c in STEERED:
            pairs = []
            for uid, row in data.get((t, c), {}).items():
                if uid in noop and row["states_target"] and np.isfinite(row["near_lo"]):
                    log_near, log_far = np.log10(row["near_delay_years"]), np.log10(row["far_delay_years"])
                    predicted = (hinge_near_log_odds(fit, row["log_h"] + row["stated_shift"], log_near, log_far)
                                 - hinge_near_log_odds(fit, row["log_h"], log_near, log_far)) if fit else np.nan
                    pairs.append((SIGN[t] * (row["near_lo"] - noop[uid]["near_lo"]), SIGN[t] * predicted))
            if pairs:
                out[f"{t} | {c}"] = {"n": len(pairs), "observed": float(np.mean([o for o, _ in pairs])),
                                     "predicted": float(np.mean([q for _, q in pairs]))}
    return out


def h2_by_landing(data, condition="effects L16", transitions=TRANSITIONS):
    """H2's oriented change split by whether the trace's stated horizon landed: n, mean, and share of the total."""
    changes = near_log_odds_change(data, condition, transitions)
    if changes is None:
        return None
    landed = {k: v for k, v in changes.items() if data[(k[0], condition)][k[1]]["states_target"]}
    others = {k: v for k, v in changes.items() if k not in landed}
    total = sum(changes.values())
    return {name: {"n": len(group), "mean": float(np.mean(list(group.values()))) if group else None,
                   "share_of_total": float(sum(group.values()) / total) if total else None}
            for name, group in (("landed", landed), ("not landed", others))}


def shift_choice_correlation(data):
    """Per transition, over steered traces that state a horizon: the correlation between how far the stated
    horizon moved toward the target (decades, signed) and the oriented change in near log-odds from noop."""
    out = {}
    for t in TRANSITIONS:
        noop = data.get((t, "noop"), {})
        pairs = [(np.sign(row["target_log_h"] - row["log_h"]) * row["stated_shift"],
                  SIGN[t] * (row["near_lo"] - noop[uid]["near_lo"]))
                 for c in STEERED for uid, row in data.get((t, c), {}).items()
                 if uid in noop and np.isfinite(row["stated_shift"]) and np.isfinite(row["near_lo"])
                 and np.isfinite(noop[uid]["near_lo"])]
        shift, change = np.array(pairs).T if pairs else (np.array([]), np.array([]))
        out[t] = {"n": len(pairs), "r": float(np.corrcoef(shift, change)[0, 1]) if len(pairs) > 2 and shift.std() > 0
                  else None}
    return out


def choice_by_stated_regime(data):
    """P(near) by the regime of the first stated horizon, every condition pooled, split by whether
    it is the prompt's own horizon or a moved one."""
    out = {}
    for moved in (False, True):
        for stated_regime in ("none", "near", "both"):
            values = [float(row["chose_near"]) for rows in data.values() for row in rows.values()
                      if row["stated_regime"] == stated_regime and row["stated_moved"] == moved]
            out[f"{'moved' if moved else 'own'}, {stated_regime}"] = bootstrap_mean(values)
    return out


# ---- the table of every cell ----

def cell_table(data):
    """One row per (transition, condition) present: the §4 measures."""
    table = []
    for (transition, condition), rows in sorted(data.items(), key=lambda kv: (TRANSITIONS.index(kv[0][0]), kv[0][1])):
        noop = data.get((transition, "noop"), {})
        flips = [row["label"] != noop[uid]["label"] for uid, row in rows.items() if uid in noop]
        changes = near_log_odds_change(data, condition, (transition,)) if condition != "noop" else {}
        entry = {"transition": transition, "condition": condition, "n": len(rows),
                 "label_flips_vs_noop": float(np.mean(flips)) if flips else None,
                 "oriented_near_log_odds_change": float(np.mean(list(changes.values()))) if changes else None}
        for measure in ("writes_target", "names_target", "names_own", "conflict", "forced", "no_horizon",
                        "chose_near", "states_target", "stated_toward", "stated_unit_changed"):
            entry[measure] = float(np.mean([row[measure] for row in rows.values()]))
        table.append(entry)
    return table


def analyse(directory):
    directory = Path(directory)
    inputs, data = load(directory)
    opposite_clamped = {uid: prompt["opposite_clamped"]
                        for uid, prompt in zip(inputs["sample_uids"], inputs["prompts"])}

    layer_profile = {}
    for layer in LAYERS:
        layer_profile[f"L{layer}"] = {"h1": h1(data, f"effects L{layer}", f"random L{layer}", TRANSITIONS),
                                      "h2": h2(data, f"effects L{layer}", TRANSITIONS)}
    gate_h1, gate_h2 = layer_profile["L16"]["h1"], layer_profile["L16"]["h2"]
    h3_opposite = h3_against_opposite(data, opposite_clamped)
    h3_unclamped = h3_against_opposite(data, opposite_clamped, exclude_clamped=True)
    h3_near = h3_near_starts(data)

    present = set(data)
    expected = [(t, c) for t in TRANSITIONS for c in expected_conditions(t)]
    return {
        "constants": {"C": C, "n_boot": N_BOOT, "h1_share": H1_SHARE, "h1_random_max": H1_RANDOM_MAX,
                      "h2_h3_min": H2_H3_MIN, "conflict_patterns": CONFLICT.pattern},
        "verdicts": {
            "H1": verdict([gate_h1["pass"]]),
            "H2": verdict([gate_h2["pass"]]),
            "H3": verdict([h3_opposite["pass"]] + [h3_near[t]["pass"] for t in FROM_NEAR])},
        "h1": gate_h1, "h2": gate_h2,
        "h3": {"against_opposite": h3_opposite, "against_opposite_without_clamped": h3_unclamped,
               "near_starts": h3_near},
        "reported": {
            "invariant_L16": {"h1": h1(data, "invariant L16", "random L16", TRANSITIONS),
                              "h2": h2(data, "invariant L16", TRANSITIONS)},
            "layer_profile": layer_profile,
            "donor_ceiling": donor_ceiling(data),
            "per_transition": {t: {"h1": h1(data, "effects L16", "random L16", (t,)),
                                   "h2": h2(data, "effects L16", (t,))} for t in TRANSITIONS},
            "in_span_share": [{k: cell.get(k) for k in ("layer", "role", "n", "k", "factor_span_dims", "in_span_share")}
                              for cell in inputs.get("cells", [])]},
        "post_hoc": {"choice_by_landing": choice_by_landing(data),
                     "within_prompt_landing": within_prompt_landing(data),
                     "behaviour_prediction": (behaviour_prediction(data, behaviour_fit)
                                              if (behaviour_fit := load_behaviour_fit(directory)) else None),
                     "h2_by_landing": h2_by_landing(data),
                     "landed_by_condition": landed_by_condition(data, behaviour_fit),
                     "natural_closes": {"choice_by_landing": choice_by_landing(natural_closes(data)),
                                        "behaviour_prediction": (behaviour_prediction(natural_closes(data), behaviour_fit)
                                                                 if behaviour_fit else None),
                                        "h2": h2(natural_closes(data), "effects L16", TRANSITIONS)},
                     "shift_choice_correlation": shift_choice_correlation(data),
                     "choice_by_stated_regime": choice_by_stated_regime(data)},
        "cells": cell_table(data),
        "pending": [f"{t}__{c.replace(' ', '_')}" for t, c in expected if (t, c) not in present],
    }


def load_behaviour_fit(directory):
    """The thinking-on hinge fit of behaviour_step.py, or None if that analysis has not been run."""
    path = Path(directory).parent / BEHAVIOUR_FIT
    return json.loads(path.read_text())["on"]["all"]["fits"]["hinge"] if path.exists() else None


def expected_conditions(transition):
    conditions = ["noop", "effects L16", "random L16", "donor L16", "invariant L16"]
    if transition in FROM_NONE_OR_BOTH:
        conditions.append("opposite L16")
    return conditions + ["effects L14", "effects L18", "random L14", "random L18"]


# ---- text ----

def fmt(stat, digits=2):
    if not stat or stat.get("mean") is None:
        return "pending"
    return f"{stat['mean']:.{digits}f} [{stat['lo']:.{digits}f}, {stat['hi']:.{digits}f}] n={stat['n']}"


def fmt_or_none(stat):
    """fmt, but an empty group reads n=0 rather than pending (post hoc groups can be empty)."""
    return fmt(stat) if stat["n"] else "n=0"


def status(passed):
    return "pending" if passed is None else "pass" if passed else "fail"


def render(summary):
    v = summary["verdicts"]
    h1_, h2_, h3_ = summary["h1"], summary["h2"], summary["h3"]
    lines = [f"H1 {v['H1']}: L16 effects share writing a target-regime horizon >= {H1_SHARE}, CI of (effects - random,"
             f" same prompts) excluding 0, random <= {H1_RANDOM_MAX}",
             f"    effects {fmt(h1_.get('share'))}  random {fmt(h1_.get('random_share'))}  difference {fmt(h1_.get('difference'))}",
             f"H2 {v['H2']}: oriented change in near log-odds from noop, all four transitions, >= {H2_H3_MIN:.2f} (0.15 C, C={C}),"
             f" CI above 0",
             f"    {fmt(h2_.get('change'))}",
             f"H3 {v['H3']}: effects - opposite (oriented, none and both starts) >= {H2_H3_MIN:.2f} with CI above 0;"
             f" near starts: both reverses CI below 0",
             f"    effects - opposite {fmt(h3_['against_opposite'].get('difference'))}  [{status(h3_['against_opposite']['pass'])}]",
             f"    without clamped opposites {fmt(h3_['against_opposite_without_clamped'].get('difference'))}"
             f"  [{status(h3_['against_opposite_without_clamped']['pass'])}, reported only]"]
    for t in FROM_NEAR:
        lines.append(f"    {t} change {fmt(h3_['near_starts'][t].get('change'))}  [{status(h3_['near_starts'][t]['pass'])}]")
    lines.append("(CIs: 2000 prompt resamples, seed 0; H1's 'CI above random' is the CI of the paired difference)")
    if summary["pending"]:
        lines.append(f"pending conditions ({len(summary['pending'])}): " + ", ".join(summary["pending"]))

    reported = summary["reported"]
    lines += ["", "Reported, not gating", "layer profile (H1 vs own random, H2):"]
    for name, entry in reported["layer_profile"].items():
        lines.append(f"  {name}: share {fmt(entry['h1'].get('share'))} vs random {fmt(entry['h1'].get('random_share'))}"
                     f" [{status(entry['h1']['pass'])}]; change {fmt(entry['h2'].get('change'))} [{status(entry['h2']['pass'])}]")
    inv = reported["invariant_L16"]
    lines.append(f"invariant L16: share {fmt(inv['h1'].get('share'))} [{status(inv['h1']['pass'])}];"
                 f" change {fmt(inv['h2'].get('change'))} [{status(inv['h2']['pass'])}]")
    ceiling = reported["donor_ceiling"]
    lines.append("donor ceiling (effects / donor): share " + ("pending" if ceiling["h1_share_ratio"] is None
                 else f"{ceiling['h1_share_ratio']:.2f}") + ", change " + ("pending" if ceiling["h2_change_ratio"] is None
                 else f"{ceiling['h2_change_ratio']:.2f}"))
    lines.append("per transition (L16 effects):")
    for t, entry in reported["per_transition"].items():
        lines.append(f"  {t}: share {fmt(entry['h1'].get('share'))} vs random {fmt(entry['h1'].get('random_share'))};"
                     f" change {fmt(entry['h2'].get('change'))}")
    lines.append("in-span share of the step (from inputs.json):")
    for cell in reported["in_span_share"]:
        share_ = cell["in_span_share"]
        lines.append(f"  L{cell['layer']} {cell['role']}: {'n/a' if share_ is None else format(share_, '.2f')}"
                     f" (n={cell['n']}, k={cell['k']}, factor span dims={cell['factor_span_dims']})")

    post_hoc = summary["post_hoc"]
    landing = post_hoc["choice_by_landing"]

    def both(stat):
        return (f"near log-odds {fmt_or_none(stat['near_log_odds'])}   P(near) {fmt_or_none(stat['p_near'])}"
                f"   ({stat['prompts']} prompts)")

    lines += ["", "Post hoc (added 2026-11-01, gates nothing): the first duration the trace states as its horizon.",
              "Choice change from the noop rerun, oriented; CIs resample prompts. Steered = effects L14-18 and invariant L16.",
              "by where the steered trace's stated horizon lands (random L14-18 and donor L16 alongside):"]
    lines += [f"  {name:24s} {both(stat)}" for name, stat in landing["groups"].items()]
    lines.append("landed traces against the donor patch of the same prompts:")
    lines += [f"  {name:24s} {both(stat)}" for name, stat in landing["landed_vs_donor_same_prompts"].items()]
    lines.append("per transition:")
    for t, groups in landing["by_transition"].items():
        lines.append(f"  {t}:")
        lines += [f"    {name:24s} {both(stat)}" for name, stat in groups.items()]
    lines.append("within prompts, landed minus not-landed steered traces (prompts with both; CI resamples prompts):")
    lines += [f"  {t}: {fmt_or_none(stat)}" for t, stat in post_hoc["within_prompt_landing"].items()]
    if post_hoc["behaviour_prediction"] is not None:
        lines.append("observed vs predicted by the unsteered behaviour map from the stated horizon's move alone (steered):")
        for t, groups in post_hoc["behaviour_prediction"].items():
            lines += [f"  {t:9s} {name:24s} n {g['n']:3d}  observed {g['observed']:+6.2f}  predicted {g['predicted']:+6.2f}"
                      for name, g in groups.items()]
    split = post_hoc["h2_by_landing"]
    if split is not None:
        lines.append("split of H2's change (effects L16) by landing: " + "; ".join(
            f"{name} n {g['n']} mean {g['mean']:+.2f} ({g['share_of_total']:.0%} of the total)"
            for name, g in split.items() if g["mean"] is not None))
    lines.append("landed traces by condition, observed vs predicted:")
    lines += [f"  {key:28s} n {g['n']:3d}  observed {g['observed']:+6.2f}  predicted {g['predicted']:+6.2f}"
              for key, g in post_hoc["landed_by_condition"].items()]
    natural = post_hoc["natural_closes"]
    lines.append(f"natural closes only (forced closes dropped everywhere, reruns included): H2 {fmt(natural['h2'].get('change'))}")
    for t, groups in natural["choice_by_landing"]["by_transition"].items():
        shown = ("random", "own or none stated", "moved, not into target", "lands in target regime", "donor L16")
        lines.append(f"  {t:9s} " + "  ".join(f"{name.split(',')[0].split(' in ')[0]} {fmt_or_none(groups[name]['near_log_odds'])}"
                                              f" ({groups[name]['prompts']})" for name in shown))
    if natural["behaviour_prediction"] is not None:
        lines.append("  natural closes, observed vs predicted:")
        for t, groups in natural["behaviour_prediction"].items():
            lines += [f"    {t:9s} {name:24s} n {g['n']:3d}  observed {g['observed']:+6.2f}  predicted {g['predicted']:+6.2f}"
                      for name, g in groups.items()]
    lines.append("correlation of the stated horizon's shift toward the target with the choice change, steered traces:")
    lines += [f"  {t}: r {'n/a' if v['r'] is None else format(v['r'], '+.2f')} (n {v['n']})"
              for t, v in post_hoc["shift_choice_correlation"].items()]
    lines.append("P(near) by the regime of the first stated horizon, all conditions (descriptive: each cell mixes start")
    lines.append("regimes and sources, so compare within a transition above, not across these cells):")
    for name, stat in post_hoc["choice_by_stated_regime"].items():
        lines.append(f"  {name:12s} {fmt_or_none(stat)}")

    lines += ["", f"{'transition':10s} {'condition':14s} {'n':>3s} {'target':>6s} {'names_t':>7s} {'names_o':>7s} {'dLO':>6s}"
                  f" {'flips':>5s} {'conflict':>8s} {'forced':>6s} {'no_hor':>6s} {'near':>5s} {'st_tgt':>6s}"
                  f" {'toward':>6s} {'unit':>5s}"]
    for c in summary["cells"]:
        change = c["oriented_near_log_odds_change"]
        flips = c["label_flips_vs_noop"]
        lines.append(f"{c['transition']:10s} {c['condition']:14s} {c['n']:3d} {c['writes_target']:6.2f} {c['names_target']:7.2f}"
                     f" {c['names_own']:7.2f} {'' if change is None else format(change, '6.2f'):>6s}"
                     f" {'' if flips is None else format(flips, '5.2f'):>5s} {c['conflict']:8.2f} {c['forced']:6.2f} {c['no_horizon']:6.2f}"
                     f" {c['chose_near']:5.2f} {c['states_target']:6.2f} {c['stated_toward']:6.2f} {c['stated_unit_changed']:5.2f}")
    lines += ["dLO = oriented change in near log-odds from noop (+ means toward the predicted direction)",
              "post hoc columns: near = P(chose near); st_tgt = first stated horizon in the target regime;",
              f"toward = first stated horizon moved (> {MOVED} decades) toward the target; unit = moved and in another unit"]
    return "\n".join(lines) + "\n"


def plot(summary, path):
    """Post hoc figure. Left: per transition, the share of traces whose first stated horizon moved
    toward the target or landed in its regime. Right: per transition, the oriented change in near
    log-odds from noop by where the stated horizon lands, the donor on the same prompts alongside."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = {(c["transition"], c["condition"]): c for c in summary["cells"]}
    # dataviz reference palette slots 1-3 (validated together), gray for the control
    bars = [("random L16", "stated_toward", "random: moved toward target (0 in every transition)", "#a3a29b"),
            ("effects L16", "stated_toward", "effects step: moved toward target", "#2a78d6"),
            ("effects L16", "states_target", "effects step: in target regime", "#1baf7a"),
            ("donor L16", "states_target", "donor patch: in target regime", "#eb6834")]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    width = 0.19
    for j, (condition, measure, label, color) in enumerate(bars):
        values = [cells[(t, condition)][measure] if (t, condition) in cells else np.nan for t in TRANSITIONS]
        left.bar(np.arange(len(TRANSITIONS)) + (j - 1.5) * width, values, width * 0.9, color=color, label=label)
    left.set_xticks(range(len(TRANSITIONS)), [t.replace("-", " → ") for t in TRANSITIONS], fontsize=8)
    left.set_ylim(0, 1.05)
    left.set_ylabel("share of traces (L16)", fontsize=8)
    left.set_title("First horizon the trace states", fontsize=9)
    left.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    # natural closes only: a forced close pulls the near log-odds toward 0 (natural_closes)
    by_transition = summary["post_hoc"]["natural_closes"]["choice_by_landing"]["by_transition"]
    # dataviz reference palette slots 1-5 and gray (the run's palette, validated); the donor twice: on the prompts
    # whose steered traces landed (the fair comparison) and on all prompts (hollow)
    series = [("random", "random step", "#a3a29b", "o"),
              ("own or none stated", "steered, stated horizon unmoved", "#eda100", "o"),
              ("moved, not into target", "steered, moved but not into the target regime", "#1baf7a", "o"),
              ("lands in target regime", "steered, stated horizon in the target regime", "#2a78d6", "o"),
              ("donor, same prompts", "donor patch, same prompts as the landed", "#eb6834", "o"),
              ("donor L16", "donor patch, all prompts", "#eb6834", "hollow")]
    rows = list(reversed(TRANSITIONS))
    labelled = set()
    for j, (key, label, color, marker) in enumerate(series):
        offset = (2.5 - j) * 0.14
        for i, t in enumerate(rows):
            group = by_transition[t][key]
            stat = group["near_log_odds"]
            if group["prompts"] < MIN_PROMPTS_SHOWN:
                continue
            right.plot([stat["lo"], stat["hi"]], [i + offset] * 2, color=color, lw=2)
            right.plot(stat["mean"], i + offset, "o", color=color, ms=6, label=None if key in labelled else label,
                       markerfacecolor="white" if marker == "hollow" else color)
            labelled.add(key)
            right.annotate(f"{group['prompts']}", (stat["hi"], i + offset), xytext=(4, -3), textcoords="offset points",
                           fontsize=6, color="#5f5e5a")
    right.set_yticks(range(len(rows)), [t.replace("-", " → ") for t in rows], fontsize=8)
    right.axvline(0, color="#7a7974", lw=0.8)
    right.set_xlabel("change in near log-odds from the noop rerun, toward the prediction (C = 19.2)", fontsize=8)
    right.set_title(f"Choice change by where the stated horizon lands (natural closes)\nsteered = effects L14-18 +"
                    f" invariant L16; n prompts at each CI; < {MIN_PROMPTS_SHOWN} prompts not shown", fontsize=9)
    if right.get_legend_handles_labels()[0]:   # a partial run can have no transition with every series
        right.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    for ax in (left, right):
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("E14 post hoc: what the steer did to the stated horizon (left) and to the choice (right)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(directory):
    directory = Path(directory)
    summary = analyse(directory)
    (directory / "summary.json").write_text(json.dumps(summary, indent=1))
    text = render(summary)
    (directory / "summary.txt").write_text(text)
    plot(summary, directory / "summary.png")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
