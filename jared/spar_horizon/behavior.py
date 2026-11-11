"""Behavior: choice parsing, what the base answers alone show, and the three
coherence tests from the paper (which regenerate swapped and relabeled
prompts)."""

from statistics import mean

import numpy as np
from scipy.special import logsumexp

from .model_io import generate_batch
from .prompts import horizon_text


def parse_choice(answer_text, labels=("a", "b")):
    """0 for the first label, 1 for the second, None if neither appears."""
    if not answer_text:
        return None
    hits = {i: answer_text.find(f"{lab})") for i, lab in enumerate(labels)}
    hits = {i: pos for i, pos in hits.items() if pos != -1}
    return min(hits, key=hits.get) if hits else None


def only_near_delivers(rec):
    """True when the horizon lets only the near option pay out in time."""
    h = rec["horizon"]
    if h is None:
        return False
    return rec["near_delay_years"] <= h < rec["far_delay_years"]


def near_index(rec):
    """0 if the near option is the first option of the record's prompt, else 1."""
    return 0 if rec.get("near_label", "a") == "a" else 1


def near_or_far(rec, answer):
    """'near' or 'far' for the option the answer picks, None if unparsed."""
    choice = parse_choice(answer)
    if choice is None:
        return None
    return "near" if choice == near_index(rec) else "far"


def _rate(hits_n):
    hits, n = hits_n
    return hits / n if n else None


def base_behavior(records, answers):
    """Behavior read from the base answers alone, with no extra generation.
    Returns a dict with
    - `by_horizon`: per horizon (None = no horizon, last): parsed answers `n`,
      `near_rate`, and the near rate split by whether the near option was
      labelled a) or b);
    - `position_bias`: mean over horizons of near_rate(near is a) minus
      near_rate(near is b). That equals P(a | near is a) + P(a | near is b) - 1:
      0 is no position bias, +1 always a), -1 always b). None when no horizon
      has the near option in both slots (the starter bank, tiny subsets);
    - `accuracy`: per regime with a defined answer, [hits, n] against `truth`."""
    bins, accuracy = {}, {}
    for rec, answer in zip(records, answers):
        pick = near_or_far(rec, answer)
        if pick is None:
            continue
        slot = "a" if near_index(rec) == 0 else "b"
        tally = bins.setdefault(rec["horizon"], {"a": [0, 0], "b": [0, 0]})[slot]
        tally[0] += pick == "near"
        tally[1] += 1
        if rec.get("truth") is not None:
            hits = accuracy.setdefault(rec["regime"], [0, 0])
            hits[0] += pick == rec["truth"]
            hits[1] += 1
    by_horizon, diffs = [], []
    for h in sorted(bins, key=lambda h: (h is None, h or 0)):
        a, b = bins[h]["a"], bins[h]["b"]
        by_horizon.append({"horizon": h, "n": a[1] + b[1], "near_rate": _rate((a[0] + b[0], a[1] + b[1])),
                           "near_rate_near_a": _rate(a), "near_rate_near_b": _rate(b)})
        if a[1] and b[1]:
            diffs.append(_rate(a) - _rate(b))
    result = {"by_horizon": by_horizon, "position_bias": mean(diffs) if diffs else None,
              "accuracy": accuracy}
    print_base_behavior(result)
    return result


def print_base_behavior(r):
    print("\nbase-answer behavior (no extra generation):")
    bias = r["position_bias"]
    print(f"  position bias       {'n/a (no horizon has the near option in both slots)' if bias is None else f'{bias:+.2f}'}"
          f"  (0 none, +1 always a), -1 always b))")
    for regime, (hits, n) in sorted(r["accuracy"].items()):
        print(f"  accuracy, {regime:<5}     {hits}/{n}  ({hits / max(n, 1):.0%} pick the rational option)")
    print("  near rate by horizon: "
          + "  ".join(f"{_horizon_label(row['horizon'])}:{row['near_rate']:.2f}" for row in r["by_horizon"]))


def _horizon_label(years):
    return "none" if years is None else horizon_text(years)


def behavior(tokenizer, model, device, records, base_answers, cfg, prompt_fn, system=None):
    """Temporal reasoning, order stability, label stability. Returns a dict of
    (hits, n) and prints the three lines. `prompt_fn(rec, swap=, labels=)`
    builds the variant prompts, so perturbed banks can reuse the tests.
    The swapped and relabeled banks are generated in batches."""
    bases = [parse_choice(a) for a in base_answers]
    live = [i for i, b in enumerate(bases) if b is not None]
    swapped_gens = generate_batch(tokenizer, model, device,
                                  [prompt_fn(records[i], swap=True) for i in live], cfg, system)
    relabeled_gens = generate_batch(tokenizer, model, device,
                                    [prompt_fn(records[i], labels=("1", "2")) for i in live], cfg, system)
    reasoning_hits, reasoning_n = 0, 0
    order_same, order_n = 0, 0
    label_same, label_n = 0, 0
    for j, i in enumerate(live):
        base, rec = bases[i], records[i]
        if only_near_delivers(rec):
            reasoning_n += 1
            reasoning_hits += base == near_index(rec)
        swapped = parse_choice(swapped_gens[j].answer_text(tokenizer))
        if swapped is not None:
            order_n += 1
            order_same += (1 - swapped) == base
        relabeled = parse_choice(relabeled_gens[j].answer_text(tokenizer), labels=("1", "2"))
        if relabeled is not None:
            label_n += 1
            label_same += relabeled == base
    result = {"reasoning": (reasoning_hits, reasoning_n), "order": (order_same, order_n),
              "label": (label_same, label_n),
              "forced": sum(g.forced for g in swapped_gens + relabeled_gens)}
    print_behavior(result)
    return result


def print_behavior(r):
    def pct(k):
        hits, n = r[k]
        return f"{hits}/{n}  ({hits / max(n, 1):.0%}"
    print("\nbehavior:")
    print(f"  temporal reasoning  {pct('reasoning')} pick the only option that delivers in time)")
    print(f"  order stability     {pct('order')} keep their choice when the options swap places)")
    print(f"  label stability     {pct('label')} keep their choice when a/b becomes 1/2)")


# --- the k-option banks (18_spec_multichoice.md §3) ---------------------------

def _chosen(rec, answer):
    """The label letter the answer picks, None if unparsed."""
    index = parse_choice(answer, rec["labels"])
    return None if index is None else rec["labels"][index]


def _option(rec, label):
    return next(o for o in rec["options"] if o["label"] == label)


def _half_best(rec, half):
    return next(o for o in rec["options"] if o["half"] == half and o["rank_in_half"] == 1)["label"]


def _score(rec, label):
    """(exact, correct half, best option of the chosen half) for a pick."""
    half = _option(rec, label)["half"]
    return label == rec["truth_label"], half == rec["truth"], label == _half_best(rec, half)


def _half_logodds(rec, logits):
    near, far = (logsumexp([logits[rec["labels"].index(lab)] for lab in rec[key]])
                 for key in ("near_labels", "far_labels"))
    return near - far


def _truth_logodds(rec, logits):
    """The truth's logit minus logsumexp of the rest of its half."""
    half = rec["near_labels"] if rec["truth"] == "near" else rec["far_labels"]
    rest = [logits[rec["labels"].index(lab)] for lab in half if lab != rec["truth_label"]]
    return logits[rec["labels"].index(rec["truth_label"])] - logsumexp(rest)


def _baseline_pick(rec, rule):
    options = rec["options"]
    if rule == "largest_payout":
        return max(options, key=lambda o: o["payout"])["label"]
    return min(options, key=lambda o: (o["delay_years"], -o["payout"]))["label"]


def _uniform_scores(rec):
    """Expected (exact, correct half, best of the chosen half) under a uniform pick."""
    k = rec["n_options"]
    return 1 / k, 0.5, 2 / k


def gap_bins(rows, n_bins=3):
    """`rows` = [(gap_in_half, exact hit)] split into `n_bins` equal-count bins by gap."""
    rows = sorted(rows)
    cuts = np.linspace(0, len(rows), n_bins + 1).astype(int)
    return [{"gap_mean": float(np.mean([g for g, _ in part])), "exact": [int(sum(h for _, h in part)), len(part)]}
            for part in (rows[a:b] for a, b in zip(cuts, cuts[1:])) if part]


def multichoice_behavior(records, answers, label_logits=None):
    """Behavior of a k-option bank from the base answers and the label logits alone.
    `records`, `answers` and `label_logits` ([n, k], columns in each record's own `labels`
    order, or None) are aligned: pass the kept prompts only. Returns a dict with, for each
    regime and for `all`:
    - `n`: parsed answers; `accuracy`: [hits, n] for `exact` (the truth option), `half` (the
      truth's half) and `within_half` (the best option of the half chosen);
    - `half_logodds`, `truth_logodds`: mean half log-odds (logsumexp near labels minus far)
      and mean within-half log-odds (truth logit minus logsumexp of the rest of its half),
      over all rows with logits; `C` = near mean minus both mean of the half log-odds;
    - `pick_rate`: share of parsed answers per displayed position;
    - `gap_bins`: exact accuracy against `gap_in_half`, in three equal-count bins;
    - `consistency`: share of bases (two or more parsed renderings) whose renderings give
      the same choice, and the mean std of the half log-odds across a base's renderings;
    - `baselines`: mean exact, half and within-half accuracy of "largest payout",
      "earliest delay, then largest" and a uniform pick."""
    groups = {}
    for i, (rec, answer) in enumerate(zip(records, answers)):
        for name in (rec["regime"], "all"):
            groups.setdefault(name, []).append((i, rec, _chosen(rec, answer)))
    result = {}
    for name, rows in groups.items():
        parsed = [(i, rec, lab) for i, rec, lab in rows if lab is not None]
        scores = np.array([_score(rec, lab) for _, rec, lab in parsed], int).reshape(-1, 3)
        k = rows[0][1]["n_options"]
        picks = np.bincount([rec["labels"].index(lab) for _, rec, lab in parsed], minlength=k)
        out = {"n": len(parsed),
               "accuracy": {key: [int(scores[:, j].sum()), len(parsed)] for j, key in enumerate(("exact", "half", "within_half"))},
               "pick_rate": (picks / max(len(parsed), 1)).tolist(),
               "gap_bins": gap_bins([(rec["gap_in_half"], bool(s[0])) for (_, rec, _), s in zip(parsed, scores)]),
               "baselines": {
                   rule: dict(zip(("exact", "half", "within_half"), np.mean(
                       [_score(rec, _baseline_pick(rec, rule)) for _, rec, _ in rows], axis=0).tolist()))
                   for rule in ("largest_payout", "earliest_then_largest")}}
        out["baselines"]["uniform"] = dict(zip(("exact", "half", "within_half"),
                                               np.mean([_uniform_scores(rec) for _, rec, _ in rows], axis=0).tolist()))
        if label_logits is not None:
            half = [_half_logodds(rec, label_logits[i]) for i, rec, _ in rows]
            out["half_logodds"] = float(np.mean(half))
            out["truth_logodds"] = float(np.mean([_truth_logodds(rec, label_logits[i]) for i, rec, _ in rows]))
        out["consistency"] = _consistency(rows, label_logits)
        result[name] = out
    if label_logits is not None and {"near", "both"} <= set(result):
        result["C"] = result["near"]["half_logodds"] - result["both"]["half_logodds"]
    print_multichoice_behavior(result)
    return result


def _consistency(rows, label_logits):
    bases = {}
    for i, rec, lab in rows:
        bases.setdefault(rec["pair_id"], []).append((i, rec, lab))
    multi = [b for b in bases.values() if sum(lab is not None for _, _, lab in b) >= 2]
    same = [len({lab for _, _, lab in b if lab is not None}) == 1 for b in multi]
    spread = ([float(np.std([_half_logodds(rec, label_logits[i]) for i, rec, _ in b])) for b in bases.values() if len(b) >= 2]
              if label_logits is not None else [])
    return {"same_choice": [int(sum(same)), len(same)], "half_logodds_std": float(np.mean(spread)) if spread else None}


def print_multichoice_behavior(r):
    print("\nmultichoice behavior (base answers):")
    for name in ("near", "both", "all"):
        if name not in r:
            continue
        g = r[name]
        acc = "  ".join(f"{key} {hits}/{n}" for key, (hits, n) in g["accuracy"].items())
        print(f"  {name:<5} n={g['n']:<4} {acc}  same choice across renderings {g['consistency']['same_choice']}")
    if "C" in r:
        print(f"  half log-odds contrast C (near minus both) {r['C']:+.2f}")
