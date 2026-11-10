"""Behavior: choice parsing, what the base answers alone show, and the three
coherence tests from the paper (which regenerate swapped and relabeled
prompts)."""

from statistics import mean

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
