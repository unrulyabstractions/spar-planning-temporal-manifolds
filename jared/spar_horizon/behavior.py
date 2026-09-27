"""Behavior: choice parsing and the three coherence tests from the paper."""

from .model_io import generate_batch
from .prompts import DELAY_YEARS, choice_prompt


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
    return DELAY_YEARS[rec["delays"][0]] <= h < DELAY_YEARS[rec["delays"][1]]


def default_prompt(rec, **kw):
    return choice_prompt(rec["rewards"], rec["delays"], rec["horizon"], **kw)


def behavior(tokenizer, model, device, records, base_answers, cfg, prompt_fn=default_prompt,
             system=None, log_every=20):
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
            reasoning_hits += base == 0
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
