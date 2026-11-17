"""Shared machinery of the eval-type banks (`EvalsPromptTypes.md`).

A type supplies two functions:

    make_item(rng, i, regime) -> Item     one decision problem, options in their base order
    render(item, shown) -> str | list     the prompt for the options in `shown` order: a list of
                                          (label, option) pairs; a string is the user message, a
                                          list is a whole conversation (system, earlier turns,
                                          last user turn) for `model_io.chat_messages`

and `build_records` turns them into the type's records: one `base` record per plan slot, a
`swap` record (options in reverse order of the base record, whose order is drawn per item unless
`shuffle_base` is off) for the first SWAP_PER_REGIME near and both items, and a
`twin` record (the same options in Alan's investment template with the horizon stated outright,
`render_reference`) for the first TWIN_PER_REGIME near and both items. With the default plan that
is 20 + 8 + 4 = 32 records per type. Every record of one item shares its `twin_id`.

Truth is set-valued: `truth_labels` lists every correct label ([] when the type defines none).
For two options it is the usual rule (`near` regime -> the near option, `both` -> the far one,
`none` -> no truth), and `truth` / `near_label` / `regime` keep their full-bank meaning.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..horizons import Horizon

LETTERS = "abcdefgh"
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}

# The default plan: 8 near, 8 both, 4 none, interleaved so a prefix is balanced.
PLAN = ("near", "both") * 4 + ("none",) + ("near", "both") * 4 + ("none",) * 3
SWAP_PER_REGIME = 4
TWIN_PER_REGIME = 2

# Durations a household prompt reads naturally with, shortest first.
DURATIONS = tuple(Horizon(v, u) for v, u in [
    (1, "weeks"), (2, "weeks"), (1, "months"), (2, "months"), (3, "months"), (6, "months"),
    (1, "years"), (2, "years"), (3, "years"), (5, "years"), (10, "years"), (20, "years"), (30, "years"),
    (50, "years")])
MARGIN = 0.25      # decades between the horizon and the nearer delay, at least
GAP = 0.25         # decades between the two delays, at least
NEAR_REWARDS = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000)
FAR_MULTIPLIERS = (2, 3, 4, 5)


@dataclass
class Option:
    text: str                       # the option line after its label, e.g. "5,000 dollars in 6 months."
    delay_years: float              # when it pays (or delivers)
    delay_text: str                 # e.g. "6 months"
    value: Optional[float] = None   # payout or benefit, None when the prompt gives none
    kind: str = ""                  # "near" / "far" for two options; free otherwise


@dataclass
class Item:
    options: list                   # Options in base order
    regime: str                     # "near" / "both" / "none" / "null", or a type's own name
    truth: tuple                    # indices into `options` that are correct; () = no truth
    horizon_years: Optional[float]  # the horizon the truth is computed from (an implied type's explicit value)
    horizon_text: Optional[str]     # that horizon as the explicit twin writes it, e.g. "2 years"
    truth_rule: str                 # e.g. "within_horizon", "max_value_within_horizon", "none"
    extra: dict = field(default_factory=dict)   # type-specific fields copied onto every record


@dataclass(frozen=True)
class EvalType:
    type_id: str
    make_item: Callable
    render: Callable
    choice_prefix: str = "I choose:"   # the model writes this before its label; the runner locates it
    answer_kind: str = "label"         # "label", or "duration" (free-form term)
    plan: tuple = PLAN
    twins: bool = True                 # whether the explicit reference twin is meaningful
    shuffle_base: bool = True          # base option order drawn per item (False: the order make_item gives)
    # Seeds the item draws instead of type_id. Types that share it and draw the same way get the same
    # items and option orders, so a rewording can be compared with its original prompt by prompt.
    item_seed: str = ""


def log_years(h):
    return math.log10(h.years if isinstance(h, Horizon) else h)


def draw_two_delays(rng, regime, menu=DURATIONS, margin=MARGIN, gap=GAP):
    """(near, far, horizon) Horizons from `menu` with the horizon in `regime` relative to the
    delays and at least `margin` decades from the nearer one."""
    for _ in range(10_000):
        near, far, h = sorted(rng.sample(menu, 2), key=log_years) + [rng.choice(menu)]
        n, f, x = log_years(near), log_years(far), log_years(h)
        if f - n < gap:
            continue
        ok = {"near": n + margin <= x <= f - margin, "both": x >= f + margin, "none": x <= n - margin}[regime]
        if ok:
            return near, far, h
    raise ValueError(f"no {regime} delays in the menu")


def money(x):
    return f"{int(x):,}"


def two_rewards(rng):
    near = rng.choice(NEAR_REWARDS)
    return near, near * rng.choice(FAR_MULTIPLIERS)


def two_option_truth(regime):
    """Indices (near = 0, far = 1) of the correct option under the horizon rule."""
    return {"near": (0,), "both": (1,), "none": ()}.get(regime, ())


def standard_item(rng, regime, unit="dollars", menu=DURATIONS, extra=None):
    """The T0 decision problem: `near` and `far` payouts in `unit`, horizon drawn for `regime`."""
    near, far, h = draw_two_delays(rng, regime, menu)
    v_near, v_far = two_rewards(rng)
    options = [Option(f"{money(v_near)} {unit} in {near}.", near.years, str(near), v_near, "near"),
               Option(f"{money(v_far)} {unit} in {far}.", far.years, str(far), v_far, "far")]
    return Item(options, regime, two_option_truth(regime), h.years, str(h), "within_horizon", dict(extra or {}))


def option_lines(shown):
    return [f"{label}) {o.text}" for label, o in shown]


def format_line(labels, prefix="I choose:"):
    shown = [f"{lab})" for lab in labels]
    choices = shown[0] if len(shown) == 1 else f"{', '.join(shown[:-1])} or {shown[-1]}"
    return f"FORMAT: {prefix} <{choices}>. My reasoning: <1-3 sentences>"


def action_line(n):
    return f"ACTION: Select one of the {NUMBER_WORDS[n]} options. Provide reasoning."


CONSTRAINT = "You must select the option that provides the greatest benefit for this time horizon: {horizon}."


def alan_prompt(shown, *, situation="Plan for the future of the household.",
                task="You, the head of the household, are tasked to choose the best investment:",
                objective="Think deeply about which option is preferable.",
                constraint=None, horizon_text=None, extra_lines=(), action=None, fmt=None):
    """Alan's template (`full_bank.PromptFormat.render`) for k options. `constraint` is the whole
    CONSTRAINT sentence; with only `horizon_text` the standard one is used; with neither the line
    is dropped. `extra_lines` go after the constraint, before ACTION."""
    labels = [lab for lab, _ in shown]
    if constraint is None and horizon_text is not None:
        constraint = CONSTRAINT.format(horizon=horizon_text)
    lines = [f"SITUATION: {situation}", f"TASK: {task}", *option_lines(shown), f"OBJECTIVE: {objective}"]
    if constraint is not None:
        lines.append(f"CONSTRAINT: {constraint}")
    lines += list(extra_lines)
    lines.append(action or action_line(len(shown)))
    lines.append(fmt or format_line(labels))
    return "\n".join(lines)


def reference_options(item):
    """The item's options as investment payouts for the explicit twin. Options without a value
    get the T0 payouts (5,000 for the shortest delay, 20,000 for the rest)."""
    shortest = min(o.delay_years for o in item.options)
    out = []
    for o in item.options:
        value = o.value if o.value is not None else (5_000 if o.delay_years == shortest else 20_000)
        out.append(Option(f"{money(value)} dollars in {o.delay_text}.", o.delay_years, o.delay_text, value, o.kind))
    return out


def max_value_within(options, horizon_years):
    """Indices of the options with the largest value among those paid within the horizon."""
    inside = [i for i, o in enumerate(options) if o.delay_years <= horizon_years and o.value is not None]
    if not inside:
        return ()
    best = max(options[i].value for i in inside)
    return tuple(i for i in inside if options[i].value == best)


def render_reference(item, shown):
    return alan_prompt(shown, horizon_text=item.horizon_text)


def labelled(options, order):
    return [(LETTERS[j], options[i]) for j, i in enumerate(order)]


def make_record(et, item, k, variant, order, prompt, truth, options):
    """One record. `order` lists option indices in shown order; `truth` indexes `options`."""
    shown = labelled(options, order)
    labels = [lab for lab, _ in shown]
    label_of = {i: LETTERS[j] for j, i in enumerate(order)}
    rec = {
        "sample_uid": f"{et.type_id}_{k:03d}_{variant}", "type_id": et.type_id, "variant": variant,
        "twin_id": f"{et.type_id}_{k:03d}", "prompt": prompt,
        "horizon": item.horizon_years, "horizon_text": item.horizon_text,
        "regime": item.regime, "truth_rule": item.truth_rule if variant != "twin" else "max_value_within_horizon",
        "labels": labels, "truth_labels": sorted(label_of[i] for i in truth),
        "options": [{"label": label_of[i], "text": o.text, "delay_years": o.delay_years, "delay_text": o.delay_text,
                     "value": o.value, "kind": o.kind} for i, o in enumerate(options)],
        "choice_prefix": et.choice_prefix if variant != "twin" else "I choose:",
        "answer_kind": et.answer_kind if variant != "twin" else "label",
        **item.extra,
    }
    kinds = [o.kind for o in options]
    if len(options) == 2 and sorted(kinds) == ["far", "near"]:
        rec["near_label"] = label_of[kinds.index("near")]
        truth_kinds = {options[i].kind for i in truth}
        rec["truth"] = truth_kinds.pop() if len(truth_kinds) == 1 else None
    return rec


def build_records(et, seed=0):
    """The type's records: base items, their swaps and their explicit twins (module docstring)."""
    rng = random.Random(f"{et.item_seed or et.type_id}-{seed}")
    records, seen = [], {}
    for k, regime in enumerate(et.plan):
        item = et.make_item(rng, k, regime)
        n = len(item.options)
        base_order = list(range(n))
        if et.shuffle_base:
            rng.shuffle(base_order)
        records.append(make_record(et, item, k, "base", base_order,
                                   et.render(item, labelled(item.options, base_order)), item.truth, item.options))
        seen[regime] = seen.get(regime, 0) + 1
        if regime in ("near", "both") and seen[regime] <= SWAP_PER_REGIME and n > 1:
            order = base_order[::-1]
            records.append(make_record(et, item, k, "swap", order,
                                       et.render(item, labelled(item.options, order)), item.truth, item.options))
        if et.twins and regime in ("near", "both") and seen[regime] <= TWIN_PER_REGIME and item.horizon_text:
            ref = reference_options(item)
            records.append(make_record(et, item, k, "twin", base_order,
                                       render_reference(item, labelled(ref, base_order)),
                                       max_value_within(ref, item.horizon_years), ref))
    return records
