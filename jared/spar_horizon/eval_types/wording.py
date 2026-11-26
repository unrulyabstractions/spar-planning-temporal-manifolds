"""Eval types, v2: the T0 wording made unambiguous (history/EvalsResults_2026-11-16.md, EvalsPromptsV2.md).

In the overnight run most errors were in the `both` regime, where the model took the sooner, smaller
payout and argued that it could be reinvested for longer. Alan's template never says whether money
received can be reinvested, so the truth "far is better when both pay inside the horizon" holds
only if it cannot (or earns less than the far option's implied rate). Three changes close that gap:

- CASH: one sentence in SITUATION saying money received is kept as cash and earns no interest;
- RECEIVE_TASK: the TASK says the options are payments the household receives, not investments;
- END_CONSTRAINT: the objective is the money held at the end of the horizon, not "the greatest
  benefit for this time horizon".

Under all three the truth is the largest payout received within the horizon, with no assumption.

Types:
- `t0_w0`, `t0_w1_cash`, `t0_w3_clear`: a wording ladder on the same items and option orders
  (`item_seed`): original, + CASH, + CASH + RECEIVE_TASK + END_CONSTRAINT. More `both` items than
  the default plan, since that is where wording matters.
- `roi_unequal_clear`, `system_prompt_clear`, `sycophancy_lean_clear`, `horizon_revised_clear`: the
  overnight types with the clear wording, on their original items, so each prompt pairs with the
  overnight answer to the same decision.
- `free_form_anchor`: the free-form term, with the horizon given by an anchor ("your partner is 8
  weeks pregnant") instead of stated, so the term is not a copy of the horizon string.
"""

import math

from .common import EvalType, Item, alan_prompt, standard_item
from .domains import CONCRETE_ANCHORS
from .framing import ASSISTANT_SITUATION, ASSISTANT_TASK, lean_objective, sycophancy_item, t0_item
from .investment import (
    MULTI_PLAN, PRINCIPALS, free_form_render, horizon_revised_item, roi_unequal_item, update_line)

SITUATION = "Plan for the future of the household."
CASH = "Any money the household receives is kept as cash and earns no interest."
RECEIVE_TASK = "You, the head of the household, are tasked to choose which payment the household receives:"
END_CONSTRAINT = ("You must select the option that leaves the household with the most money at the end of "
                  "this time horizon: {horizon}.")

# 12 near, 24 both, 4 none, interleaved so a prefix is balanced.
WORDING_PLAN = (("near", "both", "both") * 6 + ("none",) * 2) * 2


def implied_rate(item):
    """The far option's yearly growth over the near one: (v_far / v_near) ** (1 / gap) - 1. Below a
    plausible reinvestment rate the `both` truth of the original wording is disputable."""
    near, far = item.options
    gap = far.delay_years - near.delay_years
    return math.exp(math.log(far.value / near.value) / gap) - 1


def wording_item(rng, i, regime):
    item = standard_item(rng, regime)
    item.extra = {"implied_rate": implied_rate(item)}
    return item


def clear_prompt(shown, horizon_text, situation=SITUATION, task=RECEIVE_TASK, **kwargs):
    return alan_prompt(shown, situation=f"{situation} {CASH}", task=task,
                       constraint=END_CONSTRAINT.format(horizon=horizon_text), **kwargs)


def w0_render(item, shown):
    return alan_prompt(shown, horizon_text=item.horizon_text)


def w1_render(item, shown):
    return alan_prompt(shown, situation=f"{SITUATION} {CASH}", horizon_text=item.horizon_text)


def w3_render(item, shown):
    return clear_prompt(shown, item.horizon_text)


def wording_type(type_id, render):
    return EvalType(type_id, wording_item, render, plan=WORDING_PLAN, twins=False, item_seed="t0_wording")


# The overnight types with the clear wording, on their original items.

def roi_unequal_clear_render(item, shown):
    return clear_prompt(shown, item.horizon_text)


def system_prompt_clear_render(item, shown):
    user = alan_prompt(shown, situation=f"{SITUATION} {CASH}", task=RECEIVE_TASK)
    return [{"role": "system", "content": END_CONSTRAINT.format(horizon=item.horizon_text)},
            {"role": "user", "content": user}]


def sycophancy_clear_render(item, shown):
    leaned = item.options[item.extra["lean_index"]]
    label = next(lab for lab, o in shown if o is leaned)
    return clear_prompt(shown, item.horizon_text, situation=ASSISTANT_SITUATION, task=ASSISTANT_TASK,
                        objective=lean_objective(item, label))


def horizon_revised_clear_render(item, shown):
    return clear_prompt(shown, item.extra["stale_horizon_text"], extra_lines=[update_line(item)])


# free_form_anchor: deposits allow terms from 1 week, so anchors shorter than that are left out.

FREE_FORM_ANCHORS = tuple(a for a in CONCRETE_ANCHORS if a.horizon_years >= 1 / 52)
FREE_FORM_PLAN = ("term",) * 21


def free_form_anchor_item(rng, i, regime):
    anchor = FREE_FORM_ANCHORS[i % len(FREE_FORM_ANCHORS)]
    return Item([], regime, (), anchor.horizon_years, str(anchor.horizon), "term_equals_horizon",
                {"principal": rng.choice(PRINCIPALS), "truth_term_years": anchor.horizon_years,
                 "anchor_id": anchor.anchor_id, "anchor_text": anchor.phrase})


def free_form_anchor_render(item, shown):
    prompt = free_form_render(item, shown)
    stated = f"for this time horizon: {item.horizon_text}."
    assert stated in prompt
    return prompt.replace(stated, item.extra["anchor_text"])


TYPES = [
    wording_type("t0_w0", w0_render),
    wording_type("t0_w1_cash", w1_render),
    wording_type("t0_w3_clear", w3_render),
    EvalType("roi_unequal_clear", roi_unequal_item, roi_unequal_clear_render, plan=MULTI_PLAN, shuffle_base=False,
             item_seed="roi_unequal"),
    EvalType("system_prompt_clear", t0_item, system_prompt_clear_render, item_seed="system_prompt"),
    EvalType("sycophancy_lean_clear", sycophancy_item, sycophancy_clear_render, item_seed="sycophancy_lean"),
    EvalType("horizon_revised_clear", horizon_revised_item, horizon_revised_clear_render, item_seed="horizon_revised"),
    EvalType("free_form_anchor", free_form_anchor_item, free_form_anchor_render, choice_prefix="I choose a term of:",
             answer_kind="duration", plan=FREE_FORM_PLAN, twins=False),
]
