"""Eval types: boundary group (EvalsBoundarySpec.md section 3). See common.py for the contract.

Templated types (R2 `mt_*`, R4 `no_horizon`) are T0 decisions in a conversation or without a horizon.
`implied_event` (R4 N1) wraps the E18 anchors. The LLM-written types (R3 `llm_*`, R4 `persona_*`) are
filled from a scaffold file (`load_boundary_types`). Every type here has a plan of 48 base items and a
`cue`: the exact text whose last token the runner captures (None when there is no cue)."""

import copy
import json
import math
import random
import re
import warnings
from collections import Counter
from pathlib import Path

from ..full_bank import natural_horizon, regime as regime_of
from ..implied_bank import ANCHORS, draw_implied_delays, feasible
from .common import (CONSTRAINT, DURATIONS, FAR_MULTIPLIERS, NEAR_REWARDS, EvalType, alan_prompt, format_line,
                     log_years, option_lines, standard_item)
from .domains import delay_pairs, valued_item
from .framing import evaluation_turn

REGIMES = ("near", "both", "none")
BOUNDARY_PLAN = REGIMES * 16
DEFAULT_SCAFFOLDS = Path(__file__).parent / "data" / "boundary_scaffolds.json"


def explicit_cue(horizon_text):
    return f"for this time horizon: {horizon_text}"


def item_rng(tag, item):
    """A generator seeded by the item's content, so extra draws do not shift the shared item stream."""
    return random.Random(f"{tag}-{item.horizon_text}-{item.options[0].text}-{item.options[1].text}")


def last_user_with_format(messages, labels, constraint=None):
    messages[-1]["content"] += (f"\n{constraint}" if constraint else "") + f"\n{format_line(labels)}"
    return messages


# ---- multi-turn (R2) ----

OPENING = "SITUATION: Plan for the future of the household. You, the head of the household, will choose between two investments."
EVALUATE = "First evaluate each option in the format COST / BENEFIT / TIMING."


def constraint_sentence(horizon_text):
    return f"CONSTRAINT: {CONSTRAINT.format(horizon=horizon_text)} "


def three_turns(shown, first_constraint="", last_constraint=""):
    """Evaluate option a (turn 1), option b (turn 2), then choose (turn 3). Constraints are the
    sentences (with trailing space) placed in turn 1 and in turn 3."""
    (first_label, first), (second_label, second) = shown
    return [{"role": "user", "content": f"{OPENING} {first_constraint}{EVALUATE}\nOption {first_label}): {first.text}"},
            {"role": "assistant", "content": evaluation_turn(first)},
            {"role": "user", "content": f"Option {second_label}): {second.text}"},
            {"role": "assistant", "content": evaluation_turn(second)},
            {"role": "user", "content": f"{last_constraint}Using your evaluations, choose one option. "
                                        f"{format_line([first_label, second_label])}"}]


def mt_first_render(item, shown):
    return three_turns(shown, first_constraint=constraint_sentence(item.horizon_text))


def mt_last_render(item, shown):
    return three_turns(shown, last_constraint=constraint_sentence(item.horizon_text))


def mt_revised_item(rng, i, regime):
    """The old horizon is at least a decade from the new one and, when the delays allow, in another regime."""
    item = standard_item(rng, regime)
    near, far = (o.delay_years for o in item.options)
    apart = [h for h in DURATIONS if abs(log_years(h) - math.log10(item.horizon_years)) >= 1]
    other = [h for h in apart if regime_of(h.years, near, far) != regime]
    old = item_rng("mt_revised", item).choice(other or apart)
    item.extra = {"old_horizon_years": old.years, "old_horizon_text": str(old)}
    return item


def mt_revised_render(item, shown):
    changed = f"Actually, my horizon has changed: {constraint_sentence(item.horizon_text)}"
    return three_turns(shown, first_constraint=constraint_sentence(item.extra["old_horizon_text"]),
                       last_constraint=changed)


def mt_assistant_render(item, shown):
    (first_label, first), (second_label, second) = shown
    return [{"role": "user", "content": f"{OPENING} Before we start, please restate my goal as you understand it."},
            {"role": "assistant", "content": "Your goal is the investment with the greatest benefit, and "
                                             f"everything has to pay out within {item.horizon_text}."},
            {"role": "user", "content": "Thanks. Here are the options:\n"
                                        f"Option {first_label}): {first.text}\nOption {second_label}): {second.text}\n"
                                        "Evaluate each one in the format COST / BENEFIT / TIMING."},
            {"role": "assistant", "content": f"Option {first_label}): {evaluation_turn(first)}\n"
                                             f"Option {second_label}): {evaluation_turn(second)}"},
            {"role": "user", "content": f"Using your evaluations and my goal, choose one option. "
                                        f"{format_line([first_label, second_label])}"}]


# Unrelated household chat for `mt_long`: no durations, no money, no choices between investments.
FILLER = (
    ("My kettle has a white crust inside. How do I get rid of it?",
     "That crust is limescale from hard water. Fill the kettle halfway with equal parts water and white vinegar, "
     "bring it to a boil, then switch it off and let it sit until the liquid cools. Pour it out, rinse the kettle "
     "a couple of times, and boil plain water once to clear any vinegar smell. If a stubborn layer remains, scrub "
     "gently with a soft brush around the heating plate. Avoid steel wool, because it scratches the inside and gives "
     "the scale something to grip next time."
     " Once the kettle is clean it also heats faster and uses less power, so the effort pays back quietly. If your water is very hard, a filter jug for the kettle slows the crust from coming back."),
    ("The pantry is a mess. Where should I start?",
     "Empty one shelf at a time instead of the whole pantry, so you are never left with a floor covered in boxes. "
     "Wipe the shelf, then sort what came off it into three piles: keep, use up soon, and throw away. Check the "
     "printed dates on tins and packets as you go. Put the things you reach for most at eye level, heavy items low, "
     "and light snacks up high. Clear containers with labels make it easy to see what is running out."
     " When the shelves are done, stand back and look at the whole cupboard. Move anything you keep tripping over to a more convenient spot, and leave a little free space so that new shopping has somewhere to go."),
    ("I cannot decide between two shades of green for the hallway.",
     "Paint the colors on large sheets of card instead of small swatches, and tape them to different walls. Look at "
     "them in bright light and in lamp light, because greens shift a lot: a sage can turn grey in a dim corridor, "
     "and a bright leaf green can feel loud in a narrow space. Hallways usually lack windows, so the lighter, "
     "warmer shade tends to look better. Live with the samples for a couple of evenings before you commit."
     " If you still cannot choose, ask whoever uses the hallway most which one feels calmer. A satin finish wipes clean more easily than matt, which matters where coats and bags rub against the wall all the time."),
    ("Some of my houseplants have yellow leaves. What is going wrong?",
     "Yellow leaves most often mean too much water, so check the soil with a finger before you pour anything. If it "
     "is damp a knuckle deep, wait. Make sure the pot has drainage holes and the saucer is not holding a puddle. "
     "Other causes are too little light, a cold draft from a window, or a plant that has outgrown its pot and "
     "needs fresh soil. Remove the yellow leaves with clean scissors so the plant can spend its energy on healthy "
     "growth."
     " Rotate the pot a quarter turn now and then so every side gets light, and wipe dust off the leaves with a damp cloth, since a dusty leaf takes in less light and the plant slowly looks tired."),
    ("What is the easiest way to clean the grout between my bathroom tiles?",
     "Mix baking soda with a little water into a thick paste, spread it along the grout lines, and let it sit while "
     "you do something else. Then scrub with an old toothbrush, working along each line, and wipe the tiles with a "
     "damp cloth. For dark stains, spray a mix of water and a splash of hydrogen peroxide, wait until it foams, and "
     "scrub again. Open a window while you work, and never mix cleaning products that contain bleach with anything "
     "acidic."
     " Finish by sealing the grout once it is dry, using a small bottle of grout sealer and a brush. The sealer keeps water out, so stains and mildew have a much harder time getting a hold on the surface."),
    ("We keep forgetting things at the shop. Any tips for a shared grocery list?",
     "Use a shared note on everyone's phone, and add items the moment you notice they are low, such as when you "
     "finish the last of the coffee. Group the list by store section, so you walk the aisles once instead of "
     "doubling back. A short list of staples you always keep in stock helps too, since it is easy to run through "
     "before you leave. If two people shop, agree that whoever buys something ticks it off straight away, which "
     "avoids buying the same thing twice."
     " It also helps to keep a note of the brands you prefer, since the household will often argue about which pasta or soap to buy. A shared list ends those debates before anyone stands in the aisle."),
    ("A screw in my cupboard hinge is stripped and will not turn. Help?",
     "Try pressing a wide rubber band flat between the screwdriver and the screw head, then turn slowly while pushing "
     "down hard; the rubber fills the damaged grooves. If that fails, a screw extractor bit, sold in small sets at "
     "hardware stores, bites into the head and turns it backwards. For a screw that is only loose in its hole, "
     "push a few wooden toothpicks with a dab of glue into the hole, trim them flush, and drive the screw back in."
     " Afterwards, put a drop of oil on the other hinge screws and tighten them gently, because a cupboard door that sags puts extra strain on the hinge and can pull the screws loose again."),
    ("What should I look for in a vacuum cleaner for a home with a dog?",
     "Look for strong suction with a motorized brush head made for pet hair, because ordinary bristles let hair wrap "
     "around the roller. A bin you can empty without touching the dust is a real convenience, and a sealed filter "
     "keeps dander from blowing back out. Check how easy it is to cut hair free from the brush roll. Cordless models "
     "are handy for quick jobs, but a corded one usually has more power for deep cleaning of carpets and the "
     "sofa."
     " Before you buy, push the vacuum across a bare floor and a rug in the shop if they let you. Weight matters more than it seems, because a heavy machine is one you will quietly stop using."),
    ("The door to the bedroom squeaks every time someone opens it.",
     "A squeak almost always comes from the hinge pins. Slide a thin screwdriver or nail under the head of each pin "
     "and tap it upward until you can pull it out, supporting the door as you work. Wipe off the old dirt, coat the "
     "pin lightly with petroleum jelly or machine oil, and put it back. Open and close the door a few times to "
     "spread the lubricant, and wipe away any drips so they do not mark the paint or the floor."
     " Test the door afterwards by swinging it slowly and listening. If the squeak moves to the middle of the swing, the other hinge is the culprit and needs exactly the same treatment, so repeat it there."),
    ("I want to start composting kitchen scraps. Where do I begin?",
     "Choose a spot with some shade and good drainage, and use a bin with a lid so animals stay out. Aim for a "
     "balance of wet kitchen scraps, such as peelings and coffee grounds, and dry material such as torn cardboard, "
     "dry leaves or shredded paper. Chop big pieces small, and give the pile a turn with a fork now and then to "
     "let air in. Keep out meat, dairy and oily food. The pile should feel like a wrung-out sponge, never soggy and "
     "never dusty."
     " When the compost is dark, crumbly and smells like forest soil, it is ready. Spread it around plants or mix it into pots, and keep a small caddy by the sink to carry scraps out neatly."),
    ("Fitted sheets are impossible to fold. Is there a trick?",
     "Put your hands into two adjacent corners and fold one corner over the other so they sit together, then do the "
     "same with the other two corners, tucking all four into one neat pocket. Lay the sheet on a flat surface and "
     "smooth it so the straight edges line up, fold it into thirds one way, and then into thirds the other way. "
     "It will not be perfect at first, but the shape comes out as a flat rectangle that stacks well in the "
     "cupboard."
     " Store the folded sheets inside their own matching pillowcase, so that sets stay together and the cupboard stays tidy. Nobody then has to search the shelf for the one sheet that fits."),
    ("Which doormat is best for a front door that gets a lot of rain?",
     "Pick a mat with a rubber backing so it grips the step, and a surface that traps water and mud rather than "
     "spreading it. Coir looks nice but soaks through; a synthetic mat with stiff bristles scrapes boots well and "
     "dries faster. Choose a size slightly wider than the door so people can step with both feet. Shake it out "
     "often, and hose it down when it gets clogged with grit, then let it dry upright against a wall."
     " Finally, slip a thin rubber scraper or an old boot brush beside the mat, so people can clean the worst mud off before stepping on it. That keeps the mat and the hallway floor cleaner for longer."),
)


def mt_long_item(rng, i, regime):
    item = standard_item(rng, regime)
    item.extra = {"filler_ids": sorted(item_rng("mt_long", item).sample(range(len(FILLER)), 8))}
    return item


def mt_long_render(item, shown):
    messages = [{"role": "user", "content": f"{OPENING} {constraint_sentence(item.horizon_text)}"
                                            "First, though, a few quick household questions."},
                {"role": "assistant", "content": "Sure, go ahead and ask."}]
    for i in item.extra["filler_ids"]:
        question, answer = FILLER[i]
        messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    labels = [lab for lab, _ in shown]
    messages.append({"role": "user", "content": "Thanks, that helps. Back to the investments. Here are the options:\n"
                                                + "\n".join(option_lines(shown))
                                                + f"\nChoose one option. {format_line(labels)}"})
    return messages


def explicit_cue_of(item, variant):
    return explicit_cue(item.horizon_text)


def mt_assistant_cue(item, variant):
    return explicit_cue_of(item, variant) if variant == "twin" else f"has to pay out within {item.horizon_text}"


# ---- no horizon (R4 N0) ----

def no_horizon_item(rng, i, regime):
    item = standard_item(rng, regime)
    item.regime, item.truth, item.truth_rule = "null", (), "none"
    item.horizon_years = item.horizon_text = None
    return item


def no_horizon_render(item, shown):
    return alan_prompt(shown)


# ---- implied events (R4 N1) ----

IMPLIED_ANCHORS = tuple(a for a in ANCHORS if a.family != "D")


def implied_event_item(rng, i, regime):
    """Anchors from the E18 bank (families A, B, C) spread over the regime's items; an anchor with no
    delays that clear its whole interval in this regime is skipped."""
    per_regime = BOUNDARY_PLAN.count(regime)
    start = BOUNDARY_PLAN[:i].count(regime) * len(IMPLIED_ANCHORS) // per_regime
    for step in range(len(IMPLIED_ANCHORS)):
        anchor = IMPLIED_ANCHORS[(start + step) % len(IMPLIED_ANCHORS)]
        lo_log, hi_log = math.log10(anchor.lo), math.log10(anchor.hi)
        if not feasible(lo_log, hi_log, regime):
            continue
        try:
            near, far = draw_implied_delays(rng, lo_log, hi_log, regime)
        except ValueError:
            continue
        mid = 10 ** anchor.log_mid
        explicit = natural_horizon(mid)
        item = valued_item(rng, regime, "dollars", NEAR_REWARDS, FAR_MULTIPLIERS, (near, far, explicit))
        item.horizon_years = mid
        item.extra = {"anchor_id": anchor.anchor_id, "family": anchor.family, "anchor_text": anchor.phrase,
                      "anchor_lo_years": anchor.lo, "anchor_hi_years": anchor.hi, "context": anchor.context,
                      "interval_years": [anchor.lo, anchor.hi], "us_convention": anchor.us_convention}
        return item
    raise ValueError(f"no anchor is feasible in the {regime} regime")


def implied_situation(item):
    return " ".join(filter(None, ["Plan for the future of the household.", item.extra["context"]]))


def implied_event_render(item, shown):
    return alan_prompt(shown, situation=implied_situation(item),
                       constraint=CONSTRAINT.format(horizon=item.extra["anchor_text"]))


def implied_event_twin(item, shown):
    return alan_prompt(shown, situation=implied_situation(item), horizon_text=item.horizon_text)


def implied_event_cue(item, variant):
    return explicit_cue_of(item, variant) if variant == "twin" else item.extra["anchor_text"]


# ---- scaffold file (R3 llm_*, R4 persona_*) ----

LLM_TYPES = ("llm_prose", "llm_competing", "llm_document")
PERSONA_TYPES = ("persona_user", "persona_model")
HORIZONS_PER_SCAFFOLD = 6
INTERVAL_MARGIN = 0.3          # decades between a persona interval and the nearer delay

# A number followed by a unit ("30-year", "6 months"), or a time word on its own ("week", "daily").
_UNIT = r"(?:seconds?|minutes?|hours?|days?|weeks?|fortnights?|months?|years?|decades?|centur(?:y|ies))"
_TIME_WORD = r"(?:minutes?|hours?|days?|weeks?|fortnights?|months?|years?|decades?|centur(?:y|ies)|daily|weekly|monthly|yearly|hourly|annual(?:ly)?)"
DURATION_TEXT = re.compile(rf"\d+(?:[.,]\d+)?\s*-?\s*{_UNIT}\b|\b{_TIME_WORD}\b", re.IGNORECASE)
SAMPLE_HORIZON = "3 months"
SAMPLE_OPTIONS = "a) 5,000 dollars in 6 months.\nb) 20,000 dollars in 3 years."


def reject(scaffold, problem):
    raise ValueError(f"scaffold {scaffold.get('scaffold_id')!r}: {problem}")


def fill_messages(scaffold, option_text, horizon_text=None):
    """Copy of the scaffold's messages with the placeholders filled (`{horizon}` only for llm_* types)."""
    messages = copy.deepcopy(scaffold["messages"])
    for message in messages:
        if horizon_text is not None:
            message["content"] = message["content"].replace("{horizon}", horizon_text)
    messages[-1]["content"] = messages[-1]["content"].replace("{options}", option_text)
    return messages


def validate_scaffold(scaffold):
    type_id = scaffold.get("type_id")
    if type_id not in LLM_TYPES + PERSONA_TYPES:
        reject(scaffold, f"unknown type_id {type_id!r}")
    messages = scaffold.get("messages")
    if not (isinstance(messages, list) and messages and all(
            isinstance(m, dict) and m.get("role") in ("system", "user", "assistant") and isinstance(m.get("content"), str)
            for m in messages)):
        reject(scaffold, "messages must be a list of {role, content} with role system/user/assistant")
    if messages[-1]["role"] != "user":
        reject(scaffold, "the last message must be the user's")
    texts = [m["content"] for m in messages]
    if sum(t.count("{options}") for t in texts) != 1 or texts[-1].count("{options}") != 1:
        reject(scaffold, "{options} must occur exactly once, in the last user message")
    llm = type_id in LLM_TYPES
    horizon_count = sum(t.count("{horizon}") for t in texts)
    if horizon_count != (1 if llm else 0):
        reject(scaffold, f"{{horizon}} occurs {horizon_count} times, expected {1 if llm else 0}")
    cue = scaffold.get("cue")
    if not (isinstance(cue, str) and cue) or "{options}" in cue or ("{horizon}" in cue) != llm:
        reject(scaffold, "cue must be a non-empty string" + (" containing {horizon}" if llm else " without {horizon}"))
    filled_cue = cue.replace("{horizon}", SAMPLE_HORIZON)
    filled = fill_messages(scaffold, SAMPLE_OPTIONS, SAMPLE_HORIZON if llm else None)
    occurrences = sum(m["content"].count(filled_cue) for m in filled)
    if occurrences != 1:
        reject(scaffold, f"cue occurs {occurrences} times after filling, expected once")
    if not llm:
        match = DURATION_TEXT.search("\n".join(texts))
        if match:
            reject(scaffold, f"persona text contains a duration: {match.group(0)!r}")
        interval = scaffold.get("interval_years")
        if not (isinstance(interval, (list, tuple)) and len(interval) == 2
                and all(isinstance(x, (int, float)) and x > 0 for x in interval) and interval[0] < interval[1]):
            reject(scaffold, "interval_years must be [lo, hi] with 0 < lo < hi")


def scaffold_prompt(scaffold, shown, horizon_text, constraint=None):
    messages = fill_messages(scaffold, "\n".join(option_lines(shown)), horizon_text)
    return last_user_with_format(messages, [lab for lab, _ in shown], constraint)


def horizon_candidates(regime):
    """Horizons from DURATIONS that have delays in `regime` (with the usual margin)."""
    return [h for h in DURATIONS if delay_pairs(regime, h.years, h.years, DURATIONS)]


def llm_type(type_id, scaffolds):
    """6 horizons per scaffold (two per regime, all different), filled into the scaffold's `{horizon}`."""
    n = len(scaffolds)
    by_id = {s["scaffold_id"]: s for s in scaffolds}
    plan = tuple(REGIMES[(s + j) % 3] for j in range(HORIZONS_PER_SCAFFOLD) for s in range(n))

    def make_item(rng, i, regime):
        s, j = i % n, i // n
        earlier = sum(plan[jj * n + s] == regime for jj in range(j))
        candidates = horizon_candidates(regime)
        h = candidates[(s * 3 + earlier * (len(candidates) // 2 + 1)) % len(candidates)]
        near, far = rng.choice(delay_pairs(regime, h.years, h.years, DURATIONS))
        item = valued_item(rng, regime, "dollars", NEAR_REWARDS, FAR_MULTIPLIERS, (near, far, h))
        scaffold = scaffolds[s]
        item.extra = {"scaffold_id": scaffold["scaffold_id"],
                      "scaffold_cue": scaffold["cue"].replace("{horizon}", item.horizon_text)}
        return item

    def render(item, shown):
        return scaffold_prompt(by_id[item.extra["scaffold_id"]], shown, item.horizon_text)

    def cue(item, variant):
        return explicit_cue_of(item, variant) if variant == "twin" else item.extra["scaffold_cue"]

    return EvalType(type_id, make_item, render, plan=plan, cue=cue)


def persona_type(type_id, scaffolds):
    """One scaffold = one base item; the horizon is its interval. A scaffold takes the feasible regime
    with the fewest items so far; one with no feasible regime is skipped (with a warning)."""
    kept, plan = [], []
    counts = Counter()
    for scaffold in scaffolds:
        lo, hi = scaffold["interval_years"]
        feasible_regimes = [r for r in REGIMES if delay_pairs(r, lo, hi, DURATIONS, INTERVAL_MARGIN)]
        if not feasible_regimes:
            warnings.warn(f"{type_id}: scaffold {scaffold['scaffold_id']!r} skipped, no regime is feasible "
                          f"for the interval {lo}-{hi} years")
            continue
        regime = min(feasible_regimes, key=counts.__getitem__)
        counts[regime] += 1
        kept.append(scaffold)
        plan.append(regime)

    def make_item(rng, i, regime):
        scaffold = kept[i]
        lo, hi = scaffold["interval_years"]
        near, far = rng.choice(delay_pairs(regime, lo, hi, DURATIONS, INTERVAL_MARGIN))
        mid = math.sqrt(lo * hi)
        item = valued_item(rng, regime, "dollars", NEAR_REWARDS, FAR_MULTIPLIERS, (near, far, natural_horizon(mid)))
        item.horizon_years = mid
        item.extra = {"scaffold_id": scaffold["scaffold_id"], "interval_years": [lo, hi]}
        return item

    by_id = {s["scaffold_id"]: s for s in kept}

    def render(item, shown):
        return scaffold_prompt(by_id[item.extra["scaffold_id"]], shown, None)

    def render_twin(item, shown):
        return scaffold_prompt(by_id[item.extra["scaffold_id"]], shown, None, CONSTRAINT.format(horizon=item.horizon_text))

    def cue(item, variant):
        return explicit_cue_of(item, variant) if variant == "twin" else by_id[item.extra["scaffold_id"]]["cue"]

    return EvalType(type_id, make_item, render, plan=tuple(plan), cue=cue, render_twin=render_twin) if kept else None


def load_boundary_types(path=DEFAULT_SCAFFOLDS):
    """EvalTypes for the scaffold file at `path` (module docstring). Raises ValueError naming the
    scaffold on any malformed entry. A type with no scaffold, or with none feasible, is not returned."""
    scaffolds = json.loads(Path(path).read_text())
    ids = [s.get("scaffold_id") for s in scaffolds]
    for scaffold in scaffolds:
        validate_scaffold(scaffold)
    duplicates = sorted(sid for sid, n in Counter(ids).items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate scaffold_id: {', '.join(map(str, duplicates))}")
    types = []
    for type_id in LLM_TYPES + PERSONA_TYPES:
        own = [s for s in scaffolds if s["type_id"] == type_id]
        if own:
            et = llm_type(type_id, own) if type_id in LLM_TYPES else persona_type(type_id, own)
            if et:
                types.append(et)
    return types


TEMPLATED_SEED = "boundary_mt"


def templated(type_id, make_item, render, cue=explicit_cue_of, **kwargs):
    return EvalType(type_id, make_item, render, plan=BOUNDARY_PLAN, item_seed=TEMPLATED_SEED, cue=cue, **kwargs)


def t0_item(rng, i, regime):
    return standard_item(rng, regime)


TYPES = [
    templated("mt_first", t0_item, mt_first_render),
    templated("mt_last", t0_item, mt_last_render),
    templated("mt_revised", mt_revised_item, mt_revised_render),
    templated("mt_long", mt_long_item, mt_long_render),
    templated("mt_assistant", t0_item, mt_assistant_render, cue=mt_assistant_cue),
    templated("no_horizon", no_horizon_item, no_horizon_render, cue=lambda item, variant: None, twins=False),
    EvalType("implied_event", implied_event_item, implied_event_render, plan=BOUNDARY_PLAN, cue=implied_event_cue,
             render_twin=implied_event_twin),
]
if DEFAULT_SCAFFOLDS.exists():
    TYPES += load_boundary_types()
