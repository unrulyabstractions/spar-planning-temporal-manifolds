"""Eval types: robustness group (EvalsPromptTypes.md). The reference example for the contract in
common.py: a T0 item with one added sentence."""

from ..horizons import Horizon
from .common import PLAN, EvalType, alan_prompt, log_years, standard_item
from .framing import expansion_index

# Irrelevant facts that state a duration; each type uses the one whose duration is far from the horizon.
TEMPORAL_DISTRACTORS = (
    ("The household has lived in the same house for {d}.", (Horizon(40, "years"), Horizon(25, "years"))),
    ("The household's last holiday lasted {d}.", (Horizon(2, "weeks"), Horizon(10, "days"))),
)

# The expanded banks use these from item len(PLAN) on, one entry per item in turn within each group:
# long durations (at least 25 years) for a short horizon, short ones (at most 3 weeks) for a long one.
NEW_LONG_DISTRACTORS = (
    ("The family farm has been in the household for {d}.", Horizon(60, "years")),
    ("The household's grandfather worked at the same company for {d}.", Horizon(35, "years")),
    ("The household's oldest clock has kept time for {d}.", Horizon(80, "years")),
    ("The church in the household's neighbourhood has stood for {d}.", Horizon(150, "years")),
    ("The household's old radio was bought {d} ago.", Horizon(40, "years")),
    ("The household's family business has traded for {d}.", Horizon(50, "years")),
    ("The household's garden wall is {d} old.", Horizon(70, "years")),
    ("The head of the household has known their neighbours for {d}.", Horizon(30, "years")),
    ("The household's piano has been in the family for {d}.", Horizon(90, "years")),
    ("The household's street was paved {d} ago.", Horizon(45, "years")),
    ("The household's village fair has run for {d}.", Horizon(65, "years")),
    ("The household's family album goes back {d}.", Horizon(100, "years")),
    ("The household's local bakery has been open for {d}.", Horizon(55, "years")),
    ("The oldest member of the household has lived in the town for {d}.", Horizon(75, "years")),
    ("The household's roof was last replaced {d} ago.", Horizon(28, "years")),
    ("The household's family recipe book has been handed down for {d}.", Horizon(120, "years")),
)
NEW_SHORT_DISTRACTORS = (
    ("The household's summer trip lasted {d}.", Horizon(3, "weeks")),
    ("The household's kitchen renovation took {d}.", Horizon(2, "weeks")),
    ("The household's cat was away for {d}.", Horizon(5, "days")),
    ("The household's last cold lasted {d}.", Horizon(10, "days")),
    ("The household's new sofa took {d} to be delivered.", Horizon(12, "days")),
    ("The household's guests stayed for {d}.", Horizon(1, "weeks")),
    ("The household's power cut last winter lasted {d}.", Horizon(3, "days")),
    ("The household's car was in the garage for {d}.", Horizon(9, "days")),
    ("The household's last family wedding celebration went on for {d}.", Horizon(4, "days")),
    ("The household's broadband was down for {d}.", Horizon(6, "days")),
    ("The household's trip to the coast took {d}.", Horizon(8, "days")),
    ("The household's last school break was {d} long.", Horizon(2, "weeks")),
    ("The household's neighbours were away for {d}.", Horizon(3, "weeks")),
    ("The household's last parcel took {d} to arrive.", Horizon(11, "days")),
    ("The household's tomato seeds took {d} to sprout.", Horizon(7, "days")),
    ("The household's last long weekend lasted {d}.", Horizon(3, "days")),
)


def distractor_item(rng, i, regime):
    """T0 with a duration in SITUATION that is not the horizon: the long one (40 or 25 years) when
    the horizon is short, the short one (2 weeks or 10 days) when it is long, so the distractor is
    always at least a decade (log) away from the horizon."""
    item = standard_item(rng, regime)
    is_short_horizon = item.horizon_years < 3
    if i < len(PLAN):
        template, durations = TEMPORAL_DISTRACTORS[0] if is_short_horizon else TEMPORAL_DISTRACTORS[1]
        d = rng.choice(durations)
    else:
        pool = NEW_LONG_DISTRACTORS if is_short_horizon else NEW_SHORT_DISTRACTORS
        template, d = pool[expansion_index(len(pool), i)]
    item.extra = {"distractor_text": template.format(d=d), "distractor_years": d.years,
                  "distractor_log_gap": abs(log_years(d) - log_years(item.horizon_years))}
    return item


def distractor_render(item, shown):
    return alan_prompt(shown, situation=f"Plan for the future of the household. {item.extra['distractor_text']}",
                       horizon_text=item.horizon_text)


TYPES = [EvalType("distractor_temporal", distractor_item, distractor_render)]
