"""Eval types: robustness group (EvalsPromptTypes.md). The reference example for the contract in
common.py: a T0 item with one added sentence."""

from ..horizons import Horizon
from .common import EvalType, alan_prompt, log_years, standard_item

# Irrelevant facts that state a duration; each type uses the one whose duration is far from the horizon.
TEMPORAL_DISTRACTORS = (
    ("The household has lived in the same house for {d}.", (Horizon(40, "years"), Horizon(25, "years"))),
    ("The household's last holiday lasted {d}.", (Horizon(2, "weeks"), Horizon(10, "days"))),
)


def distractor_item(rng, i, regime):
    """T0 with a duration in SITUATION that is not the horizon: the long one (40 or 25 years) when
    the horizon is short, the short one (2 weeks or 10 days) when it is long, so the distractor is
    always at least a decade (log) away from the horizon."""
    item = standard_item(rng, regime)
    template, durations = TEMPORAL_DISTRACTORS[0] if item.horizon_years < 3 else TEMPORAL_DISTRACTORS[1]
    d = rng.choice(durations)
    item.extra = {"distractor_text": template.format(d=d), "distractor_years": d.years,
                  "distractor_log_gap": abs(log_years(d) - log_years(item.horizon_years))}
    return item


def distractor_render(item, shown):
    return alan_prompt(shown, situation=f"Plan for the future of the household. {item.extra['distractor_text']}",
                       horizon_text=item.horizon_text)


TYPES = [EvalType("distractor_temporal", distractor_item, distractor_render)]
