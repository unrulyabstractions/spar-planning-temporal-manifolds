"""Distractor families: sentences that mention a duration which is NOT the decision's time horizon.

Each family is one temporal role. Its templates are split in two groups:
  seen    – may appear in the decoder's training data (and in tier A tests)
  unseen  – never trained on; used only for tier B tests (same role, new wording)
Tier C holds out a whole family (all four templates) from training.

Template placeholders (filled per domain, see DOMAIN_WORDS):
  {D} duration · {Entity}/{entity} · {Role_np}/{role_np} · {thing} · {Neighbour} · {Project}
"""

from __future__ import annotations

from dataclasses import dataclass

from ptm.horizons import Horizon
from ptm.matrix import MATRIX_HORIZONS, PROSE


def _h(*texts: str) -> tuple[Horizon, ...]:
    return tuple(Horizon.parse(t) for t in texts)


# The matrix grid (1 day .. 100 years) and a version capped at 30 years for roles where a century is absurd.
GRID_12 = MATRIX_HORIZONS
GRID_TO_30Y = tuple(h for h in MATRIX_HORIZONS if h.years <= 30)
PREP_DURATIONS = _h("1 hour", "2 hours", "1 day", "3 days", "1 week", "2 weeks", "1 month")


@dataclass(frozen=True)
class Family:
    name: str
    description: str
    seen: tuple[str, str]
    unseen: tuple[str, str]
    durations: tuple[Horizon, ...]

    def templates(self, group: str) -> tuple[str, ...]:
        return {"seen": self.seen, "unseen": self.unseen, "all": self.seen + self.unseen}[group]


FAMILIES: tuple[Family, ...] = (
    Family(
        name="entity_age",
        description="age of an organisation or account (generalises Alan's 'mention' control)",
        seen=("{Entity} was established {D} ago.", "{Entity} was set up {D} ago."),
        unseen=("{Entity} has existed for {D}.", "{Entity} first opened {D} ago."),
        durations=GRID_12,
    ),
    Family(
        name="prep_time",
        description="time available to prepare the answer (generalises Alan's 'role' control)",
        seen=("You have {D} to prepare your recommendation.", "Your recommendation is due in {D}."),
        unseen=("The meeting where you present your choice is in {D}.", "You were given {D} to study the two options."),
        durations=PREP_DURATIONS,
    ),
    Family(
        name="tenure",
        description="the decision-maker's own history in the role",
        seen=("You have held this role for {D}.", "You took on this responsibility {D} ago."),
        unseen=("Your predecessor held the role for {D}.", "You last reviewed {role_np}'s plans {D} ago."),
        durations=GRID_TO_30Y,
    ),
    Family(
        name="background_event",
        description="an unrelated local event",
        seen=("The office building was last renovated {D} ago.", "The local school finished building its new gym {D} ago."),
        unseen=("The town library changed its opening hours {D} ago.", "{Project} was completed {D} ago."),
        durations=GRID_TO_30Y,
    ),
    Family(
        name="other_horizon",
        description="a horizon-like duration that belongs to a different decision (defeats the 'horizon'-sentence regex)",
        seen=("For an unrelated decision last year, {role_np} used a time horizon of {D}.",
              "{Neighbour} plans its {thing} over a horizon of {D}."),
        unseen=("The previous plan, which has now ended, covered a period of {D}.",
                "A separate long-term project of {role_np} runs for {D}."),
        durations=GRID_12,
    ),
)
FAMILY_BY_NAME = {f.name: f for f in FAMILIES}

# Domain words. entity / role_np / thing come from Alan's PROSE so wording matches his controls.
_EXTRA = {
    "investment": dict(neighbour="a neighbouring household", project="the household's kitchen renovation"),
    "climate": dict(neighbour="a neighbouring city", project="the city's new football stadium"),
}


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def domain_words(domain: str) -> dict[str, str]:
    p, x = PROSE[domain], _EXTRA[domain]
    return dict(entity=p.entity, Entity=_cap(p.entity), role_np=p.role_np, Role_np=_cap(p.role_np), thing=p.thing,
                Neighbour=_cap(x["neighbour"]), Project=_cap(x["project"]))


def fill(template: str, domain: str, d: Horizon) -> str:
    return template.format(D=str(d), **domain_words(domain))
