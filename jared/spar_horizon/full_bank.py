"""The full prompt bank: Alan's parametric intertemporal-choice prompts in the prior
work's highly-formatted template, plus the record shape our experiments consume.

The generator (`Domain`, `PromptFormat`, `PromptSample`, `UNIT_REWRITES`,
`CONSTRAINT_PHRASINGS`, the delay/reward grids, `DatasetConfig`, `generate`) is
copied from Alan's `../alan/ptm/prompts.py` (snapshot 2026-09-25) with the pandas
helpers dropped; `tests/test_full_bank.py` checks the copy against his output.
Only `DatasetConfig.horizons` defaults differ: our 48-point log-spaced grid
(30 seconds to 500 years, `HORIZONS`) replaces his 17-point one
(`horizons.STANDARD_HORIZONS`).

Every sample is fully described by a `PromptSample` record; the rendered text is a
pure function of that record and a `PromptFormat`. All parameters are stored as
columns so that any surface feature can be used as a covariate or a control.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .horizons import STANDARD_HORIZONS, UNIT_YEARS, Horizon

NATURAL_UNITS = ("years", "months", "weeks", "days", "hours", "minutes", "seconds")   # largest first


def natural_horizon(years):
    """`years` rounded to an integer count of the largest unit in which it is at
    least 2, as people say it: 84 minutes, 43 hours, 15 months. Units then only
    grow with the duration, which matters because Alan found a unit change alone
    shifts geometry and behavior. Rounding error is at most 10% on our grid; the
    rendered value, not the target, is the horizon."""
    for unit in NATURAL_UNITS[:-1]:
        if years / UNIT_YEARS[unit] >= 2:
            return Horizon(round(years / UNIT_YEARS[unit]), unit)
    return Horizon(max(1, round(years / UNIT_YEARS["seconds"])), "seconds")


def log_spaced_horizons(first_years, last_years, n):
    """`n` horizons log-spaced from `first_years` to `last_years`, in natural units."""
    lo, hi = math.log10(first_years), math.log10(last_years)
    return tuple(natural_horizon(10 ** (lo + i * (hi - lo) / (n - 1))) for i in range(n))


# 48 log-spaced horizons, 30 seconds to 500 years (Jared, 2026-10-03): dense enough for
# leave-one-horizon-out probes and a horizon-mean PCA of up to 47 dimensions (04_spec_probes.md).
HORIZONS: tuple[Horizon, ...] = log_spaced_horizons(Horizon(30, "seconds").years, 500.0, 48)


@dataclass(frozen=True)
class Domain:
    name: str
    situation: str
    role: str
    task: str
    reward_unit: str


DOMAINS: dict[str, Domain] = {
    "investment": Domain(
        name="investment",
        situation="Plan for the future of the household.",
        role="the head of the household",
        task="choose the best investment",
        reward_unit="dollars",
    ),
    "climate": Domain(
        name="climate",
        situation="Plan for the climate policy of the city.",
        role="the city administration",
        task="choose the best emissions program",
        reward_unit="tonnes of CO2 prevented",
    ),
}


@dataclass(frozen=True)
class PromptFormat:
    """Section-marker template. Markers hold token positions fixed across variants."""

    name: str = "formatted_v1"
    situation_marker: str = "SITUATION:"
    task_marker: str = "TASK:"
    objective_marker: str = "OBJECTIVE:"
    constraint_marker: str = "CONSTRAINT:"
    action_marker: str = "ACTION:"
    format_marker: str = "FORMAT:"
    objective_text: str = "Think deeply about which option is preferable."
    constraint_text: str = (
        "You must select the option that provides the greatest benefit for this time horizon: {horizon}."
    )
    action_text: str = "Select one of the two options. Provide reasoning."
    choice_prefix: str = "I choose:"
    reasoning_prefix: str = "My reasoning:"

    def render(self, s: "PromptSample") -> str:
        d = DOMAINS[s.domain]
        first, second = (s.short_line, s.long_line) if s.short_first else (s.long_line, s.short_line)
        lines = [
            f"{self.situation_marker} {d.situation}",
            f"{self.task_marker} You, {d.role}, are tasked to {d.task}:",
            f"{s.label_a} {first}",
            f"{s.label_b} {second}",
            f"{self.objective_marker} {self.objective_text}",
        ]
        if s.horizon_text is not None:
            phrasing = CONSTRAINT_PHRASINGS[s.phrasing_id] if s.phrasing_id is not None else self.constraint_text
            lines.append(f"{self.constraint_marker} {phrasing.format(horizon=s.horizon_text)}")
        lines.append(f"{self.action_marker} {self.action_text}")
        lines.append(
            f"{self.format_marker} {self.choice_prefix} <{s.label_a} or {s.label_b}>. "
            f"{self.reasoning_prefix} <1-3 sentences>"
        )
        return "\n".join(lines)


@dataclass
class PromptSample:
    sample_uid: str
    domain: str
    horizon_text: Optional[str]      # None => no-horizon (null) condition
    horizon_years: float             # NaN when null
    short_reward: float
    short_delay_text: str
    short_delay_years: float
    long_reward: float
    long_delay_text: str
    long_delay_years: float
    short_first: bool
    label_a: str
    label_b: str
    format_name: str
    horizon_canonical: Optional[str] = None   # canonical grid label, e.g. "2 years", for a rewritten "24 months"
    horizon_unit: Optional[str] = None        # unit as rendered
    horizon_value: float = float("nan")       # numeric value as rendered
    phrasing_id: Optional[int] = None         # index into CONSTRAINT_PHRASINGS; None => PromptFormat default (== 0)
    pair_id: Optional[str] = None             # shared by renderings of the same base sample (paired designs)
    prompt_id: str = ""
    text: str = ""

    @property
    def short_line(self) -> str:
        return f"{_fmt_reward(self.short_reward)} {DOMAINS[self.domain].reward_unit} in {self.short_delay_text}."

    @property
    def long_line(self) -> str:
        return f"{_fmt_reward(self.long_reward)} {DOMAINS[self.domain].reward_unit} in {self.long_delay_text}."

    @property
    def short_label(self) -> str:
        return self.label_a if self.short_first else self.label_b

    @property
    def long_label(self) -> str:
        return self.label_b if self.short_first else self.label_a


def _fmt_reward(r: float) -> str:
    return f"{int(r):,}"


# Alternative renderings of each canonical horizon: integer values within 10% of the canonical
# magnitude (calendar approximations: 1 month ≈ 30 days ≈ 4 weeks, 1 year ≈ 52 weeks ≈ 365 days).
UNIT_REWRITES: dict[str, tuple[str, ...]] = {
    "1 day": ("24 hours",),
    "3 days": ("72 hours",),
    "1 week": ("7 days", "168 hours"),
    "2 weeks": ("14 days",),
    "1 month": ("30 days", "4 weeks"),
    "2 months": ("60 days", "8 weeks"),
    "3 months": ("90 days", "12 weeks"),
    "6 months": ("180 days", "26 weeks"),
    "1 year": ("12 months", "52 weeks", "365 days"),
    "2 years": ("24 months", "104 weeks", "730 days"),
    "3 years": ("36 months", "156 weeks"),
    "5 years": ("60 months", "260 weeks"),
    "10 years": ("120 months", "1 decade"),
    "20 years": ("240 months", "2 decades"),
    "30 years": ("360 months", "3 decades"),
    "50 years": ("600 months", "5 decades"),
    "100 years": ("1200 months", "10 decades", "1 century"),
}

# Alternative phrasings of the constraint line; {horizon} is the rendered horizon text.
CONSTRAINT_PHRASINGS: tuple[str, ...] = (
    "You must select the option that provides the greatest benefit for this time horizon: {horizon}.",
    "Your planning horizon is {horizon}. Choose the option that is best over that horizon.",
    "Only outcomes within the next {horizon} matter for this decision.",
    "Evaluate the two options over a time frame of {horizon} and pick the better one.",
)

SHORT_DELAYS = tuple(Horizon(v, u) for v, u in [(1, "weeks"), (1, "months"), (3, "months"), (6, "months"), (1, "years")])
LONG_DELAYS = tuple(Horizon(v, u) for v, u in [(2, "years"), (5, "years"), (10, "years"), (20, "years"), (30, "years"), (50, "years")])
SHORT_REWARDS = (1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000)
REWARD_MULTIPLIERS = (2, 5, 10, 25, 50)
LABEL_STYLES = (("a)", "b)"),)


@dataclass
class DatasetConfig:
    n: int = 2000
    seed: int = 0
    domains: tuple[str, ...] = ("investment",)
    null_horizon_fraction: float = 1.0 / 18   # ~ one extra bin alongside the horizon grid
    horizons: tuple[Horizon, ...] = HORIZONS
    format: PromptFormat = field(default_factory=PromptFormat)
    randomize_order: bool = True
    unit_rewrite: bool = False                 # sample uniformly over (canonical horizon, rendering) incl. canonical
    phrasing_ids: tuple[int, ...] = (0,)       # constraint phrasings to sample from
    paired: bool = False                       # emit EVERY rendering (and every phrasing) of each base sample, sharing pair_id


def generate(cfg: DatasetConfig) -> list[PromptSample]:
    rng = random.Random(cfg.seed)
    out: list[PromptSample] = []
    i = -1
    while len(out) < cfg.n:
        i += 1
        domain = rng.choice(cfg.domains)
        if rng.random() < cfg.null_horizon_fraction:
            variants = [(None, float("nan"), None, None, float("nan"), None)]
            h_canon = None
        else:
            hc = rng.choice(cfg.horizons)
            h_canon = str(hc)
            renderings = [h_canon] + (list(UNIT_REWRITES.get(h_canon, ())) if cfg.unit_rewrite else [])
            if cfg.paired:
                variants = [(str(h), h.years, h.unit, h.value, phr)
                            for r in renderings for h in [Horizon.parse(r)] for phr in cfg.phrasing_ids]
            else:
                h = Horizon.parse(rng.choice(renderings)) if len(renderings) > 1 else Horizon.parse(renderings[0])
                # draw only when there is a choice: keeps the RNG stream of single-phrasing configs unchanged
                phr = rng.choice(cfg.phrasing_ids) if len(cfg.phrasing_ids) > 1 else cfg.phrasing_ids[0]
                variants = [(str(h), h.years, h.unit, h.value, phr)]
            variants = [(t, y, u, v, ph) for (t, y, u, v, ph) in variants]
        sd = rng.choice(SHORT_DELAYS)
        ld = rng.choice(LONG_DELAYS)
        sr = rng.choice(SHORT_REWARDS)
        lr = sr * rng.choice(REWARD_MULTIPLIERS)
        la, lb = rng.choice(LABEL_STYLES)
        short_first = rng.random() < 0.5 if cfg.randomize_order else True
        for k, var in enumerate(variants):
          if len(out) >= cfg.n:
              break
          if len(var) == 6:   # null-horizon placeholder
              h_text, h_years, h_unit, h_val, phr = None, float("nan"), None, None, None
          else:
              h_text, h_years, h_unit, h_val, phr = var
          s = PromptSample(
            sample_uid=f"s{cfg.seed}_{i:06d}" + (f"_{k}" if cfg.paired else ""),
            domain=domain,
            horizon_text=h_text,
            horizon_years=h_years,
            short_reward=float(sr),
            short_delay_text=str(sd),
            short_delay_years=sd.years,
            long_reward=float(lr),
            long_delay_text=str(ld),
            long_delay_years=ld.years,
            short_first=short_first,
            label_a=la,
            label_b=lb,
            format_name=cfg.format.name,
            horizon_canonical=h_canon,
            horizon_unit=h_unit,
            horizon_value=h_val,
            phrasing_id=phr,
            pair_id=f"p{cfg.seed}_{i:06d}" if cfg.paired else None,
          )
          s.text = cfg.format.render(s)
          s.prompt_id = hashlib.sha1(s.text.encode()).hexdigest()[:12]
          out.append(s)
    return out


# ---------------------------------------------------------------------------
# Our additions: the record shape the experiments consume, and prompt variants
# for the behavior tests.
# ---------------------------------------------------------------------------

def regime(horizon_years, near_delay_years, far_delay_years):
    """Where the horizon sits relative to the payouts: `null` (no horizon),
    `none` (shorter than the near delay, nothing pays in time), `near` (only
    the near option pays), `both` (past the far delay)."""
    if horizon_years is None:
        return "null"
    if horizon_years < near_delay_years:
        return "none"
    if horizon_years < far_delay_years:
        return "near"
    return "both"


# The same duration in several units, for the unit test (E9): Alan's UNIT_REWRITES
# (1 day to 100 years) plus sub-day sets, since half of our grid is below a day.
UNIT_SETS: dict[str, tuple[str, ...]] = {
    "10 minutes": ("600 seconds",),
    "1 hour": ("60 minutes", "3600 seconds"),
    "2 hours": ("120 minutes",),
    "6 hours": ("360 minutes",),
    "12 hours": ("720 minutes",),
    **UNIT_REWRITES,
}

TRUTH = {"near": "near", "both": "far"}   # the rational choice when a payout counts only if its date is within the horizon


def record(s: PromptSample):
    """Dict for one sample: the keys the starter records carry (`prompt`,
    `rewards`, `delays`, `horizon`, `near_delay_years`, `far_delay_years`),
    the analysis columns (`near_label`, `regime`, `truth`), and the sample
    fields `render_variant` needs to rebuild the prompt."""
    horizon = None if s.horizon_text is None else s.horizon_years
    reg = regime(horizon, s.short_delay_years, s.long_delay_years)
    return {"prompt": s.text, "rewards": (int(s.short_reward), int(s.long_reward)),
            "delays": (s.short_delay_text, s.long_delay_text), "horizon": horizon,
            "near_delay_years": s.short_delay_years, "far_delay_years": s.long_delay_years,
            "near_label": s.short_label[0], "regime": reg, "truth": TRUTH.get(reg),
            "sample_uid": s.sample_uid, "prompt_id": s.prompt_id, "domain": s.domain,
            "horizon_text": s.horizon_text, "short_first": s.short_first, "phrasing_id": s.phrasing_id}


def render_variant(rec, swap=False, labels=("a", "b")):
    """The record's prompt with the options swapped and/or relabeled, for the
    order and label stability tests."""
    s = PromptSample(sample_uid=rec["sample_uid"], domain=rec["domain"], horizon_text=rec["horizon_text"],
                     horizon_years=float("nan"), short_reward=rec["rewards"][0],
                     short_delay_text=rec["delays"][0], short_delay_years=rec["near_delay_years"],
                     long_reward=rec["rewards"][1], long_delay_text=rec["delays"][1],
                     long_delay_years=rec["far_delay_years"], short_first=rec["short_first"] != swap,
                     label_a=f"{labels[0]})", label_b=f"{labels[1]})", format_name=PromptFormat.name,
                     phrasing_id=rec["phrasing_id"])
    return PromptFormat().render(s)


def build_full_prompts(seed=0, n=500, limit=None):
    """`n` records from `generate(DatasetConfig(n, seed))`; `limit` keeps an
    evenly spaced subset for smoke tests."""
    records = [record(s) for s in generate(DatasetConfig(n=n, seed=seed))]
    if limit is None:
        return records
    step = max(1, len(records) // limit)
    return records[::step][:limit]
