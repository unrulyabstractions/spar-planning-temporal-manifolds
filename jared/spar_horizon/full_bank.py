"""The full prompt bank: Alan's parametric intertemporal-choice prompts in the prior
work's highly-formatted template, plus the record shape our experiments consume.

The generator (`Domain`, `PromptFormat`, `PromptSample`, `UNIT_REWRITES`,
`CONSTRAINT_PHRASINGS`, the delay/reward grids, `DatasetConfig`, `generate`) is
copied from Alan's `../alan/ptm/prompts.py` (snapshot 2026-09-25) with the pandas
helpers dropped; `tests/test_full_bank.py` checks the copy against his output.
Only `DatasetConfig.horizons` defaults differ: our 48-point log-spaced grid
(30 seconds to 500 years, `HORIZONS`) replaces his 17-point one
(`horizons.STANDARD_HORIZONS`). `DatasetConfig.rewrites` is ours too: the
rendering table `unit_rewrite` draws from, by default his `UNIT_REWRITES`.

Every sample is fully described by a `PromptSample` record; the rendered text is a
pure function of that record and a `PromptFormat`. All parameters are stored as
columns so that any surface feature can be used as a covariate or a control.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from itertools import zip_longest
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
    rewrites: dict = field(default_factory=lambda: UNIT_REWRITES)   # canonical text -> other renderings
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
            renderings = [h_canon] + (list(cfg.rewrites.get(h_canon, ())) if cfg.unit_rewrite else [])
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


# --- the units bank (14_spec_unit_overlap.md) ---------------------------------
# The same duration in its natural unit and in adjacent units, so that unit and
# magnitude are no longer confounded: months then run from "2 months" to "708 months".

MAX_RENDERED_VALUE = 1000   # no "15768000 seconds"
MAX_RENDERING_ERROR = 0.10  # as natural_horizon's rounding


def neighbour_renderings(h):
    """Integer renderings of `h` in the units adjacent to its own in NATURAL_UNITS,
    within 10% of its duration and from 1 to 1000."""
    i = NATURAL_UNITS.index(h.unit)
    out = []
    for unit in (NATURAL_UNITS[i - 1] if i > 0 else None, NATURAL_UNITS[i + 1] if i + 1 < len(NATURAL_UNITS) else None):
        if unit is None:
            continue
        value = round(h.years / UNIT_YEARS[unit])
        if 1 <= value <= MAX_RENDERED_VALUE and abs(value * UNIT_YEARS[unit] / h.years - 1) <= MAX_RENDERING_ERROR:
            out.append(str(Horizon(value, unit)))
    return tuple(out)


NEIGHBOUR_REWRITES: dict[str, tuple[str, ...]] = {
    str(h): neighbour_renderings(h) for h in HORIZONS if neighbour_renderings(h)}
UNIT_DURATIONS: tuple[Horizon, ...] = tuple(h for h in HORIZONS if str(h) in NEIGHBOUR_REWRITES)
UNITS_BASE = 250   # base prompts; each is emitted once per rendering, ~520 prompts


def build_units_prompts(seed=0, n_base=UNITS_BASE, limit=None):
    """Paired records: `n_base` base prompts, each in every rendering of its duration,
    sharing rewards, delays, order and labels. A base whose renderings fall in
    different regimes (a rendering's rounding crosses a delay) is dropped, so a
    pair differs in the phrase only. Extra keys: `pair_id`, `duration_id` (the
    natural rendering), `rendered_unit`, `natural`. `limit` keeps evenly spaced
    whole pairs, about `limit` records."""
    cfg = DatasetConfig(n=8 * n_base, seed=seed, horizons=UNIT_DURATIONS, null_horizon_fraction=0.0,
                        unit_rewrite=True, rewrites=NEIGHBOUR_REWRITES, paired=True)
    pairs = {}
    for s in generate(cfg):
        rec = record(s)
        rec.update(pair_id=s.pair_id, duration_id=s.horizon_canonical, rendered_unit=s.horizon_unit,
                   natural=s.horizon_text == s.horizon_canonical)
        pairs.setdefault(s.pair_id, []).append(rec)
    by_duration = {str(h): [] for h in UNIT_DURATIONS}
    for p in pairs.values():
        if len(p) == 1 + len(NEIGHBOUR_REWRITES[p[0]["duration_id"]]) and len({r["regime"] for r in p}) == 1:
            by_duration[p[0]["duration_id"]].append(p)
    # round-robin over durations, so each gets n_base / 33 bases (uniform draws leave some with 3)
    complete = [p for row in zip(*by_duration.values()) for p in row][:n_base]
    if len(complete) < n_base:
        raise ValueError(f"only {len(complete)} balanced pairs for n_base={n_base}")
    if limit is not None:
        n_pairs = max(1, limit // 2)
        complete = complete[::max(1, len(complete) // n_pairs)][:n_pairs]
    return [rec for p in complete for rec in p]


# --- the crossed bank (16_spec_pristine_manifold.md §4) -----------------------
# Every duration in every unit whose value lies in 0.1-1000, so that unit and magnitude
# are crossed rather than adjacent; delays sit at random log margins from the horizon,
# so every regime occurs at every duration. Results on it are ours, not a replication:
# the delays no longer follow Alan's grids.

CROSSED_VALUES = (0.1, 1000)
EXACT_INTEGER = 0.01                                # "365 days" for 365.25, not "370 days"
DELAY_LIMITS = (UNIT_YEARS["minutes"], 1000.0)      # years
MARGIN = (0.2, 1.5)                                 # decades from the horizon to the nearer delay
GAP = (0.3, 2.0)                                    # decades between the delays, when both are on one side
MIN_DISTANCE = 0.2                                  # decades between any rendered delay and any horizon rendering
CROSSED_PER_DURATION = 4                            # 192 bases, ~530 prompts
CROSSED_REGIMES = ("none", "near", "both")
MAX_DRAWS = 10_000


def crossed_rendering(years, unit):
    """`years` in `unit`, or None outside 0.1-1000: the integer when it is within 1%,
    else 2 significant figures ("0.5 years", "1.8 months")."""
    value = years / UNIT_YEARS[unit]
    if not CROSSED_VALUES[0] <= value <= CROSSED_VALUES[1]:
        return None
    integer = round(value)
    if integer >= 1 and abs(integer / value - 1) <= EXACT_INTEGER:
        return Horizon(integer, unit)
    return Horizon(float(f"{value:.2g}"), unit)


def crossed_renderings(years):
    """Every rendering of `years` in NATURAL_UNITS, largest unit first."""
    return tuple(r for unit in NATURAL_UNITS for r in [crossed_rendering(years, unit)] if r is not None)


def delays_for(log_h, regime_name, margin, other_margin, gap):
    """(near, far) delays in log10 years at the given margins from the horizon."""
    if regime_name == "none":
        return log_h + margin, log_h + margin + gap
    if regime_name == "near":
        return log_h - margin, log_h + other_margin
    return log_h - margin - gap, log_h - margin


def feasible_regimes(log_h):
    """The regimes for which some margins within MARGIN and GAP put both delays inside
    the limits (at 30 seconds only none: its near delay can reach 1 minute)."""
    lo, hi = (math.log10(y) for y in DELAY_LIMITS)
    fits = {"none": max(MARGIN[0], lo - log_h) <= min(MARGIN[1], hi - log_h - GAP[0]),
            "near": log_h - lo >= MARGIN[0] and hi - log_h >= MARGIN[0],
            "both": log_h - MARGIN[0] - GAP[0] >= lo and max(MARGIN[0], log_h - hi) <= MARGIN[1]}
    return [name for name in CROSSED_REGIMES if fits[name]]


def realism_labels(near_years, far_years):
    """Labels only (Jared, 2026-10-27): how odd the payout dates read for an investment."""
    if near_years < UNIT_YEARS["days"]:
        scale = "minutes-hours"
    elif near_years < UNIT_YEARS["months"]:
        scale = "days-weeks"
    else:
        scale = "months-years"
    alan = all(UNIT_YEARS["weeks"] <= y <= 50 for y in (near_years, far_years))
    return {"delay_scale": scale, "alan_range": alan}


def draw_crossed_delays(rng, renderings, regime_name):
    """Rendered (near, far) delays for `regime_name`, redrawn until both lie within the
    limits and at least MIN_DISTANCE decades from every horizon rendering, so rounding
    never moves a rendering into another regime."""
    log_h = math.log10(renderings[0].years)
    lo, hi = DELAY_LIMITS
    for _ in range(MAX_DRAWS):
        near, far = delays_for(log_h, regime_name, rng.uniform(*MARGIN), rng.uniform(*MARGIN), rng.uniform(*GAP))
        if not (math.log10(lo) <= near and far <= math.log10(hi)):
            continue
        delays = [rng.choice(crossed_renderings(10 ** d)) for d in (near, far)]
        if all(lo <= d.years <= hi and abs(math.log10(d.years / r.years)) >= MIN_DISTANCE
               for d in delays for r in renderings):
            return delays
    raise ValueError(f"no {regime_name} delays for {renderings[0]} in {MAX_DRAWS} draws")


def crossed_plan(per_duration=CROSSED_PER_DURATION):
    """(duration, regime) for each base of the standalone crossed bank. Regimes are
    assigned, not drawn: four uniform draws leave most durations without one regime,
    so each duration takes every feasible regime in turn, starting one further along
    CROSSED_REGIMES at each duration, so the extra bases rotate."""
    plan = []
    for j, h in enumerate(HORIZONS):
        feasible = feasible_regimes(h.log10_years)
        rotated = [name for name in CROSSED_REGIMES[j % 3:] + CROSSED_REGIMES[:j % 3] if name in feasible]
        plan += [(h, rotated[b % len(rotated)]) for b in range(per_duration)]
    return plan


def emit_crossed(plan, seed, prefix):
    """One base per (duration, regime) in `plan`, as a list of records per base: the base
    emitted once per rendering of its duration, sharing delays, rewards, order and
    labels. Extra keys: `pair_id`, `duration_id` (the natural rendering),
    `rendered_unit`, `natural`, `delay_scale`, `alan_range`."""
    rng = random.Random(seed)
    fmt = PromptFormat()
    bases = []
    for i, (h, regime_name) in enumerate(plan):
        renderings = crossed_renderings(h.years)
        near, far = draw_crossed_delays(rng, renderings, regime_name)
        short_reward = rng.choice(SHORT_REWARDS)
        long_reward = short_reward * rng.choice(REWARD_MULTIPLIERS)
        short_first = rng.random() < 0.5
        base = []
        for k, r in enumerate(renderings):
            s = PromptSample(sample_uid=f"{prefix}{seed}_{i:04d}_{k}", domain="investment", horizon_text=str(r),
                             horizon_years=r.years, short_reward=float(short_reward), short_delay_text=str(near),
                             short_delay_years=near.years, long_reward=float(long_reward),
                             long_delay_text=str(far), long_delay_years=far.years, short_first=short_first,
                             label_a="a)", label_b="b)", format_name=fmt.name, horizon_canonical=str(h),
                             horizon_unit=r.unit, horizon_value=r.value, phrasing_id=0,
                             pair_id=f"{prefix}{seed}_{i:04d}")
            s.text = fmt.render(s)
            s.prompt_id = hashlib.sha1(s.text.encode()).hexdigest()[:12]
            rec = record(s)
            rec.update(pair_id=s.pair_id, duration_id=str(h), rendered_unit=r.unit, natural=str(r) == str(h),
                       **realism_labels(near.years, far.years))
            base.append(rec)
        bases.append(base)
    return bases


def evenly_spaced_bases(bases, limit):
    """Evenly spaced whole bases, about `limit` records; all of them when `limit` is None."""
    if limit is None:
        return bases
    n_records = sum(len(b) for b in bases)
    n_bases = max(1, limit * len(bases) // n_records)
    return bases[::max(1, len(bases) // n_bases)][:n_bases]


def build_crossed_prompts(seed=0, per_duration=CROSSED_PER_DURATION, limit=None):
    """The standalone crossed bank: `per_duration` bases at each of the 48 durations
    (`crossed_plan`). `limit` keeps evenly spaced whole bases, about `limit` records."""
    bases = emit_crossed(crossed_plan(per_duration), seed, "x")
    return [rec for base in evenly_spaced_bases(bases, limit) for rec in base]


# --- the fill bank (16_spec_pristine_manifold.md §4, Jared 2026-10-29) --------
# The crossed generator restricted to the (duration, regime) cells the old full and
# units banks leave nearly empty, pooled with those runs: their delays are independent
# of the horizon, which the crossed bank's are not (corr +0.85).

FILL_MIN_OLD = 3          # a (duration, regime) cell with fewer old prompts is filled
FILL_PER_CELL = 2
ANCHOR_FULL = 12          # old full-bank prompts re-run, to check the runs can be pooled
ANCHOR_PAIRS = 6          # old units-bank pairs re-run


def old_bank_records():
    """The old 8B runs' prompts with a horizon: the full bank, then the units bank."""
    return [r for r in build_full_prompts() + build_units_prompts() if r["horizon"] is not None]


def fill_plan(per_cell=FILL_PER_CELL, min_old=FILL_MIN_OLD):
    """(duration, regime) for each fill base: `per_cell` bases in every feasible cell
    with fewer than `min_old` old prompts."""
    old = Counter((r.get("duration_id") or r["horizon_text"], r["regime"]) for r in old_bank_records())
    return [(h, name) for h in HORIZONS for name in feasible_regimes(h.log10_years)
            if old[(str(h), name)] < min_old for _ in range(per_cell)]


def fill_anchors():
    """Evenly spaced old prompts to re-run, as bases: ANCHOR_FULL full-bank prompts and
    ANCHOR_PAIRS units-bank pairs, keeping their sample_uids. Full-bank prompts get the
    paired keys as one-rendering bases (they are in natural units)."""
    full = [r for r in build_full_prompts() if r["horizon"] is not None]
    bases = []
    for r in full[::len(full) // ANCHOR_FULL][:ANCHOR_FULL]:
        h = Horizon.parse(r["horizon_text"])
        bases.append([{**r, "pair_id": r["sample_uid"], "duration_id": str(h), "rendered_unit": h.unit,
                       "natural": True}])
    pairs = {}
    for r in build_units_prompts():
        pairs.setdefault(r["pair_id"], []).append(r)
    pairs = list(pairs.values())
    bases += pairs[::len(pairs) // ANCHOR_PAIRS][:ANCHOR_PAIRS]
    return [[{**r, **realism_labels(r["near_delay_years"], r["far_delay_years"]), "anchor": True} for r in base]
            for base in bases]


def build_fill_prompts(seed=0, per_cell=FILL_PER_CELL, limit=None):
    """The fill bank (`fill_plan`, extra key `anchor` False) followed by the anchors
    (`anchor` True). `limit` keeps evenly spaced whole bases of each, in proportion."""
    fill = [[{**r, "anchor": False} for r in base] for base in emit_crossed(fill_plan(per_cell), seed, "f")]
    anchors = fill_anchors()
    if limit is not None:
        n_fill = sum(map(len, fill))
        share = n_fill / (n_fill + sum(map(len, anchors)))
        fill = evenly_spaced_bases(fill, max(1, round(limit * share)))
        anchors = evenly_spaced_bases(anchors, max(1, limit - round(limit * share)))
    return [rec for base in fill + anchors for rec in base]


def crossed_mix(records):
    """The achieved mix, for the generator's report: regimes per duration third, with the
    realism labels, and the durations whose regime mix is not all three."""
    third = {str(h): f"{i * 3 // len(HORIZONS) + 1}/3" for i, h in enumerate(HORIZONS)}
    bases = {r["pair_id"]: r for r in records}.values()
    lines = [f"{len(records)} prompts, {len(bases)} bases, "
             f"{sum(r['regime'] in ('near', 'both') for r in records)} near/both prompts",
             f"{'third':6s} {'regime':7s} {'bases':>6s} " + " ".join(f"{s:>14s}" for s in
                                                                 ("minutes-hours", "days-weeks", "months-years"))
             + f" {'alan_range':>11s}"]
    for t in ("1/3", "2/3", "3/3"):
        for name in CROSSED_REGIMES:
            group = [b for b in bases if third[b["duration_id"]] == t and b["regime"] == name]
            scales = [sum(b["delay_scale"] == s for b in group) for s in ("minutes-hours", "days-weeks", "months-years")]
            lines.append(f"{t:6s} {name:7s} {len(group):6d} " + " ".join(f"{n:14d}" for n in scales)
                         + f" {sum(b['alan_range'] for b in group):11d}")
    partial = {h: sorted({b["regime"] for b in bases if b["duration_id"] == str(h)}) for h in HORIZONS}
    lines.append("durations without all three regimes: " + ", ".join(
        f"{h} {'/'.join(names)}" for h, names in partial.items() if len(names) < 3))
    return "\n".join(lines)


# --- the choice banks (18_spec_multichoice.md, 26_spec_overnight_choice_scan.md §6) ---
# k options, half near (one shared delay) and half far (one shared delay), one principal for
# all options, so the half is decided by the horizon and the option within the half by the return.

CHOICE_BANKS = {"choice4": (4, False), "choice8": (8, False), "choice4_roi": (4, True), "choice8_roi": (8, True)}
CHOICE_PER_DURATION = 2                                 # one near and one both base: ~250 prompts per bank
CHOICE_REGIMES = ("near", "both")
NEAR_RETURNS = (2, 4, 6, 8, 10, 12, 15, 20)             # percent
FAR_RETURNS = (50, 75, 100, 150, 200, 300, 400, 500)
NUMBER_WORDS = {4: "four", 8: "eight"}
LETTERS = "abcdefgh"


def choice_labels(k):
    return tuple(LETTERS[:k])


def choice_prompt_text(options, horizon_text, principal, roi_explicit):
    """Alan's template for the options as listed (dicts with `label`, `payout`, `delay_text`),
    shown as `a)` ... . Only the TASK line differs between the implicit and the explicit ROI."""
    fmt, d = PromptFormat(), DOMAINS["investment"]
    shown = [f"{o['label']})" for o in options]
    task = (f"have {principal:,} {d.reward_unit} to invest and are tasked to {d.task}:" if roi_explicit
            else f"are tasked to {d.task}:")
    lines = [f"{fmt.situation_marker} {d.situation}", f"{fmt.task_marker} You, {d.role}, {task}"]
    lines += [f"{lab} {o['payout']:,} {d.reward_unit} in {o['delay_text']}." for lab, o in zip(shown, options)]
    lines += [f"{fmt.objective_marker} {fmt.objective_text}",
              f"{fmt.constraint_marker} {fmt.constraint_text.format(horizon=horizon_text)}",
              f"{fmt.action_marker} Select one of the {NUMBER_WORDS[len(options)]} options. Provide reasoning.",
              f"{fmt.format_marker} {fmt.choice_prefix} <{', '.join(shown[:-1])} or {shown[-1]}>. "
              f"{fmt.reasoning_prefix} <1-3 sentences>"]
    return "\n".join(lines)


def choice_variant(rec, swap=False, labels=None):
    """The record's prompt with the options in reverse order (`swap`) and/or labelled 1..k
    (`labels` given; its values are ignored, a choice bank has k labels, not two)."""
    options = rec["options"][::-1] if swap else rec["options"]
    if labels is not None:
        options = [{**o, "label": str(i + 1)} for i, o in enumerate(options)]
    return choice_prompt_text(options, rec["horizon_text"], rec["principal"], rec["roi_explicit"])


def van_der_corput(i):
    """The i-th point of the base-2 van der Corput sequence in [0, 1): 0, .5, .25, .75, .125, ..."""
    x, scale = 0.0, 0.5
    while i:
        x += (i & 1) * scale
        i >>= 1
        scale /= 2
    return x


def choice_plan(per_duration):
    """(duration, regime) for each base, in run order. The durations are those where both
    regimes are feasible; a duration's bases alternate near / both, starting with the other
    regime at the next duration. Within a regime the durations come in van der Corput order
    and the two regimes alternate, so every prefix of the list covers the duration range
    and both regimes evenly."""
    kept = [h for h in HORIZONS if {"near", "both"} <= set(feasible_regimes(h.log10_years))]
    by_regime = {name: [] for name in CHOICE_REGIMES}
    for j, h in enumerate(kept):
        for b in range(per_duration):
            by_regime[CHOICE_REGIMES[(b + j) % 2]].append((b // 2, van_der_corput(j), h))
    queues = [[(h, name) for _, _, h in sorted(by_regime[name], key=lambda t: t[:2])] for name in CHOICE_REGIMES]
    return [item for pair in zip_longest(*queues) for item in pair if item is not None]


def truth_positions(rng, k, n):
    """`n` positions in 0..k-1 as consecutive random permutations, so counts differ by at most 1."""
    out = []
    while len(out) < n:
        block = list(range(k))
        rng.shuffle(block)
        out += block
    return out[:n]


def choice_base(rng, h, regime_name, k, roi_explicit, truth_pos):
    """One base: horizon renderings, delays, principal, returns and option order (all drawn
    from `rng`; `roi_explicit` only changes the text, so ROI twins share every draw)."""
    renderings = crossed_renderings(h.years)
    near, far = draw_crossed_delays(rng, renderings, regime_name)
    principal = rng.choice(SHORT_REWARDS)
    half = k // 2
    options = []
    for name, delay, returns in (("near", near, NEAR_RETURNS), ("far", far, FAR_RETURNS)):
        ordered = sorted(rng.sample(returns, half), reverse=True)
        options += [{"payout": principal * (100 + r) // 100, "return_pct": r, "delay_text": str(delay),
                     "delay_years": delay.years, "half": name, "rank_in_half": rank + 1}
                    for rank, r in enumerate(ordered)]
    truth_half = TRUTH[regime_name]
    truth = next(o for o in options if o["half"] == truth_half and o["rank_in_half"] == 1)
    rest = [o for o in options if o is not truth]
    rng.shuffle(rest)
    shown = rest[:truth_pos] + [truth] + rest[truth_pos:]
    for label, o in zip(choice_labels(k), shown):
        o["label"] = label
    return renderings, near, far, principal, shown, truth


def emit_choice(k, roi_explicit, seed, per_duration, limit):
    rng = random.Random(f"choice{k}-{seed}")
    plan = choice_plan(per_duration)
    positions = {name: truth_positions(rng, k, sum(r == name for _, r in plan)) for name in CHOICE_REGIMES}
    taken = dict.fromkeys(CHOICE_REGIMES, 0)
    records = []
    for i, (h, regime_name) in enumerate(plan):
        pos = positions[regime_name][taken[regime_name]]
        taken[regime_name] += 1
        renderings, near, far, principal, shown, truth = choice_base(rng, h, regime_name, k, roi_explicit, pos)
        in_half = sorted((o for o in shown if o["half"] == truth["half"]), key=lambda o: o["rank_in_half"])
        base = []
        for j, r in enumerate(renderings):
            text = choice_prompt_text(shown, str(r), principal, roi_explicit)
            reg = regime(r.years, near.years, far.years)
            assert reg == regime_name
            base.append({
                "prompt": text, "horizon": r.years, "horizon_text": str(r),
                "near_delay_years": near.years, "far_delay_years": far.years, "regime": reg, "truth": TRUTH[reg],
                "sample_uid": f"c{k}_{seed}_{i:04d}_{j}", "prompt_id": hashlib.sha1(text.encode()).hexdigest()[:12],
                "domain": "investment", "phrasing_id": 0,
                "pair_id": f"c{k}_{seed}_{i:04d}", "duration_id": str(h), "rendered_unit": r.unit,
                "natural": str(r) == str(h), **realism_labels(near.years, far.years),
                "n_options": k, "principal": principal, "roi_explicit": roi_explicit,
                "options": [dict(o) for o in shown], "labels": [o["label"] for o in shown],
                "near_labels": [o["label"] for o in shown if o["half"] == "near"],
                "far_labels": [o["label"] for o in shown if o["half"] == "far"],
                "truth_label": truth["label"], "gap_in_half": in_half[0]["payout"] / in_half[1]["payout"]})
        records.append(base)
        if limit is not None and sum(map(len, records)) >= limit:
            break
    return records


def build_choice_prompts(k, roi_explicit=False, seed=0, per_duration=CHOICE_PER_DURATION, limit=None):
    """The k-option bank (k = 4 or 8) as records, in run order: `per_duration` bases at each
    duration where both `near` and `both` are feasible, each emitted once per rendering of its
    duration. `limit` keeps the first whole bases until there are at least `limit` records. The RNG is the
    bank's own, so the other banks are untouched. See `choice_plan` for the order."""
    if k not in NUMBER_WORDS:
        raise ValueError(f"k must be 4 or 8, got {k}")
    return [rec for base in emit_choice(k, roi_explicit, seed, per_duration, limit) for rec in base]
