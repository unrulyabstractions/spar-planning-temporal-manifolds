"""Tier T0: structured investment prompts (report, Appendix A.1).

Two prompt sets are built here, both from the same template:

* **Extraction bank** (`build_extraction_bank`): 500 prompts used to fit steering vectors. 471 state a horizon
  from the 48-point grid; 29 are null-control twins, identical to one of those 471 except that the CONSTRAINT
  line is removed. Most of these prompts have no correct answer (the horizon is shorter than the near delay);
  that is intended, since they are used for extraction, not for scoring behaviour.

* **Verification grid** (`build_verification_grid`): 100 items near the boundary where the correct answer flips
  (horizon H between 0.36 and 2.8 times the far delay D, H written in years like D), each presented in both
  option orders: 200 prompts. Every one has a correct answer. Behaviour is scored only on these.

The template, with one sample (correct answer a, since only the near option pays within 8 years):

    SITUATION: Plan for the future of the household.
    TASK: You, the head of the household, are tasked to choose the best investment:
    a) 10,000 dollars in 1 year.
    b) 50,000 dollars in 10 years.
    OBJECTIVE: Think deeply about which option is preferable.
    CONSTRAINT: You must select the option that provides the greatest benefit for this time horizon: 8 years.
    ACTION: Select one of the two options. Provide reasoning.
    FORMAT: I choose: <a) or b)>. My reasoning: <1-3 sentences>

Reproducibility: each set is a pure function of its config (including the seed). The random draws happen in
a fixed order, documented in each builder; reordering them changes the bank, which `prompts_sha256` in the
manifest will show.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from .durations import HORIZON_GRID, REWRITE_MAX_VALUE, REWRITE_TOLERANCE, Duration, unit_rewrites
from .records import CORRECT_OPTION, Option, PromptRecord, regime

# ---------------------------------------------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------------------------------------------

SITUATION = "SITUATION: Plan for the future of the household."
TASK = "TASK: You, the head of the household, are tasked to choose the best investment:"
OBJECTIVE = "OBJECTIVE: Think deeply about which option is preferable."
ACTION = "ACTION: Select one of the two options. Provide reasoning."

# The four phrasings of the CONSTRAINT line (report, A.1); {H} is the written horizon.
CONSTRAINT_PHRASINGS: tuple[str, ...] = (
    "You must select the option that provides the greatest benefit for this time horizon: {H}.",
    "Your planning horizon is {H}. Choose the option that is best over that horizon.",
    "Only outcomes within the next {H} matter for this decision.",
    "Evaluate the two options over a time frame of {H} and pick the better one.",
)

LABEL_STYLES: tuple[tuple[str, str], ...] = (("a)", "b)"), ("1)", "2)"))


def option_line(label: str, option: Option) -> str:
    """'a) 10,000 dollars in 1 year.'"""
    return f"{label} {option.reward:,} dollars in {option.delay}."


def render(*, near: Option, far: Option, near_first: bool, labels: tuple[str, str],
           horizon: Optional[Duration], phrasing_id: Optional[int]) -> str:
    """The prompt text. With `horizon=None` the CONSTRAINT line is left out (the null control)."""
    first, second = (near, far) if near_first else (far, near)
    lines = [
        SITUATION,
        TASK,
        option_line(labels[0], first),
        option_line(labels[1], second),
        OBJECTIVE,
    ]
    if horizon is not None:
        lines.append("CONSTRAINT: " + CONSTRAINT_PHRASINGS[phrasing_id].format(H=str(horizon)))
    lines += [
        ACTION,
        f"FORMAT: I choose: <{labels[0]} or {labels[1]}>. My reasoning: <1-3 sentences>",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# Option grids (report, A.1: near pays $1k-100k after 1 week to 1 year; far pays near x {2,5,10,25,50}
# after 2-50 years)
# ---------------------------------------------------------------------------------------------------------------

NEAR_DELAYS = tuple(Duration.parse(s) for s in ("1 week", "1 month", "3 months", "6 months", "1 year"))
FAR_DELAYS = tuple(Duration.parse(s) for s in ("2 years", "5 years", "10 years", "20 years", "30 years", "50 years"))
NEAR_REWARDS = (1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000)
REWARD_MULTIPLIERS = (2, 5, 10, 25, 50)


def make_record(*, uid: str, split: str, near_delay: Duration, far_delay: Duration, near_reward: int,
                multiplier: int, near_first: bool, labels: tuple[str, str], horizon: Optional[Duration],
                grid_value: Optional[Duration], phrasing_id: Optional[int], pair_id: Optional[str] = None,
                notes: Optional[dict] = None) -> PromptRecord:
    """Render one prompt and derive every fact about it (regime, correct label, ties) from its parameters."""
    near = Option(reward=near_reward, delay=str(near_delay), delay_years=near_delay.years)
    far = Option(reward=near_reward * multiplier, delay=str(far_delay), delay_years=far_delay.years)
    reg = regime(horizon, near_delay, far_delay)
    correct = CORRECT_OPTION.get(reg)
    record = PromptRecord(
        uid=uid, tier="T0", split=split,
        text=render(near=near, far=far, near_first=near_first, labels=labels, horizon=horizon,
                    phrasing_id=phrasing_id),
        horizon=None if horizon is None else str(horizon),
        horizon_grid_value=None if grid_value is None else str(grid_value),
        horizon_unit=None if horizon is None else horizon.unit,
        horizon_years=None if horizon is None else horizon.years,
        rewritten=horizon is not None and horizon != grid_value,
        phrasing_id=phrasing_id,
        near=near, far=far, reward_multiplier=multiplier, near_first=near_first, labels=labels,
        regime=reg, correct_option=correct, correct_label=None,
        horizon_equals_a_delay=horizon is not None and horizon.days in (near_delay.days, far_delay.days),
        h_over_far_delay=None if horizon is None else float(horizon.days / far_delay.days),
        pair_id=pair_id, notes=notes or {},
    )
    if correct is not None:
        record.correct_label = record.near_label if correct == "near" else record.far_label
    return record


# ---------------------------------------------------------------------------------------------------------------
# Extraction bank
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractionConfig:
    seed: int = 0
    n_with_horizon: int = 471          # report A.1: 471 state a horizon ...
    n_null: int = 29                   # ... and 29 don't
    null_twins: bool = True            # True: each null prompt is a copy of a horizon prompt minus its CONSTRAINT
                                       # line (report: "the same prompt with the CONSTRAINT line removed").
                                       # False: null prompts are independent draws.
    rewrite_tolerance: float = REWRITE_TOLERANCE
    rewrite_max_value: int = REWRITE_MAX_VALUE


def eligible_renderings(grid_value: Duration, near_delay: Duration, far_delay: Duration,
                        cfg: ExtractionConfig) -> list[Duration]:
    """The ways this prompt's horizon may be written: its natural form plus every unit rewrite that keeps the
    prompt in the same regime. (Rewrites differ by up to 2%, so near a delay a rewrite can cross it: "3 months"
    ties a 3-month delay and pays, while "13 weeks" is 0.3% shorter and does not. Such rewrites would change the
    correct answer, so they are excluded for that prompt.)"""
    base = regime(grid_value, near_delay, far_delay)
    rewrites = unit_rewrites(grid_value, cfg.rewrite_tolerance, cfg.rewrite_max_value)
    return [grid_value] + [r for r in rewrites if regime(r, near_delay, far_delay) == base]


def build_extraction_bank(cfg: ExtractionConfig = ExtractionConfig()) -> list[PromptRecord]:
    """The extraction bank, in a fixed order: horizon prompts x0001.., each followed by its null twin if it has
    one. Random draws, in order:

    1. Horizon grid values, balanced: every grid value appears floor(n/48) times and a random subset one
       more time, then the list is shuffled (so per-horizon averages have equal support).
    2. Per horizon prompt: near delay, far delay, near reward, reward multiplier, option order (50/50),
       CONSTRAINT phrasing (uniform over 4), label style (uniform over 2), and the written form of the horizon
       (uniform over `eligible_renderings`); redrawn if the prompt repeats an earlier one exactly.
    3. Which prompts get a null twin (or, with null_twins=False, the parameters of independent null prompts),
       skipping sources whose null text would repeat an earlier one.
    """
    rng = random.Random(cfg.seed)

    # 1. balanced horizon assignment
    reps, extra = divmod(cfg.n_with_horizon, len(HORIZON_GRID))
    grid_values = list(HORIZON_GRID) * reps + rng.sample(HORIZON_GRID, extra)
    rng.shuffle(grid_values)

    # 2. one prompt per horizon. About 10 prompts share each grid value, drawn from 16,800 combinations of the
    # other parameters, so an exact repeat of an earlier prompt happens in roughly one seed in seven; a repeat
    # is redrawn (all of step 2's draws again, for that prompt).
    records: list[PromptRecord] = []
    seen_texts: set[str] = set()
    for i, grid_value in enumerate(grid_values, start=1):
        while True:
            near_delay, far_delay = rng.choice(NEAR_DELAYS), rng.choice(FAR_DELAYS)
            near_reward, multiplier = rng.choice(NEAR_REWARDS), rng.choice(REWARD_MULTIPLIERS)
            near_first = rng.random() < 0.5
            phrasing_id = rng.randrange(len(CONSTRAINT_PHRASINGS))
            labels = rng.choice(LABEL_STYLES)
            horizon = rng.choice(eligible_renderings(grid_value, near_delay, far_delay, cfg))
            record = make_record(
                uid=f"x{i:04d}", split="extraction", near_delay=near_delay, far_delay=far_delay,
                near_reward=near_reward, multiplier=multiplier, near_first=near_first, labels=labels,
                horizon=horizon, grid_value=grid_value, phrasing_id=phrasing_id)
            if record.text not in seen_texts:
                break
        seen_texts.add(record.text)
        records.append(record)

    # 3. null controls. Removing the CONSTRAINT line removes the horizon and the phrasing, so two different
    # horizon prompts can have the same null text (about a 10% chance somewhere among 29 twins); sources are
    # taken in random order, skipping any whose null text is already in the bank.
    if cfg.null_twins:
        nulls: dict[int, PromptRecord] = {}
        seen_texts = {r.text for r in records}
        order = list(range(len(records)))
        rng.shuffle(order)
        for k in order:
            if len(nulls) == cfg.n_null:
                break
            src = records[k]
            twin = make_record(
                uid=f"{src.uid}-null", split="extraction",
                near_delay=Duration.parse(src.near.delay), far_delay=Duration.parse(src.far.delay),
                near_reward=src.near.reward, multiplier=src.reward_multiplier, near_first=src.near_first,
                labels=src.labels, horizon=None, grid_value=None, phrasing_id=None, pair_id=f"{src.uid}+null")
            if twin.text in seen_texts:
                continue
            seen_texts.add(twin.text)
            src.pair_id = twin.pair_id
            nulls[k] = twin
        records = [r for k, src in enumerate(records) for r in ([src, nulls[k]] if k in nulls else [src])]
    else:
        for j in range(1, cfg.n_null + 1):
            records.append(make_record(
                uid=f"n{j:04d}", split="extraction", near_delay=rng.choice(NEAR_DELAYS),
                far_delay=rng.choice(FAR_DELAYS), near_reward=rng.choice(NEAR_REWARDS),
                multiplier=rng.choice(REWARD_MULTIPLIERS), near_first=rng.random() < 0.5,
                labels=rng.choice(LABEL_STYLES), horizon=None, grid_value=None, phrasing_id=None))
    require_distinct_texts(records)
    return records


def require_distinct_texts(records: list[PromptRecord]) -> None:
    """Every prompt in a set must be a different text; a duplicate would silently double-weight one prompt."""
    seen: dict[str, str] = {}
    for r in records:
        if r.text in seen:
            raise ValueError(f"{r.uid} has the same text as {seen[r.text]}; change the seed or the builder")
        seen[r.text] = r.uid


# ---------------------------------------------------------------------------------------------------------------
# Verification grid
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class VerificationConfig:
    seed: int = 0
    n_items: int = 100                 # each item is presented in both option orders: 2 * n_items prompts
    ratio_span: float = 2.8            # H/D runs over [1/span, span], log-spaced
    decimals: int = 2                  # the horizon is written in years with at most this many decimals


def ratio_grid(n: int, span: float) -> list[float]:
    """`n` values of H/D, evenly spaced in log from 1/span to span. With `n` even the grid is symmetric about 1
    and never contains 1 itself: for n=100, span=2.8 the two points nearest the boundary are 0.990 and 1.010,
    1% from it. Rounding H to two decimals moves it by at most 0.005 years, which near the boundary (H close to
    D, and D >= 2 years) is at most 0.25%, so rounding cannot create a tie with D or move H across it."""
    if n % 2:
        raise ValueError("use an even number of items so that H/D = 1 is not on the grid")
    return [span ** (2 * i / (n - 1) - 1) for i in range(n)]


def horizon_in_years(years: float, decimals: int) -> Duration:
    """`years` written in years, rounded to `decimals` decimals ("17.86 years", "140 years")."""
    return Duration(Fraction(round(years * 10 ** decimals), 10 ** decimals), "years")


def build_verification_grid(cfg: VerificationConfig = VerificationConfig()) -> list[PromptRecord]:
    """The verification grid: item v001 has the smallest H/D, v100 the largest; each item yields two prompts,
    "<item>-near-first" and "<item>-far-first", identical except for the option order.

    The random stream is separate from the extraction bank's (seeded with "<seed>-verification"), so changing
    one set's size never changes the other. Draws, in order:

    1. Far delay D per item, balanced over the six far delays, shuffled against the ratio grid.
    2. Per item: near delay (uniform over those strictly shorter than H, so the near option always pays and
       never ties), near reward, reward multiplier, CONSTRAINT phrasing, label style.
    """
    rng = random.Random(f"{cfg.seed}-verification")
    ratios = ratio_grid(cfg.n_items, cfg.ratio_span)

    reps, extra = divmod(cfg.n_items, len(FAR_DELAYS))
    far_delays = list(FAR_DELAYS) * reps + rng.sample(FAR_DELAYS, extra)
    rng.shuffle(far_delays)

    records: list[PromptRecord] = []
    for i, (ratio, far_delay) in enumerate(zip(ratios, far_delays), start=1):
        horizon = horizon_in_years(ratio * far_delay.years, cfg.decimals)
        if (horizon.days < far_delay.days) != (ratio < 1) or horizon.days == far_delay.days:
            raise AssertionError(f"rounding moved H={horizon} across D={far_delay} (target ratio {ratio:.4f})")
        near_delay = rng.choice([d for d in NEAR_DELAYS if d.days < horizon.days])
        near_reward, multiplier = rng.choice(NEAR_REWARDS), rng.choice(REWARD_MULTIPLIERS)
        phrasing_id = rng.randrange(len(CONSTRAINT_PHRASINGS))
        labels = rng.choice(LABEL_STYLES)
        item = f"v{i:03d}"
        for near_first in (True, False):
            records.append(make_record(
                uid=f"{item}-{'near' if near_first else 'far'}-first", split="verification",
                near_delay=near_delay, far_delay=far_delay, near_reward=near_reward, multiplier=multiplier,
                near_first=near_first, labels=labels, horizon=horizon, grid_value=horizon,
                phrasing_id=phrasing_id, pair_id=item, notes={"target_h_over_d": ratio}))
    require_distinct_texts(records)
    return records
