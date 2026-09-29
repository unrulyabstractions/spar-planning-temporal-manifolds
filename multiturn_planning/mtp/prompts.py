"""Conversation specs for the multi-turn planning experiment.

One conversation = a fixed system prompt with the format rules (identical for every conversation), a first user
turn (scenario + goal + horizon sentence) followed by
"Continue" turns. The model answers turn 1 with an outline of step titles only (no timings), then one step
per "Continue" in a fixed format, then "Plan Completed". Every scenario is used at every H_target
(content-matched horizons), in three wordings; the no-horizon condition drops the horizon sentence.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import asdict, dataclass

N_STEPS = 5
CONTINUE = "Continue"
DONE = "Plan Completed"

# 12 goals chosen to make sense at every horizon from a week to fifty years.
SCENARIOS = {
    "bakery": "I run a small bakery and I want to grow the business.",
    "fitness": "I want to become much fitter and healthier.",
    "green_street": "Our neighbourhood association wants to make our street greener.",
    "family_finances": "I want to put my family's finances on a better footing.",
    "research_field": "I'm a researcher and I want to build real expertise in a new field.",
    "homelessness": "Our nonprofit wants to reduce homelessness in our town.",
    "piano": "I want to learn to play the piano well.",
    "company_emissions": "Our company wants to cut its carbon emissions.",
    "school_reading": "Our school wants to improve our students' reading skills.",
    "old_house": "We bought an old house and we want to restore it.",
    "flood_ready": "Our town wants to be better prepared for floods.",
    "open_source": "I maintain an open-source library and I want to grow its community.",
}

# H_target grid (years): 1 week ... 50 years, ~3.4 decades.
HORIZONS = {
    "1 week": 1 / 52, "1 month": 1 / 12, "3 months": 0.25, "1 year": 1.0,
    "3 years": 3.0, "10 years": 10.0, "25 years": 25.0, "50 years": 50.0,
}

WORDINGS = {
    "within": "The plan should achieve this within {h}.",
    "horizon": "The time horizon for this plan is {h}.",
    "from_today": "All of it needs to happen within {h} from today.",
}

SYSTEM = (
    "You are a planning assistant. You write plans one step at a time, following these rules exactly.\n"
    f"1. Your first reply is only an outline: the titles of exactly {N_STEPS} steps, one per line, numbered "
    f"1 to {N_STEPS}. No timings, no details, nothing else.\n"
    f'2. Each time the user says "{CONTINUE}", reply with only the next step (the first "{CONTINUE}" gets '
    "step 1), in exactly this format:\n"
    "Step <n>: <title>\n"
    "Time horizon: <how far from today this step will be complete, as a single duration: a number and a unit>\n"
    "Details: <two or three sentences>\n"
    f'3. You give steps {", ".join(str(i) for i in range(1, N_STEPS))} and {N_STEPS}, one per "{CONTINUE}"; never '
    f'skip a step. Only after step {N_STEPS} has been given, reply to the next "{CONTINUE}" with only "{DONE}".'
)


@dataclass(frozen=True)
class ConversationSpec:
    conv_id: str
    scenario: str
    condition: str           # "horizon" | "none"
    wording: str | None
    h_target_text: str | None
    h_target_years: float    # nan for condition "none"
    sample: int              # sampling seed index (0 for greedy)
    greedy: bool
    first_user: str

    @property
    def log_h_target(self) -> float:
        return math.log10(self.h_target_years) if self.condition == "horizon" else math.nan

    def to_dict(self) -> dict:
        return asdict(self)


def first_user_turn(scenario: str, wording: str | None, h: str | None) -> str:
    goal = SCENARIOS[scenario]
    text = goal + " Help me make a plan."
    if h is not None:
        text += " " + WORDINGS[wording].format(h=h)
    return text


def build_specs(n_none_samples: int = 16) -> list[ConversationSpec]:
    """Horizon condition: every scenario x H_target x wording, greedy (288).
    No-horizon condition: one prompt per scenario, so greedy would give one plan each; instead
    `n_none_samples` sampled plans per scenario (192 at the default)."""
    specs = []
    for sc, (h, yrs), w in itertools.product(SCENARIOS, HORIZONS.items(), WORDINGS):
        cid = f"{sc}__{w}__{h.replace(' ', '')}"
        specs.append(ConversationSpec(cid, sc, "horizon", w, h, yrs, 0, True, first_user_turn(sc, w, h)))
    for sc, s in itertools.product(SCENARIOS, range(n_none_samples)):
        specs.append(ConversationSpec(f"{sc}__none__s{s}", sc, "none", None, None, math.nan, s, False,
                                      first_user_turn(sc, None, None)))
    return specs


# ---- parsing the model's step horizon -------------------------------------------------------------------

UNIT_YEARS = {
    "minute": 1 / 525_960, "hour": 1 / 8_766, "day": 1 / 365.25, "week": 7 / 365.25,
    "fortnight": 14 / 365.25, "month": 1 / 12, "quarter": 0.25, "year": 1.0, "decade": 10.0,
}
WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "half": 0.5,
}
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(sorted(WORD_NUMBERS, key=len, reverse=True)) + r")"
_UNIT = r"(minute|hour|day|week|fortnight|month|quarter|year|decade)s?"
# optional range: "6-12 months", "6 to 12 months", "0–6 months" -> the upper end
_DURATION = re.compile(rf"{_NUM}(?:\s*(?:-|–|—|to)\s*{_NUM})?\s*{_UNIT}\b", re.IGNORECASE)
_HORIZON_LINE = re.compile(r"time\s*horizon\s*\**\s*([:：])\s*\**\s*(.+)", re.IGNORECASE)   # group 1 = colon, 2 = value
_STEP_LINE = re.compile(r"^\W*step\s*(\d+)", re.IGNORECASE | re.MULTILINE)


def _num(s: str) -> float:
    s = s.lower()
    return WORD_NUMBERS[s] if s in WORD_NUMBERS else float(s)


def parse_duration_years(text: str) -> float | None:
    """First duration in `text`, in years; a range gives its upper end."""
    m = _DURATION.search(text)
    if not m:
        return None
    n = _num(m.group(2) or m.group(1))
    return n * UNIT_YEARS[m.group(3).lower()]


def parse_step(reply: str) -> dict:
    """Step number, horizon text and years from one step reply (None where absent)."""
    step = _STEP_LINE.search(reply)
    hline = _HORIZON_LINE.search(reply)
    htext = hline.group(2).strip() if hline else None
    return {"step_no": int(step.group(1)) if step else None, "h_step_text": htext,
            "h_step_years": parse_duration_years(htext) if htext else None}


def outline_mentions_time(reply: str) -> bool:
    """True if the outline already states a duration (then step horizons may be copied from it)."""
    return _DURATION.search(reply) is not None
