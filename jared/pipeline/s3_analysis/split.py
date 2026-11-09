"""Load the answered prompts of E0 runs into one table of labels, and split it by a label.

A table is a list of dicts, one per answered prompt, with the columns in COLUMNS. It is built
from `answers.json` (written by experiments/exp0_baseline.py) and the run folder's name, so no
model and no activations are needed:

    rows = load_runs(Path("results/horizon"), "Qwen3-8B", "on", ["full", "units", "fill"])
    for regime, group in split_by(rows, "regime").items():
        print(regime, near_share(group))
"""

import re
from collections import defaultdict
from pathlib import Path

from spar_horizon.e0_runs import read_answers, run_folder
from spar_horizon.full_bank import natural_horizon
from spar_horizon.horizons import canonical_unit

COLUMNS = ("regime", "unit", "horizon", "delay", "truth", "n_choices", "forced_close", "bank", "number_format",
           "chose_near")
RUN_NAME = re.compile(r"think-(?:on|off)(?:_(?P<bank>\w+))?$")


def bank_of(run_dir):
    """The bank in an E0 run folder's name (`exp0_Qwen3-8B_think-on_units`); the starter bank has no suffix."""
    return RUN_NAME.search(Path(run_dir).name)["bank"] or "starter"


def number_format(answer):
    """How the horizon is written in the prompt: "digits" or "written" (a number word); None without a horizon.
    Rows of the paired banks carry the phrase as `duration_id`; the others are rendered in their natural unit."""
    if answer["horizon"] is None:
        return None
    phrase = answer.get("duration_id") or str(natural_horizon(answer["horizon"]))
    return "digits" if phrase[0].isdigit() else "written"


def chose_near(answer):
    """Whether the choice is the near option (a/b banks) or in the near half (choice banks); None if unknown."""
    if "near_labels" in answer:
        return answer["label"] in answer["near_labels"]
    if "near_label" in answer:
        return answer["label"] == answer["near_label"]
    return None


def table_row(answer, bank):
    """One table row from an answers.json row that has a label (the run kept it)."""
    return {"regime": answer.get("regime"),
            "unit": canonical_unit(answer["rendered_unit"]) if "rendered_unit" in answer else None,
            "horizon": answer["horizon"],
            "delay": answer.get("delay_scale"),
            "truth": answer.get("truth"),
            "n_choices": answer.get("n_options", 2),
            "forced_close": bool(answer["forced"]),
            "bank": bank,
            "number_format": number_format(answer),
            "chose_near": chose_near(answer)}


def load_run(run_dir):
    """The table of one run folder. Prompts the run did not keep (no label) are left out. (Not
    `e0_runs.load_run_activations`: that one keeps the prompts with activations, by `think_len`.)"""
    return [table_row(a, bank_of(run_dir)) for a in read_answers(run_dir) if "label" in a]


def load_runs(root, model, mode, banks):
    """The tables of `exp0_<model>_think-<mode>_<bank>` for each bank under `root`, concatenated;
    banks whose run folder is missing are skipped."""
    rows = []
    for bank in banks:
        run_dir = run_folder(root, model, mode, bank)
        if (run_dir / "answers.json").exists():
            rows += load_run(run_dir)
    return rows


def split_by(rows, column):
    """{value of `column`: rows}, in order of first appearance. Rows without a value (None) are kept
    under None, so a split never silently drops prompts."""
    groups = defaultdict(list)
    for row in rows:
        groups[row[column]].append(row)
    return dict(groups)


def near_share(rows):
    """Fraction of rows that chose the near option; NaN for no rows. Rows with an unknown choice
    (neither `near_label` nor `near_labels`) are not counted."""
    known = [r["chose_near"] for r in rows if r["chose_near"] is not None]
    return sum(known) / len(known) if known else float("nan")
