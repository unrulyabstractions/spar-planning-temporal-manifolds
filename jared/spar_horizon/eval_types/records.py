"""Eval-type rows (`<run>/<type>/rows.json`) as the steering records E13/E14 read, and the
donor prompt of a row: the same prompt with its horizon phrase replaced."""

import json
import re
from pathlib import Path

from ..full_bank import HORIZONS, regime
from ..model_io import chat_text
from ..steering import phrase_positions

MARKER = "horizon: "   # the constraint line reads "... for this time horizon: 3 weeks."
LABEL = re.compile(r"\s*([a-z])\)")


def eval_records(rows_path):
    """The rows of one eval type with the steering record fields added: the near and far
    delay in years (from `options`), the label E0's answer starts with (None if it does not
    start with one), and E0's near log-odds from `label_logits` (logit of each label, in the
    order of `labels`). Raises on a row without `near_label`: the caller picks the types."""
    records = []
    for row in json.loads(Path(rows_path).read_text()):
        if row.get("near_label") not in row["labels"]:
            raise ValueError(f"{row['sample_uid']}: no near_label; this type has no near/far rule to steer")
        delays = {option["kind"]: option["delay_years"] for option in row["options"]}
        match = LABEL.match(row["answer"])
        label = match.group(1) if match and match.group(1) in row["labels"] else None
        logits = dict(zip(row["labels"], row["label_logits"]))
        far_label = next(lab for lab in row["labels"] if lab != row["near_label"])
        records.append({**row, "near_delay_years": delays["near"], "far_delay_years": delays["far"], "label": label,
                        "near_lo": logits[row["near_label"]] - logits[far_label]})
    return records


def with_horizon_text(prompt, horizon_text, new_text):
    """`prompt` (a string or a list of {role, content} messages) with "horizon: <horizon_text>"
    replaced by "horizon: <new_text>"; the phrase must occur exactly once."""
    old, new = MARKER + horizon_text, MARKER + new_text
    if isinstance(prompt, str):
        found, replaced = prompt.count(old), prompt.replace(old, new)
    else:
        found = sum(message["content"].count(old) for message in prompt)
        replaced = [{**message, "content": message["content"].replace(old, new)} for message in prompt]
    if found != 1:
        raise ValueError(f"{old!r} occurs {found} times in the prompt, expected once")
    return replaced


def find_donor(record, target_log_h, target_regime, tokenizer, cfg):
    """(horizon text, log10 years, donor prompt) of the grid horizon nearest the target, in the
    target regime for this row's delays, whose phrase has as many tokens as the row's; or None.
    The rule of steering_prepare.find_donor, with the prompt rebuilt by replacing the phrase
    (render_variant cannot rebuild an eval-type prompt)."""
    def phrase_length(prompt, text):
        return len(phrase_positions(tokenizer, chat_text(tokenizer, prompt, cfg), text))

    recipient_length = phrase_length(record["prompt"], record["horizon_text"])
    for horizon in sorted(HORIZONS, key=lambda h: abs(h.log10_years - target_log_h)):
        if regime(horizon.years, record["near_delay_years"], record["far_delay_years"]) != target_regime:
            continue
        prompt = with_horizon_text(record["prompt"], record["horizon_text"], str(horizon))
        if phrase_length(prompt, str(horizon)) == recipient_length:
            return str(horizon), horizon.log10_years, prompt
    return None
