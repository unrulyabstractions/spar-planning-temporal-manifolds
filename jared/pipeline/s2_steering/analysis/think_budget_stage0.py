"""Stage 0 of 28_spec_choice4_side.md (CPU, saved artifacts only): what the force-closed traces of a
k-option bank say before any of them is continued, and the inputs the box run needs.

    python -m pipeline.s2_steering.analysis.think_budget_stage0 --bank choice4
    python -m pipeline.s2_steering.analysis.think_budget_stage0 --bank choice8

Reads results/horizon/exp0_Qwen3-8B_think-on_<bank>/answers.json and the generation cache next to it.
For every prompt whose think block was cut at the budget, it takes the prompt token ids and the exact
generated think token ids from the cache (the cache entry is found by re-tokenising the prompt the way
the run did; if an entry is missing, the think text is re-tokenised and the row says so), and records
  loop       the last 4,000 characters of the trace contain a 150-character block three or more times
  tentative  the last option the trace commits to in its final 1,500 characters ("so the best is b")
  orig_label the label the forced answer step gave
Writes to steering/exp17_Qwen3-8B_think_budget/ (small enough for `just push`):
  stage0_<bank>.json / .npz        the box inputs (pipeline/s2_steering/experiments/think_budget/exp17_think_budget.py reads them)
  predictions_<bank>.{json,txt}    per-trace predictions, written before any continuation runs
  tentative_check_<bank>.txt       20 seeded random traces with the detected label, for a hand check
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from spar_horizon.config import Settings
from spar_horizon.model_io import cache_load, chat_text

JARED = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = Path("/home/Jared/SPAR/results/horizon")
OUT_DIR = JARED / "steering" / "exp17_Qwen3-8B_think_budget"
MODEL = "Qwen/Qwen3-8B"

LOOP_WINDOW = 4000      # characters at the end of the think text
LOOP_BLOCK = 150        # characters of a block that repeats
LOOP_COUNT = 3
TENTATIVE_WINDOW = 1500
CHECK_N = 20
CHECK_SEED = 0

_VERBS = (r"(?:best(?: option| choice| answer)? is|(?:better|right|correct|optimal) (?:option|choice|answer) is|"
          r"(?:answer|choice|option|pick|selection) (?:is|would be|should be)|"
          r"go(?:ing)? with|choose|choosing|pick|picking|select|selecting|opt for|stick with|settle on)")
_AFTER = r"\s+(?:is\s+)?(?P<opt>(?:option|choice)\s+)?(?P<pre>[*(]*)(?P<label>[a-h])\b(?P<close>\))?"
_COMMIT = re.compile(_VERBS + _AFTER, re.IGNORECASE)
_SAYS_BEST = re.compile(r"(?P<opt>(?:option|choice)\s+)(?P<pre>[*(]*)(?P<label>[a-h])(?P<close>\))?\**\s+"
                        r"(?:is|would be|seems|looks)\s+(?:the\s+)?(?:best|optimal|right|correct)\b", re.IGNORECASE)


_CONDITIONAL = re.compile(r"\b(?:if|when|suppose|unless|whether)\s+(?:\w+\s+){0,3}$", re.IGNORECASE)


def has_loop(text):
    """True if some 150-character block occurs three or more times (not overlapping) in the last
    4,000 characters of `text`. The threshold is fixed before any continuation is run."""
    window = text[-LOOP_WINDOW:]
    return any(window.count(window[i:i + LOOP_BLOCK]) >= LOOP_COUNT
               for i in range(len(window) - LOOP_BLOCK + 1))


def tentative_matches(text, labels):
    """[(end position in text, label)] of every commitment phrase in the final 1,500 characters. The
    letter 'a' is also an article, so it counts only with 'option'/'choice' before it, a ')' after it, or
    markdown around it. A phrase right after "if", "when" or "suppose" is a hypothesis, not a commitment."""
    offset = max(0, len(text) - TENTATIVE_WINDOW)
    window = text[offset:]
    found = []
    for pattern in (_COMMIT, _SAYS_BEST):
        for m in pattern.finditer(window):
            label = m["label"].lower()
            if label not in labels:
                continue
            if label == "a" and not (m["opt"] or m["close"] or m["pre"]):
                continue
            if window[m.end():].lstrip().startswith("than") or _CONDITIONAL.search(window[:m.start()]):
                continue
            found.append((offset + m.end(), label))
    return sorted(found)


def last_committed_label(text, labels):
    """The last option label the trace commits to near its end, or None."""
    found = tentative_matches(text, labels)
    return found[-1][1] if found else None


def late_positions(k):
    """Indices of the displayed positions in the second half of the options."""
    return list(range(k // 2, k))


def position_stats(rows, picks):
    """Pick rate and truth rate per displayed position over `rows`, and excess-late: the sum over the
    second-half positions of (pick rate - truth rate). `picks` is the chosen label per row (None skipped)."""
    pairs = [(r, p) for r, p in zip(rows, picks) if p in r["labels"]]
    if not pairs:
        return {"n": 0, "pick_rate": None, "truth_rate": None, "late_pick": None, "excess_late": None}
    k = len(pairs[0][0]["labels"])
    pick = np.bincount([r["labels"].index(p) for r, p in pairs], minlength=k) / len(pairs)
    truth = np.bincount([r["labels"].index(r["truth_label"]) for r, _ in pairs], minlength=k) / len(pairs)
    late = late_positions(k)
    return {"n": len(pairs), "pick_rate": pick.tolist(), "truth_rate": truth.tolist(),
            "late_pick": float(pick[late].sum()), "excess_late": float((pick[late] - truth[late]).sum())}


def option_scores(row, label):
    """(exact, correct half, best option of the chosen half) for a pick; None for an unparsed label."""
    if label not in row["option_info"]:
        return None
    half, _ = row["option_info"][label]
    best = next(lab for lab, (h, rank) in row["option_info"].items() if h == half and rank == 1)
    return label == row["truth_label"], half == row["truth"], label == best


def accuracy(rows, picks):
    """[exact, half, within-half] hit counts and the parsed count."""
    scores = [s for r, p in zip(rows, picks) if (s := option_scores(r, p)) is not None]
    return {"n": len(scores), **{key: int(sum(s[j] for s in scores)) for j, key in enumerate(("exact", "half", "within_half"))}}


ROW_FIELDS = ("sample_uid", "pair_id", "regime", "truth", "truth_label", "labels", "near_labels", "far_labels",
              "n_options", "duration_id", "delay_scale", "rendered_unit")


def compact(answer):
    """The fields of an answers.json row that scoring needs."""
    return {**{f: answer[f] for f in ROW_FIELDS},
            "option_info": {o["label"]: [o["half"], o["rank_in_half"]] for o in answer["options"]}}


def forced_row(index, answer, think_ids, source):
    """The stage 0 record of one force-closed bank row."""
    text = answer["think_text"]
    return {"bank_index": index, **compact(answer),
            "orig_label": answer["label"], "orig_think_len": answer["think_len"], "ids_source": source,
            "loop": has_loop(text), "tentative": last_committed_label(text, answer["labels"]),
            "n_think_ids": len(think_ids)}


def trace_ids(tokenizer, cfg, answer):
    """(prompt ids, think ids, source): from the generation cache when the run's entry exists, else the
    think text re-tokenised."""
    prompt_ids = tokenizer.encode(chat_text(tokenizer, answer["prompt"], cfg), add_special_tokens=False)
    gen = cache_load(cfg, prompt_ids, None, "cpu")
    if gen is None:
        return prompt_ids, tokenizer.encode(answer["think_text"], add_special_tokens=False), "retokenised"
    think = gen.ids[gen.prompt_len:gen.prompt_len + gen.think_len].tolist()
    if gen.prompt_len != len(prompt_ids) or tokenizer.decode(think) != answer["think_text"]:
        raise ValueError(f"cache entry of {answer['sample_uid']} does not match answers.json")
    return prompt_ids, think, "cache"


def concat(arrays):
    return (np.concatenate([np.asarray(a, np.int32) for a in arrays]),
            np.r_[0, np.cumsum([len(a) for a in arrays])].astype(np.int64))


def reference(answers):
    """Forced and natural closes of the whole bank, for the summary's comparisons."""
    out = {}
    for name, flag in (("forced", True), ("natural", False)):
        chosen = [a for a in answers if a["forced"] is flag]
        rows, picks = [compact(a) for a in chosen], [a["label"] for a in chosen]
        out[name] = {"accuracy": accuracy(rows, picks), "position": position_stats(rows, picks)}
    return out


def build(answers, tokenizer, cfg):
    """(stage 0 json record, prompt ids, think ids) for one bank's answers.json rows."""
    rows, prompts, thinks = [], [], []
    for i, a in enumerate(answers):
        if not a["forced"]:
            continue
        prompt_ids, think_ids, source = trace_ids(tokenizer, cfg, a)
        rows.append(forced_row(i, a, think_ids, source))
        rows[-1]["prompt"] = a["prompt"]
        prompts.append(prompt_ids)
        thinks.append(think_ids)
    record = {"bank": cfg.bank, "model": cfg.model, "settings": {**cfg.to_dict(), "cache_dir": None},
              "think_cap": cfg.max_think_tokens, "n_bank": len(answers), "rows": rows, "reference": reference(answers)}
    return record, prompts, thinks


def predictions(record):
    """What the stage 0 text audit predicts for the continuation, per trace and in total. A trace flagged as
    looping is predicted not to close; a trace with a committed label is predicted to end on it."""
    rows = record["rows"]
    per_trace = [{"sample_uid": r["sample_uid"], "loop": r["loop"], "predicted_close": not r["loop"],
                  "predicted_label": r["tentative"] or r["orig_label"], "orig_label": r["orig_label"]}
                 for r in rows]
    with_tentative = [r for r in rows if r["tentative"]]
    agree = [r["tentative"] == r["orig_label"] for r in with_tentative]
    total = {"n": len(rows), "n_loop": sum(r["loop"] for r in rows),
             "predicted_close_rate": float(np.mean([not r["loop"] for r in rows])) if rows else None,
             "n_with_tentative": len(with_tentative),
             "forced_label_agrees_with_tentative": [int(sum(agree)), len(agree)],
             "tentative_is_truth": [sum(r["tentative"] == r["truth_label"] for r in with_tentative), len(with_tentative)],
             "orig_is_truth": [sum(r["orig_label"] == r["truth_label"] for r in rows), len(rows)]}
    return {"bank": record["bank"], "written": "before any continuation was run", "total": total, "per_trace": per_trace}


def predictions_text(pred, record):
    t = pred["total"]
    ref = record["reference"]
    lines = [f"Stage 0 predictions for {pred['bank']} (28_spec_choice4_side.md), written before any box run",
             f"force-closed traces: {t['n']} of {record['n_bank']} prompts, cap {record['think_cap']} think tokens",
             f"token ids from: {dict(sorted(count_sources(record).items()))}",
             f"loop flag (150-char block x3 in last 4,000 chars): {t['n_loop']}/{t['n']}",
             f"predicted close within the extension (= not loop): {t['predicted_close_rate']:.2f}",
             f"traces with a committed label in the last 1,500 chars: {t['n_with_tentative']}/{t['n']}",
             f"  forced label == committed label: {t['forced_label_agrees_with_tentative'][0]}/{t['forced_label_agrees_with_tentative'][1]}",
             f"  committed label is the truth: {t['tentative_is_truth'][0]}/{t['tentative_is_truth'][1]}",
             f"forced label is the truth: {t['orig_is_truth'][0]}/{t['orig_is_truth'][1]}",
             "if the cut causes the errors, a closed continuation ends on its committed label",
             "", "reference (all rows of the bank):"]
    for name in ("forced", "natural"):
        acc, pos = ref[name]["accuracy"], ref[name]["position"]
        lines.append(f"  {name}: n {acc['n']}, exact {acc['exact']}, half {acc['half']}, within-half {acc['within_half']}"
                     + (f", pick {fmt_rates(pos['pick_rate'])}, truth {fmt_rates(pos['truth_rate'])}, "
                        f"excess-late {pos['excess_late']:+.2f}" if pos["n"] else ""))
    return "\n".join(lines) + "\n"


def fmt_rates(rates):
    return " ".join(f"{r:.2f}" for r in rates)


def count_sources(record):
    counts = {}
    for r in record["rows"]:
        counts[r["ids_source"]] = counts.get(r["ids_source"], 0) + 1
    return counts


def tentative_check_text(answers, record):
    """CHECK_N seeded random forced traces: the detected label with the text around the last commitment."""
    by_uid = {a["sample_uid"]: a for a in answers}
    rng = np.random.default_rng(CHECK_SEED)
    chosen = sorted(rng.choice(len(record["rows"]), size=min(CHECK_N, len(record["rows"])), replace=False))
    out = []
    for j in chosen:
        r = record["rows"][j]
        text = by_uid[r["sample_uid"]]["think_text"]
        matches = tentative_matches(text, r["labels"])
        out.append(f"== {r['sample_uid']}  detected: {r['tentative']}  forced label: {r['orig_label']}  "
                   f"truth: {r['truth_label']}  matches in window: {len(matches)}")
        if matches:
            end = matches[-1][0]
            out.append("..." + text[max(0, end - 200):end + 80].replace("\n", " ") + "...")
        else:
            out.append("(no commitment found) final text: ..." + text[-300:].replace("\n", " "))
        out.append("")
    return "\n".join(out)


def run(bank, results, out_dir, cache_dir=None):
    from transformers import AutoTokenizer
    run_dir = Path(results) / f"exp0_Qwen3-8B_think-on_{bank}"
    answers = json.loads((run_dir / "answers.json").read_text())
    settings = json.loads((run_dir / "manifest.json").read_text())["settings"]
    cfg = Settings(**{**settings, "cache_dir": str(cache_dir or Path(results) / "gen_cache")})
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    record, prompts, thinks = build(answers, tokenizer, cfg)
    pred = predictions(record)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_ids, prompt_offsets = concat(prompts)
    think_ids, think_offsets = concat(thinks)
    np.savez(out_dir / f"stage0_{bank}.npz", prompt_ids=prompt_ids, prompt_offsets=prompt_offsets,
             think_ids=think_ids, think_offsets=think_offsets)
    (out_dir / f"stage0_{bank}.json").write_text(json.dumps(record, indent=1))
    (out_dir / f"predictions_{bank}.json").write_text(json.dumps(pred, indent=1))
    text = predictions_text(pred, record)
    (out_dir / f"predictions_{bank}.txt").write_text(text)
    (out_dir / f"tentative_check_{bank}.txt").write_text(tentative_check_text(answers, record))
    print(text)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bank", choices=("choice4", "choice8"), required=True)
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="the results/horizon folder with the E0 run and gen_cache")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--cache-dir", default=None, help="generation cache (default: <results>/gen_cache)")
    args = parser.parse_args()
    run(args.bank, args.results, Path(args.out), args.cache_dir)
