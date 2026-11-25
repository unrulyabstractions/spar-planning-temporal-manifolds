"""Export the eval-type banks (spar_horizon/eval_types, EvalsPromptTypes.md) as prompt_bank banks, so the team's
evaluation runner (`evaluation/transfer_eval.py` on the eval-types-stage branch) can replay a steering method on them.

    python -m pipeline.s1_data_vectors.analysis.export_evaltype_banks              # EXPORTED types, n_base 100
    python -m pipeline.s1_data_vectors.analysis.export_evaltype_banks --types t0_w0 coding_agent --n-base 0

Run from jared/. Writes `<out>/<type_id>/{prompts.jsonl, manifest.json}`, one bank per type. Default out: ~/SPAR/results/horizon/evaltype_banks.

One bank per type: the evaluation runs, its stage skipping and the run hash (baseline runs refuse to resume on a bank
whose prompts changed) then work per type, so editing one type invalidates only that type's runs, and any subset of
types can be run.

Each row is a `pbank.records.PromptRecord` with exactly its fields (pbank's loader takes no others):
  uid             the record's sample_uid (unique across types)
  tier, split     "evaltype", "transfer": no fitting code selects them (fitting and the shared draws take only
                  split "extraction", the T0 replay only "verification")
  text            the prompt; for a conversation-rendered record (system message, earlier turns) a role-tagged
                  transcript for reading and hashing only: the model input is notes["messages"]
  horizon, horizon_grid_value
                  the horizon the truth is computed from, as the explicit twin writes it (horizon_text); the
                  prompt may state it otherwise (unit_mismatch) or only imply it (implicit_*): see notes
  horizon_unit, horizon_years   its unit word and value in years
  rewritten, phrasing_id        False, None (T0 notions; a type's own phrasing id stays in notes)
  near, far       reward (the option's value, None when the prompt gives none), delay (as written), delay_years
  reward_multiplier             far value / near value (None without values)
  near_first, labels            from near_label; labels as written in the prompt ("a)", "b)")
  regime          near -> near_only, both -> both, none -> none
  correct_option, correct_label from the record's truth / truth_labels (None where the type defines no truth)
  horizon_equals_a_delay, h_over_far_delay
  pair_id         the twin_id for the base and swap records (the same item in both option orders, as the
                  verification grid's pairs); the sample_uid for an explicit twin, which has no order partner
  notes           the eval-type record unchanged except its prompt (type_id, variant, twin_id, regime, truth_rule,
                  truth_labels, options, choice_prefix, answer_kind and the type's own fields), plus
                  "messages" for a conversation
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from spar_horizon.eval_types import TYPES, build
from spar_horizon.eval_types.common import EXPANDED_N_BASE
from spar_horizon.runs import git_hash

# The types gated for the expanded banks (GATED in tests/test_eval_types_expanded.py).
EXPORTED = ("system_prompt", "city_no_values", "distractor_temporal", "t0_w0", "t0_w3_clear", "sycophancy_lean",
            "t0_w1_cash", "implicit_vague", "city_values", "implicit_concrete", "role_assistant", "prose_natural",
            "multiturn_analyst", "system_prompt_clear", "unit_mismatch", "role_fund_manager", "public_health",
            "horizon_revised_clear", "coding_agent")
TIER, SPLIT = "evaltype", "transfer"
REGIME = {"near": "near_only", "both": "both", "none": "none"}
DEFAULT_OUT = Path.home() / "SPAR" / "results" / "horizon" / "evaltype_banks"


def transcript(messages):
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)


def to_prompt_record(rec):
    """The eval-type record as a PromptRecord dict (module docstring)."""
    if rec["regime"] not in REGIME:
        raise ValueError(f"{rec['sample_uid']}: regime {rec['regime']!r} has no pbank equivalent")
    if len(rec["labels"]) != 2 or "near_label" not in rec:
        raise ValueError(f"{rec['sample_uid']}: not a two-option near/far record")
    by_kind = {o["kind"]: o for o in rec["options"]}
    near, far = by_kind["near"], by_kind["far"]
    conversation = not isinstance(rec["prompt"], str)
    h_years, h_text = rec["horizon"], rec["horizon_text"]
    has_values = near["value"] is not None and far["value"] is not None
    notes = {k: v for k, v in rec.items() if k != "prompt"}
    if conversation:
        notes["messages"] = rec["prompt"]
    truth = rec.get("truth")
    return {
        "uid": rec["sample_uid"], "tier": TIER, "split": SPLIT,
        "text": transcript(rec["prompt"]) if conversation else rec["prompt"],
        "horizon": h_text, "horizon_grid_value": h_text,
        "horizon_unit": h_text.split()[-1] if h_text else None, "horizon_years": h_years,
        "rewritten": False, "phrasing_id": None,
        "near": {"reward": near["value"], "delay": near["delay_text"], "delay_years": near["delay_years"]},
        "far": {"reward": far["value"], "delay": far["delay_text"], "delay_years": far["delay_years"]},
        "reward_multiplier": far["value"] / near["value"] if has_values else None,
        "near_first": rec["near_label"] == rec["labels"][0],
        "labels": [f"{lab})" for lab in rec["labels"]],
        "regime": REGIME[rec["regime"]],
        "correct_option": truth,
        "correct_label": f"{rec['truth_labels'][0]})" if truth is not None else None,
        "horizon_equals_a_delay": h_years is not None and h_years in (near["delay_years"], far["delay_years"]),
        "h_over_far_delay": h_years / far["delay_years"] if h_years is not None else None,
        "pair_id": rec["twin_id"] if rec["variant"] in ("base", "swap") else rec["sample_uid"],
        "notes": notes,
    }


def prompts_sha256(rows):
    """pbank.records.prompts_sha256: SHA-256 over "uid|text" lines."""
    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r['uid']}|{r['text']}\n".encode())
    return h.hexdigest()


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def export_type(type_id, out, seed, n_base):
    """Write one type's bank; returns its manifest."""
    rows = [to_prompt_record(r) for r in build(type_id, seed=seed, n_base=n_base)]
    d = out / type_id
    d.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "spar_horizon.eval_types", "type_id": type_id, "seed": seed, "n_base": n_base,
        "tier": TIER, "split": SPLIT, "code_commit": git_hash(),
        "conversation": any("messages" in r["notes"] for r in rows),
        "n_prompts": len(rows),
        "counts": {"variant": dict(Counter(r["notes"]["variant"] for r in rows)),
                   "regime": dict(Counter(r["regime"] for r in rows)),
                   "correct_option": dict(Counter(str(r["correct_option"]) for r in rows))},
        "prompts_sha256": prompts_sha256(rows),
    }
    write_atomic(d / "prompts.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_atomic(d / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--types", nargs="+", default=list(EXPORTED), choices=sorted(TYPES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-base", type=int, default=EXPANDED_N_BASE,
                   help="base items per type (default %(default)s, the expanded banks); 0 = the original plan")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args(argv)
    n_base = a.n_base or None
    for type_id in a.types:
        old = a.out / type_id / "manifest.json"
        old_sha = json.loads(old.read_text())["prompts_sha256"] if old.exists() else None
        m = export_type(type_id, a.out, a.seed, n_base)
        status = "new" if old_sha is None else ("unchanged" if old_sha == m["prompts_sha256"] else "CHANGED")
        print(f"{type_id:<24} {m['n_prompts']:>4} prompts  {m['counts']['regime']}"
              f"{'  conversation' if m['conversation'] else ''}  {status}")
    print(f"wrote {len(a.types)} banks to {a.out}")


if __name__ == "__main__":
    main()
