"""Step 2 of 3: run a model, unsteered, on every prompt of a bank, to characterize its baseline behaviour.

    python baseline.py --bank data/t0_s0                                   # Gemma 4 31B, thinking on
    python baseline.py --bank data/t0_s0 --model Qwen/Qwen3-32B --batch-size 8
    python baseline.py --bank data/t0_s0 --model Qwen/Qwen3.5-2B --limit 8   # local smoke test

Settings follow the report (Appendix A.6, "Models"): thinking on, greedy decoding, a token cap (8,192 by
default). The defaults reproduce the 2026-10-17 Gemma pilot's measured setup: batch 4 on one A100 80 GB
(about 13 s per prompt; batch 4 keeps a runaway trace from running out of memory).

What it writes, in runs/<model>__<bank>/ unless --out is given (see README.md for every field):

    run.json            model, software, settings, the bank's SHA-256, git commit, start/end times
    responses.jsonl     one row per prompt: thinking, answer, parsed choice, correctness, token counts, and the
                        generated token ids (which ../steering/extract.py replays to read activations)
    summary.json        accuracy, format failures and truncation per split and regime

Resumable: rows are appended and flushed after every batch, and a rerun of the same command skips prompts
already answered. A rerun with different settings, or against a different bank, is refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from pbank.answers import parse_choice, score
from pbank.local_model import LocalChatModel, log
from pbank.provenance import PACKAGE_DIR, git_commit, utc_now
from pbank.records import append_jsonl, load_prompts, prompts_sha256, read_jsonl

DEFAULT_MODEL = "google/gemma-4-31B-it"


def summarize(records, rows: list[dict]) -> dict:
    """Per split and regime: how many prompts, how many answered in format, correct, and truncated."""
    by_uid = {r.uid: r for r in records}
    table: dict = defaultdict(lambda: defaultdict(int))
    for row in rows:
        rec = by_uid[row["uid"]]
        cell = table[f"{rec.split}/{rec.regime}"]
        cell["prompts"] += 1
        cell["choice_parsed"] += row["parse_status"] == "ok"
        cell["format_failure"] += row["parse_status"] in ("format_failure", "conflicting")
        cell["thinking_never_closed"] += row["parse_status"] == "no_answer"
        cell["hit_token_cap"] += row["hit_token_cap"]
        cell["chose_near"] += row["chosen_option"] == "near"
        cell["chose_far"] += row["chosen_option"] == "far"
        if rec.correct_option is not None:
            cell["scored"] += 1
            cell["correct"] += row["correct"] is True
    return {k: dict(v) for k, v in sorted(table.items())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True, help="bank directory written by generate.py")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id (default: %(default)s)")
    p.add_argument("--out", type=Path, help="run directory (default: runs/<model>__<bank>)")
    p.add_argument("--split", default="all", choices=["all", "extraction", "verification"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--no-thinking", action="store_true", help="turn thinking mode off (default: on)")
    p.add_argument("--limit", type=int, help="only the first N prompts (smoke tests)")
    args = p.parse_args()

    records = load_prompts(args.bank)
    if not records:
        sys.exit(f"no prompts in {args.bank}; run generate.py first")
    bank_sha = prompts_sha256(records)
    selected = [r for r in records if args.split == "all" or r.split == args.split][:args.limit]

    out = args.out or PACKAGE_DIR / "runs" / f"{args.model.split('/')[-1]}__{args.bank.name}"
    # Everything that changes what the model generates. A resumed run must match these exactly. (The bank is
    # identified by its hash, not its path, so a run can be resumed from another directory or machine.)
    settings = {"model": args.model, "bank_prompts_sha256": bank_sha, "thinking": not args.no_thinking,
                "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size, "decoding": "greedy"}
    run_json = out / "run.json"
    if run_json.exists():
        previous = json.loads(run_json.read_text())["settings"]
        if previous != settings:
            sys.exit(f"{out} holds a run with different settings:\n  {previous}\nvs now\n  {settings}\n"
                     f"Use another --out, or the same settings to resume.")

    responses = out / "responses.jsonl"
    done = {row["uid"] for row in read_jsonl(responses)}
    todo = [r for r in selected if r.uid not in done]
    log(f"{args.model} on {args.bank}: {len(selected)} selected, {len(done)} already answered, {len(todo)} to run")
    if todo:
        model = LocalChatModel(args.model)
        out.mkdir(parents=True, exist_ok=True)
        meta = json.loads(run_json.read_text()) if run_json.exists() else {"started_utc": utc_now()}
        meta.update(settings=settings, bank_name=args.bank.name, model=model.describe(), code_commit=git_commit())
        run_json.write_text(json.dumps(meta, indent=2) + "\n")

        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            replies = model.generate([r.text for r in batch], thinking=not args.no_thinking,
                                     max_new_tokens=args.max_new_tokens)
            rows = []
            for rec, reply in zip(batch, replies):
                parsed = parse_choice(reply.answer, rec.labels)
                s = score(rec, parsed)
                rows.append({"uid": rec.uid, "model": args.model, "thinking": reply.thinking, "answer": reply.answer,
                             "choice_label": parsed.label, "parse_status": parsed.status,
                             "n_choice_mentions": parsed.n_mentions, "chosen_option": s.chosen_option,
                             "correct": s.correct, "n_generated": reply.n_generated,
                             "hit_token_cap": reply.hit_token_cap, "seconds": round(reply.seconds, 2),
                             "generated_ids": reply.generated_ids})
            append_jsonl(responses, rows)
            log(f"  [{len(done) + i + len(batch)}/{len(selected)}] {batch[0].uid}..{batch[-1].uid}: "
                f"{sum(r['parse_status'] == 'ok' for r in rows)}/{len(rows)} parsed, "
                f"max {max(r['n_generated'] for r in rows)} tokens, {replies[0].seconds:.1f} s/prompt")

        meta["finished_utc"] = utc_now()
        run_json.write_text(json.dumps(meta, indent=2) + "\n")

    summary = summarize(records, read_jsonl(responses))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for cell, c in summary.items():
        scored = f", correct {c['correct']}/{c['scored']}" if c.get("scored") else ""
        log(f"{cell:<26} n={c['prompts']:>3}  parsed {c['choice_parsed']:>3}  near {c['chose_near']:>3}  "
            f"far {c['chose_far']:>3}  cap {c['hit_token_cap']:>2}{scored}")


if __name__ == "__main__":
    main()
