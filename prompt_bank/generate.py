"""Step 1 of 3: build a prompt bank, then have a checker model review every prompt.

    python generate.py --out data/t0_s0                                        # build + check (local Qwen3.5-27B)
    python generate.py --out data/t0_s0 --checker openrouter:anthropic/claude-haiku-5.5   # check with Haiku
    python generate.py --out data/t0_s0 --checker none                         # build only (no GPU, no API)

What it writes (see README.md for every field):

    <out>/prompts.jsonl     the prompts: the extraction bank (500) and the verification grid (200)
    <out>/manifest.json     settings, counts, git commit, and the SHA-256 of the prompts
    <out>/checks/<checker>.jsonl  the checker's review of each prompt (+ .meta.json with its settings)

Building is deterministic: the same settings always give byte-identical prompts. Rerunning on an existing
bank directory is safe: if the prompts would be identical the build step is a no-op, and the check step resumes
where it stopped. If the prompts would differ (new code or new settings), the script refuses and asks for a
new --out directory, because the existing checks, runs and reviews describe the old prompts.

Only tier T0 exists. T1-T3 (realistic prompts) are deliberately not built until steering passes its gate on T0
(report, Appendix A.6, roadmap step 2).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from pbank.checker import checker_file_name, run_checker
from pbank.provenance import git_commit, utc_now
from pbank.records import load_prompts, prompts_sha256, record_to_dict, write_jsonl
from pbank.t0 import ExtractionConfig, VerificationConfig, build_extraction_bank, build_verification_grid

DEFAULT_CHECKER = "local:Qwen/Qwen3.5-27B"


def counts(records) -> dict:
    """Composition of the bank, per split, for the manifest and the console."""
    out = {}
    for split in ("extraction", "verification"):
        rs = [r for r in records if r.split == split]
        out[split] = {
            "prompts": len(rs),
            "regime": dict(Counter(r.regime for r in rs)),
            "horizon_unit": dict(Counter(r.horizon_unit or "none" for r in rs)),
            "rewritten": sum(r.rewritten for r in rs),
            "horizon_equals_a_delay": sum(r.horizon_equals_a_delay for r in rs),
            "phrasing_id": dict(Counter(str(r.phrasing_id) for r in rs)),
            "labels": dict(Counter("/".join(r.labels) for r in rs)),
            "near_first": sum(r.near_first for r in rs),
        }
    return out


def build(out: Path, seed: int) -> list:
    """Build the T0 prompts into `out`, or confirm that identical prompts are already there."""
    extraction_cfg, verification_cfg = ExtractionConfig(seed=seed), VerificationConfig(seed=seed)
    records = build_extraction_bank(extraction_cfg) + build_verification_grid(verification_cfg)
    sha = prompts_sha256(records)

    prompts_path = out / "prompts.jsonl"
    if prompts_path.exists():
        existing = prompts_sha256(load_prompts(out))
        if existing != sha:
            sys.exit(f"{prompts_path} exists with different prompts (sha256 {existing[:12]} vs {sha[:12]} now).\n"
                     f"Its checks, runs and reviews describe those prompts; write the new bank to a new --out.")
        print(f"{prompts_path}: identical prompts already built (sha256 {sha[:12]}); not rewriting")
        return records

    write_jsonl(prompts_path, (record_to_dict(r) for r in records))
    manifest = {
        "tier": "T0",
        "prompts_sha256": sha,
        "created_utc": utc_now(),
        "code_commit": git_commit(),
        "extraction_config": asdict(extraction_cfg),
        "verification_config": asdict(verification_cfg),
        "counts": counts(records),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(records)} prompts to {prompts_path} (sha256 {sha[:12]})")
    print(json.dumps(manifest["counts"], indent=2))
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="bank directory, e.g. data/t0_s0")
    p.add_argument("--tier", default="T0", choices=["T0"], help="only T0 exists until the T0 gate passes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checker", default=DEFAULT_CHECKER,
                   help="'local:<Hugging Face id>', 'openrouter:<OpenRouter id>', or 'none' (default: %(default)s)")
    p.add_argument("--checker-batch-size", type=int, default=8,
                   help="prompts per generate() call (local) or per round of concurrent requests (hosted)")
    p.add_argument("--checker-max-new-tokens", type=int, default=8192)
    p.add_argument("--checker-no-thinking", action="store_true", help="local checker: turn thinking mode off")
    p.add_argument("--reasoning-effort", default="medium", help="hosted checker: reasoning effort (default medium)")
    p.add_argument("--limit", type=int, help="check only the first N prompts (smoke tests)")
    args = p.parse_args()

    records = build(args.out, args.seed)
    if args.checker == "none":
        return
    to_check = records[:args.limit] if args.limit else records
    rows = run_checker(to_check, args.checker, args.out / "checks" / checker_file_name(args.checker),
                       batch_size=args.checker_batch_size, max_new_tokens=args.checker_max_new_tokens,
                       thinking=not args.checker_no_thinking, reasoning_effort=args.reasoning_effort,
                       prompts_sha256=prompts_sha256(records))
    latest = {row["uid"]: row for row in rows}.values()
    print(f"checked {len(latest)}: agree {sum(r['agrees'] is True for r in latest)}, "
          f"disagree {sum(r['agrees'] is False for r in latest)}, unreadable {sum(r['agrees'] is None for r in latest)}, "
          f"flagged a problem {sum(r['verdict'] == 'problem' for r in latest)}")


if __name__ == "__main__":
    main()
