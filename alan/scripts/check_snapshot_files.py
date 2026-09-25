#!/usr/bin/env python
"""List every figures/ and runs/ path mentioned in the progress logs and notes and report whether it
exists and was written after a given time. Usage: check_snapshot_files.py --after '2026-09-25 08:46'"""
import argparse, datetime as dt, os, re
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--after", required=True); a = ap.parse_args()
t0 = dt.datetime.fromisoformat(a.after).timestamp()
pat = re.compile(r"`?((?:figures|runs|data/prompts)/[A-Za-z0-9_./\-{},*<>]+)")
seen = {}
for f in list(Path(".").glob("progress-*.md")) + list(Path("notes").glob("*.md")) + [Path("snapshot/README-shared.md")]:
    for m in pat.finditer(f.read_text()):
        p = m.group(1).rstrip(".,;:)`")
        if any(c in p for c in "{}*<>"):   # patterns, not concrete paths
            continue
        seen.setdefault(p, set()).add(f.name)
missing, stale, fresh = [], [], []
for p in sorted(seen):
    # old run names that were renamed in the reproduction
    q = p.replace("qwen3-14b_investment_n2000_s0_v2", "qwen3-14b_investment_n2000_s0")
    if not os.path.exists(q):
        missing.append((p, sorted(seen[p])))
    elif os.path.getmtime(q) < t0 and os.path.isfile(q):
        stale.append(p)
    else:
        fresh.append(p)
print(f"paths cited: {len(seen)}  fresh: {len(fresh)}  stale: {len(stale)}  missing: {len(missing)}")
for p in stale: print("  STALE  ", p)
for p, src in missing: print("  MISSING", p, "  cited in", ",".join(src))
