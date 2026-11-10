"""Validate variants.yaml and print what it produces (run after editing it; no model needed).

  python scripts/check_variants.py [--show VARIANT]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phr.items import build_choice, build_state, check_config, load_config  # noqa: E402
from phr.multiturn import build_specs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--show", help="print one example prompt of this variant (e.g. canonical, named, implicit:lease)")
a = ap.parse_args()
cfg = load_config()
errs = check_config(cfg)
if errs:
    sys.exit("PROBLEMS:\n  " + "\n  ".join(errs))
ch, st, mt = build_choice(cfg), build_state(cfg), build_specs(cfg)
print(f"OK. choice {len(ch)} items, state {len(st)} items, multiturn {len(mt)} conversations "
      f"({sum(s.branch_turn is None for s in mt)} full, {sum(s.branch_turn is not None for s in mt)} branched)")
print(ch.groupby("family").variant.agg(["nunique", "size"]).rename(columns={"nunique": "variants", "size": "items"}).to_string())
if a.show:
    print("\n" + ch[ch.variant == a.show].text.iloc[0])
