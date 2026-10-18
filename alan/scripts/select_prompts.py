#!/usr/bin/env python
"""Select a subset of a prompt parquet with a pandas query, keeping rows (and their sample_uids) unchanged so results
can be paired with an earlier run on the full set.

Derived columns available to the query (not written out): delay_unit (unit of the long delay's text), D (long delay in
years, rounded), ratio (long/short reward, rounded), gi (grid index: the integer after the last "_" of sample_uid).

  select_prompts.py IN.parquet OUT.parquet --query "D == 10 and ratio == 5 and phrasing_id == 0 and gi % 2 == 1"
"""
import argparse, json
from pathlib import Path

import pandas as pd

from ptm.horizons import canonical_unit

ap = argparse.ArgumentParser()
ap.add_argument("inp", type=Path); ap.add_argument("out", type=Path)
ap.add_argument("--query", required=True)
a = ap.parse_args()
df = pd.read_parquet(a.inp)
aux = pd.DataFrame(index=df.index)
aux["delay_unit"] = df["long_delay_text"].str.split().str[1].map(canonical_unit)
aux["D"] = df["long_delay_years"].round().astype(int)
aux["ratio"] = (df["long_reward"] / df["short_reward"]).round().astype(int)
aux["gi"] = pd.to_numeric(df["sample_uid"].str.rsplit("_", n=1).str[1], errors="coerce")
keep = pd.concat([df, aux], axis=1).query(a.query, engine="python").index
sub = df.loc[keep]
a.out.parent.mkdir(parents=True, exist_ok=True)
sub.to_parquet(a.out, index=False)
a.out.with_suffix(".config.json").write_text(json.dumps({"source": str(a.inp), "query": a.query, "n": len(sub)}, indent=2))
print(f"wrote {len(sub)} of {len(df)} prompts to {a.out}")
print(pd.concat([sub, aux.loc[keep]], axis=1).groupby(["horizon_unit", "delay_unit"]).size().unstack().to_string())
