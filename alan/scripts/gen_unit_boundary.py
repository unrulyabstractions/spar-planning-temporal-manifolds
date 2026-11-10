#!/usr/bin/env python
"""Unit-compatibility boundary set (ptm/unit_boundary.py): horizon unit x delay unit x long delay x reward ratio x
option order x constraint phrasing, horizons swept densely around the long delay.

  gen_unit_boundary.py OUT.parquet [--n 24] [--span 1.5] [--units weeks,months,quarters,years] [--long-delays 2,5,10]
      [--ratios 5,20] [--phrasings 0,1,2,3] [--orders sf,lf]
"""
import argparse, json
from pathlib import Path

from ptm.prompts import to_frame
from ptm.unit_boundary import unit_boundary_set

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--n", type=int, default=24); ap.add_argument("--span", type=float, default=1.5)
ap.add_argument("--units", default="weeks,months,quarters,years", help="used for both the horizon and the delays")
ap.add_argument("--long-delays", default="2,5,10"); ap.add_argument("--ratios", default="5,20")
ap.add_argument("--phrasings", default="0,1,2,3"); ap.add_argument("--orders", default="sf,lf")
a = ap.parse_args()
units = tuple(a.units.split(","))
samples, dropped = unit_boundary_set(
    n=a.n, span=a.span, horizon_units=units, delay_units=units, long_delays=tuple(int(x) for x in a.long_delays.split(",")),
    ratios=tuple(int(x) for x in a.ratios.split(",")), orders=tuple(o == "sf" for o in a.orders.split(",")),
    phrasings=tuple(int(x) for x in a.phrasings.split(",")))
# shuffled (fixed seed) so that any finished prefix of a capture is a balanced subset of the crossing
df = to_frame(samples).sample(frac=1.0, random_state=0).reset_index(drop=True)
assert df.sample_uid.is_unique and df.prompt_id.is_unique, "duplicate prompts"
a.out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(a.out, index=False)
a.out.with_suffix(".config.json").write_text(json.dumps(vars(a) | {"out": str(a.out), "n_prompts": len(df), "dropped": dropped}, indent=2))
print(f"wrote {len(df)} prompts to {a.out}; dropped {dropped}")
print(df.groupby(["horizon_unit", "long_delay_text"]).size().unstack().to_string())
print("\n--- first prompt ---\n" + df.text.iloc[0])
