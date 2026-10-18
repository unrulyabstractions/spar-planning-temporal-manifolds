"""Merge behaviour-only (--no-acts) run parts captured in parallel (one per GPU, --offset/--limit) into one run dir.
Usage: merge_index_parts.py OUT_RUN PART_RUN... ; asserts the merged sample_uids equal the pilot prompt set (path below).
meta.json is copied from the first part (its n_prompts is that part's); merged_from.json lists the parts."""
import sys, json, shutil, pandas as pd
from pathlib import Path
out = Path(sys.argv[1]); parts = [Path(p) for p in sys.argv[2:]]
out.mkdir(parents=True, exist_ok=True)
d = pd.concat([pd.read_parquet(p / "index.parquet") for p in parts], ignore_index=True)
ref = pd.read_parquet("data/prompts/unit_boundary_think_pilot.parquet")
assert d.sample_uid.is_unique and set(d.sample_uid) == set(ref.sample_uid), "merged uids != pilot uids"
d.to_parquet(out / "index.parquet")
shutil.copy(parts[0] / "meta.json", out / "meta.json")
json.dump({"merged_from": [str(p) for p in parts]}, open(out / "merged_from.json", "w"))
print(out, len(d), "rows; uids unique and equal to the pilot set")
