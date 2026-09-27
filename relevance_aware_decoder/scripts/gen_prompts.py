#!/usr/bin/env python
"""Generate the prompt set: Alan's matrix (unchanged) + distractor prompts for three test tiers.

Usage: gen_prompts.py OUT_DIR [--seed 0] [--smoke]
  -> OUT_DIR/prompts_s{seed}.parquet        (5,520 prompts; input to alan/scripts/capture.py)
     OUT_DIR/prompts_s{seed}_smoke.parquet  (with --smoke: 1 configuration per split, for a local plumbing test)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rad  # noqa: E402,F401  (puts ../alan on sys.path)

from ptm.matrix import MatrixConfig  # noqa: E402
from rad.build import DistractorConfig, build_all, dataset_hash, smoke_subset  # noqa: E402
from rad.families import FAMILIES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out_dir", type=Path); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

cfg = DistractorConfig(matrix=MatrixConfig(seed=a.seed), seed=a.seed)
df = build_all(cfg)
path = a.out_dir / f"prompts_s{a.seed}.parquet"
df.to_parquet(path, index=False)
dist = df[df.condition == "distractor"]
info = dict(n=len(df), hash=dataset_hash(df), by_condition=df.condition.value_counts().to_dict(),
            distractor_by_split_group={f"{k[0]}/{k[1]}": int(v) for k, v in dist.groupby(["split", "template_group"]).size().items()},
            families={f.name: dict(description=f.description, seen=f.seen, unseen=f.unseen, durations=[str(h) for h in f.durations])
                      for f in FAMILIES})
(a.out_dir / f"prompts_s{a.seed}.json").write_text(json.dumps(info, indent=2))
print(f"wrote {path}: {len(df)} prompts, hash {info['hash']}")
print("by condition:", info["by_condition"])
print("distractor rows by split/template group:", info["distractor_by_split_group"])
print("slots per family:\n", dist.groupby(["family", "slot"]).size().unstack().to_string())
for f in FAMILIES:
    for g in ("seen", "unseen"):
        r = dist[(dist.family == f.name) & (dist.template_group == g)].iloc[0]
        print(f"\n--- {r.template_id} (slot {r.slot}; H = {r.horizon_text}, D = {r.distractor_text}) ---\n{r.text}")
if a.smoke:
    sm = smoke_subset(df)
    sm.to_parquet(a.out_dir / f"prompts_s{a.seed}_smoke.parquet", index=False)
    print(f"\nsmoke subset: {len(sm)} prompts {sm.condition.value_counts().to_dict()}")
