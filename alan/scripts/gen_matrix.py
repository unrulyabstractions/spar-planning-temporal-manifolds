#!/usr/bin/env python
"""Generate the matched prompt matrix and its control sets.
Usage: gen_matrix.py OUT_DIR [--seed 0] [--n-configs 30]   -> OUT_DIR/matrix_s{seed}.parquet, OUT_DIR/controls_s{seed}.parquet"""
import argparse, json
from pathlib import Path
import pandas as pd
from ptm.matrix import MatrixConfig, build_matrix, build_controls, MATRIX_HORIZONS, HELDOUT_HORIZONS
from ptm.prompts import to_frame

ap = argparse.ArgumentParser(); ap.add_argument("out_dir", type=Path); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-configs", type=int, default=30)
a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
cfg = MatrixConfig(seed=a.seed, n_configs=a.n_configs)
m = to_frame(build_matrix(cfg)); k = to_frame(build_controls(cfg))
m.to_parquet(a.out_dir / f"matrix_s{a.seed}.parquet", index=False); k.to_parquet(a.out_dir / f"controls_s{a.seed}.parquet", index=False)
(a.out_dir / f"matrix_s{a.seed}.config.json").write_text(json.dumps({**cfg.__dict__, "horizons": [str(h) for h in MATRIX_HORIZONS], "heldout": [str(h) for h in HELDOUT_HORIZONS]}, indent=2, default=str))
print(f"matrix: {len(m)} prompts; configs {m.config_id.nunique()}, scenarios {m.scenario_id.nunique()}, horizons {m.horizon_text.nunique()}, renderings {m.rendering.value_counts().to_dict()}")
print("split by config:", m.groupby('split').config_id.nunique().to_dict(), " rows:", m.split.value_counts().to_dict(), " heldout-horizon rows:", int(m.horizon_heldout.sum()))
print("varied variants:", m[m.rendering == 'varied'].variant.value_counts().sort_index().to_dict())
print(f"controls: {len(k)} prompts; by condition {k.condition.value_counts().to_dict()}")
for cond in ["main", "mention", "role", "factual", "scaling"]:
    df = m if cond == "main" else k[k.condition == cond]
    rows = df[df.rendering != "structured"] if cond == "main" else df
    r = rows.iloc[0]
    print(f"\n--- example [{cond}] {r.sample_uid} ---\n{r.text}")
if True:
    r = m[(m.rendering == "varied")].iloc[5]; print(f"\n--- example [main/varied variant {r.variant}] ---\n{r.text}")
