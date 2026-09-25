#!/usr/bin/env python
"""Export a shareable activation subset for a run.
Usage: export_subset.py RUN_DIR --layers 22,37 --positions T0,T1,T2,T3,T4,T5,T6,T7,T8,R0,R1,R2,R3 [--max-mb 95]"""
import argparse
from pathlib import Path
from ptm.store import RunData
from ptm.subset import export_subset
ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--layers", required=True); ap.add_argument("--positions", required=True); ap.add_argument("--max-mb", type=float, default=95.0)
a = ap.parse_args()
run = RunData(a.run_dir)
meta = export_subset(run, [int(x) for x in a.layers.split(",")], a.positions.split(","), a.max_mb)
files = sorted(Path(a.run_dir).glob("acts_subset_*.safetensors"))
print(f"{a.run_dir}: layers {meta['layers']} positions {meta['positions']} chunks {meta['chunks']} -> {len(files)} files, {sum(f.stat().st_size for f in files) / 2**20:.0f} MB, max file {max(f.stat().st_size for f in files) / 2**20:.0f} MB")
