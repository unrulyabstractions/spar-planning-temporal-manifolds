#!/usr/bin/env python
"""Content checksums for capture runs, to verify that a re-capture reproduces the originals.

Per run: n rows; sha256 over (sample_uid, gen_ids, choice) in index order; float64 sum and sha256 of the
raw bytes of the layer-22 T3 activations (or the last layer if fewer); and the same for R0.
Usage: run_checksums.py RUN_DIR [RUN_DIR ...] [--out FILE.json]
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from ptm.store import RunData

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--out", default=None)
a = ap.parse_args()
res = {}
for r in a.runs:
    run = RunData(r); df = run.index
    h = hashlib.sha256()
    for u, g, c in zip(df.sample_uid, df.gen_ids, df.choice.fillna("")):
        h.update(f"{u}|{g}|{c}\n".encode())
    layer = min(22, run.n_layers)
    rec = dict(n=len(df), index_sha256=h.hexdigest(), model=run.meta["model_name"], layer=layer)
    for pos in ["T3", "R0"]:
        X = run.get(layer, pos)
        rec[f"{pos}_sum"] = float(np.float64(X).sum()); rec[f"{pos}_sha256"] = hashlib.sha256(X.tobytes()).hexdigest()
    res[Path(r).name] = rec
    print(f"{Path(r).name}: n={rec['n']} index={rec['index_sha256'][:12]} T3sum={rec['T3_sum']:.6g} R0sum={rec['R0_sum']:.6g}")
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=2); print("wrote", a.out)
