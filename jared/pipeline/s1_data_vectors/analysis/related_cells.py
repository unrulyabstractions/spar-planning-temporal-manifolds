"""The horizon cell for the related-quality steering (EvalsRelatedQualities.md §3): E19's operating
point, `fit_cell` on the T0 full bank's thinking-on activations at the pre-label ':' (as
evaltypes_steering_prepare.bank_cells fits E19's cells), at one layer.

    python -m pipeline.s1_data_vectors.analysis.related_cells ~/SPAR/results/horizon
    -> jared/steering/related_box/cells_Qwen3-8B_on_L24.npz and .json   (shipped to the box by `just push`)

    python -m pipeline.s1_data_vectors.analysis.related_cells out/smoke_related --plumbing \\
        --model Qwen/Qwen3.5-0.8B --layer 12 --out out/smoke_related
    -> <out>/cells_Qwen3.5-0.8B_plumbing_L12.npz and .json

--plumbing is for the local smoke only: a cell fitted on the smoke's own related run (root/related_<model>_think-on),
every set's pre-label activations with FAKE log horizons, so the chain has a cell of the model's width.
Its curve means nothing.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.s1_data_vectors.analysis.horizon_shape import cell_data
from spar_horizon.e0_runs import run_dir
from spar_horizon.steering import fit_cell

MODEL = "Qwen/Qwen3-8B"
LAYER = 24
OUT = Path(__file__).resolve().parents[3] / "steering" / "related_box"
FAKE_HORIZONS = 4      # plumbing: distinct fake log horizons, so the curve has 3 dimensions


def bank_cell(root, model, layer):
    run = run_dir(root, model, "on")
    X, log_h, _ = cell_data(np.load(run / "activations.npz"), "pre-label", layer)
    return fit_cell(X, log_h), {"source": str(run), "n_prompts": len(log_h)}


def plumbing_cell(root, model, layer):
    run = Path(root) / f"related_{model.split('/')[-1]}_think-on"
    states = []
    for path in sorted(run.glob("*/acts_*.npz")):
        acts = np.load(path)
        states.append(acts[f"L{layer}"][:, acts["region_prelabel"][0]])
    X = np.concatenate(states).astype(np.float64)
    fake_log_h = np.arange(len(X)) % FAKE_HORIZONS - 1.0
    return fit_cell(X, fake_log_h), {"source": str(run), "n_prompts": len(X),
                                     "plumbing": "FAKE log horizons on the smoke's own activations"}


def main(root, out_dir, model=MODEL, layer=LAYER, plumbing=False):
    cell, meta = (plumbing_cell if plumbing else bank_cell)(root, model, layer)
    meta = {"model": model, "mode": "on", "label": "pre-label", "layer": layer, **meta,
            "k": int(cell["basis"].shape[0]),
            "curve_logh": [float(cell["curve_logh"][0]), float(cell["curve_logh"][-1])]}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cells_{model.split('/')[-1]}_{'plumbing' if plumbing else 'on'}_L{layer}"
    np.savez(out_dir / f"{stem}.npz", **{key: np.asarray(value) for key, value in cell.items()})
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    print(f"{stem}: k = {meta['k']}, curve from {meta['curve_logh'][0]:.2f} to {meta['curve_logh'][1]:.2f} "
          f"log10 years, fitted on {meta['n_prompts']} prompts{' (PLUMBING ONLY)' if plumbing else ''}")
    print(f"wrote {out_dir / stem}.npz and .json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", help="results/horizon with the bank runs; with --plumbing, the smoke's out dir")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--model", default=MODEL)
    p.add_argument("--layer", type=int, default=LAYER)
    p.add_argument("--plumbing", action="store_true", help="local smoke only: fake horizons, see the docstring")
    args = p.parse_args()
    main(args.root, args.out, args.model, args.layer, args.plumbing)
