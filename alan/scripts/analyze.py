#!/usr/bin/env python
"""Sweep a run: per-(layer, position) metrics table, sweep heatmap, best-cell scatter, behavior curve.

Usage: analyze.py RUN_DIR OUT_DIR [--layers 0,10,20] [--positions T0,T8,R0] [--supervised]
"""
import argparse
from pathlib import Path

import numpy as np

from ptm.analysis import behavior_by_horizon, plot_behavior, plot_cell, plot_sweep, sweep
from ptm.store import RunData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--positions", default=None)
    ap.add_argument("--supervised", action="store_true")
    ap.add_argument("--top", type=int, default=4, help="scatter plots for the top-k cells by |rho_pc1|")
    ap.add_argument("--replot-only", action="store_true", help="regenerate figures from OUT_DIR/sweep.csv without recomputing")
    a = ap.parse_args()
    run = RunData(a.run_dir)
    if a.replot_only:
        import pandas as pd
        t = pd.read_csv(a.out_dir / "sweep.csv")
        plot_sweep(t, run, a.out_dir / "sweep_rho_pc1_horizon.png")
        plot_sweep(t, run, a.out_dir / "sweep_rho_pc1_reward_ratio.png", metric="rho_pc1_reward_ratio")
        plot_sweep(t, run, a.out_dir / "sweep_silhouette_choice.png", metric="silhouette_choice_pc3", absolute=False)
        plot_behavior(run, a.out_dir / "behavior_vs_horizon.png")
        t["abs_rho"] = t["rho_pc1_horizon"].abs()
        for _, r in t.sort_values("abs_rho", ascending=False).head(a.top).iterrows():
            plot_cell(run, int(r["layer"]), r["position"], a.out_dir / f"cell_L{int(r['layer']):02d}_{r['position']}.png")
        print("replotted", a.out_dir); return
    layers = [int(x) for x in a.layers.split(",")] if a.layers else None
    positions = a.positions.split(",") if a.positions else None
    a.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"run: {run.meta['model_name']}  n={len(run.index)}  layers={run.n_layers}  positions={run.labels}")
    fmt_ok = run.index["choice"].notna().mean()
    print(f"format adherence: {fmt_ok:.3f}   chose_short overall: {run.index['chose_short'].dropna().astype(float).mean():.3f}")
    b = behavior_by_horizon(run)
    b.to_csv(a.out_dir / "behavior_by_horizon.csv", index=False)
    print(b.to_string(index=False))
    plot_behavior(run, a.out_dir / "behavior_vs_horizon.png")

    t = sweep(run, layers=layers, positions=positions, supervised=a.supervised)
    t.to_csv(a.out_dir / "sweep.csv", index=False)
    plot_sweep(t, run, a.out_dir / "sweep_rho_pc1_horizon.png")
    plot_sweep(t, run, a.out_dir / "sweep_rho_pc1_reward_ratio.png", metric="rho_pc1_reward_ratio")
    plot_sweep(t, run, a.out_dir / "sweep_silhouette_choice.png", metric="silhouette_choice_pc3", absolute=False)
    t["abs_rho"] = t["rho_pc1_horizon"].abs()
    top = t.sort_values("abs_rho", ascending=False).head(a.top)
    print("\ntop cells by |rho_pc1_horizon|:")
    print(top[["layer", "position", "rho_pc1_horizon", "rho_pc1_reward_ratio", "evr_pc1", "silhouette_choice_pc3", "rho_pc1_p_short"]].to_string(index=False))
    import json
    print("\nsample-level PCA spectrum at the top cells (explained variance ratio; |rho| of each PC with log horizon):")
    for _, r in top.iterrows():
        ev = json.loads(r["evr_top10"]); rh = json.loads(r["rho_top10"])
        print(f"  L{int(r['layer'])} {r['position']}: evr " + " ".join(f"{v:.3f}" for v in ev) + f"  (cum2 {r['cum_evr_2']:.2f} cum5 {r['cum_evr_5']:.2f} cum10 {r['cum_evr_10']:.2f})")
        print(f"            |rho| " + " ".join(f"{v:.2f}" if v is not None else " nan" for v in rh))
    for _, r in top.iterrows():
        plot_cell(run, int(r["layer"]), r["position"], a.out_dir / f"cell_L{int(r['layer']):02d}_{r['position']}.png")
    print(f"\nwrote {a.out_dir}")


if __name__ == "__main__":
    main()
