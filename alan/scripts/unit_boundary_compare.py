#!/usr/bin/env python
"""Side-by-side unit-boundary curves for several models (runs of scripts/gen_unit_boundary.py prompts).

One row per model, one column per horizon unit; one line per delay unit (the matched unit drawn thick). y = fraction
choosing the short option, x = H / long delay (grid points pooled over long delay, reward ratio and phrasing; same
nominal H/D across D). --order picks short-first prompts, long-first prompts, or both pooled.

  unit_boundary_compare.py OUT.png "Qwen3-14B=runs/qwen3-14b_unit_boundary_n24" "Qwen3.8-27B=runs/..." [--order sf|lf|all]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ptm.horizons import canonical_unit
from ptm.unit_boundary import UNITS

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("runs", nargs="+", help="LABEL=RUN_DIR")
ap.add_argument("--order", choices=["sf", "lf", "all"], default="sf")
a = ap.parse_args()

COLORS = dict(zip(UNITS, ["#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7"]))   # validated set (depth_profiles, 2026-10-04)
ORDER_NAME = {"sf": "short option listed first", "lf": "long option listed first", "all": "both option orders"}
fig, axs = plt.subplots(len(a.runs), len(UNITS), figsize=(3.4 * len(UNITS), 2.7 * len(a.runs) + 0.9),
                        sharex=True, sharey=True, squeeze=False)
for row, spec in zip(axs, a.runs):
    label, run = spec.split("=", 1)
    d = pd.read_parquet(Path(run) / "index.parquet")
    d = d[d["choice"].isin(["a", "b"])].copy()
    if a.order != "all":
        d = d[d["short_first"] == (a.order == "sf")]
    d["du"] = d["long_delay_text"].str.split().str[1].map(canonical_unit)
    d["r"] = d["horizon_years"] / d["long_delay_years"]
    d["y"] = d["chose_short"].astype(bool).astype(float)
    d["gi"] = d["sample_uid"].str.rsplit("_", n=1).str[1].astype(int)
    for ax, hu in zip(row, UNITS):
        for du in UNITS:
            c = d[(d.horizon_unit == hu) & (d.du == du)].groupby("gi").agg(r=("r", "median"), p=("y", "mean"))
            ax.plot(c.r, c.p, "-o", ms=2.5 if hu != du else 3.5, color=COLORS[du], lw=2.8 if hu == du else 1.1,
                    alpha=1.0 if hu == du else 0.85, label=f"delays in {du}" + (" (= horizon unit)" if hu == du else ""))
        ax.axvline(1.0, color="0.45", lw=0.8, ls="--")
        ax.axvspan(0.6, 1.0, color="0.92", zorder=0)
        ax.set_xscale("log"); ax.set_xlim(0.65, 1.53); ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0.7, 0.85, 1.0, 1.2, 1.5]); ax.set_xticklabels(["0.7", "0.85", "1", "1.2", "1.5"])
        ax.minorticks_off(); ax.grid(alpha=0.2)
        ax.set_title(f"{label}: horizon in {hu}", fontsize=9)
    row[0].set_ylabel(f"{label}\nfraction choosing short", fontsize=9)
for ax in axs[-1]:
    ax.set_xlabel("horizon / long-option delay", fontsize=9)
handles = [plt.Line2D([], [], color=COLORS[u], lw=2, marker="o", ms=3, label=f"delays in {u}") for u in UNITS]
handles.append(plt.Line2D([], [], color="0.3", lw=2.8, label="thick = delays in the horizon's unit"))
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, frameon=False)
fig.suptitle(f"Choice vs horizon/delay ratio, {ORDER_NAME[a.order]}. Shaded: short option is correct (horizon < delay)",
             fontsize=10)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
a.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=140)
print(f"wrote {a.out}")
