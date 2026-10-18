#!/usr/bin/env python
"""No-thinking vs thinking unit-boundary curves for one model (paired pilot runs, e.g. the 10-17 thinking pilots).

Same panels and style as scripts/unit_boundary_compare.py: one row per horizon unit, one line per delay unit (the
matched unit drawn thick), y = fraction choosing the short option, x = H / long delay. Left column: no thinking;
right column: thinking. Prompts whose thinking never closed have no choice and are dropped (counted in the title).

  unit_boundary_think_compare.py OUT.png LABEL NOTHINK_RUN THINK_RUN [--order sf|lf|all]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ptm.horizons import canonical_unit
from ptm.unit_boundary import UNITS

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("label")
ap.add_argument("nothink", type=Path)
ap.add_argument("think", type=Path)
ap.add_argument("--order", choices=["sf", "lf", "all"], default="all")
a = ap.parse_args()

COLORS = dict(zip(UNITS, ["#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7"]))   # validated set (depth_profiles, 2026-10-04)
ORDER_NAME = {"sf": "short option listed first", "lf": "long option listed first", "all": "both option orders"}


def load(run):
    d = pd.read_parquet(run / "index.parquet")
    if a.order != "all":
        d = d[d["short_first"] == (a.order == "sf")]
    n_all = len(d)
    d = d[d["choice"].isin(["a", "b"])].copy()
    d["du"] = d["long_delay_text"].str.split().str[1].map(canonical_unit)
    d["r"] = d["horizon_years"] / d["long_delay_years"]
    d["y"] = d["chose_short"].astype(bool).astype(float)
    d["gi"] = d["sample_uid"].str.rsplit("_", n=1).str[1].astype(int)
    return d, n_all - len(d)


cols = [("no thinking", *load(a.nothink)), ("thinking", *load(a.think))]
per_point = {"all": "2 prompts per point (one per option order)", "sf": "1 prompt per point", "lf": "1 prompt per point"}[a.order] + " in the 10-17 pilot"
fig, axs = plt.subplots(len(UNITS), 2, figsize=(3.4 * 2 + 0.6, 2.3 * len(UNITS) + 1.1), sharex=True, sharey=True)
for j, (mode, d, n_drop) in enumerate(cols):
    for ax, hu in zip(axs[:, j], UNITS):
        for du in UNITS:
            c = d[(d.horizon_unit == hu) & (d.du == du)].groupby("gi").agg(r=("r", "median"), p=("y", "mean"))
            ax.plot(c.r, c.p, "-o", ms=2.5 if hu != du else 3.5, color=COLORS[du], lw=2.8 if hu == du else 1.1,
                    alpha=1.0 if hu == du else 0.85)
        ax.axvline(1.0, color="0.45", lw=0.8, ls="--")
        ax.axvspan(0.6, 1.0, color="0.92", zorder=0)
        ax.set_xscale("log"); ax.set_xlim(0.65, 1.53); ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0.7, 0.85, 1.0, 1.2, 1.5]); ax.set_xticklabels(["0.7", "0.85", "1", "1.2", "1.5"])
        ax.minorticks_off(); ax.grid(alpha=0.2)
        ax.set_title(f"horizon in {hu}", fontsize=9)
    drop = f"; {n_drop} unclosed dropped" if n_drop else ""
    axs[0, j].annotate(f"{mode.upper()} (n = {len(d)}{drop})", xy=(0.5, 1.2), xycoords="axes fraction",
                       ha="center", fontsize=10, fontweight="bold")
for ax in axs[:, 0]:
    ax.set_ylabel("fraction choosing short", fontsize=9)
for ax in axs[-1]:
    ax.set_xlabel("horizon / long-option delay", fontsize=9)
handles = [plt.Line2D([], [], color=COLORS[u], lw=2, marker="o", ms=3, label=f"delays in {u}") for u in UNITS]
handles.append(plt.Line2D([], [], color="0.3", lw=2.8, label="thick = delays in the horizon's unit"))
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
fig.suptitle(f"{a.label}: choice vs horizon/delay ratio, {ORDER_NAME[a.order]}\n"
             f"{per_point}\nshaded: short option is correct (horizon < delay)", fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 0.96), h_pad=0.6)
a.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=140)
print(f"wrote {a.out}")
