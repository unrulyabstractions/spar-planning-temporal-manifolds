#!/usr/bin/env python
"""Cross-model depth profile of single-population horizon ordinality, from existing sweep tables.

Reads `sweep.csv` (written by analyze.py --sweep) from one figure directory per model and plots
|rho(PC1, log H)| against fractional depth l/L (stored layer l = residual stream after decoder layer l-1,
0 = embedding output, L = max stored layer), one panel per position, models overlaid. Prints and writes
per model × position: peak |rho| and its depth, the depth range within --plateau of the peak, onset (first
depth with |rho| >= --thr), the share of layers at or past onset that stay >= --thr, and the last depth >= --thr.

Usage: depth_profiles.py OUT_DIR LABEL=FIGDIR [LABEL=FIGDIR ...] [--positions T3,R0] [--thr 0.8] [--plateau 0.02]
                         [--colors C1,C2,...]   (one matplotlib color per run, in order; default color cycle)
                         [--styles S1,S2,...]   (one matplotlib linestyle per run, e.g. -,--; default solid)
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# family-neutral panel labels: transition positions are compared by index (Qwen T0 = <|im_end|>, T3 = assistant;
# Gemma 4 T0 = <turn|>, T3 = model)
TOKEN = {"T0": "end of user turn", "T3": "role token: assistant / model", "R0": "first response token"}

ap = argparse.ArgumentParser(); ap.add_argument("out_dir"); ap.add_argument("runs", nargs="+")
ap.add_argument("--positions", default="T3,R0"); ap.add_argument("--thr", type=float, default=0.8)
ap.add_argument("--plateau", type=float, default=0.02); ap.add_argument("--colors", default=None)
ap.add_argument("--styles", default=None)
a = ap.parse_args()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
positions = a.positions.split(",")
colors = a.colors.split(",") if a.colors else [None] * len(a.runs)
assert len(colors) == len(a.runs), "--colors needs one color per run"
styles = a.styles.split(",") if a.styles else ["-"] * len(a.runs)
assert len(styles) == len(a.runs), "--styles needs one linestyle per run"

rows, curves = [], {}
for spec in a.runs:
    label, fig_dir = spec.split("=", 1)
    d = pd.read_csv(Path(fig_dir) / "sweep.csv"); L = int(d.layer.max())
    for p in positions:
        s = d[(d.position == p) & (d.layer > 0)].sort_values("layer")
        f, r = s.layer.to_numpy() / L, s.rho_pc1_horizon.abs().to_numpy()
        curves[(label, p)] = (f, r)
        i = int(np.nanargmax(r)); pl = f[r >= r[i] - a.plateau]; above = r >= a.thr
        on = int(np.argmax(above)) if above.any() else None
        rows.append(dict(model=label, L=L, position=p, peak=r[i], peak_depth=f[i], plateau_lo=pl.min(), plateau_hi=pl.max(),
                         onset_depth=f[on] if on is not None else np.nan,
                         sustained=above[on:].mean() if on is not None else np.nan,
                         last_depth=f[np.where(above)[0][-1]] if on is not None else np.nan))
tab = pd.DataFrame(rows); tab.to_csv(out / "depth_profiles.csv", index=False)
with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
    print(f"|rho(PC1, log H)| by fractional depth; onset/last = first/last depth with |rho| >= {a.thr}; "
          f"sustained = share of layers from onset on that stay >= {a.thr}; plateau = depths within {a.plateau} of peak")
    print(tab.to_string(index=False), flush=True)

fig, axes = plt.subplots(1, len(positions), figsize=(5.2 * len(positions), 3.8), sharey=True, squeeze=False)
for ax, p in zip(axes[0], positions):
    for spec, c, st in zip(a.runs, colors, styles):
        label = spec.split("=", 1)[0]; f, r = curves[(label, p)]
        ax.plot(f, r, marker="o", ms=2.5, lw=1.6, label=label, color=c, ls=st)
    ax.axhline(a.thr, color="0.6", lw=0.8, ls="--")
    ax.set_title(f"{p} ({TOKEN.get(p, p)})"); ax.set_xlabel("fractional depth  l / L"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
axes[0][0].set_ylabel("|rho(PC1, log horizon)|")
# legend below the panels, one row: inside a panel it hid parts of the curves
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Horizon ordinality on PC1 by depth (canonical investment prompts, one prompt style)", fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out / "depth_profiles.png", dpi=160)
print(f"wrote {out / 'depth_profiles.png'} and {out / 'depth_profiles.csv'}")
