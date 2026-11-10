#!/usr/bin/env python
"""Unit-compatibility boundary analysis (prompts from scripts/gen_unit_boundary.py, behaviour from capture.py).

Under the constraint the short option is correct iff the horizon H is below the long delay D (D = long_delay_years, the
rendered delay: 522 weeks = 10.004 y). With x = ln(H / D), a logistic fit P(short) = sigmoid(b0 + b1 x) to the choices
gives the switch point mu = -b0 / b1, reported as the ratio exp(mu) = H/D at which the model is indifferent
(1.0 = the correct boundary; 0.85 = it starts taking the long option when the horizon is 15% short of its delay).
Shift = mu(cell) - mu(matched cell), in ln units, with a bootstrap CI over prompts (resampled within each cell).
Errors: below = long chosen though H < D; above = short chosen though H > D.

  unit_boundary.py RUN_DIR OUT_DIR [--label "Qwen3-14B"] [--n-boot 1000] [--seed 0]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from ptm.horizons import canonical_unit
from ptm.unit_boundary import UNITS, switch_point

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path); ap.add_argument("out", type=Path)
ap.add_argument("--label", default=None); ap.add_argument("--n-boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
label = a.label or a.run.name
a.out.mkdir(parents=True, exist_ok=True)

d = pd.read_parquet(a.run / "index.parquet")
n_all = len(d)
d = d[d["choice"].isin(["a", "b"])].copy()
d["hu"] = d["horizon_unit"].map(canonical_unit)
d["du"] = d["long_delay_text"].str.split().str[1].map(canonical_unit)
d["D"] = d["long_delay_years"].round().astype(int)
d["ratio"] = (d["long_reward"] / d["short_reward"]).round().astype(int)
d["order"] = np.where(d["short_first"], "short first", "long first")
d["match"] = d["hu"] == d["du"]
d["x"] = np.log(d["horizon_years"] / d["long_delay_years"])
d["y"] = d["chose_short"].astype(bool).astype(int)
d["err_below"] = (d["x"] < 0) & (d["y"] == 0)
d["err_above"] = (d["x"] > 0) & (d["y"] == 1)
rng = np.random.default_rng(a.seed)


XMIN, XMAX = d["x"].min(), d["x"].max()     # the sampled range of ln(H/D)


def switch(x, y) -> float:
    """switch_point, censored to the sampled range: -inf if the switch lies below it (or the cell always chooses long),
    +inf above it (or always short); NaN only for a rising curve."""
    y = np.asarray(y)
    if y.size and y.min() == y.max():
        return np.inf if y[0] == 1 else -np.inf
    mu = switch_point(x, y)
    return mu if not np.isfinite(mu) or XMIN <= mu <= XMAX else (-np.inf if mu < XMIN else np.inf)


def fsw(mu) -> str:
    return "n/a" if np.isnan(mu) else f"<{np.exp(XMIN):.3f}" if mu == -np.inf else f">{np.exp(XMAX):.3f}" if mu == np.inf else f"{np.exp(mu):.3f}"


def fsh(v) -> str:
    return "n/a" if np.isnan(v) else ("-inf" if v < 0 else "+inf") if np.isinf(v) else f"{v:+.3f}"


def fci(est, lo, hi, bad, f) -> str:
    return f"{f(est)} [{f(lo)}, {f(hi)}]" + ("†" if bad > 0.05 else "")


def boot(groups: list[tuple[np.ndarray, np.ndarray]], stat) -> tuple[float, float, float]:
    """stat over groups of (x, y), with a percentile CI from resampling prompts within each group."""
    est = stat(*[g for xy in groups for g in xy])
    reps = []
    for _ in range(a.n_boot):
        rs = []
        for x, y in groups:
            i = rng.integers(0, len(x), len(x))
            rs += [x[i], y[i]]
        reps.append(stat(*rs))
    reps = np.array(reps, dtype=float)
    bad = np.isnan(reps).mean()                 # resamples with no defined statistic (inf - inf, rising curve)
    lo, hi = np.nanpercentile(reps, [2.5, 97.5], method="inverted_cdf") if bad < 1 else (np.nan, np.nan)   # no interpolation across ±inf
    return est, lo, hi, bad


def xy(df):
    return df["x"].to_numpy(), df["y"].to_numpy()


lines = [f"# Unit-boundary analysis: {label}", "",
         f"Run `{a.run}`; {len(d)} of {n_all} prompts with a parsed choice (format adherence {len(d) / n_all:.3f}).",
         f"Switch point = H/D at P(short) = 0.5 (logistic fit on ln(H/D)); 95% CI: {a.n_boot} bootstrap resamples of prompts.",
         "Shift = ln(switch) - ln(switch of the matched cell); negative = the model takes the long option at shorter horizons.",
         "A switch outside the sampled H/D range is shown as <min or >max (censored, not extrapolated); a shift involving one",
         "is ±inf. † = more than 5% of bootstrap resamples gave no defined value; treat that interval as unreliable.", ""]

# 1. switch point per (horizon unit, delay unit), pooled over D, ratio, order, phrasing
rows = []
for hu in UNITS:
    for du in UNITS:
        c = d[(d.hu == hu) & (d.du == du)]
        mu, lo, hi, bad = boot([xy(c)], switch)
        rows.append(dict(hu=hu, du=du, n=len(c), switch=np.exp(mu), lo=np.exp(lo), hi=np.exp(hi), mu=mu, mu_lo=lo, mu_hi=hi, bad=bad,
                         err_below=c.loc[c.x < 0, "err_below"].mean(), err_above=c.loc[c.x > 0, "err_above"].mean()))
cells = pd.DataFrame(rows)
tab = cells.pivot(index="hu", columns="du", values="switch").loc[list(UNITS), list(UNITS)]
lines += ["## Switch point H/D by horizon unit (rows) and delay unit (columns), pooled", "",
          "| horizon \\ delays | " + " | ".join(UNITS) + " |", "|---|" + "---|" * len(UNITS)]
for hu in UNITS:
    cs = []
    for du in UNITS:
        r = cells[(cells.hu == hu) & (cells.du == du)].iloc[0]
        cs.append(f"{'**' if hu == du else ''}{fci(r.mu, r.mu_lo, r.mu_hi, r.bad, fsw)}{'**' if hu == du else ''}")
    lines.append(f"| {hu} | " + " | ".join(cs) + " |")
lines += ["", "Error rates (below: long chosen though H < D / above: short chosen though H > D):", "",
          "| horizon \\ delays | " + " | ".join(UNITS) + " |", "|---|" + "---|" * len(UNITS)]
for hu in UNITS:
    cs = [f"{r.err_below:.3f} / {r.err_above:.3f}" for _, r in cells[cells.hu == hu].set_index("du").loc[list(UNITS)].iterrows()]
    lines.append(f"| {hu} | " + " | ".join(cs) + " |")

# 2. shift relative to the matched cell, by row (same horizon rendering) and by column (same delay rendering)
def shift_stat(x1, y1, x0, y0):
    return switch(x1, y1) - switch(x0, y0)

lines += ["", "## Shift vs matched units (ln units; 95% CI)", "",
          "Same horizon unit, delays changed (controls the horizon's rendering) | same delay unit, horizon changed:", "",
          "| horizon unit | delay unit | shift vs (H, delays in H's unit) | shift vs (horizon in delays' unit, D) |", "|---|---|---|---|"]
shift_rows = []
for hu in UNITS:
    for du in UNITS:
        if hu == du:
            continue
        c = d[(d.hu == hu) & (d.du == du)]
        r_ = boot([xy(c), xy(d[(d.hu == hu) & (d.du == hu)])], shift_stat)
        c_ = boot([xy(c), xy(d[(d.hu == du) & (d.du == du)])], shift_stat)
        shift_rows.append(dict(hu=hu, du=du, row_shift=r_, col_shift=c_))
        lines.append(f"| {hu} | {du} | {fci(*r_, fsh)} | {fci(*c_, fsh)} |")

# 3. matched vs mismatched, broken down by each design factor
lines += ["", "## Matched vs mismatched by factor (pooled over unit pairs)", "",
          "| factor | level | switch matched | switch mismatched | shift (ln) | errors below, matched / mismatched |", "|---|---|---|---|---|---|"]
for fac in ["D", "ratio", "order", "phrasing_id", "hu", "du"]:
    for lev in sorted(d[fac].unique(), key=str):
        m, mm = d[(d[fac] == lev) & d.match], d[(d[fac] == lev) & ~d.match]
        sh = boot([xy(mm), xy(m)], shift_stat)
        lines.append(f"| {fac} | {lev} | {fsw(switch(*xy(m)))} | {fsw(switch(*xy(mm)))} | {fci(*sh, fsh)} | "
                     f"{m.loc[m.x < 0, 'err_below'].mean():.3f} / {mm.loc[mm.x < 0, 'err_below'].mean():.3f} |")

# 4. continuity with the 2026-10-03 2x2 (months/years x D = 10 y, ratio 5, short first, phrasing 0)
lines += ["", "## Continuity check: the 2026-10-03 conditions (months/years, D = 10 y, ratio 5, short first, phrasing 0)", "",
          "| horizon | delays | n | switch H/D | switch H (years) |", "|---|---|---|---|---|"]
for hu in ["months", "years"]:
    for du in ["years", "months"]:
        c = d[(d.hu == hu) & (d.du == du) & (d.D == 10) & (d.ratio == 5) & d.short_first & (d.phrasing_id == 0)]
        mu = switch(*xy(c))
        lines.append(f"| {hu} | {du} | {len(c)} | {fsw(mu)} | {10 * np.exp(mu):.2f} |")

# 5. option order and label: P(choose "a)") and P(short) per unit cell x H/D bin, split by order
lines += ["", "## Option order", "",
          f"P(choose a)) = {(d.choice == 'a').mean():.3f} overall; short first {(d.loc[d.short_first, 'choice'] == 'a').mean():.3f}, "
          f"long first {(d.loc[~d.short_first, 'choice'] == 'a').mean():.3f}.", ""]
BINS = [0.66, 0.75, 0.85, 0.95, 1.0, 1.05, 1.2, 1.51]
d["rb"] = pd.cut(np.exp(d["x"]), BINS)
for sf, name in [(True, "short first"), (False, "long first")]:
    t = d[d.short_first == sf].pivot_table(index=["hu", "du"], columns="rb", values="y", aggfunc="mean", observed=False)
    t = t.reindex(pd.MultiIndex.from_product([UNITS, UNITS]))
    lines += [f"P(short) by H/D bin, {name} (rows: horizon unit / delay unit):", "",
              "| H / delays | " + " | ".join(str(c) for c in t.columns) + " |", "|---|" + "---|" * len(t.columns)]
    for (hu, du), r in t.iterrows():
        lines.append(f"| {hu} / {du}{' (match)' if hu == du else ''} | " + " | ".join(f"{v:.2f}" for v in r) + " |")
    lines.append("")
lines += ["P(short) at H/D < 0.85 by phrasing and order:", "", "| phrasing | short first | long first |", "|---|---|---|"]
for p_ in sorted(d.phrasing_id.unique()):
    q = d[(d.phrasing_id == p_) & (d.x < np.log(0.85))]
    lines.append(f"| {p_} | {q[q.short_first].y.mean():.2f} | {q[~q.short_first].y.mean():.2f} |")

(a.out / "unit_boundary.md").write_text("\n".join(lines) + "\n")
cells.to_csv(a.out / "unit_boundary_cells.csv", index=False)
print("\n".join(lines))

# figure: P(short) vs H/D, one panel per horizon unit, one line per delay unit (matched bold)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d["gi"] = d["sample_uid"].str.rsplit("_", n=1).str[1].astype(int)       # grid index: same nominal H/D across D
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
cols = dict(zip(UNITS, ["#1b6ca8", "#d1495b", "#2a9d8f", "#6c4f9e"]))
for ax, hu in zip(axs, UNITS):
    for du in UNITS:
        c = d[(d.hu == hu) & (d.du == du)].groupby("gi").agg(r=("x", "mean"), p=("y", "mean"))
        ax.plot(np.exp(c.r), c.p, "-o", ms=3, color=cols[du], lw=2.6 if hu == du else 1.2, label=f"delays in {du}")
    ax.axvline(1.0, color="0.5", lw=0.8, ls="--"); ax.set_xscale("log"); ax.set_title(f"horizon in {hu}")
    ax.set_xlabel("H / long delay"); ax.grid(alpha=0.25)
axs[0].set_ylabel("fraction choosing short")
axs[0].legend(fontsize=8, loc="lower left")
fig.suptitle(f"{label}: choice switch vs horizon/long-delay ratio (pooled over D, reward ratio, order, phrasing)", fontsize=10)
fig.tight_layout(); fig.savefig(a.out / "unit_boundary_curves.png", dpi=130)
print(f"wrote {a.out / 'unit_boundary.md'}, {a.out / 'unit_boundary_curves.png'}")
