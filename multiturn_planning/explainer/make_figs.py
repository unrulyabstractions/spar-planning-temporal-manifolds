"""Figures for explainer/index.html (multi-turn planning, Qwen3-14B run qwen3-14b_mtp_s0).

Reads ../results/qwen3-14b_mtp_s0/ (in git) and ../runs/qwen3-14b_mtp_s0/ (not in git: the capture, on Hugging Face as
anicola/ptm-multiturn-planning-qwen3-14b, same layout; the behavior and PCA figures need its index.parquet and
activation shards). Writes PNGs to figs/.
Run from anywhere:  python multiturn_planning/explainer/make_figs.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

MTP = Path(__file__).resolve().parents[1]                # multiturn_planning/
RUN, RES = MTP / "runs/qwen3-14b_mtp_s0", MTP / "results/qwen3-14b_mtp_s0"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(MTP))
from mtp import analysis as an  # noqa: E402

INK, MUTED, RULE, BAND = "#1A242D", "#56636E", "#D6DDE2", "#EEF1F3"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (validated all-pairs)
RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # ordinal blue
BLUE = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": RULE, "axes.linewidth": 1, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 160, "savefig.bbox": "tight",
})
TARGETS = ["1 week", "1 month", "3 months", "1 year", "3 years", "10 years", "25 years", "50 years"]
DUR_TICKS = {np.log10(1 / 365.25): "1 day", np.log10(7 / 365.25): "1 week", np.log10(1 / 12): "1 month",
             np.log10(0.5): "6 months", 0: "1 year", 1: "10 years", np.log10(50): "50 years"}
BOUNDARY = [f"P{i}" for i in range(9)] + ["U"]

df = pd.read_parquet(RUN / "index.parquet")
df["log_h_step"] = np.log10(df.h_step_years.astype(float))
df["log_h_target"] = np.log10(df.h_target_years.astype(float))
st = df[(df.kind == "step") & df.log_h_step.notna()].copy()
st["k"] = st.turn - 1


def dur_axis(ax, which="y", lim=None):
    ticks = {v: k for v, k in DUR_TICKS.items() if lim is None or lim[0] <= v <= lim[1]}
    (ax.set_yticks if which == "y" else ax.set_xticks)(list(ticks), list(ticks.values()))


# 1 · behavior: median step horizon per step, one line per target -----------------------------------------
fig, ax = plt.subplots(figsize=(8.4, 4.4))
hor = st[st.condition == "horizon"]
ends = {}
for i, t in enumerate(TARGETS):
    m = hor[hor.h_target_text == t].groupby("k").log_h_step.median()
    ax.plot(m.index, m.values, color=RAMP[i], lw=2, marker="o", ms=5, zorder=3)
    ends.setdefault(round(m.values[-1], 2), []).append(t.replace(" years", "").replace(" year", "") if len(ends) >= 0 else t)
for y, ts in ends.items():                      # targets whose lines end at the same value share one label
    lab = " & ".join(ts)
    lab += "" if lab.endswith(("week", "month", "months")) else (" years" if len(ts) > 1 or ts[0] != "1" else " year")
    ax.text(5.12, y, lab, color=INK, fontsize=9, va="center")
nm = st[st.condition == "none"].groupby("k").log_h_step.median()
ax.plot(nm.index, nm.values, color=S2, lw=2, ls="--", marker="o", ms=5, zorder=3)
ax.text(5.12, nm.values[-1] - 0.12, "no horizon", color=S2, fontsize=9, va="center", fontweight="bold")
ax.set_xticks(range(1, 6), [f"step {k}" for k in range(1, 6)])
ax.set_xlim(0.8, 5.9)
dur_axis(ax)
ax.set_ylabel("step horizon (median over conversations)")
fig.savefig(OUT / "behavior.png"); plt.close(fig)

# 2 · A: reading H_target turn by turn ----------------------------------------------------------------------
A = pd.read_csv(RES / "A_persistence.csv")
Ab = A[A.pos.str.startswith("P")]            # P0-P8 only: at turn 1, U is the "." right after the horizon phrase
same = Ab.groupby("turn").r2_same_turn.max()
trans = Ab.groupby("turn").r2.max()
fig, ax = plt.subplots(figsize=(7.6, 3.8))
ax.axhline(0, color=MUTED, lw=1)
ax.plot(same.index, same.values, color=S1, lw=2, marker="o", ms=6, label="probe refit at that turn")
ax.plot(trans.index, trans.values, color=S2, lw=2, marker="o", ms=6, label="probe trained at turn 1, applied at that turn")
ax.set_xticks(range(1, 8), ["1\noutline", "2\nstep 1", "3\nstep 2", "4\nstep 3", "5\nstep 4", "6\nstep 5", "7\ndone"])
ax.set_ylim(-0.05, 1.02); ax.set_xlabel("turn (boundary window before the reply)")
ax.set_ylabel("R² for log H_target\n(best boundary cell)")
ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 0.08))
fig.savefig(OUT / "A_persistence.png"); plt.close(fig)

# 3 · A: PCA at turn 1 vs turn 4, same cell -------------------------------------------------------------------
df2, acts, valid, meta = an.load_run(RUN)
_t1 = A[(A.turn == 1) & A.pos.str.startswith("P")].sort_values("pc1_rho", ascending=False).iloc[0]
LAYER, POS = int(_t1.layer), _t1.pos                  # the boundary cell where PC1 orders H_target best at turn 1
print(f"PCA cell: layer {LAYER}, {POS}")
li, pi = meta["layers"].index(LAYER), meta["positions"].index(POS)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
for ax, t in zip(axes, (1, 4)):
    r = df2[(df2.turn == t) & (df2.condition == "horizon")]
    Z = PCA(2, random_state=0).fit_transform(acts[r.row.values, li, pi].astype(np.float32))
    rho = abs(pd.Series(Z[:, 0]).corr(pd.Series(r.log_h_target.values), method="spearman"))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=r.log_h_target.values, cmap=BLUE, s=26, edgecolor="white", linewidth=0.6)
    ax.set_title(f"turn {t} ({'before the outline' if t == 1 else 'before step 3'})\nPC1 |ρ| = {rho:.2f}",
                 fontsize=10.5, loc="left")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
cb.set_ticks([np.log10(1 / 52), np.log10(0.25), 0, 1, np.log10(50)], labels=["1 week", "3 months", "1 year", "10 years", "50 years"])
cb.set_label("H_target"); cb.outline.set_visible(False)
fig.suptitle(f"layer {LAYER}, boundary token {POS}", x=0.02, ha="left", fontsize=9, color=MUTED, y=1.04)
fig.savefig(OUT / "A_pca.png"); plt.close(fig)

# 4 · B: activations over the text baseline, by layer ---------------------------------------------------------
B = pd.read_csv(RES / "B_next_step.csv")
Bb = B[B.pos.isin(BOUNDARY)]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
for ax, cond, title in zip(axes, ("horizon", "none"), ("target stated (1,411 steps)", "no target stated (912 steps)")):
    d = Bb[Bb.condition == cond]
    best = d.groupby("layer").delta_r2.max()
    floor = d.groupby("layer").delta_r2_shuffled.max()
    ax.axhline(0, color=MUTED, lw=1)
    ax.fill_between(floor.index, -0.5, floor.values, color="#DCE2E7", lw=0, label="shuffled labels (best cell)")
    ax.plot(best.index, best.values, color=S1, lw=2, marker="o", ms=5, label="real labels (best boundary cell)")
    ax.set_title(f"{title}: text alone R² = {d.r2_text.iloc[0]:.2f}", fontsize=10.5, loc="left")
    ax.set_xlabel("layer")
axes[0].set_ylabel("ΔR² from adding activations\nto the text baseline")
axes[0].set_ylim(-0.16, 0.08)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")   # above the band, so its swatch shows
fig.savefig(OUT / "B_delta.png"); plt.close(fig)

# 5 · C: turn-1 activations and each later step -------------------------------------------------------------
C = pd.read_csv(RES / "C_first_turn.csv")
fig, ax = plt.subplots(figsize=(7.2, 3.5))
ax.axhline(0, color=MUTED, lw=1)
for j in range(1, 6):
    d = C[C.step == j]
    ax.plot([j, j], [d.delta_r2_shuffled.max(), d.delta_r2.max()], color=RULE, lw=2, zorder=1)
    ax.scatter([j], [d.delta_r2.max()], s=60, color=S1, edgecolor="white", linewidth=1.5, zorder=3,
               label="real labels" if j == 1 else None)
    ax.scatter([j], [d.delta_r2_shuffled.max()], s=60, color="white", edgecolor=MUTED, linewidth=1.5, zorder=3,
               label="shuffled labels" if j == 1 else None)
ax.set_xticks(range(1, 6), [f"step {j}" for j in range(1, 6)])
ax.set_ylabel("ΔR² over H_target + wording\n(best of 110 turn-1 cells)")
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.savefig(OUT / "C_first_turn.png"); plt.close(fig)

# 6 · D: transfer to self-chosen horizons, by layer ------------------------------------------------------------
D = pd.read_csv(RES / "D_transfer.csv")
fig, ax = plt.subplots(figsize=(7.4, 3.5))
for pos_set, c, lab in ((BOUNDARY, S1, "boundary window (before the step)"), (["H"], S2, "'Time horizon:' header (title already written)")):
    d = D[D.pos.isin(pos_set)].groupby("layer").rho.max()
    ax.plot(d.index, d.values, color=c, lw=2, marker="o", ms=5, label=lab)
ax.set_ylim(0, 1); ax.set_xlabel("layer"); ax.set_ylabel("Spearman ρ, predicted vs chosen")
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.savefig(OUT / "D_transfer.png"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
