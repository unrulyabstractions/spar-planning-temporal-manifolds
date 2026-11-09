"""Is the model's own choice a threshold on the delays, or graded? (2026-10-30; post hoc)

The crossing split of stage A (pipeline/s2_steering/analysis/crossing_split.py) found steering graded: prompts
whose steered horizon stays on the same side of a delay move almost as much per unit of
step as those that cross. Whether that counts against a horizon reading depends on what
the unsteered model does. A model that compares the horizon with the delays jumps where
a delay crosses the horizon; one that weighs them smoothly slopes.

Same model as delay_step.py, with the near log-odds of E0 in place of the code readout:
within a duration (bank and duration fixed effects), with m = log10(delay / horizon),

    near log-odds = a_d + Σ_{near, far} [ b·m + c·max(m, 0) + s·1(m < 0) ] + controls + e

  s         the jump where that delay falls inside the horizon
  b, b + c  the slope inside and outside (continuous at 0, so a kink counts as smooth)
  controls  option order, log unit size, log near reward, log reward ratio (rewards are
            drawn per prompt and move the choice)

Specs as in delay_step.py (hinge, spline in the absolute log delay, steps only). Rows: the
8B full + units + fill runs, anchors dropped; with thinking on, natural closes only. Rows
whose answer lead is not the normal one have no valid label logits and are dropped
(steering_analysis.saved_decision). CIs resample bases (fe_regression).

Figure: the within-duration residual of the near log-odds, binned by the far margin on
prompts whose near delay is inside the horizon, and by the near margin on prompts whose
far delay is outside, so each panel varies one crossing.

    python -m pipeline.s3_analysis.by_label.behaviour_step /home/Jared/SPAR/results/horizon
    -> <root>/analysis/behaviour_step.{json,txt,png}
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.s3_analysis.by_label.confound_sweep import dataset, within
from pipeline.s3_analysis.by_label.delay_margin import binned
from pipeline.s3_analysis.by_label.delay_step import BANKS, MARGIN_BINS, MODEL, SPECS, fmt, row_records, step_fits
from spar_horizon.e0_runs import NORMAL_LEAD
from pipeline.s3_analysis.by_datatype.unit_overlap import load_run

CONTROL_COLUMNS = {"near first": 4, "log unit size": 5, "log reward": 2, "log ratio": 3}   # dataset feature columns
FORCED = 8
COLORS = {"off": "#2a78d6", "on": "#eb6834"}   # dataviz reference palette slots 1-2, as delay_step


def near_log_odds(root, mode):
    """The near log-odds per dataset() row (same banks, anchors dropped in the same
    order), NaN where the answer lead makes the saved label logits the wrong pair."""
    out = []
    for bank in BANKS:
        z, labels = load_run(root, mode, bank, MODEL)
        lo = np.where(z["lead"][labels["rows"]] == NORMAL_LEAD, labels["near_log_odds"], np.nan)
        out.append(lo[~labels["anchor"]])
    return np.concatenate(out)


def fit_subset(lo, data, rows, rng):
    """Step fits (in log-odds) and margin bins on `rows`."""
    y, V = data["y"][rows], data["features"][rows]
    groups, bases = data["durations"][rows], data["pairs"][rows]
    margins = {"near": V[:, 0] - y, "far": V[:, 1] - y}
    log_delays = {"near": V[:, 0], "far": V[:, 1]}
    controls = {name: V[:, j] for name, j in CONTROL_COLUMNS.items()}
    residual = within(lo[rows], groups)
    # each panel varies one crossing: the far delay with the near one inside, the near
    # delay with the far one outside
    near_inside, far_outside = margins["near"] < 0, margins["far"] >= 0
    return {"n": int(rows.sum()), "bases": int(len(np.unique(bases))),
            "fits": step_fits(lo[rows], 1.0, margins, log_delays, controls, groups, bases, rng),
            "bins": {"far": binned(residual[near_inside], margins["far"][near_inside], bases[near_inside], rng,
                                   MARGIN_BINS),
                     "near": binned(residual[far_outside], margins["near"][far_outside], bases[far_outside], rng,
                                    MARGIN_BINS)}}


def summary_lines(results):
    lines = ["Near log-odds of the unsteered 8B against the delay margins, within duration; * = 95% CI",
             "(bases resampled) excludes 0. Step = jump where the delay falls inside the horizon (inside minus",
             "outside); slopes per decade of margin, below 0 (inside) and the change above. C = mean near",
             "log-odds of near-regime minus both-regime prompts on these rows.", ""]
    for mode, by_subset in results.items():
        for which, m in by_subset.items():
            lines.append(f"thinking {mode}, {which}: n {m['n']}, {m['bases']} bases, C {m['C']:.2f}")
            for spec in SPECS:
                shown = [k for k in m["fits"][spec] if "spline" not in k and k not in CONTROL_COLUMNS]
                lines.append(f"  {spec:9s} " + " | ".join(f"{k} {fmt(m['fits'][spec][k])}" for k in shown))
            far = m["fits"]["hinge"]
            lines.append(f"  in units of C (hinge): far step {far['far step'][0] / m['C']:+.2f}, "
                         f"far slope inside {far['far slope'][0] / m['C']:+.3f}/decade, outside "
                         f"{(far['far slope'][0] + far['far slope change above 0'][0]) / m['C']:+.3f}/decade")
        lines.append("")
    return lines


def plot(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, role in zip(axes, ("far", "near")):
        for mode, by_subset in results.items():
            pts = np.array(by_subset["all"]["bins"][role])
            if len(pts):
                ax.plot(pts[:, 0], pts[:, 1], color=COLORS[mode], lw=2, marker="o", ms=4, label=f"thinking {mode}")
                ax.fill_between(pts[:, 0], pts[:, 2], pts[:, 3], color=COLORS[mode], alpha=0.15, lw=0)
        ax.axhline(0, color="#7a7974", lw=0.8)
        ax.axvline(0, color="#3d3d3a", lw=1.0, ls="--")
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        other = "near delay inside" if role == "far" else "far delay outside"
        ax.set_title(f"{role} delay ({other} the horizon)", fontsize=9)
        ax.set_xlabel(f"log10({role} delay / horizon)   (left of 0: inside the horizon)", fontsize=8)
    axes[0].set_ylabel("near log-odds − duration mean", fontsize=8)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("The unsteered 8B's choice against the delay margins (8B full + units + fill).\n"
                 "A step at the dashed line = a threshold on the delay; a slope = graded.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(root):
    root = Path(root)
    rng = np.random.default_rng(0)
    results = {}
    for mode in ("off", "on"):
        _, data = dataset(root, MODEL, mode, BANKS)
        lo = near_log_odds(root, mode)
        in_range = np.array([r["alan_range"] for r in row_records(root, mode)])
        usable = ~np.isnan(lo) & (data["features"][:, FORCED] < 0.5)
        results[mode] = {}
        for which, rows in (("all", usable), ("Alan range", usable & in_range)):
            out = fit_subset(lo, data, rows, rng)
            regime = data["regime"][rows]
            out["C"] = float(lo[rows][regime == "near"].mean() - lo[rows][regime == "both"].mean())
            results[mode][which] = out
            print(f"thinking {mode}, {which}: done", flush=True)
    out = root / "analysis"
    (out / "behaviour_step.json").write_text(json.dumps(results, indent=1))
    text = "\n".join(summary_lines(results))
    (out / "behaviour_step.txt").write_text(text)
    plot(results, out / "behaviour_step.png")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
