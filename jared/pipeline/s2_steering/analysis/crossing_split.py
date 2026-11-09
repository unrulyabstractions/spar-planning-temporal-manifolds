"""Is pristine steering horizon-specific? The crossing split of 15 §3 on stage A (16 §5).

A horizon edit should change the choice only for prompts whose target horizon
(log h + α) crosses a delay, by about what the model's own behaviour does across that
delay; a decision push moves every prompt. Per cell and direction, from the saved
per-prompt effects of stage A (`exp8_Qwen3-8B_pristine/results.json`):

  target regime   regime(log h + α, clamped to the curve as curve_delta clamps it)
  crossers        prompts whose target regime differs from their own; stayers keep it
  predicted sign  s = sign(behaviour-map mean of the target regime - of the start regime)
  stratum         (start regime, α), used when its crossers all reach one target regime
                  and crossers and stayers each number >= MIN_GROUP
  gap G           Σ_strata w · (mean s·e over crossers - over stayers),
                  w ∝ n_cross · n_stay / (n_cross + n_stay), normalized to 1

The behaviour map is the mean near log-odds per regime on the pooled E0 runs (full,
units, fill; natural closes only with thinking on). Stage A's prompts are near and both
regime only, so the map, not the run's baseline, supplies `none`.

Criteria (15 §3, with the pooled effects curve in place of the horizon-average curve,
16 §5). Confirmation cells: effects-curve effect toward rational at |α| = 2 >= 0.05 C.
  A1  G(effects curve) >= 0.05 C with CI above 0; reported for the invariant part too.
  A2  as written it compares with E8's decision direction, which stage A did not steer.
      Reported instead, post hoc: paired differences of the invariant part against the
      in-span part, unit U and the mean of the random directions.
  A3  a prompt-clustered regression of s·e on crossing, |baseline near log-odds|, step
      length and forced: the crossing coefficient; G on natural closes (thinking on).
  A4  at α = -2 on both-regime prompts, e(-> none) - e(-> near).

    python -m pipeline.s2_steering.analysis.crossing_split /home/Jared/SPAR/results/horizon
    -> <root>/analysis/crossing_split_pristine.{json,txt,png}
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
from pipeline.s2_steering.experiments.ablate_steer.exp8_steering import load_inputs, selected_records
from pipeline.s1_data_vectors.analysis.pristine_manifold import POOLED_BANKS
from pipeline.s2_steering.analysis.steering_analysis import saved_decision
from spar_horizon import steering as S
from spar_horizon.full_bank import regime

INPUTS = ROOT / "steering" / "inputs_Qwen3-8B_pristine.npz"
RUN = "exp8_Qwen3-8B_pristine"
MIN_GROUP = 10
PASS = 0.05
N_BOOT = 2000
CONTROLS = ("in-span part", "unit U", "random")   # the invariant part against each (post hoc A2)


def behaviour_map(root, mode):
    """{regime: mean near log-odds} over the pooled E0 runs: horizon prompts, the fill's
    anchors dropped, natural closes only with thinking on."""
    near, regimes = [], []
    for bank in POOLED_BANKS:
        run = root / f"exp0_Qwen3-8B_think-{mode}_{bank}"
        if not run.exists():
            continue
        z = np.load(run / "activations.npz")
        keep = ~np.isnan(z["horizons"])
        rows = [a for a, k in zip(json.loads((run / "answers.json").read_text()), keep) if k]
        lo = saved_decision(run, z)[1]
        use = ~np.isnan(lo) & ~np.array([r.get("anchor", False) for r in rows])
        if mode == "on":
            use &= ~z["forced_mask"][keep].astype(bool)
        near.append(lo[use])
        regimes.append(np.array([r["regime"] for r in rows])[use])
    near, regimes = np.concatenate(near), np.concatenate(regimes)
    return {r: float(near[regimes == r].mean()) for r in np.unique(regimes)}


def target_regimes(records, alpha, grid):
    """The regime of each prompt's horizon moved by alpha decades, clamped to the curve's
    range as curve_delta clamps the step."""
    log_h = np.log10([r["horizon"] for r in records])
    target = np.clip(log_h + alpha, grid.min(), grid.max())
    return np.array([regime(10 ** t, r["near_delay_years"], r["far_delay_years"]) for t, r in zip(target, records)])


def strata(regimes, targets, behaviour):
    """(used, skipped): used is a list of dicts with the start regime, alpha, target,
    predicted sign and the crosser / stayer masks; skipped says why a stratum was not used.
    `targets` is {alpha: target regimes}."""
    used, skipped = [], []
    for alpha, target in targets.items():
        for start in np.unique(regimes):
            here = regimes == start
            cross, stay = here & (target != start), here & (target == start)
            reached = sorted(str(t) for t in set(target[cross]))
            name = f"{start}@{alpha:+d}"
            if len(reached) != 1:
                skipped.append(f"{name}: crossers reach {reached or 'nothing'}")
            elif cross.sum() < MIN_GROUP or stay.sum() < MIN_GROUP:
                skipped.append(f"{name}: {cross.sum()} crossers, {stay.sum()} stayers")
            else:
                sign = float(np.sign(behaviour[reached[0]] - behaviour[start]))
                used.append({"name": name, "start": start, "alpha": alpha, "target": reached[0], "sign": sign,
                             "cross": cross, "stay": stay})
    return used, skipped


def stratum_weights(used):
    w = np.array([s["cross"].sum() * s["stay"].sum() / (s["cross"].sum() + s["stay"].sum()) for s in used])
    return w / w.sum()


def pooled_means(effects, used, counts):
    """(crossers, stayers): the stratum-weighted means of s·e. `effects` is {alpha: [n]};
    `counts` [n] is how often each prompt is drawn (all ones for the plain estimate)."""
    w = stratum_weights(used)
    cross = stay = 0.0
    for weight, s in zip(w, used):
        oriented = s["sign"] * effects[s["alpha"]]
        cross += weight * (counts * oriented)[s["cross"]].sum() / counts[s["cross"]].sum()
        stay += weight * (counts * oriented)[s["stay"]].sum() / counts[s["stay"]].sum()
    return cross, stay


def gaps(effects_by_direction, used, n_boot=N_BOOT, seed=0):
    """{direction: (crossers, stayers, gap)} on the data, and {direction: [n_boot] gaps}
    from one shared resampling of prompts, so differences between directions are paired."""
    n = len(used[0]["cross"])
    point = {}
    for d, e in effects_by_direction.items():
        c, s = pooled_means(e, used, np.ones(n))
        point[d] = (c, s, c - s)
    rng = np.random.default_rng(seed)
    boot = {d: np.empty(n_boot) for d in effects_by_direction}
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        # every group has >= MIN_GROUP prompts, so an empty group is rare; redraw it
        while any(counts[s["cross"]].sum() == 0 or counts[s["stay"]].sum() == 0 for s in used):
            counts = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        for d, e in effects_by_direction.items():
            c, s = pooled_means(e, used, counts)
            boot[d][b] = c - s
    return point, boot


def interval(samples):
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def crossing_regression(effects, used, base, step_length, forced, n_boot=N_BOOT, seed=0):
    """(coefficient, low, high) of crossing in an OLS of s·e on crossing, |baseline near
    log-odds|, step length and forced (dropped when constant), over the (prompt, stratum)
    rows; the CI resamples prompts."""
    rows = []
    for s in used:
        for i in np.flatnonzero(s["cross"] | s["stay"]):
            rows.append((i, s["sign"] * effects[s["alpha"]][i], float(s["cross"][i]), abs(base[i]),
                         step_length[s["alpha"]][i], float(forced[i])))
    prompt, y, *columns = (np.array(c) for c in zip(*rows))
    columns = [c for c in columns if c.std() > 0]
    X = np.column_stack([np.ones(len(y)), *columns])

    def crossing_coef(take):
        return np.linalg.lstsq(X[take], y[take], rcond=None)[0][1]

    rng = np.random.default_rng(seed)
    prompts = np.unique(prompt)
    by_prompt = {p: np.flatnonzero(prompt == p) for p in prompts}
    boot = [crossing_coef(np.concatenate([by_prompt[p] for p in rng.choice(prompts, len(prompts))]))
            for _ in range(n_boot)]
    return (float(crossing_coef(np.arange(len(y)))), *interval(boot))


def none_minus_near(effects, regimes, target, n_boot=N_BOOT, seed=0):
    """A4: on both-regime prompts at alpha = -2, mean e of those sent to none minus those
    sent to near, bootstrapping prompts within each group."""
    to_none = effects[(regimes == "both") & (target == "none")]
    to_near = effects[(regimes == "both") & (target == "near")]
    if len(to_none) == 0 or len(to_near) == 0:
        return {"n_none": len(to_none), "n_near": len(to_near), "difference": [float("nan")] * 3}
    rng = np.random.default_rng(seed)
    boot = [rng.choice(to_none, len(to_none)).mean() - rng.choice(to_near, len(to_near)).mean()
            for _ in range(n_boot)]
    return {"n_none": len(to_none), "n_near": len(to_near),
            "difference": [float(to_none.mean() - to_near.mean()), *interval(boot)]}


def analyse_cell(cell, steer, rows, base, forced, behaviour, confirm):
    """Everything 15 §3 reports for one cell of the stage A run."""
    regimes = np.array([r["regime"] for r in rows])
    log_h = np.log10([r["horizon"] for r in rows])
    alphas = sorted(int(a) for a in next(iter(steer.values())))
    targets = {a: target_regimes(rows, a, cell["curve_logh"]) for a in alphas}
    used, skipped = strata(regimes, targets, behaviour)
    effects = {d: {a: np.asarray(steer[d][str(a)]["effect"], float) for a in alphas} for d in steer}
    randoms = [d for d in effects if d.startswith("random")]
    effects["random"] = {a: np.mean([effects[d][a] for d in randoms], 0) for a in alphas}
    out = {"confirmation": confirm, "strata": [{k: v for k, v in s.items() if k not in ("cross", "stay")} |
                                               {"n_cross": int(s["cross"].sum()), "n_stay": int(s["stay"].sum())}
                                               for s in used], "skipped": skipped}
    if not used:
        return out
    point, boot = gaps(effects, used)
    out["gap"] = {d: {"crossers": c, "stayers": s, "gap": [g, *interval(boot[d])]} for d, (c, s, g) in point.items()}
    out["invariant minus"] = {d: [point["invariant part"][2] - point[d][2],
                                  *interval(boot["invariant part"] - boot[d])] for d in CONTROLS}
    g, lo, _ = out["gap"]["effects curve"]["gap"]
    gi, loi, _ = out["gap"]["invariant part"]["gap"]
    out["A1"] = {"effects curve": bool(g >= PASS and lo > 0), "invariant part": bool(gi >= PASS and loi > 0)}
    step_length = {a: np.linalg.norm(S.curve_delta(cell, log_h, a), axis=1) for a in alphas}
    # what crossers and stayers differ in besides crossing, per stratum (read with A3)
    out["by stratum"] = {s["name"]: {
        "step length": [float(step_length[s["alpha"]][g].mean()) for g in (s["cross"], s["stay"])],
        "|base|": [float(np.abs(base[g]).mean()) for g in (s["cross"], s["stay"])],
        **{d: [float((s["sign"] * effects[d][s["alpha"]])[g].mean()) for g in (s["cross"], s["stay"])]
           for d in ("effects curve", "invariant part", "random")}} for s in used}
    out["A3 regression"] = {d: crossing_regression(effects[d], used, base, step_length, forced)
                            for d in ("effects curve", "invariant part")}
    if forced.any():
        natural = [{**s, "cross": s["cross"] & ~forced, "stay": s["stay"] & ~forced} for s in used]
        natural = [s for s in natural if s["cross"].sum() >= MIN_GROUP and s["stay"].sum() >= MIN_GROUP]
        if natural:
            p, b = gaps({d: effects[d] for d in ("effects curve", "invariant part")}, natural)
            out["A3 natural closes"] = {d: [p[d][2], *interval(b[d])] for d in p}
    if -2 in targets:
        out["A4"] = {d: none_minus_near(effects[d][-2], regimes, targets[-2])
                     for d in ("effects curve", "invariant part")}
    return out


def ci_text(values):
    value, low, high = values
    return f"{value:+.3f} [{low:+.3f}, {high:+.3f}]"


def report(results, behaviour):
    lines = ["Crossing split of stage A (15 §3 on the pooled manifold, 16 §5). Effects in units of C;",
             "oriented by the behaviour map's predicted sign. Behaviour map (mean near log-odds by regime):"]
    lines += [f"  {mode}: " + "  ".join(f"{r} {v:+.2f}" for r, v in m.items()) for mode, m in behaviour.items()]
    for key, out in results.items():
        lines.append(f"\n=== {key}  ({'confirmation' if out['confirmation'] else 'descriptive'})")
        lines.append("  strata: " + ", ".join(f"{s['name']}->{s['target']} (s {s['sign']:+.0f}, "
                                              f"{s['n_cross']} cross / {s['n_stay']} stay)" for s in out["strata"]))
        if out["skipped"]:
            lines.append("  skipped: " + "; ".join(out["skipped"]))
        if "gap" not in out:
            continue
        lines.append(f"  {'direction':16s} {'crossers':>9s} {'stayers':>9s}   gap G [95% CI]")
        for d, g in out["gap"].items():
            lines.append(f"  {d:16s} {g['crossers']:+9.3f} {g['stayers']:+9.3f}   {ci_text(g['gap'])}")
        lines.append("  by stratum, crossers / stayers: step length, |baseline near log-odds|, s·e per direction")
        for name, row in out["by stratum"].items():
            lines.append(f"    {name:8s} " + "  ".join(f"{k} {row[k][0]:.2f}/{row[k][1]:.2f}" if k in ("step length", "|base|")
                                                   else f"{k} {row[k][0]:+.3f}/{row[k][1]:+.3f}" for k in row))
        lines.append(f"  A1 (G >= {PASS} C, CI above 0): effects curve {out['A1']['effects curve']}, "
                     f"invariant part {out['A1']['invariant part']}")
        lines.append("  A2 post hoc, invariant minus: " + "  ".join(
            f"{d} {ci_text(v)}" for d, v in out["invariant minus"].items()))
        lines.append("  A3 crossing coefficient: " + "  ".join(
            f"{d} {ci_text(v)}" for d, v in out["A3 regression"].items()))
        if "A3 natural closes" in out:
            lines.append("  A3 G on natural closes: " + "  ".join(
                f"{d} {ci_text(v)}" for d, v in out["A3 natural closes"].items()))
        if "A4" in out:
            lines.append("  A4 both@-2, e(->none) - e(->near): " + "  ".join(
                f"{d} {ci_text(v['difference'])} (n {v['n_none']}/{v['n_near']})" for d, v in out["A4"].items()))
    return "\n".join(lines)


def plot(results, path):
    keys = [k for k, out in results.items() if "gap" in out]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.6), sharey=True, squeeze=False)
    for ax, key in zip(axes[0], keys):
        gap = results[key]["gap"]
        directions = [d for d in gap if d != "random"]
        x = np.arange(len(directions))
        ax.bar(x - 0.18, [gap[d]["crossers"] for d in directions], 0.34, color="#2a6fdb", label="crossers")
        ax.bar(x + 0.18, [gap[d]["stayers"] for d in directions], 0.34, color="#a8b3c4", label="stayers")
        ax.axhline(0, color="#555", lw=0.8)
        ax.set_xticks(x, directions, rotation=45, ha="right", fontsize=8)
        ax.set_title(key + (" (confirmation)" if results[key]["confirmation"] else ""), fontsize=9)
    axes[0][0].set_ylabel("oriented effect s·e (C)")
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(root):
    root = Path(root)
    meta, cells, post = load_inputs(INPUTS)
    records = selected_records(meta)
    run = json.loads((root / RUN / "results.json").read_text())
    behaviour = {mode: behaviour_map(root, mode) for mode in ("off", "on")}
    rows_by_mode = {"off": records, "on": [records[j] for j in post["record"]]}
    results = {}
    for key, res in run["cells"].items():
        mode = key.split("|")[0]
        confirm = res["toward"]["effects curve"]["2"][0] >= PASS
        results[key] = analyse_cell(cells[key], res["steer"], rows_by_mode[mode], np.asarray(run["base"][mode]),
                                    np.asarray(run["forced"][mode], bool), behaviour[mode], confirm)
    out = root / "analysis"
    text = report(results, behaviour)
    print(text)
    (out / "crossing_split_pristine.txt").write_text(text)
    (out / "crossing_split_pristine.json").write_text(json.dumps({"behaviour": behaviour, "cells": results},
                                                                  indent=1, default=float))
    plot(results, out / "crossing_split_pristine.png")


if __name__ == "__main__":
    main(sys.argv[1])
