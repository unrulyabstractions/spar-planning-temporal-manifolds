"""E14 post hoc (2026-11-01; 22_results_phrase_steering.md): what did the phrase step leave out?

Saved states and traces only; runs no model.

1. Decomposition. For each prompt with a donor, the real change of the phrase state when the horizon text changes
   from the prompt's own to the donor's. It is estimated as the difference of the mean states of the pooled prompts
   with those texts, after removing the fit's regime, option-order and delay contributions, which differ between
   prompts with the same text. That change is compared with:
     - the duration step the run used (the effects curve, unit held at its average), and
     - the duration step plus the fit's own unit-factor change (donor unit minus own unit).
   R² = 1 - |change - approximation|² / |change|². Medians over prompts, per role, at L16.
2. Unit-word probe: a logistic probe of the unit word on the unit-token states (5-fold accuracy reported), applied
   to the prompt's own state plus each step. Does the step move the state across a unit boundary?
3. Reachability: can the target regime be written in the prompt's own unit with a number from 1 to 999? Landing,
   per direction, split by that.
4. Step size: the share of effects traces whose first stated horizon moved toward the target, by step size
   (decades), in thirds per direction.

    python -m pipeline.s1_data_vectors.analysis.exp14_geometry /home/Jared/SPAR/results/horizon/exp14_Qwen3-8B_phrase
    -> <dir>/geometry.json, geometry.txt, geometry.png
"""

import json
import sys
from pathlib import Path

import numpy as np

from pipeline.s3_analysis.by_label import exp14_analysis as ea
from spar_horizon.horizons import UNIT_YEARS, canonical_unit
from spar_horizon.steering import curve_delta, fit_duration_effects, span_part
from pipeline.s3_analysis.by_datatype.written_unit import durations

LAYER = 16
UP = ("none-near", "near-both")      # transitions whose target is a longer horizon
MAX_NUMBER = 999                     # the largest number a digit edit is taken to reach in the own unit
LARGE = 2.0                          # decades: a step that moves the stated horizon about half the time
NOT_FIT = ("unit",)                  # the factors the residualized states keep


def unit_of(horizon_text):
    return canonical_unit(horizon_text.split()[-1])


def level_effects(cell, factor):
    """{level: effect [d]} of a fit factor, the first (dropped) level at zero."""
    levels, effects = cell["factors"][factor]
    return dict(zip(levels, np.vstack([np.zeros(effects.shape[1]), effects])))


def residualized(X, cell, factors, covariates):
    """X without the fit's contributions of every factor but the unit, and of the covariates."""
    out = X.copy()
    for name, labels in factors.items():
        if name not in NOT_FIT:
            effects = level_effects(cell, name)
            out -= np.array([effects[label] for label in labels])
    for name, values in covariates.items():
        out -= np.outer(values - values.mean(), cell["covariates"][name])
    return out


def median_finite(values):
    """Median of the finite values; NaN when there are none (a cosine with no unit change is undefined)."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def r2(change, approximation):
    return float(1 - ((change - approximation) ** 2).sum() / (change ** 2).sum())


def cosine(a, b):
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norms) if norms > 0 else float("nan")


def decompose(change, step, unit_change, factor_span):
    """How much of a real text change [d] the duration step, and the step plus the unit change, account for;
    and how the step, and its invariant part (the step minus its part in the factor span), line up with the
    unit change."""
    invariant = step - span_part(step[None], factor_span)[0]
    return {"r2_step": r2(change, step), "r2_step_unit": r2(change, step + unit_change),
            "cos_step": cosine(change, step),
            "unit_share": float((unit_change ** 2).sum() / (change ** 2).sum()),
            "cos_step_unit": cosine(step, unit_change), "cos_invariant_unit": cosine(invariant, unit_change),
            "invariant_norm_share": float((invariant ** 2).sum() / (step ** 2).sum())}


def reachable_in_own_unit(row):
    """Can a number from 1 to MAX_NUMBER in the prompt's own unit fall inside the target regime?"""
    unit = durations(row["horizon_text"])[0][1]
    near, far = row["near_delay_years"], row["far_delay_years"]
    low, high = {"none": (0.0, near), "near": (near, far), "both": (far, np.inf)}[row["target_regime"]]
    return max(low, UNIT_YEARS[unit]) < min(high, MAX_NUMBER * UNIT_YEARS[unit])


def thirds(size, value):
    """[(low, high, n, mean value)] over the three equal-count bins of `size`."""
    edges = np.quantile(size, [0, 1 / 3, 2 / 3, 1])
    bins = []
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        inside = (size >= low) & ((size < high) if i < 2 else (size <= high))
        bins.append((float(low), float(high), int(inside.sum()), float(value[inside].mean())))
    return bins


# ---- the run ----

def load_fit_inputs(directory):
    """States, the pooled records' columns and the E14 prompts, as steering_prepare --phrase saw them."""
    from pipeline.s1_data_vectors.analysis.steering_prepare import OUT, phrase_columns, selected_records, varying_factors   # heavy imports

    inputs = json.loads((directory / "inputs.json").read_text())
    z = np.load(directory / "states.npz")
    prompts_path = Path(inputs["prompts_file"])
    if not prompts_path.exists():   # a path on the box; the file ships in steering/
        prompts_path = OUT / prompts_path.name
    records = selected_records(json.loads(prompts_path.read_text()))[:len(z["sample_uids"])]
    columns = phrase_columns(records)
    row_of = {uid: i for i, uid in enumerate(z["sample_uids"])}
    return {"inputs": inputs, "states": z, "records": records, "columns": columns,
            "factors": varying_factors(columns), "rows": np.array([row_of[u] for u in inputs["sample_uids"]]),
            "steps": np.load(directory / "inputs.npz")}


def geometry(loaded):
    """Sections 1 and 2 per role at LAYER."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    columns, factors, prompts, rows = loaded["columns"], loaded["factors"], loaded["inputs"]["prompts"], loaded["rows"]
    covariates = {"log near delay": columns["log_near"], "log far delay": columns["log_far"]}
    texts = np.array([r["horizon_text"] for r in loaded["records"]])
    up = np.array([p["transition"] in UP for p in prompts])
    halves = np.random.default_rng(0).permutation(len(texts)) % 2 == 0
    groups = {"all donors": lambda g: True, "donor in the same unit": lambda g: g["same_unit"],
              "unit changes, up": lambda g: not g["same_unit"] and g["up"],
              "unit changes, down": lambda g: not g["same_unit"] and not g["up"]}

    def summarize(per_prompt):
        return {name: {"n": len(sel), **{k: median_finite([g[k] for g in sel]) for k in
                                         ("r2_step", "r2_step_unit", "cos_step", "unit_share", "cos_step_unit",
                                          "cos_invariant_unit", "invariant_norm_share")}}
                for name, keep in groups.items() if (sel := [g for g in per_prompt if keep(g)])}

    def per_prompt_changes(X, fit_rows, eval_rows):
        """decompose() per E14 prompt with a donor: the fit on `fit_rows`, the text change measured on `eval_rows`."""
        cell = fit_duration_effects(X[fit_rows], columns["duration"][fit_rows],
                                    {name: labels[fit_rows] for name, labels in factors.items()}, orthogonal=False,
                                    covariates={name: values[fit_rows] for name, values in covariates.items()})
        adjusted = residualized(X[eval_rows], cell, {name: labels[eval_rows] for name, labels in factors.items()},
                                {name: values[eval_rows] for name, values in covariates.items()})
        eval_texts, unit_effect, out = texts[eval_rows], level_effects(cell, "unit"), []
        for i, p in enumerate(prompts):
            donor, own = p["donor_horizon_text"], texts[rows[i]]
            if donor is None or not (eval_texts == donor).any() or not (eval_texts == own).any():
                continue
            change = adjusted[eval_texts == donor].mean(0) - adjusted[eval_texts == own].mean(0)
            step = curve_delta(cell, np.array([p["log_h"]]), np.array([p["donor_log_h"] - p["log_h"]]))[0]
            out.append({"up": bool(up[i]), "same_unit": unit_of(donor) == unit_of(own),
                        **decompose(change, step, unit_effect[unit_of(donor)] - unit_effect[unit_of(own)],
                                    cell["factor_span"])})
        return out

    # how much of the variation in log duration the unit word accounts for: with the unit a factor, the curve is
    # identified from the variation left within a unit, where the duration is the written number
    duration, units = columns["duration"], factors["unit"]
    within_unit = sum(((duration[units == u] - duration[units == u].mean()) ** 2).sum() for u in np.unique(units))
    out = {"unit_share_of_log_duration_variance": float(1 - within_unit / ((duration - duration.mean()) ** 2).sum())}
    for role in ("number", "unit"):
        X = loaded["states"][f"L{LAYER}|{role}"].astype(np.float64)
        everything = np.ones(len(X), bool)
        cell = fit_duration_effects(X, columns["duration"], factors, orthogonal=False, covariates=covariates)
        log_h = np.array([p["log_h"] for p in prompts])
        target = np.array([p["target_log_h"] for p in prompts])
        saved = loaded["steps"][f"step|L{LAYER}|effects|{role}"]
        out[role] = {
            "refit_max_relative_error": float(np.abs(curve_delta(cell, log_h, target - log_h) - saved).max()
                                              / np.abs(saved).max()),
            "decomposition": summarize(per_prompt_changes(X, everything, everything)),
            # fit on one half of the pooled prompts, text change measured on the other, both ways
            "cross_fitted": summarize(per_prompt_changes(X, halves, ~halves) + per_prompt_changes(X, ~halves, halves))}
        if role == "unit":
            labels = factors["unit"]
            probe = LogisticRegression(max_iter=3000, C=0.1)
            accuracy = float(cross_val_score(probe, X, labels, cv=5).mean())
            probe.fit(X, labels)
            own_units = labels[rows]
            changed = {kind: probe.predict(X[rows] + loaded["steps"][f"step|L{LAYER}|{kind}|unit"]) != own_units
                       for kind in ("effects", "invariant", "random")}
            # not run on the model: the step to the donor's horizon plus the fit's change from the own unit to the
            # donor's. Does the probe then read the donor's unit?
            unit_effect = level_effects(cell, "unit")
            with_donor = [i for i, p in enumerate(prompts) if p["donor_horizon_text"] is not None]
            reads_donor_unit = [
                probe.predict((X[rows[i]] + curve_delta(cell, np.array([prompts[i]["log_h"]]),
                                                        np.array([prompts[i]["donor_log_h"] - prompts[i]["log_h"]]))[0]
                               + unit_effect[unit_of(prompts[i]["donor_horizon_text"])] - unit_effect[own_units[i]])[None])[0]
                == unit_of(prompts[i]["donor_horizon_text"]) for i in with_donor]
            out["unit_probe"] = {"cv_accuracy": accuracy, "changed_after_step": {
                kind: {"up": float(c[up].mean()), "down": float(c[~up].mean())} for kind, c in changed.items()},
                "step_plus_unit_reads_donor_unit": {"n": len(with_donor), "share": float(np.mean(reads_donor_unit))}}
    return out


def behaviour(data):
    """Sections 3 and 4 from the traces (exp14_analysis.load)."""
    steered = [(t, row) for (t, c), rows in data.items() if c in ea.STEERED for row in rows.values()]
    reach = {}
    for reachable in (True, False):
        for direction in ("up", "down"):
            group = [row for t, row in steered if (t in UP) == (direction == "up")
                     and reachable_in_own_unit(row) == reachable]
            reach[f"{'reachable' if reachable else 'needs a unit change'}, {direction}"] = {
                "n": len(group), "moved_toward": float(np.mean([r["stated_toward"] for r in group])),
                "landed": float(np.mean([r["states_target"] for r in group])),
                "landed_with_unit_change": float(np.mean([r["states_target"] and r["stated_unit_changed"]
                                                          for r in group]))}
    effects = [(t, row) for (t, c), rows in data.items() if c.startswith("effects") for row in rows.values()]
    size = np.array([abs(row["target_log_h"] - row["log_h"]) for _, row in effects])
    moved = np.array([row["stated_toward"] for _, row in effects], float)
    is_up = np.array([t in UP for t, _ in effects])
    # how far a moved stated horizon goes, within the prompt's unit and with a new unit
    moved_rows = [row for _, row in effects if row["stated_moved"]]
    distance = {}
    for name, keep in (("same unit", lambda r: not r["stated_unit_changed"]), ("new unit", lambda r: r["stated_unit_changed"])):
        group = [r for r in moved_rows if keep(r)]
        shift = np.array([abs(r["stated_shift"]) for r in group])
        step = np.array([abs(r["target_log_h"] - r["log_h"]) for r in group])
        distance[name] = {"n": len(group), "prompts": len({(r["transition"], r["sample_uid"]) for r in group}),
                          "r_shift_step": float(np.corrcoef(shift, step)[0, 1]) if len(group) > 2 else float("nan"),
                          "median_decades": float(np.median(shift)),
                          "quartiles": [float(q) for q in np.percentile(shift, [25, 75])],
                          "median_share_of_step": float(np.median(shift / step))}
    # large steps only (>= LARGE decades), split by direction and reachability: magnitude vs unit
    large = {}
    for direction in ("up", "down"):
        for reachable in (True, False):
            group = [row for t, row in effects if (t in UP) == (direction == "up") and reachable_in_own_unit(row) == reachable
                     and abs(row["target_log_h"] - row["log_h"]) >= LARGE]
            if group:
                large[f"{direction}, {'own unit' if reachable else 'needs a unit change'}"] = {
                    "n": len(group), "moved_toward": float(np.mean([r["stated_toward"] for r in group])),
                    "landed": float(np.mean([r["states_target"] for r in group]))}
    return {"landing_by_reachability": reach,
            "moved_by_step_size": {"up": thirds(size[is_up], moved[is_up]),
                                   "down": thirds(size[~is_up], moved[~is_up])},
            "digit_edits": {"moved": len(moved_rows),
                            "moved_prompts": len({(r["transition"], r["sample_uid"]) for r in moved_rows}),
                            "digit_edit": int(sum(r["digit_edit"] for r in moved_rows)),
                            "digit_edit_prompts": len({(r["transition"], r["sample_uid"]) for r in moved_rows
                                                       if r["digit_edit"]}),
                            "states_a_delay": int(sum(r["stated_is_delay"] for r in moved_rows)),
                            "landed_states_a_delay": int(sum(r["stated_is_delay"] and r["states_target"]
                                                             for r in moved_rows))},
            "moved_distance": distance, "large_steps": large}


def render(result):
    lines = [f"E14 post hoc geometry at L{LAYER} (exp14_geometry.py). Medians over the prompts with a donor.",
             f"The unit word accounts for {result['geometry']['unit_share_of_log_duration_variance']:.2f} of the variance of log"
             " duration over the fit's prompts (η²), so the curve is fitted from the variation within a unit.",
             "R² of the real text change (residualized mean states, own text -> donor text) by the duration step,",
             "and by the step plus the fit's unit-factor change; unit share = |unit change|² / |text change|².", ""]
    for role in ("number", "unit"):
        entry = result["geometry"][role]
        lines.append(f"{role} tokens (refit reproduces the run's step to {entry['refit_max_relative_error']:.0e}):")
        for which, title in (("decomposition", "in sample"), ("cross_fitted", "cross-fitted (fit on one half, measure on the other)")):
            lines.append(f"  {title}:")
            for name, g in entry[which].items():
                lines.append(f"    {name:24s} n {g['n']:3d}  R² step {g['r2_step']:+.2f}  R² step + unit "
                             f"{g['r2_step_unit']:+.2f}  cos(change, step) {g['cos_step']:.2f}  unit share {g['unit_share']:.2f}")
                lines.append(f"    {'':24s}        cos(step, unit change) {g['cos_step_unit']:+.2f}  cos(invariant part, unit"
                             f" change) {g['cos_invariant_unit']:+.2f}  |invariant|²/|step|² {g['invariant_norm_share']:.2f}")
    probe = result["geometry"]["unit_probe"]
    lines += ["", f"unit-word probe on the unit-token states: 5-fold accuracy {probe['cv_accuracy']:.3f}",
              "share of prompts whose probed unit changes after the step:"]
    for kind, c in probe["changed_after_step"].items():
        lines.append(f"  {kind:9s} up {c['up']:.2f}  down {c['down']:.2f}")
    plus = probe["step_plus_unit_reads_donor_unit"]
    lines.append(f"  not run: step to the donor's horizon + the fit's unit change reads as the donor's unit in "
                 f"{plus['share']:.2f} of {plus['n']} prompts")
    lines += ["", f"steered traces ({', '.join(ea.STEERED)}): first stated horizon, by whether the target regime can be",
              f"written in the prompt's own unit with a number from 1 to {MAX_NUMBER}"]
    for name, g in result["behaviour"]["landing_by_reachability"].items():
        lines.append(f"  {name:26s} n {g['n']:3d}  moved toward {g['moved_toward']:.2f}  landed {g['landed']:.2f}"
                     f"  (with a unit change {g['landed_with_unit_change']:.2f})")
    lines += ["", "effects traces (L14-18): share moved toward the target, by step size in thirds"]
    for direction, bins in result["behaviour"]["moved_by_step_size"].items():
        for low, high, n, value in bins:
            lines.append(f"  {direction:4s} {low:.1f}-{high:.1f} decades: n {n:3d}  moved {value:.2f}")
    b = result["behaviour"]
    d_ = b["digit_edits"]
    lines += ["", f"effects traces (L14-18) whose stated horizon moved: {d_['moved']} ({d_['moved_prompts']} prompts);"
                  f" a digit edit of the prompt's number in the same unit (6 -> 66, 213 -> 13): {d_['digit_edit']}"
                  f" ({d_['digit_edit_prompts']} prompts)",
              f"moved traces whose stated horizon is within 10% of a payout delay: {d_['states_a_delay']}"
              f" ({d_['landed_states_a_delay']} of them landed)",
              "how far the stated horizon moved (decades):"]
    for name, d in b["moved_distance"].items():
        lines.append(f"  {name:9s} n {d['n']:3d} ({d['prompts']} prompts)  median {d['median_decades']:.2f} (IQR "
                     f"{d['quartiles'][0]:.2f}-{d['quartiles'][1]:.2f})  median share of the step "
                     f"{d['median_share_of_step']:.2f}  r(shift, step size) {d['r_shift_step']:+.2f}")
    lines.append(f"steps of {LARGE:.0f}+ decades, by direction and whether the target fits in the prompt's own unit:")
    for name, g in b["large_steps"].items():
        lines.append(f"  {name:26s} n {g['n']:3d}  moved toward {g['moved_toward']:.2f}  landed {g['landed']:.2f}")
    return "\n".join(lines) + "\n"


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
    roles = ("unit", "number")
    width = 0.36
    for j, (key, label, color) in enumerate((("r2_step", "the step that was steered", "#2a78d6"),
                                             ("r2_step_unit", "the step + the fit's unit term", "#eb6834"))):
        values = [result["geometry"][role]["cross_fitted"]["all donors"][key] for role in roles]
        left.bar(np.arange(len(roles)) + (j - 0.5) * width, values, width * 0.9, color=color, label=label)
        for x, v in zip(np.arange(len(roles)) + (j - 0.5) * width, values):
            left.annotate(f"{v:+.2f}", (x, max(v, 0)), xytext=(0, 3), textcoords="offset points", ha="center",
                          fontsize=8, color="#3d3d3a")
    left.set_xticks(range(len(roles)), [f"{r} token" for r in roles], fontsize=8)
    left.set_ylim(-0.1, 1)
    left.axhline(0, color="#7a7974", lw=0.8)
    left.set_ylabel("R² of the real text change (median, cross-fitted)", fontsize=8)
    left.set_title("How much of a real change of horizon text the step covers (L16)", fontsize=9)
    left.legend(fontsize=7, frameon=False, loc="upper left")
    for direction, color in (("up", "#2a78d6"), ("down", "#eb6834")):
        bins = result["behaviour"]["moved_by_step_size"][direction]
        right.plot([(lo + hi) / 2 for lo, hi, _, _ in bins], [v for *_, v in bins], "o-", color=color, lw=2, ms=6,
                   label=f"{direction}-steps")
    right.set_ylim(0, 1)
    right.set_xlabel("step size (decades to the target)", fontsize=8)
    right.set_ylabel("share whose stated horizon moved toward the target", fontsize=8)
    right.set_title("Share moved, by step size (effects L14-18; across prompts:\nthe largest up-steps are mostly"
                    " none→near prompts with short horizons)", fontsize=9)
    right.legend(fontsize=7, frameon=False)
    for ax in (left, right):
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(directory):
    directory = Path(directory)
    _, data = ea.load(directory)
    result = {"geometry": geometry(load_fit_inputs(directory)), "behaviour": behaviour(data)}
    (directory / "geometry.json").write_text(json.dumps(result, indent=1))
    text = render(result)
    (directory / "geometry.txt").write_text(text)
    plot(result, directory / "geometry.png")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
