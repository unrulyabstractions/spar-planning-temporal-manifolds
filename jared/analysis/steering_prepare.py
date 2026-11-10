"""E8 inputs, prepared locally (spec: 07_spec_steering.md, section 6): per-cell
directions fitted on a run's saved activations, and the cached thinking-on
reasoning of its near- and both-regime prompts.

    python analysis/steering_prepare.py results/horizon
    -> jared/steering/inputs_Qwen3-8B_answer_slot.npz and .json   (shipped to the box by `just push`)

    python analysis/steering_prepare.py results/horizon --name first_run \
        --cells "off:suffix newline:16,off:pre-label:21,on:think close:19,on:pre-label:21"
    -> the first run's cells (07_spec_steering.md section 3)

    python analysis/steering_prepare.py results/horizon --name decision_L24 --cells on:pre-label:24 --natural-only
    -> fitted on natural think closes only, as in 07_spec_steering.md section 10

    python analysis/steering_prepare.py results/horizon --model Qwen/Qwen3-14B --name patch \
        --cells off:pre-label:27 --no-reasoning
    -> thinking-off inputs only, before the thinking-on run exists (10_spec_scale_14b.md section 3)

    python analysis/steering_prepare.py results/horizon --pooled
    -> inputs_Qwen3-8B_pristine: stage A of 16_spec_pristine_manifold.md §5, the effects
       curve at the off suffix L16, off pre-label L22/L24 and on pre-label L22/L24

    python analysis/steering_prepare.py results/horizon --pooled --stage-b
    -> inputs_Qwen3-8B_pristine_suffix: stage B, the on suffix L16, for exp13_suffix_steer.py

With --pooled the cells come from the pooled banks (full, units and fill, whichever
exist; anchors left out), fitted with steering.fit_duration_effects without the ⟂ step
and with the factors and delay covariates of pristine_manifold.py. Each cell carries
the span of the factor and covariate effects (`factor_span`) and the unit direction U
(`unit_direction`), from which E8 splits the effects curve into its invariant and
in-span parts. Every cell is pooled when the fill run's anchors match the old runs at
the suffix (pristine_manifold.gated_jobs). Steered are natural renderings whose delays
lie in Alan's range (1 week - 50 years; Jared, 2026-10-30): stage A those in the near
and both regimes, stage B two per duration in any regime, with their E0 outcome as the
baseline. Prints, per cell, how the effects step splits.

    python analysis/steering_prepare.py results/horizon --phrase-prompts
    -> steering/phrase_prompts_Qwen3-8B.json: the pooled thinking-on prompts whose horizon-phrase
       states E14 extracts (21_spec_phrase_steering.md); needs no activations

    python analysis/steering_prepare.py --phrase <exp14 folder> [--per-transition 32] [--seed 0]
    -> <folder>/inputs.npz, inputs.json and fit_report.txt: the pooled fit at the phrase tokens
       (L14, L16, L18 x number, unit), the drawn prompts with their targets, and the steps

Every other cell also carries `decision`: the near-vs-far direction inside its horizon
subspace (steering.decision_direction), so E8 can split the curve into its
decision part and the rest. With --no-decision it does not, and E8 steers the
original set instead (probe line, curve, mean difference, random).

Needs only CPU, the run folders (exp0_*_think-off_full, exp0_*_think-on_full, and
with --pooled the units and fill runs too; gen_cache/) and the model's tokenizer.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from exp8_steering import selected_records  # noqa: E402
from horizon_shape import cell_data  # noqa: E402  (same folder)
from steering_analysis import saved_decision  # noqa: E402  (same folder)
from spar_horizon.banks import build_records  # noqa: E402
from spar_horizon.config import Settings  # noqa: E402
from spar_horizon.full_bank import HORIZONS, natural_horizon, regime, render_variant  # noqa: E402
from spar_horizon.horizons import Horizon  # noqa: E402
from spar_horizon.model_io import cache_load, chat_text, label_ids, think_end_id  # noqa: E402
from spar_horizon.steering import (curve_delta, decision_direction, encode, fit_cell,  # noqa: E402
                                   fit_duration_effects, phrase_positions, span_part, unit_direction)

# Second run (07_spec_steering.md section 9): the answer slot where patching says the decision lives
CELLS = "off:pre-label:24,off:pre-label:27,off:pre-label:30,on:pre-label:24,on:pre-label:27,on:pre-label:30"
NAME = "answer_slot"
# 16_spec_pristine_manifold.md §5: stage A at the answer slot and the thinking-off suffix,
# stage B at the thinking-on suffix newline L16, before the comparison forms (18 §10)
PRISTINE_CELLS = "off:suffix newline:16,off:pre-label:22,off:pre-label:24,on:pre-label:22,on:pre-label:24"
PRISTINE_NAME = "pristine"
STAGE_B_CELLS = "on:suffix newline:16"
STAGE_B_NAME = "pristine_suffix"
STAGE_B_PER_DURATION = 2
# the generation settings a stage B trace must share with E0's for E0 to be its baseline
GENERATION_KEYS = ("model", "thinking", "n_response", "max_think_tokens", "prefill", "choice_prefix",
                   "answer_window")
OUT = Path(__file__).resolve().parent.parent / "steering"


def parse_cells(text):
    """"mode:label:layer,..." -> [(mode, label, layer)]."""
    cells = []
    for item in text.split(","):
        mode, label, layer = item.split(":")
        if mode not in ("off", "on"):
            raise ValueError(f"cell {item!r}: mode must be off or on")
        cells.append((mode, label, int(layer)))
    return cells


def run_dir(root, model, mode, bank="full"):
    """The E0 run folder of `model` for `mode` on `bank`; a results folder can hold several models."""
    run = Path(root) / f"exp0_{model.split('/')[-1]}_think-{mode}_{bank}"
    if not (run / "manifest.json").exists():
        raise FileNotFoundError(f"no finished E0 run at {run}")
    return run


def run_settings(root, model, mode, bank="full"):
    """The Settings of the E0 run in `root` for `model`, `mode` and `bank`, cache under root."""
    manifest = json.loads((run_dir(root, model, mode, bank) / "manifest.json").read_text())
    known = {f.name for f in dataclasses.fields(Settings)}
    cfg = Settings(**{k: v for k, v in manifest["settings"].items() if k in known})
    cfg.cache_dir = str(root / "gen_cache")
    return cfg


def matches_e0(g, z, i):
    """Whether the cached Generation of prompt `i` is the one E0 extracted: same think
    length and close, and the same token ids at E0's kept positions. The cache is shared
    by concurrent runs, and batch composition can change a bf16 greedy generation, so a
    cached trace can differ from the one E0's activations were taken on."""
    row = int(np.cumsum(z["kept"])[i]) - 1
    positions = z["keep_idx"][row]
    return (g.think_len == z["think_len"][row] and g.forced == bool(z["forced_mask"][row])
            and positions.max() < len(g.ids) and np.array_equal(g.ids.numpy()[positions], z["keep_ids"][row]))


def cached_reasoning(cfg, records, selected, tokenizer, z):
    """Arrays for the thinking-on prompts in `selected` whose generation is cached
    and equals E0's (`z`, the thinking-on activations):
    their ids up to the label (concatenated, with offsets), the position of the
    </think> token, think length, forced flag, the label token pair in the form
    the model used, and the index into `selected`. Also the indices it could not
    find, and those whose cached generation differs from E0's (dropped, so T4 never
    patches reasoning other than the one the saved activations and decisions belong to). A forced close is "\n</think>\n\n" appended to the cut-off reasoning,
    so </think> is located, not computed, as in extract()."""
    rows, missing, mismatched = [], [], []
    close_id = think_end_id(tokenizer)
    with_space, without_space = label_ids(tokenizer, cfg=cfg), label_ids(tokenizer)
    for j, i in enumerate(selected):
        g = cache_load(cfg, encode(tokenizer, records[i]["prompt"], cfg), None, "cpu")
        if g is None:
            missing.append(i)
            continue
        if not matches_e0(g, z, i):
            mismatched.append(i)
            continue
        ids = g.ids.tolist()
        closes = [p for p in range(g.prompt_len + g.think_len, g.answer_start) if ids[p] == close_id]
        if len(closes) != 1:
            raise ValueError(f"prompt {i}: {len(closes)} </think> tokens between the think span and the label")
        label_text = tokenizer.decode([ids[g.answer_start]])
        rows.append({"ids": ids[:g.answer_start], "close": closes[0], "think_len": g.think_len,
                     "forced": g.forced, "labels": with_space if label_text.startswith(" ") else without_space,
                     "record": j})
    if not rows:
        return None, missing, mismatched
    lengths = [len(r["ids"]) for r in rows]
    arrays = {"ids": np.concatenate([r["ids"] for r in rows]).astype(np.int32),
              "offsets": np.r_[0, np.cumsum(lengths)].astype(np.int64),
              "close": np.array([r["close"] for r in rows]),
              "think_len": np.array([r["think_len"] for r in rows]),
              "forced": np.array([r["forced"] for r in rows]),
              "labels": np.array([r["labels"] for r in rows]),
              "record": np.array([r["record"] for r in rows])}
    return arrays, missing, mismatched


def main(root, out_dir, cells_spec, name, model, natural_only=False, reasoning=True, decision=True):
    root = Path(root)
    cell_list = parse_cells(cells_spec)
    if not reasoning and any(mode == "on" for mode, _, _ in cell_list):
        raise ValueError("thinking-on cells need the thinking-on run; drop --no-reasoning")
    # both modes render the same bank from the same seed, so either run gives the records
    cfg = run_settings(root, model, "on" if reasoning else "off")
    records = build_records(cfg)
    selected = [i for i, r in enumerate(records) if r["regime"] in ("near", "both")]
    arrays, cells = {}, []
    for mode, label, layer in cell_list:
        run = run_dir(root, model, mode)
        z = np.load(run / "activations.npz")
        X, log_h, forced = cell_data(z, label, layer)
        _, near_lo, _ = saved_decision(run, z)
        # forced closes separate perfectly from natural ones at these cells and their rate
        # varies with the horizon, so directions fitted on all prompts partly read "was cut off"
        keep = ~forced if natural_only else np.ones(len(log_h), bool)
        cell = fit_cell(X[keep], log_h[keep])
        has_decision = keep & np.isfinite(near_lo)   # NaN where the saved label logits are the wrong token pair
        decision_vector, decision_r2 = decision_direction(X[has_decision], near_lo[has_decision], cell)
        if decision:
            cell["decision"] = decision_vector
        key = f"{mode}|{label}|{layer}"
        for field, value in cell.items():
            arrays[f"{key}|{field}"] = np.asarray(value)
        cells.append({"mode": mode, "label": label, "layer": layer, "k": int(cell["basis"].shape[0]),
                      "n": int(keep.sum()), "natural_only": natural_only, "decision_r2": decision_r2})
        print(f"{key}: {keep.sum()} prompts{' (natural closes)' if natural_only else ''}, "
              f"horizon-average basis k={cell['basis'].shape[0]}, near log-odds from the basis R² {decision_r2:.2f}")
    post = None
    if reasoning:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        post, missing, mismatched = cached_reasoning(cfg, records, selected, tokenizer,
                                                     np.load(run_dir(root, model, "on") / "activations.npz"))
        if post is not None:
            arrays.update({f"post|{name}": value for name, value in post.items()})
        print(f"cached reasoning: {0 if post is None else len(post['record'])} of {len(selected)} near/both prompts"
              + (f"; missing {len(missing)} (thinking-on test T4 will skip them)" if missing else ""))
        if mismatched:
            print(f"WARNING: dropped {len(mismatched)} prompts whose cached reasoning differs from E0's "
                  f"(thinking-on test T4 will skip them): {mismatched}")
    meta = {"model": cfg.model, "n_prompts": cfg.n_prompts, "seed": cfg.seed, "limit": cfg.limit,
            "selected": selected, "sample_uids": [records[i]["sample_uid"] for i in selected], "cells": cells,
            "n_post": 0 if post is None else int(len(post["record"])), "source": str(root)}
    save_inputs(out_dir, name, arrays, meta)


def save_inputs(out_dir, name, arrays, meta):
    """inputs_<model>_<name>.npz (cell arrays, cached reasoning) and .json (meta)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"inputs_{meta['model'].split('/')[-1]}_{name}.npz"
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) and {path.with_suffix('.json').name}")


# --- pooled banks: the effects curve and its split (16_spec_pristine_manifold.md §5) -----

def pristine_cell(X, duration, log_h, factors, covariates):
    """E8's cell from the effects curve: the fit's duration effects without the ⟂ step,
    the span of its factor and covariate effects, and U. The ridge probe and mean
    difference come from fit_cell on the same rows, for E8's readout column. Also
    returns the old curve (duration averages, no factors) to compare the steps with."""
    effects = fit_duration_effects(X, duration, factors, orthogonal=False, covariates=covariates)
    probe = fit_cell(X, log_h)
    cell = {**{k: effects[k] for k in ("mean", "basis", "curve_logh", "curve_coords", "factor_span")},
            "unit_direction": unit_direction(effects), **{k: probe[k] for k in ("w", "b", "caa")}}
    return cell, fit_duration_effects(X, duration, {}, orthogonal=False)


def split_summary(cell, old, log_h):
    """How the effects step at alpha = +1 and -1 splits over the steered prompts: the mean
    length of the step and of its invariant and in-span parts, the in-span share of its
    squared length, and the mean cosine of each part, and of U, with the old curve's
    step. Steps clamped to zero at a curve's end are left out."""
    steps = np.vstack([curve_delta(cell, log_h, alpha) for alpha in (1, -1)])
    old_steps = np.vstack([curve_delta(old, log_h, alpha) for alpha in (1, -1)])
    unit = np.outer(np.r_[np.ones(len(log_h)), -np.ones(len(log_h))], cell["unit_direction"])
    keep = (np.linalg.norm(steps, axis=1) > 1e-12) & (np.linalg.norm(old_steps, axis=1) > 1e-12)
    steps, old_steps, unit = steps[keep], old_steps[keep], unit[keep]
    in_span = span_part(steps, cell["factor_span"])
    invariant = steps - in_span

    def length(rows):
        return float(np.linalg.norm(rows, axis=1).mean())

    def cos(a, b):
        return float(np.mean((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))

    return {"steps": int(keep.sum()), "effects length": length(steps), "invariant length": length(invariant),
            "in-span length": length(in_span),
            "in-span share": float(np.mean((in_span ** 2).sum(1) / (steps ** 2).sum(1))),
            "cos effects/old": cos(steps, old_steps), "cos invariant/old": cos(invariant, old_steps),
            "cos in-span/old": cos(in_span, old_steps), "cos U/effects": cos(unit, steps),
            "cos U/old": cos(unit, old_steps)}


def steerable(record):
    """Whether a record may be steered (Jared, 2026-10-30): a natural rendering with a
    horizon whose delays lie within Alan's range, 1 week to 50 years, where the curves
    were validated (19 §5). Old-bank prompts have no `alan_range` label: their delays are
    Alan's by construction. Anchors (re-runs of old prompts) are left out."""
    return (record["horizon"] is not None and record.get("natural", True) and not record.get("anchor", False)
            and record.get("alan_range", True))


def stage_a_rows(records):
    """Indices of the records stage A steers: steerable ones in the near or both regime
    (E8's contrast and rational-answer framing)."""
    return [i for i, r in enumerate(records) if steerable(r) and r["regime"] in ("near", "both")]


def stage_b_rows(durations, regimes, per_duration=STAGE_B_PER_DURATION, seed=0):
    """Row indices, `per_duration` per duration: drawn at random, but in distinct regimes
    where the duration has several, so that steering crosses regimes both ways."""
    rng = np.random.default_rng(seed)
    chosen = []
    for duration in np.unique(durations):
        rows = rng.permutation(np.flatnonzero(durations == duration))
        first_of_regime = [r for i, r in enumerate(rows) if regimes[r] not in regimes[rows[:i]]]
        rest = [r for r in rows if r not in first_of_regime]
        chosen += (first_of_regime + rest)[:per_duration]
    return np.array(chosen, dtype=int)


def concatenate_reasoning(blocks):
    """One set of cached_reasoning arrays from per-bank blocks [(arrays or None, offset)],
    with each block's record indices shifted by its offset into the pooled selection."""
    blocks = [(arrays, offset) for arrays, offset in blocks if arrays is not None]
    if not blocks:
        return None
    out = {k: np.concatenate([arrays[k] for arrays, _ in blocks]) for k in ("ids", "close", "think_len", "forced",
                                                                             "labels")}
    out["record"] = np.concatenate([arrays["record"] + offset for arrays, offset in blocks])
    lengths = np.concatenate([np.diff(arrays["offsets"]) for arrays, _ in blocks])
    out["offsets"] = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    return out


def generation_settings(root, model, banks):
    """The thinking-on E0 settings that determine a trace, the same in every bank's run."""
    settings = {bank: run_settings(root, model, "on", bank).to_dict() for bank in banks}
    shared = {k: settings[banks[0]][k] for k in GENERATION_KEYS}
    for bank, values in settings.items():
        if any(values[k] != shared[k] for k in GENERATION_KEYS):
            raise ValueError(f"the thinking-on {bank} run was generated with other settings than the {banks[0]} run")
    return shared


def pooled_main(root, out_dir, cells_spec, name, model, stage_b=False):
    """Steering inputs from the pooled banks (16 §4-§5). Stage A: the steerable near/both
    prompts, with their cached reasoning for thinking-on cells. Stage B: two steerable
    prompts per duration, any regime, with their E0 thinking-on outcome as the baseline.
    Every cell is fitted on the pooled banks (Jared, 2026-10-30), so a mode whose fill
    anchors do not match the old runs at the suffix (pristine_manifold.gated_jobs) stops it."""
    # imported here: pristine_manifold imports off_curve_content, which imports this module
    from pristine_manifold import (POOLED_BANKS, anchor_check, cell_matrix, fit_columns, gated_jobs, load_parts,
                                   pooled_labels)

    root = Path(root)
    cell_list = parse_cells(cells_spec)
    if stage_b and any(mode != "on" for mode, _, _ in cell_list):
        raise ValueError("stage B steers before thinking-on reasoning: its cells must be thinking on")
    modes = sorted({mode for mode, _, _ in cell_list})
    parts = {mode: load_parts(root, mode, "pooled", model.split("/")[-1]) for mode in modes}
    labels = {mode: pooled_labels(parts[mode]) for mode in modes}
    found = {mode: {lab["bank"][0] for _, lab in parts[mode]} for mode in modes}
    banks = [b for b in POOLED_BANKS if all(b in found[mode] for mode in modes)]
    print(f"pooled runs: {banks}" + ("" if "fill" in banks else " (no fill run: the old runs only)"))
    # the gate looks at the suffix newline, before the trace, at every layer used
    gate = sorted({(mode, "suffix newline", layer) for mode, _, layer in cell_list} | set(cell_list))
    anchors = {mode: anchor_check(root, mode, parts[mode], [(p, l) for m, p, l in gate if m == mode])
               for mode in modes}
    print(f"anchor check: {json.dumps(anchors)}")
    if any(job not in gated_jobs(gate, anchors) for job in gate):
        raise ValueError("the fill run's anchors do not match the old runs at the suffix (19 §2): "
                         "its cells cannot be pooled")
    settings = {bank: run_settings(root, model, modes[0], bank) for bank in banks}
    records = {bank: build_records(cfg) for bank, cfg in settings.items()}
    extra = {}
    if stage_b:
        on = labels["on"]
        index = {bank: {r["sample_uid"]: i for i, r in enumerate(records[bank])} for bank in banks}
        candidates = np.array([j for j in range(len(on["y"]))
                               if steerable(records[on["bank"][j]][index[on["bank"][j]][on["sample_uids"][j]]])], int)
        picked = candidates[stage_b_rows(on["durations"][candidates], on["regime"][candidates])]
        rows = [(on["bank"][j], index[on["bank"][j]][on["sample_uids"][j]]) for j in picked]
        answers = {bank: {r["sample_uid"]: r for r in json.loads((run_dir(root, model, "on", bank)
                                                                  / "answers.json").read_text())} for bank in banks}
        extra["baseline"] = [{"near_lo": float(on["near_log_odds"][j]), "forced": bool(on["forced"][j]),
                              "label": answers[on["bank"][j]][on["sample_uids"][j]].get("label"),
                              "think_text": answers[on["bank"][j]][on["sample_uids"][j]].get("think_text", "")}
                             for j in picked]
        extra["generation"] = generation_settings(root, model, banks)
        print(f"stage B: {len(rows)} prompts over {len(np.unique(on['durations'][picked]))} durations "
              f"(of {len(candidates)} steerable), regimes "
              + ", ".join(f"{r} {int((on['regime'][picked] == r).sum())}" for r in ("none", "near", "both"))
              + "; banks " + ", ".join(f"{bank} {int((on['bank'][picked] == bank).sum())}" for bank in banks))
    else:
        rows = [(bank, i) for bank in banks for i in stage_a_rows(records[bank])]
        print(f"stage A: {len(rows)} steerable near/both prompts ("
              + ", ".join(f"{bank} {sum(b == bank for b, _ in rows)}" for bank in banks) + ")")
    log_h = np.log10([records[bank][i]["horizon"] for bank, i in rows])
    arrays, cells = {}, []
    for mode, label, layer in cell_list:
        lab = labels[mode]
        factors, covariates = fit_columns(lab, pooled=True)
        cell, old = pristine_cell(cell_matrix(parts[mode], label, layer), lab["duration"], lab["y"],
                                  factors, covariates)
        split = split_summary(cell, old, log_h)
        key = f"{mode}|{label}|{layer}"
        for field, value in cell.items():
            arrays[f"{key}|{field}"] = np.asarray(value)
        cells.append({"mode": mode, "label": label, "layer": layer, "k": int(cell["basis"].shape[0]),
                      "n": int(len(lab["y"])), "factor_span_dims": int(cell["factor_span"].shape[0]), "split": split})
        print(f"split {key}: {len(lab['y'])} prompts in the fit, k={cell['basis'].shape[0]}, factor span "
              f"{cell['factor_span'].shape[0]} dims; alpha=±1 step length: effects {split['effects length']:.2f}, "
              f"invariant {split['invariant length']:.2f}, in-span {split['in-span length']:.2f} "
              f"(in-span share {split['in-span share']:.2f}); cos with the old curve's step: effects "
              f"{split['cos effects/old']:+.2f}, invariant {split['cos invariant/old']:+.2f}, in-span "
              f"{split['cos in-span/old']:+.2f}, U {split['cos U/old']:+.2f}; cos U/effects {split['cos U/effects']:+.2f}",
              flush=True)
    post = None
    if not stage_b and "on" in modes:
        tokenizer = AutoTokenizer.from_pretrained(model)
        blocks, offset = [], 0
        for bank in banks:
            selected = [i for b, i in rows if b == bank]
            block, missing, mismatched = cached_reasoning(
                run_settings(root, model, "on", bank), records[bank], selected, tokenizer,
                np.load(run_dir(root, model, "on", bank) / "activations.npz"))
            blocks.append((block, offset))
            offset += len(selected)
            print(f"cached reasoning, {bank} bank: {0 if block is None else len(block['record'])} of {len(selected)}"
                  + (f"; missing {len(missing)}" if missing else "")
                  + (f"; WARNING: dropped {len(mismatched)} that differ from E0's" if mismatched else ""))
        post = concatenate_reasoning(blocks)
        if post is not None:
            arrays.update({f"post|{field}": value for field, value in post.items()})
    first = settings[banks[0]]
    meta = {"model": model, "n_prompts": first.n_prompts, "seed": first.seed, "limit": first.limit,
            "selected": [int(i) for _, i in rows], "selected_banks": [str(b) for b, _ in rows],
            "bank_settings": {b: {"n_prompts": c.n_prompts, "seed": c.seed, "limit": c.limit}
                              for b, c in settings.items()},
            "sample_uids": [records[bank][i]["sample_uid"] for bank, i in rows], "cells": cells,
            "n_post": 0 if post is None else int(len(post["record"])), "source": str(root), "pooled": True,
            "anchor_check": anchors, **extra}
    save_inputs(out_dir, name, arrays, meta)


# --- E14: steering at the horizon phrase (21_spec_phrase_steering.md) --------------------

PHRASE_PROMPTS_NAME = "phrase_prompts"
PHRASE_LAYERS = (14, 16, 18)
PHRASE_ROLES = ("number", "unit")
DONOR_LAYER = 16        # the layer of the opposite-step and invariant-part conditions
RANGE_SLACK = 1e-6      # the fit rounds its durations to 9 decimals; far = 50 years + 1 decade is the curve's end
MIN_DELAY_GAP = 0.6     # decades between the delays for a midpoint target: 0.3 from each
# (name, start regime, target regime); the order is also the run order. near-both is drawn
# before near-none, which gives up the prompts it took.
TRANSITIONS = (("both-near", "both", "near"), ("none-near", "none", "near"),
               ("near-both", "near", "both"), ("near-none", "near", "none"))


def phrase_prompts_main(root, out_dir, model):
    """steering/phrase_prompts_<model>.json: every record with a horizon (anchors left out)
    of each pooled bank whose thinking-on run exists, in the shape E8's selected_records reads."""
    # imported here: pristine_manifold imports off_curve_content, which imports this module
    from pristine_manifold import POOLED_BANKS

    root = Path(root)
    banks = [b for b in POOLED_BANKS
             if (root / f"exp0_{model.split('/')[-1]}_think-on_{b}" / "manifest.json").exists()]
    if not banks:
        raise FileNotFoundError(f"no finished thinking-on run of {model} in {root}")
    settings = {bank: run_settings(root, model, "on", bank) for bank in banks}
    records = {bank: build_records(settings[bank]) for bank in banks}
    rows = [(bank, i) for bank in banks for i, r in enumerate(records[bank])
            if r["horizon"] is not None and not r.get("anchor", False)]
    meta = {"model": model, "generation": generation_settings(root, model, banks),
            "bank_settings": {b: {"n_prompts": c.n_prompts, "seed": c.seed, "limit": c.limit}
                              for b, c in settings.items()},
            "selected": [i for _, i in rows], "selected_banks": [b for b, _ in rows],
            "sample_uids": [records[bank][i]["sample_uid"] for bank, i in rows]}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{PHRASE_PROMPTS_NAME}_{model.split('/')[-1]}.json"
    path.write_text(json.dumps(meta))
    print(f"wrote {path}: {len(rows)} prompts (" + ", ".join(f"{b} {meta['selected_banks'].count(b)}" for b in banks) + ")")


def rendering_labels(rec):
    """(duration id, rendered unit) of a record, as unit_overlap.load_run labels the rows:
    the paired banks carry them, a full-bank record is its own natural rendering."""
    if "duration_id" in rec:
        return rec["duration_id"], rec["rendered_unit"]
    h = natural_horizon(rec["horizon"])
    return str(h), h.unit


def phrase_columns(records):
    """The fit's columns from the records (16 §2, §4, as pristine_manifold.fit_columns):
    the duration of each rendering (the grid horizon), the factors unit, regime and option
    order, and the covariates log near and far delay; also each prompt's own log horizon."""
    labels = [rendering_labels(rec) for rec in records]
    return {"duration": np.array([Horizon.parse(duration).log10_years for duration, _ in labels]),
            "log_h": np.log10([rec["horizon"] for rec in records]),
            "log_near": np.log10([rec["near_delay_years"] for rec in records]),
            "log_far": np.log10([rec["far_delay_years"] for rec in records]),
            "regime": np.array([rec["regime"] for rec in records]),
            "factors": {"unit": np.array([unit for _, unit in labels]),
                        "regime": np.array([rec["regime"] for rec in records]),
                        "order": np.array([rec["short_first"] for rec in records])}}


def varying_factors(columns):
    """The fit's factors that take several levels within some duration. A smoke run on the first
    rows of the file holds the full bank only, where the unit follows from the duration and the
    fit cannot separate the two; the unit factor is then left out (the report says so)."""
    return {name: values for name, values in columns["factors"].items()
            if any(len(np.unique(values[columns["duration"] == d])) > 1 for d in np.unique(columns["duration"]))}


def transition_target(target_regime, log_near, log_far):
    """The horizon (log10 years) a prompt is steered to: the log-midpoint of the delays for
    near, one decade past the far delay for both, one decade under the near delay for none."""
    if target_regime == "near":
        return (log_near + log_far) / 2
    return log_far + 1 if target_regime == "both" else log_near - 1


def eligible_rows(columns, steerable_rows, start_regime, target_regime, curve_range):
    """Row indices of the prompts that can make this transition: steerable, in the start
    regime, with a target inside the curve (the fitted durations) and, for a midpoint
    target, delays at least MIN_DELAY_GAP decades apart."""
    target = transition_target(target_regime, columns["log_near"], columns["log_far"])
    ok = steerable_rows & (columns["regime"] == start_regime)
    ok &= (target >= curve_range[0] - RANGE_SLACK) & (target <= curve_range[1] + RANGE_SLACK)
    if target_regime == "near":
        ok &= columns["log_far"] - columns["log_near"] >= MIN_DELAY_GAP
    return np.flatnonzero(ok)


def allocate(counts, total):
    """Draws per stratum: proportional to `counts`, at least one per non-empty stratum
    (when `total` allows), largest remainder, never more than a stratum holds."""
    counts = np.asarray(counts)
    total = min(total, counts.sum())
    quota = total * counts / counts.sum()
    smallest = 1 if total >= (counts > 0).sum() else 0
    drawn = np.minimum(counts, np.maximum(smallest * (counts > 0), np.floor(quota))).astype(int)
    while drawn.sum() < total:
        drawn[np.where(drawn < counts, quota - drawn, -np.inf).argmax()] += 1
    while drawn.sum() > total:   # the minimum of one overshot
        drawn[np.where(drawn > smallest, drawn - quota, -np.inf).argmax()] -= 1
    return drawn


def stratified_draw(rows, banks, per_transition, rng):
    """`per_transition` of `rows`, stratified by the bank of each row (`banks` is per row index)."""
    if len(rows) == 0:
        return rows
    pools = [rng.permutation(rows[banks[rows] == b]) for b in dict.fromkeys(banks[rows])]
    drawn = allocate([len(p) for p in pools], per_transition)
    return np.concatenate([p[:n] for p, n in zip(pools, drawn)])


def draw_prompts(columns, steerable_rows, banks, curve_range, per_transition, seed, allow_short):
    """({transition: drawn row indices}, {transition: eligible count}). A near prompt serves one
    transition: near-none draws from the near prompts near-both left."""
    drawn, eligible, taken = {}, {}, set()
    for index, (name, start, target) in enumerate(TRANSITIONS):
        rows = eligible_rows(columns, steerable_rows, start, target, curve_range)
        eligible[name] = len(rows)
        pool = np.array([r for r in rows if r not in taken], dtype=int)
        if len(pool) < per_transition and not allow_short:
            raise ValueError(f"{name}: {len(pool)} eligible prompts for {per_transition} requested "
                             f"(pass --allow-short to take what exists)")
        drawn[name] = stratified_draw(pool, banks, per_transition, np.random.default_rng([seed, index]))
        if start == "near":
            taken |= set(drawn[name].tolist())
    return drawn, eligible


def opposite_targets(log_h, target, curve_range):
    """The same number of decades the other way, clamped to the curve; (targets, clamped flags)."""
    mirrored = 2 * log_h - target
    opposite = np.clip(mirrored, *curve_range)
    return opposite, ~np.isclose(opposite, mirrored)


def random_like(steps, rng):
    """A Gaussian direction per row, as long as that row of `steps`."""
    noise = rng.normal(size=steps.shape)
    return noise * (np.linalg.norm(steps, axis=1) / np.linalg.norm(noise, axis=1))[:, None]


def find_donor(rec, target_log_h, target_regime, tokenizer, cfg):
    """(horizon text, log10 years) of the horizon nearest the target, in the target regime
    for this prompt's delays, whose phrase has as many tokens as the recipient's; or None."""
    def phrase_length(prompt, horizon_text):
        return len(phrase_positions(tokenizer, chat_text(tokenizer, prompt, cfg), horizon_text))

    recipient_length = phrase_length(rec["prompt"], rec["horizon_text"])
    for h in sorted(HORIZONS, key=lambda h: abs(h.log10_years - target_log_h)):
        if regime(h.years, rec["near_delay_years"], rec["far_delay_years"]) != target_regime:
            continue
        if phrase_length(render_variant({**rec, "horizon_text": str(h)}), str(h)) == recipient_length:
            return str(h), h.log10_years
    return None


def phrase_cell_report(layer, role, cell, X, columns, steps, transition_of):
    """Fit size, factor span and the effects steps' lengths and in-span share, with the
    regime AUC of the fit's first JUDGED_K coordinates (P4; P1-P3 need written traces and
    forced closes, which exist only after reasoning)."""
    from pristine_manifold import JUDGED_K, regime_auc   # imported here: see phrase_prompts_main

    in_span = (span_part(steps, cell["factor_span"]) ** 2).sum(1) / (steps ** 2).sum(1)
    length = np.linalg.norm(steps, axis=1)
    coords = (X - cell["mean"]) @ cell["basis"][:JUDGED_K].T
    return {"layer": layer, "role": role, "n": len(X), "k": int(cell["basis"].shape[0]),
            "factor_span_dims": int(cell["factor_span"].shape[0]), "in_span_share": float(in_span.mean()),
            "step_length": float(length.mean()),
            "by_transition": {name: {"step_length": float(length[transition_of == name].mean()),
                                     "in_span_share": float(in_span[transition_of == name].mean())}
                              for name, _, _ in TRANSITIONS if (transition_of == name).any()},
            "p4_regime_auc": regime_auc(coords, columns["regime"], columns["duration"])}


def phrase_report_text(cells, eligible, per_transition, bank_counts, donors, short, left_out):
    lines = ["E14 pooled fit at the horizon phrase (21_spec_phrase_steering.md §7, task 2)",
             "P1-P3 are left out: P1 needs written traces and P3 forced closes, and neither exists before reasoning.",
             "P4 = largest regime AUC of the first four coordinates within a duration (5-fold, in-sample basis).", ""]
    if left_out:
        lines += [f"WARNING: factors {left_out} follow from the duration in these rows and are not in the fit.", ""]
    for c in cells:
        lines.append(f"L{c['layer']} {c['role']:6s}: {c['n']} rows, k={c['k']}, factor span {c['factor_span_dims']} dims, "
                     f"P4 {c['p4_regime_auc']:.2f}; effects step length {c['step_length']:.1f}, "
                     f"in-span share {c['in_span_share']:.2f}")
        lines += [f"    {name:9s} step length {v['step_length']:.1f}, in-span share {v['in_span_share']:.2f}"
                  for name, v in c["by_transition"].items()]
    lines += ["", f"eligible prompts (before near-none gives up near-both's), requested {per_transition} per transition"
              + (" (short: all that exist were taken)" if short else "") + ":"]
    for name, _, _ in TRANSITIONS:
        lines.append(f"  {name:9s} eligible {eligible[name]:4d}; drawn "
                     + ", ".join(f"{bank} {n}" for bank, n in bank_counts[name].items())
                     + f"; donors found {donors[name][0]} of {donors[name][1]}")
    return "\n".join(lines) + "\n"


def phrase_main(directory, per_transition=32, seed=0, allow_short=False, prompts_file=None):
    """E14's inputs from the phrase states in `directory` (exp14_phrase_states.py): the pooled
    fit per layer and role, the drawn prompts with targets, steps and donors."""
    directory = Path(directory)
    states_meta = json.loads((directory / "states.json").read_text())
    z = np.load(directory / "states.npz")
    prompts_path = Path(prompts_file or states_meta["prompts_file"])
    if not prompts_path.exists():
        prompts_path = OUT / prompts_path.name
    prompts_meta = json.loads(prompts_path.read_text())
    n = states_meta["n"]
    records = selected_records(prompts_meta)[:n]
    if [r["sample_uid"] for r in records] != [str(u) for u in z["sample_uids"]]:
        raise ValueError("the states are not those of the prompts file's first rows")
    banks = np.array(prompts_meta["selected_banks"][:n])
    columns = phrase_columns(records)
    steerable_rows = np.array([steerable(r) for r in records])
    factors = varying_factors(columns)
    left_out = sorted(set(columns["factors"]) - set(factors))
    if left_out:
        print(f"WARNING: factors {left_out} follow from the duration in these rows and are left out of the fit "
              f"(a smoke run on the first rows; the real states pool the units and fill banks)")
    fits = {}
    for layer in PHRASE_LAYERS:
        for role in PHRASE_ROLES:
            X = z[f"L{layer}|{role}"].astype(np.float64)
            fits[layer, role] = fit_duration_effects(X, columns["duration"], factors, orthogonal=False,
                                                     covariates={"log near delay": columns["log_near"],
                                                                 "log far delay": columns["log_far"]})
            print(f"fit L{layer} {role}: k={fits[layer, role]['basis'].shape[0]}", flush=True)
    grid = fits[PHRASE_LAYERS[0], PHRASE_ROLES[0]]["curve_logh"]
    curve_range = (grid[0], grid[-1])
    drawn, eligible = draw_prompts(columns, steerable_rows, banks, curve_range, per_transition, seed, allow_short)
    short = any(len(rows) < per_transition for rows in drawn.values())

    rows = np.concatenate([drawn[name] for name, _, _ in TRANSITIONS]).astype(int)
    transition_of = np.concatenate([[name] * len(drawn[name]) for name, _, _ in TRANSITIONS])
    target_regime = {name: target for name, _, target in TRANSITIONS}
    log_h = columns["log_h"][rows]
    target = np.array([transition_target(target_regime[t], columns["log_near"][r], columns["log_far"][r])
                       for t, r in zip(transition_of, rows)])
    moves_across = np.isin(transition_of, ("both-near", "none-near"))   # none and both starts have an opposite step
    opposite, clamped = opposite_targets(log_h, target, curve_range)

    generation = {**prompts_meta["generation"], "model": states_meta["model"]}
    cfg = Settings(**generation)
    tokenizer = AutoTokenizer.from_pretrained(states_meta["model"])
    donors = [find_donor(records[r], t, target_regime[name], tokenizer, cfg)
              for r, t, name in zip(rows, target, transition_of)]

    arrays, cell_reports = {}, []
    for (layer, role), cell in fits.items():
        effects = curve_delta(cell, log_h, target - log_h)
        kinds = {"effects": effects,
                 "random": random_like(effects, np.random.default_rng([seed, layer, PHRASE_ROLES.index(role)]))}
        if layer == DONOR_LAYER:
            kinds["invariant"] = effects - span_part(effects, cell["factor_span"])
            kinds["opposite"] = np.where(moves_across[:, None], curve_delta(cell, log_h, opposite - log_h), 0.0)
        arrays.update({f"step|L{layer}|{kind}|{role}": step.astype(np.float32) for kind, step in kinds.items()})
        X = z[f"L{layer}|{role}"].astype(np.float64)
        cell_reports.append(phrase_cell_report(layer, role, cell, X, columns, effects, transition_of))
    np.savez(directory / "inputs.npz", **arrays)

    prompts = [{"transition": t, "start_regime": columns["regime"][r], "target_regime": target_regime[t],
                "log_h": float(h), "target_log_h": float(tg),
                "opposite_log_h": float(o) if across else None, "opposite_clamped": bool(c and across),
                "donor_horizon_text": None if d is None else d[0], "donor_log_h": None if d is None else float(d[1])}
               for t, r, h, tg, o, c, across, d in zip(transition_of, rows, log_h, target, opposite, clamped,
                                                        moves_across, donors)]
    meta = {"model": states_meta["model"], "generation": generation,
            "bank_settings": prompts_meta["bank_settings"],
            "selected": [prompts_meta["selected"][r] for r in rows], "selected_banks": [str(banks[r]) for r in rows],
            "sample_uids": [records[r]["sample_uid"] for r in rows], "prompts": prompts,
            "transitions": [name for name, _, _ in TRANSITIONS], "layers": list(PHRASE_LAYERS),
            "eligible": eligible, "left_out_factors": left_out, "cells": cell_reports, "states_file": str(directory / "states.npz"),
            "prompts_file": str(prompts_path), "per_transition": per_transition, "seed": seed}
    (directory / "inputs.json").write_text(json.dumps(meta, indent=1))
    bank_counts = {name: {b: int((banks[drawn[name]] == b).sum()) for b in dict.fromkeys(banks)}
                   for name, _, _ in TRANSITIONS}
    donor_counts = {name: (sum(d is not None for d, t in zip(donors, transition_of) if t == name),
                           len(drawn[name])) for name, _, _ in TRANSITIONS}
    report = phrase_report_text(cell_reports, eligible, per_transition, bank_counts, donor_counts, short, left_out)
    (directory / "fit_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", nargs="?", help="folder with the full-bank E0 runs and gen_cache/")
    parser.add_argument("--out", type=Path, default=OUT, help=f"output folder (default {OUT})")
    parser.add_argument("--cells", default=None,
                        help='"mode:label:layer,..." (default: the answer slot at L24/27/30; '
                             f'with --pooled "{PRISTINE_CELLS}"; with --stage-b "{STAGE_B_CELLS}")')
    parser.add_argument("--name", default=None, help=f"suffix of the inputs file and of E8's output folder "
                                                     f"({NAME}; with --pooled {PRISTINE_NAME}; --stage-b {STAGE_B_NAME})")
    parser.add_argument("--model", default=Settings.model, help=f"model whose E0 runs to read ({Settings.model})")
    parser.add_argument("--natural-only", action="store_true",
                        help="fit directions on prompts whose think block closed on its own")
    parser.add_argument("--no-reasoning", action="store_true",
                        help="thinking-off cells only, without the thinking-on run or its cached reasoning")
    parser.add_argument("--no-decision", action="store_true",
                        help="leave out the decision direction, so E8 steers the probe line set")
    parser.add_argument("--pooled", action="store_true",
                        help="the effects curve on the pooled banks (16_spec_pristine_manifold.md §5), stage A")
    parser.add_argument("--stage-b", action="store_true",
                        help="with --pooled: stage B's prompts and E0 baseline, for exp13_suffix_steer.py")
    parser.add_argument("--phrase-prompts", action="store_true",
                        help="write E14's prompts file (21_spec_phrase_steering.md), the pooled thinking-on prompts")
    parser.add_argument("--phrase", type=Path, default=None, metavar="DIR",
                        help="E14: fit the phrase states in DIR and write its inputs.npz/json and fit_report.txt")
    parser.add_argument("--per-transition", type=int, default=32, help="with --phrase: prompts per transition")
    parser.add_argument("--seed", type=int, default=0, help="with --phrase: seed of the draw and the random steps")
    parser.add_argument("--allow-short", action="store_true",
                        help="with --phrase: take every eligible prompt of a transition that has fewer than requested")
    args = parser.parse_args()
    if args.phrase:
        phrase_main(args.phrase, args.per_transition, args.seed, args.allow_short)
        sys.exit()
    if args.root is None:
        parser.error("root is required (except with --phrase)")
    if args.phrase_prompts:
        phrase_prompts_main(args.root, args.out, args.model)
        sys.exit()
    if args.stage_b and not args.pooled:
        parser.error("--stage-b needs --pooled")
    if args.pooled:
        pooled_main(args.root, args.out, args.cells or (STAGE_B_CELLS if args.stage_b else PRISTINE_CELLS),
                    args.name or (STAGE_B_NAME if args.stage_b else PRISTINE_NAME), args.model, args.stage_b)
    else:
        main(args.root, args.out, args.cells or CELLS, args.name or NAME, args.model, args.natural_only,
             not args.no_reasoning, not args.no_decision)
