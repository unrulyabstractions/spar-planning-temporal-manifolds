"""Inputs of the phrase number-patch experiment (27_spec_phrase_number_patch.md §2): same-unit
crossing pairs, and per pair the manifold step, its random twins and the horizon subspace. CPU only.

    python -m pipeline.s1_data_vectors.analysis.patch_prepare --phrase-dir ../results/horizon/exp14_Qwen3-8B_phrase
    -> steering/exp16_Qwen3-8B_patch/{inputs.json, inputs.npz, fit_report.txt}
       (small; `just push` ships steering/ but not results/)

Needs locally: <phrase-dir>/states.json and states.npz (E14's phrase states, whose L16 number and
unit rows are refitted here as scan_prepare does), the prompts file states.json names
(steering/phrase_prompts_Qwen3-8B.json) and the Qwen3-8B tokenizer from the Hugging Face cache.
No E0 activations, no GPU and no model weights.

Pairs: `--pairs` (24) steerable none-start and both-start prompts, half each, whose delays are
Alan's, with a donor: the same prompt rendered with another horizon in the same unit word, in the
near regime for the prompt's delays, with as many phrase tokens as the prompt's own (digits are
single tokens, so the number tokens align one to one), the one nearest the log-midpoint of the
delays. Drawn like E14 (steering_prepare.stratified_draw, stratified by bank, seed 0) and
interleaved, so that any prefix of the pairs holds both transitions.

Per pair, at the patch layer (16): the fit's `manifold` step along the duration curve from the
prompt's horizon to the donor's, on the number tokens (`number`) and for the unit token (`unit`);
`random`, a Gaussian direction as long as the number step; `inspan`, a Gaussian direction inside the
number fit's top-8 subspace, as long as the number step; and `basis`, that subspace [8, d].
Donor states are not stored: the driver takes them from a forward pass of the donor-rendered
prompt, as E14 does.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from pipeline.s1_data_vectors.analysis.scan_prepare import interleave, load_phrase_states
from pipeline.s1_data_vectors.analysis.steering_prepare import (
    OUT, eligible_rows, fit_phrase_cell, phrase_columns, random_like, steerable, stratified_draw,
    transition_target, varying_factors)
from spar_horizon.banks import selected_records
from spar_horizon.config import Settings
from spar_horizon.full_bank import HORIZONS, regime, render_variant
from spar_horizon.model_io import chat_text
from spar_horizon.steering import curve_delta, phrase_positions

PATCH_LAYER = 16
DEPTH_LAYERS = (12, 22)
PAIRS = 24
PAIR_TRANSITIONS = (("none-near", "none"), ("both-near", "both"))
PATCH_NAME = "exp16_{}_patch"


def phrase_length(tokenizer, cfg, prompt, horizon_text):
    return len(phrase_positions(tokenizer, chat_text(tokenizer, prompt, cfg), horizon_text))


def same_unit_donor(rec, target_log_h, tokenizer, cfg):
    """(horizon text, log10 years) of the horizon nearest `target_log_h` that has the recipient's
    unit word, lies in the near regime for the recipient's delays and has as many phrase tokens as
    the recipient's own; or None."""
    unit_word = rec["horizon_text"].split()[-1]
    recipient_length = phrase_length(tokenizer, cfg, rec["prompt"], rec["horizon_text"])
    for h in sorted(HORIZONS, key=lambda h: abs(h.log10_years - target_log_h)):
        if str(h).split()[-1] != unit_word:
            continue
        if regime(h.years, rec["near_delay_years"], rec["far_delay_years"]) != "near":
            continue
        if phrase_length(tokenizer, cfg, render_variant({**rec, "horizon_text": str(h)}), str(h)) == recipient_length:
            return str(h), h.log10_years
    return None


def in_span_like(steps, basis, rng):
    """A Gaussian direction inside the span of `basis` [k, d] per row, as long as that row of `steps`."""
    noise = rng.normal(size=(len(steps), len(basis))) @ basis
    return noise * (np.linalg.norm(steps, axis=1) / np.linalg.norm(noise, axis=1))[:, None]


def draw_pairs(records, columns, steerable_rows, banks, curve_range, per_transition, seed, tokenizer, cfg):
    """({transition: drawn row indices}, {transition: {row: donor}}, {transition: eligible count},
    {transition: eligible with a donor}). Raises if a transition has fewer pairs than asked."""
    drawn, donors, eligible, with_donor = {}, {}, {}, {}
    for index, (name, start) in enumerate(PAIR_TRANSITIONS):
        rows = eligible_rows(columns, steerable_rows, start, "near", curve_range)
        eligible[name] = len(rows)
        found = {}
        for r in rows:
            target = transition_target("near", columns["log_near"][r], columns["log_far"][r])
            donor = same_unit_donor(records[r], target, tokenizer, cfg)
            if donor is not None:
                found[r] = donor
        with_donor[name] = len(found)
        if len(found) < per_transition:
            raise ValueError(f"{name}: {len(found)} eligible prompts with a same-unit donor for {per_transition} requested")
        pool = np.array(sorted(found), dtype=int)
        drawn[name] = stratified_draw(pool, banks, per_transition, np.random.default_rng([seed, index]))
        donors[name] = found
    return drawn, donors, eligible, with_donor


def pair_report_text(prompts, step_lengths, eligible, with_donor, n_pairs):
    lines = [f"Phrase number patch: {n_pairs} same-unit crossing pairs (27_spec_phrase_number_patch.md §2)",
             f"manifold step mean length {np.mean(step_lengths):.1f}; its random twins (anywhere, and inside the "
             f"number fit's top-8 subspace) have the same length", ""]
    for name, _ in PAIR_TRANSITIONS:
        rows = [p for p in prompts if p["transition"] == name]
        lines.append(f"{name}: eligible {eligible[name]}, with a same-unit donor {with_donor[name]}, drawn {len(rows)}")
        lines += [f"  {p['horizon_text']:>12s} -> {p['donor_horizon_text']:<12s} delays {p['near_delay_years']:.3g} / "
                  f"{p['far_delay_years']:.3g} years  (bank {p['bank']})" for p in rows]
    lines.append("")
    lines.append("manifold step length per transition: " + ", ".join(
        f"{name} {np.mean([l for p, l in zip(prompts, step_lengths) if p['transition'] == name]):.1f}"
        for name, _ in PAIR_TRANSITIONS))
    return "\n".join(lines) + "\n"


def patch_main(phrase_dir, out_dir, pairs=PAIRS, seed=0, layer=PATCH_LAYER):
    """Write inputs.json, inputs.npz and fit_report.txt of the patch experiment to `out_dir`."""
    if pairs % len(PAIR_TRANSITIONS):
        raise ValueError(f"{pairs} pairs cannot be balanced over {len(PAIR_TRANSITIONS)} transitions")
    phrase_dir, out_dir = Path(phrase_dir), Path(out_dir)
    states_meta = json.loads((phrase_dir / "states.json").read_text())
    states, state_uids = load_phrase_states(phrase_dir, (layer,))
    prompts_path = Path(states_meta["prompts_file"])
    if not prompts_path.exists():
        prompts_path = OUT / prompts_path.name
    prompts_meta = json.loads(prompts_path.read_text())
    records = selected_records(prompts_meta)[:len(state_uids)]
    if [r["sample_uid"] for r in records] != state_uids:
        raise ValueError("the phrase states are not those of the prompts file's first rows")
    banks = np.array(prompts_meta["selected_banks"][:len(records)])
    columns = phrase_columns(records)
    factors = varying_factors(columns)
    fits = {role: fit_phrase_cell(states[f"L{layer}|{role}"], columns, factors) for role in ("number", "unit")}
    curve_range = (fits["number"]["curve_logh"][0], fits["number"]["curve_logh"][-1])

    generation = {**prompts_meta["generation"], "model": states_meta["model"]}
    cfg = Settings(**generation)
    tokenizer = AutoTokenizer.from_pretrained(states_meta["model"])
    drawn, donors, eligible, with_donor = draw_pairs(
        records, columns, np.array([steerable(r) for r in records]), banks, curve_range,
        pairs // len(PAIR_TRANSITIONS), seed, tokenizer, cfg)
    rows = interleave(*(drawn[name] for name, _ in PAIR_TRANSITIONS))
    transition_of = [next(name for name, _ in PAIR_TRANSITIONS if r in drawn[name]) for r in rows]
    donor = [donors[t][r] for t, r in zip(transition_of, rows)]
    log_h = columns["log_h"][rows]
    donor_log_h = np.array([d[1] for d in donor])

    number_step = curve_delta(fits["number"], log_h, donor_log_h - log_h)
    basis = fits["number"]["basis"]
    arrays = {f"step|L{layer}|manifold|number": number_step,
              f"step|L{layer}|manifold|unit": curve_delta(fits["unit"], log_h, donor_log_h - log_h),
              f"step|L{layer}|random|number": random_like(number_step, np.random.default_rng([seed, layer, 0])),
              f"step|L{layer}|inspan|number": in_span_like(number_step, basis, np.random.default_rng([seed, layer, 1])),
              f"basis|L{layer}": basis}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "inputs.npz", **{k: v.astype(np.float32) for k, v in arrays.items()})

    prompts = [{"transition": t, "start_regime": columns["regime"][r], "target_regime": "near",
                "log_h": float(h), "target_log_h": float(g), "donor_horizon_text": d[0], "donor_log_h": float(g),
                "midpoint_log_h": float(transition_target("near", columns["log_near"][r], columns["log_far"][r])),
                "horizon_text": records[r]["horizon_text"], "bank": str(banks[r]),
                "near_delay_years": records[r]["near_delay_years"], "far_delay_years": records[r]["far_delay_years"]}
               for t, r, h, g, d in zip(transition_of, rows, log_h, donor_log_h, donor)]
    meta = {"model": states_meta["model"], "generation": generation, "bank_settings": prompts_meta["bank_settings"],
            "selected": [prompts_meta["selected"][r] for r in rows], "selected_banks": [str(banks[r]) for r in rows],
            "sample_uids": [records[r]["sample_uid"] for r in rows], "prompts": prompts,
            "transitions": [name for name, _ in PAIR_TRANSITIONS], "patch_layer": layer,
            "depth_layers": list(DEPTH_LAYERS), "eligible": eligible, "with_donor": with_donor, "pairs": pairs,
            "seed": seed, "phrase_dir": str(phrase_dir), "prompts_file": str(prompts_path)}
    (out_dir / "inputs.json").write_text(json.dumps(meta, indent=1))
    report = pair_report_text(prompts, np.linalg.norm(number_step, axis=1), eligible, with_donor, len(rows))
    (out_dir / "fit_report.txt").write_text(report)
    print(report)


def main(args):
    phrase_dir = Path(args.phrase_dir)
    model = json.loads((phrase_dir / "states.json").read_text())["model"].split("/")[-1]
    out_dir = Path(args.out) if args.out else OUT / PATCH_NAME.format(model)
    patch_main(phrase_dir, out_dir, args.pairs, args.seed, args.layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phrase-dir", required=True, help="exp14_<model>_phrase folder with states.json and states.npz")
    parser.add_argument("--out", default=None, help=f"output folder (default: {OUT}/exp16_<model>_patch)")
    parser.add_argument("--pairs", type=int, default=PAIRS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer", type=int, default=PATCH_LAYER)
    main(parser.parse_args())
