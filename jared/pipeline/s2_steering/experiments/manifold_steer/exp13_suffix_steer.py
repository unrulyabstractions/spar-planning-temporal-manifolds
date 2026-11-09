"""E13 stage B: steer at the thinking-on suffix, then let the model reason
(16_spec_pristine_manifold.md §5). Qwen3-8B, the pooled banks.

At the suffix newline (L16, before the reasoning starts) each prompt's state gets a
step along one part of the effects curve (steering_prepare.py --pooled --stage-b):
its invariant part, its in-span part, U, or a random direction, at alpha = ±1 decade,
each as long as the effects curve's step. The step is added on every prefill pass
(E10's prompt patch), and the model regenerates its reasoning and label greedily.
The unsteered baseline is E0's own trace of the same prompt, not regenerated.

    python -m pipeline.s2_steering.experiments.manifold_steer.exp13_suffix_steer --inputs steering/inputs_Qwen3-8B_pristine_suffix.npz \\
        --out /workspace/results/horizon
    python -m pipeline.s2_steering.experiments.manifold_steer.exp13_suffix_steer --inputs steering/inputs_Qwen3.5-0.8B_pristine_suffix_smoke.npz \\
        --limit 2 --batch-size 2 --score-batch-size 2 --out out      # local smoke (plumbing only)

Per condition and prompt: the near log-odds before the generated label, the label,
forced close, think length; whether the reasoning writes the prompt's horizon, and
whether it writes the steered-to one (within 10%, mentions that match a delay left
out); E0's values alongside. Per condition (summary.json): the shift toward the
answer the steered-to horizon makes rational, E8's framing (a longer horizon turns
near-regime prompts toward far, a shorter one both-regime prompts toward near), in
log-odds against E0 and in label flips; the near log-odds shift per regime; the
mention rates against E0's.

Writes <out>/exp13_<inputs name>/ (e.g. exp13_Qwen3-8B_pristine_suffix):
conditions/<name>.json and .npz, summary.json. As in E10, each condition is saved
when it finishes and skipped on a rerun, and several copies (one per GPU) share the
queue by claiming conditions with conditions/<name>.lock.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pipeline.s2_steering.experiments.ablate_steer.exp8_steering import delta_rows, load_inputs, selected_records, suffix_newlines
from pipeline.s2_steering.experiments.patch.exp10_patch_reasoning import generate, generation_arrays, names_horizon, run_claimed, score
from spar_horizon import steering as S
from spar_horizon.config import Settings
from spar_horizon.durations import close_to, durations
from spar_horizon.model_io import load_model, suffix_length
from spar_horizon.runs import code_hash, out_base, stamp

DIRECTIONS = ("invariant part", "in-span part", "unit U", "random 0")
CONDITIONS = tuple((f"{direction} {alpha:+d}", direction, alpha) for direction in DIRECTIONS for alpha in (1, -1))


def names_duration(think, years, delays):
    """Whether the reasoning writes a duration within 10% of `years`, not counting
    mentions that also match a delay (the payout dates are written anyway)."""
    return any(close_to(y, years) and not any(close_to(y, d) for d in delays) for y, _ in durations(think))


def chose_near(label, near_label):
    """1 if the label is the near option, 0 if the far one, NaN if there is none."""
    if label not in ("a", "b"):
        return float("nan")
    return float(label == near_label)


def run_condition(name, direction, alpha, ctx):
    """Generate and score one condition; returns the per-prompt record and the ids."""
    model, device, tokenizer, cfg, cell = ctx["model"], ctx["device"], ctx["tokenizer"], ctx["cfg"], ctx["cell"]
    records, baseline, log_h = ctx["records"], ctx["baseline"], ctx["log_h"]
    deltas = torch.tensor(delta_rows(cell, direction, alpha, log_h), dtype=torch.float32)

    def patch(idx):
        return S.prompt_patch(model, cell["layer"], [ctx["newline"][i] for i in idx], [deltas[i:i + 1] for i in idx],
                              add=True)

    generations = generate(model, device, tokenizer, cfg, ctx["ids"], ctx["batch_size"], patch)
    near_first = [r["near_label"] == "a" for r in records]
    near_lo, labels = score(model, device, tokenizer, cfg, generations, near_first, ctx["score_batch_size"], patch)
    # the step is clamped at the curve's ends, so the steered-to horizon is too
    target = 10 ** np.clip(log_h + alpha, cell["curve_logh"][0], cell["curve_logh"][-1])
    rows = []
    for r, b, g, lo, label, years in zip(records, baseline, generations, near_lo, labels, target):
        think = g.think_text(tokenizer)
        delays = (r["near_delay_years"], r["far_delay_years"])
        rows.append({"sample_uid": r["sample_uid"], "regime": r["regime"], "horizon_text": r["horizon_text"],
                     "near_label": r["near_label"], "target_years": float(years),
                     "near_lo": float(lo), "label": label, "forced": g.forced, "think_len": g.think_len,
                     "names_own": names_horizon(think, r["horizon_text"]),
                     "names_target": names_duration(think, years, delays),
                     "e0_near_lo": b["near_lo"], "e0_label": b["label"], "e0_forced": b["forced"],
                     "e0_names_own": names_horizon(b["think_text"], r["horizon_text"]),
                     "e0_names_target": names_duration(b["think_text"], years, delays)})
    result = {"name": name, "direction": direction, "alpha": alpha, "layer": cell["layer"], "rows": rows}
    return result, generation_arrays(generations)


def summarize(results):
    """Per condition: the shift toward the rational answer at the steered-to horizon,
    against E0, in near log-odds and in label flips; the near log-odds shift per
    regime; the mention and forced-close rates beside E0's."""
    out = {}
    for name, res in results.items():
        rows, alpha = res["rows"], res["alpha"]
        regimes = np.array([r["regime"] for r in rows])
        shift = np.array([r["near_lo"] - r["e0_near_lo"] for r in rows], float)
        near = np.array([chose_near(r["label"], r["near_label"]) for r in rows])
        e0_near = np.array([chose_near(r["e0_label"], r["near_label"]) for r in rows])
        flips = S.toward_rational(near - e0_near, regimes, alpha)   # +1 toward, -1 away, 0 unchanged
        defined = ~np.isnan(flips)

        def rate(key):
            return float(np.mean([r[key] for r in rows]))

        out[name] = {"direction": res["direction"], "alpha": alpha, "n": len(rows),
                     "n_crossing": int(defined.sum()),
                     "toward_log_odds": S.bootstrap_mean(S.toward_rational(shift, regimes, alpha)),
                     "flipped_toward": float(np.mean(flips[defined] == 1)) if defined.any() else float("nan"),
                     "flipped_away": float(np.mean(flips[defined] == -1)) if defined.any() else float("nan"),
                     "near_shift_by_regime": {regime: S.bootstrap_mean(shift[regimes == regime])
                                              for regime in ("none", "near", "both") if (regimes == regime).any()},
                     **{key: rate(key) for key in ("names_own", "e0_names_own", "names_target", "e0_names_target",
                                                   "forced", "e0_forced")}}
    return out


def report(out_dir, name=None):
    """summary.json over every condition finished so far, by any worker; prints one
    condition's line, or every condition's when `name` is None."""
    results = {}
    for file in sorted((out_dir / "conditions").glob("*.json")):
        result = json.loads(file.read_text())
        results[result["name"]] = result
    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    for key in ([name] if name is not None else [c[0] for c in CONDITIONS if c[0] in summary]):
        s = summary[key]
        t = s["toward_log_odds"]
        print(f"toward rational, {key:18} log-odds {t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}] "
              f"({s['n_crossing']} near/both), labels flipped toward {s['flipped_toward']:.2f} / away "
              f"{s['flipped_away']:.2f}; names own horizon {s['names_own']:.2f} (E0 {s['e0_names_own']:.2f}), "
              f"steered-to {s['names_target']:.2f} (E0 {s['e0_names_target']:.2f}); forced {s['forced']:.2f} "
              f"(E0 {s['e0_forced']:.2f})", flush=True)


def main(args):
    meta, cells, _ = load_inputs(args.inputs)
    if "baseline" not in meta or len(cells) != 1:
        raise ValueError(f"{args.inputs}: not stage B inputs (steering_prepare.py --pooled --stage-b)")
    (cell,) = cells.values()
    records, baseline = selected_records(meta), meta["baseline"]
    if args.limit:
        records, baseline = records[:args.limit], baseline[:args.limit]
    # E0's generation settings, so that its trace is the unsteered baseline
    generation = {**meta["generation"], "max_think_tokens": args.max_think_tokens or meta["generation"]["max_think_tokens"]}
    cfg = Settings(**generation, batch_size=args.batch_size)
    out_dir = out_base(args.out) / Path(args.inputs).stem.replace("inputs_", "exp13_", 1)
    (out_dir / "conditions").mkdir(parents=True, exist_ok=True)
    tokenizer, model, device = load_model(cfg)
    ids = [S.encode(tokenizer, r["prompt"], cfg) for r in records]
    ctx = {"model": model, "device": device, "tokenizer": tokenizer, "cfg": cfg, "cell": cell, "records": records,
           "baseline": baseline, "ids": ids, "log_h": np.log10([r["horizon"] for r in records]),
           "newline": suffix_newlines(ids, tokenizer, suffix_length(tokenizer, records[0]["prompt"], cfg)),
           "batch_size": args.batch_size, "score_batch_size": args.score_batch_size}
    stamp(f"{len(records)} prompts, cell {cell['mode']} {cell['label']} L{cell['layer']}, "
          f"think budget {cfg.max_think_tokens}")
    (out_dir / "manifest.json").write_text(json.dumps({
        "inputs": str(args.inputs), "settings": cfg.to_dict(), "prompts": len(records), "code_hash": code_hash()},
        indent=1))
    todo = [c for c in CONDITIONS if not args.only or c[0] in args.only.split(";")]
    for name, direction, alpha in todo:
        if run_claimed(out_dir, name, lambda: run_condition(name, direction, alpha, ctx)):
            report(out_dir, name)
    report(out_dir)
    stamp(f"wrote {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--inputs", required=True,
                        help="steering/inputs_<model>_<name>.npz from steering_prepare.py --pooled --stage-b")
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts (smoke)")
    parser.add_argument("--max-think-tokens", type=int, default=None,
                        help="think budget (default: E0's, which the baseline needs)")
    parser.add_argument("--batch-size", type=int, default=12, help="generation batch (E0's on the 4090)")
    parser.add_argument("--score-batch-size", type=int, default=8, help="batch for the scoring passes")
    parser.add_argument("--only", default=None, help='";"-separated condition names to run (default: all)')
    parser.add_argument("--out", default=None, help="output base (default: out/ or $WORKSPACE/results/horizon)")
    main(parser.parse_args())
