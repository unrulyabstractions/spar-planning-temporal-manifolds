"""Stage 2 of 2: compute the steering directions and the manifold curve from extracted activations.

    python fit.py --activations activations/gemma-4-31B-it__t0_s0 --bank ../prompt_bank/data/t0_s0 --layer 38
    python fit.py ... --regimes near_only both --name no-none       # leave the "none" regime out

For one layer and site it writes fits/<activations name>/<name>/L<layer>_<site>/:

    directions.npz   unit vectors: caa, reward (control), choice (diagnostic only), random [n, d] (controls);
                     typical_norm (the strength unit); raw norms
    manifold.npz     the curve (steer/manifold.py, Manifold.load)
    per_horizon.jsonl  one row per stated horizon: count, regimes, its mean's position along CAA and on the curve
    fit.json         settings, inputs and their hashes, class sizes, and the diagnostics printed below

Definitions (report, Appendix A.3; details in the steer modules):

* CAA: mean activation of prompts with H/D >= 4 minus that of prompts with H/D <= 1/4, unit-normalized; D is the
  far option's delay (Appendix A.1), so H/D is the bank's `h_over_far_delay`.
* Reward contrast (control): the same with the reward multiplier (far reward / near reward) in place of H/D:
  multiplier >= 25 against <= 5. The report does not define this direction further; this is our reading.
* Random controls: isotropic random unit vectors, seeded.
* Choice (diagnostic, not a control): mean of replies that chose the far option minus those that chose near. Its
  cosine with CAA measures how much of CAA is "which option was chosen" rather than "how long the horizon is".
* Manifold: per-horizon means -> PCA (k components) -> local-linear curve in log10(days) (steer/manifold.py).

Which prompts: those with a stated horizon whose regime is in --regimes (default: all three, as the report's
Methods section says "none" prompts are used for extraction). The no-horizon (null) prompts are never used: they
have no horizon to place on the curve.

Diagnostics, all in fit.json:

* class sizes by regime, for CAA and reward (the CAA classes are not balanced by regime; see README);
* cosines of CAA with the reward and choice directions, and the largest |cosine| with a random control;
* monotonicity: Spearman correlation between log-horizon and each horizon mean's projection on CAA;
* curve: variance explained by the k components; in-sample error of t-hat for the horizon means; and held-out
  error, by grouped k-fold over prompts (curve fitted without the fold, t-hat read for the fold's prompts),
  overall and by regime. Held-out here means held-out *prompts*: every horizon is still in the training folds.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from pbank.local_model import log
from pbank.provenance import git_commit, utc_now
from pbank.records import load_prompts, prompts_sha256, write_jsonl
from steer import activations
from steer.fitting import FIT_REGIMES, FitSettings, fit, select

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None, fits_root: Path = HERE / "fits") -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations", type=Path, required=True, help="directory written by extract.py")
    p.add_argument("--bank", type=Path, required=True, help="the bank the activations were extracted from")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--site", default="label", choices=activations.POSITIONS)
    p.add_argument("--regimes", nargs="+", default=list(FIT_REGIMES), choices=FIT_REGIMES)
    p.add_argument("--caa-long-min", type=float, default=4.0, help="H/D at or above which a prompt is 'long'")
    p.add_argument("--caa-short-max", type=float, default=0.25, help="H/D at or below which a prompt is 'short'")
    p.add_argument("--reward-high-min", type=float, default=25)
    p.add_argument("--reward-low-max", type=float, default=5)
    p.add_argument("--n-random", type=int, default=100, help="random control directions")
    p.add_argument("--components", type=int, default=4, help="PCA components of the curve (report: 3-8)")
    p.add_argument("--bandwidth", type=float, default=0.3, help="curve smoother bandwidth, decades")
    p.add_argument("--folds", type=int, default=5, help="folds for the held-out curve check")
    p.add_argument("--seed", type=int, default=0, help="seeds the random controls and the fold assignment")
    p.add_argument("--name", default="default", help="name of this fit's settings (output subdirectory)")
    args = p.parse_args(argv)

    acts = activations.load(args.activations, args.layer, args.site)
    records = {r.uid: r for r in load_prompts(args.bank)}
    if acts.meta["settings"]["bank_prompts_sha256"] != prompts_sha256(records.values()):
        sys.exit(f"{args.activations} was extracted from a different bank than {args.bank}")
    out = Path(fits_root) / args.activations.name / args.name / f"L{args.layer:02d}_{args.site}"
    if out.exists():
        sys.exit(f"{out} exists; choose another --name")

    # The prompts this fit uses, and the per-prompt quantities it needs.
    use = select(acts.uids, records, args.regimes)
    X = acts.X[use]
    recs = [records[acts.uids[i]] for i in use]
    chosen = np.array([acts.index[i]["chosen_option"] for i in use])
    settings = FitSettings(regimes=tuple(args.regimes), caa_long_min=args.caa_long_min,
                           caa_short_max=args.caa_short_max, reward_high_min=args.reward_high_min,
                           reward_low_max=args.reward_low_max, n_random=args.n_random, components=args.components,
                           bandwidth=args.bandwidth, folds=args.folds, seed=args.seed)
    log(f"layer {args.layer} {args.site}: {len(use)} of {len(acts.uids)} extracted prompts, "
        f"regimes {dict(Counter(r.regime for r in recs))}, d = {X.shape[1]}")
    result = fit(X, recs, chosen, settings)
    diagnostics, norm = result.diagnostics, result.typical_norm
    caa, reward, choice = result.caa, result.reward, result.choice

    out.mkdir(parents=True)
    np.savez(out / "directions.npz", caa=caa.vector, reward=reward.vector, choice=choice.vector, random=result.random,
             typical_norm=norm, caa_raw_norm=caa.raw_norm, reward_raw_norm=reward.raw_norm)
    result.manifold.save(out / "manifold.npz")
    write_jsonl(out / "per_horizon.jsonl", result.per_horizon)
    array = activations.array_name(args.layer, args.site)
    meta = {"created_utc": utc_now(), "code_commit": git_commit(),
            "inputs": {"activations": args.activations.name, "array": array,
                       "array_sha256": acts.meta["arrays"][array]["sha256"], "bank_name": args.bank.name,
                       "bank_prompts_sha256": acts.meta["settings"]["bank_prompts_sha256"],
                       "model": acts.meta["run_settings"]["model"], "thinking": acts.meta["run_settings"]["thinking"]},
            "settings": {k: v for k, v in vars(args).items() if k not in ("activations", "bank")},
            "n_prompts": result.n_prompts, "regimes": result.regime_counts, "diagnostics": diagnostics}
    (out / "fit.json").write_text(json.dumps(meta, indent=2) + "\n")

    c, m = diagnostics["caa"], diagnostics["manifold"]
    log(f"CAA: long {c['n_long']} {c['long_by_regime']}, short {c['n_short']} {c['short_by_regime']}")
    log(f"     |mean difference| = {c['raw_norm']:.3g} ({c['raw_norm_over_typical_norm']:.3f} of typical norm "
        f"{norm:.3g}); cos with reward {c['cos_reward']:+.3f}, with choice {c['cos_choice']:+.3f}, "
        f"max |cos| with {args.n_random} random {c['max_abs_cos_random']:.3f}")
    log(f"     Spearman(log H, horizon mean along CAA) = {c['spearman_logH_vs_mean_projection']:+.3f}")
    log(f"curve: k={args.components}, explained {m['explained_variance_of_means']}, in-sample median |t-hat - t| "
        f"{m['in_sample_median_abs_err_decades']:.3f} decades")
    h = m["held_out"]
    log(f"       held-out prompts: Spearman {h['spearman']:+.3f}, median |t-hat - t| "
        f"{h['median_abs_err_decades']:.3f} decades, by regime {h['median_abs_err_by_regime']}, "
        f"at range ends {h['share_at_range_ends']:.1%}")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
