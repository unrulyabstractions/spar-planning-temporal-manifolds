"""Stage 2 of 2: compute the steering directions and the manifold curve from extracted activations.

    python fit.py --activations activations/gemma-4-31B-it__t0_s0 --bank ../prompt_bank/data/t0_s0 --layer 38
    python fit.py ... --no-include-none --name no-none               # leave the "none" regime out
    python fit.py ... --uids ../prompt_bank/data/t0_s0/splits/reserved_horizons.jsonl --uids-role fit --name reserved

For one layer and site it writes fits/<activations name>/<name>/L<layer>_<site>/:

    directions.npz   unit vectors: caa, reward (control), choice (diagnostic only), trend and probe (horizon
                     directions with the regime removed), random [n, d] (controls);
                     the within-regime CAAs caa_none, caa_near_only, caa_both and their pool caa_within (those that
                     could be formed); typical_norm (the strength unit); raw norms
    manifold.npz     the curve (steer/manifold.py, Manifold.load)
    per_horizon.jsonl  one row per stated horizon: count, regimes, its mean's position along CAA and on the curve
    fit_uids.jsonl   one row per fitted prompt: its activation row, uid, regime, horizon, H/D, reward multiplier,
                     option order, label style, far option's label, baseline choice, CAA and reward class, fold
    fit.json         settings, inputs and their hashes, class sizes, and the diagnostics printed below

and appends one line to fits/<activations name>/search.jsonl on every run, whatever the outcome (written, refused,
error): when, which code, which settings, and the headline diagnostics. It is the record of what was tried before a
fit was chosen, so it is append-only and written by this script, not by hand.

Definitions (report, Appendix A.3; details in the steer modules):

* CAA: mean activation of prompts with H/D >= 4 minus that of prompts with H/D <= 1/4, unit-normalized; D is the
  far option's delay (Appendix A.1), so H/D is the bank's `h_over_far_delay`.
* Within-regime CAA (comparison methods): CAA's classes differ in regime, and so in which answer is correct (its
  long class is all "both"). Inside one regime the correct answer is the same for every prompt, so there the
  contrast is: mean of the regime's prompts with the longest horizons minus mean of those with the shortest, each
  class about --within-quantile (default one third) of the regime, split on log-horizon. One direction per regime
  (caa_none, caa_near_only, caa_both), and caa_within: the average of the raw differences of the --within-pool
  regimes (default near_only and both: in "none" the horizon cannot change the answer), weighted by regime size,
  then unit-normalized. H/D cannot define these classes: inside "none" and "near_only" it is always below 1.
* Reward contrast (control): the same with the reward multiplier (far reward / near reward) in place of H/D:
  multiplier >= 25 against <= 5. The report does not define this direction further; this is our reading.
* Random controls: isotropic random unit vectors, seeded. With --draws, the shared draws written by
  evaluation/shared_draws.py for these activations are used instead (isotropic, covariance-matched and real
  differences), so the fit's random baseline is the same set of vectors as the evaluation's nulls. The draws are
  made from every fitted-split prompt of the extraction, including any this fit leaves out (--exclude-horizons,
  --no-include-none); they enter the diagnostics only, never a method's parameters.
* Choice (diagnostic, not a control): mean of replies that chose the far option minus those that chose near. Its
  cosine with CAA measures how much of CAA is "which option was chosen" rather than "how long the horizon is".
* Trend and probe (horizon directions with the regime removed): within each regime, subtract the regime's mean
  activation and mean log10 H. trend is the sum over prompts of (activation) x (log10 H), unit-normalized: the
  covariance pattern of a horizon decoder (Haufe et al. 2014), the right direction for an additive edit. probe is the
  ridge decoder's weight vector for the same target, alpha from logspace(0, 6, 13) by 5-fold CV that holds out whole
  horizons (sorted distinct horizons dealt to folds in turn), unit-normalized. Both point toward longer horizons.
  Diagnostics: held-out-horizon Spearman of each one's projection with the (demeaned) log10 H, probe's alpha, cosines
  with caa, choice and reward, and the prompt counts (steer/horizon_decoders.py).
* Manifold: per-horizon means -> PCA (k components) -> local-linear curve in log10(days) (steer/manifold.py).

Which prompts: extraction-split prompts with a stated horizon. "None"-regime prompts are included by default, as the
report's Methods section says they are "used for extraction"; --no-include-none leaves them out of everything (CAA,
the controls and the curve). --exclude-horizons leaves out every prompt at the given stated horizons, e.g. to hold
them out for calibration. --uids FILE keeps only prompts whose uid is a row of the jsonl FILE (with --uids-role ROLE,
only rows whose "role" is ROLE, e.g. reserved_horizons.jsonl's fit/test); it combines with the rules above. fit.json
records the file, its sha256, the role and, in `excluded`, the prompts it leaves out ("not_in_uids", counted only for
prompts the other rules keep). Verification-grid prompts are never used, even if extract.py was run on them, and neither
are the no-horizon (null) prompts, which have no horizon to place on the curve. fit.json counts the prompts left out
for each reason.

Diagnostics, all in fit.json:

* trend and probe: see their definition above (diagnostics.horizon_decoders);
* class sizes by regime, for CAA and reward (the CAA classes are not balanced by regime; see README);
* the within-regime CAAs: class sizes and horizon cut-offs per regime, each class's share of far choices, cosines
  with CAA, choice and reward, and the cosines between the regimes' directions (do they agree on one direction?);
* cosines of CAA with the reward and choice directions, and the largest and 95th-percentile |cosine| with the
  random controls, per family;
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
from pbank.provenance import utc_now
from pbank.records import load_prompts, prompts_sha256, read_jsonl, write_jsonl
from steer import activations
from steer.activations import file_sha256
from steer.draws import load_shared
from steer.provenance import code_commit
from pbank.durations import Duration
from steer.fitting import FIT_REGIMES, FitSettings, direction_keys, fit, select

HERE = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations", type=Path, required=True, help="directory written by extract.py")
    p.add_argument("--bank", type=Path, required=True, help="the bank the activations were extracted from")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--site", default="label", choices=activations.POSITIONS)
    p.add_argument("--include-none", action=argparse.BooleanOptionalAction, default=True,
                   help='use "none"-regime prompts (default: yes, as the report says)')
    p.add_argument("--exclude-horizons", nargs="+", default=[], metavar="HORIZON",
                   help='stated horizons (grid values, e.g. "3 years") whose prompts are left out of the fit')
    p.add_argument("--uids", type=Path, metavar="FILE",
                   help="jsonl with a uid per row (and a role): fit only on the prompts it lists")
    p.add_argument("--uids-role", help='with --uids: only rows whose "role" is this (e.g. fit)')
    p.add_argument("--caa-long-min", type=float, default=4.0, help="H/D at or above which a prompt is 'long'")
    p.add_argument("--caa-short-max", type=float, default=0.25, help="H/D at or below which a prompt is 'short'")
    p.add_argument("--within-quantile", type=float, default=1 / 3,
                   help="within-regime CAA: share of a regime's prompts in each of its long and short classes")
    p.add_argument("--within-pool", nargs="+", default=["near_only", "both"], choices=FIT_REGIMES,
                   help="regimes whose within-regime differences are pooled into caa_within")
    p.add_argument("--reward-high-min", type=float, default=25)
    p.add_argument("--reward-low-max", type=float, default=5)
    p.add_argument("--n-random", type=int, default=100, help="random control directions (not used with --draws)")
    p.add_argument("--draws", type=Path, metavar="DIR",
                   help="shared random draws for this model, layer and site (evaluation/shared_draws.py output, "
                        "draws/<model>/L<layer>_<site>); replaces the fit's own random directions")
    p.add_argument("--components", type=int, default=4, help="PCA components of the curve (report: 3-8)")
    p.add_argument("--bandwidth", type=float, default=0.3, help="curve smoother bandwidth, decades")
    p.add_argument("--folds", type=int, default=5, help="folds for the held-out curve check")
    p.add_argument("--seed", type=int, default=0, help="seeds the random controls and the fold assignment")
    p.add_argument("--name", default="default", help="name of this fit's settings (output subdirectory)")
    return p.parse_args(argv)


def recorded_settings(args: argparse.Namespace) -> dict:
    return {k: v for k, v in vars(args).items() if k not in ("activations", "bank", "draws", "uids")}


def read_uids(path: Path, role: str | None) -> set[str]:
    rows = read_jsonl(path)
    if role is not None:
        rows = [r for r in rows if r.get("role") == role]
    if not rows:
        sys.exit(f"--uids {path}: no rows" + (f" with role {role!r}" if role else ""))
    return {r["uid"] for r in rows}


def append_search(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None, fits_root: Path = HERE / "fits") -> None:
    args = parse_args(argv)
    search = Path(fits_root) / args.activations.name / "search.jsonl"
    entry = {"time_utc": utc_now(), "code_commit": code_commit(), "activations": args.activations.name,
             "bank": args.bank.name, "settings": recorded_settings(args)}
    try:
        summary = run(args, Path(fits_root))
    except SystemExit as e:
        append_search(search, {**entry, "outcome": "refused", "reason": str(e.code)})
        raise
    except Exception as e:
        append_search(search, {**entry, "outcome": "error", "reason": f"{type(e).__name__}: {e}"})
        raise
    append_search(search, {**entry, "outcome": "written", **summary})


def run(args: argparse.Namespace, fits_root: Path) -> dict:
    """The fit for parsed `args`, written under `fits_root`; returns the search-log summary."""
    acts = activations.load(args.activations, args.layer, args.site)
    records = {r.uid: r for r in load_prompts(args.bank)}
    if acts.meta["settings"]["bank_prompts_sha256"] != prompts_sha256(records.values()):
        sys.exit(f"{args.activations} was extracted from a different bank than {args.bank}")
    out = fits_root / args.activations.name / args.name / f"L{args.layer:02d}_{args.site}"
    if out.exists():
        sys.exit(f"{out} exists; choose another --name")

    # The prompts this fit uses, and the per-prompt quantities it needs.
    grid = {r.horizon_grid_value for r in records.values() if r.horizon_grid_value}
    for h in args.exclude_horizons:
        if h not in grid:
            sys.exit(f"--exclude-horizons {h!r} is not a stated horizon of {args.bank.name}; grid values look like "
                     f"{sorted(grid, key=lambda g: Duration.parse(g).days)[:3]}")
    if args.uids_role and not args.uids:
        sys.exit("--uids-role needs --uids")
    allowed = read_uids(args.uids, args.uids_role) if args.uids else None
    if allowed is not None and not allowed <= records.keys():
        sys.exit(f"--uids {args.uids}: {len(allowed - records.keys())} uids are not in {args.bank.name}")
    selection = select(acts.uids, records, args.include_none, args.exclude_horizons, allowed)
    use = selection.rows
    X = acts.X[use]
    recs = [records[acts.uids[i]] for i in use]
    chosen = np.array([acts.index[i]["chosen_option"] for i in use])
    settings = FitSettings(caa_long_min=args.caa_long_min,
                           caa_short_max=args.caa_short_max, reward_high_min=args.reward_high_min,
                           reward_low_max=args.reward_low_max, n_random=args.n_random, components=args.components,
                           bandwidth=args.bandwidth, folds=args.folds, seed=args.seed,
                           within_quantile=args.within_quantile, within_pool=tuple(args.within_pool))
    log(f"layer {args.layer} {args.site}: {len(use)} of {len(acts.uids)} extracted prompts, "
        f"regimes {dict(Counter(r.regime for r in recs))}, d = {X.shape[1]}; left out {selection.excluded}")
    array = activations.array_name(args.layer, args.site)
    if args.draws:
        try:
            shared = load_shared(args.draws, acts.meta["model"]["model_id"], args.layer, args.site, X.shape[1],
                                 acts.meta["arrays"][array]["sha256"])
        except (FileNotFoundError, ValueError) as e:
            sys.exit(f"--draws: {e}")
        random_sets, random_source = shared.families, shared.source
    else:
        random_sets, random_source = None, {"kind": "own_isotropic", "n": args.n_random, "seed": args.seed}
    result = fit(X, recs, chosen, settings, random_sets)
    diagnostics, norm = result.diagnostics, result.typical_norm
    caa, reward, choice = result.caa, result.reward, result.choice

    out.mkdir(parents=True)
    within = {key: result.direction(key) for key in direction_keys(result) if key.startswith("caa_")}
    np.savez(out / "directions.npz", caa=caa.vector, reward=reward.vector, choice=choice.vector,
             trend=result.trend.vector, probe=result.probe.vector, random=result.random,
             typical_norm=norm, caa_raw_norm=caa.raw_norm, reward_raw_norm=reward.raw_norm,
             **{key: d.vector for key, d in within.items()},
             trend_raw_norm=result.trend.raw_norm, probe_raw_norm=result.probe.raw_norm,
             **{f"{key}_raw_norm": d.raw_norm for key, d in within.items()})
    result.manifold.save(out / "manifold.npz")
    write_jsonl(out / "per_horizon.jsonl", result.per_horizon)
    write_jsonl(out / "fit_uids.jsonl", (
        {"row": i, "uid": r.uid, "regime": r.regime, "horizon": r.horizon, "horizon_grid_value": r.horizon_grid_value,
         "h_over_far_delay": r.h_over_far_delay, "reward_multiplier": r.reward_multiplier, "near_first": r.near_first,
         "label_style": "/".join(r.labels), "far_label": r.far_label, "chosen_option": acts.index[i]["chosen_option"],
         "choice_label": acts.index[i]["choice_label"], "caa_class": str(cc) or None, "reward_class": str(rc) or None,
         "caa_within_class": str(wc) or None, "fold": int(f)}
        for i, r, cc, rc, wc, f in zip(use, recs, result.caa_class, result.reward_class, result.within_class,
                                       result.fold)))
    meta = {"created_utc": utc_now(), "code_commit": code_commit(),
            "inputs": {"activations": args.activations.name, "array": array,
                       "array_sha256": acts.meta["arrays"][array]["sha256"], "bank_name": args.bank.name,
                       "bank_prompts_sha256": acts.meta["settings"]["bank_prompts_sha256"],
                       "model": acts.meta["run_settings"]["model"], "thinking": acts.meta["run_settings"]["thinking"]},
            "settings": recorded_settings(args),
            "uids_file": None if allowed is None else {"path": str(args.uids), "sha256": file_sha256(args.uids),
                                                       "role": args.uids_role, "n_uids": len(allowed)},
            "random_source": random_source,
            "n_prompts": result.n_prompts, "regimes": result.regime_counts, "excluded": selection.excluded,
            "diagnostics": diagnostics}
    (out / "fit.json").write_text(json.dumps(meta, indent=2) + "\n")

    c, m = diagnostics["caa"], diagnostics["manifold"]
    log(f"CAA: long {c['n_long']} {c['long_by_regime']}, short {c['n_short']} {c['short_by_regime']}")
    log(f"     |mean difference| = {c['raw_norm']:.3g} ({c['raw_norm_over_typical_norm']:.3f} of typical norm "
        f"{norm:.3g}); cos with reward {c['cos_reward']:+.3f}, with choice {c['cos_choice']:+.3f}, "
        f"max |cos| with random: " + ", ".join(f"{f} ({v['n']}) {v['max']:.3f}" for f, v in c["abs_cos_random"].items()))
    log(f"     Spearman(log H, horizon mean along CAA) = {c['spearman_logH_vs_mean_projection']:+.3f}")
    w = diagnostics["caa_within"]
    for g, v in w["by_regime"].items():
        log(f"within {g}: long {v['n_long']} (t >= {v['long_min_log10_days']:.2f}), short {v['n_short']} "
            f"(t <= {v['short_max_log10_days']:.2f}); chose far {v['chose_far_share_long']:.2f} / "
            f"{v['chose_far_share_short']:.2f}; cos with CAA {v['cos_caa']:+.3f}, with choice {v['cos_choice']:+.3f}")
    if w["pooled"]:
        log(f"within, pooled over {w['pool']}: cos with CAA {w['pooled']['cos_caa']:+.3f}, with choice "
            f"{w['pooled']['cos_choice']:+.3f}; agreement between regimes {w['agreement']}")
    if w["not_formed"]:
        log(f"within: no classes could be formed in {w['not_formed']}")
    hd = diagnostics["horizon_decoders"]
    log(f"trend: held-out-horizon Spearman {hd['trend']['held_out_horizon_spearman']:+.3f}, cos with CAA "
        f"{hd['trend']['cos_caa']:+.3f}, choice {hd['trend']['cos_choice']:+.3f}, reward {hd['trend']['cos_reward']:+.3f}")
    log(f"probe: alpha {hd['probe']['alpha']:g}, held-out-horizon Spearman {hd['probe']['held_out_horizon_spearman']:+.3f}, "
        f"cos with CAA {hd['probe']['cos_caa']:+.3f}, with trend {hd['cos_trend_probe']:+.3f}")
    log(f"curve: k={args.components}, explained {m['explained_variance_of_means']}, in-sample median |t-hat - t| "
        f"{m['in_sample_median_abs_err_decades']:.3f} decades")
    h = m["held_out"]
    log(f"       held-out prompts: Spearman {h['spearman']:+.3f}, median |t-hat - t| "
        f"{h['median_abs_err_decades']:.3f} decades, by regime {h['median_abs_err_by_regime']}, "
        f"at range ends {h['share_at_range_ends']:.1%}")
    log(f"wrote {out}")
    return {"out": str(out.relative_to(fits_root)), "n_prompts": result.n_prompts, "excluded": selection.excluded,
            "headline": {"caa_cos_choice": c["cos_choice"], "caa_cos_reward": c["cos_reward"],
                         "caa_spearman_logH": c["spearman_logH_vs_mean_projection"],
                         "curve_held_out_spearman": h["spearman"],
                         "curve_held_out_median_abs_err_decades": h["median_abs_err_decades"]}}


if __name__ == "__main__":
    main()
