# Steering Qwen3-8B, thinking on, in the evaluation suite (spec v2, 2026-11-26, not approved)

Branch `qwen8b-steering`. Budget $5-10 (Jared, 2026-11-25): plan for $3, stop and `just down` at $5 spent; $10 is the
hard cap. Replay in float32 (approved 2026-11-26).

## 1. Diagnosis of the first Qwen3-8B run (CPU, saved artifacts of `results/qwen8b_quick_20261125/`)

That run found CAA at L24 at the covariance-null level (+0.22 logits, p = 0.45). Checks on the 456 fit prompts, per
candidate direction and site (split-half = median cosine of the direction fitted on two random halves, 30 splits;
letter = first-listed minus second-listed answer label; choice = chose far minus chose near):

| direction | far rate, + class / - class | split-half (label / prelabel) | cos with choice (label / prelabel) | cos with letter (label / prelabel) |
|---|---|---|---|---|
| `caa` (H/far delay >= 4 vs <= 0.25) | 0.47 / 0.31 | 0.35 / 0.62 | 0.45 / 0.45 | 0.33 / 0.04 |
| `caa-within` (longer vs shorter inside a regime) | near_only 0.02 / 0.00; both 0.42 / **0.58** | 0.20 / 0.16 | -0.01 / 0.19 | **0.67** / 0.04 |
| `choice` | 1 / 0 by construction | 0.95 / 0.79 | 1 | 0.16 / -0.12 |
| regime: `both` minus `near_only` (`none` left out) | **0.45 / 0.01** | 0.76 / 0.84 | 0.75 / 0.55 | 0.23 / 0.13 |

Same regime direction at the earlier sites: `think_end` split-half 0.86, cos with choice 0.85; `prompt_end` split-half
0.79, cos with choice **0.08** (before reasoning the regime direction is not yet the decision).

Findings:
1. **The horizon only changes behaviour where it crosses the far delay** (the regime boundary). CAA's classes are set by
   H/far delay thresholds, and 275 of its 346 short prompts are `none` (neither option arrives), which pick far 39% of
   the time; so its classes barely differ in behaviour, and the direction is unreliable (split-half 0.35 at `label`).
   Inside a regime, longer horizons do not change behaviour at all (or reverse it in `both`): `caa-within` is noise
   (split-half <= 0.2) and, at `label`, mostly the answer letter.
2. **Calibration picks a strength past the peak.** On the held-out fold the CAA effect is +0.65 logits at x = 0.2 and
   -1.16 at 0.8. The rule (smallest healthy x with |mean odd| >= 1 logit) judges size only, so the reversed effect at
   0.8 counts as reaching the target and 0.8 is frozen (`calibration.json`: reached_target true). This explains the
   sign change between the held-out fold and the grid. Fix: `--calib-rule peak` (largest effect before the first sign
   change along the sweep: x = 0.2 here).
3. **The regime contrast is reliable and moves behaviour, but at the answer it is largely the decision** (cos 0.55-0.85
   with choice). Only at `prompt_end` is it reliable and separate from the decision.

## 2. Question

Does a direction with a behavioural handle (regime) move the reasoned choice on the T0 verification grid more than
covariance-matched random directions, at the answer (prelabel) and before the reasoning (prompt_end)? Is the readout
able to show an effect at all (`choice` as positive control)?

## 3. Run (replay only: baseline traces and extractions reused, no new generation)

- **Directions:** `choice` (positive control), regime (`both` minus `near_only`, fitted on the same 456 prompts),
  `caa` (reference). `caa-within` is dropped (finding 1).
- **Sites:** `prelabel` (all three) and `prompt_end` (regime only). `label` is not rerun.
- **Packaging:** regime is not a method in `steering/` (Alan's code, out of scope). It is written as a standard
  `LinearEdit` bundle (`method.json` + `arrays.npz` with `direction`, `typical_norm`) by a script in this branch, so
  `evaluation/` reads it unchanged. `choice` and `caa` are refitted at `prelabel` with `steering/fit.py --site`.
- **Strength:** same held-out fold and x grid (0.02-0.8); new flag `--calib-rule peak` (largest |odd effect| among
  healthy x before the first sign change; default stays `target`), with its dose curve saved. Nulls matched at the chosen x.
- **Readout:** float32 replay, K = 19 covariance draws (shared draws for `prelabel` and `prompt_end` on the box).

## 4. Readouts, fixed before the run

Per direction and site: meaning (order-balanced logits), rank p vs covariance nulls, switch-point shift, label mass,
dose curve. Read in order: (1) `choice` at `prelabel` beats the nulls, else stop (the readout is broken); (2) regime at
`prelabel` beats the nulls; (3) regime at `prompt_end` beats the nulls (a horizon edit, not a decision edit).
Prediction (written now): (1) yes; (2) yes, smaller than `choice`; (3) weak under replay, because the reasoning is held
fixed and the decision is made in it.

## 5. Cost and stop rule

This morning's calibrate + replay (2 directions + 19 nulls, bf16, A100) took about 34 min. Here: about 4 directions-sites
+ 2 x 19 nulls, float32 (about 2x): about 2.5 h with boot, on a 40-48 GB card at about $0.5-0.9/h: $1.5-2.5. Run
`choice` and regime at `prelabel` first; stop at $5 spent with whatever is complete.

## 6. Code changes

- `jared/.../regime_bundle.py`: writes the regime bundle from the saved arrays and fit rows (CPU), with a test.
- `evaluation/replay_eval.py`: `--calib-rule {target,peak}`, default `target`, with a test on hand-made calibration rows.
- The diagnostics of §1 as a script (see §7), so they are rerun on every fit.

## 7. Follow-up (separate spec): pre-flight diagnostics for `evaluation/`

A CPU stage after fit and before any GPU replay, writing `diagnostics.json` and a report-card section, with gates:
class behaviour gap, split-half reliability, alignment with nuisance directions (letter, label style, option order,
forced close, trace length), alignment with the choice direction, dose-curve shape and the rule's chosen x, fold sign
agreement; plus a planted-direction test of the full statistics path, and a minimal-pair bank (same item, horizon just
below and above the far delay) for fitting.
