# Results: second overnight, after the fill run (2026-10-30)

Post hoc, on saved activations only (8B, full + units + fill, both modes); no model was
run. Builds on `19_results_fill.md`. Equivalent decades and CIs as in `18` (95%, bases
resampled; * = excludes 0). Jared's decisions from the evening of 10-29: pool every
cell, steer with directions from the pooled fit, steer only prompts whose delays lie
in Alan's range (1 week – 50 years).

## 1. Summary

- **Stage A and B are ready to run** (§6). The inputs are prepared from the pooled fit
  and the code is tested; the model paths still need Jared's 0.8B smokes.
- **Before reasoning, the horizon code moves smoothly with delay magnitude; there is no
  step where a delay crosses the horizon** (§3). At suffix L16 no step survives two
  specifications or two readouts, and pooled CIs bound one at about ±0.25 decades.
  The fill's "regime both" (−0.6 decades with steps alone) goes to −0.03…−0.37, all CIs
  including 0, once smooth delay terms are in. So the regime question from `19` §4
  resolves as: the model encodes how long the delays are, not whether they fit.
  After reasoning (pre-label) the far delay's crossing is a 3–5 decade step along the
  horizon axis: that is the decision, not the horizon.
- **The pooled effects ⟂ curve is the clean axis** (§4). A one-decade horizon step
  along it carries 0.05–0.10 decades of delay, against 0.16–0.40 for the plain effects
  curves (old or pooled). In range its readout is calibrated (slope 0.98–1.00) and
  delay-free (≤ 0.10). Out of range (fill delays of seconds to hours) every curve is
  pulled by the delays.
- **The fill's +2-decade misread is absolute delay scale, not regime** (§3.3): far
  delays 3–5 decades below a week predict +1.3 to +3 decades.
- **Correlationally, the horizon code barely predicts the choice beyond the true
  horizon** (§5), and no steering direction predicts it better than random directions.
  At the stage B cell (on suffix L16) nothing does. Steering at α = ±1 is 3–4 SD
  outside the natural spread along these directions, so a causal effect would be new
  information, not a contradiction.
- **The stage B directions are stable** (fold-to-fold cosine 0.80–0.98); **the
  thinking-on pre-label directions for stage A are not** (0.42–0.56), so stage A on
  those cells will be hard to read.
- **Pooling the trace cells** (thinking-on think close and pre-label) makes their
  per-prompt readouts worse than on the old runs alone (§2), as the anchor check
  predicted. Rendering averages still pass P2 at pre-label.

## 2. Pooling every cell (Jared's decision)

`gated_jobs` (`analysis/pristine_manifold.py`) now pools all of a mode's cells when its
suffix anchors match (median gap ≤ 0.1 at every suffix layer). Thinking-on traces do
not reproduce under a different batch, so the trace cells cannot test the run setup
(`19` §2). Its test was rewritten for the new rule (a requirement change, not a
weakened assertion); the suffix-only outputs are kept in
`results/horizon/analysis/fill_gated_suffix_only/`.

Effects ⟂, k = 4, old runs only (`17` §8) → pooled with the fill:

| cell | P2 per prompt | P2 R² (rendering averages) | P4 |
|---|---|---|---|
| on think close L22 | 682% → 1258% | 0.02 → 0.13 | 0.55 → 0.55 |
| on think close L24 | 822% → 1165% | 0.11 → 0.17 | 0.54 → 0.55 |
| on pre-label L22 | 231% → 459% | 0.75 → 0.65 | 0.54 → 0.59 |
| on pre-label L24 | 283% → 432% | 0.62 → 0.62 | 0.60 → 0.71 |

The suffix cells are unchanged from `19` (on L16: 96% per prompt, R² 0.91, P4 0.60).
With the trace cells pooled, the confound table at on pre-label L22 (effects ⟂) keeps
regime both +0.55* and forced +0.21*; delays, order and unit are ≤ 0.10.

## 3. Step or slope at the delay crossing

`analysis/delay_step.py`; figures `delay_step_fill.png`, `delay_step_pooled.png`.

Model, fixed before fitting: within bank × duration, readout = duration mean +
Σ over the near and far delay of [slope·m + hinge·max(m, 0) + step·1(m < 0)] + order +
log unit size, with m = log10(delay / horizon). Robustness: a restricted cubic spline
in absolute log delay. Three cross-fitted readouts: a fill-only probe, pooled effects ⟂
(which projects regime out, so it leans toward no step), and the horizon axis (nothing
projected out). Equivalent decades use the calibration on old prompts, since on the
fill the delays move with the horizon.

### 3.1 Suffix (before reasoning)

Step, hinge spec, near / far:

| cell | fill: probe | fill: ⟂ | fill: axis | pooled: probe | pooled: ⟂ | pooled: axis |
|---|---|---|---|---|---|---|
| on L16 | +0.21 / −0.18 | +0.24 / −0.21 | −0.31 / −0.06 | +0.16 / −0.02 | +0.15* / −0.09 | −0.05 / +0.14* |
| on L24 | −0.16 / +0.16 | −0.09 / −0.03 | −0.54* / +0.02 | +0.50* / +0.07 | +0.17 / +0.18* | +0.14 / +0.05 |
| off L16 | +0.19 / −0.03 | +0.14 / −0.37 | −0.28 / −0.05 | +0.12 / +0.03 | +0.05 / −0.06 | −0.05 / +0.14* |
| off L24 | −0.23 / +0.20 | −0.27 / +0.21 | −0.50* / +0.05 | +0.51* / +0.09 | +0.21* / +0.31* | +0.12 / +0.04 |

- **L16:** no step survives both specifications or two readouts; with the spline no
  step is significant (|step| ≤ 0.40 on the fill, ≤ 0.10 pooled). The smooth part: the
  probe and ⟂ readouts rise 0.1–0.3 equivalent decades per decade of margin (a longer
  delay reads a longer horizon); the axis is flat below the crossing.
- **L24:** steps appear but disagree in sign between readouts (axis −0.5 on the fill,
  +0.1…+0.5 pooled). No consistent step.
- **Pre-label (after reasoning):** the horizon axis drops 3.4–5.4 equivalent decades
  when the far delay is inside the horizon (figure, bottom rows). That is the decision
  written into the answer slot.
- Caveat: the fill keeps every delay at least 0.2 decades from the horizon, so a step
  on fill prompts is fitted across that gap.

### 3.2 What this does to `19` §4

"Regime both" on the fill: with steps alone the far step is −0.56 to −0.64*; adding the
smooth delay terms takes it to −0.03 to −0.37, every CI including 0. The regime
residual was a delay slope.

### 3.3 Why the old curves misread the fill

Old-runs effects ⟂, cross-fitted, mean error on fill prompts at on L16: near regime
+1.98 (delays of minutes–hours), +1.13 (days–weeks), −0.54 (months–years); both +1.21,
+0.48, −0.06; none (months–years only) +0.11. Regressed over all prompts (on L16 / L24):
per decade of far delay −0.32* / −0.58*, of near delay −0.08* / −0.18*; far inside the
horizon −0.47* / −0.55*; near inside +0.09 / −0.19 (n.s.). So it is the absolute delay
scale; the regime term has the opposite sign. This rests on old-vs-fill contrasts at
matched horizons, so any other bank difference would be absorbed into it.

## 4. Which axis carries horizon and not delay

`analysis/axis_delay_purity.py`, `axis_delay_purity.png`. Candidates: (a) old-runs
effects curve, (b) pooled effects curve, (c) pooled effects ⟂, (d) fill-only effects ⟂.
"Step" = decades of near / far delay contained in a one-decade horizon step along the
axis (held-out, within duration) — what steering along the axis would add.

| cell | (a) old effects | (b) pooled effects | (c) pooled ⟂ | (d) fill ⟂ |
|---|---|---|---|---|
| on L16 | −0.24 / +0.22 | −0.34 / +0.31 | **+0.07 / +0.05** | +0.11 / +0.04 |
| on L24 | +0.40 / −0.30 | +0.26 / −0.17 | **+0.05 / +0.05** | +0.12 / −0.01 |
| off L16 | −0.24 / +0.23 | −0.36 / +0.33 | **+0.10 / +0.01** | +0.12 / +0.04 |
| off L24 | +0.40 / −0.30 | +0.23 / −0.16 | **+0.07 / +0.02** | +0.11 / −0.00 |

- Readout view, on L16 in range (curve readout per decade of horizon / near / far):
  (c) 1.00 / +0.02 / +0.02; (a) 0.55 and (b) 0.70 are compressed (nearest point on a
  non-orthogonal curve). Out of range (c) reads 0.54 / +0.15 / +0.23.
- (a) and (b) carry the unit (≈ −0.25 per decade of unit size) and at L24 option
  order (+0.3). (c) carries regime both (+0.28 at L16), which §3 now reads as delay
  slope that the linear covariates do not fully capture.
- Cosines between main directions: (a)·(b) 0.89–0.92, (c)·(d) 0.73–0.79, plain vs ⟂
  0.24–0.39. The ⟂ direction is a third of the full horizon direction's length.
- A "pull" measure (how far a decade of delay moves activations along the axis, in
  horizon decades) ranks the axes the other way, because it divides by the axis's
  length; for steering the step view is the relevant one.

**For the steering conditions:** stage A/B's "invariant part" is the effects step with
its component in the factor + delay-covariate span removed, so, in sample, it contains
no linear delay, unit, regime or order effect by construction; it is the steering
counterpart of (c). The "effects curve" condition is (b), whose step carries 0.31–0.34
decades of delay at L16. The invariant part's held-out delay content was not measured
separately.

## 5. Does the code predict the choice?

`analysis/choice_from_code.py`, `choice_from_code.png`. Linear probability of choosing
near on the cross-fitted effects ⟂ residual (readout − true log horizon), fixed effects
duration × regime × bank, controls for delays, rewards, forced, order (× bank, × regime).
Thinking off on all banks (n = 1,386); thinking on on the old banks (n = 986; the fill is
at ceiling). Expected signs fixed beforehand: near −, both −, none ambiguous.

| cell | all regimes | near | both |
|---|---|---|---|
| off suffix L16 | +0.00 [−0.05, +0.05] | −0.08 [−0.15, +0.01] ✓ | +0.02 ✗ |
| off pre-label L22 | −0.01 | −0.06* ✓ | −0.04 ✓ |
| on suffix L16 (stage B) | −0.02 [−0.08, +0.04] | +0.03* ✗ (P(near) 0.98) | +0.05 [−0.24, +0.29] ✗ |
| on think close L24 | −0.02* | +0.00 | −0.04* ✓ |
| on pre-label L24 | +0.02 | +0.00 | −0.03 ✓ |

Per decade of residual; the residual SD at the suffix is 0.3–0.4 decades, so one SD of
misreading moves P(near) by ≤ 0.03. The fill-free probe check at off suffix L16, near:
−0.12* [−0.22, −0.02].

Which part predicts the choice (coordinates from the other fold, per α = 1): at the
stage B cell invariant −0.01, in-span −0.03, U −0.02, all CIs spanning 0; across cells
no direction beats random directions per SD (smallest share of random directions doing
better: 0.14). Read-out cells after reasoning may run backwards (the choice moves the
code). Correlational expectation for stage B: |ΔP(near)| < 0.09 at α = ±1 per direction —
but α = 1 is 3–4 SD beyond the natural spread along these directions (0.26–0.38 α).

## 6. Steering directions and what is ready

`analysis/steering_prepare.py --pooled [--stage-b]`, `experiments/exp8_steering.py`
(stage A), `experiments/exp13_suffix_steer.py` (stage B), `queue_pristine_steer.sh`.
Inputs: `steering/inputs_Qwen3-8B_pristine{,_suffix}.{npz,json}`. 1,386 prompts in the fit
(anchors out), all cells pooled; the factor span is 11-D thinking off, 12-D on.

| cell | effects / invariant / in-span step length | in-span share | cos with old curve step: effects / invariant / in-span / U |
|---|---|---|---|
| A: off suffix L16 | 1.30 / 0.82 / 1.00 | 0.60 | +0.75 / +0.73 / +0.40 / +0.20 |
| A: off pre-label L22 | 4.29 / 2.84 / 3.18 | 0.52 | +0.72 / +0.56 / +0.45 / +0.10 |
| A: off pre-label L24 | 13.06 / 8.10 / 10.11 | 0.55 | +0.78 / +0.53 / +0.58 / −0.17 |
| A: on pre-label L22 | 3.69 / 2.56 / 2.63 | 0.50 | +0.67 / +0.54 / +0.41 / +0.02 |
| A: on pre-label L24 | 7.33 / 4.86 / 5.44 | 0.54 | +0.60 / +0.55 / +0.32 / −0.10 |
| B: on suffix L16 | 1.57 / 1.00 / 1.20 | 0.61 | +0.78 / +0.63 / +0.50 / +0.10 |

Stability (`analysis/direction_stability.py`; effects / invariant / in-span / U):

| cell | fold 0 vs 1 | old vs pooled |
|---|---|---|
| suffix L16 (off, on) | 0.92 / 0.80 / 0.90 / 0.98 | 0.87 / 0.73 / 0.64 / 0.87 |
| off pre-label L22 | 0.79 / 0.74 / 0.76 / 0.94 | 0.80 / 0.73 / 0.69 / 0.79 |
| on pre-label L22 | 0.56 / 0.47 / 0.55 / 0.78 | 0.78 / 0.74 / 0.68 / 0.79 |
| on pre-label L24 | 0.56 / 0.42 / 0.56 / 0.81 | 0.76 / 0.72 / 0.66 / −0.75 |

- The in-span share (0.50–0.61) overstates real overlap: the span includes every small,
  noisy factor effect; a synthetic test gives > 0.15 with no planted overlap.
- U's sign is set to point from short to long along the curve; at the answer slot its
  alignment with the curve is only 0.07–0.40, so U's sign there is close to arbitrary
  (it flips between old and pooled fits at pre-label L24).
- Prompts steered (natural renderings, in Alan's range, no anchors): stage A 275
  near/both (176 in E8); stage B 96 over all 48 durations: 56 none, 19 near, 21 both.
- Time: stage A about 1.3 h, stage B about 3.6 h on one GPU.

### Decision for Jared: crossing-eligible prompts in stage B

At α = ±1 the steered horizon passes a delay in only 44 of the 96 stage B prompts
(20 for +1, 24 for −1). The in-range pool has 311 prompts with a delay within one
decade of the horizon (134 near, 87 both, 90 none), so a stage B set made only of
crossing-eligible prompts is possible. It would double the crossings at the same
cost, but departs from 16 §5's "2 per duration". Not changed.

## 7. What to run

From `jared/`, on the xpu env, in order:
```
~/miniconda3/envs/intel_ai/bin/python local_checks.py pooledbank
~/miniconda3/envs/intel_ai/bin/python local_checks.py stagea
~/miniconda3/envs/intel_ai/bin/python local_checks.py stageb
```
Then on the box, after `just push` (the inputs ship in `steering/`):
```
cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
tmux new -s steer 'bash queue_pristine_steer.sh; bash'
```
Smoke notes: the 0.8B smoke pools full + units only (`anchor_check` reads the 8B
folders); a 32-token think budget can make "forced" collinear with a duration and the
fit refuse; stage B's `--limit` takes the shortest durations (regime none), so the smoke
checks plumbing, not crossings.

## 8. Reading stage A/B against tonight

- Stage B, invariant condition: the clean test. If it moves choices and the reasoning's
  stated horizon, the decision reads a delay-free horizon code at the suffix.
- Stage B, effects-curve condition: carries ~0.3 decades of delay per decade of
  horizon; a difference between it and the invariant condition is attributable to the
  in-span content.
- Stage A thinking-on pre-label results rest on directions with fold cosine ≈ 0.5;
  treat them as exploratory.

## Files

New: `analysis/delay_step.py`, `analysis/axis_delay_purity.py`,
`analysis/choice_from_code.py`, `analysis/direction_stability.py`,
`experiments/exp13_suffix_steer.py`, `queue_pristine_steer.sh`, tests
`test_delay_step.py`, `test_axis_delay_purity.py`, `test_choice_from_code.py`,
`test_direction_stability.py`, `test_pristine_steering.py`, `synthetic_pooled.py`.
Changed: `analysis/pristine_manifold.py` (gate), `analysis/delay_margin.py` (optional
`bins`), `analysis/steering_prepare.py`, `experiments/exp8_steering.py`,
`experiments/exp10_patch_reasoning.py` (shared claim loop, behaviour unchanged),
`spar_horizon/steering.py` (`span_part`, `unit_direction`, `factor_span`, additive
`prompt_patch`), `local_checks.py` (`pooledbank`, `stagea`, `stageb`), 16 §6 task 7 note.
Outputs in `results/horizon/analysis/`. Minor: recomputing `realism_labels` from
log-rounded delays marks ~20% of old prompts out of range at the 1-week boundary; use
the records' `alan_range` (old records are in range by construction).
