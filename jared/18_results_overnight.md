# Results: overnight confound sweep on the saved runs (2026-10-29)

Unattended session, saved activations and answers only: no model was run. Runs used:
8B full and units banks, both modes (986 prompts with a horizon), and 14B full bank,
both modes (471). The question throughout is what, other than the horizon, moves the
horizon code **within a duration**. Duration fixed effects throughout, so nothing that
only varies with the horizon can score. Effects are in equivalent decades of horizon:
- 0.1 decades = 26%;
- 0.3 decades = 2×.

Scripts (each with tests in `tests/`), outputs in `results/horizon/analysis/`:
- `analysis/confound_sweep.py` → `confound_sweep.{json,txt}`, about 40 min
- `analysis/delay_margin.py` → `delay_margin.{json,txt,png}`
- `analysis/far_margin.py` → `far_margin.{json,txt,png}`
- `analysis/confound_robustness.py` → `confound_robustness.{json,txt}`
- `analysis/delay_position.py` → `delay_position.{json,txt}`
- `analysis/unit_staircase.py` → `unit_staircase.{json,txt,png}`
- `analysis/pristine_confounds.py` → `pristine_confounds.{json,txt}`

§8 and the behaviour and E8 regroupings in §1 were one-off computations in the session,
not scripts; their numbers are as printed. Folds are assigned by a stable hash of the
base id, so every script reproduces its numbers exactly.

## 1. Summary

- **The readout decides what looks confounded.** The same cell gives different
  confound profiles under different readouts:
  - probes with 1–16 duration-average PCs;
  - the projection on the horizon axis.

  A probe with enough components learns to subtract a nuisance that has a direction of
  its own. A low-dimensional readout can't, and the 1–4-dimensional steering curve is
  one. So "the probe is clean" doesn't mean "the steering curve is clean". Only effects
  that hold across readouts are reported as robust below.
- **Every readout fitted to all prompts reads some regime as horizon** on the old banks,
  because the regime mix changes with the duration: long durations are mostly near/both.
  - **The clean test:** fit the probe and the axis on one regime and read the others
    against it.
    - Up to about 1 year: the reference is regime-none prompts, read against the near
      delay (§3).
    - Above 1 year: the reference is regime-near prompts, read against the far delay
      (§4).

    At 8B suffix L16 and 14B suffix L20 the code is then flat within about ±0.1
    decades around both crossings, against +0.2–0.8 for "regime" under the all-prompt
    probe. So most of that is the probe absorbing regime. The one exception is far
    delays 10–30× inside the horizon at 8B L16 (−0.27, n = 12).
  - **At the later suffix layers (8B L24, 14B L27) a graded horizon-versus-delay signal
    remains** with a clean probe: 0.2–0.6 decades, growing the deeper a delay sits inside
    the horizon, with no step at the crossing.
  - **What the fill bank adds:** none prompts at long durations (13 cells), the one
    comparison the old banks cannot make.
- **Robust confound 1, the near delay, before reasoning, both models.** On regime-none
  prompts with everything fitted on them, where no delay is inside the horizon, a 10×
  longer near delay reads the horizon shorter, under every readout:
  - 8B suffix L16: −0.04 to −0.15 decades;
  - 14B suffix L20: −0.08 to −0.39;
  - 14B suffix L27: −0.09 to −0.50.

  The smallest value in each range is the 16-PC probe's, which can subtract part of it.
  This is the graded horizon-versus-delay comparison of `13` E2, now measured as a bias
  on the horizon code itself.

  At 8B L16 it is positional, not semantic (§6): the delay listed second, the one nearer
  the horizon in the text, does the pulling, and that is also where 8B's option-order
  effect comes from. By 8B L24 and 14B L27 the pull follows the near/far roles instead. It is not in the units-bank prototype's fit: the delays
  enter only in the pooled fit (`16` §4), so pooling with delay covariates was the right
  call.
- **Robust confound 2, the unit: a number line, 8B and now 14B.** At a fixed duration,
  a bigger unit (smaller number) reads shorter. Its size depends on the axis (§7):
  - −0.6 decades per decade of unit along the factor fit's duration axis, as `17` §6
    said. This is the curve we would steer along.
  - −0.17 along the raw duration-average axis (the old E8 curve, all probes). That axis
    has already absorbed part of the unit, because natural units grow with the horizon.

  The two unit-effect estimates agree (cosine 0.96, equal length), so only the axis
  differs.

  It also shows on the full bank's natural renderings, as a drop in the readout at unit
  boundaries ("43 hours" → "3 days"; §7):
  - 8B L24: −0.20 decades;
  - 14B L20 and L27: −0.12 and −0.18;
  - 8B L16: none (+0.02).

  That is the first number-line evidence for 14B, which has no units bank.
- **Option order is real but readout-dependent.** It is large in low-dimensional
  readouts and the axis, and gone with 16 PCs:
  - 8B L16: −0.02 to +0.30 on all prompts, 0.00 to −0.19 on none only;
  - 14B L20: down to −0.61.

  It moves the code, so a low-dimensional steering curve must have it projected out (the
  pristine fit does).
- **Rewards barely move the code** (≤ 0.1 decades per decade of reward, sign varies by
  readout). There is no sign of a generic "big number" axis shared with the rewards.
- **The answer slot (pre-label) carries everything:**
  - regime: up to ±1 decade;
  - forced closes: +0.4–0.9;
  - think length: up to −1.4 per decade;
  - the choice: ±0.2–0.6.

  Nothing new; it confirms the answer slot as an average-level manifold only.
- **Behaviour: option order drives 8B's choice with thinking off.**
  - In regime none, 8B off picks near 88% of the time when near is listed first and 44%
    when second.
  - In "both" (units bank): 71% against 0%.
  - Thinking on shrinks it (62/46), and 14B barely has it (70/63 off).
- **The planned pristine fit removes most of this at 8B suffix L16 (§9).** Pooled
  runs, with the delays as covariates and nuisance directions projected out (k = 4):
  - near delay: −0.14 → −0.03;
  - unit size: −0.09 → +0.01;
  - regime near: +0.30 → +0.09.

  Regime both (+0.38 → +0.23) and order (+0.07 → −0.05) remain, and L24 keeps a
  near-delay and unit pull of −0.07 to −0.09.
- **E8's answer-slot steering is not an order or forced-close effect.** Its per-prompt
  effects toward the rational answer, grouped by option order and by forced vs natural
  close, differ by at most 0.03 C, inside the CIs, both modes.

## 2. The sweep: one readout, every cell (`confound_sweep.txt`)

Probe: top-8 duration-average PCs, cross-fitted by base. Equivalent decades (readout
coefficient / calibration); * = 95% CI (bases resampled) excludes 0. Selected rows:

| cell | near delay | far delay | reward | order | unit size | regime near | regime both |
|---|---|---|---|---|---|---|---|
| 8B full+units on, suffix L16 | −0.13* | +0.05* | +0.01 | +0.13* | −0.08* | +0.22* | +0.40* |
| 8B full+units on, suffix L24 | −0.12* | +0.06* | −0.05* | +0.19* | −0.22* | +0.39* | +0.54* |
| 8B full+units on, pre-label L22 | −0.24* | +0.18* | +0.01 | +0.23* | +0.08 | +0.22 | +0.80* |
| 14B full on, suffix L20 | −0.22* | +0.11* | +0.06* | −0.05 | – | +0.48* | +0.82* |
| 14B full on, suffix L27 | −0.10* | −0.15* | +0.05 | −0.08* | – | +0.23* | +0.12 |

Read the regime columns with §1's caveat: this probe is fitted to all prompts.

Grouped by duration third (8B, 14B, suffix): the near-delay effect is strongest in the
middle third (about 7 hours to 7 months, −0.18 to −0.53), where the old banks' delays
(1 week – 1 year) sit closest to the horizon.

## 3. Delay margin, against regime-none prompts (`delay_margin.png`)

Within a duration, each prompt's readout minus the none prompts' mean, binned by
log10(near delay / horizon). The probe and axis are fitted on none prompts only.

![delay margin](../../results/horizon/analysis/delay_margin.png)

- **Suffix L16 (8B) and L20 (14B):** flat within about ±0.1 decades over the whole range,
  including where the near delay falls inside the horizon. There is no step at 0, and a
  shallow inverted U (highest at +0.25 to +1.25).
- **8B suffix L24:** −0.19 to −0.30 when the near delay is inside the horizon (CI excludes
  0). That is the opposite sign to the all-prompt probe's "regime near" effect, so at L24
  regime has its own direction, which readouts weigh with either sign.
- **Answer slot:** a drop of decades on the axis once the near delay is inside the
  horizon. This is the decision, not the horizon.

## 4. Far margin at long durations, against regime-near prompts (`far_margin.png`)

Above about 1 year the old banks have no none prompts, so the reference is regime near
(only the near option pays). The probe and axis are fitted on near prompts, and near and
both prompts are read against them, binned by log10(far delay / horizon).

![far margin](../../results/horizon/analysis/far_margin.png)

- **8B suffix L16:** flat within ±0.1, except far delays 10–30× inside the horizon:
  −0.27 (n = 12).
- **14B suffix L20:** flat within ±0.05 (9 durations only).
- **8B suffix L24:** a graded fall the deeper the far delay is inside: −0.07, −0.24 and
  −0.60 decades at margins of −0.25, −0.75 and −1.25. No step at 0.
- **14B suffix L27:** +0.27 (probe) to +0.47 (axis) just inside, falling to about 0 by
  +0.75.
- The near-fitted probes are less calibrated (0.72–0.83, against 0.95 overall), because
  near prompts cover only 9–17 durations.

## 5. Readout robustness (`confound_robustness.txt`)

Equivalent decades by readout, thinking on; first all prompts, then none only:

| cell | readout | order | regime both | unit size | near delay |
|---|---|---|---|---|---|
| 8B suffix L16, all | k=1 / k=4 / k=16 / axis | +0.03 / +0.30 / −0.02 / +0.06 | +0.01 / +0.31 / +0.43 / +0.08 | −0.18 / −0.09 / +0.02 / −0.17 | −0.15 / −0.16 / −0.08 / −0.15 |
| 8B suffix L16, none only | k=1 / k=4 / k=16 / axis | −0.13 / −0.19 / +0.00 / −0.16 | – | −0.09 / +0.07 / +0.03 / −0.05 | −0.15 / −0.14 / −0.04 / −0.15 |
| 14B suffix L20, none only | k=1 / k=4 / k=16 / axis | −0.61 / −0.42 / −0.06 / −0.49 | – | – | −0.39 / −0.22 / −0.08 / −0.31 |
| 14B suffix L27, none only | k=1 / k=4 / k=16 / axis | −0.26 / −0.19 / −0.02 / −0.25 | – | – | −0.50 / −0.35 / −0.09 / −0.47 |

## 6. Which delay pulls: role or position? (`delay_position.txt`; exploratory)

Regime-none prompts, everything fitted on them. The same delays described two ways,
each with option order:
- **role:** near, far;
- **position:** listed first, listed second.

The second-listed delay is the one nearer the CONSTRAINT line in the text.

| cell (readout) | role: near / far / order (R²) | position: first / second / order (R²) |
|---|---|---|
| 8B L16 (k=4) | −0.13 / −0.11 / −0.19 (0.18) | −0.05 / −0.18 / +0.03 (0.20) |
| 8B L16 (axis) | −0.14 / −0.08 / −0.16 (0.15) | −0.05 / −0.17 / +0.05 (0.16) |
| 8B L24 (axis) | +0.19 / −0.29 / +0.07 (0.21) | −0.02 / −0.01 / +0.04 (0.01) |
| 14B L20 (axis) | −0.31 / −0.07 / −0.49 (0.51) | −0.16 / −0.28 / −0.30 (0.48) |
| 14B L27 (axis) | −0.46 / +0.00 / −0.25 (0.43) | −0.20 / −0.36 / −0.00 (0.32) |

- **8B L16:** position fits slightly better, and the order effect vanishes there. The
  horizon code is pulled by whichever delay is textually nearer, which is a proximity
  effect, not a comparison.
- **8B L24 and 14B L27:** role fits clearly better. The pull follows near and far, as a
  comparison would.
- **14B L20:** the two fit equally well.
- Post hoc, on prompts up to about 1 year. A bank that varies text position and role
  separately (for example, the CONSTRAINT line before the options) would test it.

## 7. Unit staircase on the natural renderings (`unit_staircase.{txt,png}`)

Full bank, natural units only. Leave-one-duration-out readout (top-8 PCs), averaged per
duration, residual from a straight line. The two durations at each end are dropped,
because they read pulled inward. A number line runs steeper within a unit and drops at
each boundary.

![unit staircase](../../results/horizon/analysis/unit_staircase.png)

| cell | overall slope | within-unit slope | mean boundary step [CI] |
|---|---|---|---|
| 8B on suffix L16 | 0.97 | 0.95 | +0.03 [−0.10, +0.14] |
| 8B on suffix L24 | 0.98 | 1.09 | −0.20 [−0.33, −0.07] |
| 14B on suffix L20 | 0.97 | 0.99 | −0.13 [−0.25, −0.03] |
| 14B on suffix L27 | 0.95 | 0.98 | −0.18 [−0.34, −0.04] |

- Thinking off matches thinking on at every suffix cell.
- The answer slot is too noisy to read.
- **Weak evidence:**
  - The per-duration residuals scatter by about ±0.5 decades (about 10 prompts per
    duration on the full bank), as much as the boundary steps.
  - The six boundaries differ a lot from one another (±0.5).
  - The CIs resample prompts, not boundaries, so they understate the uncertainty.

  Read only the sign and the layer trend (8B: none at L16, present at L24); the units
  bank's within-duration measure (§1) is the stronger evidence for a number line.

**Why §1's two numbers for the unit differ (−0.6 against −0.17).** Pooled 8B, thinking on,
suffix L16:
- the within-duration unit vector from the all-feature regression and the factor fit's
  unit trend agree (cosine 0.96, length ratio 1.0);
- projected on the factor fit's duration axis (unit held fixed, |axis| 1.12) both give
  −0.58 to −0.60;
- projected on the raw duration-average axis (|axis| 0.69, cosine 0.70 with the
  other) both give −0.16 to −0.17.

The raw axis already contains part of the unit trend, because natural units grow with
the horizon. So it is rotated away from the unit direction, and a probe or curve built on
it shows a smaller unit bias. L24 is the same (−0.50 to −0.57 against −0.16 to −0.28).

## 8. Can the full and units runs be pooled? (8B, matched durations)

The two runs were made with different thinking-on batch sizes (12 and 20). For each of the
33 durations both banks share, a probe fitted on the full bank without that duration
reads the units bank's natural renderings and the full bank's own prompts there:

| cell | units − full readout, mean over durations [95% CI] |
|---|---|
| 8B off suffix L16 / L24 / pre-label L24 | +0.03 [−0.03, +0.08] / +0.04 [−0.01, +0.08] / +0.01 [−0.10, +0.13] |
| 8B on suffix L16 / L24 / pre-label L24 | +0.03 [−0.03, +0.09] / +0.03 [−0.01, +0.08] / +0.05 [−0.07, +0.16] |

No offset (≤ 0.05 decades, every CI includes 0). The fill bank's anchors (`16` §4) test
the same thing for the new run.

## 9. Does the pristine fit remove it? (`pristine_confounds.txt`)

8B pooled old runs, both banks. Each readout is a curve cross-fitted by base,
nearest-point readout, k = 4:
- **old curve:** horizon averages;
- **effects ⟂:** factors unit, regime, forced and order, log near and far delays as
  covariates, their span projected out (`16` §2, §4).

Then the sweep's within-duration regression, in equivalent decades:

| cell | readout | near delay | far delay | order | unit size | regime near | regime both | reward |
|---|---|---|---|---|---|---|---|---|
| on suffix L16 | old curve | −0.14* | +0.01 | +0.07* | −0.09* | +0.30* | +0.38* | +0.07* |
| | effects ⟂ | −0.03 | −0.03 | −0.05* | +0.01 | +0.09 | +0.23* | −0.06* |
| on suffix L24 | old curve | −0.08* | −0.10* | +0.01 | −0.15* | +0.48* | +0.63* | +0.05* |
| | effects ⟂ | −0.09* | −0.06* | −0.03 | −0.07* | +0.06 | +0.12 | −0.04* |
| off suffix L16 | old curve | −0.12* | −0.01 | +0.08* | −0.06* | +0.27* | +0.34* | +0.06* |
| | effects ⟂ | −0.02 | −0.04* | −0.05* | +0.01 | +0.07 | +0.15 | −0.08* |
| on pre-label L22 | old curve | −0.27* | +0.28* | +0.40* | −0.17 | +1.61* | +2.94* | +0.02 |
| | effects ⟂ | −0.07 | +0.15* | −0.17* | +0.07 | +0.71* | +1.26* | +0.10* |

- **Suffix L16, both modes: the fit does what it was built for.**
  - Near delay, unit and regime near fall to ≤ 0.09 decades (≤ 23%).
  - Regime both keeps +0.15–0.23. That is the regime × duration interaction, which a
    main-effects fit cannot remove (`17` §1, P4). On the old banks, regime both exists
    only at long durations.
  - Reward (−0.06 to −0.08) is not in the fit and is small.
- **Suffix L24:** the regime effects drop (+0.48/+0.63 → +0.06/+0.12), but a near-delay
  and unit pull of −0.07 to −0.09 stays. That is the graded comparison and the number
  line of §4 and §7, which vary within the factors' levels.
- **Pre-label:** the fit halves everything, but regime stays at +0.7 to +1.3. As before,
  the answer slot can't be cleaned by a main-effects fit.
- Calibration of the nearest-point readout is 0.99 at the suffix (0.75 → 0.93 at
  pre-label).

## 10. What this means for the manifold and the box run

- **Keep the delays as covariates in the pooled fit (`16` §4).** The near-delay bias is
  real before reasoning in both models and lies along the axis that low-dimensional
  readouts use. With them, the fit removes it at 8B L16 (§9).
- **Order must stay a factor, and be projected out.** It is real in low dimensions.
- **Steer before the comparison forms: 8B suffix L16, and 14B suffix L20 for any 14B
  replication.** Under a clean probe these cells are flat across both delay crossings and
  have no unit staircase in 8B. 8B L24 and 14B L27 carry a graded horizon-versus-delay
  signal and a number line, which the fit only partly removes (§9). That argues for
  stage B at L16, as planned.
- **The remaining L16 confound is regime both at long durations** (+0.15–0.23 after the
  fit). The fill bank's long-duration none and near cells are what let a fit separate
  it, so the pooled rerun after the fill should be checked with
  `analysis/pristine_confounds.py` first.
- **At L16 the remaining delay pull is textual proximity** (the second-listed delay).
  The fill bank keeps Alan's template, so this stays. The delay covariates absorb it,
  but a template-order variant would be needed to remove it at the source.
- **The fill bank is what can test the regime confound above 1 year.** The old banks have
  no none prompts there, so no clean probe can be fitted.
- **Steering contrast:** 8B's thinking-off choices are half order-driven, so its effects
  in C units mix the order bias into the baseline. Keep reporting thinking-on as primary.
