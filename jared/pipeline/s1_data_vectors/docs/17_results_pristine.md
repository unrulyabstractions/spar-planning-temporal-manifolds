# Results: E13 prototype, a unit-free horizon manifold on the 8B units bank (2026-10-27)

Spec: `16_spec_pristine_manifold.md` §3. Saved activations only:
- `pipeline/s1_data_vectors/analysis/pristine_manifold.py` → `results/horizon/analysis/pristine_manifold.{json,txt,png}`;
- `pipeline/s3_analysis/by_datatype/written_unit.py` → `written_unit.json`.

Post hoc in one respect: the P1 definition changed after the first two cells were
seen (§3, decision needed).

## 1. Summary

- **Projecting the nuisance directions out of the basis is what makes the manifold
  clean** ("effects ⟂"). Fitting duration effects without that step is worse than
  the old curve everywhere: the unit stays inside the basis and the readout jumps
  (P1 300–700%).
- **Suffix newline (before reasoning), both modes, L16–L27:** effects ⟂ cuts the
  unit gap to a third or a half of the old curve's:
  - systematic P1: 23–43% against 45–165%;
  - forced closes (P3 0.50–0.54) and, at L16, regime (P4 0.55) are gone;
  - within-unit ordering on rendering averages: 0.82–0.88, against 0.42–0.80.

  It still misses the absolute thresholds (P1 ≤ 20%, P2 ≤ 50%).
- **Pre-label, thinking on (the E8 steering cells):**
  - effects ⟂ removes the forced-close signal that dominated the old curve (P3 0.51–0.56 against 0.88–0.99);
  - on rendering averages it orders durations within a unit at 0.50–0.60 (L22/L24), against 0.25–0.29;
  - **per prompt it reads no better than the unit word**: P2 249–336% against a unit lookup's 284%, within-unit R² −0.9 to −2.1.

  The magnitude code there exists on averages and is swamped by reasoning noise per prompt.
- **Think close (thinking on):** no usable magnitude code with any fit. On averages it is
  ≤ 0.31 within unit, and per prompt P2 is 540–770%.
- **P4 fails at the suffix from L22 (0.65–0.69).** Projecting out the regime main effect
  leaves its interaction with duration, which is the horizon × delay signal of `12` §2 (E1).
  A main-effect regression cannot remove it.

## 2. Table (k = 4; old curve → effects ⟂)

Percent errors are `10^|r − t| − 1`:
- **P1 sys:** each rendering's mean pair gap over its bases, median over renderings;
- **P1 pair:** median per-pair gap;
- **P2:** leave-one-duration-out per prompt, unit lookup 284%;
- **wuR² avg:** within-unit R² on rendering averages, unit-only floor ≈ −0.25;
- **P3:** forced-close AUC;
- **P4:** largest within-duration AUC over the regime pairs.

| cell | P1 sys | P1 pair | P2 | wuR² avg | P3 | P4 |
|---|---|---|---|---|---|---|
| off suffix L16 | 99 → 40 | 79 → 36 | 92 → 74 | 0.80 → 0.82 | – | 0.54 → 0.55 |
| off suffix L24 | 55 → 34 | 100 → 18 | 140 → 73 | 0.65 → 0.88 | – | 0.62 → 0.66 |
| off pre-label L22 | 72 → 25 | 17 → 14 | 150 → 138 | 0.56 → 0.79 | – | 0.63 → 0.57 |
| off pre-label L24 | 77 → 52 | 34 → 27 | 149 → 149 | 0.32 → 0.68 | – | 0.57 → 0.54 |
| on suffix L16 | 68 → 43 | 91 → 43 | 86 → 74 | 0.79 → 0.83 | 0.54 → 0.50 | 0.54 → 0.55 |
| on suffix L24 | 92 → 16 | 80 → 28 | 136 → 78 | 0.70 → 0.86 | 0.54 → 0.53 | 0.63 → 0.67 |
| on think close L24 | 588 → 277 | 3695 → 1204 | 1202 → 647 | −0.04 → 0.30 | 1.00 → 0.54 | 0.67 → 0.56 |
| on pre-label L22 | 191 → 114 | 771 → 195 | 458 → 260 | 0.25 → 0.60 | 0.99 → 0.53 | 0.67 → 0.63 |
| on pre-label L24 | 202 → 109 | 515 → 109 | 464 → 249 | 0.29 → 0.50 | 0.98 → 0.51 | 0.72 → 0.68 |

All cells and k = 2/8 are in `pristine_manifold.txt`.

The P1 sys column above is the first estimator: the median over renderings of |mean|. It
is inflated by the noise between traces, and its CIs were biased upward. §7 replaces it
with the debiased root mean square.

## 3. Pre-registered verdicts, and the decisions needed

As written in `16` §3 (k = 4), effects ⟂ passes all four checks at no cell:
- **P3** passes everywhere (thinking on);
- **P2 ≤ 50%** fails everywhere: the best is 67% at suffix L22;
- **P1 ≤ 20%** passes on the per-pair gap at off suffix L24 (18%) and off pre-label L22/L27,
  and on the systematic gap at on suffix L24 (16%).

**Decisions for Jared:**
1. **P1 definition.** The per-pair gap counts the reasoning noise between two traces as
   unit error. At pre-label with thinking on, that alone is hundreds of percent. Proposed:
   judge the systematic gap, as `unit_overlap.py` and E9 do.
2. **P2 level.** Per prompt, ≤ 50% is under one grid step; no cell reaches it before or
   after reasoning. Steering moves along a manifold fit on averages. Proposed: judge P2 on
   rendering averages (within-unit R² ≥ 0.5, the earlier suffix criterion) and report per
   prompt alongside. Changed after seeing the data, so it is marked post hoc.
3. **P4 at the suffix from L22.** Either accept that a duration manifold carries the
   duration × delay comparison, or remove it in the crossed bank, where delays are
   relative to the horizon and the regime is balanced per duration.

## 4. Written unit (step 0, side view)

- 511 of 515 traces mention the horizon. Natural renderings are always reasoned about in
  their own unit.
- Rewrites usually state a conversion ("600 seconds, which is 10 minutes") but keep
  reasoning in the rendered unit. The exception is months: 48 of 62 "N months" prompts
  reason mainly in years.
- Only 60 of 248 pairs reason in the same main unit.
- P1 is lower for those pairs, on the old curve and on effects ⟂ (on suffix L16: 23% same
  against 57% different).

## 5. What this means for the crossed bank and steering

- The fit works where the horizon is a clean input, at the suffix, and partly at the
  answer slot.
- The cleanest steering target is the suffix (L16–L24), before reasoning: the E10 position
  family, now with a magnitude manifold.
- The answer slot is usable only as an average-level manifold. Per prompt, the model's
  reasoning noise dominates.

## 6. Added after the write-up: the unit directions are not orthogonal to the horizon axis

Measured on the same fits (scratch check, not yet in `pristine_manifold.py`):
- **55–62% of the duration curve's variance lies in the unit-effect span** at the
  suffix, both modes, at L16 and L24; at pre-label it is 36–56%. That in-span part is
  ordered by horizon: one component gives R² 0.99 against log horizon at the suffix.
- **At a fixed duration, the main axis follows the number, not the unit.** The unit
  effects projected onto the curve's per-decade axis (thinking on, suffix L16) are, in
  decades: years −1.57, months −1.11, weeks −0.44, days 0, hours +0.76, minutes +1.57,
  seconds +2.74.

  A bigger unit means a smaller number, and reads shorter: about 0.6 decades per decade
  of unit. So "36 months" sits further along the axis than "3 years". Pre-label L24 shows
  the same pattern, weaker (years −1.27 … seconds +0.73, hours out of order).
- **Implications:**
  - The main horizon axis is partly a number line. The model needs the unit to get the
    magnitude, but along this axis it applies it only partly.
  - "Effects ⟂" removes that whole axis. What remains is the component that is the same
    for every rendering of a duration: still ordered by horizon (top PC R² 0.83–0.96),
    but the smaller part of the curve.
  - "Effects" without ⟂ keeps the axis, so a step along it is "a bigger number" as much
    as "a longer time".
  - Within the additive fit this cannot be split further: a linear number effect is
    duration minus unit, so it is absorbed into those two factors. The crossed bank
    does not change that.
- **For steering (`16` §5):** steer three directions at matched length: the ⟂ curve
  (rendering-invariant duration), the removed main axis (number-heavy), and the unit
  direction U.

  If choices follow the ⟂ curve and not the main axis, the decision reads the integrated
  duration. Behaviour already hints at this: rewrites move the main axis by about a
  decade at the suffix, yet the signed choice barely moves (+0.28 log-odds, 14% of labels
  flip, no noise floor yet).
- The synthetic test planted unit effects in random directions, nearly orthogonal to the
  duration curve, so it could not show this case.
- **Checked overnight (`18` §1, §7):**
  - The −0.6 holds on the pooled banks, for any k.
  - It is measured along the factor fit's duration axis. Along the raw duration-average
    axis (E8's curve, all probes), which has absorbed part of the unit, the same unit
    vector reads −0.17.
  - The pristine fit removes it at suffix L16 (`18` §9).

## 7. Rerun under the approved criteria (Jared, 2026-10-27)

Decisions, recorded in `16` §3:
- P1 is judged on the systematic gap;
- P2 on rendering averages (within-unit R² ≥ 0.5);
- P4 is left for the crossed bank to decide.

**The P1 estimator changed.** It is now the root mean square over renderings of each
rendering's mean pair shift, with the squared mean reduced by var / n, and a CI that
resamples durations. On synthetic data (60 renderings, 4 bases, noise 0.3 decades):
- the old median |mean| read 21% with no shift at all;
- the new estimator reads 7% (clipping at zero keeps it slightly positive);
- at a true 26% it reads 24%, and at 58% it reads 58%;
- CI coverage was 35–40 of 40.

It is not comparable to §2's sys column: a root mean square weights the worst renderings
more than a median.

k = 4, old curve → effects ⟂:

| cell | P1 sys | wuR² avg | P2 per prompt | P3 | P4 | ⟂ passes |
|---|---|---|---|---|---|---|
| off suffix L16 | 111 → 48 [29, 63] | 0.80 → 0.82 | 74 | – | 0.55 | P2 P4 |
| off suffix L24 | 115 → 51 [21, 86] | 0.65 → 0.88 | 73 | – | 0.66 | P2 |
| off pre-label L22 | 192 → 63 [17, 106] | 0.56 → 0.79 | 138 | – | 0.57 | P2 P4 |
| on suffix L16 | 110 → 50 [31, 69] | 0.79 → 0.83 | 74 | 0.50 | 0.55 | P2 P3 P4 |
| on suffix L24 | 133 → 49 [19, 86] | 0.70 → 0.86 | 78 | 0.53 | 0.67 | P2 P3 |
| on think close L24 | 446 → 0 [0, 206] | −0.04 → 0.30 | 647 | 0.54 | 0.56 | P1 P3 P4 |
| on pre-label L22 | 113 → 0 [0, 117] | 0.25 → 0.60 | 260 | 0.53 | 0.63 | P1 P2 P3 |
| on pre-label L24 | 357 → 65 [0, 168] | 0.29 → 0.50 | 249 | 0.51 | 0.68 | P2 P3 |

All cells are in `pristine_manifold.txt`.

- **No cell passes all four.** The closest is on suffix L16, which fails only P1 (50% against 20%).
- **At the suffix,** effects ⟂ cuts the systematic unit gap to between a half and a tenth of the old curve's at every cell
  (39–57% against 110–375%), but leaves about 50%. That residue is not the unit's main
  effect, which is projected out. It is the unit effect varying with duration, an
  interaction.
- **P1 "passes" at think close and pre-label only for lack of power.** The point
  estimate is 0, but the CI reaches 117–206%: the noise between traces swamps any
  systematic gap. A pass there is not evidence that no unit is left. On the crossed bank,
  with about 4 bases per rendering, the same will hold.
- **Consequence for the steering gate (`16` §5):** as written, it would block steering
  at every cell. A decision for Jared:
  - keep the gate;
  - judge P1 on "at most half the old curve's" (every suffix cell passes);
  - or run the three-direction comparison ungated, since it asks which part of the
    code the decision reads, which does not need a pristine manifold.

## 8. Pooled baseline: the old runs only, before the fill bank (2026-10-29)

`pristine_manifold.py <root> pooled` with no fill run fits the old full and units runs
together: 986 prompts with a horizon, both modes, log near and far delays as
covariates. It is the baseline the fill bank should improve.
→ `pristine_manifold_pooled.{json,txt,png}`

| cell | P1 sys, old → ⟂ | wuR² avg ⟂ | P3 | P4 ⟂ | ⟂ passes |
|---|---|---|---|---|---|
| off suffix L16 | 190 → 78 [50, 105] | 0.85 | – | 0.59 | P2 P4 |
| on suffix L16 | 175 → 77 [51, 106] | 0.85 | 0.57 | 0.54 | P2 P3 P4 |
| on suffix L24 | 170 → 99 [50, 149] | 0.82 | 0.58 | 0.59 | P2 P3 P4 |
| on pre-label L22 | 142 → 39 [0, 139] | 0.75 | 0.57 | 0.54 | P2 P3 P4 |
| on pre-label L24 | 9 → 0 [0, 77] | 0.62 | 0.55 | 0.60 | P1 P2 P3 |

Compared with the units bank alone (§7):
- **Better:**
  - Magnitude on averages at pre-label with thinking on: 0.62–0.75, against 0.50–0.60.
  - The regime leak at the suffix is gone: P4 0.54–0.61, against 0.65–0.69.
- **Worse:** the systematic unit gap at the suffix is 59–99%, against 39–57%. Only the
  units bank has pairs, so P1 measures the same pairs under a fit that the natural-unit
  full bank now dominates. The fill bank adds crossed renderings at every duration with
  a gap, which is where the unit factor is estimated.
- **On pre-label L24, P1 passes only for lack of power** (CI to 77%), and P4 sits at the
  bar (0.60).
- The run takes 38 min locally for 986 prompts; with the fill, about 55 min.
