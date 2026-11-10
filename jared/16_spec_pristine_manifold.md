# Spec: E13, a horizon manifold free of the unit word (and of regime, forced closes, order)

Date: 2026-10-27. Draft for Jared's approval. Decisions (Jared, 2026-10-27):
- prototype the fit locally on the 8B units bank first;
- then a crossed bank: 2-significant-figure renderings over 0.1–1000, with the
  delays crossed too;
- 8B only.

Supersedes the steering parts of `15_spec_crossing.md` (B, and the units-bank
Part C discussed): those run with the manifold built here. `15` Part A stays as
a cheap check of the existing, confounded runs.

## 1. Why

Every steering vector so far (E8, E11) is a curve through horizon averages on
the full bank. There, three things move with the horizon:
- the unit word (it explains 0.97 of log horizon);
- the regime (every short horizon is `none`);
- the forced-close rate.

A step along that curve changes all three, so no steering result can be called
"magnitude". No rendering rule removes the unit confound over 30 s – 500 y:
the range is 7.7 decades and a readable unit covers about 3.5. 2-sig-fig
renderings over 0.1–1000 still leave a unit share of 0.76 (computed
2026-10-27). So the bank has to be combined with a fit that removes the unit
explicitly.

## 2. The fit (replaces horizon averages)

Per cell (position, layer), least squares over prompts:

`x_i ≈ μ + D[duration_i] + U[unit_i] + R[regime_i] + F·forced_i + O·near_first_i`

with one-hot factors, one level of each dropped.

The **duration effects** `D` (one vector per duration) replace the horizon
averages. The basis is the top-k PCs of `D`, and the manifold is `D` on that
basis, ordered by log duration. It is a drop-in for `fit_cell`'s output
(`basis`, `curve_logh`, `curve_coords`, `mean`), so `curve_delta` and E8 use it
unchanged.

The factors can be estimated separately only because units overlap across
durations:
- `U` comes from same duration / different unit;
- `D` comes from same unit / different duration.

**Readout:** project onto the basis and take the nearest point on the
interpolated curve. That point is a log duration.

## 3. Prototype: 8B units bank, saved activations, local

Data: `exp0_Qwen3-8B_think-{on,off}_units`. 515 prompts in pairs, 33 durations
from 3 min to 59 y, units overlapping by design (unit share 0.90). Its delays
are not crossed, so the prototype cannot test the delay comparison. It tests
only whether the fit can strip the unit, regime and forced closes.

**Step 0, the written unit (side view only, Jared 2026-10-27).** Decode the
cached traces with the tokenizer (no model). Record which unit the reasoning
first uses for the horizon, and how often it converts ("708 months" →
"59 years"). Report P1 split by whether the pair's traces wrote the same unit.
The fit does not use it.

**Cells:**
- suffix newline: L16, the best-decoded cell;
- `</think>` and pre-label: L22, L24, L27, around the steering cells and the hand-off;
- thinking on and off.

**Cross-fitting:** fit on half the base prompts (split by `pair_id`), check on
the other half, then swap. Both renderings of a pair stay on the same side.

**Error in percent (Jared, 2026-10-27).** A readout `r` (log10 years) against
the true duration `t` has error `10^|r − t| − 1`. So reading 2× or ½× is 100%,
and one step of the 48-point grid (0.18 decades) is 52%. Errors are reported as
median and 90th percentile, overall, per unit and per duration. Over- and
under-reading are reported separately: mean signed `r − t`, as a percent.

**Checks (criteria fixed now; k = 2, 4, 8 reported, k = 4 judged):**

| | what | pass |
|---|---|---|
| P1 | no unit left: for held-out pairs (same base, two renderings), the readout gap minus the true gap between the renderings, `10^|(r_A − r_B) − (t_A − t_B)| − 1` | median ≤ 20%, and at most half the horizon-average curve's on the same cell (E9's unit shift at the suffix was 0.30 decades, 100%) |
| P2 | magnitude present: leave-one-duration-out readout against the true duration, within unit | median error ≤ 50% (under one grid step), and below a unit lookup's on the same prompts; within-unit R² reported alongside |
| P3 | no forced closes: AUC of forced vs natural from the k coordinates | ≤ 0.60 (horizon-average curve: 1.00 at `</think>`) |
| P4 | no regime: AUC of regime within duration from the k coordinates | ≤ 0.60 |

The horizon-average curve on the same cells is run through the same checks, so
the table shows what the fit removes.

**Criteria as judged on the crossed bank (Jared, 2026-10-27, after the prototype:
`17` §3; post hoc with respect to the prototype, fixed before the crossed bank):**
- **P1** is judged on the systematic gap: per rendering, the mean pair gap over its
  bases, debiased for the noise between traces (`17` §2), root mean square over
  renderings. The per-pair median is reported alongside.
- **P2** is judged on rendering averages: within-unit R² ≥ 0.5. The per-prompt
  leave-one-duration-out error and the unit lookup are reported alongside.
- **P4** stays ≤ 0.60. The crossed bank's balanced regimes decide whether the
  suffix leak from L22 in the prototype was the units bank's imbalance.

**Reading:**
- **All four pass at a cell:** the fit works there, and the crossed bank is worth building.
- **P1 or P3/P4 fail at the answer slot but pass at the suffix:** the answer-slot
  code is not separable into duration and the rest, and steering there cannot be
  made pristine with this design.
- **P2 fails where the others pass:** removing the confounds removed the signal.
  There is no magnitude code at that cell to steer.

Output: `analysis/pristine_manifold.py` → `<results>/analysis/pristine_manifold.json`,
a table, and one figure per cell: both curves, with the paired renderings
joined by lines.

## 4. Then: crossed bank (outline; full spec after the prototype)

**Replaced by a fill bank pooled with the old runs (Jared, 2026-10-29).** The
standalone crossed bank below ties its delays to the horizon (corr +0.85 between
log horizon and log delay), so on its own a duration direction could be a delay
direction. The old 8B runs (full and units banks, 986 prompts with a horizon,
same settings and layout) have independent delays (corr −0.11), but regime is
nearly fixed by duration there: 45 of 48 durations lack a feasible regime. So:
- **Fill bank (`--bank fill`):** the crossed generator below, restricted to the
  (duration, regime) cells with fewer than 3 old prompts (72 cells: 31 both,
  28 near, 13 none), 2 bases per cell, each in every rendering of its
  duration. About 400 prompts, about 45 min of E0 thinking on.
- **Anchors:** 24 old prompts re-run in the same E0 run (12 full-bank prompts,
  6 units-bank pairs), marked `anchor`. Greedy decoding should reproduce their
  traces; their activations must match the old runs' within bf16 noise before
  anything is pooled. Anchors are left out of the fit.
- **Pooled fit:** old full + old units + fill, both modes, with the log near and
  far delays as two continuous factors next to unit, regime, forced and order;
  their effect directions are projected out with the rest (effects ⟂).
- The rules below (renderings, margins, limits, realism labels) apply to the fill.

- **Horizons:** the 48 durations, 30 s – 500 y, each rendered in every unit
  whose value lies in 0.1–1000. Use the exact integer when there is one
  (365 days, 168 hours), else 2 significant figures ("0.5 years", "1.4 hours").
  Singular units at 1. That gives about 133 renderings:
  - per duration: 1 rendering (5 durations, at the range ends), 2 (14), 3 (16), 4 (13);
  - per unit: 9 (seconds) to 22.
- **Regimes balanced, delays relative to the horizon (Jared, 2026-10-27).** Each
  duration's bases take every feasible regime in turn, starting one further
  along none / near / both at each duration so the fourth base rotates. (Changed
  in the build from a uniform draw per base, which left 33 of 48 durations
  without a regime.) Then delays at a log margin from the horizon `h` (decades;
  `m ~ U(0.2, 1.5)`, `g ~ U(0.3, 2)` the near-to-far gap):
  - none: near = h + m, far = near + g;
  - near: near = h − m, far = h + m′ (an independent draw of m);
  - both: far = h − m, near = far − g.

  The regime mix and the margin distribution are then the same at every
  duration, so regime no longer moves with duration. It also gives the
  crossing ladder (`15` §4) the same margins everywhere.
- **Delay limits:** 1 minute to 1,000 years. A regime is feasible at a duration
  if some margins put both delays inside the limits; margins are redrawn until
  they do. As built: 30–71 s are none only, 108 s and 3 min lack both, and
  326 and 500 y lack none. The achieved mix is 71 none / 62 near / 59 both bases.
- **Delay renderings:** the horizon rule. One rendering per delay is drawn per base
  and shared by its horizon renderings, so delay units are crossed with horizon
  units. After rounding, every delay must stay at least 0.2 decades from `h`, or
  the base is redrawn, so rounding never flips a regime.
- **Each base prompt** draws its context once (delays, rewards as Alan's, option
  order) and is emitted per horizon rendering, as in the units bank.
- **Size:** 192 bases, 4 per duration, 532 prompts, 338 of them near/both
  (121 independent bases). About 1 h of E0 on one GPU
  (thinking on, 8B). Steering statistics resample by base.
- **Realism cost:** payouts in minutes or hours ("5,000 dollars in 40 minutes")
  read oddly for an investment.
- **Realism labels (Jared, 2026-10-27): labels only; the analysis comes later.**
  Each record carries:
  - `delay_scale`, from the near delay, the first payout:
    - `minutes-hours`: under 1 day;
    - `days-weeks`: 1 day to under 1 month;
    - `months-years`: 1 month up;
  - `alan_range`: both delays within Alan's grid span, 1 week to 50 years.

  The delays in years are already on each record for anything finer. The
  generator prints the count per label × regime × duration third. Nothing is
  balanced on these labels or analysed with them in this spec.
- Generator: `build_crossed_prompts` in `spar_horizon/full_bank.py`, `--bank crossed`;
  `crossed_mix(records)` prints the achieved mix. Tests in `tests/test_full_bank.py`.
- Replication framing: the delays no longer follow Alan's generator, so results
  on this bank are ours, not a replication.

## 5. Then: steering with the pristine manifold

The `15` criteria (crossing, ladder), with the manifold from §2 in place of the
horizon-average curve.

**Directions (Jared, 2026-10-27, after `17` §6).** At a cell, the fit's duration
curve without the ⟂ step ("effects") splits into two parts that add up to it:
- **invariant:** its part outside the span of the factor effects, the same for every
  rendering of a duration (the ⟂ manifold);
- **in-span:** its part inside that span, mostly the main axis, which the number moves
  as well as the duration (`17` §6).

Steered at each α (decades along the effects curve, as E8's `curve`):

| direction | step |
|---|---|
| effects curve | `curve_delta` on the effects curve |
| invariant part | the effects step without its in-span component |
| in-span part | the remainder; the two parts add up to the effects step |
| unit `U` | the first principal component of the unit effects, at the effects step's length |
| random ×2 | at the effects step's length |

This is E8's decision split, with the factor span in place of the decision vector.

Reading:
- **Choices follow the invariant part, not the in-span part:** the decision reads the
  rendering-invariant duration.
- **Choices follow the in-span part, and `U` moves them too:** the decision reads a
  code that the unit word and the number move.
- **Both move choices:** the duration is read from both.

**Stage A, the answer slot (forward passes, cheap).** E8's steer test with the
directions above:
- thinking off: suffix newline L16 and pre-label L22/L24;
- thinking on: pre-label L22/L24;
- the crossed bank's natural renderings;
- α = ±1, ±2.

Thinking on is judged on rendering averages, per the P2 decision. From E8's logs
(8B, 176 records, five directions): about 1.5 min per thinking-off cell and 18 min
per thinking-on cell. Stage A is about 1 h with the extra directions.

**Stage B, the suffix with thinking on (regenerates the reasoning, expensive).**
Add the step at the suffix newline, then regenerate the trace and the label, as E10's
patch-before-reasoning does.
- Cell: on suffix L16. It is E10's layer, and in the prototype regime is clean there
  (P4 0.55) but leaks from L22.
- Prompts: 96 bases, 2 per duration (natural renderings, all regimes, so crossings
  both ways).
- Conditions: invariant, in-span, `U` and one random direction, at α = ±1. That is
  8 conditions; the unsteered baseline is E0's own traces.
- Cost: E10 took about 1,600 s per condition for 96 prompts (8B, 3,072-token budget),
  so about 3.6 h on one GPU, under 1 h on four with the queue.

**No gate (Jared, 2026-10-29).** As written, P1 ≤ 20% would block steering at
every cell (`17` §7), and the three-direction comparison asks which part of the
code the decision reads, which does not need a fully unit-free manifold. P1–P4 on
the pooled fit are reported with the steering results, not used to select cells.

## 6. Tasks

1. Step 0: parse the written unit from the units-bank traces (tokenizer only), with tests.
2. The fit (`fit_duration_effects` in `spar_horizon/steering.py`, returning a
   `fit_cell`-shaped dict), with tests on synthetic data where duration, unit and
   regime effects are planted and must be recovered separately.
3. `analysis/pristine_manifold.py`: cross-fit, P1–P4 for both curves, figure.
   Results in a short `17_results_pristine.md`.
4. With your approval of those results: the crossed-bank spec (§4). Done:
   generator and tests; regimes rotated rather than drawn (§4).
5. Debias P1 systematic and fix its CI (bootstrap by duration). Done; prototype rerun.
6. `local_checks.py crossbank`: E0 smoke on the 0.8B (Jared runs it).
7. `pristine_manifold.py` (and `unit_overlap.load_run`) read the crossed run, not only
   the units run. Steering inputs: the effects curve and its invariant / in-span split, `U`, in
   `steering_prepare.py`; the new directions in `exp8_steering.py` (stage A); a
   suffix steer-and-regenerate driver from E10's patch code (stage B). Each with tests
   and a `local_checks.py` smoke. Done (steering part, 2026-10-30): `steering_prepare.py
   --pooled [--stage-b]`, `exp8_steering.py` (pooled cells), `exp13_suffix_steer.py`,
   `tests/test_pristine_steering.py`, `local_checks.py pooledbank / stagea / stageb`,
   `queue_pristine_steer.sh`. Steered: natural renderings with delays in Alan's range.
8. Fill bank and anchors (`--bank fill`), delay covariates in `fit_duration_effects`,
   pooled loading and the anchor check in `pristine_manifold.py`.
9. Box: E0 fill (both modes) → anchor check → `pristine_manifold.py` pooled → stage A → stage B.
