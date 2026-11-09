# Results: the fill bank, pooled with the old 8B runs (2026-10-29)

Run: E0 on the fill bank (`--bank fill`), Qwen3-8B, thinking off and on. Each mode has
425 prompts: 400 fill prompts and 25 anchors. It ran on one RTX 5090
(`logs/queue_fill.log`): thinking off took 12 min, thinking on 1 h 45 min.
- Spec: `16` §2 and §4.
- Questions from `17` §8 and `18` §9–§10.

Everything after the run used saved activations only; no model was run.

Outputs are in `results/horizon/analysis/`. No old-run output was overwritten:
- `pristine_manifold_pooled_fill.{json,txt,png}`: P1–P4 on old + fill (§2). The
  old-runs-only baseline is still `pristine_manifold_pooled.*`.
- `pristine_confounds_fill.{json,txt}`: confounds after the fill (§3). The old version
  is still `pristine_confounds.*`, with a spare copy at `pristine_confounds_oldruns.*`.
- One-off computations (§1, §3–§6): `anchor_readout_fill.json`, `fill_transfer.txt`,
  `delay_margin_fill.{json,txt}` and `behaviour_fill.txt`. Their scripts are in
  `results/horizon/analysis/fill_oneoffs/`, not in the repo.

**What was pre-registered (`16` §4):** the anchor gate, the pooled fit and its P1–P4
bars. Everything else here is post hoc, including the tolerance that turns the anchor
check into a per-cell gate (§1).

## 1. Summary

- **Anchors: they pass with thinking off and at the suffix with thinking on. They fail
  after the reasoning trace with thinking on.**
  - Thinking off: every anchor's activations match the old runs bit for bit.
  - Thinking on: greedy decoding reproduced only 4 of 25 traces, so the trace cells (think
    close, pre-label) differ by about one distance between two distinct prompts.
  - So the fill is pooled at every thinking-off cell and at the thinking-on suffix only.
    The thinking-on pre-label cells (E8's steering cells) are not pooled.
- **Regime both at on suffix L16 reads +0.23* → −0.01 [−0.18, +0.16], but the fill did
  not remove it. Two opposite effects cancel:**
  - the old prompts still show +0.21 [+0.05, +0.38] under the new fit;
  - in the fill, regime is almost a function of the delays at a fixed duration
    (regime both against log far delay: r = −0.89).

  The fill decouples regime from duration but couples it to the delay level, which is
  built into its design (delays at a margin from the horizon).
- **The surprise: curves fitted on the old runs do not read the fill.** On fill prompts
  in the shortest third (horizon up to about 2.5 days, delays in seconds to hours):
  - at suffix L16 they read the horizon about 2 decades (100×) too long;
  - at L24, 3.4–4.2 decades too long.

  This holds for the old curve and for effects ⟂, with thinking off and on (slope of
  readout on truth 0.57–0.64 at L16). The reverse works: a curve fitted on the fill
  reads the old prompts with slope 0.99–1.12. Delays far outside the old banks' range
  (1 week to 50 years) move the suffix code far more than `18`'s in-range
  −0.14 decades per decade suggested.
- **Pooled P1–P4 (k = 4, effects ⟂):**
  - Within-unit ordering on rendering averages holds or improves at on suffix L16
    (0.85 → 0.91).
  - Everything else gets worse or stays where it was:
    - per-prompt error: 64% → 96% at on L16, 66% → 117% at off L16;
    - P4 (regime) now fails at every suffix cell except on L27 (0.60–0.74, against 0.54–0.61);
    - P1 is unchanged at L16 (76–75%) and worse from L22 on.
  - No cell passes all four.
- **The clean regime-none reference past 1 year (`18` §3) can't be read from the fill.**
  - Above 1 year the only none prompts are fill prompts with delays of 2–1000 years.
  - Near and both prompts there read 0.7–1.3 decades shorter than the none prompts (L16
    and L24). That gap mixes regime with a 1–3 decade difference in delay level.
- **Behaviour with thinking on is near ceiling on the fill:**
  - near 98.5%, both 97.5%, against 54–61% for both on the old banks;
  - no option-order effect;
  - 92% of bases give the same choice in every rendering (80% on the units bank).
- **Behaviour with thinking off:** a strong preference for option b) when the delays are
  minutes to hours (P(a) = 0.12–0.18). The option-order gap reverses to −60 points.

## 2. Anchor check (pre-registered gate, `16` §4)

25 anchors per mode: 12 full-bank prompts and 13 units-bank renderings of 6 bases.
Activation gap = ‖fill − old‖, as a fraction of the median distance between two distinct
old (units-bank) prompts in the same cell.

| mode | same trace and label | same label | suffix newline L16–L27, median / max gap | think close L16–L27 | pre-label L16–L27 |
|---|---|---|---|---|---|
| off | 25/25 | 25/25 | 0 / 0 (bit-identical) | – | 0 / 0 |
| on | 4/25 | 24/25 | 0 / 0.27–0.31 | 0.74–0.76 / 1.69–1.80 | 0.42–0.73 / 0.87–1.41 |

- **Thinking off passes exactly.** Every off run used batch 12. The answer text after the
  label differs in 5 of 25, and the label never does.
- **Thinking on, suffix (before the trace): bf16 rounding.**
  - At L16, 14 of 25 anchors are bit-identical. The other 11 differ by 0.17–0.27 of the
    distance between two prompts (0.16–0.31 over L16–L27), about 1% of the activation
    norm.
  - The prompt tokens are identical, so the difference can only come from the batch:
    padding and batch-mates in the extraction pass.
  - In readout terms, fitted on the old runs, the fill-minus-old offset is −0.004
    [−0.016, +0.004] decades at L16 for the old curve and −0.003 [−0.015, +0.008] for
    effects ⟂. Mean |difference| is ≤ 0.017 decades.
  - At L22 and L24 the means stay within ±0.05. Single anchors jump up to 1.0–1.6
    decades there, because the nearest-point readout switches branches.
- **Thinking on, after the trace: fails.**
  - Greedy decoding in bf16 isn't batch-invariant: the traces diverge after 20–2,900
    characters. This happens even for units-bank anchors run at the same batch size (20),
    so the batch's contents matter, not only its size.
  - One anchor flips its label (s0_000206: a → b, now force-closed).
  - Readouts there differ by 0.4–1.8 decades per anchor (mean |difference|). The mean
    offset's CI includes 0 at every cell but is ±0.5–1 decade wide, so it can't rule out a
    bank shift.
- **Decision applied (post hoc):** a cell is pooled only if its median anchor gap is at
  most 0.1 (`ANCHOR_TOLERANCE`, `gated_jobs` in `pristine_manifold.py`, reused by
  `pristine_confounds.py`). The observed medians are 0 or at least 0.42, so any
  tolerance between those gives the same split.
  - Pooled: all 8 thinking-off cells and the 4 thinking-on suffix cells.
  - Not pooled: thinking-on think close and pre-label.
  - `pristine_confounds` fits on pre-label L22 on the old runs only, so that row repeats
    `18` §9.
- **The same applies to the old runs.** Their thinking-on traces are one draw from a
  batch-dependent process, so the old full and units runs (batch 12 and 20) never had
  matching traces either.

**Decision for Jared:** whether the trace cells may be pooled anyway. Each prompt's trace
is one draw in any case, and no offset is detectable, but the CIs are wide. As written,
the gate says no.

## 3. Pooled P1–P4 (`pristine_manifold_pooled_fill.txt`)

Old full + units + fill, anchors left out: 1,386 prompts with a horizon (986 old + 400
fill) per mode. Log near and far delays are covariates. k = 4; bars as in `17` §7.
Effects ⟂ unless a column says otherwise. "old → fill" means old runs only (`17` §8) →
with the fill.

| cell | P1 sys ⟂ | old-curve P1 sys | wuR² avg ⟂ | P2 per prompt ⟂ | P3 ⟂ | P4 ⟂ | old-curve P4 | ⟂ passes, old → fill |
|---|---|---|---|---|---|---|---|---|
| off suffix L16 | 78 → 75 [0, 153] | 190 → 216 | 0.85 → 0.77 | 66 → 117 | – | 0.59 → 0.65 | 0.57 → 0.83 | P2 P4 → P2 |
| off suffix L24 | 95 → 130 [19, 252] | 162 → 0* | 0.84 → 0.67 | 68 → 149 | – | 0.58 → 0.63 | 0.51 → 0.83 | P2 P4 → P2 |
| off pre-label L22 | 51 → 67 [0, 147] | 89 → 62 | 0.87 → 0.79 | 87 → 168 | – | 0.62 → 0.66 | 0.62 → 0.83 | P2 → P2 |
| off pre-label L24 | 122 → 57 [0, 105] | 262 → 362 | 0.83 → 0.78 | 122 → 197 | – | 0.54 → 0.58 | 0.68 → 0.77 | P2 P4 → P2 P4 |
| on suffix L16 | 77 → 76 [15, 123] | 175 → 227 | 0.85 → **0.91** | 64 → 96 | 0.57 → 0.56 | 0.54 → 0.60 | 0.57 → 0.86 | P2 P3 P4 → P2 P3 |
| on suffix L22 | 91 → 86 [40, 124] | 145 → 0* | 0.84 → 0.88 | 66 → 100 | 0.57 → 0.57 | 0.58 → 0.68 | 0.56 → 0.85 | P2 P3 P4 → P2 P3 |
| on suffix L24 | 99 → 110 [61, 166] | 170 → 0* | 0.82 → 0.84 | 68 → 132 | 0.58 → 0.55 | 0.59 → 0.69 | 0.52 → 0.84 | P2 P3 P4 → P2 P3 |
| on suffix L27 | 67 → 171 [78, 291] | 303 → 0* | 0.78 → 0.76 | 74 → 149 | 0.52 → 0.56 | 0.61 → 0.53 | 0.54 → 0.86 | P2 P3 → P2 P3 P4 |

The unit lookup's P2 is 271% (old runs: see `17` §8).

- **P2 (magnitude on rendering averages) still passes everywhere.** It is best at on
  suffix L16 (0.91, k = 8: 0.86).
- **Per prompt the pooled manifold reads worse:** median error about 100–200%, against
  64–122% before. That is still better than the unit lookup (271%). This is the fill
  prompts' delay content (§5), which the linear delay covariates only partly absorb.
- **P4 fails because regime is now readable within a duration (effects ⟂ 0.58–0.74, old
  curve 0.77–0.86, against 0.51–0.68).** In the fill, regime at a fixed duration means
  delays 1–3 decades apart, so P4 now partly measures the delays (§4). On suffix L16 sits
  just over the bar (0.602).
- **P1 is unchanged at L16 and worse from L22 on**, with wider CIs.
- **\* Old-curve P1 sys of 0 is an artefact.** Those CIs reach 176%. The fill's pairs have
  very noisy old-curve readouts (p90 above 1,800%), and the var / n correction of `17` §7
  then removes everything. Read the pair medians for the old curve instead (on L16:
  83%).
- **P3 (forced) is unaffected:** 0.55–0.57.

## 4. Confounds after the fill (`pristine_confounds_fill.txt`)

Same readouts and regression as `18` §9. Equivalent decades; * means the 95% CI (bases
resampled) excludes 0. Each cell shows the old runs → with the fill. On pre-label L22 the
fill is not pooled (§2), so that row is unchanged from `18` §9.

| cell | readout | near delay | far delay | order | unit size | regime near | regime both |
|---|---|---|---|---|---|---|---|
| on suffix L16 | old curve | −0.14* → −0.06* | +0.01 → +0.13* | +0.07* → +0.21* | −0.09* → −0.04 | +0.30* → −0.07 | +0.38* → −0.14 |
| | effects ⟂ | −0.03 → +0.09* | −0.03 → +0.11* | −0.05* → −0.00 | +0.01 → −0.02 | +0.09 → **+0.16*** | **+0.23* → −0.01 [−0.18, +0.16]** |
| on suffix L24 | old curve | −0.08* → +0.11* | −0.10* → +0.04 | +0.01 → −0.01 | −0.15* → +0.01 | +0.48* → +0.15 | +0.63* → +0.10 |
| | effects ⟂ | −0.09* → +0.14* | −0.06* → +0.18* | −0.03 → −0.01 | −0.07* → −0.03 | +0.06 → +0.16 | +0.12 → **+0.26*** |
| off suffix L16 | old curve | −0.12* → −0.06 | −0.01 → +0.16* | +0.08* → +0.20* | −0.06* → −0.03 | +0.27* → −0.01 | +0.34* → −0.08 |
| | effects ⟂ | −0.02 → +0.08* | −0.04* → +0.06 | −0.05* → −0.03 | +0.01 → −0.02 | +0.07 → +0.06 | +0.15 → −0.11 |

Calibration at the suffix is 0.93 for the old curve and 0.98–0.99 for effects ⟂ (it was
0.99–1.00).

**The pooled coefficients hide a split (post hoc, `fill_transfer.txt`).** The pooled
effects ⟂ readout was regressed within duration on the old prompts and on the fill prompts
separately:

| cell | old prompts: regime both | fill prompts: regime both, with delays | fill prompts: regime both, without delay columns | fill: corr(both, log far delay) within duration |
|---|---|---|---|---|
| on suffix L16 | +0.21 [+0.05, +0.38] | +0.17 [−0.26, +0.65] | −0.69 [−0.95, −0.48] | −0.89 |
| off suffix L16 | +0.08 [−0.11, +0.23] | −0.27 [−1.14, +0.48] | −0.77 [−1.20, −0.35] | −0.89 |
| on suffix L24 | +0.51 [+0.29, +0.73] | +0.05 [−0.47, +0.62] | −1.09 [−1.55, −0.67] | −0.89 |

- **The old prompts keep their regime-both residual** under the new fit (+0.21 at on
  L16, +0.51 at L24).
- **In the fill, regime and delay level can't be separated.** At a fixed duration a
  "both" prompt has delays decades below a "none" prompt. Its total contrast reads 0.7–1.1
  decades shorter. With the delays in the regression, the regime coefficient matches the
  old prompts' (+0.17) but its CI spans ±0.45.
- **The pooled −0.01 is the two cancelling,** together with the delay coefficients
  turning positive (+0.09 to +0.18 per decade). It is not evidence that the residual is
  gone.
- **Answer to the open question: not answered by this bank.** Regime is defined by the
  delays relative to the horizon. So at a fixed duration it can be varied only by moving
  the delays, and only a design with independent delays can separate a step at the
  crossing from a smooth delay slope (`18` §3's margin plot). The fill's margins do span the
  crossing at fixed horizons, so that margin plot on fill prompts, with a readout fitted on
  the fill, is the remaining test (not run).

## 5. The old curves do not read fill prompts (post hoc, `fill_transfer.txt`)

Curves were fitted on the old runs only, as in `17` §8 (k = 4), and read the fill prompts.
Mean signed error in decades by regime and duration third (cuts at 2.5 days and 7
months):

| cell | readout | slope | near, t1 | near, t2 | both, t1 | both, t2 | both, t3 | none, t3 |
|---|---|---|---|---|---|---|---|---|
| on suffix L16 | old curve | 0.64 | +2.10 | +0.97 | +2.00 | +0.82 | +1.03 | +0.55 |
| | effects ⟂ | 0.58 | +2.14 | +1.01 | +1.80 | +0.49 | +0.45 | +0.34 |
| off suffix L16 | effects ⟂ | 0.57 | +2.17 | +1.11 | +1.79 | +0.54 | +0.35 | +0.27 |
| on suffix L24 | effects ⟂ | 0.36 | +4.10 | +1.77 | +3.35 | +1.90 | +1.18 | +0.88 |

- **Fitted the other way (on the fill, reading the old prompts)** the slope is 0.99–1.12,
  with median |error| 0.5–1.1 decades.
- **It is not a rendering effect.** Natural and other renderings err alike (near, t1:
  +2.80 / +2.83 with a ridge probe fitted on the old runs).
- **The worst cases are horizons of 2–4 minutes with delays in seconds and minutes,**
  read 4 decades too long, for example "horizon 108 seconds; 66 seconds / 22 minutes".
- **Reading:** the old banks' delays sit in 1 week – 50 years at every horizon, so a
  readout fitted on them never had to ignore delay content. Delays in seconds to hours
  move the suffix code by decades, in the same direction as `18` §1's near-delay pull (a
  shorter delay reads a longer horizon) but 4–7× more than a linear extrapolation
  (−0.14/decade × 4 decades = 0.56) predicts.
- **For steering:** a curve fitted on the old runs is only valid for prompts with Alan's
  delay range. The pooled curve is valid more widely, at the cost of per-prompt precision
  (§3).

## 6. Delay margin with the fill (one-off, `delay_margin_fill.txt`)

Same probe, axis and bins as `delay_margin.py` (probe fitted on regime-none prompts),
with residuals split at 1 year.
- **Old runs only:** reproduces `18` §3 exactly (flat within ±0.1 at L16).
- **With the fill, above 1 year (10 durations, 58 none prompts):** near and both prompts
  read 0.6–1.2 decades shorter than none at L16, probe and axis alike. At L24 it is
  0.8–1.4 shorter. There is no step at the crossing:
  - about −1.1 across near margins from −1.75 to −0.75, and −0.7 at −0.25;
  - about −1 across far margins from −1.25 to +0.75.
- **With the fill, up to 1 year:** fill near and both prompts read +1.3 to +2.2 decades
  against the old none prompts. That is §5's transfer failure, not a regime effect.
- **So the clean-reference result is not extended past 1 year.** The fill's none prompts
  there have delays of decades to centuries, outside the old range, so regime and delay
  level move together.

## 7. Behaviour (`behaviour_fill.txt`; anchors and null prompts excluded)

Rational choice: near in regime near, far in regime both. CIs resample bases. The fill
has 400 prompts: 64 none (all in the longest third), 136 near, 200 both.

| mode | bank | near acc | both acc | none: chose near | order gap, chose near when near is first − second (near / both / none) | forced | median think |
|---|---|---|---|---|---|---|---|
| off | full | 85.5 [78, 93] | 77.4 [69, 86] | 64.1 | +31 / +16 / +44 | – | – |
| off | units | 84.4 [77, 92] | 61.3 [45, 77] | 68.9 | +29 / +71 / +31 | – | – |
| off | fill | 59.6 [48, 71] | 66.0 [54, 78] | 98.4 | **−60 / −62** / +3 | – | – |
| on | full | 97.6 [94, 100] | 53.8 [44, 63] | 53.2 | +5 / +20 / +15 | 21% | 1,799 |
| on | units | 97.9 [95, 100] | 61.3 [48, 74] | 57.4 | +5 / +25 / +5 | 26% | 1,917 |
| on | fill | 98.5 [96, 100] | **97.5 [95, 100]** | 85.9 | −0 / −3 / −13 | 39% | 2,370 |

- **Thinking on is near ceiling on the fill.**
  - "Both" accuracy is 97.5%, against 54–61% on the old banks. The fill's far delay sits
    at least 0.2 decades inside the horizon at every duration. The old "both" prompts are
    long durations with Alan's delays, where 8B often still takes the near option.
  - Regime-none prompts (long horizons, delays beyond them) force-close 89% of the time.
- **Thinking off: a position bias toward option b) appears on the fill.**
  - P(a) is 0.19–0.20 in near/both, against 0.58–0.84 on the old banks, and 0.12–0.18
    when the delays are minutes to hours. E0's log reports position bias −0.40.
  - Accuracy is near chance (60–66%) and the order gap reverses.
  - So `18` §1's "order drives 8B's thinking-off choice" holds with the opposite sign
    here: the bias depends on the content.
- **Label stability across renderings of a base:** on 92.2% of 128 bases (units bank
  80.0%); off 83.6% (84.8%).
- **Same choice off and on:** fill 67.5%, against 72.4% (full) and 77.7% (units).
- **Realism labels (side view, labels only):**
  - The fill is almost entirely outside Alan's range: `alan_range` is true for 12 of 400
    prompts, all "both", 100% correct both modes.
  - By `delay_scale` (near / both accuracy):
    - minutes-hours (n = 104 / 158): off 58% / 60%, on 98% / 97%;
    - days-weeks (18 / 38): off 44% / 87%, on 100% / 100%;
    - months-years (14 / 4): off 93% / 100%, on 100% / 100%.
  - The none prompts are all months-years.

  Thinking on doesn't care about realism. Thinking off is worst on the
  minutes-to-hours payouts, which is also where the b) bias is strongest.

## 8. What this means

- **The fill did what it could for P2.** Within-unit ordering at on suffix L16 is 0.91.
- **The fill did not settle regime,** and it can't by construction (§4).
- **The fill exposed a delay-content dependence at the suffix** that is much larger than
  anything measured within the old banks (§5). Before any steering with a pooled curve:
  - decide whether the steering prompts stay within Alan's delay range (then the old
    curve is fine on them);
  - or steer with the pooled curve and accept 96–117% per-prompt error at L16.
- **E8's thinking-on pre-label cells cannot use the fill** under the gate as written (§2).
- **Possible next step (not run):** the `18` §3 margin plot on fill prompts only, with a
  readout fitted on the fill, separating a step at the delay crossing from a smooth delay
  slope.
