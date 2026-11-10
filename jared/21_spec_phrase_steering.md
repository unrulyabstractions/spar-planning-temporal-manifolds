# Spec: E14, targeted horizon steering at the horizon phrase (thinking on)

Date: 2026-10-30, revised 2026-10-31. Decisions (Jared, 2026-10-30 and 2026-10-31):
- the four transitions: none → near and both → near, plus the reverses near → both
  and near → none;
- the target horizon lands at the log-midpoint of the new regime;
- steer along the pooled manifold fitted at the phrase tokens, not a donor patch
  (the donor patch stays as the ceiling condition);
- steer at L16 and at L14 and L18 (skip one layer each side). L16 carries every
  condition; L14 and L18 carry the effects step and the random control
  (2026-10-31);
- one box session: extraction, fit and the full run back to back, with no pilot
  and no go/no-go pause (2026-10-31).

8B, thinking on. Follows `16` §5 stage B, which steered at the suffix newline and
moved nothing.

## 1. Why

Three results point here:
- **Stage B (`exp13_Qwen3-8B_pristine_suffix`):** a ±1 decade step at the suffix
  newline (L16) left the reasoning naming the prompt's own horizon in 99-100% of
  traces. No direction beat random.
- **Behaviour (`analysis/behaviour_step.py`):** with thinking on, the model's choice
  is a threshold. Near log-odds drop by 17.5 (0.91 C) where the far delay falls
  inside the horizon, with a flat slope on either side. With thinking off it is
  graded.
  - *Correction (2026-11-01, after the run; no criterion changes).* The slopes are not
    flat. With the far delay outside the horizon, near log-odds fall by about 5 per
    decade of longer horizon. In the none regime they fall steeply (about 8 per
    decade) as the horizon shortens. Only the slope inside the far delay is flat.
    See `22` Appendix A.
- **E10 (`08` §11):** the reasoning reads the horizon back from the phrase tokens.
  Patching them at L12-16 made the trace write the donor's horizon, and the choice
  followed the trace.

So a thinking-on steer has to act where the reasoning reads the horizon (the
phrase), and has to move it far enough to cross a delay. Stage B did neither: a
fixed α = ±1 crossed a delay in 44 of 96 prompts.

## 2. What is steered

**Positions:** the horizon phrase tokens (`steering.phrase_positions`). Two roles:
- the number tokens ("59", "1.8");
- the unit token (" years"), the last token of the phrase.

The number of number tokens varies across renderings, so each role has its own
curve. The number tokens are fitted on their mean, and the step is added to each of
them.

**Layers:** L16, E10's layer, is the main layer. L14 and L18 (one layer skipped
each side) show whether the effect is specific to L16 or spread across the band.
Each layer is steered on its own, never two at once, with its own fit and its own
step. The steer is applied in the prefill only (reasoning and answer passes),
never at decode steps, as in E10.

**The manifold:** the `16` §2 pooled fit (duration effects; unit, regime, order as
factors; log near and far delays as covariates). It is fitted on the pooled
thinking-on prompts (full + units + fill, anchors dropped), per role and per layer. Forced closes
are not a factor here: the phrase comes before any reasoning. The phrase states are
causal in the prompt, so one forward pass over the prompts (no generation) gives
them for thinking on and off alike.

**The step, per prompt:** `curve(target) − curve(log h)` (`curve_delta`), with the
target chosen per prompt:

| transition | start regime | target log horizon | eligible (steerable) |
|---|---|---|---|
| none → near (up) | none | (log near + log far) / 2 | 426 |
| both → near (down) | both | (log near + log far) / 2 | 122 |
| near → both (up) | near | log far + 1 | 148 |
| near → none (down) | near | log near − 1 | 148 |

Counts from the 1,386 pooled prompts (2026-10-31). The earlier draft's 437/161/164/188
counted natural renderings only; `steerable` also requires delays in Alan's range
(Jared, 2026-10-30), which leaves 3 fill-bank prompts, so the draw is almost all full
and units bank. near → none draws from the 116 near prompts near → both leaves.

Eligibility rules:
- the target must lie inside the curve (30 s – 500 y);
- for the midpoint targets, log far − log near ≥ 0.6, so the target is at least
  0.3 decades from both delays.

The steps are a median of 1.6-3.5 decades, real horizon changes (computed
2026-10-30).

## 3. Prompts and conditions

**Prompts:** 32 per transition, 128 in all, natural renderings, drawn at random
(seed 0) from the eligible prompts. They are stratified by bank where a bank has
any eligible prompts. A near-regime prompt is used in one transition only.

**Conditions (per prompt unless noted):**

| condition | layers | edit at the phrase | what it controls |
|---|---|---|---|
| no-op rerun | none | none | batch noise. E10 found 12/96 labels change from batching alone |
| effects step | L14, L16, L18 | the full step to the target | the test |
| invariant part | L16 | the step minus its in-span part (`16` §5) | is the rendering-invariant part enough? |
| random | L14, L16, L18 | a random vector, as long as that layer's step (per role) | movability |
| opposite step (none and both prompts only) | L16 | the same number of decades in the other direction, which stays in the regime | a monotone push of the same size |
| donor patch | L16 | E10's full-phrase patch from a prompt whose horizon is the target (nearest available) | the ceiling: what a real horizon change does here |

Cost: 128 no-op + L16 (128 × 4 + 64) + L14 and L18 (2 × 2 × 128) = 1,216
prompt-generations, less the prompts without a same-length donor (about 24: near → both
finds a donor for about half its prompts, since a horizon past the far delay often
tokenizes longer), so about 1,190. At stage B's rate (about 13 s per prompt) that is
about 4.3 h on one 5090. Add a few minutes for the phrase extraction and the fit.

**Run order:** the driver runs one transition at a time, starting with both → near,
and within a transition the L16 effects step, random and donor patch first. A run
cut short still leaves complete prompts for the main test.

## 4. Measures

- **Stated horizon:**
  - the share of traces that write a duration in the target regime's interval,
    not counting mentions within 10% of a delay;
  - `names_duration` for the target ±10% and for the prompt's own horizon.
- **Choice:** the label, and the change in near log-odds from the no-op rerun. The
  predicted direction comes from the thinking-on behaviour map:
  none +0.1, near +12.2, both −7.0.
  - none → near: more near;
  - both → near: more near;
  - near → both: less near;
  - near → none: less near.
- **Side effects:**
  - the share of traces flagging the conflict ("typo", "wait, the horizon is");
  - forced closes;
  - traces that state no horizon.

## 5. Criteria (fixed before the run)

Judged on the L16 effects step. The invariant part is reported against the same
thresholds. At L14 and L18, H1 and H2 are reported against the same thresholds
(each layer against its own random control) but do not gate; H3 needs the
opposite step and is L16 only. CIs come from 2,000 prompt resamples.

- **H1, the reasoning reads the steer:** the share writing a target-regime horizon
  is ≥ 0.30 with CI above the random condition's, and random is ≤ 0.10.
- **H2, the choice crosses:** across all four transitions, the oriented change in
  near log-odds (sign from §4) is ≥ 0.15 C with CI above 0. The no-op rerun
  is the baseline.
- **H3, not a monotone push:** two checks.
  - On none and both prompts, effects step minus opposite step (oriented) is
    ≥ 0.15 C with CI above 0.
  - On near prompts, both reverses lower the near log-odds (CIs below 0). A
    monotone push gives them opposite signs.
- **Reported, not gating:**
  - the share of the donor-patch ceiling reached (H1 and H2 ratios);
  - per transition;
  - invariant part vs effects step;
  - the in-span share of the step per role, as at the answer slot (44/56);
  - the layer profile: H1 share and H2 log-odds change at L14, L16, L18.

Reading:
- **H1 + H2 + H3:** steering along the manifold sets the horizon the reasoning
  uses, and the choice follows by crossing the delays. This is the causal claim
  stage B could not make.
- **H1 without H2:** the trace reads the steered horizon but the decision does not
  follow. That is a decision made from the text, not the stated horizon; the
  traces will say which.
- **Neither, while the donor patch works:** the manifold does not contain what
  the phrase carries. At the unit token that may be the unit (E10). The invariant
  part, with the unit projected out, is the check.

## 6. Risks

- **At the unit token the duration code may be mostly unit.** E10's last-token
  patch swapped the unit and kept the number. Projecting the unit factor out
  there could leave a small invariant part. The extraction reports the in-span
  share per role before the run.
- **A step of several decades may push the state off the data manifold.** The
  "maybe a typo?" share and the forced-close rate show whether that happens. The
  donor patch is the on-manifold comparison.
- **The opposite step for none prompts clamps at 30 s** where log h − 30 s is less
  than the step. Those prompts are reported with the clamped length, and the
  contrast is rerun without them.

## 7. Tasks

1. **Extraction** (`experiments/exp14_phrase_states.py`): one prefill forward pass
   over the pooled thinking-on prompts. It saves L12, L14, L16 and L18 states at
   the number tokens (mean) and the unit token. Tests use the tiny random Qwen3 and the 0.8B
   tokenizer: positions, alignment with records.
2. **Fit** (`analysis/steering_prepare.py --phrase`): the pooled fit per role and per
   layer (L14, L16, L18); P1-P4
   and the in-span share, reported; then target selection (§2-3). It writes
   `steering/inputs_Qwen3-8B_phrase.{npz,json}`. Tests: targets, eligibility,
   stratified draw, opposite steps.
3. **Driver** (`experiments/exp14_phrase_steer.py`): stage B's driver with
   per-prompt targets, two-role phrase steps at the condition's layer, the opposite
   step and the donor patch (from E10), in the §3 run order. It saves traces, labels and near log-odds per condition.
   Tests on the tiny Qwen3.
4. **Analysis** (`analysis/exp14_analysis.py`): §4-5 with tests on planted data.
5. **`local_checks.py phrase`:** a 0.8B smoke (Jared runs it).
6. **Box, one session (about 4.5 h):** extraction, then the fit (CPU, on the box),
   then the full run, chained in one queue script with no pause. The fit's report
   (in-span share per role and layer) is written before the run starts, so it can
   be read while the run goes. Its partial results can be synced at any time.

## 8. Out of scope

- Thinking off: its choice is graded. Its test is the stage A rerun with about 20
  random directions; that is a separate addendum.
- 14B.
- Steering at layers other than L14, L16, L18, or at two layers at once. L12
  states are saved for a follow-up only.
