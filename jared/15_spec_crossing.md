# Spec: E12, is curve steering horizon-specific? (delay crossing)

Date: 2026-10-27. Draft for Jared's approval. Criteria in §3 and §4 are fixed
before any confirmation cell is split.

**Status 2026-10-27: on hold behind `16_spec_pristine_manifold.md`.** Every
curve here comes from full-bank horizon averages, where the unit word (0.97 of
log horizon), the regime and forced closes move with the horizon. Part A stays
a cheap check of the existing runs. Part B runs with the `16` manifold in place
of that curve.

## 1. Question and why now

Curve steering at the 8B answer slot (L24, thinking on) moves the choice 0.13 C,
but two readings stay open (`01_plan.md` §4): the curve carries the horizon,
which the model compares with the delays, or it carries the average decision per
horizon. A horizon reading makes one prediction a decision push does not: **an
edit should change the choice only when the target horizon crosses a delay**,
by about what the model's own behavior does across that delay.

The behavior map (mean near log-odds by regime, E0, natural closes for thinking
on) gives the predicted sign of each crossing:

| | none | near | both |
|---|---|---|---|
| 8B on | −0.60 | 13.28 | −0.60 |
| 14B on | 1.33 | 20.97 | −4.69 |
| 14B off | 4.27 | 15.93 | −9.22 |

On 8B, `none` sits with `both`, so near→none should move toward far. That is
what the "non-specific" −0.14 at α = −2 in §4 was. On 14B, `none` sits between
the other two regimes.

**Discovery (already seen, 8B on L24, `exp8_Qwen3-8B_decision_L24`).** Split by
crossing, in units of C. The curve moves prompts that cross (−0.13 to −0.18;
+0.08 to +0.14) and barely moves prompts that stay (−0.04 to +0.01). The matched
decision push moves both groups equally (≈0.15–0.30). One part does not fit:
both→none (+0.14) equals both→near, where the behavior map predicts about 0.
This cell is reported, not used to confirm.

## 2. Definitions (both parts)

- **Effect** `e`: the saved per-prompt change in near log-odds, in units of C
  (`steer[direction][alpha]["effect"]`).
- **Target regime**: the regime of `log h + α`, clamped to the curve's range as
  `curve_delta` clamps it, against the prompt's own delays. **Crossers** change
  regime; **stayers** don't.
- **Predicted sign** `s`: the sign of (behavior-map mean of the target regime − mean of the
  start regime) for the crossers in a stratum. Oriented effect: `s · e`.
- **Stratum**: (start regime, α). A stratum is used when its crossers all reach
  one target regime and crossers and stayers each number ≥ 10.
  - 8B L24: near@±1, near@±2, both@−1. Not both@−2: only 5 stayers.
- **Bootstrap**: 2,000 resamples of prompts, all strata and directions resampled
  together, 95% percentile CI.

## 3. Part A: local, saved E8 results (no box)

**Statistic.** The crossing gap
`G = Σ_strata w · (mean s·e over crossers − mean s·e over stayers)`, with
`w ∝ n_cross · n_stay / (n_cross + n_stay)`, normalized to sum to 1. Computed
per direction: curve, curve without decision, decision (matched in length to
the curve).

**Criteria, per cell:**

- **A1, crossing matters:** G(curve) ≥ 0.05 C with CI above 0.
- **A2, not just movability:** G(curve) − G(decision) ≥ 0.05 C with CI above 0, paired bootstrap.
  The decision push has no horizon content, so any gap it shows comes from
  crossers being easier to move.
- **A3, reported, not gating:**
  - a prompt-clustered regression of `s·e` on crossing, |baseline near LO|, step
    length and the forced flag: the crossing coefficient and its CI;
  - the same G on natural closes only.
- **A4, diagnostic, not gating:** "Compare with both delays" or "shorter means nearer"? At
  α = −2 on both-regime prompts, take e(→none) − e(→near). 8B predicts ≤ −0.05;
  a CI covering 0 leaves the monotone reading open. Part B decides it.

**Cells.**
- **Confirmation:** the untouched cells whose published curve effect at |α| = 2
  is ≥ 0.05 C. That is 14B off L27 (0.148) and 14B on L33 (0.056).
- **Descriptive only:** 8B on L22 (0.044) and 14B on L27 and L30. These move too
  little for a null to mean anything.
- **14B off L30/L33:** listed in the table, no verdict.

**Reading.** Horizon-specific steering is **supported (8B and 14B)** if A1 and A2
pass at 14B off L27, and **strong** once Part B passes on 8B. If 14B off L27
fails A2, the curve effect is a decision push that happens to correlate with
crossing; that goes into `13` as *against*.

Output: `analysis/crossing_split.py` → `<results>/analysis/crossing_split.json`,
a table on stdout, and one figure: per cell, the oriented effect of crossers
against stayers for each direction.

## 4. Part B: box, the boundary ladder (8B on L24)

Part A uses whatever α the run had, so crossing is confounded with how far a
prompt sits from its delays. Part B sets the target directly. For each prompt
and each delay `d` (near, far), step to `log d + δ`, δ ∈ {−0.8, −0.3, +0.3, +0.8}
decades. Each prompt is its own control across δ.

- **Far ladder:** near prompts step up, both prompts step down, all to around the far delay. Keep
  a prompt only if `log far − 0.8 > log near`, so the low side stays in `near`.
- **Near ladder:** all prompts step down to around the near delay. Keep a prompt only if
  `log near + 0.8 < log far`, so the high side stays in `near`. This is the test
  A4 can't settle: on 8B, crossing below the near delay should *lower* the near
  log-odds (into `none`, at −0.60). "Shorter means nearer" predicts it rises.
- **Directions:** curve, curve without decision, decision (matched length).
  2 ladders × 4 δ × 3 = 24 conditions, about 26 min on the cached reasoning,
  the same as one E8 steer phase.
- **Statistics:**
  - the jump at the delay `J = mean[e(+0.3) − e(−0.3)]`;
  - the slope on each side `W+ = mean[e(+0.8) − e(+0.3)]` and `W− = mean[e(−0.3) − e(−0.8)]`.
  - All are paired within prompts and oriented by the predicted sign across that delay.

**Criteria, fixed now:**

- **B1, a jump at the far delay:** oriented J(curve) ≥ 0.05 C, CI above 0, and
  J − (W+ + W−)/2 above 0 (CI). A threshold, not a smooth trend.
- **B2, a jump at the near delay, in the behavior map's direction:** the same test
  at the near ladder. If B1 passes and B2 fails with the opposite sign, the
  reading is "shorter means nearer", not "compared with the delays".
- **B3, not movability:** J(curve) − J(decision) ≥ 0.05 C, CI above 0, at whichever
  ladders pass B1/B2.

**Optional, only if the box has a 48 GB card:** the same ladder at 14B off L27,
thinking off, a few minutes. Its behavior map puts `none` between the others,
which is a different prediction to test.

## 5. Code changes

- `spar_horizon/steering.py`: `ladder_alphas(log_h, log_delay, offset)`, the per-row α,
  and a mask for rows kept by the §4 rules. `curve_delta` and `matched_deltas`
  already take per-row α.
- `experiments/exp8_steering.py`: a `ladder` test. It needs per-prompt delays,
  which it reads from the records it already loads. It writes per-prompt
  `effect`, the kept mask and the step length per (ladder, δ, direction) into
  `results.json`.
- `analysis/crossing_split.py` (Part A) and a ladder section in it (Part B).
- Tests:
  - pure python: `ladder_alphas` and the kept mask;
  - target-regime and stratum building, on synthetic records with known answers;
  - the ladder test on the tiny random Qwen3, as for the existing steering tests.

## 6. Tasks

1. `analysis/crossing_split.py` and its tests. Run Part A on all cells, Part A verdicts in `16_results_crossing.md`.
2. `ladder_alphas`, the `ladder` test in `exp8_steering.py`, tests, a 0.8B
   smoke added to `local_checks.py` (Jared runs it).
3. Box: 8B on L24 ladder (Jared rents; I give the exact command).
4. Part B analysis, verdicts, and updates to `01_plan.md` §4/§5 and `13_evidence.md` B1.

## 7. Out of scope

Within-unit magnitude (the E10 same-unit follow-up, `01_plan.md` §5.4) and
steering before reasoning. Both are separate specs.
