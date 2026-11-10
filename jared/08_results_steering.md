# Results: E8 patching, ablation and steering (Qwen3-8B, full bank)

Box run 2026-10-11 (4090), `experiments/exp8_steering.py --inputs
steering/inputs_Qwen3-8B.npz`; results in `results/horizon/exp8_Qwen3-8B_full/`.
Spec and fixed criteria: `07_spec_steering.md`. Wall time 81 min, of which the
thinking-on cells took 65.

The setup reproduces the original run: the near-vs-both contrast is 11.39
log-odds thinking off and 13.01 thinking on (saved run: 11.2 and 12.9), and 92
of 96 patching pairs differ by more than 1 log-odds (mean gap 11.6).

## 1. Where the horizon travels (T1, thinking off)

Transfer fraction when one position group is patched from a prompt with a
horizon in the other regime (1 = the whole choice shift moves with it):

| layers | horizon tokens | prompt after the horizon | `<|im_start|>` | `choose` | `:` (pre-label) | suffix newline |
|---|---|---|---|---|---|---|
| 1-7 | 1.00 -> 0.88 | 0.00 -> 0.11 | 0 | 0 | 0 | 0 |
| 8-14 | 0.79 -> 0.74 | 0.21 -> 0.26 | 0 | 0 | 0 | 0 |
| 15-19 | 0.66 -> 0.41 | 0.27 -> 0.50 | 0.04-0.08 | 0.0-0.01 | 0.0-0.01 | 0 |
| 20-23 | 0.32 -> 0.15 | 0.45 -> 0.26 | 0.13-0.23 | 0.05-0.14 | 0.03-0.24 | 0 |
| 24-35 | 0.00 | 0.15 -> 0.00 | 0 | 0.11-0.24 | 0.63 -> 0.89 | 0 |

The horizon stays on its own tokens through layer 14, spreads into the
prompt that follows it (peak 0.50 at L19), passes briefly through
`<|im_start|>` (0.2 at L20-23), and lands on the answer slot from L24, where
patching `:` alone carries 63-89% of the effect. Every other template token,
including the suffix newline, carries nothing at any layer.

## 2. The pre-registered cells (T2-T4): all null

| cell | ablation removes | random | line effect (near, +1 decade) | curve - line |
|---|---|---|---|---|
| off, suffix newline, L16 | -0.00 | 0.00 | 0.00 | 0.00 |
| off, pre-label, L21 | 0.01 | 0.00 | -0.02 | 0.02 [0.01, 0.02] |
| on, think close, L19 | -0.01 | 0.00 | 0.01 | 0.00 |
| on, pre-label, L21 | 0.06 | 0.00 | 0.01 | 0.02 [0.01, 0.02] |

All twelve verdicts fail (necessary, sufficient, curve beats line). The edits
are applied (effects are small but nonzero where the table says so) and large:
one decade along the curve is 2.9 units at the off pre-label cell, against a
typical spread of 4.95.

## 3. Reading

- **The patching map explains the nulls.** The cells were chosen where the
  horizon is most decodable, and those are not on the causal path. The suffix
  newline is decodable at R² 1.00 and carries 0.00 of the effect at every
  layer: a passenger. The pre-label token at L21 is decodable at 0.98 but
  carries 0.07: the decision reaches it only from L22-24.
- **Decodable is not used.** This is the central result so far: a perfect
  linear horizon code at a position says nothing about whether the choice is
  computed from it there.
- **Thinking on** was not patched (T1 is thinking off only), but its cells
  behave the same way, so the same late hand-off to the answer slot is the
  first hypothesis.

## 4. Next after run 1 (done: section 5)

1. Repeat T2-T3 where the map says the effect lives: the pre-label `:` at
   L24-L30 (transfer 0.63-0.82), thinking off and on. Directions can be fitted
   from the saved activations, which keep the pre-label token at every layer.
2. Patch the answer slot with thinking on (only the last position aligns across
   different reasoning traces), to see whether the same L24 hand-off holds.
3. The "prompt after the horizon" group (0.50 at L19) is the other carrier;
   its positions are not in the saved activations, so it can be patched but
   not yet probed.
4. Speed: compute logits at the last position only and use batch 8 for the
   cached reasoning; the thinking-on cells dominate the run time.

## 5. Second run: the answer slot, layers 24-30 (2026-10-12)

`exp8_Qwen3-8B_answer_slot/`, spec section 9; 95 min on a 4090. Contrast
11.39 off, 12.92 on (unchanged). Figures and the check below:
`results/horizon/analysis/steering_*.png`, `analysis/steering_analysis.py`.

**Answer-slot patching across prompts (T5)** reproduces T1 with thinking off
(0.58 at L24 and 0.88 at L34, against 0.63 and 0.88) and shows the same
hand-off with thinking on: near 0 before L16, a trickle of 0.05-0.09 at
L16-20, then 0.22 at L22, 0.56 at L24, 0.82 at L34. Reasoning does not move
where the decision is read from.

| cell | ablation removes, k = 1 / 2 / 4 / 8 | random | curve, toward rational, 2 decades | line | verdicts passed |
|---|---|---|---|---|---|
| off L24 | 0.00 / 0.33 / 0.37 / 0.46 | 0.00 | 0.39 | 0.01 | curve beats line |
| off L27 | 0.08 / 0.11 / 0.66 / 0.74 | 0.00 | 0.06 | 0.01 | none |
| off L30 | 0.21 / 0.21 / 0.80 / 0.83 | 0.00 | 0.04 | 0.01 | none (k = 2 judged) |
| on L24 | 0.19 / 0.27 / 0.30 / 0.47 | 0.00 | 0.15 | 0.00 | curve beats line |
| on L27 | 0.09 / 0.39 / 0.74 / 0.74 | 0.00 | 0.05 | 0.00 | necessary |
| on L30 | 0.67 / 0.73 / 0.83 / 0.83 | 0.00 | 0.04 | 0.00 | necessary |

The probe line does nothing at any cell (at most 0.01); the mean difference
is small and at thinking-on L24 goes the wrong way (-0.13).

## 6. Horizon or decision? (local check, 2026-10-12)

At the late answer slot the horizon averages differ partly because the
decision differs by horizon, so the subspace could be the decision itself.
From the saved activations and the a-vs-b label logits (5-fold CV R², top-4
horizon-average PCs, refit on each fold's training prompts; corrected
2026-10-26, the first version fitted the PCs on all prompts including the
test fold, which raised the thinking-on numbers by up to 0.16):

| cell | horizon from subspace | horizon from the decision alone | subspace adds beyond the decision | a-vs-b decision from subspace | horizon within the same choice (near / far) |
|---|---|---|---|---|---|
| off L24 | 0.83 | 0.00 | +0.83 | 0.39 | 0.78 / 0.86 |
| off L27 | 0.80 | 0.00 | +0.83 | 0.95 | 0.77 / 0.86 |
| off L30 | 0.77 | 0.00 | +0.80 | 0.97 | 0.73 / 0.82 |
| on L24 | 0.73 | 0.06 | +0.70 | 0.38 | 0.60 / 0.73 |
| on L27 | 0.67 | 0.06 | +0.62 | 0.95 | 0.56 / 0.73 |
| on L30 | 0.62 | 0.06 | +0.58 | 0.95 | 0.50 / 0.68 |

- **The subspace carries real horizon information at every cell.** The
  decision itself predicts log horizon at 0.00-0.06 (it depends on the horizon
  relative to the delays, not on the horizon), yet the subspace predicts it at
  0.62-0.83, adds 0.58-0.83 beyond the decision, and still separates horizons
  among prompts that made the same choice (0.50-0.86).
- **At L27 and L30 the subspace also contains the a-vs-b decision** (0.95-0.97).
  Ablating it removes the decision by construction, so the two "necessary"
  passes there (thinking on) do not show that the horizon is necessary.
- **L24 is the cleanest cell.** The subspace is mostly horizon there (decision
  0.38-0.39), ablating it removes 30-37% of the contrast with random at 0, and
  moving along the curve by 2 decades shifts the choice 39% (off) and 15% (on)
  of the contrast toward the rational answer, where the straight probe line
  does nothing. This is the first evidence that the horizon representation is
  used, and that its curved geometry matters for the effect.

## 7. Review, 2026-10-13

From the per-direction, per-alpha table in `exp8_Qwen3-8B_answer_slot/results.json`:

- **The L24 subspace carries the abstract decision too.** §6 reads L24 as
  mostly horizon from the a-vs-b column (0.38-0.39, per-fold PCs), but the near-vs-far
  log-odds decode from it at 0.64 (off) and 0.65 (on). L24 looks like the
  point where a near-vs-far preference exists before it is bound to a label.
  The curve is fitted on horizon averages, so it can include that preference
  and the probe line does not: "curve beats line" is not yet "the horizon is used".
- **Thinking on, the curve effect is partly non-specific.** Near-regime
  prompts lose near preference whichever way the curve moves: -0.14 at -2
  and at +2 decades (L24), -0.16 / -0.07 (L27), -0.16 / -0.06 (L30). The
  `toward` metric counts only the +alpha side for them. A shorter horizon
  may land in the `none` regime, where lower near log-odds could be
  legitimate; comparing with real `none`-regime prompts settles it.
  Thinking off, L24 behaves as a horizon edit should: near -0.20 / -0.39 at
  +1 / +2, both +0.17 / +0.38 at -1 / -2.
- **The mean difference steers backwards thinking on** at L24 (+0.13 near at
  +1 decade, -0.20 at -2). Suspected cause: long horizons are force-closed
  more often, so long-minus-short partly reads "was cut off" (`06` §4).
- **Patching consistency:** in run 1's T1 the transfer fractions of all
  position groups sum to 0.92-1.00 at every layer, so the groups are close to
  additive and nothing is missing from the map.

## 8. Next

Thinking on only from 2026-10-14; the list is in `01_plan.md` §5.

## 9. Third run: horizon or decision at L24 (2026-10-14)

`exp8_Qwen3-8B_decision_L24/`, spec `07` §10; 37 min on a 4090. Thinking on,
pre-label L24, directions refit on the 371 natural closes. The decision
direction `u` (ridge from the 8-dim horizon subspace to the near log-odds)
predicts the near log-odds at R² 0.86. Effects in units of C = 12.9.

| | removed (ablation, k = 4) | toward rational, \|alpha\| = 2 |
|---|---|---|
| horizon subspace | 0.30 [0.26, 0.34] | curve 0.13 |
| without `u` (3 dims) / curve without `u` | 0.11 [0.07, 0.15] | 0.07 [0.05, 0.09] |
| `u` alone / decision part of the curve | 0.09 [0.08, 0.10] | 0.05 |
| random | 0.00 | 0.01 |

**Both criteria of `07` §10 pass.** The reading does not follow from them:

- **The ablation rows are not a split of the ablated subspace (audit,
  2026-10-26).** `u` is fitted in all 8 dims, ablation is judged on the top 4,
  and only 0.25 of `u`'s squared norm lies in the top 4. "Without `u`" removes
  the top 4 minus a direction that is not `u`; "`u` alone" ablates a direction
  that is 75% outside the top 4. The steering rows (k = 8) do split exactly.

- **Removing `u` does not remove the decision.** The subspace without `u`
  still predicts the near log-odds at R² 0.52 (horizon unchanged at 0.85):
  `u` is the ridge direction, not the whole decision content. "Beyond the
  decision" is beyond one direction of it.
- **The curve follows some regime transitions, not all, mostly through the
  horizon rather than the prompt's delays.** By transition (saved
  `results.json`), the curve moves near->none -0.17, near->both -0.14,
  both->near +0.11, and leaves near->near and both->both at 0.00-0.05. But it
  also moves both->none by +0.14, where real behavior moves +0.04 (with
  thinking on, `none`-regime prompts are answered like `both`: near log-odds
  -0.25 vs -0.78; `near` +12.2), so it does not follow the pattern of real
  behavior. The pushed decision direction instead moves every prompt the same
  way (+0.15 to +0.23 at -alpha). The curve edit depends only on (horizon,
  alpha), and so does the average decision at each horizon. Among prompts
  given the identical edit, the part of the effect that follows each prompt's
  own delays is 0.03 [0.00, 0.06] of the real regime-change effect, against
  0.02 [0.01, 0.04] for the pure decision push. This check was added after
  the criteria passed and cannot separate the two readings: the intervals
  overlap, and the relevant ceiling is the curve's overall 0.14, not 1.
- **The mean difference steers backwards because of the median split, not
  forced closes.** Refit on natural closes it still goes the wrong way. The
  short half of the horizons is entirely `none`-regime (answered like
  `both`), the long half holds every `near`-regime prompt, so long minus short
  points toward "near".

**Reading:** at the answer slot after reasoning, the horizon subspace is used,
and the curve's effect is consistent with the average decision attached to
each horizon; the tests here cannot tell that apart from a horizon value
being compared with this prompt's delays at L24 (see the two caveats above). That comparison happens before the answer slot
(during reasoning or earlier), which only steering before reasoning can test.

## 10. E9: unit rewrites at the suffix (2026-10-14)

`exp9_Qwen3-8B_units/`, spec `07` §11; 22 s of forward passes. 22 durations
in 2-4 units each, in 24 contexts; analysis `analysis/units_analysis.py`
(`results/horizon/analysis/units.*`). The originals match the saved E0
activations (cosine 0.9999-1.0000 at every layer).

- **R3 passes as fixed:** at the suffix newline, L16, the unit share of the
  probe readout's variance is 0.02 (curve readout 0.03), and at most 0.08 from
  L12 on at both suffix positions. The readout tracks the duration across
  7 decades with slope 0.90.
- **But the share hides shifts of a third of a decade.** Mean |shift| of a
  rewrite from its canonical rendering is 0.30 decades, twice the grid
  spacing (0.15) and about twice the probe's held-out error (RMSE 0.16): "1 century"
  -0.80 (read as about 16 years), "600 seconds" and "3600 seconds" +0.74,
  "7 days" -0.69 vs "1 week", "1 decade" -0.48, "720 minutes" +0.60. Neither
  the numeral nor the unit size alone explains the shifts (they are nearly
  collinear here; R² 0.10-0.39).
- Units the bank never uses (decades, century) are read at the right order
  of magnitude but short (-0.25 to -0.80).

**Reading:** before reasoning the representation is mostly duration, not
unit (R3), with unit- and numeral-dependent errors of about 0.3 decades. That
bounds the "within-unit" worry from `06` §5 at the suffix: it is real but
second order.

## 11. E10: patching the horizon before reasoning (2026-10-18)

`exp10_Qwen3-8B_reasoning/`, spec `07` §12; 2.7 h on 2 GPUs (with the L22
run, §12). 96 pairs from the crossing pairs (seed 0), thinking on; the patch
is applied in both prefill passes (reasoning and answer), never at decode
steps. Analysis `analysis/exp10_analysis.py` (`results/horizon/analysis/exp10.*`,
`figures/exp10_reasoning.png`). Transfer = (patched − recipient) /
(donor text − recipient) in near log-odds, on the 88 pairs with a gap above
1 log-odds.

**Pre-registered criteria:** plumbing passes (self patch changes 0 of 24
labels). Subspace read passes at L16 (subspace 0.54 vs full 0.59, random
0.08) and fails at L12 only on "random ≤ 0.1" (subspace 0.85, full 0.71,
random 0.20 [0.01, 0.41]). The random transfer at L12 comes from the 23
same-unit pairs, whose gaps are small (median 4 log-odds vs 17 for pairs
whose units differ), so a ratio of re-rolled reasoning is noisy there (0.57
on same-unit, 0.07 on different-unit pairs). Counted by labels, random L12
moves 6 of the 52 pairs the donor text flips, the subspace 29.

**The patch swaps the unit, not the horizon.** The last token of the phrase
is the unit word (" years"). Patching it, in full or only its top-8 horizon
subspace, makes the reasoning write the recipient's number with the donor's
unit in 70 of the 71 pairs whose units differ at L16 and 68 of 71 at L12
(random 2 of 71). A hybrid can only show where the units differ; 25 of the 96
pairs share a unit. Corrected 2026-10-26: this section first gave 71-73%,
which divided by all 96 pairs.

| condition | writes recipient, units differ / same | writes donor | hybrid, units differ (of 71) | follows donor (label flips, units differ) | same unit | changes label where donor text does not |
|---|---|---|---|---|---|---|
| donor text | 0.01 / 0.00 | 0.98 | 5 | 46/46 | 6/6 | 0/44 |
| full phrase L16 / L12 | 0.03 / 0.00 | 0.99 | 4, 4 | 41/46, 41/46 | 2/6, 3/6 | 5/44, 8/44 |
| full last L16 / L12 | 0.14, 0.03 / 1.00 | 0.05 | 70, 68 | 28/46, 30/46 | 0/6, 0/6 | 7/44, 6/44 |
| subspace last L16 / L12 | 0.17, 0.04 / 1.00 | 0.05 | 70, 68 | 30/46, 29/46 | 0/6, 0/6 | 4/44, 5/44 |
| random last L16 / L12 | 0.97 / 1.00 | 0.03 | 2, 2 | 8/46, 5/46 | 0/6, 1/6 | 7/44, 9/44 |

Batch noise (the check `07` §12 asked for): rerunning the unpatched prompts
in E10's batches changes 12 of the 96 labels against the E0 cache, so the
"changes label" column has a floor of about 0.1 from batching alone.

Example (`s0_000005`, "59 years" with donor "10 days", full last L16): "needs
to choose between two investment options over a 59-day time horizon".
The choice then follows the horizon the trace wrote. Where the units match,
patching the unit token moves nothing (0 of 6 flips): the number is not
carried there.

Side effects: at L16 the model notices the conflict with the prompt text
("Wait, the time horizon is 7 months? ... maybe a typo?" in 22-30% of
traces vs 1% unpatched; 5-8% at L12), and more traces hit the 3,072-token
cap (0.35-0.40 vs 0.26).

**Reading:**
- The top-8 horizon subspace at the unit token, L12-16, is causally the
  unit: swapping 8 dims does what swapping the whole state does (hybrid
  70 vs 70 of 71 at L16, flips 30 vs 28). A random 8-dim swap does nothing,
  but it is a much smaller edit than the horizon-subspace swap (not
  size-matched), so it is a weak control.
- Reasoning uses it by reading it back as text: the trace writes a horizon
  and the choice follows that horizon. This is a horizon representation, not
  a decision one: the model reports it as the horizon, before any reasoning.
- It says nothing about magnitude within a unit (the number), which is where
  the "unit word alone gives R² 0.96" worry (`01` §4) sits. The test is the
  same patch on the number tokens, or the subspace over the whole phrase, on
  same-unit pairs.

## 12. L22: horizon or decision one layer earlier (2026-10-18)

`exp8_Qwen3-8B_decision_L22/`, same tests as §9 at pre-label L22; 41 min.
All three criteria fail: ablating the top-4 horizon subspace removes 0.09
[0.07, 0.10] of C (random 0.00; L24: 0.30), without `u` 0.03, `u` alone 0.03;
the curve without `u` steers 0.02 (random 0.01). The mean difference steers
0.08-0.11 (readout ±5-7 decades) and the decision direction 0.07-0.09. The
horizon subspace at the answer slot carries little at L22 and what it
carries is not separable from the decision, consistent with the decision
arriving there at L22-24 (§5). (The run logs random controls only at k = 4,
the size the criteria use.)
