# Results: the 8B map on Qwen3-14B, thinking off and on (E11)

Box run 2026-10-20/21 (2 GPUs), `queue_14b.sh`; spec and fixed criteria:
`10_spec_scale_14b.md`. Results in `results/horizon/exp{0,8}_Qwen3-14B_*`,
`coherence_Qwen3-14B_think-on_full/`; analysis `analysis/stage_markers.py`,
`analysis/cross_model_alignment.py` (outputs in `results/horizon/analysis/`).

**Run later (2026-10-24, `queue_14b_rest.sh`, shared-prefix scoring):** the
thinking-on cells at L30 and L33 (`exp8_Qwen3-14B_on_L30_L33/`) and the probe
line at all six cells (`exp8_Qwen3-14B_line_{off,on}/`), which closes S4.
Their numbers are in `13_evidence.md` B1b and B2, not in the tables below.

The setup reproduces: the near-vs-both contrast is 21.97 log-odds thinking off
and 25.31 on, about twice the 8B's (11.39 / 12.92). 14B force-closes 27 of 500
reasoning traces, against 103 for 8B (median think length 1,346 vs 1,749).

## 1. Stage order (S1, S5): pass

Markers from T1 (thinking off) and T5 (both modes), relative depth = layer /
n_layers (36 / 40); `frac` interpolates the crossing between layers.

| marker | 8B | 14B | 14B − 8B (frac depth) |
|---|---|---|---|
| `encode_end` (horizon tokens >= 0.5), off | L18 (0.51) | L18 (0.46) | −0.05 |
| `spread_peak` (prompt after horizon), off | L19 (0.53), peak 0.50 | L23 (0.58), peak 0.66 | +0.05 |
| `handoff` (answer slot >= 0.5), off | L24 (0.66) | L26 (0.64) | −0.02 |
| `bind` (a/b decodes at R² >= 0.9), off | L25 (0.69) | L27 (0.68) | −0.02 |
| `handoff`, on | L24 (0.66) | L27 (0.67) | +0.01 |
| `bind`, on | L25 (0.69) | L27 (0.67) | −0.02 |

`bind` off corrected 2026-10-26 (was L30, +0.06): with thinking off, 14B writes
`I choose: **b)**` in 53 of 500 prompts, and E0 stored the logits of `' a'` /
`' b'` there, not of the unspaced label the model writes next. Without those
rows a/b R² at L27 goes from 0.43 to 0.90. The `**` lead and the claim that
14B binds later than 8B are struck. `encode_end` and `spread_peak` may still
shift: T1's "horizon" group left out a shared unit word in 33 of 96 pairs
(fixed in code; needs a T1 rerun on the box).

- **S1 passes:** the order holds in both modes and every marker is within
  0.05 of its 8B depth.
- **S5 passes:** with thinking on the 14B hand-off is one layer later than off.
- **14B binds the label one layer after the hand-off with thinking off**, as
  8B does. With thinking on the hand-off and
  binding coincide: a/b R² jumps from 0.10 at L26 to 0.94 at L27, and the
  interpolated crossings are 26.73 and 26.95. The order holds by 0.22 layers.
  Ordered on whole layers it would be a tie and S1 would fail. Read this as
  "hand-off and binding in the same layer".
- **Two corrections to the spec's §2 reference values:** 8B `encode_end` is
  L18, not L17 (0.55 at L18). 8B `bind` is L25, not L27: per layer, the a/b
  R² is 0.34 at L24 and 0.95 at L25. `08` §6 only sampled L24/27/30. Neither
  correction changes a verdict.
- 14B spreads more of the effect into the prompt after the horizon (peak 0.66
  vs 0.50), and its hand-off is sharper (answer-slot transfer 0.11 at L24 →
  0.64 at L26). As on 8B, every template token except `choose` and `:`
  carries at most 0.11 at any layer.

## 2. Cross-model alignment (S2, S3): both fail as pre-registered

Linear CKA of the 48 horizon averages between every 8B and every 14B layer,
at the suffix newline and pre-label; thinking on averages both models over
the 353 prompts that closed naturally in both.

**Corrected 2026-10-26.** The first version's null permuted the rows (horizon
averages) of one model, so it compared the two models on different prompts.
The models see identical prompts, and shared prompt content alone makes
horizon averages alike (any grouping of the same prompts does). The null is
now a label shuffle: horizon labels shuffled over prompts (within unit for
the within-unit CKA), the same shuffle in both models, averages recomputed.
Against it the suffix match is not significant, and S3 fails at every cell.
The readout R² is also below a unit-word lookup on the same 48 averages
(0.957), so S3's 0.8 threshold was weaker than a lookup. The thinking-on cells
first used each model's own natural closes (371 vs 445); the layer pick now
refits the PCs per fold, which moved suffix-on from L19 to L20.

| matched cell (8B / 14B) | CKA (shuffle p95, p) | readout R² (unit lookup) | within-unit CKA (shuffle p95, p) | row-permutation p95, CKA / within | rank ρ 8B / 14B | S3 |
|---|---|---|---|---|---|---|
| suffix newline off (L20 / L20) | 0.88 (0.91, 0.28) | 0.94 (0.96) | 0.77 (0.78, 0.12) | 0.12 / 0.19 | 0.98 / 0.98 | fail |
| suffix newline on (L20 / L20) | 0.89 (0.92, 0.22) | 0.95 (0.96) | 0.76 (0.79, 0.19) | 0.12 / 0.20 | 0.98 / 0.98 | fail |
| pre-label off (L24 / L26) | 0.45 (0.30, 0.001) | 0.22 (0.96) | 0.17 (0.24, 0.27) | 0.12 / 0.11 | 0.58 / 0.20 | fail |
| pre-label on (L24 / L27) | 0.75 (0.56, 0.001) | 0.78 (0.96) | 0.44 (0.52, 0.26) | 0.15 / 0.19 | 0.51 / 0.60 | fail |

So the suffix CKA of 0.88 and its within-unit 0.77 are what the shared prompts
give on their own; they are not evidence that the two models share a horizon
code beyond the prompt. At pre-label the overall CKA is above the label-shuffle
null (p = 0.001), but its within-unit part is not.

S2 fails in all four position × mode combinations (Spearman of the best-match
layer −0.24 to 0.79; mean depth gap 0.15–0.34).

**Diagnostic, added after the run (not pre-registered).** For each 8B layer:
the row maximum, the CKA at the same relative depth, and how many 14B layers
come within 0.02 of the maximum (`results/horizon/analysis/cross_model_alignment.json`):

| | 8B L4 | L8 | L12 | L16 | L20 | L24 | L28 | L35 |
|---|---|---|---|---|---|---|---|---|
| suffix off: max (at 14B) | .92 (L4) | .85 (L8) | .93 (L16) | .92 (L21) | .88 (L21) | .89 (L20) | .90 (L20) | .88 (L20) |
| suffix off: same depth | .92 | .84 | .84 | .90 | .86 | **.59** | .74 | .78 |
| pre-label off: max | .33 | .35 | .24 | .31 | .39 | .47 (L29) | .41 | .26 |
| pre-label off: same depth | .12 | .14 | .09 | .11 | .15 | **.46** | .40 | .18 |
| pre-label off: # within .02 | 9 | 11 | 11 | 11 | 9 | 8 | 8 | 11 |

- **At the suffix, the two models match layer for layer up to about 0.55 of
  depth** (CKA 0.84–0.92 on the diagonal). After that the 8B code stays like
  its mid-depth code, while 14B's moves away from its own: every late 8B
  layer is closest to 14B L20–22, and the same-depth 14B layer reaches only
  0.59–0.78. This is a real difference, not a flat-maximum artefact (at most
  3 layers within 0.02).
- **At pre-label before the hand-off there is little to align:** CKA is
  0.1–0.4 everywhere and flat (8–11 layers within 0.02 of the maximum). The
  argmax there is noise, which is what fails S2's monotonicity. **At the
  hand-off the depths do correspond:** for 8B L24 and L28, the same-depth
  14B layer is within 0.01 of the row maximum, thinking off and on.
- **Whether the horizon code is shared across models is not shown.** At the
  suffix, a rotation maps 14B onto 8B well enough that the 8B probe reads
  the held-out 14B horizons at R² 0.94, but a unit lookup alone gives 0.957,
  and CKA and within-unit CKA are at the level the shared prompts give under
  the label shuffle (struck 2026-10-26: "more than a shared unit code"). At
  the answer slot the overall CKA (0.45 off, 0.75 on) is above that null,
  the within-unit part is not, and horizon is not the first aligned
  coordinate. The layer-by-layer depth pattern in the diagnostic table above
  does not depend on the null.

## 3. Causal tests at the answer slot (E8, decision split)

Pre-label, effects in units of the contrast; k = 2 off / 4 on for ablation;
steering "toward rational" at |α| = 2. 8B reference: thinking on L24 (`08` §9).

| cell | ablate horizon subspace | without decision dir. | decision dir. alone | curve | curve without decision | decision part | random |
|---|---|---|---|---|---|---|---|
| 14B off L27 | 0.15 | −0.00 | 0.11 | 0.15 | 0.05 | 0.10 | 0.01 |
| 14B off L30 | 0.01 (k=4: 0.61) | 0.01 | 0.01 | 0.02 | 0.00 | 0.00 | 0.00 |
| 14B off L33 | 0.04 (k=4: 0.72) | 0.03 | 0.00 | 0.02 | 0.01 | 0.00 | 0.00 |
| 14B on L27 | 0.45 | **0.33** (low 0.31) | 0.07 | 0.04 | 0.01 | 0.03 | 0.01 |
| 8B on L24 | 0.30 | 0.11 | 0.09 | 0.13 | 0.07 | 0.05 | 0.01 |

- **Caveat on the three ablation columns (audit, 2026-10-26).** The decision
  direction is fitted in the 8-dim horizon basis, but ablation is judged on
  the top k (2 off, 4 on). Only part of it lies in that top k: 0.23 at off
  L27, 0.59 / 0.53 at off L30 / L33, 0.61–0.64 on, 0.25 at 8B L24. So
  "without decision dir." removes the top k minus a direction that is not the
  decision direction, and "decision dir. alone" ablates mostly directions
  outside the top k. The columns are not a split of the ablated subspace; the
  pre-registered verdicts stand, the readings below that rest on the split do
  not. Redoing it needs the direction projected into the top k (box).
- **At the 14B hand-off, thinking off, steering along the decision direction
  moves the choice.** (The ablation split behind "the horizon subspace without
  it removes nothing" is not valid, see the caveat above.) Steering along it alone moves near prompts by 0.21–0.27 and both
  prompts by 0.23–0.31. The curve's effect (0.15) is two thirds its
  decision part.
- **Thinking on, the horizon subspace beyond the decision direction is
  necessary but not sufficient.** Ablating it removes 0.33 of the contrast
  (`ablate_beyond_decision` passes; 8B: 0.11; but the removed direction holds
  only 0.63 of the decision direction, so part of that 0.33 may still be
  decision), but steering along the curve
  without the decision moves 0.01 (8B: 0.07). The one sizeable steering
  effect is again non-specific: near prompts lose 0.13–0.14 of their
  preference at α = −2 along both curves, as on 8B (`08` §7).
- **At L30/L33 the a/b decision lives in the 3rd–4th components** (ablating
  k = 2 removes ≤ 0.04, k = 4 removes 0.61–0.72), and no steering direction
  does anything there. This matches 8B's L27/L30: once the label is bound,
  ablating the subspace removes the decision by construction.
- **The mean difference steers backwards**, as on 8B (+0.15 to +0.20 near
  for a longer horizon at off L27): the median split again puts every
  `near`-regime prompt in the long half (`08` §9).

## 4. Behavior and coherence

| | temporal reasoning | order stability | label stability | position bias |
|---|---|---|---|---|
| 8B off (500) | 71/83 (86%) | 276/500 (55%) | 420/500 (84%) | +0.34 |
| 14B off (500) | 79/83 (95%) | 371/500 (74%) | 326/500 (65%) | +0.08 |
| 14B on (80, `coherence.py`) | 16/16 | 65/80 (81%) | 65/80 (81%) | −0.09 (base, all 500) |

14B is more order-stable and less position-biased than 8B, but less
label-stable thinking off. 8B has no thinking-on coherence numbers yet; the
same script runs on it. The 14B thinking-on subset is the first 314 cached
prompts, sampled evenly (regime mix 46 none / 16 near / 16 both / 2 null,
close to the full bank's).

## 5. Reading

- **The pipeline replicates by stage:** encoding on the horizon tokens,
  spreading into the following prompt, hand-off to the answer slot at
  0.64–0.67 of depth in both models and modes, binding to the label right
  after. Prediction 1 of the working hypothesis (fixed relative depth) holds
  for the hand-off at two sizes.
- ~~The early horizon code is the same object across models~~ (struck
  2026-10-26, §2): the suffix alignment does not beat a null that keeps the
  shared prompts, so this run cannot say whether the models share a horizon
  code. A test needs prompts that differ between the models with the
  horizons held fixed, or a null built from within-horizon prompt variation.
- **Steering the horizon at the answer slot gets weaker with scale.** On
  14B the curve's effect is mostly its decision part (thinking off) or
  nearly nothing (thinking on). Yet with thinking on, ablating the horizon
  subspace beyond the decision removes a third of the contrast (with the
  §3 caveat: that ablation does not fully remove the decision direction). Removing
  the subspace breaks the choice; pushing along it does not steer the
  choice. That is the 8B "decodable ≠ used" result one step further:
  "necessary ≠ steerable".

## 6. Next

1. ~~`queue_14b_rest.sh`~~ done 2026-10-24 (`on_L27_check` reproduced §3's
   row within 0.002; L30/L33 thinking on and the probe line: `13` B1b, B2).
2. Thinking-on coherence for 8B (`coherence.py --model Qwen/Qwen3-8B`).
3. The decision direction steers well at 14B off L27 (0.2–0.3). Check
   whether it is the same direction as 8B's at L24–25, via the cross-model
   map at the hand-off.
