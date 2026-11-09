# Spec: ablate and steer one subspace, by layer, thinking on (E12, drafted 2026-10-27)

Status: **draft, awaiting approval.** Criteria in §5 to be fixed on approval, before any run.
8B and 14B, thinking on only (decision 2026-10-14), pre-label `:` only, cached reasoning.

## 1. Question

At 14B thinking-on L27, mean-ablating the horizon subspace without the decision
direction removes 0.33 of the near-vs-both contrast, while steering along the curve
without the decision moves 0.01 (`11` §3, `13` B2), read as "necessary ≠ steerable".
Two problems with that comparison:

1. **The two tests act on different subspaces.** Ablation uses B3 = top 4
   horizon-average PCs minus the decision. Steering uses the curve in the top 8
   minus the decision (`exp8_steering.py:143`, `steering.py:249`). Only 43% (8B L24)
   to 86% (14B L33) of the steering move lies in B3. This is the open item from the
   2026-10-26 audit (`01` §5.6).
2. **B3 carries the a/b label and reasoning length.** `pipeline/s1_data_vectors/analysis/off_curve_content.py`
   (post hoc; `results/horizon/analysis/off_curve_content.json`) gives the
   within-horizon r on near/both natural closes. Nulls are about 0.15; "a-vs-b" is
   shown with log think length partialled out.

   | | 8B L16 | 8B L20 | 8B L22 | 8B L24 | 14B L17 | 14B L21 | 14B L23 | 14B L25 | 14B L26 | 14B L27 |
   |---|---|---|---|---|---|---|---|---|---|---|
   | a-vs-b log-odds | −0.14 | 0.09 | 0.79 | 0.93 | 0.02 | 0.23 | 0.29 | 0.55 | 0.48 | **0.99** |
   | regime | 0.19 | 0.23 | 0.51 | 0.61 | 0.41 | 0.43 | 0.60 | 0.69 | 0.79 | 0.70 |
   | log far | 0.37 | 0.40 | 0.33 | 0.21 | 0.11 | 0.11 | 0.30 | 0.27 | 0.35 | 0.18 |
   | log think length (raw) | 0.99 | 0.98 | 0.95 | 0.86 | 0.98 | 0.97 | 0.97 | 0.94 | 0.92 | 0.85 |
   | T5 transfer at the last position | 0.05 | 0.09 | 0.22 | 0.56 | 0.00 | 0.01 | 0.04 | 0.15 | 0.42 | 0.53 |

   The T5 values come from `exp8_Qwen3-8B_answer_slot` and `exp8_Qwen3-14B_answer_slot_off`
   (the "on" key). 3 random directions orthogonal to the top 8 score ≤ 0.48 on every target.

Reading, to be tested here rather than assumed:
- **The 14B L27 ablation is probably the label.** The label fills B3 (0.99) from
  L27, the layer where 14B thinking on binds. `13` B2 already says this for L30/L33.
- **B3 tracks reasoning length everywhere** (0.85–0.99). That is most likely the
  position of the last token. A mean ablation moves it, and so does any steering
  along the curve, since think length varies with horizon.
- **The usable window is narrow.** The last position carries the choice only from
  T5 ≈ 0.15 (8B L22, 14B L25). By then the label is entering B3 (0.55–0.79). Any
  last-position edit at layer L is capped by T5 at L. Edits before 8B L22 or
  14B L25 can at most give small effects.
- The regime signal in B3 (14B 0.41–0.79 from L17) may be read off the reasoning
  text (`13` E1). It is not evidence of an internal comparison.

The questions:
- **Q1.** Once the decision, label and think-length directions are taken out, is the
  rest of the horizon subspace necessary for the choice?
- **Q2.** Is that same subspace steerable, along the curve or by swapping it in from
  a donor?
- **Q3.** Does an edit at a window layer change the answer through the decision
  and label coordinates at the binding layer, or around them?

## 2. Directions, per (model, layer), cross-fitted

The natural closes are split into 2 folds by prompt (seeded). Each tested prompt
uses directions fitted on the other fold, so no prompt is tested with directions
fitted on itself. Per fold, with `fit_cell` and `decision_direction` reused as is:

| name | definition |
|---|---|
| H8 | top 8 horizon-average PCs (current `basis`) |
| d | decision: ridge from H8 coordinates to the near log-odds (current) |
| ℓ | label: the same ridge, target the a-vs-b log-odds |
| τ | think length: the same ridge, target log think length |
| **C** | H8 minus span(d, ℓ, τ): the cleaned subspace, 5 dims |
| B3 | top 4 minus d (current; kept for continuity with `11` §3) |

C is the one subspace that both ablation and steering act on. The tests use the 176
cached near/both prompts, as in E8, and the results are also reported on natural
closes only.

## 3. Edits per layer, last position only

| test | edits | count |
|---|---|---|
| **A, ablate** | mean-ablate H8, C, B3, d, ℓ, τ; C on-curve part only; C off-curve part only; 3 random directions orthogonal to H8, added with the per-prompt norm of the C ablation's displacement | 11 |
| **S, steer** | curve delta projected into C at α ±1, ±2; random in C's complement, per-prompt norm-matched, α ±2 | 6 |
| **I, interchange** | C coordinates set to a donor's (other regime, near option under the same label, T5 pairing over all 176); the full residual from the same donor (T5, the ceiling) | 2 |
| **M, mediation** | interchange C at L, and at the binding layer (8B L25, 14B L27) reset the d and ℓ coordinates to the recipient's unedited values; window layers only | 1 |

Layers: 8B L18–L30, 14B L21–L33 (13 each). Metrics are as in E8: removed share of
the contrast for A, effect toward rational in contrast units for S, transfer toward
the donor for I and M, all with prompt-bootstrap 95% CIs.

## 4. Cost (box; 14B timings from `exp8_Qwen3-14B_on_L27_check`)

A prefix pass over 176 cached traces takes ~85 s on 14B. A one-token edit takes ~1 s.
Every edit sits at the last position, so **one prefix pass serves every layer and
every edit**. That needs one call to `score_last_edits` with all (layer, edit) pairs,
instead of one call per cell as now.

Per model and fold half, three passes: the baseline, recipient capture (for M) and
donor capture (for I). Then 13 layers × ~20 edits ≈ 260 one-token passes.
- 14B: ~4 min of prefix passes + ~5 min of edits ≈ 10 min.
- 8B: about half that.
- With 2 GPUs (one model each), ~15 min wall clock, inside one short rental.

## 5. Success criteria (fixed on approval)

The window means 8B L22–L24 and 14B L25–L26, where T5 ≥ 0.15 and binding has not
yet happened. A layer-level pass needs two adjacent layers, against the multiple
comparisons over 13 layers.

- **E12.1 The label confound (14B L27).** Ablating C removes ≤ half of what B3
  removes, and ℓ alone removes ≥ 0.2. Passing withdraws "necessary ≠ steerable" at
  14B L27 (`11` §5, `13` B2).
- **E12.2 Necessary beyond decision and label.** In the window, ablating C removes
  ≥ 0.10 with CI low > 0, and at least 0.05 more than the largest norm-matched random.
- **E12.3 Steerable in the same subspace.** In the window, the C interchange
  transfers ≥ 0.05 (CI low > 0) and ≥ a third of the full-residual T5 transfer at
  that layer, with random ≤ 0.02. Curve-in-C steering toward rational is reported
  next to it with the same threshold, ≥ 0.05 with CI low > 0.
- **E12.4 Around the decision.** Descriptive: the mediated fraction is
  1 − M / I at each window layer. "Not through the decision and label" is claimed
  only if M ≥ 0.05 with CI low > 0.
- **Descriptive, no verdict:** on-curve vs off-curve ablation, the τ ablation, and
  H8 vs C by layer.

If E12.1 passes and E12.2 fails, the conclusion is that at the answer slot the
horizon is used only as the decision and the label. Any horizon use beyond them
would then be earlier in the prompt or in the reasoning, which last-position edits
cannot reach (§7).

## 6. Tasks

1. `steering.py`:
   - a `basis_without` over several directions;
   - a `ProjectedReplace` edit (`x + ((donor - x) Cᵀ) C`);
   - a norm-matched random `Add`;
   - multi-layer specs in `score_last_edits`, for M.
   Tests: the projections are idempotent, and C ⊥ {d, ℓ, τ}. The shared-prefix path
   must equal `score` for two-layer edits.
2. `steering_prepare.py --sweep`: 2-fold cross-fitted ℓ, τ, C per layer, with the fold
   assignment saved.
3. `exp8_steering.py` test `sweep`: all layers × edits in one shared-prefix call;
   prints §5's verdicts.
4. A `local_checks.py` entry (0.8B plumbing, Jared runs it) and `queue_e12.sh`.
5. After the run: a results section and updates to `13` B2, `11` §3 and `01`.

## 7. Out of scope

- DAS-style learned subspaces: next, if E12.3 fails in a way that leaves room.
- Edits at prompt positions: thinking on, they change the reasoning (E10), and
  thinking off is closed (2026-10-14).
- Thinking off at all.
- Doc corrections from §1 before this runs. Until then the table above is post hoc.
