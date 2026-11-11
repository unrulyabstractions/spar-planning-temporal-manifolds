# Results: overnight multichoice banks and phrase-steering scan (2026-11-06)

Spec `26_spec_overnight_choice_scan.md` (approved with the §6 amendments). Qwen3-8B, thinking on, one
RTX 4090 (vast.ai, $0.392/hr). **Part A is complete. Part B is a partial screen: 34 of 40 cells.**
Everything here is descriptive; Part B is a screen with 10 prompts per cell and makes no significance claim.

Outputs (`SPAR/results/horizon/`): `exp0_Qwen3-8B_think-on_choice{4,8}/` (`answers.json`, `manifest.json`,
`chunks/`, `choice_structure/`), `exp15_Qwen3-8B_scan/` (`conditions/`, `scan_summary.{txt,json}`,
`fit_report.txt`), `exp14_Qwen3-8B_phrase/states_L10-22-26.*`.

## 1. Part A: 4- and 8-choice banks (implicit ROI)

Both banks: 246 prompts, 86 bases (43 of 48 durations kept; 30-71 s, 108 s and 3 min dropped), 123 near and
123 both. All 246 answers parsed in each. The truth rule is the best payout that arrives within the horizon:
in `near` it is always an option in the near half, in `both` always in the far half.

| | choice4 | choice8 | full bank (2 options) |
|---|---|---|---|
| matches truth, near | 107/123 (87%) | 107/123 (87%) | |
| matches truth, both | 95/123 (77%) | 97/123 (79%) | |
| think block force-closed | 105/246 (43%) | 74/246 (30%) | 103/500 (21%) |
| median think length (tokens) | 2,639 | 2,351 | 1,748 |

- **Errors sit in the forced closes.** Natural closes: choice4 near 92/93, both 45/48; choice8 near 101/103,
  both 63/69. Forced closes: choice4 near 15/30, both 50/75; choice8 near 6/20, both 34/54. Forced closes are
  the minority of prompts and most of the errors.
- **Which kind of error.** In `near` the model picks a near-half option in 108/123 (choice4) and 111/123
  (choice8), and nearly every near-half pick is the truth, so near-regime errors are mostly choosing the far
  half. In `both` it picks the far half in 112/123 and 119/123, but only 95 and 97 are the truth, so both-regime
  errors are mostly the wrong option inside the right half (17 of 28 errors in choice4, 22 of 26 in choice8).
- **Position bias.** choice4 pick rate by position 0.18 / 0.21 / 0.29 / 0.32 against truth rate
  0.26 / 0.27 / 0.24 / 0.22. Accuracy given the truth's position: 0.63 / 0.75 / 0.98 / 0.96, so the model
  misses the truth mostly when it is early. choice8 pick rate 0.07 to 0.17, accuracy by truth position
  0.52 / 0.79 / 0.72 / 0.94 / 0.92 / 0.91 / 1.00 / 0.89.

### Activation structure (reduced set, grouped CV by base, `choice_structure/results.txt`)

Layers 10/14/16/18/22/24/26/30; scores are balanced accuracy or out-of-fold R². Chance: 0.50 for the half,
0.25 (k=4) or 0.125 (k=8) for position. Figures: `probes_{prompt_main,position,options}.png`.

- **Which option is chosen is not readable before the reasoning ends.** `chosen_pos` at the suffix, phrase and
  think@0.25/0.5 is at chance in both banks. It appears at `think@1` only weakly (0.36-0.43 in choice4, 0.24-0.30
  in choice8), rises at `</think>` (0.59 / 0.39 at L22), and is high at the pre-label token from L22 (choice4
  0.90 at L22-24, 0.94 at L26; choice8 0.66 at L22, 0.85 at L24, 0.95-0.97 at L26-30). Below L18 it is 0.24-0.58.
- **On natural closes the choice is already at `think@1`**: `chosen_pos` 0.96 (choice4, L22) and 0.74 (choice8,
  L22); on forced closes `think@1` is 0.35 and 0.17. So whether a trace has finished the comparison decides
  whether the last think token carries the answer.
- **The near/far half chosen is mostly the regime.** `half_chosen` at the suffix is 0.70-0.74 and `regime`
  0.80-0.82 (choice4), and the half chosen follows the regime in 88-90% of `near` and 90-97% of `both`
  prompts. I did not test half-chosen with the regime held fixed, so the two are not separated here.
- **Horizon.** `log_horizon` R² is 0.93-0.95 at the suffix newline and 0.98-0.99 at the unit token, and only
  0.68-0.78 at the number tokens. The unit token is the best place to read the horizon, which is the unit word
  (as in E14, `22`). The PC1 correlation with log horizon at the suffix peaks at L10 (0.91) and L16 (0.83) for
  choice4 and L16 (0.81) for choice8, then falls. The unit's share of variance at the suffix is 0.07-0.29, and
  0.31-0.57 at the unit token.
- **Option positions** (prompt-side, before any reasoning):
  - The option's line-end `.` token carries its delay (`log_delay` R² 0.95 / 0.92 at L10, falling to 0.83 / 0.80
    at L22) and its half (0.89-0.93 / 0.92-0.94), and partly its return (R² 0.41-0.56 / 0.47-0.63). The label
    token `a)` carries almost none of this (return R² about 0, delay 0.58-0.69).
  - `rank_in_half` is weakly readable at the line end in choice8 (R² up to 0.32) and not in choice4 (up to 0.12).
  - `is_chosen` at the line end is 0.59-0.65 (choice4) and 0.57-0.74 (choice8), against 0.50 at the label. That
    is above chance before reasoning, but an option's half and delay predict whether it is chosen, so this is not
    shown to be a value signal.

Not yet done on Part A: the explicit-ROI banks (`choice4_roi`, `choice8_roi`), and no causal test on any of the above.

## 2. Part B: steering scan (34 of 40 cells)

Setup as in spec §2: the same 10 prompts in every cell (5 none→near, 5 both→near), the manifold step to the
per-prompt target against a norm-matched random vector, thinking on, steer applied on every prefill pass. A flip
is a label different from the unsteered baseline; excess = cell flips minus the mean of the two no-op reruns.

- **Noise:** both no-op reruns flip 2 of 10 (1 toward near each), so the floor is about 2.
- **Reported cells (excess toward-target ≥ 4): none.** The largest excess toward flips are +3: L10 unit
  manifold (5 flips: 4 toward, 1 away; near log-odds shift +6.04, the largest in the scan) and L10 number
  random (5 flips, 3 toward, 2 away).
- **Random controls match or beat the manifold step.** Over the 13 regenerating non-answer cells present in both
  directions (L18, L14, L22 and L10 number/unit/suffix, and L26 number), the manifold cells average 2.5 flips and the random
  cells 2.9. L10 unit random, the twin of the top manifold cell, has 3 flips (excess +1).
- **Answer-slot cells are all 0 flips** at L18, L14, L22 and L10 (steps of 1-16 residual-norm units, baseline
  re-score of cached reasoning). Their "excess" of −2 only reflects the baseline re-score being exact while the
  no-op reruns have batch noise, so they are not comparable with the regenerating cells.
- **In-span caveat** (`fit_report.txt`): only 50-85% of each step lies in directions that track unit, regime,
  order or delay. At the unit token it is about 0.5 at every layer, so even a hit there would be largely a
  unit/regime/delay push rather than a pure horizon step. Phrase-token steps are 22-118 long, suffix and answer
  steps 1-16.

Reading: at 10 prompts per cell this screen catches only large effects, and none appeared at L18, L14, L22 or
L10. It does not support a layer or position to follow up. The one cell worth a look (L10 unit manifold) has an
equal-sized random twin one cell away and is the position where the manifold step is least pure.

### What did not run

- **L26: six cells** (unit manifold and random, suffix manifold and random, answer slot manifold and random).
  L26 number-token cells ran (2 and 4 flips, neither toward-target beyond noise). The L26 unit manifold cell was
  killed unfinished.
- **All four extras** (explicit-ROI banks, decision-direction/CAA, alpha sweep, second seed): less than $2 was
  left to the $6 line.

## 3. How the run went and what it cost

- Three rentals before a working box: one stuck in `loading` for 13+ minutes, two failed with CUDA error 804
  (compat libs in `/usr/local/cuda/compat`; moved aside on the third).
- Two unexplained host stops (about 10:50 and 13:00 UTC) with the GPU busy; each resume took 6-7 minutes and
  killed tmux, and the runs resumed from saved chunks and cells. Two CUDA out-of-memory crashes (choice8 forward
  pass, the scan on L18 unit random), fixed with expandable segments and, for choice8, batch 8.
- Time against the spec: Part A took longer than planned (choice4 2 h 10 min vs about 1.1 h) and Part B much longer
  (about 7.5 min per regenerating cell; the spec's 13 s per generation ignored that a batch of 10 decodes up to
  3,072 tokens). I stopped Part B at 37 units to fit the instance lifetime.
- Spend: about $4.5 by my sum of the `just burn` segments (it resets on each resume), under the $6 target and
  the $8 cap. Not verified against the account's billing.
- `just push` shipped `jared/` dotfiles to the rented host (destroyed with it); the push step should exclude them.
- Nothing is committed.
