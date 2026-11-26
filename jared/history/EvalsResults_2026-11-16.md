# Eval-type baselines on Qwen3-8B, thinking on (2026-11-16)

The 20 types kept in `EvalsPromptIdeas.md`, run once on Qwen3-8B with thinking on.
The question was whether each type is coherent (parses, is stable across option order, has a truth the
model can reach) and what its baseline accuracy is. Activations were saved but are not analysed here.

- Run: `results/horizon/evaltypes_Qwen3-8B_think-on/<type>/{rows.json, acts_*.npz}`, scored with
  `python -m pipeline.s3_analysis.eval_types_score <run dir>` (`summary.md`, `summary.json`).
- Setup: think budget 3,072 tokens, then the think block is force-closed and the answer generated;
  answer parsed after `</think>` from the type's prefix (`I choose:`, `"choice":`, `I choose a term of:`).
  Activations at layers 10/14/16/18/22/24/26/30 (suffix, think at 25/50/100%, pre-label, label, answer).
- Per type: 20 base items, 8 swaps (reversed option order), 4 explicit twins (Alan's T0 investment template,
  horizon stated). 628 records in all. Accuracy is over base + swap rows that have a truth (24 per type).
- Box: one single-GPU vast.ai box at $0.449/hr, batch 5, about 13-15 minutes per type. Tonight's spend:
  **$2.39** of the $3 cap (account total $8.99).

## Results

`acc natural` counts only answers given after a natural end of thinking. `twin` uses the T0 template, so
across types it measures how much the baseline itself varies.

| type | forced | acc | chance | acc natural | near | both | twin | order agree | both correct after swap |
|---|---|---|---|---|---|---|---|---|---|
| multiturn_analyst | 0/32 | 23/24 | 0.50 | 23/24 | 12/12 | 11/12 | 3/4 | 7/8 | 7/8 |
| json_io | 5/32 | 23/24 | 0.50 | 21/21 | 12/12 | 11/12 | 3/4 | 7/8 | 7/8 |
| city_values | 1/32 | 24/24 | 0.50 | 24/24 | 12/12 | 12/12 | 3/4 | 8/8 | 8/8 |
| city_no_values | 1/32 | 12/12 | 0.50 | 12/12 | 12/12 | (no truth) | 3/4 | 8/8 | 4/4 |
| implicit_concrete | 11/32 | 22/24 | 0.50 | 15/17 | 11/12 | 11/12 | 4/4 | 8/8 | 8/8 |
| loss_framing | 5/32 | 22/24 | 0.50 | 20/20 | 10/12 | 12/12 | 4/4 | 6/8 | 6/8 |
| distractor_temporal | 4/32 | 21/24 | 0.50 | 19/22 | 12/12 | 9/12 | 4/4 | 8/8 | 7/8 |
| implicit_vague | 5/32 | 21/24 | 0.50 | 21/23 | 11/12 | 10/12 | 4/4 | 6/8 | 6/8 |
| coding_agent | 1/32 | 21/24 | 0.50 | 21/23 | 11/12 | 10/12 | 4/4 | 7/8 | 7/8 |
| unit_mismatch | 16/32 | 21/24 | 0.50 | 9/10 | 11/12 | 10/12 | 2/4 | 5/8 | 5/8 |
| prose_natural | 6/32 | 21/24 | 0.50 | 20/22 | 12/12 | 9/12 | 3/4 | 7/8 | 6/8 |
| role_fund_manager | 11/32 | 21/24 | 0.50 | 15/17 | 12/12 | 9/12 | 2/4 | 7/8 | 7/8 |
| roi_equal_10 | 28/32 | 21/24 | 0.66 | 2/2 | 13/16 | 8/8 | 4/4 | 2/8 | 7/8 |
| public_health | 8/32 | 20/24 | 0.50 | 16/17 | 12/12 | 8/12 | 3/4 | 7/8 | 7/8 |
| sycophancy_lean | 5/32 | 19/24 | 0.50 | 17/22 | 12/12 | 7/12 | 3/4 | 7/8 | 5/8 |
| horizon_revised | 7/32 | 18/24 | 0.50 | 13/18 | 12/12 | 6/12 | 4/4 | 5/8 | 5/8 |
| role_assistant | 5/32 | 18/24 | 0.50 | 17/22 | 11/12 | 7/12 | 4/4 | 6/8 | 5/8 |
| system_prompt | 3/32 | 17/24 | 0.50 | 15/22 | 12/12 | 5/12 | 3/4 | 8/8 | 6/8 |
| roi_unequal | 24/32 | 12/24 | 0.125 | 5/6 | 7/16 | 5/8 | 3/4 | 3/8 | 3/8 |
| free_form | 3/20 | term = horizon in 20/20 | - | - | - | - | - | - | - |

For `roi_equal_10` and `roi_unequal`, "near" and "both" are the 4-/8-option regimes (12+4 and 4+4 rows).
Every row parsed (628/628).

## What the run shows

**1. The cut-off is robust; picking the larger payout inside the horizon is not.** In `near` (only
the near option pays out within the horizon) the model is right on 11-12 of 12 in every two-option type.
Almost all errors are in `both` (both pay out within the horizon; the larger, later one is right),
where the model picks the sooner, smaller payout. The T0 twins show the same split: 36/38 in `near`,
27/38 in `both`. Reading the traces, the model uses the horizon as a reinvestment window: "investing the
$1,000 now gives 20 years of compounding, more than $4,000 in a year invested for 19". Compounding is
mentioned in 52/56 wrong `both` answers, and in 126/156 right ones, so it is the model's usual frame and
the error is in the comparison, not in reading the horizon.

**2. Most per-type differences are within noise.** With 12 rows per regime and a T0 `both` baseline
near 0.71, a type needs about 5/12 or fewer in `both` to stand out, before correcting for 18 types.
Only `system_prompt` (5/12) reaches that. `horizon_revised` (6/12), `role_assistant` (7/12) and
`sycophancy_lean` (7/12) point the same way, but none of these is a result yet. They are the types to
run at larger n.

**3. Type-specific readouts behave as intended.**
- `loss_framing`: the near rate flips with the framing (near regime 2/12 near, both regime 12/12),
  and the near log-odds flip sign (-8.2 vs +14.8). The model reads the horizon as a cut-off on costs too.
- `horizon_revised`: follows the stale horizon on 6/24 rows.
- `sycophancy_lean`: follows a lean towards the wrong option on 5/24 rows; the mean `both` log-odds
  is +3.1 (towards near), against -6 to -14 in most other types.
- `implicit_concrete` / `implicit_vague` (22/24, 21/24): inferring the horizon from an anchor costs
  little. `implicit_concrete` thinks longer (median 1,920 tokens, 11/32 forced).
- `city_no_values` `both` (no truth): picks the later delivery on 8/12, and the earlier one in `none`
  on 4/4.
- The explicit horizon string appears in the think trace for 95-100% of explicit base rows in most types
  (`loss_framing` and `city_no_values` 15/20).

**4. Three types need changes before they are useful.**
- `free_form`: the named term equals the horizon on 20/20 (median log error 0). The model copies the
  horizon, so this measures copying, not a horizon judgment. To make it informative: offer discrete
  terms that do not include the horizon, or give a rate schedule where a shorter term is better.
- `roi_unequal` (8 options) and `roi_equal_10` (4 options): 24/32 and 28/32 force-closed at 3,072 think
  tokens. The model works out every option's value. Natural closes are mostly right (5/6, 2/2), forced
  closes much less so (7/18 for `roi_unequal`). These need a think budget of about 8k tokens, or the
  values given in the prompt. Order agreement on `roi_equal_10` (2/8) is low because the truth is a
  set; both picks are correct on 7/8 swap pairs. `roi_unequal`'s expected log delay tracks log horizon
  (Spearman 0.61, n=20); `roi_equal_10`'s does not (0.09), as expected when every inside option is
  equally good.
- `unit_mismatch`: 16/32 forced (median think 2,940) and twins 2/4. The model converts units at
  length. It is still 9/10 on natural closes, so a larger budget may fix it.

## Caveats

- n = 24 graded rows per type (12 per regime); the confidence interval on a regime cell is about ±0.25.
  This run checks coherence and gives baselines, not effect sizes.
- One greedy generation per prompt (batched bf16, so not exactly batch-invariant); no repeats.
- Forced closes: the answer after a cut-off trace is often positional, so `acc` mixes two answer modes;
  `acc natural` separates them.
- The queue was restarted with batch 5 after repeated out-of-memory step-downs from batch 12; generations
  already cached were reused, so no rows are duplicated.
- The first box rented (instance 60888185, a host in China) had about 50 kB/s egress and was destroyed
  before use. Adding its machine to `EXCLUDE_MACHINES` in `.env` would stop it being picked again.
- Activations are saved (`acts_*.npz`, one per prompt format) but not analysed.

## Next

1. Run `system_prompt`, `horizon_revised`, `role_assistant`, `sycophancy_lean` and the T0 twin at
   about 100 rows per regime to test finding 2.
2. Fix `free_form` and rerun `roi_unequal` / `unit_mismatch` with an 8k think budget.
3. Probe the saved activations: does the horizon probe trained on the main bank read the horizon on these
   types, and does it differ between right and wrong `both` answers? (Finding 1 says the horizon is
   read correctly in wrong answers, so it should not.)
