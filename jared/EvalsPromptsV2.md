# Eval prompts v2: making the truth unambiguous (2026-11-17)

Follows `EvalsResults_2026-11-16.md`. That run showed the model rarely takes a payout that comes
after the horizon (`near` regime 11-12/12 in every two-option type), but often takes the sooner,
smaller payout when both pay out within the horizon (`both`). This doc traces those errors to the
template, proposes a clearer wording, and records the predictions made before the v2 run.

## 1. What the `both` errors are (local analysis of the overnight rows)

**The template leaves reinvestment open.** Alan's template asks for "the best investment" among
lines like "1,000 dollars in 1 month" and for "the greatest benefit for this time horizon". It never
says whether money received can be reinvested. The model nearly always assumes it can: compounding is
mentioned in 52/56 wrong `both` answers and 126/156 right ones. Our truth (`both` -> the far option)
holds only if it cannot, or if reinvestment earns less than the far option's implied yearly rate
`(v_far / v_near) ** (1 / gap) - 1`.

| implied yearly rate of the far option | `both` rows correct (overnight, 2-option types) |
|---|---|
| < 8% | 2/9 |
| 8-15% | 7/11 |
| 15-30% | 8/12 |
| 30-100% | 19/29 |
| > 100% | 69/92 |

17 overnight `both` items are lost to the near option reinvested at 10%/yr; the model is "right" on
5 of them. These items have no defensible truth under the original wording.

**Not all errors are disputable.** Above a 30%/yr implied rate the model still takes the near
option often, and more so at long horizons: correct on 32/35 with a horizon under a year, 53/64 at
1-10 years, 51/75 at 10 years or more. In the traces this is a heuristic, not arithmetic: "the
$1,000 now has 20 years to compound, so it beats $4,000 in a year".

**The horizon is read correctly in wrong answers.** A ridge probe for log horizon, trained on the
full bank (Qwen3-8B, thinking on) and applied to the eval types, has the same error on wrong and
right `both` rows (median 0.74 vs 0.75 decades at L18, end of prompt; p = 0.85). The decision fails
after the horizon is encoded.

**The main bank has the same problem.** In the full bank (Alan's replication, 500 prompts),
25 of 93 `both` items have an implied rate under 10%/yr. On the overnight Qwen3-8B think-on run of
that bank the model takes the far option in 50/93 `both` items (8/25 below 10%/yr, 27/44 at 25%/yr
or more) against 81/83 correct in `near`. The crossed bank is nearly clean (4/166 under 10%/yr).

## 2. Probe transfer (local, saved activations)

The full-bank probe reads the horizon well on 8 of 20 eval types at the end of the prompt (L18:
Spearman 0.91-0.97, error 0.2-0.6 decades); the two roi types rank well (0.91, 0.95) but sit 2 decades
low. It fails where the horizon is not near the end
of a single user turn: `system_prompt` 0.22, `multiturn_analyst` 0.36, `prose_natural` 0.45,
`json_io` biased by 8 decades. Adding the other eval types to the training set
(leave-one-type-out) barely helps them at the end of the prompt (0.51, 0.35, 0.65, 0.74), but at the
label slot after thinking (L22, prelabel) they transfer much better (0.72, 0.65, 0.67, 0.92). The
horizon direction at the end of the prompt is tied to the template; after thinking it is closer to
format-free. Script: `pipeline/s3_analysis/eval_types_probe.py`; table in the run directory's
`probe_transfer.md`.

## 3. The v2 wording

Three changes, each one sentence (`spar_horizon/eval_types/wording.py`):

| part | original | v2 |
|---|---|---|
| SITUATION | Plan for the future of the household. | ... Any money the household receives is kept as cash and earns no interest. |
| TASK | ... tasked to choose the best investment: | ... tasked to choose which payment the household receives: |
| CONSTRAINT | ... the greatest benefit for this time horizon: X. | ... leaves the household with the most money at the end of this time horizon: X. |

With all three the truth is the largest payout received within the horizon, with no assumption
about rates. The horizon still decides which options count, so the eval still tests horizon use.
The option lines are unchanged, so the tokens of each option are the same in both wordings.

## 4. The v2 run

Types (each pairs prompt by prompt with its counterpart, via `EvalType.item_seed`):

- `t0_w0`, `t0_w1_cash`, `t0_w3_clear`: T0 in the original wording, with CASH, and with all three
  changes; the same 40 items (12 near, 24 both, 4 none) and 8 swaps.
- `roi_unequal_clear`, `system_prompt_clear`, `sycophancy_lean_clear`, `horizon_revised_clear`: four
  overnight types with all three changes, on their overnight items; the overnight rows are the
  comparison.
- `free_form_anchor`: the free-form term with the horizon given by an anchor ("Your partner is 8
  weeks pregnant"), so the term cannot copy the horizon string. 21 prompts, 7 anchors.

Analysis: `pipeline/s3_analysis/eval_types_paired.py` (paired accuracy, exact McNemar on discordant
pairs) and `eval_types_score.py`.

### Predictions (written before the run)

1. CASH raises `both` accuracy (`t0_w0` -> `t0_w1_cash`, 28 paired rows) from about 0.7 to 0.9 or
   more, with `near` unchanged; the gain is largest below a 25%/yr implied rate.
2. All three changes (`t0_w3_clear`) do at least as well as CASH alone, with shorter thinking.
3. `roi_unequal_clear`: force-closed rows drop from 24/32 to under 12/32, and accuracy rises from 12/24.
4. `system_prompt_clear` `both` accuracy rises from 5/12; `sycophancy_lean_clear` follows the lean less
   than the overnight 5/24; `horizon_revised_clear` still follows the stale horizon about as often
   (6/24), since that error is about which horizon, not about reinvestment.
5. `free_form_anchor`: the term is within a factor 1.5 of the anchored horizon on at least 15/21.

## 5. Results

(Filled in after the run.)
