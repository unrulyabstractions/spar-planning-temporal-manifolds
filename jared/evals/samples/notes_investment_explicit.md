# Notes: explicit-horizon investment families (5 samples each)

Numbers (payouts from stated rates, `return_pct_annual`, regime, `truth_label`) come from a generator script kept outside the repo; the JSONL was re-loaded and checked (schema keys, truth = max value with `h_years <= H`, labels, returns, 20% margin).

## inv_t1_2opt
- Horizons: 1 week, 18 months, 2, 8 and 30 years. Regimes: 2 `near`, 3 `both`. Correct label: a in 2, b in 3.
- Risk is crossed with horizon: in samples 1 and 4 the near option is safe and the far one riskier; in 5 the far option is the safe one; in 2 the far (4-day) option is safer; in 3 the far option is riskier. `truth` follows value only, so risk is a distractor for the model, never part of the label.
- In every `both` sample the far option also pays more, so `both` truth is always `far`. A `both` case where the short option pays more would need the 2-option truth to be `near` in `both`; I did not make one. Jared: want it?
- The 1-week sample uses days with a 365-day year; annualized returns there are huge (57-73%) because the payouts are small over days. Harmless for the label, but `return_pct_annual` is meaningless at that scale.
- Prose carries payouts in dollars; there is no stated rate, so `roi_condition` is `not_stated`.

## inv_t1_k_equal10 / inv_t1_k_equal5
- Used Alan's structured template (`surface`="structured"), with one added sentence in TASK stating the rate ("Every option grows your money at 10% per year, compounded, until it pays out"). Deviation from the template: the FORMAT line shows `<a), b), c) or d)>` (copied from `choice_prompt_text` behaviour of `shown`). Say so if you want the letters bare.
- Payout = round(P * 1.1^h) (1.05^h), computed per option; shown order shuffled so the correct label varies. 3 samples with 4 options, 2 with 8.
- Interpretation of "ROI equal for options before / after" in EvalsSpec: I read it as "every option, shorter or longer than H, has the same annualized return", so only duration separates payouts and the truth is simply the longest option within H. If you meant equal return only on the options before H (or after H), that is a different design and needs a decision.
- Rounding to whole dollars makes `return_pct_annual` 9.99 / 10.01 for 1-3 month options; the label is unaffected.
- equal5 sample 3 has H (40 y) beyond every option, so all four pay; this is the extreme of `k_within_n`. No sample has zero options within H (truth_label null): the prompt would then be unanswerable by the rule. Jared: include such items?
- Principal is stated in the prompt (needed to make the payout/rate story checkable) and `roi_explicit`-style: the rate sentence is the explicit-ROI variant.

## inv_t1_k_unequal
- No rate stated; per-option rates range 1.5%-40%, chosen so payouts differ. Samples 1, 3 and 4 have truth that is not the longest option within H (shorter beats longer): 1 (1y at 30% beats 3y and 6y), 3 (4y beats 7y at 1.5%), 4 (9-month beats 18-month). Samples 2 and 5 are standard (longest within H wins), with 5 containing a shorter-beats-longer pair that is not decisive (5y beats 10y but 20y wins).
- The model must compare payouts across options; the prompt gives only the principal and the payouts, so "unequal return" is visible only by arithmetic.
- Rates in the generator are not in the prompt; `return_pct_annual` is recomputed from the rounded payouts.

## inv_role_hedgefund
- Five system wordings (multi-strategy PM, senior finance professional at an asset manager, long/short equity PM, systematic quant PM, CFA strategist), roles `hedge_fund_pm` and `finance_professional`. Horizons: 18 months, 3 years, 7 years, 6 months, 10 years. Options: 2, 2, 4, 3, 2.
- Option counts of 3 and 4 use `k_within_n` regimes as the schema says; 2-option samples use `near`/`both`. In samples 1 and 4 only the shortest, lowest-payout option is within H, so larger, longer-dated lures are present.
- Dollar scale is millions (fund-like) so principals differ from the household families; the label depends only on durations and payouts.
- The "role" key is two values, not a single `hedge_fund_pm`, since the spec said "hedge-fund PM / finance professional"; analysis may want them grouped.

## inv_json
- User turn is one JSON object (situation, principal, options with `payout_usd` and `delay {value, unit}`, horizon, task, answer_format). A one-line preface asks for `{"choice": ..., "reasoning": ...}`, and the same shape is repeated inside the object. Drop one of the two if you prefer.
- Horizons: 9 months, 45 days, 5 years, 2 years, 12 years; delays in days, months and years. 2-4 options.
- `options[].text` holds the short description; payout/delay live in the structured fields only.
- Descriptions are bare ("Index fund") so the format carries the content; the model sees no prose risk language, but `risk` is still recorded in the label.

## Open questions
- Where, if anywhere, should risk enter the truth rule? Currently never.
- Do we need items with zero options within H (null truth) or `both` items where the near option pays more?
- Does "ROI equal for options before / after" mean all options equal (what I did) or equal only on one side of H?
