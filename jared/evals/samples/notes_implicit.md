# Notes: implicit-horizon samples (T2 two-option, T2 k-option, T3 open)

Records were generated from specs by a throwaway script (truth computed with `max_value_within_H`, checked at `horizon_lo_years`, `horizon_years`, `horizon_hi_years`). All truths are stable on their intervals.

## inv_t2_implicit_2opt
- Anchors: friday_monday, baby_8wk, toddler_school (vague), birthday_40, mortgage_30. Regimes: 3 `near`, 2 `both`; truth labels b, a, a, b, a. The risky option is the truth for 3 records and the non-truth for 2.
- Every option is a fixed-maturity product with a stated payout in USD. Far options are described as locked until their date, so "beyond H" means unavailable at H. Without that, an index fund could be sold early and the rule would not apply.
- In `both` records (birthday_40, mortgage_30) the horizon only matters if the model avoids the shorter option. A short-biased model fails; a long-biased one passes without using H. mortgage_30 cannot have a far option beyond H (it would need a >50-year product). Decide whether `both` belongs in the implicit bank or only `near`.
- The explicit twin keeps the context sentence ("my partner is 8 weeks pregnant", "she just turned 2") and swaps only the anchor phrase, so it is a clean A/B on the cue.
- `return_pct_annual` is null for h < 3 months. Returns for risky options use the stated expected value.

## inv_t2_implicit_k
- Anchors: year_end_oct (4 options), wedding_spring (8, vague), lease (4), university_8 (8), first_job (4, vague). Disjoint from family 1.
- The interval rule removes the natural duration-matched option: university_8 has no 10-year option (4 y then 25 y), lease has no 12-month one. This makes the bank fair under uncertainty but less realistic. Decide whether to accept it.
- Minimum margin to the interval is 2.0x for all but first_job (1.15x on the 55-year option; no realistic product is longer). first_job is vague, so its interval is wide (35-48 y) and the truth is still stable, but a reviewer may argue a 55-year equity fund could be sold at year 40. I describe it as locked.
- The truth is rarely the shortest option (ranks 2, 3, 2, 5, 3 by horizon), so the "pick the longest option" heuristic fails; ties in `kind` rank were avoided.
- Risk is crossed: the truth is medium risk in year_end_oct, wedding_spring, university_8 and first_job; low in lease.

## inv_t3_open
- Anchors: busy_season, baby_new, retire_60_65, kids_grown, newborn_adult. Readouts: three `equity_share` (as a 0-1 fraction), two `bond_duration_years`. `expected_direction` is `up` throughout. `truth_rule` is `none`, `options` is `[]`.
- Reference ranges are rules of thumb, not sourced facts (stated in each `notes`): cash/T-bills for money needed within about a year, duration matching, "100/110/120 minus age", target-date and age-based 529 glide paths. I recalled the glide-path figures and did not check them against current fund documents. Jared should decide whether to cite real documents or keep the main metric as the direction of change, not the range.
- retire_60_65 is confounded: retirement money is spent for decades afterwards, so "invest until I retire" is not a hard liability date. Consider rewording to "I need to withdraw all of it at 65".
- Readout extraction needs a parser (percentages to equity share, a stated duration in years); not part of this sample.

## Questions across families
- Do anchors such as "my partner just found out she is pregnant" or "starts school" carry cultural and knowledge assumptions (school start age 4-7, 18 vs 21 as adult, 40-week term)? I widened the intervals for the vague ones but the concrete ones assume the US/UK norm.
- Schema gap: no key for "option is locked / fixed maturity", though truth depends on it. I wrote it into option text. A `locked` boolean on options may be worth adding.
- Schema gap: `rendering` is `implied` for these records and the explicit twin lives in `explicit_twin_prompt`; the twin has no `truth` of its own, which is fine because it has the same options and H.
