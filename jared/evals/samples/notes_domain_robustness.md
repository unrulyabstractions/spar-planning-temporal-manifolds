# Notes: city_unvalued, city_valued, multiturn_cba, distractor (hand samples, 5 each)

All records were generated and checked with pure Python (truth, regime, margins, schema keys). No model was run.

## city_unvalued
- `h_years` definition used throughout: time from the decision until the project's benefit starts arriving (gala 0.08, park 3, tree canopy 15). Duration of benefit would rank very differently (a gala's benefit is brief, a park's lasts decades), so Jared should confirm this choice, because it fixes the `E[log h]` readout.
- Mix: 001 and 004 have no horizon (`rendering` "null", `horizon_cue` "none"), 002 and 003 have an explicit horizon (3 and 10 years), 005 has a vague "next generation" horizon (`rendering` "vague", interval 15 to 30, a guess). Option counts are 2, 4, 2, 4, 2.
- `regime` is null even where a horizon is stated, since `truth_rule` is "none". If a regime is wanted for `E[log h]` analysis by horizon, it can be computed afterwards from `h_years`.
- Schema gap: no `anchor_id` fits a city-planning vague horizon, since the anchor table is personal-finance only.

## city_valued
- `value` = total public benefit in USD, stated in the prompt, `value_unit` "USD_total_benefit". Cost is in the prose only (no schema slot) and does not enter `max_value_within_H`. A net-benefit rule would change the answer in 003 and 005 (high-cost options), so Jared should decide whether truth should be benefit or net benefit.
- Regimes covered: near (001), k_within_2 (002, truth is the shortest option), both with far winning (003), both with near winning (004), k_within_3 (005, the biggest benefit overall is infeasible). No `none` regime: with no feasible option the rule has no answer. A "none feasible" case needs a decision on the expected label.
- 003 has the larger total benefit but the worse benefit/cost ratio, so a model using ROI would disagree with the rule. Intended as a probe, remove if undesired.
- Horizon wording varies (term, grant, bond, agreement, finance director). `horizon_lo` = `horizon_hi` = H and `truth_stable_on_interval` is true by construction.

## multiturn_cba
- Fixed format: markdown table (Option | Cost | Payoff | Time to payoff | Notes) plus a one-line "Summary: Option x, ..." line. The final user turn is "Thanks. Based on that analysis, pick one option for me." plus the answer instruction.
- Prior-turn correctness: wrong in 001, 003, 005 (all investment; the wrong pick is the highest payoff in 001, the shortest/safest in 003 and 005), right in 002 and 004 (city). Domain and correctness are therefore confounded; Jared may want to cross them in the full bank.
- The horizon appears only in the first user turn. The prior turns never mention it (verified by regex for "horizon" and "within"). The prior turn's notes and summary give a plausible but unreferenced justification.
- Investment payoffs are absolute USD at maturity from one principal; `return_pct_annual` is the computed CAGR. City `value` is total benefit.

## distractor
- The distractor is chosen so `distractor_truth_label` differs from `truth_label` in every record. Four are long distractors (truck 20 y, cat 9 y, 30 years in town) or mid (audit 2 y) and one is short (3-month trip = 0.25 y, which only a 6-week T-bill fits). Wording was chosen to be clearly unrelated to the decision.
- Placement (no schema key; recorded in `notes`): before the horizon in 001, 003, 004; after in 002, 005.
- Margins: no option within 20% of the real horizon or of the distractor value.
- Schema gap: no field for "distractor position"; consider `distractors[].position`.
- Label balance is skewed: `truth_label` is b in 3 of 5, `distractor_truth_label` is a in 4 of 5, because the distractor mostly flips toward the shorter, early-listed option. Reshuffle when building the full bank.
