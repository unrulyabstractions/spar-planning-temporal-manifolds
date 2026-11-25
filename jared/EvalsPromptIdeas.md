# Eval prompt types: 40 ideas, cut to 20 (2026-11-16)

Follows `EvalsSpec.md` and `EvalsPromptTypes.md` (the first 20, with five candidates each). This doc
lists 40 ideas, scores them, and keeps 20. The 20 kept are all implemented in `spar_horizon/eval_types/`
and run overnight on Qwen3-8B with thinking on (results: `EvalsResults_2026-11-16.md`).

## How the cut was made

Each idea was scored 0-2 on four criteria, and the top 20 kept (ties broken by cost to run):

- **Q — answers a project question:** moves R1-R6 or the horizon-vs-decision question in `01_plan.md` §4.
- **T — clean truth:** a rational answer that a careful reader would not dispute.
- **R — runnable now:** a single conversation with an answer the runner can locate (no judge, no tools).
- **N — not redundant:** tests something no kept type already tests.

## The 40

| # | id | What it tests | Q | T | R | N | Kept |
|---|---|---|---|---|---|---|---|
| 1 | `roi_equal_10` | 4 options, equal value at the horizon at 10%: the horizon as a pure cut-off | 2 | 2 | 2 | 2 | yes |
| 2 | `roi_equal_5` | same at 5% (smaller far payouts) | 1 | 2 | 2 | 0 | no: a one-line twin of 1, run after 1 shows an effect |
| 3 | `roi_unequal` | 8 options, payouts not monotone in delay: largest payout inside the horizon | 2 | 2 | 2 | 2 | yes |
| 4 | `free_form` | the model names a deposit term: a continuous readout of the horizon it uses | 2 | 2 | 2 | 2 | yes |
| 5 | `implicit_concrete` | horizon inferred exactly from an anchor ("8 weeks pregnant") | 2 | 2 | 2 | 2 | yes |
| 6 | `implicit_vague` | horizon only a range ("before the wedding next spring") | 2 | 1 | 2 | 2 | yes |
| 7 | `json_io` | JSON in and out, same horizon string | 1 | 2 | 2 | 2 | yes |
| 8 | `system_prompt` | the constraint moved into the system message | 2 | 2 | 2 | 2 | yes |
| 9 | `role_fund_manager` | professional finance role, horizon unchanged | 1 | 2 | 2 | 2 | yes |
| 10 | `role_assistant` | advising a user instead of deciding | 1 | 2 | 2 | 2 | yes |
| 11 | `city_no_values` | delivery dates only, no magnitudes | 1 | 1 | 2 | 2 | yes |
| 12 | `city_values` | cost, opening date and users per day | 1 | 2 | 2 | 2 | yes |
| 13 | `multiturn_analyst` | horizon in turn 1, two neutral evaluation turns, decision in turn 4 | 2 | 2 | 2 | 2 | yes |
| 14 | `distractor_numeric` | irrelevant counts in the situation | 1 | 2 | 2 | 0 | no: 15 is the stronger distractor |
| 15 | `distractor_temporal` | an irrelevant duration at least a decade from the horizon | 2 | 2 | 2 | 2 | yes |
| 16 | `public_health` | domain change, countable neutral unit | 2 | 2 | 2 | 1 | yes |
| 17 | `conservation` | domain change, hectares restored | 2 | 2 | 2 | 0 | no: third domain change of the same form |
| 18 | `career` | personal domain, lump sums | 1 | 2 | 2 | 0 | no: dollars again, close to T0 |
| 19 | `coding_agent` | the model as an agent; short-horizon menu (days to 2 years) | 2 | 2 | 2 | 2 | yes |
| 20 | `loss_framing` | costs: the correct label flips relative to the regime | 2 | 2 | 2 | 2 | yes |
| 21 | `role_prior_conflict` | pension-fund vs day-trader roles, crossed with regime | 2 | 2 | 2 | 1 | no: needs 9 to show a role effect first |
| 22 | `system_user_conflict` | system says 20 years, user says 2 years | 1 | 0 | 2 | 2 | no: precedence is a convention, not a truth |
| 23 | `sycophancy_lean` | the user leans towards the wrong option | 2 | 2 | 2 | 2 | yes |
| 24 | `leading_analysis` | an earlier turn recommends the wrong option | 1 | 2 | 2 | 1 | no: close to 23, and to 13 with a lean |
| 25 | `stream_vs_lump` | salary raises (flows) vs lump sums | 1 | 1 | 2 | 2 | no: truth needs accumulation; separate factor |
| 26 | `json_number_unit` | horizon as `{"value": 2, "unit": "years"}` | 2 | 2 | 2 | 1 | no: good R3 follow-up once 7 parses |
| 27 | `risk_crossed` | the far option is safer or riskier | 1 | 0 | 2 | 2 | no: truth depends on risk preference |
| 28 | `horizon_position` | constraint before vs after the options | 1 | 2 | 2 | 1 | no: position effects already seen via E14 |
| 29 | `horizon_revised` | stale horizon, then "UPDATE: the horizon is now X" | 2 | 2 | 2 | 2 | yes |
| 30 | `date_arithmetic` | horizon from two stated dates | 1 | 2 | 2 | 0 | no: E18 family A |
| 31 | `unit_mismatch` | horizon "30 months", delays in years | 2 | 2 | 2 | 2 | yes |
| 32 | `number_words` | "two years" vs "2 years" | 2 | 2 | 2 | 1 | no: cheap R3 follow-up, unit/number split |
| 33 | `null_baseline` | no horizon at all, per domain | 1 | 0 | 2 | 1 | no: no truth; the full bank has nulls |
| 34 | `boundary` | horizon exactly equal to a delay | 1 | 0 | 2 | 2 | no: inclusive or not is ambiguous |
| 35 | `third_party` | deciding for a grandchild vs for oneself | 1 | 1 | 2 | 2 | no: who benefits is a different question |
| 36 | `translation` | the prompt in Spanish / Chinese | 2 | 2 | 2 | 2 | no: tokenization changes the positions; later |
| 37 | `agent_observation` | horizon arrives in a tool output | 2 | 2 | 0 | 2 | no: needs a tool-call template |
| 38 | `negated_horizon` | "outcomes after 2 years do not matter" | 1 | 2 | 2 | 1 | no: close to R1 paraphrases |
| 39 | `reinvest_beats_far` | stated rate makes near + reinvestment beat far inside the horizon | 2 | 2 | 2 | 1 | no: overlaps 1; run after 1 |
| 40 | `prose_natural` | first-person prose, no section markers | 2 | 2 | 2 | 2 | yes |

Kept (20): 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 23, 29, 31, 40.

## What each kept type adds over T0

| Question | Types |
|---|---|
| Is the horizon read as a cut-off, not a magnitude prior? | `roi_equal_10`, `roi_unequal`, `loss_framing` |
| Is the readout continuous? | `free_form` (term), `roi_equal_10` / `roi_unequal` (expected log delay from the label distribution) |
| Does it survive not being written as a duration? | `implicit_concrete`, `implicit_vague`, `unit_mismatch` |
| Does it survive placement and format? | `system_prompt`, `json_io`, `multiturn_analyst`, `prose_natural` |
| Does it survive pressure? | `distractor_temporal`, `sycophancy_lean`, `horizon_revised` |
| Does it survive a role or domain change (R2)? | `role_fund_manager`, `role_assistant`, `city_no_values`, `city_values`, `public_health`, `coding_agent` |

## Record layout (all types)

Per type 20 base items (8 near, 8 both, 4 none; the 4-/8-option types 12 / 4 / 4; `free_form` 20 terms),
8 option-order swaps, 4 explicit twins in Alan's investment template: 32 records (`free_form` 20), 628 in all.
`truth_labels` is a set; see `spar_horizon/eval_types/common.py`.

## Changes from `EvalsPromptTypes.md`

- `roi_equal_10` and `roi_unequal` draw delays per item from a 14-duration menu (1 week to 50 years),
  so payouts like "10,018 dollars in 1 week" appear.
- `free_form`'s CONSTRAINT says "select the term", and the deposit allows terms from 1 week.
- `prose_natural` has three horizon phrasings, chosen per item.
- `json_io` option values keep their label ("a": "a) 5,000 dollars in 6 months").
- `implicit_concrete` adds a passport anchor (valid 10 years, issued 7 years ago: 3 years).
- `coding_agent` horizons include hours so the `none` regime exists.
