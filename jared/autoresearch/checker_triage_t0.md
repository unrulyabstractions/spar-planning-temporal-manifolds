# Triage of Haiku checker flags on the T0 bank (seed 0)

Source: `prompt_bank/data/t0_s0/checks/anthropic__claude-haiku-5-5.jsonl`. 700 prompts, 274 flagged (39%), 460 flags. The checker agrees with the generator's answer on all 700. I also recomputed every regime from the prompt text with exact fractions: 0 mismatches, and in the verification split the near delay is always shorter than H. So no flag exposes a label bug. Theme counts come from keyword rules on the flag text, with samples read. They are approximate (about 5% of flags fit no theme).

## 1. Themes

| Cat | Theme | n | Example | Verdict |
|---|---|---|---|---|
| instructions | FORMAT/ACTION force "a) or b)" but no option pays off, so "none" cannot be answered (all in `none`) | 62 | x0005 "format ... does not clearly allow a 'none' response" | Design property, intended (README: none-regime is for extraction, never scored). Not a bug, but it is why `none` flags dominate. |
| instructions | `<a) or b)>` angle-bracket placeholder: write "a)" or "a"? Brackets literal? | 63 | x0138 "unclear whether to reproduce the label exactly" | Taste. It is the report's template. Cheap to ignore. |
| instructions | other ("Think deeply" vs short window, etc.) | 11 | x0033 | Noise |
| implausible | Short horizon (seconds to days) against years-scale payouts | 122 | x0075 "A 46-second horizon is wildly out of step" | Intended: the grid spans 30 s to 500 y by design. Taste. |
| implausible | Very long horizon (59-500 y) | 13 | x0101 "330-year planning horizon is unrealistic" | Intended, harmless. |
| ambiguity | "greatest benefit" could mean discounted or time-valued, not nominal payout | 57 | v080-near-first "'benefit' to mean present value or discounted value" | Real issue, phrasing 0 only (53 of 55). Score it, since it can change what a careful model optimises (see 2c). |
| ambiguity | Horizon read as time to deliberate, not payout horizon | 19 | x0002 "15-minute 'time frame' ... time the decision-maker has to deliberate" | Real for phrasing 3 ("Evaluate over a time frame of X") with short H (13 of 20); 7 in phrasing 0. Only matters in `none` prompts, where a model that reads it this way would answer differently. |
| ambiguity | "Plan for the future" implies a horizon (null prompts) / overrides a short H | 3 | x0259-null | Low. Only 2 of 29 null prompts. |
| inconsistency | Short H contradicts "Plan for the future of the household" / "Think deeply" | 44 | x0083 "plan for the future ... yet the stated horizon is only 10 minutes" | Design (SITUATION and OBJECTIVE are fixed template text). Taste. |
| inconsistency | Forced choice vs "none" is the correct answer | 22 | x0384 "FORMAT requires choosing 1) or 2) ... format contradicts the study rule" | Same as the first row. |
| other | Restates "no option pays off" or "best investment" vs the rule | 24 | x0014 | Noise. Several flags say "no conflict found". |
| grammar | "dollars" without a symbol; odd decimal horizon | 8 | v012-far-first "2.24 years ... non-round decimal" | Noise |

Overlap: 122 flags hit the "none/neither" wording across categories, and all of them are in `none` prompts.

## 2. Anything that could make a LABEL wrong or debatable

Checker `options_within_horizon` and answers match for all 700 prompts, so these are risks for the *model*, not generator errors.

| uid | Reason | Label if rule changed |
|---|---|---|
| x0004 (60 months, far = 5 y, rewrite of "5 years", `both`) | Exact tie on the FAR delay. Under strict "<" the label flips to near. | both -> near_only |
| x0311 (5 years, far = 5 y, `both`) | Exact tie on far. | both -> near_only |
| x0042, x0224, x0234 (3 months, near = 3 months, `near_only`) | Exact tie on the NEAR delay. Phrasings 3, 0, 2 respectively. Phrasing 0 ("for this time horizon: 3 months") reads less clearly as inclusive than phrasing 2 ("within the next"). | near_only -> none |
| x0238, x0272, x0290, x0340 ("183 days" vs "6 months" near, `near_only`) | Unit rewrite 0.2% over the delay (182.625 d). Correct only under the stated conversion. A calendar reading (183-184 d) can give "not within". x0272 and x0290 were flagged for exactly this. | near_only -> none |
| v049, v050, v051, v052 (both orders) | Ratio H/D 0.969, 0.990, 1.010, 1.032 with a 2-decimal H ("1.98 years" vs "2 years"). Labels are exact. Models must compare decimals, so errors here are arithmetic, not label noise. | none |
| All 200 verification prompts | H has 2 decimals ("6.67 years"); flagged as odd in v012, v031. 5 flags total. | none |
| `none` regime (279 prompts) | ACTION says "Select one of the two options" but nothing pays off, so any answer is a forced guess. The label is "none", which the model cannot give. Fine for fitting vectors, but flags/responses here must never be scored. | none |
| "within" vs "at" | Only phrasing 2 says "within the next X". Phrasings 0, 1, 3 say "this time horizon", "planning horizon", "time frame". Inclusive ties are therefore explicit in only 1 of 4. Hits the 5 ties above (3 of them in phrasings 0 or 3). | see ties |

Prompts needing no change: the other rewrites are safe. I scanned all rewritten horizons for ratio within 5% of any delay: only x0004, x0238, x0272, x0290, x0340 qualify.

## 3. Flag rate by factor (prompts with >=1 flag)

| Factor | Level | n | flagged | rate |
|---|---|---|---|---|
| split | extraction / verification | 500 / 200 | 237 / 37 | 47% / 18% |
| regime (ext) | none / both / near_only / no_horizon | 279 / 86 / 106 / 29 | 62% / 37% / 27% / 7% | |
| regime (ver) | both / near_only | 100 / 100 | 26% / 11% | |
| phrasing | 0 / 1 / 2 / 3 | 183 / 171 / 152 / 165 | 51% / 41% / 36% / 33% | |
| phrasing x none | 0 / 1 / 2 / 3 | 69 / 74 / 61 / 75 | 67% / 61% / 57% / 64% | flat |
| phrasing x both (ext) | 0 / 1 / 2 / 3 | 25 / 16 / 22 / 23 | 60% / 50% / 23% / 17% | phrasing 0 stands out |
| label style | a) / 1) | 350 / 350 | 42% / 36% | small |
| unit rewrite | no / yes | 528 / 172 | 43% / 27% | rewrites are lower, because they are less often in `none` |
| option order | far first / near first | 367 / 333 | 41% / 37% | none |
| horizon unit | seconds / minutes / hours / days / weeks / months / years | 61/84/87/62/42/55/280 | 72/61/67/29/29/24/27% | all short units are `none` |

Reading: the main confound is regime, not any template factor. Short-horizon (`none`) prompts draw 62% of flags from "nothing pays off, but choose one". Within `none`, flag rates are flat across phrasings (57-67%), so phrasing is not the cause there. Phrasing 0 carries the "benefit" ambiguity, so it attracts extra flags in `both` and in verification (68 of the 200 verification prompts use phrasing 0). That is why verification flags are mostly "benefit" (21 of 37 in `both`). Unit rewrite and label style are not confounds.

## 4. Recommendations (ranked by impact on steering/eval labels; existing splits unchanged)

| # | What | Why |
|---|---|---|
| 1 | Verification (68 of 200 prompts use phrasing 0): decide whether "greatest benefit for this time horizon" is acceptable. If scoring depends on the model choosing the larger nominal payout among options that pay off, add a note when reporting, or check the model's reasoning for discounting on the 26 flagged `both` verification prompts. | 20-21 flags in `both`, the scored split. A discounting reading could favour the near option and look like "horizon sensitivity". |
| 2 | Treat the 5 tie prompts and 4 "183 days" prompts as ambiguous: exclude from fitting or report with and without (x0004, x0311, x0042, x0224, x0234, x0238, x0272, x0290, x0340). | Their labels flip under a strict reading. 9 of 471 horizon prompts. |
| 3 | Never score `none` prompts. Keep them in extraction only (README already says so). For analysis of what the model does on them, note that the forced choice makes any answer arbitrary. | 174 flagged prompts here. |
| 4 | Short-H prompts in phrasing 3 ("time frame of X") and 0: expect a deliberation-time reading. Check the model's reasoning on a few before trusting contrasts that include them. | 20 flags. Only affects `none`. |
| 5 | Ignore: placeholder brackets, "dollars" formatting, implausible-horizon flags. They come from the report's template and are by design. | Taste or noise. |

## Summary

- 274 of 700 prompts (460 flags) flagged; zero label errors: regimes recomputed exactly, checker agrees on all 700.
- 62% of `none` prompts are flagged for "nothing pays but choose one" and "unrealistic short horizon". That is by design and `none` is never scored.
- Real issue: "greatest benefit" in phrasing 0 (57 flags, 53 in phrasing 0), including 27 verification flags (21 on `both` prompts), could invite discounting.
- Debatable labels: 5 exact ties (x0004, x0311, x0042, x0224, x0234) and 4 "183 days" vs "6 months" rewrites (x0238, x0272, x0290, x0340).
- Flag rate is driven by regime, not by label style, option order or rewrites. Phrasing 0 is the only template confound.

## Correction and behaviour check (added by the main session, 2026-11-26 04:30)

- The draft said all verification prompts use phrasing 0. They do not: 68 / 52 / 38 / 42 of the 200 use phrasings
  0 / 1 / 2 / 3 (`prompt_bank/data/t0_s0/prompts.jsonl`). Fixed above.
- Flags are not behaviour. On the Qwen3-8B thinking-on baseline (`results/qwen8b_quick_20261125/.../items.jsonl`),
  accuracy on `both` prompts by phrasing is 0.43 / **0.26** / 0.56 / 0.49 (n = 51 / 35 / 36 / 35); on `near_only`
  0.94 / 1.00 / 1.00 / 0.96. Phrasing 0, which draws the "benefit" flags, is no worse than the rest on `both`
  (0.43 vs 0.43). **Phrasing 1** ("Choose the option that is best over that horizon") is: 0.26 vs 0.50 for the
  others, Fisher p = 0.013 (uncorrected, one of four). A likely reading, untested: "best over that horizon" invites
  the reinvestment arithmetic behind the `both` errors (`EvalsPromptsV2.md` §1).
- So the template confound that matters for scoring is phrasing 1, which the checker did not flag. Phrasing belongs in
  the diagnostics page (accuracy by phrasing x regime) and in the C5 held-out split.
- `both` accuracy is 0.43 overall: on more than half of the prompts where both options pay off, the model picks the
  smaller, sooner one. Any steering effect on `both` rows is read against that baseline.
