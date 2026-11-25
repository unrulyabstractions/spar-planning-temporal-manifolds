# Draft: one label schema for every eval prompt bank (NOT APPROVED, 2026-11-10)

Goal: every bank (T0 structured, T1 free-text, T2 implicit, T3 open-ended, domain shifts,
format shifts, multi-turn, distractors) emits records with the same keys, so one analysis
reads all of them and the expected-horizon readout `E[log h] = sum_i p_i log h_i` works on
every bank that has options. Keys a bank does not use are `null`, never missing.

Sources: `EvalsSpec.md` (this folder), midterm report appendix A.2-A.5, the existing records in
`spar_horizon/full_bank.py` (`record`, `emit_choice`) and `spar_horizon/implied_bank.py`.
Existing key names are kept where they exist (`regime`, `truth`, `truth_label`, `n_options`,
`options`, `roi_explicit`, `twin_id`, `anchor_id`, `rendering`).

## Record (one JSON object per line)

### Identity
| key | type | meaning |
|---|---|---|
| `id` | str | unique, `<family>_<nnn>` |
| `family` | str | prompt type (list below) |
| `tier` | str | `T0` structured, `T1` explicit in prose, `T2` implicit, `T3` open-ended |
| `twin_id` | str | shared by renderings of one base (implicit / explicit / null twins, option-order swaps) |
| `rendering` | str | `explicit`, `implied`, `null` (no horizon at all), `vague` |
| `source` | str | `hand_sample` now; later `template`, `llm:<model>` |

### Content
| key | type | meaning |
|---|---|---|
| `system` | str or null | system prompt, null if none |
| `turns` | list | `[{"role": "user"/"assistant", "content": str}]`; the model answers after the last user turn |
| `answer_format` | str | `label` (a/b/...), `json`, `free` |
| `domain` | str | `investment`, `city_planning`, ... |
| `role` | str or null | who the model is asked to be: `household`, `hedge_fund_pm`, `city_council`, ... |
| `surface` | str | `structured` (Alan template), `prose`, `json` |

### Options (empty list for free-form)
`options`: list of `{label, text, value, value_unit, h_years, return_pct_annual, risk, kind}`
- `h_years`: the option's duration, used by `E[log h]`: for an investment the time until it pays
  out (or the holding period); for a project the time until its benefit arrives.
- `value`: payout or benefit in `value_unit` (null when the prompt gives none).
- `return_pct_annual`: implied annual return (null when not computable).
- `risk`: `low` / `medium` / `high` / null, crossed with horizon so long does not mean safe.
- `kind`: `near` / `far` for 2-option; for k options the rank of `h_years` (1 = shortest).
Plus `labels` (shown order), `n_options`, `truth_label`.

### Horizon
| key | type | meaning |
|---|---|---|
| `horizon_cue` | str | `explicit`, `implicit`, `none` |
| `horizon_text` | str or null | the surface phrase ("before our baby is born") |
| `horizon_years` | float or null | point value (the explicit twin's value for an implicit item) |
| `horizon_lo_years`, `horizon_hi_years` | float or null | interval a reasonable reader could hold |
| `anchor_id` | str or null | row id from the implicit-horizon table in `EvalsSpec.md` |
| `anchor_type` | str or null | `concrete` / `vague` (the table's Type column) |
| `explicit_twin_prompt` | str or null | sample stage only: the same prompt with the horizon stated outright |

### Truth
| key | type | meaning |
|---|---|---|
| `truth_rule` | str | `max_value_within_H` (an option counts only if `h_years <= H`; pick the largest value among those), `none` (no defined answer) |
| `regime` | str or null | 2-option: `none` / `near` / `both` / `null` as in `full_bank.regime`; k options: `k_within_<n>` |
| `truth` | str or null | `near` / `far` / null (2-option) |
| `truth_label` | str or null | label of the correct option |
| `truth_stable_on_interval` | bool or null | the same option is correct for every H in [lo, hi] |
| `roi_condition` | str | `equal_10` (all options compound at 10%/yr from one principal), `equal_5`, `unequal`, `not_stated` |

### Readout for free-form and judged answers
| key | type | meaning |
|---|---|---|
| `readout` | str or null | `choice`, `equity_share`, `bond_duration_years`, `horizon_of_recommended_action` |
| `expected_direction` | str or null | what a longer horizon should do to the readout (`up` / `down`) |
| `reference_range` | [lo, hi] or null | a defensible answer range, with its source in `notes` |

### Robustness
| key | type | meaning |
|---|---|---|
| `distractors` | list | `[{"text": str, "years": float}]`: durations in the prompt that are not the horizon |
| `distractor_truth_label` | str or null | what `truth_label` would be if the model used the distractor as H |
| `prior_turn_recommendation` | str or null | multi-turn: label the earlier assistant turn favoured |
| `prior_turn_correct` | bool or null | whether that earlier recommendation equals `truth_label` |
| `notes` | str | anything a reviewer should know (assumptions, ambiguity) |

## Families for the first samples (5 each)

| family | tier | what varies |
|---|---|---|
| `inv_t1_2opt` | T1 | free-text, stated horizon, 2 options, risk crossed |
| `inv_t1_k_equal10` | T1 | 4 or 8 options, all at 10%/yr from one principal |
| `inv_t1_k_equal5` | T1 | same at 5%/yr |
| `inv_t1_k_unequal` | T1 | 4 or 8 options, unequal returns, no rate stated |
| `inv_role_hedgefund` | T1 | system prompt makes the model a fund manager / finance role |
| `inv_json` | T1 | the prompt and answer are JSON |
| `inv_t2_implicit_2opt` | T2 | horizon from the anchor table, 2 options |
| `inv_t2_implicit_k` | T2 | horizon from the anchor table, 4 or 8 options |
| `inv_t3_open` | T3 | free-form allocation / duration, horizon implied |
| `city_unvalued` | T1/T2 | myopic vs long-term project, no values given (no truth; `E[log h]` only) |
| `city_valued` | T1 | projects with costs, benefits and times; truth by `max_value_within_H` |
| `multiturn_cba` | T1 | earlier assistant turn holds a cost-benefit table in a fixed format; then choose |
| `distractor` | T1 | an irrelevant duration in the prompt ("the movie lasted 3 hours") |

## Open questions for Jared
See the review doc that accompanies the samples.
