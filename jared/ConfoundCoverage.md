# Confound coverage: what our banks, splits, labels and checks can show (2026-11-26)

For each confound: where it can be seen now, and the cheapest fix where it cannot. Confounds C1-C8 are those of
`evaluation_design/README.md` §8; the rest were found in the 2026-11-25 Qwen3-8B run (`SteeringQwen8B.md` §1) or are
built into the bank design. "Diag" = `evaluation/diagnose.py` (`EvalDiagnosticsSpec.md`).

T0 bank facts used below (`prompt_bank/data/t0_s0/prompts.jsonl`): extraction 500 prompts (279 `none`, 106
`near_only`, 86 `both`, 29 `no_horizon`), 4 CONSTRAINT phrasings (115-123 each), `rewritten` unit flag, `labels`
(a/b or 1/2), `near_first`, reward multiplier 2-50 (90-109 each), `horizon_equals_a_delay`; verification 200 prompts
(100 `near_only`, 100 `both`, each item in both orders), multiplier varied (32-48 each), all 4 phrasings.

## Matrix

| Confound | Labels in the records | Where it can be seen | Check that shows it | Gap | Cheapest fix |
|---|---|---|---|---|---|
| **C1 answer letter / position** | `labels`, `near_first`, far label; `choice_label` in runs | verification grid (both orders) | `analyze.py` meaning vs letter; Diag C letter cosine, B letter/order bias | fit classes are not letter-balanced; the `label` site holds the letter itself | Goals task 3c (balanced fit subsets); fit at `prelabel` or earlier |
| **C2 decision vs horizon** | `regime`, `correct_option`, `chosen_option` | verification grid; extraction regimes | `analyze.py` plateaus/slope; Diag C cos with `choice` and regime | no horizon-only direction tested; at the answer, regime ≈ decision (cos 0.55-0.85) | regime direction at `prompt_end` (cos 0.08 with choice), `SteeringQwen8B.md` v2 |
| **C3 reward sensitivity** | `reward_multiplier` | both splits (varied) | reward direction fitted; cos in Diag C | the switching-multiplier test (design §8 C3) is not built; and **far reward > near reward and far delay > near delay in every prompt**, so "prefer more" and "prefer later" are never separated | build the C3 test on `both` rows; add a bank tier where the later option is the smaller one (correct = near even in `both`) |
| **C4 disruption** | runs: hit cap, trace length | any run | even part, slope, N6 null, gencheck cap rate; Diag F | no next-token KL on neutral text; replay holds reasoning fixed, so disruption of reasoning only shows in gencheck | KL on a fixed neutral text set per strength (cheap, replay-only) |
| **C5 surface form** | `phrasing_id`, `rewritten`, `labels` | extraction split | none yet | no held-out-variant split | Goals task 3b (fit on some phrasings/renderings, test on others); eval-type banks are the far version |
| **C6 rescale vs replace** | `no_horizon` (29 null twins) | extraction split | none yet | analysis not built | realised Δlog H* vs stated horizon (replay data already has it) + effect on `no_horizon` twins |
| **C7 stated horizon** | — | needs generation | none yet | needs steered generation + a judge | later; generation is the costly part |
| **C8 calibration** | `horizon_grid_value` | extraction split | dose curve in Diag D | no reserved horizons | Goals task 3a (hold out grid values from fitting) |
| **`none` regime** (neither pays off) | `regime` | extraction (279 of 471 horizon prompts) | Diag B far rate on `none` (Qwen3-8B: 0.39, like a long horizon) | it fills CAA's short class (275 of 346) | leave `none` out of horizon contrasts, or make it its own class; report it separately |
| **horizon length ≡ regime** | `h_over_far_delay`, `regime` | extraction | Diag C class composition by regime | in the extraction split long horizons are mostly `both`, short mostly `none`: horizon and regime move together | Goals task 3d (minimal pairs: same item, H just below / above the far delay) |
| **ties** | `horizon_equals_a_delay` (5 prompts) | extraction | label audit (task 2) | — | keep flagged; drop from fitting |
| **trace length / forced close** | runs: hit cap, tokens | any run | Diag C cosine with hit-cap direction; gencheck cap rate | not a fit-time control | add trace-length bins to the per-prompt table |
| **thinking derailment** (loops, cap) | runs | gencheck, `check_loops.py` | cap rate per arm | steered generation hit the cap 15/16 at x = 0.8 | strength sweep for generation (separate from replay calibration) |
| **readout precision** | run settings | N0 check | Diag F | bf16 batched vs alone 0.5 logit | fp32 replay (suite default) |
| **checker-flagged prompts** | `checks/*.jsonl` problems | T0 | Diag A counts; triage doc | 460 flags not yet triaged | Goals task 5 (`autoresearch/checker_triage_t0.md`) |
| **eval types: labels** | `family`, `variant`, twin, `regime`, truth | eval-type banks | label audit (task 2) | no second-model check; v1 wording had ambiguous truths (`EvalsPromptsV2.md` §1); label schema draft not approved | rule-based recheck (task 2); checker run when API spend is approved |
| **eval types: splits** | `split = transfer` | eval-type banks | transfer report | no fit/test split inside a type; 32 rows per original bank | use the 100-item expanded banks; a per-type held-out half |
| **distance from Alan's prompt** | `eval_distance_rows.json` | eval-type banks | E19-E22 tables | weak (ρ ≈ -0.3, n.s., n = 13-18 types) | more types, or steer before reasoning where distance is measured |

## Biggest design gaps (in order)

1. **Horizon and regime are not separable in the extraction split, and `none` prompts behave like long horizons.**
   Every contrast fitted on it mixes horizon, regime and the correct answer. Minimal pairs (task 3d) fix the fit data.
2. **Later is always larger.** No prompt has a later-but-smaller option, so a "prefer the bigger number" direction
   and a "prefer later" direction are indistinguishable on every bank. A small tier with the far option smaller (and
   correct = near in every regime) is the only way to separate them.
3. **No held-out splits for C5 and C8** (tasks 3a, 3b), although the labels to build them are already in the records.
