# Spec E20: phrase donor patching on the eval types, 2026-11-21

Status: **approved 2026-11-21** (with Q1 yes, Q2 keep arm C; extras trimmed to fit the approved cost, §2). Decisions already taken (2026-11-21): steering directions are
fitted on the full bank only (a transfer test); GPU budget for this session is $5; the clear wording (domain
truth sentence + end-of-horizon constraint) goes into the 4 domain types only, never the full bank.

## 1. Why the plan changes from "run E8/E13/E14 on the new prompts"

What moves the choice on Alan's bank with thinking on (Qwen3-8B), which is the mode every eval type ran in:

| method | position | bank effect, thinking on | source |
|---|---|---|---|
| E8 fitted curve, ±2 decades | answer slot L22/L24 | ≤ 0.03 C (pristine fit); 0.14-0.15 of contrast on crossing rows (E19's measure) | `22` appendix A; `EvalsPromptsV2.md` §8 |
| E13 fitted step, ±1 decade | suffix newline L16 | nothing beats random (all CIs span 0) | `exp13_Qwen3-8B_pristine_suffix/summary.json` |
| E14 fitted manifold step | horizon phrase L16 | H1–H3 fail; +1.4 near log-odds | `22` §4 |
| E14 donor patch (phrase states of the same prompt re-rendered at the target horizon) | horizon phrase L16 | **+9.9 near log-odds** | `22` |
| E16 number-token donor, same unit | number tokens L16 | **+5.2, donor number written 24/24** | `30` (27 section) |

**E19 (another session, same day, `EvalsPromptsV2.md` §8) already ran the E8 transfer:** the bank's L24
answer-slot curve moves log-odds on almost every eval type, beating random (specific 0.04-0.24 thinking on).
But it flips 0-3 answers per type, and on `coding_agent` it does nothing. E20 does not repeat it.
Transferring E13/E14's fitted directions would measure a null effect on new domains, and a null there
says nothing about the domain. So the main measurement is the **donor patch**, the one intervention that works
on the bank. It asks whether, in a new domain, the horizon phrase still carries the horizon the decision uses.
One bank-fit direction stays in as the transfer arm (§3, arm C), mainly to show whether it stays null.

## 2. Prompts

The 4 domain types we have: `city_no_values`, `city_values`, `public_health`, `coding_agent`. Each has 32 rows,
2 options and Alan's template: 14 near, 14 both and 4 none, all with `near_label`, all with `choice_prefix`
"I choose:", and `"horizon: " + horizon_text` present and unique in the prompt.
Extra types (Q1, approved), to give the distance axis more than 4 points. They pass the same filters (2 options,
`near_label` on every row, phrase found once, prefix "I choose:"): `distractor_temporal`, `multiturn_analyst`,
`role_assistant`, `role_fund_manager`, `sycophancy_lean`, `system_prompt`, `system_prompt_clear`, `t0_w0`,
`t0_w1_cash`, `t0_w3_clear`. Dropped: `json_io` (mixes two choice prefixes) and `sycophancy_lean_clear` (built
wrong, V2 §5). The extras get **up to 16 rows each (8 near, 8 both, fixed seed, drawn among the rows that
have a donor) and arms A and B only**. Rows after the donor filter (Qwen3-8B tokenizer): domain types 21, 24,
21, 28 (all rows); extras 11-16, where near rows are short (3-8) because a near→both donor must pass the far
delay with the same phrase token count.

## 3. Arms (thinking on, max 3072 think tokens, E0 generation settings plus the type's `answer_window`)

- **A. noop:** regenerate unpatched. bf16 batching is not bit-stable, so this sets the flip noise floor; we don't
  reuse the E0 answer.
- **B. donor patch at L16, phrase tokens:** copy the L16 phrase states from the same prompt with `horizon_text`
  replaced by a donor horizon with the **same phrase token count** (the patch is token to token). The unit may
  change, as in E14, where 95 of 104 donors changed unit. Requiring the same unit would leave 0-9 rows per type
  (0 of 28 for `coding_agent`). The target is E14's `transition_target` (near→both, both→near, none→near).
  The donor rule is E14's `find_donor`. The donor prompt is built by replacing the string, not by
  `render_variant`, which can't rebuild eval-type prompts.
- **C. bank-fit transfer:** E14's `effects L16` step, refitted from `<exp14 dir>/states.npz` with
  steering_prepare's own functions (`inputs.npz` holds only the bank prompts' steps; the refit reproduces them
  to float32 rounding), with the same targets as B, plus `random L16` at the same norm.

Per row: near log-odds, label, forced, whether the trace names the target horizon.
Per type: the flip rate toward the target regime minus noop's flip rate, the shift in near log-odds versus noop
with a bootstrap CI, and the forced rate.

## 4. Report

Plot effect (B, C) against the horizon-subspace distance (`eval_distance_rows.json`, joined on `sample_uid`,
last prompt token). The bank's own E14 donor/effects numbers serve as the reference point. With 4 types it is a
per-type table with CIs, **not a correlation**: there are only 4 x-values, and n ≈ 28 steerable rows per type.

## 5. Code (one adapter, one script; E8/E13/E14 unchanged)

1. `spar_horizon/eval_types/records.py: eval_records(rows_path)` normalizes eval rows to the steering record fields:
   `near_delay_years`/`far_delay_years` from `options[kind]`, the label from `answer`, and `near_lo` from
   `label_logits` and `near_label`. It drops rows with no `near_label`.
2. `pipeline/s2_steering/experiments/manifold_steer/exp20_evaltype_transfer.py --types ... --exp14-dir ... --arms` (run with `python -m`):
   it reuses E14's `generate`/`score`, `S.phrase_positions` and `S.prompt_patch`, and is resumable per (type, arm)
   with E13's `run_claimed`.
3. Tests (fast suite, no model): field normalization on real rows.json; donor string replacement keeps everything
   outside the phrase; the donor's token count matches on the Qwen tokenizer; phrase positions are found for all
   128 rows.
4. Local smoke for Jared (xpu, `local_checks.py`-style): 2 rows × arm B with 256 think tokens, before renting.

## 6. Cost (4090 NA/EU at ~$0.43/hr; reference: the bank's E13 took ~21 min per condition on 96 prompts)

Pace ≈ 0.22 GPU-min per row per arm. 4 domain types: 4 runs (A, B, C effects, C random) × 94 rows; 10 extra
types: 2 runs × 139 rows. 654 row-runs ≈ 2.4 h, plus ~0.5 h setup ≈ **2.9 h ≈ $1.25**. That leaves ~$3.5 of the
$5 for the v3 prompt run (next spec: conservation, career, domain truth sentence, TWIN_PER_REGIME 8, saving the
phrase token). I'll check `just burn` per type and stop
at $4.50.

## 7. Run order

Domain types first, arm B before A and C, so the main measurement lands first if the run is cut short.
Then the extras.
