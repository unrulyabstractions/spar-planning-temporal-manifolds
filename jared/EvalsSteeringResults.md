# E20 results: phrase donor patching on the eval types (2026-11-21)

Spec: `EvalsSteeringSpec.md`. Run: one RTX 4090, 2026-11-21 21:52 to 2026-11-22 02:47 UTC (box 61864062),
$1.72 this session including ~1 h idle after the OOM crash below. Outputs:
`~/SPAR/results/horizon/exp20_Qwen3-8B_evaltype_transfer/` (`<type>/conditions/<arm>.json`, `<type>/summary.json`,
`donor_table.md/.json`, run log `../exp20_run.log`). Table: `python -m pipeline.s3_analysis.eval_types_donor`.

**Question.** Steering directions fitted on Alan's prompts barely move a reasoned choice (E13, E14 null on the bank;
E19: the L24 answer-slot curve moves log-odds on the eval types but flips 0-3 answers per type, and nothing on
`coding_agent`). The one intervention that does flip answers on the bank is the donor patch: copy the L16 states
of the horizon phrase from the same prompt with a different horizon (E14: +9.9 near log-odds). Does it still work
on prompts far from Alan's, in other domains (city, public health, coding agent) and formats (system prompt,
multi-turn, sycophancy, role)?

**Arms** (thinking on, 3072 think tokens, E0 generation settings). The donor horizon is the grid horizon nearest
the target (log-midpoint across the nearest delay), in the target regime, with the same phrase token count.
- `donor`: L16 phrase states replaced by the donor prompt's.
- `noop`: an unpatched rerun, the noise floor (bf16 batching is not bit-stable).
- `effects` / `random` (4 domain types only): E14's bank-fit step at L16 from the row's horizon to the target, and
  a random step of the same length.

Flipped = share of rows whose E0 label was not the target answer that now give it. Log-odds = near log-odds
change versus noop, signed toward the target, bootstrap 95% CI. Dist = horizon-subspace distance from Alan's
prompt at L18 suffix end (`eval_distance.json` rank cell; 0 is Alan's wording, `t0_w0`).

| type | dist | n (no donor) | noop flipped | donor flipped − noop | donor log-odds vs noop | effects log-odds | random log-odds | forced noop / donor |
|---|---|---|---|---|---|---|---|---|
| t0_w0 | 0.00 | 13 (16) | 0.11 | +0.67 | +10.3 [+4.8, +16.0] | | | 0.15 / **0.69** |
| distractor_temporal | 1.30 | 16 (4) | 0.07 | +0.80 | +12.0 [+7.0, +16.5] | | | 0.12 / 0.25 |
| t0_w1_cash | 1.57 | 13 (16) | 0.08 | +0.92 | +23.1 [+20.1, +25.7] | | | 0.08 / 0.08 |
| city_no_values | 2.24 | 21 (11) | 0.07 | +0.86 | +19.5 [+13.9, +24.6] | +2.5 [-1.4, +6.3] | +0.8 [-0.7, +2.6] | 0.00 / 0.00 |
| t0_w3_clear | 3.63 | 13 (16) | 0.00 | +1.00 | +29.1 [+26.2, +31.7] | | | 0.00 / 0.15 |
| public_health | 3.83 | 21 (11) | 0.07 | +0.27 | +7.9 [+3.7, +12.4] | +0.9 [-1.8, +4.0] | +0.9 [-1.6, +3.5] | 0.29 / 0.29 |
| city_values | 5.18 | 24 (8) | 0.05 | +0.77 | +22.0 [+17.8, +25.9] | +1.3 [-0.4, +3.6] | +0.7 [-1.1, +2.9] | 0.04 / 0.21 |
| role_assistant | 5.55 | 12 (15) | **0.50** | +0.12 | +0.5 [-4.3, +5.5] | | | 0.08 / 0.42 |
| role_fund_manager | 6.40 | 15 (8) | 0.21 | +0.43 | +11.8 [+7.8, +16.4] | | | 0.13 / 0.47 |
| system_prompt | 7.59 | 15 (8) | 0.00 | +0.90 | +13.6 [+8.2, +19.1] | | | 0.20 / 0.27 |
| sycophancy_lean | 8.20 | 11 (13) | 0.12 | +0.88 | +13.0 [+6.7, +19.3] | | | 0.00 / 0.00 |
| coding_agent | 8.78 | 28 (4) | 0.09 | +0.48 | +13.2 [+8.5, +18.3] | +0.8 [-1.0, +2.5] | +1.2 [-0.4, +2.8] | 0.14 / 0.07 |
| multiturn_analyst | 9.37 | 16 (6) | 0.00 | +0.71 | +20.1 [+15.7, +23.7] | | | 0.06 / 0.06 |
| system_prompt_clear | 10.24 | 15 (8) | 0.00 | +0.93 | +28.3 [+25.9, +30.7] | | | 0.07 / 0.07 |

Spearman across the 14 types, distance vs donor effect: flipped −0.00 (p = 0.99), log-odds +0.19 (p = 0.52).

## Findings

- **The horizon phrase drives the choice in every domain and format tested.** The donor patch beats noop in 13 of
  14 types, by +7.9 to +29.1 near log-odds (CIs clear of 0), and flips 0.27-1.00 of the flippable rows over noop's
  0-0.21. On the bank the same patch gave +9.9 (E14), so the eval types respond at least as strongly.
- **Distance from Alan's prompt does not predict it** (ρ ≈ 0 for flips, +0.19 for log-odds, n = 14). The farthest
  types (`system_prompt_clear` 10.2, `multiturn_analyst` 9.4, `coding_agent` 8.8) respond fully.
- **The bank-fit step does not transfer, and was null on the bank too.** On the 4 domain types E14's effects step
  moves near log-odds by +0.8 to +2.5 with CIs spanning 0, the same as a random step (+0.7 to +1.2).
- **`coding_agent` responds to the donor (+13.2) although E19's answer-slot steering does nothing there.** Its
  decision does read the horizon phrase; the bank's horizon curve is not the direction it reads at L24. The
  donor works because it carries the real state, not a direction fitted on investment prompts.
- **`public_health` is the weakest real effect** (+7.9, flipped +0.27), with 0.29 of traces force-closed in both
  noop and donor.

## Caveats

- **`role_assistant` has no usable baseline.** An unpatched rerun changes E0's answer on 6 of 12 rows (noop
  flipped 0.50), so a flip rate against E0 cannot show an effect. Label against label, donor versus noop, the
  donor moves 2 rows toward the target and none away; on the 4 near→both rows it changes nothing.
- **Donor traces force-close more often in some types** (`t0_w0` 0.69 vs 0.15 noop, `role_fund_manager` 0.47 vs
  0.13, `role_assistant` 0.42 vs 0.08). A forced close ends the reasoning at 3072 tokens, so the answer is read
  after a cut-off trace. `t0_w0`'s +10.3 still clears 0.
- **Few rows, and few near rows.** 11-28 rows per type. A near→both donor needs a horizon past the far delay with
  the same phrase token count, so 4-15 rows per type have no donor (most of them near rows). The extras have
  3-8 near rows each.
- **The donor is a within-prompt ceiling, not a deployable steer.** It needs the prompt re-rendered at the target
  horizon. What it shows is where the decision reads the horizon (the phrase tokens at L16), in every format.
- **First run crashed** at 22:08 UTC with CUDA OOM in `score` (8 traces of up to ~3,300 tokens). Rerun with
  `--score-batch-size 4` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, after removing the stale
  `effects.lock` (a crashed run's lock makes `run_claimed` skip the arm silently).

## Next

1. A donor-free steer that carries what the donor carries: the per-prompt real phrase-state change is not the
   bank curve (E14: R² 0.02 by the steer over all donors, 0.44 for the 9 same-unit donors). Fit the step on
   pooled eval-type donor deltas, then test it on held-out types.
2. Score donor against noop label by label (as for `role_assistant`) for every type, so unstable baselines don't
   blur the flip rate; rerun `role_assistant` with more rows if it matters.
