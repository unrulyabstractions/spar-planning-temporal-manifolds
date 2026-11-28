# Draft spec: horizon patching and steering with the reasoning regenerated (for approval, 2026-11-28)

Follows `REPORT_2026-11-28.md`. Under the teacher-forced replay, a ×10 rewrite of the stated horizon moves the answer
only +0.52 logits, because the replayed reasoning decides it. E14/E16/E20 regenerated the reasoning and got +5 to +29
near log-odds from an L16 phrase patch. This experiment measures tonight's arms the same way, so the patch can serve as
the baseline / ceiling for steering that it was meant to be.

## Question

With the reasoning regenerated, how much of a real horizon change (text rewrite) does each prompt-side edit carry:
1. the full L16 patch (whole state);
2. the k = 4 curve subspace of it;
3. a horizon-token steering push?
And which of them beat random controls of the same form?

## Arms

Each edit runs at both signs (source 10× longer / 10× shorter, or push ±x). All edits are prefill-only
(`steer.hooks.at_prompt`), so the edited prompt states reach the reasoning through the KV cache, as in E20.

| arm | what | runs |
|---|---|---|
| noop | regenerate unedited (noise floor) | 1 |
| patch:text | horizon ×10 / ÷10 in the prompt | 2 |
| patch:full | L16, horizon's last token through prompt end, from the ×10 / ÷10 source | 2 |
| patch:curve-displace | the same, restricted to the k = 4 curve PCs | 2 |
| patch-null:k4[0..2] | the same, random 4-dim bases from the covariance draws | 6 |
| steer caa | L12 horizon_last direction, written from the horizon to prompt end, x = ±0.4 (tonight's winner) | 2 |
| covariance[0..2] | random directions, same span and strength | 6 |

21 runs × 48 items (24 verification pairs, near_only and both, from the 178 readable).

## Readout, per generation

- **y after the new reasoning:** logit(far) − logit(near) at the answer slot, teacher-forcing the generated trace.
  This is the same quantity as the replay readout, so it is on tonight's scale and continuous, with more power than
  label counts. E20 used this too.
- **P(far):** chose far, from the parsed label.
- **Restates the source horizon:** whether the trace names the source's horizon (E20's `names_target`), or the
  original one.
- **Hit cap:** forced at the think cap.
- **Summary per arm:** meaning, letter and even (analyze.py's split by sign and option order), each against noop,
  with a paired bootstrap over pairs. Rank p against the nulls of the same form.

## Settings

- Greedy decoding, like E20: arms then differ only by the edit. The baseline was sampled, so noop is the reference,
  not the saved baseline.
- Think cap 4096, as in the baseline. bf16 with batched generation.
- Hosting:
  - GPU: A100 80 GB. Generation is bandwidth-bound, so it is cheaper per trace than an H100 (~$1.10/hr vs $3.23/hr
    for ~1.6× speed).
  - The tonight-style driver runs under `timeout`, with the deadline set from the budget.

## Cost

- **Per trace:** baseline traces average 1834 tokens; batch 16 on an A100 40 GB took ~10 s per prompt.
- **This run:** 1008 traces at ~8 s (batch 32 on 80 GB) is ~2.2 h. The scoring pass, model load and the source
  forwards add ~0.4 h. That is ~2.6 h, ~$2.90.
- **Cap risk:** strong pushes ran to the cap in the 2026-11-25 gencheck. A 1.5× allowance gives ~$4.30.
- **Pilot first:** time one batch of noop and patch:full, then fix the deadline from the measured speed.
- **Priority order if time runs short:** noop, text, full, curve, its nulls; then steering and its nulls.

## Code to add (evaluation/, ~150 lines + tests)

1. **gencheck accepts patch arms.**
   - Build `patch:full`, `patch:<method>` and `patch-null:k{k}[i]` with `EditFactory.make` (it already does).
   - Call `edit.prepare(rep, items, span, layer)` before generating.
   - Pass `rows` to `steering(...)`.
   - Map the span through `patching.PATCH_SPANS` for `at_prompt`.
2. **Scoring pass.** For each generated trace, teacher-force it with the edit applied as during generation, and read
   y at the answer slot. Reuse `Replayer` with the generated reasoning in place of the baseline's.
3. **Restates-horizon check**, and `--gen-greedy`.
4. **Tests** on the tiny random Qwen3 with the real tokenizer (as `test_patching.py`):
   - the patch changes only prefill states at the aligned tokens;
   - decades = 0 gives noop's tokens exactly (greedy);
   - the scoring pass matches the replay when the trace is the baseline's.

## Open choices (Jared)

1. **Budget.** ~$3-4.50 planned. `.env` shows $44.70 of $50 spent, so this needs a new session budget.
2. **Nulls.** K = 3 per family keeps it affordable, but then p cannot go below 0.25. That is enough to see whether an
   arm sits inside or outside the null range, not to claim significance. K = 9 for the curve patch only would add
   12 runs, ~+$1.40.
3. **Keep the steering arm?** Tonight's CAA push did not beat its random directions under replay. Generation may
   change that. Dropping it saves ~$1.40.
4. **Donor rule.** ×10 in the same unit (tonight; only the last token aligns), or E20's same-token-count donor
   (whole phrase aligns; usually another unit, not one decade). Recommended: ×10, for comparability with tonight's
   replay.
