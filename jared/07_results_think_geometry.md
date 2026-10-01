# Results: the horizon axis during reasoning (Qwen3-8B, thinking on)

Run: `results/horizon_v3_think/exp0_Qwen3-8B_think-on`, snapshot `v3_think`,
2026-09-30 on the 4090. 116/116 prompts kept, 48 force-closed at 3072 think
tokens (median think length 2858). Code hash `0782decd493d`. Post-analysis
numbers from `analysis/think_probe.py` and the manifest.

## 1. The horizon signal dissolves per token, persists on average, and re-forms at the close

| where | fresh probe R² (best layer) | suffix-fit probe, rho with horizon |
|---|---|---|
| suffix `\n` after `<|im_end|>` | 1.000 (L11) | 1.00 |
| think@0.25 / 0.5 / 0.75 | 0.46 / 0.31 / 0.09 | 0.12 / 0.07 / 0.18 |
| think@1 (last think token) | 0.71 (L29); 0.87 non-forced | 0.13 |
| `</think>` | 0.92 (L19) | 0.60 |
| `\n\n` after the close | 0.97 (L33) | -0.15 |
| prefill `I choose :` | 0.96 (L24-35) | 0.4-0.7 |

- **Single tokens inside the think block do not carry the horizon** in any
  linearly readable way, at any layer.
- **The block as a whole does, weakly.** The dense track (suffix direction,
  L11, every think token) averaged over each prompt's block correlates with
  the true horizon at rho 0.89, but with slope 0.10: the ordering survives,
  the amplitude is compressed tenfold (within-prompt std 1.13 vs
  between-prompt std 0.26 in log10 years). Binned by fraction of the block,
  rho falls from 0.90 (first tenth) to 0.73 (last tenth).
- **A better direction exists in deeper layers.** A probe fit on the mean of
  the four think columns reaches R² 0.64 at L24 (0.60-0.65 across L19-35),
  and its direction is nearly orthogonal to the suffix direction
  (cosine 0.09-0.15). The suffix probe reads the think mean at |rho| up to
  0.62 at L30 but only ~0.2-0.3 at L11. So the dense track at L11 is the
  wrong instrument for the think block; the next box run should take the
  dense pass at L24-30 with the think-fit direction.
- **At the close the horizon is re-encoded**, in deeper layers (L19-35) and
  not along the suffix direction.

## 2. The drift is regression to the mean, not the decision

A commit phrase ("the answer is", "I'll go with", ...) appears in 106/116
think texts, at a median 34% of the block (IQR 20-59%). Around it, in the
200 tokens before vs after:

| | before | after |
|---|---|---|
| rho(reading, horizon) | 0.85 | 0.83 |
| slope vs horizon | 0.113 | 0.099 |
| mean abs error (log10 y) | 1.77 | 1.80 (Wilcoxon p = 0.13) |

Nothing changes at the commit. Per-prompt drift over the block is
uncorrelated with the commit point (rho 0.03) but strongly anticorrelated
with the horizon itself (rho -0.61): long horizons drift down, short ones
up. The suffix-direction reading relaxes toward the population mean over
thousands of tokens, regardless of when the model decides.

## 3. The answer is readable before reasoning and perfectly at its end

Logistic probe for the chosen label, 5-fold CV, majority baseline 0.62:

| position | all prompts | non-forced |
|---|---|---|
| suffix probe position | 0.91 (L28) | 0.88 |
| last think token | 0.91 (L21) | **1.00** (L17-25) |
| token before the label | 1.00 (L25) | 1.00 |
| label token (ceiling) | 1.00 | 1.00 |

For prompts that closed their own think block the decision is fully linear
at the last think token. Forced prompts were cut mid-argument: their label
logit margin is 2.6 vs 6.8 for natural closes. The suffix row says the
choice is 91% predictable before any reasoning is written.

## 4. Behavior with thinking

| | thinking off | thinking on |
|---|---|---|
| temporal reasoning (only near pays in time) | 27/27 | 27/27 |
| order stability (keep choice when options swap) | 47/116 (41%) | 86/116 (74%) |
| label stability (a/b -> 1/2) | 102/116 (88%) | 92/116 (79%) |
| answers "b" | 17 | 44 |

Reasoning removes most of the first-option bias.

## Caveats and next

- Forced closes are the long horizons (22/30 at >= 10 years loop), not the
  short ones as planned; a 6144-token run would tell how many close given
  room, and the think@1 and label numbers are cleanest on natural closes.
- bfloat16 batched greedy on the 4090 is not bit-stable across batch
  composition: 46 forced in one run, 48 in the rerun (two prompts changed).
- The pre-label slot is the prefill colon for all but the lead-token
  prompts; its 1.00 reflects a state that has attended to the whole block.
- Next box run: dense pass at the think-fit layer and direction
  (`dense_track` currently uses the suffix probe at the probe layer), and
  the 6144 budget.
