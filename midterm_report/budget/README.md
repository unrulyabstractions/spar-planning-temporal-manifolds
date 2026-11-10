# Budget estimate for the midterm report (Table 1)

Augusto Nicola, 2026-11-02. This is how the numbers in the midterm report's budget table (`../latex/report.tex`,
Table 1) were computed. Please compare it with your own estimates, and flag any assumption you think is off.
Each number is a single constant at the top of `budget_estimate.py`, so a disagreement can be settled by editing
it and rerunning.

```
python3 budget_estimate.py      # standard library only
```

## Method

Cost of a tier = (number of thinking-on generations it needs) × (GPU seconds per generation) × (price per GPU-hour),
times a slowdown factor for steered traces and an overhead factor, plus API costs. Each factor has a low, mid and high
value, and the table shows mid (low–high). Low uses all the low values together, and high uses all the high ones, so
the ranges are deliberately wide.

## Measured anchors

| Quantity | Value | Source |
|---|---|---|
| Gemma 4 31B-it, thinking on, HF generate, batch 4, 1× A100 80 GB | 384 prompts in 1 h 23 min → **13.0 s / generation** (median 716 tokens, p90 1,014, none hit the 8,192 cap) | `alan/progress-20261017.md` |
| Qwen3.8-27B, thinking on, batch 20, 1× A100 | 384 in 3 h 11 min → **29.8 s / generation** (median 1,354, p90 5,816, 22 hit the cap). Used for Qwen3-32B and Qwen3.5-27B. | same file |
| Qwen3-14B, thinking on, batch 24, one process per GPU | 192 per GPU in 2 h 13 min → **41.6 s / generation** (long traces hold the whole batch) | same file |
| A100 80 GB rental | $0.81–1.36/h observed (Alan's rentals, $1.19/h on ours); 2× A100 once at $1.07/h total | Alan's progress logs, `phrasing_study` runs |

## Assumptions (not measured)

| Assumption | Low / mid / high | Why |
|---|---|---|
| Price per GPU-hour | $0.9 / 1.2 / 1.4 | Range of observed A100 prices |
| Steered traces vs unsteered | 1.0 / 1.3 / 1.8× | Steering may lengthen traces or push more of them to the cap; one runaway trace holds its batch |
| Overhead | 1.2 / 1.4 / 1.7× | Setup, weight download, idle time, reruns and failed runs |
| Trace length by tier | T0 1×, T1–T2 1.5×, T3 2.5× | Realistic and open-ended prompts should produce longer reasoning and answers |
| Qwen3-8B | 15 s / generation | Not measured |
| API (prompt generation and checking, coherence judge) | $5–60 per tier | Small next to GPU time; judge model prices not checked |

## Generation counts (the biggest design lever)

These are my reading of the report's appendix A, which does not state them. Settings × prompts dominates every tier.

- **Tier 1 (Gemma 4 31B):** the 500-prompt extraction bank, then a gate and calibration sweep on 200 verification
  prompts × 69 settings = 13,800 generations. The 69 settings are: unsteered, CAA at 12 strengths, the manifold at
  12 shifts, the reward contrast at 4, and 20 random directions at 2 strengths each. Checks (generated-only
  steering, reward indifference, prompting baseline) add 1,700, and the H2 extras 2,000.
- **Tier 2:** T0 template variants, 5 settings × 1,000 = 5,000. T1–T2: about 400 items, including the 4/8-option
  versions, × 2 option orders = 1,200 prompts × 22 settings = 26,400. The 22 settings are: unsteered, prompting
  baseline, horizon rewrite (R), CAA ×4, manifold ×4, reward, and 10 random directions.
- **Tier 3:** T3, 200 items × 15 settings. Qwen3-32B replication: the same gate (14,300), plus T1–T2 with 5 random
  directions (19,200).
- **Tier 4 (rough):** a gate with 10 random directions on Qwen3-8B, Qwen3-14B and Qwen3.5-27B (7,800 each);
  multi-turn "are you sure?" at 8,000; durations outside planning at 2,000.

## Output

| Tier | Mid | Range | Cumulative (mid, range) |
|---|---|---|---|
| 1. Core | $157 | $75–308 | $157 ($75–308) |
| 2. Transfer | +$382 | $184–750 | $539 ($259–1,058) |
| 3. Full scope | +$868 | $425–1,704 | $1,407 ($684–2,763) |
| 4. Stretch | +$539 | $262–1,059 | $1,946 ($945–3,821) |

Against the confirmed $800, Tier 1 fits even at the high end, and Tiers 1–2 fit at the mid estimate. Tier 3 does not
fit as designed. Most of it is the Qwen3-32B replication, whose long traces cost about 2.3× Gemma's per prompt.

## Things that would change the numbers most

1. **Batching.** All anchors use HF `generate` with fixed batches, where the longest trace sets each batch's time.
   Continuous batching, or a GPU with more memory than the 80 GB that holds Gemma's 62.6 GB weights at batch 4,
   should cut GPU time substantially. Not measured.
2. **Fewer settings.** Random directions on every realistic prompt are the largest single item in Tier 2. A
   random-direction null on a subset would cut it.
3. **Replication scope.** Replicating only the T0 gate on Qwen3-32B would remove about $520 of Tier 3 (mid).
4. **Token cap.** Gemma never exceeded 5,700 tokens in the pilot, so a lower cap than 8,192 limits runaway traces.
