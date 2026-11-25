# Transfer to the eval types (replay readout)

- Strength frozen on T0 (/workspace/SPAR/spar-planning-temporal-manifolds/evaluation/results/gemma-4-31B-it__t0_s0/calibration.json): push x = 0.1 × typical norm, target shift = 1 decades
- Groups: `prompts` = the type's own prompts (base and swap), `twins` = the same options in the T0 template with the horizon stated. Only near_only / both prompts are run.
- meaning: order-balanced effect on the base/swap pairs (40 per type); odd: mean odd part over all items; Δlog10 H*: switch-point shift, only where ≥ 5 readable items have each correct option.
- unsteered acc: share of items whose unsteered replay prefers the correct option.
- Check set `quick_system_prompt`: Quickest transfer check, 40 prompts: the eval type most expected to work (system_prompt), 10 near and 10 both items with their swaps, no twins (the T0 replay is the positive control). CAA and curve-snap against the covariance-matched null family, with the prompting arms beside them.
  - system_prompt: expected to work (ranked 1st on Qwen3-8B: phrase donor flips 0.90, answer-slot patch 0.48, stacked steer 0.38)

## Summary

| bank | group | n (unreadable) | unsteered mass / acc | method | meaning | odd | Δlog10 H* | label mass ± | max p over nulls (meaning / odd) |
|---|---|---|---|---|---|---|---|---|---|
| T0 verification grid | grid | 200 (0) | 1.000 / 0.80 | caa | +1.539 | +1.539 | +0.007 | 1.000 / 1.000 | 0.30 / 0.30 |
| T0 verification grid | grid | 200 (0) | 1.000 / 0.80 | curve-snap | -1.045 | -1.045 | +1.080 | 1.000 / 1.000 | – / – |
| system_prompt (expected to work (ranked 1st on Qwen3-8B: phrase donor flips 0.90, answer-slot patch 0.48, stacked steer 0.38)) | prompts | 40 (0) | 1.000 / 1.00 | caa | +0.744 | +0.744 | +0.001 | 1.000 / 1.000 | 0.35 / 0.35 |
| system_prompt (expected to work (ranked 1st on Qwen3-8B: phrase donor flips 0.90, answer-slot patch 0.48, stacked steer 0.38)) | prompts | 40 (0) | 1.000 / 1.00 | curve-snap | -0.578 | -0.578 | +0.831 | 1.000 / 1.000 | – / – |
| system_prompt (expected to work (ranked 1st on Qwen3-8B: phrase donor flips 0.90, answer-slot patch 0.48, stacked steer 0.38)) | prompts | 40 (0) | 1.000 / 1.00 | prompt:disposition | +0.306 | +0.306 | +0.000 | 1.000 / 1.000 | – / – |
| system_prompt (expected to work (ranked 1st on Qwen3-8B: phrase donor flips 0.90, answer-slot patch 0.48, stacked steer 0.38)) | prompts | 40 (0) | 1.000 / 1.00 | prompt:shift | +0.398 | +0.398 | +0.000 | 1.000 / 1.000 | – / – |
| system_prompt (expected to work (ranked 1st on Qwen3-8B: phrase donor flips 0.90, answer-slot patch 0.48, stacked steer 0.38)) | prompts | 40 (0) | 1.000 / 1.00 | prompt:neutral | -0.042 | -0.042 | -0.000 | 1.000 / – | – / – |

## Against the nulls (rank p; |null| 95th percentile)

| bank | group | method | null family | K | meaning: stat / p95 / p | odd: stat / p95 / p | Δlog10 H*: stat / p95 / p |
|---|---|---|---|---|---|---|---|
| T0 verification grid | grid | caa | covariance | 19 | +1.539 / 3.803 / 0.30 | +1.539 / 3.803 / 0.30 | +0.007 / 0.331 / 0.75 |
| system_prompt (expected to work (ranked 1st on Qwen3-8B: phrase donor flips 0.90, answer-slot patch 0.48, stacked steer 0.38)) | prompts | caa | covariance | 19 | +0.744 / 1.906 / 0.35 | +0.744 / 1.906 / 0.35 | +0.001 / 0.599 / 0.75 |
