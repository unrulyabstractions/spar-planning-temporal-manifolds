# Transfer to the eval types (replay readout)

- Strength frozen on T0 (/workspace/SPAR/spar-planning-temporal-manifolds/evaluation/results/Qwen3-8B__t0_s0__think/calibration.json): push x = 0.8 × typical norm, target shift = 1 decades
- Groups: `prompts` = the type's own prompts (base and swap), `twins` = the same options in the T0 template with the horizon stated. Only near_only / both prompts are run.
- meaning: order-balanced effect on the base/swap pairs (40 per type); odd: mean odd part over all items; Δlog10 H*: switch-point shift, only where ≥ 5 readable items have each correct option.
- unsteered acc: share of items whose unsteered replay prefers the correct option.
- Check set `quick_system_prompt`: Quickest transfer check, 40 prompts: the eval type most expected to work (system_prompt), 10 near and 10 both items with their swaps, no twins (the T0 replay is the positive control). CAA and curve-snap against the covariance-matched null family, with the prompting arms beside them.
  - system_prompt: expected to work (ranked 1st on Qwen3-8B: phrase donor flips 0.90, answer-slot patch 0.48, stacked steer 0.38)

## Summary

| bank | group | n (unreadable) | unsteered mass / acc | method | meaning | odd | Δlog10 H* | label mass ± | max p over nulls (meaning / odd) |
|---|---|---|---|---|---|---|---|---|---|
| T0 verification grid | grid | 178 (22) | 0.999 / 0.71 | caa | +0.217 | +0.263 | +0.651 | 0.997 / 0.998 | 0.45 / 0.30 |
| T0 verification grid | grid | 178 (22) | 0.999 / 0.71 | curve-snap | -0.352 | -0.356 | +0.053 | 0.996 / 0.997 | – / – |
| system_prompt (expected to work (ranked 1st on Qwen3-8B: phrase donor flips 0.90, answer-slot patch 0.48, stacked steer 0.38)) | prompts | 38 (2) | 0.998 / 0.79 | caa | +0.183 | +0.207 | +0.087 | 0.994 / 0.999 | 0.65 / 0.60 |
| system_prompt (expected to work (ranked 1st on Qwen3-8B: phrase donor flips 0.90, answer-slot patch 0.48, stacked steer 0.38)) | prompts | 38 (2) | 0.998 / 0.79 | curve-snap | -0.295 | -0.298 | +0.133 | 0.993 / 0.995 | – / – |
| system_prompt (expected to work (ranked 1st on Qwen3-8B: phrase donor flips 0.90, answer-slot patch 0.48, stacked steer 0.38)) | prompts | 38 (2) | 0.998 / 0.79 | prompt:disposition | +1.682 | +1.660 | +0.045 | 0.998 / 0.999 | – / – |
| system_prompt (expected to work (ranked 1st on Qwen3-8B: phrase donor flips 0.90, answer-slot patch 0.48, stacked steer 0.38)) | prompts | 38 (2) | 0.998 / 0.79 | prompt:shift | +1.174 | +1.146 | +0.045 | 0.995 / 0.998 | – / – |
| system_prompt (expected to work (ranked 1st on Qwen3-8B: phrase donor flips 0.90, answer-slot patch 0.48, stacked steer 0.38)) | prompts | 38 (2) | 0.998 / 0.79 | prompt:neutral | +0.344 | +0.299 | -0.013 | 0.998 / – | – / – |

## Against the nulls (rank p; |null| 95th percentile)

| bank | group | method | null family | K | meaning: stat / p95 / p | odd: stat / p95 / p | Δlog10 H*: stat / p95 / p |
|---|---|---|---|---|---|---|---|
| T0 verification grid | grid | caa | covariance | 19 | +0.217 / 0.901 / 0.45 | +0.263 / 0.869 / 0.30 | +0.651 / 1.439 / 0.15 |
| system_prompt (expected to work (ranked 1st on Qwen3-8B: phrase donor flips 0.90, answer-slot patch 0.48, stacked steer 0.38)) | prompts | caa | covariance | 19 | +0.183 / 0.896 / 0.65 | +0.207 / 0.831 / 0.60 | +0.087 / 0.883 / 0.40 |
