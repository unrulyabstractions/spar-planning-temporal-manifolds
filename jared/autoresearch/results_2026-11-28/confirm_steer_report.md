# Report card (replay readout)

- Verification items readable: 178 (unreadable: 22, listed in skipped.jsonl)
- Push strength x = 0.4 × typical norm; target shift = 1.0 decades (fixed by screen.py select: highest S (peak-rule order-balanced mean odd effect, logits toward far) among methods other than reward and choice whose chosen strength is healthy; S within 0.25 logits of the best: the smallest strength, compared by position in its own grid, then the higher S)
- Write span: from_horizon; TF32: off; replay weights: float32
- N0: zero edit vs hook-free max |Δy| = 9.5367431640625e-05; unsteered replay matches the generated choice on 99.4%
- Unsteered switch point log10(H/D) = +0.381, slope 3.43, label mass 0.999

## Methods, controls and comparisons

95% intervals: paired bootstrap over verification pairs (1000 resamples).

| arm | x | meaning (logit) | letter | even | Δlog10 H* | slope ± | low/high plateau (+x) | label mass ± |
|---|---|---|---|---|---|---|---|---|
| caa | 0.4 | +1.252 [+1.043, +1.460] | +0.634 [+0.437, +0.830] | -1.778 [-2.042, -1.516] | -0.000 [-0.022, +0.007] | 3.49 / 3.90 | 0.03 / 0.32 | 1.000 / 0.995 |

## Against the nulls (rank p; |null| 95th percentile)

| method | null family | K | meaning: stat / p95 / p | Δlog10 H*: stat / p95 / p |
|---|---|---|---|---|
| caa | covariance | 8 | +1.252 / 1.447 / 0.22 | -0.000 / 0.005 / 1.00 |
