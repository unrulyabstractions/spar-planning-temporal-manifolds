# Report card (replay readout)

- Verification items readable: 178 (unreadable: 22, listed in skipped.jsonl)
- Push strength x = 0.4 × typical norm; target shift = 1.0 decades (fixed by screen.py select: highest S (peak-rule order-balanced mean odd effect, logits toward far) among methods other than reward and choice whose chosen strength is healthy; S within 0.25 logits of the best: the smallest strength, compared by position in its own grid, then the higher S)
- Write span: all; TF32: off; replay weights: float32
- N0: zero edit vs hook-free max |Δy| = 8.58306884765625e-05; unsteered replay matches the generated choice on 99.4%
- Unsteered switch point log10(H/D) = +0.381, slope 3.43, label mass 0.999

## Methods, controls and comparisons

95% intervals: paired bootstrap over verification pairs (1000 resamples).

| arm | x | meaning (logit) | letter | even | Δlog10 H* | slope ± | low/high plateau (+x) | label mass ± |
|---|---|---|---|---|---|---|---|---|
| caa | 0.4 | +6.812 [+6.396, +7.187] | +0.670 [+0.463, +0.887] | +0.356 [-0.232, +0.890] | +0.804 [+0.452, +1.967] | 0.69 / 3.10 | 0.59 / 0.63 | 0.981 / 0.999 |

## Against the nulls (rank p; |null| 95th percentile)

| method | null family | K | meaning: stat / p95 / p | Δlog10 H*: stat / p95 / p |
|---|---|---|---|---|
| caa | covariance | 19 | +6.812 / 4.087 / 0.05 | +0.804 / 0.489 / 0.05 |

Plumbing run: the model is small and the numbers are not evidence about steering.
