# Report card (replay readout)

- Verification items readable: 178 (unreadable: 22, listed in skipped.jsonl)
- Push strength x = 0.8 × typical norm (calibrated on held-out fold 0, target 1.0 logit, reached: True); target shift = 1 decades
- N0: zero edit vs hook-free max |Δy| = 0.0; unsteered replay matches the generated choice on 99.4%
- Unsteered switch point log10(H/D) = +0.382, slope 3.42, label mass 0.999

## Methods and controls

| arm | x | meaning (logit) | letter | even | Δlog10 H* | slope ± | low/high plateau (+x) | label mass ± |
|---|---|---|---|---|---|---|---|---|
| caa | 0.8 | +0.217 | +9.797 | +3.942 | +0.651 | 0.54 / 0.01 | 0.48 / 0.55 | 0.997 / 0.998 |
| curve-snap | 1 | -0.352 | -0.448 | +3.460 | +0.053 | 1.84 / 1.06 | 0.22 / 0.44 | 0.996 / 0.997 |
| curve-snap@0 | 0 | +2.112 | -0.661 | +0.000 | +0.027 | 3.00 / nan | 0.07 / 0.35 | 0.997 / nan |

## Against the nulls (rank p; |null| 95th percentile)

| method | null family | K | meaning: stat / p95 / p | Δlog10 H*: stat / p95 / p |
|---|---|---|---|---|
| caa | covariance | 19 | +0.217 / 0.901 / 0.45 | +0.651 / 1.439 / 0.15 |

## Generation check (subset)

| arm | samples | parsed | P(far) | hit cap | mean tokens |
|---|---|---|---|---|---|
| caa+ | 1 | 1/16 | 0.0 | 15 | 3896 |
| caa- | 1 | 4/16 | 0.5 | 11 | 3296 |
| none | 1 | 15/16 | 0.07 | 1 | 1968 |

Plumbing run: the model is small and the numbers are not evidence about steering.
