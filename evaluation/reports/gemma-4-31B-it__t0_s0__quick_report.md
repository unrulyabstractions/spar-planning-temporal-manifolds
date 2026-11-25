# Report card (replay readout)

- Verification items readable: 200 (unreadable: 0, listed in skipped.jsonl)
- Push strength x = 0.1 × typical norm (calibrated on held-out fold 0, target 1.0 logit, reached: True); target shift = 1 decades
- N0: zero edit vs hook-free max |Δy| = 0.1328125; unsteered replay matches the generated choice on 100.0%
- Unsteered switch point log10(H/D) = +0.190, slope 5.11, label mass 1.000

## Methods and controls

| arm | x | meaning (logit) | letter | even | Δlog10 H* | slope ± | low/high plateau (+x) | label mass ± |
|---|---|---|---|---|---|---|---|---|
| caa | 0.1 | +1.539 | -0.311 | +1.204 | +0.007 | 5.03 / 5.11 | 0.06 / 0.50 | 1.000 / 1.000 |
| curve-snap | 1 | -1.045 | +1.802 | +3.913 | +1.080 | 0.80 / 0.07 | 0.29 / 0.35 | 1.000 / 1.000 |
| curve-snap@0 | 0 | +2.370 | -7.606 | +0.000 | -0.151 | 2.30 / nan | 0.15 / 0.43 | 1.000 / nan |

## Against the nulls (rank p; |null| 95th percentile)

| method | null family | K | meaning: stat / p95 / p | Δlog10 H*: stat / p95 / p |
|---|---|---|---|---|
| caa | covariance | 19 | +1.539 / 3.803 / 0.30 | +0.007 / 0.331 / 0.75 |

## Generation check (subset)

| arm | samples | parsed | P(far) | hit cap | mean tokens |
|---|---|---|---|---|---|
| caa+ | 1 | 15/16 | 0.33 | 1 | 990 |
| caa- | 1 | 16/16 | 0.25 | 0 | 708 |
| none | 1 | 16/16 | 0.25 | 0 | 804 |

Plumbing run: the model is small and the numbers are not evidence about steering.
