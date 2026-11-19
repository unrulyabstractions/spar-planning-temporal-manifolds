# Report card (replay readout)

- Verification items readable: 79 (unreadable: 121, listed in skipped.jsonl)
- Push strength x = 0.02 × typical norm (calibrated on held-out fold 0, target 1.0 logit, reached: False); target shift = 1 decades
- N0: zero edit vs hook-free max |Δy| = 0.0; unsteered replay matches the generated choice on 100.0%
- Unsteered switch point log10(H/D) = -0.272, slope 4.83, label mass 0.903

## Methods and controls

| arm | x | meaning (logit) | letter | even | Δlog10 H* | slope ± | low/high plateau (+x) | label mass ± |
|---|---|---|---|---|---|---|---|---|
| caa | 0.02 | -0.031 | -0.268 | +0.009 | -0.002 | 4.86 / 4.78 | 0.41 / 1.00 | 0.864 / 0.936 |
| reward | 0.02 | +0.018 | -0.621 | -0.025 | -0.006 | 4.90 / 4.71 | 0.41 / 1.00 | 0.870 / 0.937 |
| choice | 0.02 | +0.464 | +0.299 | -0.017 | +0.007 | 4.70 / 4.90 | 0.43 / 1.00 | 0.905 / 0.901 |
| curve-displace | 1 | -0.205 | +0.232 | -0.684 | -0.012 | 4.69 / 4.53 | 0.42 / 1.00 | 0.945 / 0.890 |
| curve-snap | 1 | -0.464 | -1.402 | -5.101 | +0.030 | 4.24 / 4.49 | 0.48 / 0.99 | 0.635 / 0.669 |
| curve-snap@0 | 0 | -2.317 | +0.576 | +0.000 | -0.011 | 4.96 / nan | 0.39 / 1.00 | 0.712 / nan |

## Against the nulls (rank p; |null| 95th percentile)

| method | null family | K | meaning: stat / p95 / p | Δlog10 H*: stat / p95 / p |
|---|---|---|---|---|
| caa | isotropic | 19 | -0.031 / 0.081 / 1.00 | -0.002 / 0.003 / 1.00 |
| caa | covariance | 19 | -0.031 / 0.213 / 1.00 | -0.002 / 0.008 / 1.00 |
| caa | real_diff | 19 | -0.031 / 0.197 / 1.00 | -0.002 / 0.008 / 1.00 |
| caa | in_span | 19 | -0.031 / 0.179 / 1.00 | -0.002 / 0.009 / 1.00 |
| caa | shuffled:caa | 19 | -0.031 / 0.115 / 1.00 | -0.002 / 0.005 / 1.00 |
| reward | isotropic | 19 | +0.018 / 0.081 / 0.95 | -0.006 / 0.003 / 1.00 |
| reward | covariance | 19 | +0.018 / 0.213 / 0.95 | -0.006 / 0.008 / 1.00 |
| reward | real_diff | 19 | +0.018 / 0.197 / 1.00 | -0.006 / 0.008 / 1.00 |
| reward | in_span | 19 | +0.018 / 0.179 / 0.90 | -0.006 / 0.009 / 1.00 |
| choice | isotropic | 19 | +0.464 / 0.081 / 0.05 | +0.007 / 0.003 / 0.05 |
| choice | covariance | 19 | +0.464 / 0.213 / 0.05 | +0.007 / 0.008 / 0.15 |
| choice | real_diff | 19 | +0.464 / 0.197 / 0.05 | +0.007 / 0.008 / 0.15 |
| choice | in_span | 19 | +0.464 / 0.179 / 0.05 | +0.007 / 0.009 / 0.25 |
| curve-displace | shuffled:curve-displace | 19 | -0.205 / 0.557 / 1.00 | -0.012 / 0.019 / 1.00 |
| curve-snap | shuffled:curve-snap | 19 | -0.464 / 0.851 / 1.00 | +0.030 / 0.079 / 0.50 |

## Generation check (subset)

| arm | parsed | P(far) | hit cap | mean tokens |
|---|---|---|---|---|
| caa+ | 12/16 | 0.33 | 4 | 2154 |
| caa- | 14/16 | 0.21 | 2 | 2017 |
| curve-displace+ | 11/16 | 0.18 | 6 | 2304 |
| curve-displace- | 11/16 | 0.36 | 5 | 2466 |
| isotropic[0]+ | 12/16 | 0.17 | 5 | 2318 |
| isotropic[0]- | 11/16 | 0.27 | 5 | 2300 |
| none | 13/16 | 0.31 | 4 | 2123 |

Plumbing run: the model is small and the numbers are not evidence about steering.
