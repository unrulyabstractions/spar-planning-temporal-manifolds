# Unit-boundary analysis: google/gemma-4-31B-it think (pilot, A100)

Run `runs/gemma-4-31b_think_pilot`; 384 of 384 prompts with a parsed choice (format adherence 1.000).
Switch point = H/D at P(short) = 0.5 (logistic fit on ln(H/D)); 95% CI: 1000 bootstrap resamples of prompts.
Shift = ln(switch) - ln(switch of the matched cell); negative = the model takes the long option at shorter horizons.
A switch outside the sampled H/D range is shown as <min or >max (censored, not extrapolated); a shift involving one
is ±inf. † = more than 5% of bootstrap resamples gave no defined value; treat that interval as unreliable.

## Switch point H/D by horizon unit (rows) and delay unit (columns), pooled

| horizon \ delays | weeks | months | quarters | years |
|---|---|---|---|---|
| weeks | **1.018 [0.974, 1.061]** | 1.018 [0.975, 1.064] | 1.018 [0.975, 1.059] | 1.018 [0.972, 1.066] |
| months | 1.020 [0.976, 1.063] | **1.020 [0.978, 1.062]** | 1.020 [0.978, 1.067] | 1.020 [0.978, 1.066] |
| quarters | 1.017 [0.972, 1.062] | 1.018 [0.973, 1.064] | **1.018 [0.974, 1.062]** | 1.018 [0.976, 1.065] |
| years | 1.017 [0.972, 1.061] | 1.018 [0.974, 1.061] | 1.018 [0.975, 1.061] | **1.018 [0.975, 1.065]** |

Error rates (below: long chosen though H < D / above: short chosen though H > D):

| horizon \ delays | weeks | months | quarters | years |
|---|---|---|---|---|
| weeks | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| months | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| quarters | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| years | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |

## Shift vs matched units (ln units; 95% CI)

Same horizon unit, delays changed (controls the horizon's rendering) | same delay unit, horizon changed:

| horizon unit | delay unit | shift vs (H, delays in H's unit) | shift vs (horizon in delays' unit, D) |
|---|---|---|---|
| weeks | months | +0.001 [-0.063, +0.063] | -0.002 [-0.069, +0.066] |
| weeks | quarters | +0.001 [-0.058, +0.065] | +0.000 [-0.064, +0.063] |
| weeks | years | +0.001 [-0.067, +0.058] | +0.000 [-0.056, +0.061] |
| months | weeks | -0.000 [-0.063, +0.055] | +0.002 [-0.062, +0.070] |
| months | quarters | -0.000 [-0.054, +0.059] | +0.002 [-0.060, +0.060] |
| months | years | -0.000 [-0.064, +0.068] | +0.002 [-0.061, +0.053] |
| quarters | weeks | -0.000 [-0.068, +0.068] | -0.000 [-0.069, +0.070] |
| quarters | months | +0.000 [-0.055, +0.053] | -0.002 [-0.055, +0.057] |
| quarters | years | +0.000 [-0.056, +0.060] | -0.000 [-0.066, +0.057] |
| years | weeks | -0.001 [-0.067, +0.066] | -0.000 [-0.072, +0.064] |
| years | months | +0.000 [-0.060, +0.060] | -0.002 [-0.060, +0.055] |
| years | quarters | -0.000 [-0.059, +0.060] | +0.000 [-0.058, +0.064] |

## Matched vs mismatched by factor (pooled over unit pairs)

| factor | level | switch matched | switch mismatched | shift (ln) | errors below, matched / mismatched |
|---|---|---|---|---|---|
| D | 10 | 1.018 | 1.018 | +0.000 [-0.006, +0.006] | 0.000 / 0.000 |
| ratio | 5 | 1.018 | 1.018 | +0.000 [-0.006, +0.006] | 0.000 / 0.000 |
| order | long first | 1.018 | 1.019 | +0.000 [-0.013, +0.012] | 0.000 / 0.000 |
| order | short first | 1.018 | 1.019 | +0.000 [-0.015, +0.012] | 0.000 / 0.000 |
| phrasing_id | 0 | 1.018 | 1.018 | +0.000 [-0.006, +0.007] | 0.000 / 0.000 |
| hu | months | 1.020 | 1.020 | -0.000 [-0.044, +0.044] | 0.000 / 0.000 |
| hu | quarters | 1.018 | 1.018 | -0.000 [-0.046, +0.044] | 0.000 / 0.000 |
| hu | weeks | 1.018 | 1.018 | +0.000 [-0.043, +0.044] | 0.000 / 0.000 |
| hu | years | 1.018 | 1.018 | -0.000 [-0.046, +0.044] | 0.000 / 0.000 |
| du | months | 1.020 | 1.018 | -0.002 [-0.045, +0.042] | 0.000 / 0.000 |
| du | quarters | 1.018 | 1.019 | +0.001 [-0.045, +0.046] | 0.000 / 0.000 |
| du | weeks | 1.018 | 1.018 | +0.000 [-0.044, +0.043] | 0.000 / 0.000 |
| du | years | 1.018 | 1.019 | +0.001 [-0.049, +0.044] | 0.000 / 0.000 |

## Continuity check: the 2026-10-03 conditions (months/years, D = 10 y, ratio 5, short first, phrasing 0)

| horizon | delays | n | switch H/D | switch H (years) |
|---|---|---|---|---|
| months | years | 12 | 1.020 | 10.20 |
| months | months | 12 | 1.020 | 10.20 |
| years | years | 12 | 1.018 | 10.18 |
| years | months | 12 | 1.018 | 10.18 |

## Option order

P(choose a)) = 0.500 overall; short first 0.500, long first 0.500.

P(short) by H/D bin, short first (rows: horizon unit / delay unit):

| H / delays | (0.66, 0.75] | (0.75, 0.85] | (0.85, 0.95] | (0.95, 1.0] | (1.05, 1.2] | (1.2, 1.51] |
|---|---|---|---|---|---|---|
| weeks / weeks (match) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| weeks / months | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| weeks / quarters | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| weeks / years | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| months / weeks | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| months / months (match) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| months / quarters | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| months / years | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| quarters / weeks | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| quarters / months | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| quarters / quarters (match) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| quarters / years | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| years / weeks | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| years / months | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| years / quarters | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| years / years (match) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |

P(short) by H/D bin, long first (rows: horizon unit / delay unit):

| H / delays | (0.66, 0.75] | (0.75, 0.85] | (0.85, 0.95] | (0.95, 1.0] | (1.05, 1.2] | (1.2, 1.51] |
|---|---|---|---|---|---|---|
| weeks / weeks (match) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| weeks / months | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| weeks / quarters | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| weeks / years | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| months / weeks | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| months / months (match) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| months / quarters | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| months / years | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| quarters / weeks | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| quarters / months | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| quarters / quarters (match) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| quarters / years | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| years / weeks | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| years / months | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| years / quarters | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| years / years (match) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |

P(short) at H/D < 0.85 by phrasing and order:

| phrasing | short first | long first |
|---|---|---|
| 0 | 1.00 | 1.00 |
