# Unit-boundary analysis: Qwen/Qwen3.8-27B think (pilot, A100)

Run `runs/qwen3.8-27b_think_pilot`; 362 of 384 prompts with a parsed choice (format adherence 0.943).
Switch point = H/D at P(short) = 0.5 (logistic fit on ln(H/D)); 95% CI: 1000 bootstrap resamples of prompts.
Shift = ln(switch) - ln(switch of the matched cell); negative = the model takes the long option at shorter horizons.
A switch outside the sampled H/D range is shown as <min or >max (censored, not extrapolated); a shift involving one
is ±inf. † = more than 5% of bootstrap resamples gave no defined value; treat that interval as unreliable.

## Switch point H/D by horizon unit (rows) and delay unit (columns), pooled

| horizon \ delays | weeks | months | quarters | years |
|---|---|---|---|---|
| weeks | **1.054 [0.982, 1.182]** | 1.018 [0.975, 1.064] | >1.501 [1.057, >1.501] | 1.054 [0.983, 1.158] |
| months | 1.020 [0.977, 1.066] | **1.056 [0.994, 1.180]** | 1.111 [0.949, 1.485] | 1.020 [0.975, 1.066] |
| quarters | 1.090 [0.986, 1.336] | 0.983 [0.922, 1.043] | **1.184 [1.015, >1.501]** | 1.018 [0.973, 1.061] |
| years | 1.012 [0.955, 1.061] | 0.982 [0.915, 1.039] | 1.173 [1.013, 1.456] | **0.981 [0.888, 1.117]** |

Error rates (below: long chosen though H < D / above: short chosen though H > D):

| horizon \ delays | weeks | months | quarters | years |
|---|---|---|---|---|
| weeks | 0.000 / 0.083 | 0.000 / 0.000 | 0.000 / 0.571 | 0.000 / 0.083 |
| months | 0.000 / 0.000 | 0.000 / 0.083 | 0.083 / 0.222 | 0.000 / 0.000 |
| quarters | 0.000 / 0.111 | 0.083 / 0.000 | 0.000 / 0.375 | 0.000 / 0.000 |
| years | 0.000 / 0.000 | 0.091 / 0.000 | 0.000 / 0.250 | 0.167 / 0.083 |

## Shift vs matched units (ln units; 95% CI)

Same horizon unit, delays changed (controls the horizon's rendering) | same delay unit, horizon changed:

| horizon unit | delay unit | shift vs (H, delays in H's unit) | shift vs (horizon in delays' unit, D) |
|---|---|---|---|
| weeks | months | -0.035 [-0.161, +0.044] | -0.037 [-0.151, +0.040] |
| weeks | quarters | +inf [+0.005, +inf] | +inf [-0.175, +inf] |
| weeks | years | +0.000 [-0.150, +0.125] | +0.072 [-0.087, +0.219] |
| months | weeks | -0.035 [-0.150, +0.044] | -0.033 [-0.145, +0.044] |
| months | quarters | +0.050 [-0.140, +0.356] | -0.064 [-inf, +0.304] |
| months | years | -0.035 [-0.144, +0.051] | +0.039 [-0.106, +0.146] |
| quarters | weeks | -0.083 [-inf, +0.152] | +0.034 [-0.120, +0.254] |
| quarters | months | -0.187 [-inf, -0.030] | -0.072 [-0.200, +0.011] |
| quarters | years | -0.151 [-0.352, -0.003] | +0.036 [-0.119, +0.148] |
| years | weeks | +0.031 [-0.122, +0.133] | -0.040 [-0.170, +0.035] |
| years | months | +0.001 [-0.155, +0.119] | -0.073 [-0.201, +0.023] |
| years | quarters | +0.179 [-0.019, +0.443] | -0.009 [-0.340, +0.285] |

## Matched vs mismatched by factor (pooled over unit pairs)

| factor | level | switch matched | switch mismatched | shift (ln) | errors below, matched / mismatched |
|---|---|---|---|---|---|
| D | 10 | 1.059 | 1.046 | -0.013 [-0.080, +0.046] | 0.042 / 0.021 |
| ratio | 5 | 1.059 | 1.046 | -0.013 [-0.076, +0.046] | 0.042 / 0.021 |
| order | long first | 1.044 | 1.016 | -0.027 [-0.133, +0.058] | 0.083 / 0.029 |
| order | short first | 1.074 | 1.081 | +0.007 [-0.088, +0.086] | 0.000 / 0.014 |
| phrasing_id | 0 | 1.059 | 1.046 | -0.013 [-0.087, +0.045] | 0.042 / 0.021 |
| hu | months | 1.056 | 1.036 | -0.019 [-0.131, +0.083] | 0.000 / 0.028 |
| hu | quarters | 1.184 | 1.023 | -0.147 [-inf, +0.003] | 0.000 / 0.028 |
| hu | weeks | 1.054 | 1.095 | +0.038 [-0.087, +0.142] | 0.000 / 0.000 |
| hu | years | 0.981 | 1.032 | +0.050 [-0.100, +0.164] | 0.167 / 0.030 |
| du | months | 1.056 | 0.992 | -0.063 [-0.174, +0.009] | 0.000 / 0.057 |
| du | quarters | 1.184 | 1.216 | +0.026 [-inf, +0.247] | 0.000 / 0.029 |
| du | weeks | 1.054 | 1.032 | -0.021 [-0.134, +0.068] | 0.000 / 0.000 |
| du | years | 0.981 | 1.031 | +0.049 [-0.109, +0.158] | 0.167 / 0.000 |

## Continuity check: the 2026-10-03 conditions (months/years, D = 10 y, ratio 5, short first, phrasing 0)

| horizon | delays | n | switch H/D | switch H (years) |
|---|---|---|---|---|
| months | years | 12 | 1.020 | 10.20 |
| months | months | 12 | 1.020 | 10.20 |
| years | years | 12 | 1.018 | 10.18 |
| years | months | 12 | 1.018 | 10.18 |

## Option order

P(choose a)) = 0.530 overall; short first 0.583, long first 0.478.

P(short) by H/D bin, short first (rows: horizon unit / delay unit):

| H / delays | (0.66, 0.75] | (0.75, 0.85] | (0.85, 0.95] | (0.95, 1.0] | (1.05, 1.2] | (1.2, 1.51] |
|---|---|---|---|---|---|---|
| weeks / weeks (match) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| weeks / months | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| weeks / quarters | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |
| weeks / years | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| months / weeks | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| months / months (match) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| months / quarters | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.67 |
| months / years | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| quarters / weeks | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.50 |
| quarters / months | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| quarters / quarters (match) | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.67 |
| quarters / years | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| years / weeks | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| years / months | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| years / quarters | 1.00 | 1.00 | 1.00 | 1.00 | nan | 0.67 |
| years / years (match) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |

P(short) by H/D bin, long first (rows: horizon unit / delay unit):

| H / delays | (0.66, 0.75] | (0.75, 0.85] | (0.85, 0.95] | (0.95, 1.0] | (1.05, 1.2] | (1.2, 1.51] |
|---|---|---|---|---|---|---|
| weeks / weeks (match) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.25 |
| weeks / months | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| weeks / quarters | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |
| weeks / years | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.00 |
| months / weeks | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| months / months (match) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.25 |
| months / quarters | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| months / years | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| quarters / weeks | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| quarters / months | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| quarters / quarters (match) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| quarters / years | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| years / weeks | 1.00 | 1.00 | 1.00 | nan | 0.00 | 0.00 |
| years / months | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| years / quarters | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| years / years (match) | 1.00 | 1.00 | 0.50 | 0.00 | 0.00 | 0.25 |

P(short) at H/D < 0.85 by phrasing and order:

| phrasing | short first | long first |
|---|---|---|
| 0 | 1.00 | 1.00 |
