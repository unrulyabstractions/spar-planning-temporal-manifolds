# Report card (replay readout)

- Verification items readable: 178 (unreadable: 22, listed in skipped.jsonl)
- Push strength x = None × typical norm; target shift = 1.0 decades (patch confirm: x = ±1 is the source prompt 10x longer / shorter; nothing calibrated)
- Write span: from_horizon; TF32: off; replay weights: float32
- N0: zero edit vs hook-free max |Δy| = 9.5367431640625e-05; unsteered replay matches the generated choice on 99.4%
- Unsteered switch point log10(H/D) = +0.381, slope 3.43, label mass 0.999

## Methods, controls and comparisons

95% intervals: paired bootstrap over verification pairs (1000 resamples).

| arm | x | meaning (logit) | letter | even | Δlog10 H* | slope ± | low/high plateau (+x) | label mass ± |
|---|---|---|---|---|---|---|---|---|
| patch:text | 1 | +0.520 [+0.416, +0.625] | +0.021 [-0.044, +0.088] | +0.160 [-0.090, +0.392] | +0.001 [-0.011, +0.005] | 3.50 / 3.72 | 0.03 / 0.32 | 0.999 / 0.999 |
| patch:full | 1 | +0.681 [+0.559, +0.803] | -0.015 [-0.076, +0.056] | -0.315 [-0.448, -0.187] | +0.001 [-0.013, +0.007] | 3.46 / 3.73 | 0.03 / 0.32 | 0.999 / 0.999 |
| patch:curve-displace | 1 | +0.326 [+0.258, +0.402] | -0.025 [-0.067, +0.015] | +0.058 [+0.006, +0.113] | +0.000 [-0.007, +0.003] | 3.36 / 3.48 | 0.03 / 0.32 | 0.999 / 0.999 |

Patch arms (patching.py) use a source prompt whose horizon is 10x longer (+) or shorter (-): a perfect horizon edit has Δlog10 H* = 1 decade. patch:text rewrites the horizon in the prompt and keeps the reasoning; patch:full replaces the whole state at the patched tokens.

## Against the nulls (rank p; |null| 95th percentile)

| method | null family | K | meaning: stat / p95 / p | Δlog10 H*: stat / p95 / p |
|---|---|---|---|---|
| patch:curve-displace | patch-null:k4 | 9 | +0.326 / 0.228 / 0.10 | +0.000 / 0.000 / 0.60 |
