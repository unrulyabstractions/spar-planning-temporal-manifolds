# Spec: E7, linear and MLP probes on the top-k principal components

Date: 2026-10-03. Decisions from Jared in §5. Companion to `01_plan.md`.

## 1. Question

Is the horizon extractable from the residual stream, and is the information
held on a flat (linear) subspace or a curved, low-dimensional manifold? How
many dimensions describe it?

## 2. Method

For each (layer, position) cell, after the fact, on saved activations (CPU):

- **Target:** log10 horizon in years, regression. No-horizon prompts excluded.
- **Cross-validation:** leave-one-horizon-out. Every prompt at one horizon
  value is held out together, so a probe can only score by placing an unseen
  horizon between its neighbours. Out-of-fold predictions are pooled; R² is
  reported on all prompts and on interior horizons only (the two end values
  are extrapolation).
- **Two orderings of dimensions, both fitted on the training fold only:**
  - `act`: PCA of the activations (ordered by total variance). Answers where
    the horizon ranks among everything the model represents.
  - `cent`: PCA of the per-horizon mean activations (ordered by separation
    between horizons). Answers how many dimensions the horizon manifold
    itself spans. At most (training horizons - 1) components.
- **k grid:** 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, plus all dimensions
  (ridge only) as the ceiling.
- **Probes at each k:** ridge with the penalty chosen by internal CV
  (`RidgeCV`) on the raw components; an MLP (one hidden layer of 32, tanh,
  L-BFGS) on standardized components and target. Its L2 penalty is 1.0, from
  a sweep over 0.01-10 on three v3 cells (suffix L11, think close L19, last
  think token L24): lower penalties blow up on noisy cells (R² down to -28),
  1.0 matches or beats the linear probe on clean ones.
- **Interior R² is the headline metric.** Holding out one of the two extreme
  horizons is extrapolation, where a tanh MLP saturates by construction (on a
  planted curve: interior 0.92 vs 0.74 on all horizons). `k90` uses interior
  R²; all-horizon R² is reported alongside.
- **Control:** the same pipeline with horizon labels permuted across horizon
  groups (`act` ordering), for selectivity.
- **Reading the curves:** `k90` is the smallest k at which a probe reaches
  90% of the best R² either probe reaches at that cell. Linear and MLP `k90`
  equal: flat subspace. MLP `k90` smaller: curved, low-dimensional manifold.
  Neither reaches it at small k: diffuse, or not a high-variance direction
  (compare `act` with `cent`).

## 3. Data

- **Now:** v3 (Qwen3-8B, 116 prompts, 96 with a horizon, 16 horizons), both
  modes, all 37 layers. Positions: the suffix newline after `<|im_end|>`, the
  last suffix token, the think fractions, the think close, the pre-label
  token. With 96 prompts, k above ~24 is near the overfitting regime, and the
  `cent` ordering is capped at 14 components.
- **Next:** the 500-prompt full-bank box run, with 48 log-spaced horizons and
  three think fractions (§5). Activations stay float32: Qwen residual streams
  carry outlier dimensions that can exceed float16's 65,504 range, and numpy
  has no bfloat16. About 6 GB per mode; loading is per layer.
- Later, as a replication: Alan's public 2,000-prompt Qwen3-14B activations.

## 4. Output

`analysis/probe_dimensionality.py <run root>` writes
`<root>/analysis/probe_dimensionality.json` and a figure: R² against k for
selected cells (linear vs MLP, both orderings, control), and per mode a
layer × position map of `k90` for each probe.

## 5. Decisions (Jared, 2026-10-03)

- Target: log horizon, regression.
- Full-bank box run: 48 log-spaced horizons over 30 seconds to 500 years.
- Positions: suffix and think checkpoints matter; the answer token least.
  Box runs keep three think fractions instead of four.
- k grid as above; computed after the fact from saved activations, no extra
  model runs.
