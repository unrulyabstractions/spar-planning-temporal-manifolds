# Results: linear vs MLP probes on the top-k PCs, full bank (Qwen3-8B)

Box run 2026-10-04 (4090, batch 12): E0 on the full bank, 500 prompts per
mode, 471 with a horizon over 48 log-spaced horizons (30 s to 500 y), option
delays and order randomized. Thinking on: 124 min, 103/500 force-closed,
median think length 1,748. Probes run locally, `pipeline/s1_data_vectors/analysis/probe_dimensionality.py
results/horizon` (spec `04_spec_probes.md`). Outputs in
`results/horizon/analysis/`. Compare `history/05_results_probes_v3.md` (116-prompt
starter bank, 16 horizons).

**Superseded in part (2026-10-26), see `12_results_local_checks.md` §1 and §4.**
The R² tables stand (reproduced within ±0.01). The readings do not all stand:
- §2 "one curved direction" at the suffix holds on the 48 horizon averages
  (within-unit 0.50 linear, borderline: 95% CI 0.20–0.67) but per prompt it
  is mostly the unit word.
- §2 pre-label thinking off "largely a staircase by unit": on the horizon
  averages there is also a slope within each unit step (0.75–0.80).
- §4 the thinking-on k = 2 arc at the answer slot loses 0.13–0.17 R² on
  natural closes, so part of it is the forced-close signal.
- §5's open items (grouped inner CV, within-unit R², natural closes only)
  were done in `12`; grouping the inner CV changed R² by ≤ 0.02.

Metric throughout: held-out R² of log horizon on interior horizons,
leave-one-horizon-out (48 folds; each holds out every prompt at one horizon,
3-18 prompts, median 10). The shuffled-horizon control is at most
-0.04 in every cell.

## 1. Where the horizon is readable

| position | best layer | full-dim R² | v3 (starter bank) |
|---|---|---|---|
| suffix newline after `<|im_end|>` (identical in both modes by construction) | L16 | 1.00 | 0.97 |
| last suffix token, thinking on / off | L10 / L35 | 1.00 / 0.99 | 0.95 / 0.94 |
| think@0.25 / think@0.5 | L28 / L26 | 0.32 / 0.28 | 0.39 / 0.23 |
| think@1 (last think token) | L36 | 0.81 | 0.74 |
| think close `</think>` | L19 | 0.96 | 0.92 |
| pre-label `:`, thinking on / off | L21 / L21 | 0.98 / 0.99 | 0.93 / 0.92 |

The pattern replicates on a bank with randomized delays and order, three
times the horizons and five times the prompts: near-perfect before and after
reasoning, weak in the middle of the block, recovering at its end.

## 2. How many dimensions, and is it curved

Interior R² at fixed layers; "variance" orders components by total variance,
"horizon-mean" by separation between horizons:

| cell | probe, ordering | k=1 | k=2 | k=3 | k=4 | k=8 | k=64 |
|---|---|---|---|---|---|---|---|
| suffix newline L16 | linear, variance | 0.90 | 0.91 | 0.90 | 0.94 | 0.97 | 0.99 |
| | MLP, variance | 0.97 | 0.97 | 0.97 | 0.98 | 0.99 | 0.99 |
| last suffix token L10 (on) | linear, variance | 0.81 | 0.82 | 0.82 | 0.86 | 0.97 | 0.99 |
| | MLP, variance | 0.90 | 0.91 | 0.96 | 0.98 | 0.99 | 0.99 |
| pre-label L21 (off) | linear, horizon-mean | 0.83 | 0.95 | 0.96 | 0.97 | 0.97 | |
| | MLP, horizon-mean | 0.83 | 0.97 | 0.98 | 0.98 | 0.98 | |
| pre-label L21 (on) | linear, horizon-mean | -0.05 | 0.74 | 0.76 | 0.85 | 0.87 | |
| | MLP, horizon-mean | -0.02 | 0.78 | 0.86 | 0.90 | 0.93 | |
| think close L19 | linear, horizon-mean | -0.06 | 0.27 | 0.39 | 0.39 | 0.80 | |
| | MLP, horizon-mean | 0.03 | 0.55 | 0.60 | 0.62 | 0.84 | |
| think@1 L36 | linear, horizon-mean | -0.03 | 0.00 | 0.02 | 0.25 | 0.46 | |
| | MLP, horizon-mean | -0.02 | 0.03 | 0.11 | 0.43 | 0.49 | |

- **Before reasoning, the horizon is essentially one curved direction.** The
  top component alone gives the MLP 0.97 and the linear probe 0.90; the linear
  probe needs about 8 components to reach 0.97. This is a sharper version of
  v3 (k=1: 0.85 vs 0.73), and matches Alan's 14B picture: horizon along PC1
  with a bend into PC2.
- **Without thinking, the answer slot holds it in two dimensions, nearly
  flat**: 0.95 linear and 0.97 MLP from two horizon-separating components.
- **With thinking, the answer slot and the think close hold a curved core in
  2-4 dimensions plus a long tail.** At the think close, two components give
  the MLP 0.55 against 0.27 linear; both need about 8 components for 0.8, and
  the last 0.16 of the 0.96 ceiling is spread over many more.
- **The end of the think block is high-dimensional.** The all-dimension probe
  reaches 0.81 at think@1, but the top 64 components hold at most 0.66. In the
  middle of the block (0.25, 0.5) nothing low-dimensional is readable at all.
- **The "k to 90%" summary understates curvature at the suffix**: both probes
  clear 90% of 1.00 with one component, so the MLP's lead there only shows in
  the curves.

**Reading:** the horizon starts as a single, gently curved direction, stays
low-dimensional and slightly curved in the answer slot, and while the model
reasons it is spread across many dimensions, recovering a low-dimensional
curved core at the close. Linear readout is enough everywhere once enough
components are included; curvature shows up as the MLP needing fewer.

## 3. Compared with v3

The largest MLP-minus-linear gaps in v3 (up to +1.27 at early layers) came
from cells where the linear probe on 96 prompts was badly negative; on the
full bank the gaps are smaller and consistent, +0.10 to +0.29 at k = 1-2
across positions. The qualitative picture is the same; the full bank is the
one to quote.

## 4. Caveats

- **Forced closes confound the thinking-on cells.** The top horizon-average
  component at the think close separates force-closed from natural closes
  perfectly (AUC 1.00; 0.88 at the pre-label token), and the forced rate
  varies with horizon. Part of what the thinking-on probes read there may be
  "was the reasoning cut off". Rerun the probes on the ~370 natural closes
  before quoting the thinking-on curvature numbers. The suffix and the
  thinking-off cells are unaffected.
- **The unit word is partly confounded with the horizon.** The bank's units
  grow with the duration by design (seconds, minutes, ..., years), so the unit
  is a coarse binning of log horizon (it explains 0.97 of log horizon itself).
  Along the main horizon-average direction, a smooth cubic in log horizon
  explains 0.971 at the suffix (L16) and adding the unit word reaches 0.982
  (F-test p = 0.004): mostly magnitude, a small unit step. At the thinking-off
  answer slot (L21) the cubic explains 0.923 and the unit word lifts it to
  0.992 (p < 0.001): there the direction is largely a staircase by unit, and
  ordering within a unit is weaker (weeks not ordered). Separating unit from
  duration needs the same horizon in different units: E3 on the full bank
  with Alan's unit rewrites. Figure: `results/horizon/analysis/horizon_shape.png`
  (`pipeline/s1_data_vectors/analysis/horizon_shape.py`).
- **What "decodable from one component" means at the suffix.** The horizon
  averages lie mostly along one direction (55% of their variance; ordering
  rho 0.99 with log horizon), with spacing that shrinks toward long horizons
  (neighbour gaps 0.39 / 0.22 / 0.08 for the short / middle / long thirds).
  The MLP's lead at k=1 is undoing that compression, a 1-D nonlinearity, not
  a bend into extra dimensions. With thinking on, the answer slot is
  different: the horizon is not on the top component and traces an arc
  (second component quadratic in log horizon, R² 0.88).
- Best layers are picked from 37 per position, which is optimistic.
- MLP penalty fixed at 1.0 (sweep on v3); not retuned for 471 prompts.
- Decodable, not used: whether the model's choice runs through the curved
  direction needs steering along the line vs along the curve (R5, box).

## 5. Review, 2026-10-13: the unit-word baseline

Under the same leave-one-horizon-out protocol, predicting each held-out
horizon as the mean log horizon of the training horizons with the same unit
word (seconds, minutes, ..., years) gives **R² 0.962 (interior 0.961)**, with
no activations. The held-out horizon always has same-unit neighbours in
training, so a unit detector scores almost everything above.

- Only the margin above ~0.96 shows magnitude within a unit. The k = 1
  linear probes at the suffix (0.90) and the thinking-off pre-label (0.83)
  are below a unit lookup; part of the MLP's lead at small k may be mapping
  unit clusters to their values rather than a curved manifold.
- The ridge picks its penalty by leave-one-prompt-out inside the training
  fold (`geometry._fit`), where same-horizon prompts remain, so it can
  under-regularize for this task while the MLP's penalty is fixed. No effect
  at k = 1-2; it may understate the linear probe at k >= 8.
- Needed before quoting the dimensionality numbers: within-unit R² (target
  and predictions with unit means removed), the inner CV grouped by horizon,
  and the natural-closes-only run (§4). Next steps: `01_plan.md` §5.
