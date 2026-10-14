# Results: linear vs MLP probes on the top-k PCs (v3, Qwen3-8B)

Run 2026-10-03, locally on CPU, `analysis/probe_dimensionality.py
results/horizon_v3_think` (12 min, 14 cores). Spec: `04_spec_probes.md`.
Data: the v3 activations, 96 prompts with a horizon, 16 horizons, all 37
layers, both modes. Metric: held-out R² of log horizon on interior horizons,
leave-one-horizon-out. Outputs in `results/horizon_v3_think/analysis/`:
`probe_dimensionality.json`, `.txt`, `probe_curves.png`, `probe_k90.png`.

## 1. Where the horizon is readable

| position (thinking on) | best layer | full-dim linear R² | best R² any probe |
|---|---|---|---|
| suffix newline after `<|im_end|>` | L2 | 0.97 | 0.99 |
| last suffix token | L2 | 0.95 | 0.99 |
| think@0.25 / 0.5 / 0.75 | L23 / L28 / L19 | 0.39 / 0.23 / 0.17 | same |
| think@1 (last think token) | L33 | 0.74 | 0.74 |
| think close `</think>` | L18 | 0.92 | 0.92 |
| pre-label `:` | L25 | 0.93 | 0.94 |

Thinking off gives the same picture at the suffix (0.98-0.99) and the
pre-label token (0.98). The shuffled-horizon control is negative at every
cell (max -0.08), so no probe reads structure that is not the horizon.

## 2. How many dimensions, and is it curved

Interior R² at small k, fixed layers:

| cell | probe, ordering | k=1 | k=2 | k=3 | k=4 | k=8 |
|---|---|---|---|---|---|---|
| suffix newline L11 (on) | linear, variance PCs | 0.73 | 0.93 | 0.92 | 0.93 | 0.94 |
| | MLP, variance PCs | 0.85 | 0.93 | 0.92 | 0.92 | 0.94 |
| think close L18 (on) | linear, horizon-mean PCs | 0.04 | 0.19 | 0.70 | 0.73 | 0.74 |
| | MLP, horizon-mean PCs | 0.01 | 0.60 | 0.74 | 0.73 | 0.77 |
| pre-label L25 (off) | linear, variance PCs | 0.32 | 0.26 | 0.51 | 0.52 | 0.85 |
| | MLP, variance PCs | 0.30 | 0.25 | 0.81 | 0.88 | 0.93 |
| think@1 L33 (on) | linear, horizon-mean PCs | -0.08 | -0.06 | 0.04 | 0.06 | 0.39 |
| | MLP, horizon-mean PCs | -0.09 | -0.09 | 0.23 | 0.23 | 0.51 |

- **At the suffix the horizon lives in about two dimensions, on a curve.**
  Two components hold 0.93 of a 0.94-0.95 ceiling for both probes. With one
  component the MLP is ahead (0.85 vs 0.73), so the top direction is a bent
  function of log horizon; the second component lets a linear readout unroll
  it. In early layers (L2-L4) the gap is extreme: one component gives the
  MLP 0.86 and the linear probe 0.12, and the linear probe needs three. This
  matches Alan's 14B finding: centroids in a plane, PC1 linear and PC2
  quadratic in log horizon.
- **At the think close and the pre-label token the same pattern repeats with
  a few more dimensions.** The MLP reaches 0.60 from two horizon-separating
  directions where the linear probe gets 0.19; both meet at k = 3-4. Thinking
  off, the pre-label token needs about 3 dimensions for the MLP and about 8
  for the linear probe.
- **Inside the think block the horizon is not a low-dimensional feature.**
  At 0.25-0.75 nothing in the top 64 components beats the control; only the
  all-dimension ridge finds 0.17-0.39. At the last think token both probes
  need about 48 components (0.74). That is a diffuse signal, and 96 prompts
  cannot say whether it is curved.

**Reading:** where the horizon is strong it sits on a low-dimensional (2-4
dims) curved manifold, and it is fully linearly accessible once those few
dimensions are included. "Linear or nonlinear" is therefore not either/or:
the geometry is curved, the readout is linear in a slightly larger subspace.

## 3. Caveats

- 96 prompts and 16 horizons. Per-cell best layers are chosen from 37, which
  is optimistic. The horizon-mean ordering is capped at 14 components.
- The MLP penalty is fixed (1.0, from a sweep on three cells).
- Probes show what is decodable, not what the model uses. "Curved" here
  means "nonlinear in its top few dimensions"; the shape needs the centroid
  geometry, and use needs steering along the line vs along the curve (box).

## 4. Next

The 500-prompt full-bank run with 48 horizons (spec `02_spec_full_bank.md`)
lifts both limits: ~470 horizon prompts and up to 47 horizon-mean
dimensions. Then the same analysis, and Alan's 2,000-prompt 14B set as a
replication.
