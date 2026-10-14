# Results: E8 patching, ablation and steering (Qwen3-8B, full bank)

Box run 2026-10-11 (4090), `experiments/exp8_steering.py --inputs
steering/inputs_Qwen3-8B.npz`; results in `results/horizon/exp8_Qwen3-8B_full/`.
Spec and fixed criteria: `07_spec_steering.md`. Wall time 81 min, of which the
thinking-on cells took 65.

The setup reproduces the original run: the near-vs-both contrast is 11.39
log-odds thinking off and 13.01 thinking on (saved run: 11.2 and 12.9), and 92
of 96 patching pairs differ by more than 1 log-odds (mean gap 11.6).

## 1. Where the horizon travels (T1, thinking off)

Transfer fraction when one position group is patched from a prompt with a
horizon in the other regime (1 = the whole choice shift moves with it):

| layers | horizon tokens | prompt after the horizon | `<|im_start|>` | `choose` | `:` (pre-label) | suffix newline |
|---|---|---|---|---|---|---|
| 1-7 | 1.00 -> 0.88 | 0.00 -> 0.11 | 0 | 0 | 0 | 0 |
| 8-14 | 0.79 -> 0.74 | 0.21 -> 0.26 | 0 | 0 | 0 | 0 |
| 15-19 | 0.66 -> 0.41 | 0.27 -> 0.50 | 0.04-0.08 | 0.0-0.01 | 0.0-0.01 | 0 |
| 20-23 | 0.32 -> 0.15 | 0.45 -> 0.26 | 0.13-0.23 | 0.05-0.14 | 0.03-0.24 | 0 |
| 24-35 | 0.00 | 0.15 -> 0.00 | 0 | 0.11-0.24 | 0.63 -> 0.89 | 0 |

The horizon stays on its own tokens through layer 14, spreads into the
prompt that follows it (peak 0.50 at L19), passes briefly through
`<|im_start|>` (0.2 at L20-23), and lands on the answer slot from L24, where
patching `:` alone carries 63-89% of the effect. Every other template token,
including the suffix newline, carries nothing at any layer.

## 2. The pre-registered cells (T2-T4): all null

| cell | ablation removes | random | line effect (near, +1 decade) | curve - line |
|---|---|---|---|---|
| off, suffix newline, L16 | -0.00 | 0.00 | 0.00 | 0.00 |
| off, pre-label, L21 | 0.01 | 0.00 | -0.02 | 0.02 [0.01, 0.02] |
| on, think close, L19 | -0.01 | 0.00 | 0.01 | 0.00 |
| on, pre-label, L21 | 0.06 | 0.00 | 0.01 | 0.02 [0.01, 0.02] |

All twelve verdicts fail (necessary, sufficient, curve beats line). The edits
are applied (effects are small but nonzero where the table says so) and large:
one decade along the curve is 2.9 units at the off pre-label cell, against a
typical spread of 4.95.

## 3. Reading

- **The patching map explains the nulls.** The cells were chosen where the
  horizon is most decodable, and those are not on the causal path. The suffix
  newline is decodable at R² 1.00 and carries 0.00 of the effect at every
  layer: a passenger. The pre-label token at L21 is decodable at 0.98 but
  carries 0.07: the decision reaches it only from L22-24.
- **Decodable is not used.** This is the central result so far: a perfect
  linear horizon code at a position says nothing about whether the choice is
  computed from it there.
- **Thinking on** was not patched (T1 is thinking off only), but its cells
  behave the same way, so the same late hand-off to the answer slot is the
  first hypothesis.

## 4. Next (built 2026-10-11: `07_spec_steering.md` section 9)

1. Repeat T2-T3 where the map says the effect lives: the pre-label `:` at
   L24-L30 (transfer 0.63-0.82), thinking off and on. Directions can be fitted
   from the saved activations, which keep the pre-label token at every layer.
2. Patch the answer slot with thinking on (only the last position aligns across
   different reasoning traces), to see whether the same L24 hand-off holds.
3. The "prompt after the horizon" group (0.50 at L19) is the other carrier;
   its positions are not in the saved activations, so it can be patched but
   not yet probed.
4. Speed: compute logits at the last position only and use batch 8 for the
   cached reasoning; the thinking-on cells dominate the run time.

## 5. Second run: the answer slot, layers 24-30 (2026-10-12)

`exp8_Qwen3-8B_answer_slot/`, spec section 9; 95 min on a 4090. Contrast
11.39 off, 12.92 on (unchanged). Figures and the check below:
`results/horizon/analysis/steering_*.png`, `analysis/steering_analysis.py`.

**Answer-slot patching across prompts (T5)** reproduces T1 with thinking off
(0.58 at L24 and 0.88 at L34, against 0.63 and 0.88) and shows the same
hand-off with thinking on: near 0 before L16, a trickle of 0.05-0.09 at
L16-20, then 0.22 at L22, 0.56 at L24, 0.82 at L34. Reasoning does not move
where the decision is read from.

| cell | ablation removes, k = 1 / 2 / 4 / 8 | random | curve, toward rational, 2 decades | line | verdicts passed |
|---|---|---|---|---|---|
| off L24 | 0.00 / 0.33 / 0.37 / 0.46 | 0.00 | 0.39 | 0.01 | curve beats line |
| off L27 | 0.08 / 0.11 / 0.66 / 0.74 | 0.00 | 0.06 | 0.01 | none |
| off L30 | 0.21 / 0.21 / 0.80 / 0.83 | 0.00 | 0.04 | 0.01 | none (k = 2 judged) |
| on L24 | 0.19 / 0.27 / 0.30 / 0.47 | 0.00 | 0.15 | 0.00 | curve beats line |
| on L27 | 0.09 / 0.39 / 0.74 / 0.74 | 0.00 | 0.05 | 0.00 | necessary |
| on L30 | 0.67 / 0.73 / 0.83 / 0.83 | 0.00 | 0.04 | 0.00 | necessary |

The probe line does nothing at any cell (at most 0.01); the mean difference
is small and at thinking-on L24 goes the wrong way (-0.13).

## 6. Horizon or decision? (local check, 2026-10-12)

At the late answer slot the horizon averages differ partly because the
decision differs by horizon, so the subspace could be the decision itself.
From the saved activations and the a-vs-b label logits (5-fold CV R², top-4
horizon-average PCs):

| cell | horizon from subspace | horizon from the decision alone | subspace adds beyond the decision | a-vs-b decision from subspace | horizon within the same choice (near / far) |
|---|---|---|---|---|---|
| off L24 | 0.84 | 0.00 | +0.83 | 0.34 | 0.79 / 0.88 |
| off L27 | 0.82 | 0.00 | +0.84 | 0.95 | 0.77 / 0.88 |
| off L30 | 0.78 | 0.00 | +0.81 | 0.97 | 0.74 / 0.86 |
| on L24 | 0.77 | 0.06 | +0.72 | 0.23 | 0.72 / 0.78 |
| on L27 | 0.72 | 0.06 | +0.67 | 0.95 | 0.68 / 0.76 |
| on L30 | 0.69 | 0.06 | +0.64 | 0.95 | 0.66 / 0.73 |

- **The subspace carries real horizon information at every cell.** The
  decision itself predicts log horizon at 0.00-0.06 (it depends on the horizon
  relative to the delays, not on the horizon), yet the subspace predicts it at
  0.69-0.84, adds 0.64-0.84 beyond the decision, and still separates horizons
  among prompts that made the same choice.
- **At L27 and L30 the subspace also contains the a-vs-b decision** (0.95-0.97).
  Ablating it removes the decision by construction, so the two "necessary"
  passes there (thinking on) do not show that the horizon is necessary.
- **L24 is the clean cell.** The subspace is mostly horizon there (decision
  0.23-0.34), ablating it removes 30-37% of the contrast with random at 0, and
  moving along the curve by 2 decades shifts the choice 39% (off) and 15% (on)
  of the contrast toward the rational answer, where the straight probe line
  does nothing. This is the first evidence that the horizon representation is
  used, and that its curved geometry matters for the effect.

## 7. Next

1. **Decision-orthogonal ablation at L27-L30:** remove the a-vs-b direction
   (fitted from the label logits) from the horizon basis, then ablate. This
   asks whether the horizon content beyond the decision is necessary there.
2. **Finer around the hand-off:** L22-L26 at the answer slot, and the
   "prompt after the horizon" positions at L15-L21 (T1's other carrier), with
   curve steering, the direction that works.
3. **Calibrate the curve effect:** compare a 2-decade curve move with real
   prompts whose horizon is 2 decades away, to state the effect as a share of a
   real horizon change.
4. T5 steering before reasoning remains deferred until the site is settled.
