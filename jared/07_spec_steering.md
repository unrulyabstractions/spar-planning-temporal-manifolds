# Spec: E8, causal tests of the horizon code (patching, ablation, steering)

Date: 2026-10-11. Approved by Jared ("write that into spec and write some
code for the box tests"). Companion to `06_results_probes_full.md`, whose
cells and directions this tests. Success criteria in §5 are fixed before the
run.

**Status 2026-10-14: done.** Both runs are in `08_results_steering.md`. Kept as
the method record (directions, units, criteria). From 2026-10-14 nothing new
runs with thinking off; open follow-ups are in `01_plan.md` §5.

## 1. Question

The probes show the stated duration is decodable, ordered and low-dimensional
(`06_results_probes_full.md`). They do not show it is used. E8 asks, on
Qwen3-8B and the full bank:

1. **Where** is the horizon causally carried from the prompt to the choice?
2. Is the low-dimensional code the probes found **necessary** for the choice?
3. Is it **sufficient**: does moving along it change the choice as a real
   horizon change would, and does following the curve beat the straight line?
4. After reasoning, does the end-of-reasoning code **drive the answer**?

## 2. Why this order, and the outcome variable

The saved full-bank logits (`results/horizon/exp0_*_full`) show:

| | thinking off | thinking on |
|---|---|---|
| option order effect on near log-odds | +6.3 | +2.5 |
| content effect (only-near-pays vs both-pay) | +11.2 | +12.9 |
| within-regime dependence on the exact horizon | sign flips with option order | consistent, modest |

So thinking off reads the content coarsely with a large additive position
bias; thinking on is coherent. Hence:

- **Outcome:** the near-option log-odds, `logit(near label) - logit(far label)`
  at the token before the label, read in one forward pass (no sampling).
- **Thinking off:** the prompt is teacher-forced through `I choose:` and
  scored with its option-swapped twin; the two near log-odds are averaged,
  which cancels an additive order effect.
- **Thinking on:** the cached reasoning from the full-bank run is fed back up
  to the label (no generation). No swap twin exists (it would need new
  reasoning); near-option slots are balanced in the bank and results are also
  reported per slot.
- **Prompts:** the `near` and `both` regimes only (176 of 500), where the
  horizon decides the rational answer.
- **Units:** the regime contrast `C = mean LO(near regime) - mean LO(both
  regime)` on unedited prompts. Every effect is `mean(LO_edited - LO) / C`.
  No per-decade calibration: thinking off has no coherent per-decade slope.

## 3. Cells and directions (fixed now)

Cells, from `06_results_probes_full.md` (layer = hidden_states index, as in
the saved activations; an edit at layer L hooks decoder layer L-1):

| mode | cell | layer |
|---|---|---|
| off | suffix newline after `<|im_end|>` | 16 |
| off | pre-label `:` | 21 |
| on | think close `</think>` | 19 |
| on | pre-label `:` | 21 |

Per cell, fitted locally on the saved activations (471 horizon prompts):

- the top-k PCs of the 48 horizon averages (k = min(8, horizons - 1)), and
  each average's coordinates in them (the **curve**). **Curve**: a prompt at
  horizon h moves by `curve(h + alpha) - curve(h)`, interpolated piecewise
  linearly in log h and clamped at the ends.
- `mean`, and the ridge probe `w, b` for log10 horizon. **Line**: along `w`.
- **Mean difference** (the paper's CAA): mean of the long half minus the short
  half (split at the median log horizon).
- **Random**: 2 random directions.

**Every direction moves each prompt as far as the curve does** for the same
alpha, so line, mean difference and random differ from the curve in direction
only. A step that moves the probe's readout by one decade would be far too
short to compare with: measured on the 8B, it is 10-19 times shorter than the
real spacing of horizon averages per decade (line 0.17-0.36 vs curve
1.8-5.2 per decade, against a typical distance from the mean of 2.9-11.5).
The probe weights fast-changing, low-variance directions, so moving its
readout takes little. The readout shift each direction achieves is reported.

`alpha` in {-2, -1, +1, +2} decades. Positive = longer horizon, which should
lower the near log-odds for near-regime prompts; negative = shorter, which
should raise them for both-regime prompts.

## 4. Tests

| | test | mode | edit | output |
|---|---|---|---|---|
| T1 | **interchange patching** | off | pairs identical except the horizon, crossing near <-> both, equal token length; replace the recipient's residual with the donor's at one layer and one position group | transfer fraction `(LO_patched - LO_r) / (LO_d - LO_r)` per layer x group, pairs with `|LO_d - LO_r| > 1` |
| T2 | **subspace ablation** | off cells | mean-ablate the top-k horizon-average subspace at the cell, k in {1, 2, 4, 8}; control: 3 random k-dim subspaces | fraction of the contrast C removed |
| T3 | **steering, line vs curve** | off cells | add the line / curve / mean-difference / random delta at the cell | effect per direction and alpha, near- and both-regime prompts separately |
| T4 | **post-reasoning** | on cells, cached reasoning | T2 and T3 at the think close and the pre-label | as T2 and T3 |

T1 position groups (thinking-off, teacher-forced): the horizon token span, the
prompt tokens after it, and each template-suffix token (`<|im_end|>` through
`I choose:`). Positions before the horizon are identical in a pair and skipped.
Layers 1 to the last. Pairs: up to 96, one per eligible prompt, alternative
horizon drawn with a fixed seed.

## 5. Success criteria (fixed before the run)

Per cell, bootstrap 95% intervals over prompts (1,000 resamples):

- **Necessary:** ablating k = 2 (off) or k = 4 (on) removes at least 0.5 of C,
  and every random subspace of that k removes at most 0.1.
- **Sufficient (line):** at alpha = +1, near-regime effect <= -0.25; at alpha =
  -1, both-regime effect >= +0.25; and the mean random-direction |effect| at
  |alpha| = 2 is at most 0.1 (all at the curve's per-decade distance).
- **Curve beats line:** at |alpha| in {1, 2}, the curve's effect toward the
  rational answer exceeds the line's by at least 0.1, interval excluding 0.
- **Patching** (T1) has no pass/fail; it locates the causal path and is read
  before deciding T5's steering site.

## 6. Data flow and code

1. **Local, CPU** (`analysis/steering_prepare.py results/horizon`): fits the
   per-cell directions and curve from `activations.npz`, exports the cached
   thinking-on reasoning of the 176 prompts (ids, prompt length, think length,
   answer start) from `results/horizon/gen_cache`. Writes
   `jared/steering/inputs_<model>.npz` (+ `.json`), a few MB, shipped by
   `just push`.
2. **Box, GPU** (`experiments/exp8_steering.py --inputs steering/inputs_Qwen3-8B.npz`):
   rebuilds the bank from the generator (asserting the same sample ids),
   runs T1-T4, writes `exp8_<model>_full/results.json`, prints tables and the
   §5 verdicts.
3. Package code in `spar_horizon/steering.py`: the residual-edit hook (add,
   mean-ablate, replace), batched left-padded scoring with explicit position
   ids, direction fitting, curve interpolation, crossing-pair construction.

Estimated box time on a 4090: T1 about 5 min, T2-T3 about 5 min, T4 about
20 min (forward passes over ~2.5k-token cached sequences), plus model load.

Box commands (from `spar-planning-temporal-manifolds/jared/` on the box, after
`just push`, inside tmux):

```
mkdir -p /workspace/results/horizon/logs
/venv/main/bin/python experiments/exp8_steering.py --inputs steering/inputs_Qwen3-8B.npz --out /workspace/results/horizon 2>&1 | tee /workspace/results/horizon/logs/exp8.log
```

`results.json` is rewritten after every cell, so a crash keeps what finished.

## 6b. Built and checked (2026-10-11)

- `spar_horizon/steering.py`, `analysis/steering_prepare.py`,
  `experiments/exp8_steering.py`; 15 tests in `tests/test_steering.py`.
- Inputs prepared from the full-bank run: 4 cells on 471 prompts each, cached
  reasoning for all 176 near/both prompts, 2.7 MB.
- End to end on CPU with a tiny random Qwen3 sharing the 8B tokenizer and
  synthetic inputs: all four tests run, results and verdicts written.
- Found while building: transformers records the last hidden state after the
  final norm, so edits are allowed at indices 1 to n_layers - 1 only (the hook
  refuses the last); a forced think close is `\n</think>\n\n`, so `</think>`
  is located, not computed; a markdown lead changes the label token's form, so
  labels are recorded per prompt.

## 7. Tests (local)

- tiny random Qwen3 (as `test_attribution.py`): a zero edit leaves logits
  unchanged; an edit at layer L position p appears in `hidden_states[L]` at p
  and leaves earlier positions untouched; replacing every position at a layer
  with a donor's reproduces the donor's logits; batched left-padded scoring
  equals single-row scoring; mean-ablation removes the subspace component.
- pure: the line step moves the readout by exactly alpha; the curve delta is 0
  at alpha = 0 and exact at grid points; direction fitting recovers a planted
  direction.
- with the 0.8B tokenizer: crossing pairs have equal length, differ only in
  the horizon span, and cross regimes.
- `local_checks.py steer`: prepare + E8 end to end on the 0.8B smoke outputs
  (plumbing only; 6 prompts give no meaningful numbers).

## 8. Deferred

- T5: steering at the suffix before reasoning, then letting the model reason
  (about 1 h for 100 near-regime prompts x 3 strengths). Site and direction
  from T1-T4.
- T6: representation run with unit rewrites (E3) and Augusto's distractor
  families on the full bank.

## 9. Second run: the answer slot (fixed 2026-10-11, before the run)

The first run's cells were chosen by decodability and were off the causal path
(`08_results_steering.md`): patching puts the decision on the pre-label `:`
from layer 24 (transfer 0.63 at L24, 0.82 at L30), not before. The second run
tests there.

- **Cells:** the pre-label token at L24, L27 and L30, thinking off and on (six
  cells). Directions fitted on the same saved activations
  (`steering/inputs_Qwen3-8B_answer_slot.npz`; the first run's inputs are kept
  as `inputs_Qwen3-8B_first_run.npz`).
- **T2-T4 unchanged,** with the same section 5 criteria. One economy: the
  random-subspace control runs only at the k where "necessary" is judged (k = 2
  off, k = 4 on); other k report the horizon subspace alone.
- **T5, answer-slot patching across prompts (new):** overwrite the recipient's
  pre-label state at one layer with that of a donor prompt in the other regime
  whose near option has the same label; transfer fraction per layer, both modes
  (96 pairs off, 64 on). It is the only patch that lines up across different
  reasoning traces. Consistency check: thinking off should reproduce T1's `:`
  column (near 0 before L20, 0.6-0.9 from L24). No pass/fail; it locates the
  hand-off with thinking on.
- **Speed:** logits at the last position only (`logits_to_keep=1`), batch 8
  for cached reasoning, donors captured at the last position only.
- **Output:** `exp8_Qwen3-8B_answer_slot/`, so the first run's folder stays.

Box command (from `jared/`, inside tmux):

```
mkdir -p /workspace/results/horizon/logs
/venv/main/bin/python experiments/exp8_steering.py --inputs steering/inputs_Qwen3-8B_answer_slot.npz \
    --tests ablate,steer,post,answer --out /workspace/results/horizon 2>&1 | tee /workspace/results/horizon/logs/exp8_answer_slot.log
```

Estimate on a 4090: thinking-off cells about 6 min, T5 about 2 min off and
10-15 min on, thinking-on cells 27 passes each over 176 cached traces, about
15-25 min per cell with the speedups (first run: 54 s per pass at batch 4 with
all logits). About 1 to 1.5 hours in all. To shorten, drop a thinking-on layer:
prepare with `--cells` and rerun `steering_prepare.py`.
