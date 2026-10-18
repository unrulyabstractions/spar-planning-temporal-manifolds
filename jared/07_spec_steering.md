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

## 10. Third run: horizon or decision at the answer slot (fixed 2026-10-14, before the run)

Thinking on only, on the cached reasoning. Question: at L24, where moving
along the curve shifts the choice and the probe line does not, is the effect
carried by the horizon content of the curve or by the near-vs-far decision it
also contains (`08` §7: the L24 subspace decodes the near log-odds at 0.65)?

- **Directions refit on natural closes only** (`steering_prepare.py
  --natural-only`, 371 of 471 horizon prompts), since forced
  closes separate perfectly at these cells and their rate varies with the horizon.
- **Decision direction** `u`: ridge from the coordinates on the cell's
  horizon-average basis (k = 8) to the saved near log-odds, mapped back to
  the residual stream and normalized; inside the horizon subspace by
  construction, so the curve splits exactly into its part along `u` and the rest.
- **Steering** (alpha in {-2, -1, +1, +2} decades): the curve; the curve
  without its `u` component; the `u` component alone (the two add up to the
  curve); `u` itself at the curve's length; the mean difference (refit); one
  random direction at the curve's length. The probe line is dropped (inert
  at every cell so far).
- **Ablation** at k = 4: the horizon subspace (reference), the same subspace
  with `u` projected out (3 dimensions), `u` alone, 3 random 4-dim subspaces;
  plus the horizon subspace at k = 1, 2, 8 as before.
- **Cells:** pre-label L24 (main). Pre-label L22 as a second, optional run if
  box time remains (the hand-off: answer-slot transfer 0.22 there).
- Per-prompt effects and base log-odds are saved so every number can be
  split into natural and forced closes afterwards.

**Criteria** (all 176 near/both prompts; effects toward the rational answer
at |alpha| = 2, both signs pooled, in units of C; 95% bootstrap intervals):

- **Horizon beyond the decision is used (steering):** the curve without `u`
  moves the choice by at least 0.05 toward the rational answer, interval
  excluding 0, and by at least a third of the full curve's effect.
- **Horizon beyond the decision is necessary (ablation):** ablating the
  horizon subspace without `u` removes at least 0.1 of C, interval excluding
  0, with every random subspace at most 0.05.
- **Reading if both fail:** the L24 curve effect is the decision, and "curve
  beats line" (`08` §5) does not show the horizon representation is used.

Box command (from `jared/`, inside tmux), after `steering_prepare.py
results/horizon --name decision_L24 --cells on:pre-label:24 --natural-only`:

```
/venv/main/bin/python experiments/exp8_steering.py --inputs steering/inputs_Qwen3-8B_decision_L24.npz \
    --tests post --out /workspace/results/horizon 2>&1 | tee /workspace/results/horizon/logs/exp8_decision_L24.log
```

Estimate: 1 baseline + 9 ablation + 24 steering passes over the 176 cached
traces at about 56 s each (run 2), about 33 minutes.

## 11. Unit rewrites at the suffix, E9 (fixed 2026-10-14, before the run)

R3: does the representation the probes read before reasoning track the
duration or the unit word? On the full bank the unit word alone predicts a
held-out horizon at R² 0.96 (`06` §5), so the probes cannot tell.

- **Prompts** (`experiments/exp9_units.py`): 22 durations from 10 minutes to
  100 years, each rendered in 2-4 units (`full_bank.UNIT_SETS`: Alan's
  `UNIT_REWRITES` plus 5 sub-day sets; 62 renderings in all, including
  "decades" and "century", which the bank never uses), placed in the same 24
  full-bank prompts (delays, rewards, option order unchanged), plus each
  prompt's original. 1,512 prompts, thinking-on template, one forward pass
  to the end of the prompt, no reasoning.
- **Kept:** hidden states at the suffix newline and the last suffix token,
  every layer (about 1.8 GB). The originals are checked against the saved E0
  activations of the same prompts.
- **Analysis (local):** fit the ridge probe and the horizon-average curve on
  the 471 main-bank prompts at each layer, apply them to the renderings.
  Per duration, the spread of the readout across its units (averaged over
  the 24 contexts) against the spread between durations.
- **Criterion (R3, from `01_plan.md` §1):** at the suffix newline, L16
  (the best layer in `06`), the unit share of the readout variance (within
  durations, over units) is below 20% of the total over durations and
  units. Also reported: the shift, in decades, between the units of one
  duration (e.g. "12 months" read as how much longer than "1 year"), and
  whether unseen units (decades, century) land at their duration.

Box command (from `jared/`, inside tmux): `/venv/main/bin/python
experiments/exp9_units.py --out /workspace/results/horizon`. Estimate: under
5 minutes including the model load.

## 12. E10: patch the horizon before reasoning, then let the model reason (drafted 2026-10-18)

Status: built on the defaults proposed to Jared on 2026-10-17 ("can you start
building that please"); the conditions below are open to change before the run.

**Question.** Every causal result so far edits the answer slot after the
reasoning is fixed (`09` §7). Does the reasoning *use* the horizon code the
probes find in the prompt? If the horizon's prompt states are swapped for those
of a horizon in the other regime, does the model reason its way to the other
answer, as it would if the text said that horizon?

**Pairs.** The T1 construction with the thinking-on template: each of the first
96 near/both prompts (bank order) with a horizon from the other regime that
tokenizes to the same length (`steering.crossing_pairs`, seed 0). Recipient =
the prompt; donor = the same prompt with the other horizon in the text.

**Positions.** The horizon phrase: the tokens of "<horizon text>" after
"time horizon: ", in both prompts (1-3 tokens). `last` = the phrase's last token.

**Layers.** L12 and L16. In thinking-off patching the horizon tokens still carry
0.74-0.79 of the effect through L14 and start to hand it to the rest of the
prompt from L15 (0.66 -> 0.41 over L15-19), so a patch confined to the phrase is
informative only up to about L16.

**Conditions** (greedy, think block force-closed at 3,072 as in E0; the patch is
applied to every prefill pass, i.e. the reasoning pass and the answer pass, never
to generated tokens):

| name | edit | why |
|---|---|---|
| `recipient` | none | baseline, regenerated in this run (batch-noise check against the E0 cache) |
| `donor text` | none, donor prompt | the real effect of the horizon change: the ceiling |
| `full phrase L` | donor states at every phrase token | does the phrase carry the horizon into the reasoning |
| `full last L` | donor state at the last phrase token | reference for the subspace patch at the same position |
| `subspace last L` | swap only the top-8 horizon-average subspace at the last token: `x + ((x_donor - x) B^T) B` | is the low-dimensional code what the reasoning reads |
| `random last L` | the same with a random 8-dim subspace | control |
| `self last 16` | the recipient's own states (24 pairs) | plumbing: must match `recipient` up to batch noise |

The subspace `B` is fitted in the same run: one forward pass over the 471
horizon prompts (prompt only), hidden states at the last phrase token at L12
and L16, top-8 PCs of the 48 horizon averages (`steering.fit_cell`).

**Outcomes per pair:** near log-odds at the token before the label (one forward
pass over the generated sequence with the same patch), the chosen label, forced
close, think length, and whether the reasoning names the recipient's or the
donor's horizon text. **Transfer** = `(LO_c - LO_recipient) / (LO_donor text -
LO_recipient)` over pairs whose gap exceeds 1 log-odds, bootstrap 95% interval.

**Criteria (fixed before the run):**
- The subspace code is read by the reasoning: `subspace last` transfers at least
  half of `full last` at the same layer, interval of the difference to `random
  last` excluding 0, and `random last` transfers at most 0.1.
- No pass/fail on `full phrase`: it locates whether the phrase still carries the
  horizon into reasoning with thinking on.
- Plumbing: `self last 16` changes the chosen label on at most 2 of 24 pairs.

**Cost:** 96 x 2 + 96 x 8 + 24 = 984 generations; E0 took 10.7 s per prompt at
batch 12, so about 3 hours on one 4090, plus a few minutes of extraction.
Conditions run in priority order and each is saved when done, so a stopped run
resumes. `queue.sh` runs one copy per GPU; the copies claim conditions one at a
time (11 conditions, about 17 minutes each on one 4090), so 4 GPUs take about an
hour and 6 about 40 minutes, with the L22 run (`07` §10) on one of them.
