# Spec: the 8B map on Qwen3-14B, thinking off and on (E11, drafted 2026-10-20)

Status: approved 2026-10-20; criteria in §5 fixed before the run. Data first: the
runs (§6 items 1, 2, 5) are built; the analysis scripts (items 3, 4) follow
once the data is in. Alan and Augusto's 14B activations are a sanity check only.

## 1. Question

On Qwen3-8B the horizon goes through four stages (`08` §1, §5, §6, §12):
it is encoded on its own tokens, spreads into the following prompt, is
handed off to the answer slot (`:`) and is bound to the a/b label. The
choice uses it at only two places, and the positions where it decodes best
are passengers. Two questions for Qwen3-14B (40 layers, against 36):

1. **Stage order:** do the same stages occur in the same order and at the
   same relative depth (layer / n_layers), thinking off and on?
2. **Cross-model alignment:** is the 14B horizon manifold the same object as
   the 8B one (same shape and ordering after alignment), at matching relative
   depth, and does the alignment hold within a unit as well as across units?

Out of scope (deferred): E10 on 14B, patching the option/delay tokens
(prediction 3), and the natural-closes / within-unit fixes for the 8B
probes, except where §5 needs them.

## 2. 8B reference values (relative depth = layer / 36)

| marker | definition (applied identically to both models) | 8B layer | depth |
|---|---|---|---|
| `encode_end` | last layer where the horizon-token group carries >= 0.5 of T1 transfer (off) | 17 | 0.47 |
| `spread_peak` | argmax of the "prompt after the horizon" transfer (T1, off) | 19 | 0.53 |
| `handoff` | first layer where answer-slot transfer >= 0.5 (T5; off and on separately) | 24 / 24 | 0.67 |
| `bind` | first layer where the a/b decision decodes at CV R² >= 0.9 from the top-4 horizon-average PCs at pre-label | 27 / 27 | 0.75 |
| `decode_peak` | best-R² layer per position (`06` §1) | per position | |

`analysis/stage_markers.py` (§6) computes these from saved results. Its
first check is that it reproduces this table from the 8B folders, and it
interpolates transfer between layers to give fractional crossings.

## 3. Runs (box, 2 GPUs, 80 GB or 48 GB cards)

Same full bank, same seed and same 500 prompts as the 8B run, so every
prompt has a twin in both models. 14B weights are ~30 GB bf16; KV at 3.3k
tokens is ~0.16 MB/token, so batch 12 fits a 48 GB card.

| step | GPU | what | 8B time | 14B estimate |
|---|---|---|---|---|
| A | 0 | E0 thinking on, full bank | 124 min (4090) | 1.5–2.5 h |
| B | 1 | E0 thinking off, then steering_prepare (off cells), then exp8 `patch` (T1) | ~20 min T1 | 0.5–1 h |
| C | both | steering_prepare (all cells), then exp8 `answer` (T5, both modes) | part of 95 min | ~1 h split |
| D | both | exp8 `ablate,steer,post` at the pre-registered cells (§4) | part of 95 min | ~1 h split |

Total about 3–4 h wall clock. A 4-prompt smoke on the box (`--limit 4`)
comes first, to time it and to check the activations shapes. Generation
caches and activations are kept under `results/horizon/exp*_Qwen3-14B_*`
(~2 GB of activations per mode at d = 5120).

## 4. Steering cells (fixed now)

Pre-label `:`, thinking off and on, at the 8B relative depths of L24 / L27 /
L30 → **14B L27 / L30 / L33**. Thinking on uses natural closes only (as in
`08` §9). Directions and the §6-style decision check are those of the 8B
answer-slot run, fitted by `steering_prepare.py --cells` on the 14B
activations. If T5 puts the 14B hand-off more than 2 layers from L27, a short
extra run (step D') repeats D at the observed hand-off layer. That run is
reported as exploratory.

## 5. Success criteria (fixed before the run)

**S1. Stage order.** Thinking off: `encode_end < spread_peak < handoff <
bind`. Thinking on: `handoff < bind`, since T1 is thinking off only. Passes
if the order holds in both modes and every marker is within 0.08 relative
depth (~3 layers) of its 8B value. Fails if any order inverts or any marker
moves by more than 0.15. A result in between is reported as "shifted".

**S2. Depth correspondence.** For each pair of 8B and 14B layers, linear CKA
of the horizon-average matrices (48 horizons x d), at the suffix newline and
at pre-label, in each mode. Passes if the best-matching 14B layer is monotone
in the 8B layer (Spearman >= 0.9) and its mean |relative-depth difference| is
<= 0.08. It also passes only if the 14B layer that best matches 8B L24
(pre-label) is within 2 layers of the 14B `handoff`.

**S3. Same manifold.** At matched cells (suffix newline at each model's
`decode_peak`, pre-label at each model's `handoff`):
- linear CKA of horizon averages >= 0.8, and above the 95th percentile of a
  null that shuffles horizon labels in one model (1,000 shuffles);
- **cross-model readout:** orthogonal Procrustes from the 14B top-8 PCs to
  the 8B top-8 PCs, fitted leave-one-horizon-out. The 8B probe applied to
  the mapped held-out 14B horizon gives R² >= 0.8;
- **within-unit version** (horizon averages with unit-word means removed,
  target likewise): CKA above the shuffle null's 95th percentile, and the
  value reported. This guards against the obvious trivial pass: both models
  encode the unit word, and a unit lookup alone gives R² 0.96 (`06` §5). If
  only the across-unit version passes, the claim is "same unit code", not
  "same manifold".
- horizon rank order along the aligned first coordinate: Spearman with log
  horizon >= 0.95 in both models.

**S4. Decodable is not used, on 14B.** The suffix newline decodes at R² >=
0.95 and carries <= 0.05 T1 transfer at every layer. Probe-line steering is
<= 0.05 of the contrast at every §4 cell. At the hand-off cell, the curve
beats the line (bootstrap CI excludes 0).

**S5. Reasoning doesn't move the hand-off.** The 14B T5 `handoff` with
thinking on is within 2 layers of thinking off (8B: identical).

Sanity, not a criterion: the near-vs-both contrast is > 5 log-odds in both
modes, and the 14B thinking-off decode is close to Alan and Augusto's 14B
activations at the matching position (`../relevance_aware_decoder/`).

## 6. Code

1. **Port check.** `exp0_baseline.py`, `steering_prepare.py`,
   `exp8_steering.py` and `probe_dimensionality.py` run with `--model
   Qwen/Qwen3-14B` with nothing hard-coded to 36 layers or d = 4096 (exp8
   already uses `len(model.layers)`). Any hard-coded value is fixed, with a
   fast tokenizer-only test.
2. **`steering_prepare.py` off-only cells.** Step B can start T1 before the
   thinking-on E0 finishes. Check whether it needs both run folders today.
3. **`analysis/stage_markers.py`.** Computes the markers of §2 for any model
   and reproduces the 8B table.
4. **`analysis/cross_model_alignment.py`.** Layer x layer CKA, shuffle null,
   Procrustes readout, within-unit version, rank order; S2 and S3 verdicts
   printed. Unit tests on synthetic data (a known rotation and rescale of one
   manifold must give CKA 1 and readout R² 1; independent noise must fall
   inside the null).
5. **`queue_14b.sh`.** Steps A–D on 2 GPUs in the style of `queue.sh`
   (`.done` markers, rerun-safe, does not stop the box).
6. Results in `11_results_14b.md`.

Built 2026-10-20: items 1 (nothing hard-coded to 36 layers or d = 4096), 2
(`steering_prepare.py --model --no-reasoning`; the 8B inputs reproduce within
the script's own run-to-run noise) and 5 (`queue_14b.sh`, dry-run with stubs;
inputs go to `$OUT/steering_inputs/`).

Done 2026-10-21: items 3, 4 and 6; results in `11_results_14b.md`. Added for the
rest of the run: shared-prefix scoring in E8 (`steering.score_last_edits`, checked
by `local_checks.py prefix`), `steering_prepare.py --no-decision`, and
`queue_14b_rest.sh` (thinking-on L27 check, L30/L33, probe line at all cells). Figures: stage markers by relative depth
   (8B vs 14B, off and on), and the CKA heatmap with the relative-depth
   diagonal drawn.

## 7. Risks

- **Thinking-on traces differ between models**, so only the suffix, think
  close and pre-label positions are compared across models. The forced-close
  rate may differ too; S3 thinking-on cells use natural closes only.
- **CKA at small n (48 horizons)** is biased upward. That is why the
  shuffle null is part of the criterion and not an afterthought.
- One larger model gives two points, not a trend. If S1 shifts, Qwen3-32B
  (64 layers) is the third point.
