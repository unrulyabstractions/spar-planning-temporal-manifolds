# First big experiment: costs from measured runs, and a recommendation

2026-11-04. Read-only: no GPU job was run. Everything below comes from saved box
results in `$RESULTS` = `/home/Jared/SPAR/results/horizon` (`pipeline/README.md`; manifests, `results.json` phase timers,
queue logs, generation condition files), the Qwen3-8B tokenizer on CPU, and
arithmetic.

- `pipeline/s3_analysis/by_datatype/run_throughput.py`: reproduces every measured number in Steps 1-2
  (`~/miniconda3/envs/mp/bin/python -m pipeline.s3_analysis.by_datatype.run_throughput $RESULTS`).
- `pipeline/s3_analysis/by_datatype/big_experiment_costs.py`: the cost model for Steps 3-5. All
  assumptions are constants at the top; change one and rerun
  (`python -m pipeline.s3_analysis.by_datatype.big_experiment_costs`).

## Summary

- **The logit-readout steering grid is cheap.** Design A (36 layers × every
  pre-think position × 500 × ±α) is 4.85 M forward passes, which is 17.5 GPU-h:
  **$7 on a 4090, $11 on a 5090**. Prefix-KV caching (C) cuts that to 10.3 GPU-h.
  Restricting to positions that can see the horizon (C\*) cuts it to 1.6 GPU-h.
- **Thinking-on generation costs about 1,000× more per run** (11.7 s against 3-13 ms).
  The full thinking-on grid (E) is 14.5 M generations, which is 47,000 GPU-h
  ($19k-$31k). It is out of reach. Thinking on at the top 5 cells (F) is
  73 GPU-h ($29 on a 4090), or 53 GPU-h ($21) with a shared unsteered baseline.
- **The GPU makes no difference on this stack.** The 5090 ran forward passes at
  the 4090's rate (76.8 vs 75-84 passes/s) and generated at the same rate
  (151-179 vs 174 tok/s). The stack is software-bound (HF `generate`, hooks,
  stragglers). An H100 would have to be 4.8× faster than a $0.40 4090 to pay
  off, and nothing measured suggests it would be. Rent several 4090s or 5090s.
- **Recommended bundle: 73 GPU-h, about $30 on 4090s ($48 on 5090s), about
  19 h on 4 GPUs.** 73% of it is the thinking-on check (Step 4).
- **Three design corrections, from the project's own results (Step 4).**
  1. Only 42 of the 135 pre-think positions can see the horizon (causal attention).
  2. On the full bank the unit word explains 0.97 of log horizon, so log-linearity
     and unit invariance must be tested on the units bank.
  3. The thinking-off logit readout needs the swapped-twin average: option order
     alone moves it +6.3 log-odds.
- **No dollar amounts are recorded anywhere.** Prices are Nov 2026 Vast.ai market
  snapshots. Fix them in the script once you look up what you actually paid.

---

## Step 1. Inventory

### Prompts (Qwen3-8B tokenizer, the project's own `build_records` + `chat_text`)

| bank | prompts | tokens up to `<think>` (chat template, thinking on) | thinking-off readout sequence (empty think + `I choose:`) |
|---|---|---|---|
| **full** (the n = 500 bank) | 500 (29 have no horizon) | **mean 134.7, median 136, max 141** (min 110) | mean 141.7, median 143, max 148 |
| fill | 425 | mean 139.6, median 140, max 146 | mean 146.6, max 153 |
| units (unit overlap) | 515 | mean 136.3, median 136, max 141 | mean 143.3, max 148 |

The template adds 8 tokens around the user content (mean 126.7 user tokens on
the full bank). **On the full bank, the first horizon token sits 41-43 tokens
before `<think>`** (mean 41.7): only 31% of pre-think positions come at or after it.
The horizon value itself is 2.7 tokens (number + unit, max 4).
Cached-reasoning sequences that the thinking-on exp8 cells score: 8B mean 1,718 tokens
(median 1,450, max 3,219); 14B mean 1,082.

### Box runs

GPU type: docs and queue logs name it for each run. From 2026-10-25 on, the queue
logs print the device. **Cost: not recorded for any run.** `infra/vast.sh` keeps the
live rate in `.state/dph` and deletes it on stop. `.state/spend.log` was never
written. The only price in the repo is the 4090 search cap `MAX_PRICE=0.50` $/h.

| date | run | what | model | GPU | passes / generations | batch | thinking | wall clock | cost |
|---|---|---|---|---|---|---|---|---|---|
| 09-24 to 10-01 | horizon_v1-v3 exp0 (starter bank) | generation + activations | 8B | 4090 (history docs) | 116 gens per run | 8 | on/off | on: 4,084 / 2,157 / 3,956 s | not recorded |
| 10-04 | exp0 think-on full | generation + activation capture | 8B | 4090 (`06_results`) | 500 gens, ~0.93 M gen tokens | 12 | on | gen+extract 5,330 s; total 7,420 s | not recorded |
| 10-04 | exp0 think-off full | forward + capture + behavior | 8B | 4090 | 500 | 12 | off | gen+extract 56 s; total 206 s | not recorded |
| 10-11 | exp8 full | T1 patching, ablation, steering, logit readout | 8B | 4090 (`08_results`) | T1 47,232; off cells 25,344; on cells 12,672 (cached reasoning, no generation) | 32 off / 8 on | off + on | 4,896 s (81 min) | not recorded |
| ~10-13 | exp8 answer_slot | answer-slot patch, ablation, steering | 8B | 4090 | off 28,512; on 14,256 | 32 / 8 | off + on | 5,712 s (95 min) | not recorded |
| ~10-14 | exp8 decision_L24 | ablation + steering | 8B | 4090 | 5,808 on | 8 | on | 2,239 s | not recorded |
| 10-18 | exp8 decision_L22 | ablation + steering | 8B | 2-GPU box, **type not recorded** | 5,808 on | 8 | on | 41 min | not recorded |
| 10-18 | exp10 reasoning | patched thinking-on generation | 8B | same 2-GPU box | 984 gens, 1.88 M gen tokens | 12 | on | 16,309 GPU-s; 2 h 41 min box | not recorded |
| 10-20 | exp0 14B think-on full | generation + capture | 14B | 2-GPU box, **type not recorded** (spec: "80 GB or 48 GB cards") | 500 gens, ~0.72 M tokens | 12 | on | gen+extract 10,808 s; total 17,488 s | not recorded |
| 10-20 | exp0 14B think-off full | forward + capture | 14B | same | 500 | 12 | off | 2,801 s (mostly CPU probe sweep) | not recorded |
| 10-20 | exp8 14B patch | T1 patching | 14B | same | 52,608 | 32 | off | 1,388 s | not recorded |
| 10-21 | exp8 14B answer_slot off / on | patch, ablation, steering | 14B | same | off 34,848 + patch; on 5,808 + patch | 32 / 8 | off / on | 2,546 s / 3,080 s | not recorded |
| ~10-21 | coherence 14B | label-format coherence | 14B | not recorded | 80 prompts; gen count not recorded | ? | on | ~48 min | not recorded |
| 10-24 | exp8 14B line_off, line_on, L27, L30_L33 | ablation + steering (prefix cache) | 14B | 2-GPU box, type not recorded | 21,120 off; 14,256 + 3,520 + ... on | 32 / 8 | off / on | 71 / 724 / 307 / 532 s | not recorded |
| 10-25 | exp0 units on / off | generation + capture | 8B | 2× RTX 5090 (queue log) | 515 | 20 | on / off | off 809 s. On: **first attempt crashed after 97 min; the rerun took 1,299 s from the warmed cache, so it is not a throughput measurement** | not recorded |
| 10-29 | exp0 fill on / off | generation + capture | 8B | 1× RTX 5090 | 425 gens, ~0.97 M tokens | 20, **OOM, continued at 13** | on / off | on: gen+extract 5,429 s, total 6,296 s; off 685 s | not recorded |
| 10-30 | exp8 pristine (E13 stage A) | steering (prefix cache at pre-label) | 8B | RTX 5090 | 26,400 off + 12,864 on | 32 / 8 | off + on | 599 s | not recorded |
| 10-30 | exp13 stage B | steered thinking-on generation | 8B | RTX 5090 | 768 gens, 1.45 M tokens | 12 | on | 10,383 s (2 h 53 min) | not recorded |
| 11-01 | exp14 phrase steering | steered/patched thinking-on generation | 8B | RTX 5090 | 1,192 gens, 2.10 M tokens | 12 | on | 13,896 s (3 h 52 min) | not recorded |

All runs use HF transformers with forward hooks, bf16, greedy decoding (E13/E14
included). None used vLLM. 2026-11-03 autoresearch was CPU-only on saved artifacts.

---

## Step 2. Measured throughput (Qwen3-8B unless noted)

### Forward-only

| workload | GPU | source runs | rate |
|---|---|---|---|
| full pass, thinking-off sequence (142 tok), batch 32 | 4090 | exp8 full (4 phases), T1 patch, answer_slot (6 phases) | **75.2-84.0 passes/s** = 10.7-11.9 k tok/s |
| same | **5090** | exp8 pristine, suffix-newline steer (13,200 passes) | **76.8 passes/s** = 10.9 k tok/s |
| full pass over cached reasoning (1,718 tok), batch 8 | 4090 | exp8 full, answer_slot, decision_L24 | 2.7-3.3 passes/s = 4.6-5.6 k tok/s |
| same | 2-GPU box (unknown) | decision_L22 | 2.4 passes/s = 4.2 k tok/s |
| prefix-cached extra edit at the last token, thinking-off | 5090 | pristine pre-label L22/L24 | **1.6 ms per edit per sequence** (461-468 edits/s incl. prefix) |
| prefix-cached extra edit, cached reasoning | 5090 | pristine on L22/L24 | 8.3-8.4 ms per edit per sequence; prefix 0.32 s per sequence |
| activation capture (11 kept positions × 37 layers) | 4090 | exp0 think-off full | 56 s for 500 prompts |
| 14B: full pass off / on; cached edit off / on | unknown | 14B exp8 runs | 38.2/s (5.4 k tok/s); 1.9/s (2.1 k tok/s); 1.7 ms; 9.7 ms |

**Model used for every cost below.** A full thinking-off pass costs 13.07 ms (1/77 s).
A cached edit that recomputes a suffix of s tokens costs **1.6 ms + 0.0809 ms × s**.
This line is fit through the two measured points: s = 1 (the cached last-token
edit) and s = 141.7 (the full pass).

### Generation (thinking on, 3,072-token think budget, HF `generate`, greedy)

| run | GPU | batch | gens | mean generated | **tok/s per GPU** | **s per generation** |
|---|---|---|---|---|---|---|
| exp0 full | 4090 | 12 | 500 | ~1,850 (think 1,799) | 174 | 10.66 (incl. extraction) |
| exp0 fill | 5090 | 20 → 13 | 425 | ~2,290 (think 2,237) | 179 | 12.77 |
| exp14 (steered) | 5090 | 12 | 1,192 | 1,760 | 151 | **11.66** (used below) |
| exp13 (steered) | 5090 | 12 | 768 | 1,886 | 140 | 13.52 |
| exp10 (patched) | 2-GPU, unknown | 12 | 984 | 1,914 | 115 | 16.57 |
| exp0 14B | unknown | 12 | 500 | ~1,430 (think 1,381) | 66 | 21.62 |

**Thinking-trace length**, 8B full bank: mean 1,799, median 1,748 tokens. 103 of 500
prompts (21%) were force-closed at 3,072. Fill bank: mean 2,237 (38% forced).
Units bank: mean 1,872. 14B: mean 1,381 (5% forced).

### $/GPU-hour

Not recorded (see Step 1). Assumed, Vast.ai on-demand, Nov 2026 snapshots:
**4090 $0.40** (trackers $0.27-0.59; the repo's search cap is $0.50),
**5090 $0.65** (median offer $0.63-0.65, floor $0.20-0.40),
**A100 80 GB $1.10** ($0.74 floor, $1.50+ when busy),
**H100 SXM $1.93** ($1.53 PCIe to ~$2.40).
Sources are at the end.

### Why runs disagree

- **4090 = 5090.** Same forward rate (75-84 vs 76.8/s) and same generation rate
  (174 vs 151-179 tok/s). The stack is bound by Python, hooks and kernel launches,
  not by FLOPs or bandwidth. Generation is also straggler-bound: a batch runs until
  its longest member finishes, and 21-38% of prompts hit the 3,072 cap. Decode
  step rate falls about 3× as the cache grows: 54 → 17 steps/s on the 4090 at
  batch 12, 25 → 16 on the 5090, 20 → 8 on the 14B. A theoretical
  bandwidth-bound HF decode would lose far less (KV at 3k × 12 rows is
  ~5 GB on top of 16 GB of weights).
- **Full vs prefix-cached thinking-on cells.** Twenty edits per prompt cost
  20 × 0.32 s = 6.4 s as full passes, against 0.32 + 19 × 0.0084 = 0.48 s cached.
  That is 13× faster. The exp8 runs before 2026-10-24 predate the cache path.
- **E10 slower (16.6 s/gen).** It has prompt-patching hooks on every batch and an
  unknown GPU. E13 is slower than E14 because its traces are longer (1,886 vs 1,760).
- **Anomaly, unresolved.** The "4090" thinking-off rate of 84 passes/s × 141.7 tokens
  × 13.9 GFLOP/token works out to ~165 TFLOP/s. That equals the 4090's dense BF16
  peak, which is implausible for HF + hooks. The 5090 figure is 151 TFLOP/s, about
  72% of its peak, which is plausible. Either the GPU label on the Oct 11-14 runs is
  off, or something reduces the work. Planning is unaffected, since both cards
  measure about 77/s.

---

## Step 3. Cost of the candidate designs

### Formulas (`pipeline/s3_analysis/by_datatype/big_experiment_costs.py`)

```
runs (grid)     = layers × positions × prompts × coefficients (× V; V = 2 with the swapped twin)
full-pass GPU-h = runs / 77 / 3600
cached GPU-h    = runs × (1.6 ms + 0.0809 ms × s̄) / 3.6e6
                  s̄ = 141.7 − (t_first + t_last)/2   # mean recomputed suffix, t uniform over the steered positions
generation GPU-h = generations × 11.66 s / 3600
cost            = GPU-h / speed_factor × price      # speed_factor 1.0 for every card (measured: 5090 = 4090)
```

V = 1 below, as you specified. Multiply A-C by 2 for the project's swapped-twin average.

| design | runs | tokens processed / generated | GPU-h | 4090 $0.40 | 5090 $0.65 | A100 80GB $1.10 | H100 $1.93 |
|---|---|---|---|---|---|---|---|
| **A** 36 layers × 134.7 positions × 500 × ±α, full passes (+500 baseline) | 4,849,700 | 687 M | 17.50 | $7.00 | $11.37 | $19.24 | $33.77 |
| **B** 18 layers, full passes | 2,425,100 | 344 M | 8.75 | $3.50 | $5.69 | $9.62 | $16.88 |
| **C** = A with prefix KV cache (s̄ = 74.8) | 4,849,700 | 363 M | 10.32 | $4.13 | $6.71 | $11.35 | $19.91 |
| **C** = B with prefix KV cache | 2,425,100 | 182 M | 5.16 | $2.06 | $3.35 | $5.67 | $9.96 |
| **C\*** = C, 36 layers, only the 41.7 positions from the horizon on (s̄ = 28.3) | 1,501,700 | 43 M | 1.62 | $0.65 | $1.06 | $1.79 | $3.14 |
| **D** edit at t only vs at t..end | same as A/B/C | | | | | | |
| **E** thinking-on generation, every cell of A, 3 samples | 14,549,100 | 25.6 B generated | 47,123 | $18,849 | $30,630 | $51,835 | $90,947 |
| **E** same, 18 layers | 7,275,300 | 12.8 B | 23,564 | $9,426 | $15,317 | $25,920 | $45,478 |
| **F** 5 cells × 500 × 3 coefficients × 3 samples | 22,500 | 39.6 M | 72.9 | $29.15 | $47.37 | $80.16 | $140.65 |
| **F′** same, α = 0 run once and shared by the 5 cells | 16,500 | 29.0 M | 53.4 | $21.38 | $34.74 | $58.79 | $103.14 |
| **G** activation cache, every layer × position, one pass per prompt | 500 | 0.07 M | 0.25 | $0.10 | $0.16 | $0.28 | $0.49 |
| **H** patching, top 5 layers × 2 position groups × {full, proj-1d, proj-8d, random-1d}, logit readout | 21,000 | 3.0 M | 0.08 | $0.03 | $0.05 | $0.08 | $0.15 |
| **H′** same patches, thinking-on readout (1 sample) | 20,000 | 35 M | 64.8 | $25.91 | $42.11 | $71.26 | $125.02 |

Notes:

- **A vs B.** Halving the layers halves the cost and saves $3.50 on a 4090. The
  layer curve is the headline result, so keep all 36.
- **C.** Saves 41%, not 2×. A recomputed suffix still pays the fixed ~1.6 ms per run.
  This cost is extrapolated from last-position edits: mid-prompt cached
  steering is **new code**, since `steering.score_last_edits` handles only the last
  position. A per-prompt batch works: the 36 layers × 2 α at one t share
  one prefix cache, which gives 72 rows. Skipping layers below L too (cache the
  residual entering L) could save up to another ~2×. That is unmeasured and needs
  custom code.
- **D. Cost is identical.** Both variants recompute positions t..end.
  Scientifically they differ: steering t..end also edits the readout token at the
  steered layer, so late-layer cells pick up a direct unembedding effect.
  Use "only at t" for the map and "t..end" at the top cells.
- **E is not feasible.** Even a 10× faster stack (vLLM-class, unmeasured, and hooks
  would need nnsight or custom code) leaves E at $1,900 or more.
- **G.** Compute is seconds. The cost is capture and disk:
  500 × 134.7 × 37 × 4096 × 2 B = **20.4 GB in bf16** (40.8 GB in fp32). float16
  overflows on Qwen's outlier dimensions (`01_plan` §7); bf16 has fp32's range.
  Fit the probes on the box and sync summaries, not 20 GB.
- **H.** Projection-only and full-residual patching cost the same: the projection
  is one matmul in the hook.
- **H100/A100 columns assume the measured speed (1.0×).** At an optimistic 2.5×
  (H100) or 1.5× (A100), the bundle costs $57 and $54, still above $30 on 4090s.
  Break-even over a $0.40 4090 needs **4.8× (H100), 2.75× (A100), 1.6× (5090)**.

---

## Step 4. Recommendation

### Three corrections first, from the project's own results

1. **Positions before the horizon cannot carry it.** Causal attention means the
   first 93 of 135 pre-think positions never see the horizon phrase. A CAA
   direction fit at the same position there is noise. Steering them only asks
   whether a direction transported from a later position works earlier, which is
   a different question. The 41-43 positions from the horizon on are nearly
   fixed-layout (the text after the horizon is identical across prompts). So
   "every token" aligns across prompts as an offset from the first horizon token.
   This also fixes a hidden problem with A as specified: absolute token indices
   do not line up across prompts that range from 110 to 141 tokens.
2. **Log-linearity on the full bank is mostly the unit staircase.** The unit word
   explains 0.97 of log horizon (`01_plan` §7). A unit lookup alone scores
   leave-one-horizon-out R² 0.96. E14 found the pre-reasoning curve to be "a
   written-number direction by construction" (unit explains 88% of log duration).
   `24` found the prompt-end code is "unit word + log written number". The
   prior that "one log-linear horizon direction" is really number × unit is
   strong. So the units bank (same durations written in several units) is a
   primary test, not a side control. A **written-number direction** is the
   control that matters most.
3. **The thinking-off readout needs the twin average, and reverses a dated
   decision.** On 2026-10-14 you stopped thinking-off runs: option order adds
   +6.3 log-odds, and the within-regime horizon dependence flips sign with option
   order. Averaging each prompt with its swapped twin (V = 2, as exp8 does) cancels
   the additive part and doubles the screen's cost. The screen is still a
   no-reasoning policy measurement. Stage 2 (thinking on) is what ties it to the
   behavior you care about. Also, the k = 1 probe line was causally inert at every
   E8 cell, and the long-minus-short mean difference steered backwards at L24
   (short horizons are all `none` regime). So fit the CAA direction as a
   regression on log h with regime, log reward and option order as covariates,
   not as a mean difference.

### The bundle: Qwen3-8B, banks full (500) + units (515), swapped twin V = 2

| # | item | runs | GPU-h |
|---|---|---|---|
| 1 | Activation cache, full + units, all 37 residual streams × all pre-think positions (V = 1) | 1,015 passes | 0.50 |
| 2a | **Steering screen, horizon direction**: 36 layers × 41.7 post-horizon positions × (full + units) × V × ±α, edit at t only, prefix cache | 6.09 M | 6.59 |
| 2b | **Control screen, same grid, full bank**: norm-matched random, reward-magnitude (log $ amount), written-number (log numeral, ignoring unit) | 9.01 M | 9.74 |
| 2c | Null: horizon direction transported from the unit token to every 4th pre-horizon position, 18 layers | 0.84 M | 2.17 |
| 3 | **Dose-response at the top 5 cells**: 8 α (±0.5, 1, 2, 3 decades) × 7 directions (horizon, magnitude, written-number, 4 random seeds) × (full + units) × V | 0.57 M | 0.61 |
| 4 | **Patching, top 5 layers**: full residual vs projection onto the 1-d direction vs top-8 subspace vs random 1-d. Position groups: horizon value tokens, last pre-think token. Pairs: different duration / same unit, same duration / different unit (invariance), random | 85,260 passes | 0.31 |
| 5 | **Thinking-on check (F′)**: top 5 cells × ±α + shared α = 0 × 500 × 3 samples (T = 0.6) | 16,500 gens | 53.44 |
| | **total work** | | **73.4** |

**Total: 73.7 GPU-h on one GPU, $29.47 on a 4090 or $47.89 on a 5090.** On 4 GPUs
it is 74.6 GPU-h with setup (0.3 GPU-h per card), **$29.83 on 4090s or $48.48 on 5090s,
about 18.6 h wall clock**. A100 is $54-82 and H100 $57-144, depending on the
unmeasured speed factor.

**Run it as two rentals.** Stage 1 (items 1-4) is 19.9 GPU-h, about 5 h on 4 GPUs.
Pick the 5 cells locally, then run stage 2 (item 5): 53.4 GPU-h, about 13.4 h on
4 GPUs. Pre-register the selection rule in the spec, for example: the largest
held-out effect on the represented horizon, with a CI excluding the random
control, at least 2 layers apart.

### How each required component is covered

- **Layer/token steering curve.** Item 2a. Two readouts come from every pass at no
  extra cost: (i) twin-averaged near-vs-far log-odds at the pre-label `:`;
  (ii) the *represented* horizon, in decades. Readout (ii) is the last-token
  residual at a fixed late cell, projected onto that cell's held-out-calibrated
  probe. It separates "moves the horizon" from "moves the decision". At the
  answer slot that matters: L24-30 decode the decision at up to 0.95.
- **Probe vs steering dissociation per layer.** Item 1 gives probe R² per
  layer × position. Use leave-one-horizon-out and within-unit, with the −0.25
  "unit only" floor. Item 2a gives the steering effect on the same cells. The
  dissociation map is CPU work. E8 already shows the pattern to expect: the
  suffix newline decodes at 1.00 and carries 0.00 of the effect.
- **Train/test split for the CAA direction.** 2-fold cross-fitting **by horizon
  value**: the 48 log-spaced horizons split into two interleaved folds of 24, and
  prompts follow their horizon. Fit on fold A, steer fold B, and vice versa.
  Every steered prompt is held out from its direction at no GPU cost. The 29
  no-horizon prompts are steer-only targets: does adding the direction create
  a horizon?
- **Held-out horizon values for log-linearity.** The same split. Compare, on the
  held-out fold, readouts linear in log h, linear in h, a unit staircase, and
  unit + log numeral. On the units bank, test the within-unit slope. Calibrate
  steering against rewriting: +1 decade of steering should move the choice as
  much as the bank's real 10× horizon change within the same regime.
- **Projection-only vs full patching.** Item 4. If the projection carries the
  effect, it should transfer most of the full-residual effect, and random 1-d
  should transfer about 0.
- **Controls.**
  - Unit-word invariance ("1 year" / "12 months" / "365 days"): units bank in 2a, 3
    and 4. E9 found a 0.30-decade shift per rewrite; that is the number to beat.
  - Generic magnitude: log reward, fit on the same activations, so it is free.
  - Written number: the strongest alternative.
  - Random direction: norm-matched, 1 seed on the grid and 4 at the top cells.
- **Thinking on.** Item 5. Primary readout: the first stated horizon in the trace
  (as in E14, now pre-registered), plus the choice. E10/E14 found pre-reasoning
  edits are read back as discrete text edits (unit swaps, digit edits). The
  readout has to see that.

### What to defer

- Qwen3-14B and larger: 2.3× slower per generation (21.6 s), and the
  unit-overlap bank was 8B-only by decision on 2026-10-25.
- Full thinking-on sweeps (E): $19k-$31k.
- Thinking-on patching readout (H′): $26.
- The fill bank: +425 prompts, +~40% on stage 1, about $3 more. Add it if the
  units bank leaves within-unit magnitude ambiguous.
- A dense pre-horizon grid.
- A vLLM port: only worth it if stage 2 grows by 10× or more.

Per the workflow, this needs a spec and your approval before code. The new code is
the mid-prompt prefix-cached steering path and the cross-fit regression direction.

---

## Step 5. Sanity check

The arithmetic was run by `pipeline/s3_analysis/by_datatype/big_experiment_costs.py`. Hand checks:

- A: 36 × 134.7 × 500 × 2 + 500 = 4,849,700 runs. 4,849,700 / 77 / 3600 = 17.50 GPU-h,
  × $0.40 = $7.00.
- C: 1.6 + 0.0809 × 74.8 = 7.66 ms per run. × 4.85 M = 10.3 GPU-h.
- F: 22,500 × 11.66 s = 72.9 GPU-h.
- The cost model reproduces the full pass: 1.6 + 0.0809 × 141.7 = 13.07 ms = 1/76.5 s.

**The estimates are most sensitive to these, in order:**

1. **Seconds per thinking-on generation** (73% of the bundle). The measured range
   is 10.66-16.57 s, which moves the bundle 69-96 GPU-h (−6% / +31%). It rises
   if steering lengthens traces (more forced closes at 3,072) or batches shrink
   (the 5090 OOM'd at 20).
2. **The cost of mid-prompt cached steering** (new code, extrapolated from
   last-position edits). A fixed cost of 4 ms instead of 1.6 ms gives +12% (82 GPU-h).
   **If the prefix cache cannot be used at all, stage 1 goes from 19 to 60 GPU-h**
   (+$16 on 4090s).
3. **Card price and speed factor.** Unmeasured for H100/A100. 5090 = 4090
   suggests the stack is software-bound. The card choice alone swings the bundle
   from $30 to $144.

A full-pass rate of 60/s instead of 77/s adds only +5%.

### Pilot (~1,000 runs; Jared runs it on the box, about 15 min GPU, ≈ $0.20 on a 4090)

1. **Cached vs full equivalence and speed.** 4 prompts (2 regimes, 2 units) × twin
   × 4 layers (8, 16, 24, 32) × 16 post-horizon positions × ±α = 1,024 steered runs.
   Compute each one twice: full pass and prefix cache. Report the max |Δ log-odds|
   between paths (expect bf16 noise) and ms/run at batch 32, 72 and 128.
   Go if cached ≤ 6 ms/run.
2. **Steered generation.** 24 steered + 24 unsteered thinking-on generations at
   T = 0.6, batch 12 then 16/24. Report s/gen, tok/s, trace length and forced share.
   Go if ≤ 14 s/gen.
3. **Capture.** All layers × positions for 64 prompts: seconds and GB written.

Weighing an H100? Run the same pilot on one for about $0.70. Rent it for the bundle
only if it is ≥ 4.8× faster than the 4090, or ≥ 3× faster than the 5090.

### Price sources (Nov 2026 snapshots; check the live Vast.ai listings before renting)

- [gpu-cost.pages.dev, Vast.ai](https://gpu-cost.pages.dev/provider/vast-ai/), [RTX 4090](https://gpu-cost.pages.dev/gpu/rtx-4090/)
- [computeprices.com, Vast 4090](https://computeprices.com/providers/vast/gpus/rtx4090), [Vast 5090](https://computeprices.com/providers/vast/gpus/rtx5090)
- [deploybase.ai, Vast.ai pricing guide](https://deploybase.ai/articles/vast-ai-gpu-cloud-pricing-complete-guide-vs-hr-for-every-gpu), [5090 on Vast.ai](https://deploybase.ai/articles/rtx-5090-on-vastai-pricing-specs-how-to-rent)
- [Beam, RTX 5090 pricing](https://www.beam.cloud/blog/nvidia-rtx-5090-pricing), [gpus.io, RTX 5090](https://gpus.io/en/gpus/rtx5090)
- [DEV Community, RunPod vs Vast.ai vs Lambda 2026](https://dev.to/jovan_chan_9500711396d4e6/cloud-gpu-pricing-compared-runpod-vs-vastai-vs-lambda-labs-2026-35bn)
