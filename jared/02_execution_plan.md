# Execution plan: first experiments, local test, ship to GPU

Status: Phase A and B DONE (2026-09-23). Phase C (ship to GPU) waits on Jared's vast.ai setup.
Date: 2026-09-23
Companion to `01_plan.md` (the what and why). This file is the how.

## 1. Goal of this round

Get the experiments below from `01_plan.md` running end to end, verified on the
laptop with Qwen3.5-0.8B on a handful of prompts, then pushed to the vast.ai
4090 and run on Qwen3-8B with thinking on and off. Results come back as
self-describing JSON plus figures.

The three chosen are the cheapest to build and each answers something on
its own:

| Experiment | Plan row | Why first |
|---|---|---|
| E0 baseline | Phase 1 | Already exists. Establishes peak layer, probe layer, and whether the 8B model reads content. Everything else keys off its layer choice. |
| E1 nuisance and specificity | R4 | Reuses the E0 prompt bank with swap / relabel / reward recolor. No new prompts, one new analysis. Cheapest possible second result. |
| E2 paraphrase transfer | R1 | First real robustness test. Needs a small prompt-template registry, which E3 and the domain test later reuse. |
| E3 unit invariance | R3 | Same registry, plus a duration-to-unit spelling table. Tests whether the axis is log-time or surface form. |
| E4 prompt perturbation | R1 ext. | Jared's addition (2026-09-23). Structural edits that keep the horizon fixed: distractor sentence, horizon moved before the options, typo in the horizon word, number spelled out, system-prompt persona. Probe transfer plus behavior stability under each. |
| E5 attribution patching | new | Jared's addition. Which layers, positions, and components carry the horizon signal to the answer. Clean/corrupt pairs differ only in horizon; metric is the a) vs b) logit difference at the answer position. Reuses the prior repo's design (`src/attribution_patching/`). |
| 3D PCA | E0/E1 ext. | Jared's addition. PC1-3 per token at the display layer, colored by horizon, with explained variance and Spearman for PC2 and PC3. Shows whether the horizon manifold is a curve in 3D rather than a line. |

Deferred to the next round: R2 domain transfer (needs new scenario text
that should be written carefully, not rushed) and R5 steering (uses the E5
attribution map to choose where to steer).

## 2. Code layout

`horizon_geometry.py` stays the notebook-shaped entry point for E0. The
shared pieces move into a small importable package so the experiments do
not copy code:

```
jared/
  spar_horizon/
    __init__.py
    prompts.py       # choice_prompt, build_prompts, template registry, unit spellings
    model_io.py      # load_model, chat_text, suffix_length, generate, answer_start
    extract.py       # extract() -> activations, tokens, answers, kept, n_suffix
    geometry.py      # PCA sweep, horizon probe (ridge on log horizon, k-fold), transfer matrix
    behavior.py      # parse_choice, three coherence tests
    attribution.py   # hooks on resid/attn_out/mlp_out, (clean - corrupt) * grad, logit-diff metric
    plots.py         # 2D and 3D PCA panels, transfer heatmaps, attribution heatmaps
    runs.py          # OUT_DIR resolution, manifest.json writer, save/load npz
  experiments/
    exp0_baseline.py
    exp1_nuisance.py
    exp2_paraphrase.py
    exp3_units.py
    exp4_perturb.py
    exp5_attribution.py
    run_all.sh       # runs every experiment for THINKING in on off
  tests/
    test_prompts.py  # counts, uniqueness, metadata round-trip, unit table
    test_behavior.py # parser on real 0.8B outputs ("a)<|im_end|>", "2)", empty)
    test_model_io.py # suffix_length == 9 for Qwen3.5 off, 7 on; think-end detection
    test_geometry.py # probe recovers a planted linear signal; transfer matrix shape
  requirements.txt   # scipy, scikit-learn, matplotlib, numpy (torch/transformers from the image)
  horizon_geometry.py / .ipynb   # thin: imports the package, keeps the knobs cell
```

Every experiment takes the same CLI: `--model`, `--thinking on|off`,
`--limit N`, `--out DIR`. On the box `--out` defaults to
`/workspace/results/horizon/<exp>_<model>_think-<on|off>/` so `just sync`
brings it home. Locally it defaults to `out/`.

Every run writes `manifest.json` (model, thinking, git hash, knobs, prompt
count, wall time, skipped count) next to its figures and npz. A run without
a manifest is treated as failed.

## 3. Task list

Each task is small, has a check, and is done before the next starts.

### Phase A: package and tests (local, no GPU, no model download beyond the 0.8B already cached)

- [x] A1 Create `spar_horizon/` by moving code out of `horizon_geometry.py`, unchanged in behavior. Check: the 6-prompt 0.8B smoke test gives the same suffix count and behavior lines as before.
- [x] A2 `tests/test_prompts.py`: 116 prompts, no duplicates, horizon metadata reparses from the prompt text. Check: pytest green.
- [x] A3 `tests/test_behavior.py` using the real strings the 0.8B model emitted. Check: green.
- [x] A4 `tests/test_model_io.py` with the real Qwen3.5 tokenizer (cached locally). Check: 9 off, 7 on, `</think>` id found.
- [x] A5 Horizon probe in `geometry.py`: ridge regression residual to log horizon, 5-fold, returns R² per layer and the unit direction. `tests/test_geometry.py` plants a linear signal in random data and checks recovery. Check: green.
- [x] A6 `runs.py` manifest writer and OUT_DIR resolution. Check: a run under a fake `WORKSPACE` env var lands in the results path.

### Phase B: experiments (local on 0.8B with `--limit 12`, thinking off only, since 0.8B never closes its think block)

- [x] B1 `exp0_baseline.py`: E0 as a CLI. Adds the probe table and saves the best-layer direction to npz. Check: runs on 12 prompts, manifest present.
- [x] B2 `exp1_nuisance.py`: loads E0's npz if present, else extracts. Recolors PC1/PC2 by reward pair and by delay pair, reports horizon rho vs reward rho per position at the display layer. Check: runs, figure has two colorings.
- [x] B3 Template registry in `prompts.py`: 5 constraint paraphrases and 3 framings, each a format string with the same slots. Test in A2 extended. Check: 5×3 variants of one prompt all parse.
- [x] B4 `exp2_paraphrase.py`: for each template, extract at the probe layer only (cheap), train probe on template i, test on j, write the transfer matrix as CSV and heatmap. Check: diagonal high on 0.8B, matrix is 5×5 or 15×15 depending on B3 decision.
- [x] B5 Unit spellings in `prompts.py`: for durations {1 hour, 1 day, 1 week, 1 month, 1 year, 10 years}, 3 spellings each (e.g. "1 year", "12 months", "365 days"). Check: test.
- [x] B6 `exp3_units.py`: project each spelling onto the probe axis, report within-duration spread vs between-duration spread. Check: runs, one bar chart.
- [x] B7 3D PCA in `plots.py`, wired into E0 and E1: PC1-3 scatter per token at the display layer, explained-variance table, Spearman for PC2/PC3. Check: figure renders on the 0.8B smoke run; PC1 Spearman matches the 2D sweep.
- [x] B8 Perturbation registry in `prompts.py`: each perturbation is a function prompt -> prompt that leaves the horizon text intact. Test asserts the horizon reparses after every perturbation. Check: green.
- [x] B9 `exp4_perturb.py`: for each perturbation, extract at probe layers, probe transfer from the clean template, and the three behavior tests. Output: one row per perturbation with probe rho, order stability, label stability. Check: runs on 0.8B.
- [x] B10 `attribution.py`: forward hooks that retain activations and grads at resid_post, attn_out, mlp_out per layer; metric = logit(a) - logit(b) at the answer position; score = (clean - corrupt) * grad, summed over d_model. Params frozen, only activation grads. Test: on a 2-layer random model the score for an untouched position is 0. Check: green.
- [x] B11 `exp5_attribution.py`: pairs = same rewards/delays with a short and a long horizon whose rational answers differ. Both directions (noising and denoising). Output: layer × position heatmap per component, top-10 components table. Check: runs on 0.8B thinking off; heatmap not all zeros.
- [x] B12 `run_all.sh`: loops experiments × thinking modes, stops on first failure, prints wall time per run. E5 runs thinking off only unless decided otherwise. Check: runs locally with `--limit 6`.

### Phase C: ship (needs Jared: vast.ai account, .env, HF token optional)

- [x] C1 `requirements.txt` for the box (written; not yet installed anywhere but the mp env).
- [ ] C2 `just up`, `just push spar-planning-temporal-manifolds`, then on the box: `pip install -r .../requirements.txt`. Check: `python -c "import spar_horizon"` on the box.
- [ ] C3 Smoke on the box: `exp0_baseline.py --model Qwen/Qwen3-8B --thinking off --limit 12`. Check: manifest, GPU in use, under 2 minutes.
- [ ] C4 Smoke thinking on, `--limit 12`. Check: skipped count is 0 or near it; median think length recorded. This sets `MAX_THINK_TOKENS` for the full run.
- [ ] C5 Full `run_all.sh`. Check: 8 manifests (4 experiments × 2 modes). `just sync`, then `just stop`.
- [ ] C6 Write results into `04_results_qwen3-8b.md` with the R1/R3/R4 pass/fail rows from `01_plan.md` filled in.

## 4. Estimates

| Item | Estimate |
|---|---|
| Phase A | 2-3 hours of work, all local |
| Phase B | 3-4 hours, local 0.8B runs are seconds to minutes at `--limit 12` |
| Phase C GPU time | E0 and E1 minutes each. E2 is 5 templates × 116 prompts, E3 is 18 × 6, E4 is ~5 × 116 × 3 variants. E5 is ~50 pairs × 3 passes with gradients, bf16 8B on 24 GB is fine with params frozen. Thinking on multiplies generation cost ~50-100×. Whole run_all under 3 GPU-hours, ~$1.50 at 4090 rates. |

## 5. Decisions

Decided by Jared (2026-09-23):
1. **E2 size:** 5 constraint paraphrases only, no framings. 580 prompts per
   thinking mode, a 5x5 transfer matrix. Framings are a later round.
2. **Layers for E2/E3:** E0's best probe layer +/- 2 (five layers).
   `run_all.sh` runs E0 first so the layer choice exists.

Assumed unless Jared says otherwise:
3. Package name `spar_horizon`.
4. Tests under `jared/tests/` so `just push` ships them.
5. Nothing in the box's `/workspace/results/` collides with a `horizon/`
   subfolder.

Revision 2026-09-23: add prompt perturbation (E4), attribution patching
(E5), and 3D PCA. Jared's choices:
6. **E2 paraphrases** are the wording variants Jared named: deadline, time
   limit, "must be realized within", "you need the money by", planning
   horizon. Five constraint templates, all explicit about the duration.
7. **E4 perturbations:** (a) implicit horizon, where the duration is implied
   by a life event ("before my daughter starts college in 5 years") rather
   than stated as a constraint; (b) distractor sentence between options and
   constraint; (c) horizon sentence moved before the options. A couple of
   structural ones plus the implicit form, per Jared.
8. **E5 method:** standard attribution patching, own hooks, no extra deps.
9. **E5 thinking:** off only.

**Phase A and B approved and completed 2026-09-23.** Phase C still needs Jared.

## 7. Build notes (what changed from the plan while building)

- The probe uses RidgeCV over a log grid of penalties instead of a fixed
  alpha; a fixed alpha overfit 4096-dim activations on ~100 samples.
- Attribution feeds the embeddings as a grad-requiring leaf, since all
  parameters are frozen and otherwise nothing in the graph carries grad.
- E5 pairs are generated from integer month/year phrases with matching digit
  counts (Qwen tokenizes digit by digit), giving 48 equal-length pairs
  instead of 6 from the fixed horizon list. E5 also reports how many pairs
  flip the a/b metric sign, which says whether the metric is meaningful on
  that model.
- E3's smoke subset keeps whole spelling groups so the within-duration
  spread is defined.
- The behavior tests accept a prompt builder so E4 can run them on perturbed
  and implicit-horizon prompts.
- Local smoke of all six experiments on the 0.8B (6 prompts, thinking off)
  takes about 8 minutes on CPU and writes every manifest.
- **Thinking on, Qwen3-8B (box smoke 2026-09-23):** every prompt exceeded a
  1024-token think budget. A 6144-token probe showed a 5-year prompt closing
  at 2705 tokens with the correct answer and reasoning, but a 300-second
  prompt looping past 6150 tokens ("neither is feasible, maybe a typo").
  Short horizons loop; a bigger budget cannot fix that. Two changes:
  (1) the think block is **force-closed** at `max_think_tokens` (now 3072)
  by appending `</think>`, so every prompt yields an answer and hidden
  states, and each run records how many were forced; (2) generation is
  **batched** (left-padded greedy, `batch_size` 8), verified token-identical
  to single-prompt generation on the 0.8B. Single-stream E0 with thinking on
  would have been ~8 GPU-hours; batched it is about one.
- **Answer position (2026-09-24):** Qwen3-8B thinking off writes a paragraph
  before naming its choice, so the 6-token answer window held prose and the
  behavior tests had almost no parseable answers. Jared chose an assistant
  prefill (`I choose:`) over regex localization. Leading formatting tokens
  before the label are skipped. The thinking-off half must be rerun; the
  suffix-position findings from the first run stand and are archived.

## 6. What I will not do without asking

Rent the box, push anything to it, or commit to git. Everything in Phase A
and B runs locally and writes only under `jared/`.
