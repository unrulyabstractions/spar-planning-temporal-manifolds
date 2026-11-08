# Stage 1: initial data and steering-vector extraction

## Question

How do we get prompts, model activations, and a steering vector? This stage builds the prompt banks'
runs (E0: generate, record activations, read out behaviour), probes and describes the horizon
representation, and fits the directions and curves that Stage 2 pushes on.

## Inputs

| path pattern | from |
|---|---|
| `spar_horizon/` (banks, prompts, models, probes; shared library) | in the repo |
| model weights (`--model`, default Qwen3-8B) | Hugging Face, on the box |
| `<out>/exp0_<model>_think-<on/off>_<bank>/` (saved activations, answers.json) | this stage, read by its own analysis scripts |
| `<out>/exp14_<model>_phrase/` (phrase states) | this stage (`exp14_phrase_states`), read by `scan_prepare`, `patch_prepare` |

`<out>` is `--out`, else `$WORKSPACE/results/horizon`, else `jared/out`.

## Outputs

| path pattern | written by | read by |
|---|---|---|
| `<out>/exp0_<tag>/{activations.npz, answers.json, manifest.json, *.png}` | `exp0_baseline` | this stage's analysis, Stage 3 |
| `<out>/exp0_<tag>/chunks/chunk_NNN.{npz,json}`, `answers.json` | `exp0_choice` (choice4/choice8 banks) | Stage 3 (`choice_structure`) |
| `<out>/coherence_<tag>/` | `coherence` | Stage 3 |
| `<out>/exp14_<model>_phrase/states.npz` | `exp14_phrase_states` | `scan_prepare`, `patch_prepare` |
| `steering/inputs_<model>_<name>.{npz,json}` | `steering_prepare` | Stage 2 (`exp8_steering`, `exp13_suffix_steer`) |
| `steering/exp16_<model>_patch/{inputs.json,inputs.npz,fit_report.txt}` | `patch_prepare` | Stage 2 (`exp16_phrase_patch`) |
| `<out>/exp15_<model>_scan/{inputs.json,inputs.npz,fit_report.txt}` | `scan_prepare` | Stage 2 (`exp15_scan`) |
| `<out>/analysis/*.{json,txt,png}` (probes, shape, stability, manifold fits) | the analysis scripts | the docs, Stage 3 |

## Contents

Run every command from `jared/`.

Experiments (`experiments/`, need a model, run on the box):

| script | what it does | example |
|---|---|---|
| `common.py` | shared CLI (`parse_args`); Stage 2 imports it | not run directly |
| `exp0_baseline.py` | E0: generate on a bank, record activations at all layers, horizon probe, PCA | `python -m pipeline.s1_data_vectors.experiments.exp0_baseline --model Qwen/Qwen3-8B --thinking on --bank full --out /workspace/results/horizon` |
| `exp0_choice.py` | E0 for the 4/8-option banks: chunked, resumable, reduced layers | `python -m pipeline.s1_data_vectors.experiments.exp0_choice --model Qwen/Qwen3-8B --bank choice4 --batch-size 12` |
| `exp14_phrase_states.py` | hidden states at the horizon phrase, one forward pass per prompt | `python -m pipeline.s1_data_vectors.experiments.exp14_phrase_states --prompts steering/phrase_prompts_Qwen3-8B.json --out /workspace/results/horizon` |
| `warm_cache.py` | fill the generation cache for one shard so generation splits over GPUs | `CUDA_VISIBLE_DEVICES=0 python -m pipeline.s1_data_vectors.experiments.warm_cache --model Qwen/Qwen3-14B --thinking on --bank full --shard 0/2` |
| `coherence.py` | the paper's coherence tests with thinking on, from a warm cache | `python -m pipeline.s1_data_vectors.experiments.coherence --model Qwen/Qwen3-14B --thinking on --bank full` |
| `check_prefix_cache.py` | does shared-prefix scoring match full forward passes in bf16 | `python -m pipeline.s1_data_vectors.experiments.check_prefix_cache --model Qwen/Qwen3-8B` |

Analysis (`analysis/`, saved activations only, run locally; `steering_prepare`, `patch_prepare`,
`scan_prepare` load a tokenizer or small model pieces but generate nothing):

| script | what it does | example |
|---|---|---|
| `probe_dimensionality.py` | E7: linear vs MLP probes of log horizon on the top-k PCs, per layer and position | `python -m pipeline.s1_data_vectors.analysis.probe_dimensionality results/horizon` |
| `probe_natural_within_unit.py` | E7 follow-up: thinking-on probes, within-unit R^2 | `python -m pipeline.s1_data_vectors.analysis.probe_natural_within_unit results/horizon` |
| `horizon_shape.py` | PCA of horizon averages at key cells | `python -m pipeline.s1_data_vectors.analysis.horizon_shape results/horizon` |
| `stage_markers.py` | encode/spread/hand-off/bind layers for 8B and 14B | `python -m pipeline.s1_data_vectors.analysis.stage_markers results/horizon` |
| `cross_model_alignment.py` | 8B vs 14B layer-by-layer CKA of horizon averages | `python -m pipeline.s1_data_vectors.analysis.cross_model_alignment results/horizon` |
| `pristine_manifold.py` | horizon manifold free of unit word, regime, forced closes, order | `python -m pipeline.s1_data_vectors.analysis.pristine_manifold results/horizon` |
| `pristine_confounds.py` | does the pristine fit remove what the confound sweep found | `python -m pipeline.s1_data_vectors.analysis.pristine_confounds results/horizon` |
| `direction_stability.py` | are the steering directions stable enough to read a steering result | `python -m pipeline.s1_data_vectors.analysis.direction_stability results/horizon` |
| `caa_confound.py` | what the CAA direction is made of (forced closes, regime) | `python -m pipeline.s1_data_vectors.analysis.caa_confound results/horizon OUT_DIR` |
| `off_curve_content.py` | what the off-curve part of the steering subspace holds | `python -m pipeline.s1_data_vectors.analysis.off_curve_content results/horizon` |
| `exp14_geometry.py` | decomposition of the phrase-state change between own and donor horizon | `python -m pipeline.s1_data_vectors.analysis.exp14_geometry <phrase dir>` |
| `steering_prepare.py` | per-cell directions and cached reasoning: the inputs of E8, E13 | `python -m pipeline.s1_data_vectors.analysis.steering_prepare results/horizon` |
| `patch_prepare.py` | inputs of the phrase number patch | `python -m pipeline.s1_data_vectors.analysis.patch_prepare --phrase-dir results/horizon/exp14_Qwen3-8B_phrase` |
| `scan_prepare.py` | inputs of the Part B steering scan | `python -m pipeline.s1_data_vectors.analysis.scan_prepare --phrase-dir results/horizon/exp14_Qwen3-8B_phrase --e0-root results/horizon` |

Each script's docstring has its exact arguments; `exp14_geometry` and `caa_confound` take the paths
given there.

## How to run

Experiments run on the GPU box (`just up 2h`, `just push`, then a queue file in tmux). Analysis runs
locally with `~/miniconda3/envs/mp/bin/python` on results synced with `just sync`. Never run
model-backed code from a Claude session (see the project `CLAUDE.md`).

| queue | what it runs |
|---|---|
| `queues/queue.sh` | E10 patching split over GPUs, plus the E8 decision split at L22 (runs Stage 2 scripts) |
| `queues/queue_14b.sh` | E11: the 8B map on Qwen3-14B (`exp0_baseline`, `steering_prepare`, E8) |
| `queues/queue_14b_rest.sh` | the rest of E11, with E8's shared-prefix scoring |
| `queues/queue_units.sh` | E0 on the units (default) or fill bank, thinking on and off |
| `queues/queue_multichoice.sh` | E0 on the choice4/choice8 banks, then `choice_structure` (Stage 3) |

```
tmux new -s units 'bash pipeline/s1_data_vectors/queues/queue_units.sh; bash'
tmux new -s fill 'BANK=fill bash pipeline/s1_data_vectors/queues/queue_units.sh; bash'
```

Fast tests (no model), from `jared/`:
`~/miniconda3/envs/mp/bin/python -m pytest -q tests --ignore=tests/test_generation.py`.
Tests stay in `jared/tests/`; those for this stage are `test_exp0_choice`, `test_choice_wiring`,
`test_exp14_states`, `test_warm_cache`, `test_probe_natural_within_unit`, `test_cross_model_alignment`,
`test_stage_markers`, `test_pristine_manifold`, `test_pristine_confounds`, `test_direction_stability`,
`test_off_curve_content`, `test_exp14_geometry`, `test_steering_prepare`, `test_phrase_prepare`,
`test_patch_prepare`, `test_scan_prepare`, `test_pristine_steering`, and the CAA part of
`test_autoresearch_2026_11_03`.

## Moved from

| old | new |
|---|---|
| `experiments/{exp0_baseline,exp0_choice,exp14_phrase_states,warm_cache,check_prefix_cache,coherence,common}.py` | `pipeline/s1_data_vectors/experiments/` |
| `analysis/{probe_dimensionality,probe_natural_within_unit,horizon_shape,direction_stability,pristine_manifold,pristine_confounds,steering_prepare,patch_prepare,scan_prepare,cross_model_alignment,stage_markers,exp14_geometry,caa_confound,off_curve_content}.py` | `pipeline/s1_data_vectors/analysis/` |
| `queue.sh`, `queue_14b.sh`, `queue_14b_rest.sh`, `queue_units.sh`, `queue_multichoice.sh` | `pipeline/s1_data_vectors/queues/` |
| `04_spec_probes.md`, `06_results_probes_full.md`, `10_spec_scale_14b.md`, `11_results_14b.md`, `14_spec_unit_overlap.md`, `16_spec_pristine_manifold.md`, `17_results_pristine.md`, `18_spec_multichoice.md`, `19_results_fill.md`, `20_results_overnight_fill.md` | `pipeline/s1_data_vectors/docs/` |

Scripts that patched `sys.path` now import as `pipeline.s1_data_vectors...`, and `steering_prepare.OUT`
still points at `jared/steering/`.

## Docs

- `04_spec_probes.md`: E7, linear and MLP probes on the top-k principal components.
- `06_results_probes_full.md`: probe results on the full bank (Qwen3-8B).
- `10_spec_scale_14b.md`: E11, the 8B map repeated on Qwen3-14B, with the stage markers.
- `11_results_14b.md`: results of E11.
- `14_spec_unit_overlap.md`: the unit-overlap (units) bank, horizon vs unit word.
- `16_spec_pristine_manifold.md`: E13, a horizon manifold free of the unit word, regime, forced closes, order.
- `17_results_pristine.md`: the E13 prototype on the 8B units bank.
- `18_spec_multichoice.md`: the 4- and 8-choice investment banks.
- `19_results_fill.md`: the fill bank pooled with the old 8B runs.
- `20_results_overnight_fill.md`: second overnight, after the fill run.

## Status and known gaps

- Nothing was rerun in the reorganisation; only paths and imports changed. Compile checks and the
  model-free tests pass for this stage.
- `queue.sh` and `queue_14b*.sh` mix stages: they start Stage 2 scripts (`exp8_steering`,
  `exp10_patch_reasoning`) by their new `python -m pipeline.s2_steering...` paths.
- Analysis scripts here import helpers from Stage 3 (`unit_overlap`, `written_unit`, `confound_sweep`,
  `exp14_analysis`, `regime_comparison`) and Stage 2 (`steering_analysis`, `exp8_steering`,
  `exp10_patch_reasoning`); these are cross-stage dependencies.
- `spar_horizon` stays the shared library; the per-stage `docs/` hold results as written, with
  only file paths updated.
