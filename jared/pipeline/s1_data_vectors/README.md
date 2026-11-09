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
| `$RESULTS/exp0_<model>_think-<on/off>_<bank>/` (saved activations, answers.json) | this stage, read by its own analysis scripts |
| `$RESULTS/exp14_<model>_phrase/` (phrase states) | this stage (`exp14_phrase_states`), read by `scan_prepare`, `patch_prepare` |

`$RESULTS` and the other roots are defined in `../README.md` (What lives where); on the box `--out` is `/workspace/results/horizon`.

## Outputs

| path pattern | written by | read by |
|---|---|---|
| `$RESULTS/exp0_<tag>/{activations.npz, answers.json, manifest.json, *.png}` | `exp0_baseline` | this stage's analysis, Stage 3 |
| `$RESULTS/exp0_<tag>/chunks/chunk_NNN.{npz,json}`, `answers.json` | `exp0_choice` (choice4/choice8 banks) | Stage 3 (`choice_structure`) |
| `$RESULTS/coherence_<tag>/` | `coherence` | Stage 3 |
| `$RESULTS/exp14_<model>_phrase/states.npz` | `exp14_phrase_states` | `scan_prepare`, `patch_prepare` |
| `steering/inputs_<model>_<name>.{npz,json}` | `steering_prepare` | Stage 2 (`exp8_steering`, `exp13_suffix_steer`) |
| `steering/exp16_<model>_patch/{inputs.json,inputs.npz,fit_report.txt}` | `patch_prepare` | Stage 2 (`exp16_phrase_patch`) |
| `$RESULTS/exp15_<model>_scan/{inputs.json,inputs.npz,fit_report.txt}` | `scan_prepare` | Stage 2 (`exp15_scan`) |
| `$RESULTS/analysis/*.{json,txt,png}` (probes, shape, stability, manifold fits) | the analysis scripts | the docs, Stage 3 |

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
| `probe_dimensionality.py` | E7: linear vs MLP probes of log horizon on the top-k PCs, per layer and position | `python -m pipeline.s1_data_vectors.analysis.probe_dimensionality $RESULTS` |
| `probe_natural_within_unit.py` | E7 follow-up: thinking-on probes, within-unit R^2 | `python -m pipeline.s1_data_vectors.analysis.probe_natural_within_unit $RESULTS` |
| `horizon_shape.py` | PCA of horizon averages at key cells | `python -m pipeline.s1_data_vectors.analysis.horizon_shape $RESULTS` |
| `stage_markers.py` | encode/spread/hand-off/bind layers for 8B and 14B | `python -m pipeline.s1_data_vectors.analysis.stage_markers $RESULTS` |
| `cross_model_alignment.py` | 8B vs 14B layer-by-layer CKA of horizon averages | `python -m pipeline.s1_data_vectors.analysis.cross_model_alignment $RESULTS` |
| `pristine_manifold.py` | horizon manifold free of unit word, regime, forced closes, order | `python -m pipeline.s1_data_vectors.analysis.pristine_manifold $RESULTS` |
| `pristine_confounds.py` | does the pristine fit remove what the confound sweep found | `python -m pipeline.s1_data_vectors.analysis.pristine_confounds $RESULTS` |
| `direction_stability.py` | are the steering directions stable enough to read a steering result | `python -m pipeline.s1_data_vectors.analysis.direction_stability $RESULTS` |
| `caa_confound.py` | what the CAA direction is made of (forced closes, regime) | `python -m pipeline.s1_data_vectors.analysis.caa_confound $RESULTS OUT_DIR` |
| `off_curve_content.py` | what the off-curve part of the steering subspace holds | `python -m pipeline.s1_data_vectors.analysis.off_curve_content $RESULTS` |
| `exp14_geometry.py` | decomposition of the phrase-state change between own and donor horizon | `python -m pipeline.s1_data_vectors.analysis.exp14_geometry $RESULTS/exp14_Qwen3-8B_phrase` |
| `steering_prepare.py` | per-cell directions and cached reasoning: the inputs of E8, E13 | `python -m pipeline.s1_data_vectors.analysis.steering_prepare $RESULTS` |
| `patch_prepare.py` | inputs of the phrase number patch | `python -m pipeline.s1_data_vectors.analysis.patch_prepare --phrase-dir $RESULTS/exp14_Qwen3-8B_phrase` |
| `scan_prepare.py` | inputs of the Part B steering scan | `python -m pipeline.s1_data_vectors.analysis.scan_prepare --phrase-dir $RESULTS/exp14_Qwen3-8B_phrase --e0-root $RESULTS` |

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

The code, queues and docs of this stage were moved here from `jared/experiments/`, `jared/analysis/`,
`jared/queue*.sh` and the numbered docs in `jared/`, with imports rewritten to
`pipeline.s1_data_vectors...`. Old paths: `git log --follow <new path>`.
`steering_prepare.OUT` still points at `jared/steering/`.

## Docs

`docs/INDEX.md`: one line per doc, labelled spec / result / draft.

## Known gaps

- `queue.sh` and `queue_14b*.sh` start Stage 2 scripts (`exp8_steering`, `exp10_patch_reasoning`)
  by their `python -m pipeline.s2_steering...` paths. Cross-stage imports: see `../README.md`.
- The per-stage `docs/` hold results as written, with only file paths updated.
