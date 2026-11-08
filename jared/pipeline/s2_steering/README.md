# Stage 2: testing and steering, by modality

## Question

Given an input and a steering vector (or a donor state), does the behaviour move? Each modality
below is one way of intervening on the model and one question it answers. All of it needs the
model (box runs, thinking on unless noted); the `analysis/` scripts here are the CPU summaries that
only make sense next to one runner.

| Modality (`experiments/`) | Runners | Question it answers | Spec | Results |
|---|---|---|---|---|
| `patch/` donor patching | `exp10_patch_reasoning`, `exp16_phrase_patch` | Does the horizon phrase (and which part of it: number, unit, subspace) carry the horizon the model reasons from? Swap in a donor state and read what the trace states and what is chosen. | `07` §12 (E10), `27` (E16) | `09`, `30` |
| `ablate_steer/` ablate, steer, patch at the answer slot | `exp8_steering` | Is the probe-found horizon subspace causal? Mean-ablation, line/curve/random steering and answer-slot patching, thinking off and on cached reasoning; horizon vs decision split. | `07` (E8), `15_spec_same_subspace_sweep` (E12 draft) | `08`, `09`, `18_results_overnight` |
| `manifold_steer/` steer along the manifold | `exp13_suffix_steer`, `exp14_phrase_steer`, `exp15_scan` | Does moving the state along the effects curve (at the thinking-on suffix E13, at the horizon phrase E14, over a layer x position x direction scan E15) change the horizon the model reasons from and its choice? Is it horizon-specific (delay crossing)? | `15_spec_crossing`, `21` (E14), `25` (draft), `26_spec_overnight_choice_scan` (E15) | `22`, `26_results_overnight_choice_scan`, `30` |
| `think_budget/` | `exp17_think_budget` | Are the force-closed traces of a k-option bank (choice4/8) a think-budget artefact? Extend the budget and see if the choice changes. | `28` (E17) | `30` |
| `units_implied/` | `exp9_units`, `exp18_implied` | E9: does the horizon representation track the duration or the unit word? E18: do prompts that imply a horizon (no stated duration) put the model in the same state as stated ones? | `07` §11 (E9), `29` (E18) | `30` |

Specs fix their criteria before a run; results docs record what happened. Findings are not
restated here.

## Inputs

Produced by stage 1 (`pipeline/s1_data_vectors/`), read-only here:

- `steering/inputs_<model>_<variant>.{json,npz}`: cells, directions, donors, steps. Written by
  `steering_prepare.py` (E8, E13). Read by `exp8_steering`, `exp13_suffix_steer`, `crossing_split`,
  `steering_analysis`.
- `steering/exp14_*`, `steering/exp15_<model>_scan/`, `steering/exp16_<model>_patch/`
  (`inputs.{json,npz}`, `fit_report.txt`): written by `steering_prepare.py --phrase`,
  `scan_prepare.py`, `patch_prepare.py` after `exp14_phrase_states` has saved phrase states
  (`<out>/exp14_<model>_phrase/states_*.npz`). Read by `exp14_phrase_steer`, `exp15_scan`,
  `exp16_phrase_patch`.
- `steering/exp17_<model>_think_budget/stage0_<bank>.{json,npz}`: written by
  `think_budget_stage0.py` (this stage, CPU, from the saved E0 choice runs `out/exp0_*/`).
- `out/exp0_*/` runs (saved activations, answers, reasoning traces) for `exp9_units`,
  `exp18_implied`, and as the E0 reference in several summaries.

## Outputs

Run folders under the results root (`--out`, `/workspace/results/horizon` on the box, synced
down to `results/horizon` or `out/`); the code paths are unchanged by the reorganisation:

| Pattern | Written by | Read by |
|---|---|---|
| `<out>/exp8_<inputs name>/results.json` | `exp8_steering` | `steering_analysis`, `crossing_split`, stage 3 |
| `<out>/exp10_<model>_reasoning/` | `exp10_patch_reasoning` | `exp10_analysis` |
| `<out>/exp13_<inputs name>/` | `exp13_suffix_steer` | `crossing_split`, stage 3 (`by_label`) |
| `<out>/exp14_<model>_phrase/` (steer units, `summary.txt`) | `exp14_phrase_steer` | `s3_analysis.by_label.exp14_analysis` |
| `<out>/exp15_<model>_scan/` (cell files, `scan_summary.{txt,json}`) | `exp15_scan` | `scan_summary`, `patch_summary` |
| `<out>/exp16_<model>_patch/` (unit files, `patch_summary.{txt,json}`) | `exp16_phrase_patch` | `patch_summary`, `s3_analysis.figures.patch_figures` |
| `<out>/exp17_<model>_think_budget/<bank>/` (chunks, `think_budget_summary.{txt,json}`) | `exp17_think_budget` | `think_budget_summary` |
| `<out>/exp9_<model>_units/` | `exp9_units` | `s3_analysis.by_datatype` unit scripts |
| `<out>/exp18_<model>_implied/units/`, `implied_summary.{txt,json}` | `exp18_implied` | `implied_summary`, `s3_analysis.figures.choice4_implied_figures` |
| `crossing_split_pristine.{json,txt,png}`, `figures/` | `crossing_split`, `steering_analysis`, `exp10_analysis` | docs `15`, `08` |

## Contents

Run everything from `jared/` as `python -m pipeline.s2_steering...`. Examples are the real
invocations from the scripts' own docstrings (box paths; add `--limit` or a 0.8B `--model` for a
smoke).

| Script | What it does | Example |
|---|---|---|
| `experiments/ablate_steer/exp8_steering.py` | E8: patch, ablate, steer, post, answer-slot tests | `python -m pipeline.s2_steering.experiments.ablate_steer.exp8_steering --inputs steering/inputs_Qwen3-8B_answer_slot.npz --tests ablate,steer,post,answer --out /workspace/results/horizon` |
| `experiments/patch/exp10_patch_reasoning.py` | E10: patch the horizon phrase before reasoning | `python -m pipeline.s2_steering.experiments.patch.exp10_patch_reasoning --out /workspace/results/horizon` |
| `experiments/patch/exp16_phrase_patch.py` | E16: patch parts of the phrase of same-unit crossing pairs (13 units) | `python -m pipeline.s2_steering.experiments.patch.exp16_phrase_patch --dir /workspace/results/horizon/exp16_Qwen3-8B_patch` |
| `experiments/manifold_steer/exp13_suffix_steer.py` | E13 stage B: steer at the thinking-on suffix, then reason | `python -m pipeline.s2_steering.experiments.manifold_steer.exp13_suffix_steer --inputs steering/inputs_Qwen3-8B_pristine_suffix.npz --out /workspace/results/horizon` |
| `experiments/manifold_steer/exp14_phrase_steer.py` | E14: steer the horizon phrase toward a target horizon | `python -m pipeline.s2_steering.experiments.manifold_steer.exp14_phrase_steer --dir /workspace/results/horizon/exp14_Qwen3-8B_phrase` |
| `experiments/manifold_steer/exp15_scan.py` | E15: layer x position x direction scan, one cell at a time | `python -m pipeline.s2_steering.experiments.manifold_steer.exp15_scan --dir /workspace/results/horizon/exp15_Qwen3-8B_scan` |
| `experiments/think_budget/exp17_think_budget.py` | E17: continue force-closed traces with extra think tokens | `python -m pipeline.s2_steering.experiments.think_budget.exp17_think_budget --bank choice4 --unit 1` |
| `experiments/units_implied/exp9_units.py` | E9: same duration in different unit words | `python -m pipeline.s2_steering.experiments.units_implied.exp9_units --out /workspace/results/horizon` |
| `experiments/units_implied/exp18_implied.py` | E18: implied-horizon prompts, modules M1-M4 | `python -m pipeline.s2_steering.experiments.units_implied.exp18_implied --out /workspace/results/horizon --modules M1` |
| `analysis/exp10_analysis.py` | what the reasoning reads after the E10 patch | `python -m pipeline.s2_steering.analysis.exp10_analysis results/horizon` |
| `analysis/steering_analysis.py` | E8: horizon subspace or decision? figures for both steering runs | `python -m pipeline.s2_steering.analysis.steering_analysis results/horizon` |
| `analysis/crossing_split.py` | pristine steering, horizon-specific? the delay-crossing split | `python -m pipeline.s2_steering.analysis.crossing_split /home/Jared/SPAR/results/horizon` |
| `analysis/scan_summary.py` | E15 flips against the baseline, per cell | `python -m pipeline.s2_steering.analysis.scan_summary results/horizon/exp15_Qwen3-8B_scan` |
| `analysis/patch_summary.py` | E16 what a trace states and the flips, per unit | `python -m pipeline.s2_steering.analysis.patch_summary steering-or-results/exp16_Qwen3-8B_patch` |
| `analysis/think_budget_stage0.py` | E17 stage 0 (CPU): what the force-closed traces say, and the box inputs | `python -m pipeline.s2_steering.analysis.think_budget_stage0 --bank choice4` |
| `analysis/think_budget_summary.py` | E17 criteria P0, C1-C3 from whatever chunks exist | `python -m pipeline.s2_steering.analysis.think_budget_summary /workspace/results/horizon/exp17_Qwen3-8B_think_budget` |
| `analysis/implied_summary.py` | E18 measures and criteria C0-C4 | `python -m pipeline.s2_steering.analysis.implied_summary <dir> [--probe-runs <E0 run dir> ...]` |

The runners import `spar_horizon` (shared library), `pipeline.s1_data_vectors.experiments.common`,
`exp0_choice` and `exp14_phrase_states` (stage 1), and `exp14_analysis`, `written_unit`
(stage 3: `s3_analysis.by_label`, `s3_analysis.by_datatype`) for parsing what a trace states.
Within the stage: E10 supplies `generate`, `score`, `states_at` etc. to E13-E16, E8 supplies
`selected_records` and `load_inputs`, and `scan_summary` / `patch_summary` are imported by the
runners that write the files they read.

## How to run

On the box (Qwen3-8B, one GPU), from `jared/`, in tmux, through the queue files; each skips
finished work, so a rerun resumes:

| Queue | Runs |
|---|---|
| `queues/queue_pristine_steer.sh` | E13 stage A (`exp8_steering`) then stage B (`exp13_suffix_steer`) |
| `queues/queue_phrase_steer.sh` | E14: states (stage 1), fit (stage 1), `exp14_phrase_steer`, `exp14_analysis` (stage 3) |
| `queues/queue_scan.sh` | E15: states and `scan_prepare` (stage 1), `exp15_scan`, `scan_summary` |
| `queues/queue_patch.sh` | E16: `exp16_phrase_patch`, `patch_summary` (inputs prepared locally by `patch_prepare.py`) |
| `queues/queue_think_budget.sh` | E17: `SMOKE=1` first, then the real run, `think_budget_summary` |
| `queues/queue_implied.sh` | E18: modules M1-M4, `implied_summary` after each |

```
cd /workspace/SPAR/spar-planning-temporal-manifolds/jared
tmux new -s scan 'bash pipeline/s2_steering/queues/queue_scan.sh; bash'
```

Locally (CPU, Jared's `mp` env): only the `analysis/` summaries on synced results. The runners need
the model; per the project notes, model-backed runs and 0.8B smokes go through `local_checks.py`
on the xpu env, not the Claude session. Box lifecycle (`just up`, `push`, `sync`, `down`) is in
`SPAR/CLAUDE.md`.

Tests (fast, no model weights except tiny random-init Qwen3): `tests/test_exp8_runner.py`,
`test_exp14_steer.py`, `test_exp15_scan.py`, `test_exp16_patch.py`, `test_exp17_think_budget.py`,
`test_exp18_implied.py`, `test_pristine_steering.py`, `test_crossing_split.py`, `test_scan_summary.py`,
`test_patch_summary.py`, `test_think_budget_stage0.py`, `test_think_budget_summary.py`,
`test_implied_summary.py`.

## Moved from (old path -> new path)

| Old (under `jared/`) | New (under `pipeline/s2_steering/`) |
|---|---|
| `experiments/exp10_patch_reasoning.py`, `exp16_phrase_patch.py` | `experiments/patch/` |
| `experiments/exp8_steering.py` | `experiments/ablate_steer/` |
| `experiments/exp13_suffix_steer.py`, `exp14_phrase_steer.py`, `exp15_scan.py` | `experiments/manifold_steer/` |
| `experiments/exp17_think_budget.py` | `experiments/think_budget/` |
| `experiments/exp9_units.py`, `exp18_implied.py` | `experiments/units_implied/` |
| `analysis/{exp10_analysis,patch_summary,scan_summary,think_budget_stage0,think_budget_summary,implied_summary,steering_analysis,crossing_split}.py` | `analysis/` |
| `queue_{patch,phrase_steer,pristine_steer,scan,think_budget,implied}.sh` | `queues/` |
| `07`, `08`, `09`, `15_spec_crossing`, `15_spec_same_subspace_sweep`, `18_results_overnight`, `21`, `22`, `25`, `26_results_overnight_choice_scan`, `26_spec_overnight_choice_scan`, `27`, `28`, `29`, `30` (`.md`) | `docs/` |

`experiments/common.py` moved to stage 1 (`pipeline.s1_data_vectors.experiments.common`). Tests
stayed in `jared/tests/` (spec rule 3) with their imports rewritten.

## Docs

- `07_spec_steering.md`: E8 spec: patching, ablation, steering, E9 units, E10 phrase patch.
- `08_results_steering.md`: E8 results on Qwen3-8B, full bank.
- `09_causal_tests_explained.md`: the causal tests explained: baselines, horizon vs decision.
- `15_spec_crossing.md`: E12 spec: is curve steering horizon-specific (delay crossing).
- `15_spec_same_subspace_sweep.md`: E12 draft: ablate and steer one subspace by layer, thinking on.
- `18_results_overnight.md`: overnight confound sweep on saved runs (no model run; kept here by the spec).
- `21_spec_phrase_steering.md`: E14 spec: steer at the horizon phrase.
- `22_results_phrase_steering.md`: E14 results.
- `25_spec_arithmetic_mediation.md`: draft (not approved): does the arithmetic decide the both-regime choice.
- `26_spec_overnight_choice_scan.md` / `26_results_overnight_choice_scan.md`: multichoice banks and the E15 scan, spec and results.
- `27_spec_phrase_number_patch.md`: E16 spec: what the horizon phrase carries, by donor patch.
- `28_spec_choice4_side.md`: E17 spec: are choice4 errors a think-budget artefact.
- `29_spec_implied_horizon_side.md`: E18 spec: implied horizons.
- `30_results_overnight_patch_budget_implied.md`: results of E16, E17 and E18 (overnight 2026-11-07).

## Status and known gaps

- Nothing was rerun in the reorganisation; the code is only moved and its imports and paths
  rewritten. The model-backed runners were compile-checked, not executed.
- Cross-stage imports: stage 2 reads stage 1 (`exp14_phrase_states`, `exp0_choice`, `common`,
  `horizon_shape`, `pristine_manifold`) and stage 3 (`exp14_analysis`, `written_unit`). A stage-3
  rename breaks `patch_summary`, `implied_summary`, `exp13_suffix_steer`.
- Docs `25` and `27`-`29` are drafts, not approved at the time they were written; check
  `01_plan.md` for the current status before running their queues.
- Old docs and the queue headers still name stage-1 and stage-3 scripts by bare filename in prose.
- Box queues assume the checkout root `jared/` as the working directory (for `python -m`).
