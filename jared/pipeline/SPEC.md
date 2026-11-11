# Pipeline reorganisation: shared spec

Goal: turn `jared/experiments/`, `jared/analysis/`, the `queue*.sh` files and the
numbered docs into three stages with one README each. Nothing is rerun; this is a
move-and-clean job. Three agents build one stage each, in parallel, in the
`pipeline` worktree. This file is the contract between them.

## Layout (all under `jared/pipeline/`)

```
pipeline/
  README.md                  overview + data-flow diagram (owned by the lead, written last)
  SPEC.md                    this file
  __init__.py
  s1_data_vectors/           Stage 1: initial data and steering-vector extraction
  s2_steering/               Stage 2: testing and steering, by modality
  s3_analysis/               Stage 3: analysis split by label and by datatype
```

Each stage directory has the same shape:

```
s?_name/
  README.md        (required sections below)
  __init__.py
  experiments/     runnable CLIs (__init__.py inside)
  analysis/        post-processing scripts for this stage (__init__.py inside)
  queues/          the box queue_*.sh files that belong to this stage
  docs/            the numbered spec/results .md files that belong to this stage
```

Stage 3 has no `experiments/`; it has `by_label/` and `by_datatype/` (below).

Stays where it is (shared library, do NOT edit): `spar_horizon/`, `tests/conftest.py`
fixtures, `local_checks.py`, `out/`, `steering/` (data), `figures/`, `snapshots/`,
`history/`, `requirements.txt`. Output/data paths in code do not change.

## Ownership (disjoint, so agents never touch the same file)

### Stage 1, `s1_data_vectors` (agent A)
Question answered: how do we get prompts, model activations, and a steering vector?
- experiments: `exp0_baseline`, `exp0_choice`, `exp14_phrase_states`, `warm_cache`,
  `check_prefix_cache`, `coherence`, `common` (shared CLI helper; stage 2 imports it from here)
- analysis (probes, geometry, vector/input builders): `probe_dimensionality`,
  `probe_natural_within_unit`, `horizon_shape`, `direction_stability`, `pristine_manifold`,
  `pristine_confounds`, `steering_prepare`, `patch_prepare`, `scan_prepare`,
  `cross_model_alignment`, `stage_markers`, `exp14_geometry`, `caa_confound`, `off_curve_content`
- queues: `queue.sh`, `queue_14b.sh`, `queue_14b_rest.sh`, `queue_units.sh`, `queue_multichoice.sh`
- docs: `04`, `06`, `10`, `11`, `14`, `16`, `17`, `18_spec_multichoice`, `19`, `20`
- tests: those testing the moved modules above

### Stage 2, `s2_steering` (agent B)
Question answered: given an input and a vector, does the behaviour move? Organise by
modality, one subfolder per modality inside `experiments/`:
`patch/` (donor patching: exp10, exp16), `ablate_steer/` (exp8), `manifold_steer/`
(exp13, exp14_phrase_steer, exp15), `think_budget/` (exp17), `units_implied/` (exp9, exp18).
- queues: `queue_patch.sh`, `queue_phrase_steer.sh`, `queue_pristine_steer.sh`, `queue_scan.sh`,
  `queue_think_budget.sh`, `queue_implied.sh`
- docs: `07`, `09`, `15_*`, `21`, `22`, `25`, `26_*`, `27`, `28`, `29`, `30`
- analysis: per-experiment summaries that only make sense next to one runner:
  `exp10_analysis`, `patch_summary`, `scan_summary`, `think_budget_stage0`,
  `think_budget_summary`, `implied_summary`, `steering_analysis`, `crossing_split`
- tests: those testing moved modules, plus `exp*` runner tests

### Stage 3, `s3_analysis` (agent C)
Question answered: what happened, split by label and by datatype?
- Everything else in `analysis/` (not claimed above) moves here, sorted into:
  - `by_label/`: splits on record labels `regime` (none/near/both/null), `unit`, `horizon`,
    `delay`, `truth`, choice count (2/4/8), forced vs natural close.
  - `by_datatype/`: splits on the data the prompt/bank carries: bank (full, units, fill,
    pristine, choice4/8, implied) and number format (digits, written, unit word).
  - `figures/`: scripts that only draw (`evidence_figures`, `presentation_figures`,
    `choice4_implied_figures`, `patch_figures`).
- Build one small shared loader, `s3_analysis/split.py`: load a run's records into a table
  with the label columns above and `bank`, and `split_by(table, "regime")`-style helpers.
  New scripts in `by_label/` and `by_datatype/` use it; moved scripts keep their logic.
- docs: `00`, `01`, `05`, `12`, `13`, `23`, `24`, `31` stay in `jared/` root (index docs);
  `08`, `18_results_overnight`, `30` stay with stage 2. Stage 3 owns the doc index only.
- tests: those testing moved modules

Any file in `analysis/` or `experiments/` not listed above goes to the stage whose README
question it answers best; the owning agent of the *destination* decides and records it in
its README "Moved from" table. Unclear cases: stage 3.

## Rules for every agent

1. Use `git mv` so history follows. Commit in the worktree on branch `pipeline`; only
   stage your own paths (`git add <your dir> <your tests>`), never `git add -A`.
2. Imports: modules are addressed as `pipeline.s1_data_vectors.analysis.foo`. Remove the
   `sys.path.insert(...)` hacks in moved scripts. Run scripts as
   `python -m pipeline.s2_steering.experiments.patch.exp10_patch_reasoning ...` from `jared/`.
   Update every import that pointed at a moved module, including in other stages' files only
   when the import line names your module: tell the lead rather than editing their files.
3. Tests live in `jared/tests/` as before. After moves, run (from the worktree `jared/`):
   `~/miniconda3/envs/mp/bin/python -m pytest -q tests --ignore=tests/test_generation.py`
   Never delete, weaken or skip a test; if one looks wrong, stop and report it.
4. Never run model-backed code (no generation, no `local_checks.py`, no GPU box verbs).
   Compile-check only: `python -m py_compile` and `--help` on CLIs that do not load a model.
5. Docs: update paths mentioned in the docs you moved. Do not rewrite findings.
6. Queues: fix the script paths inside `queue_*.sh` to the new `python -m` form.
7. Work only in the `pipeline` worktree. Do not edit `spar_horizon/`.
8. Interface hand-offs: stage 1 writes `out/exp0_*/` runs and `steering/inputs_*.{json,npz}`;
   stage 2 reads those and writes `steering/exp*_*/`; stage 3 reads both. README must list
   these inputs and outputs by path pattern.

## README template (each stage, same headings)

```
# Stage N: <name>
## Question
## Inputs  (paths/patterns, which stage produced them)
## Outputs (paths/patterns, who reads them)
## Contents (table: script | what it does | one-line example command)
## How to run (box vs local; link queue files)
## Moved from (old path -> new path)
## Docs (the numbered docs in docs/, one line each)
## Status and known gaps
```

## Done means

- `git status` clean in your paths, committed on `pipeline`.
- Full fast suite passes (or failures are listed with cause, not hidden).
- `grep -rn "sys.path.insert" pipeline/<your stage>` returns nothing.
- Report back: files moved, import edits, anything you left for another stage, test result.
