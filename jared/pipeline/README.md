# Pipeline

Three stages, each with its own README. Run everything from `jared/` as
`python -m pipeline.<stage>.<folder>.<script>`. Model-backed runs go to the box or to
`local_checks.py`; see `../CLAUDE.md` (SPAR root) for the rules.

```
 Stage 1  s1_data_vectors      prompts -> generation + activations -> probes -> steering inputs
    |   exp0_*/ (activations, answers.json)      steering/inputs_*.{json,npz}
    v
 Stage 2  s2_steering          donor patching, ablation, steering, think-budget, implied horizons
    |   exp*_*/
    v
 Stage 3  s3_analysis          results split by label (regime, unit, horizon, ...) and datatype (bank, ...)
```

| Stage | Question | README |
|---|---|---|
| 1. Initial data and steering vectors | How do we get prompts, activations and a steering vector? | `s1_data_vectors/README.md` |
| 2. Testing and steering | Given an input and a vector, does behaviour move? By modality: `patch/`, `ablate_steer/`, `manifold_steer/`, `think_budget/`, `units_implied/` | `s2_steering/README.md` |
| 3. Analysis | What happened, by label and by datatype? Shared loader `s3_analysis/split.py` | `s3_analysis/README.md` |

Shared library, not part of a stage: `../spar_horizon/`. Tests are one suite in `../tests/`.

## What lives where

Every README uses `$RESULTS` for the data root, passed as the first argument of analysis scripts
and as `--out` / `--dir` base of runners. Set it once: `export RESULTS=/home/Jared/SPAR/results/horizon`
(fish: `set -x RESULTS /home/Jared/SPAR/results/horizon`).

| What | Where | Notes |
|---|---|---|
| Run folders `exp<N>_<model>_*`, `gen_cache/`, `analysis/` (real data) | `$RESULTS` = `/home/Jared/SPAR/results/horizon` | filled by `just sync`; outside git; the same for every worktree |
| The same on the box | `/workspace/results/horizon` | runners write here with `--out /workspace/results/horizon` |
| Steering inputs `inputs_<model>_<name>.{json,npz}` | `steering/` under `jared/` (box: the pushed checkout) | written by `steering_prepare`, `patch_prepare`, `scan_prepare`; older ones are also in `$RESULTS/steering_inputs/` |
| Local smoke runs (0.8B plumbing only) | `out/` under `jared/` (default `--out` when `$WORKSPACE` is unset) | gitignored, not evidence |
| Figures and tables in the docs | `figures/` under `jared/`, and `<run>/figures/` inside `$RESULTS` | |

`out/` and `steering/` exist in the main checkout only; a worktree sees `$RESULTS` and passes
`steering/` paths from wherever it ran `steering_prepare`.

## Reviewing evidence

Figures show n and an interval (or the raw points) for every bar or curve. A claim in a results
doc names the test behind it (which interval, which null, which criterion), or says it is a screen
or post hoc and gates nothing.

## Cross-stage dependencies

Shared helpers that several stages used now live in `../spar_horizon/` (`e0_runs.py` run
folders and loaders, `durations.py` duration parsing, plus a few functions in `probes.py`,
`banks.py`, `steering.py`). Stage 1 no longer imports stage 2, and stage 2 no longer imports
stage 3. Three imports still run against the data flow, all stage 1 into stage 3:

- `pristine_confounds.py` and `direction_stability.py` use helpers from `by_label/confound_sweep`
  (eleven stage 3 scripts use the same ones).
- `exp14_geometry.py` imports `by_label/exp14_analysis`.

Fixing them means moving those three scripts into stage 3, or the `confound_sweep` helpers into
`spar_horizon/`. Also: `queue.sh` and `queue_14b*.sh` (stage 1) mostly launch stage 2 work, and
`s2 docs/18_results_overnight.md` is a stage 3 style confound sweep.

## Open points

- "Datatype" is read as bank plus rendered unit word. Every current bank writes the horizon in
  digits, so a digits/written split exists only in the model's traces (`written_unit.py`).
- `split.py` reads `answers.json` only. Run-folder naming is shared (`spar_horizon/e0_runs.py`); two activation/record loaders stay separate on purpose (see their comments).
- Older docs and queue headers still cite bare script names (`exp8_steering.py`) without the stage path.
- Nothing was rerun in the reorganisation: paths and imports changed, model-backed runners were
  compile-checked only.
