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

The stages are not perfectly layered. These imports go against the data flow:

- `s1 steering_prepare` imports `s2 exp8_steering`; stage 2 runners import `s1` builders back
  (`common`, `exp0_choice`, `exp14_phrase_states`, `horizon_shape`, `pristine_manifold`).
- `s1` analysis scripts import stage 3 helpers (`unit_overlap`, `written_unit`, `confound_sweep`,
  `exp14_analysis`, `regime_comparison`).
- `s2 patch_summary`, `implied_summary`, `exp13_suffix_steer` import stage 3 modules
  (`exp14_analysis`, `written_unit`); a stage 3 rename breaks them.
- `s1 queue.sh`, `queue_14b*.sh` mostly launch stage 2 work (E10, E8 decision split).
- `s2 docs/18_results_overnight.md` is a stage 3 style confound sweep.

Fixing these means moving a few functions into `spar_horizon/`; not done here.

## Open points

- "Datatype" is read as bank plus rendered unit word. Every current bank writes the horizon in
  digits, so a digits/written split exists only in the model's traces (`written_unit.py`).
- `split.py` reads `answers.json` only; older stage 3 scripts keep their own loaders.
- Older docs and queue headers still cite bare script names (`exp8_steering.py`) without the stage path.
- Nothing was rerun in the reorganisation: paths and imports changed, model-backed runners were
  compile-checked only.
