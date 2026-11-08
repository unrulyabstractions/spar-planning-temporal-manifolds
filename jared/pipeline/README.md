# Pipeline

Three stages, each with its own README. Run everything from `jared/` as
`python -m pipeline.<stage>.<folder>.<script>`. Model-backed runs go to the box or to
`local_checks.py`; see `../CLAUDE.md` (SPAR root) for the rules.

```
 Stage 1  s1_data_vectors      prompts -> generation + activations -> probes -> steering inputs
    |   out/exp0_*/ (activations, answers.json)      steering/inputs_*.{json,npz}
    v
 Stage 2  s2_steering          donor patching, ablation, steering, think-budget, implied horizons
    |   steering/exp*_*/ , out/exp*_*/
    v
 Stage 3  s3_analysis          results split by label (regime, unit, horizon, ...) and datatype (bank, ...)
```

| Stage | Question | README |
|---|---|---|
| 1. Initial data and steering vectors | How do we get prompts, activations and a steering vector? | `s1_data_vectors/README.md` |
| 2. Testing and steering | Given an input and a vector, does behaviour move? By modality: `patch/`, `ablate_steer/`, `manifold_steer/`, `think_budget/`, `units_implied/` | `s2_steering/README.md` |
| 3. Analysis | What happened, by label and by datatype? Shared loader `s3_analysis/split.py` | `s3_analysis/README.md` |

Shared library, not part of a stage: `../spar_horizon/`. Data and results stay in
`../out/`, `../steering/`, `../figures/`. Tests are one suite in `../tests/`.

## Known cross-stage dependencies

The stages are not perfectly layered. These imports go against the data flow:

- `s1 steering_prepare` imports `s2 exp8_steering`; stage 2 runners import `s1` builders back.
- `s2 patch_summary`, `implied_summary`, `exp13_suffix_steer` import stage 3 modules
  (`exp14_analysis`, `written_unit`).
- `s1 queue.sh`, `queue_14b*.sh` mostly launch stage 2 work (E10, E8 decision split).
- `s2 docs/18_results_overnight.md` is a stage 3 style confound sweep.

Fixing these means moving a few functions into `spar_horizon/`; not done here.

## Open points

- "Datatype" is read as bank plus rendered unit word. Every current bank writes the horizon in
  digits, so a digits/written split exists only in the model's traces (`written_unit.py`).
- `split.py` reads `answers.json` only; older stage 3 scripts keep their own loaders.
- Older docs still cite bare script names (`exp8_steering.py`) without the stage path.
