# Pipeline reorganisation and multi-model review (2026-11-09)

Branch `pipeline` (worktree `.claude/worktrees/pipeline`), from snapshot `3fc94f9` on `jared`.
Start with `pipeline/README.md` for the layout; this file records what was done and how it was judged.

## What was done

1. Snapshot commit of everything (except `.claude/settings.local.json`).
2. Reorganisation into three stages from one shared spec (`SPEC.md`): `s1_data_vectors`,
   `s2_steering` (by modality), `s3_analysis` (by label / by datatype), each with a README.
3. A review by three models (haiku, sonnet, opus) on one rubric (below),
   a round of fixes, then the same three models re-scored without being told what changed.

## Rubric

Clarity, brevity, evidence quality, collaborability; 1-10, each score backed by `path:line` or a figure name.

## Scores (haiku / sonnet / opus)

| Criterion | Before | After | Mean change |
|---|---|---|---|
| Clarity | 6 / 6 / 6 | 7 / 7 / 7 | +1.0 |
| Brevity | 5 / 4 / 4 | 5 / 4 / 5 | +0.3 |
| Evidence | 5 / 6 / 6 | 6 / 7 / 7 | +1.0 |
| Collaborability | 6 / 6 / 6 | 6 / 5 / 6 | -0.3 |

Caveats: three models of one family; reviewers read different amounts (several read little code and
none regenerated a figure or checked numbers against raw JSON); one point is within reviewer noise.
Brevity and collaborability did not improve: the fixes added files (new modules, tests, indexes).

## Fixes applied

- Figures `choice4_*`, `implied_*`: Wilson 95% intervals, titles and n built from the JSON,
  "C0 failed, provisional" in titles, families defined; the script is testable and takes the results root as an argument.
- Doc 30: 7-vs-13 mediation denominator explained, "only family C tracks" replaced by what the
  McNemar tests support, C3 stated as a fail, subset comparison (105 vs 68) disclosed, stale lines removed,
  unreconciled spend flagged. `01_plan.md` brought in line.
- Code: run-folder naming and loaders shared in `spar_horizon/e0_runs.py`; duration parsing in
  `spar_horizon/durations.py`; stage 1 no longer imports stage 2, stage 2 no longer imports stage 3.
- Docs: `01_plan.md` refreshed (E16-E18), one `$RESULTS` data root, duplicate indexes cut,
  `docs/INDEX.md` for stages 1 and 2.
- Full fast suite: 567 passed.

## Still open (all three reviewers or opus; ranked)

1. Older figures (`pres_1`, `pres_2`, `causal_2`, `causal_4`, `exp10_reasoning`) lack n and defined error bars;
   "Decodable is not causal" and "same plane" titles say more than one patched position shows.
2. `implied_3_readout` Spearman bars have no interval; `choice4_3` shows paired data as two independent bars (add McNemar).
3. Three stage-1 scripts still import stage 3 (`pristine_confounds`, `direction_stability`, `exp14_geometry`);
   stage 3 imports stages 1 and 2 (that direction is by design).
4. Fill-bank one-off scripts live outside git (`results/horizon/analysis/fill_oneoffs/`, per doc 19).
5. Three copies of a Wilson helper; duplicate doc numbers (two 15_*, two 26_*, 18_* in both stages).
6. No seeds plumbed through the model-backed runners (not touched: untestable from the session).
7. Doc 30: "the scan's null was the direction, not the placement" is an inference, not a test; the
   E18 stop rule (C0 fail -> stop and fix) was not followed, though results are labelled provisional;
   spend $6.33 at $0.421/h (about 15 h) does not match the ~10 h window.
8. Model-backed runners were only compile-checked after the move; run the E0 smoke via `local_checks.py`.
