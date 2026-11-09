# Stage 2 docs: index

One line per doc. Label: spec (criteria fixed before a run), result, explainer, draft (not approved),
on hold. Method specs of finished work stay as the method record. Findings are in `01_plan.md` §4.

| Doc | Label | What it is |
|---|---|---|
| `07_spec_steering.md` | spec (done) | E8 patching, ablation, steering; also E9 units and E10 phrase patch |
| `08_results_steering.md` | result | E8-E10 on Qwen3-8B, full bank |
| `09_causal_tests_explained.md` | explainer | the E8 tests: baselines, patched prompts, horizon vs decision |
| `15_spec_crossing.md` | spec (on hold) | E12 part 1: is curve steering horizon-specific (delay crossing) |
| `15_spec_same_subspace_sweep.md` | draft | E12 part 2: ablate and steer one subspace by layer |
| `18_results_overnight.md` | result | confound sweep on saved runs (no model run) |
| `21_spec_phrase_steering.md` | spec (run) | E14: steer at the horizon phrase |
| `22_results_phrase_steering.md` | result | E14: H1-H3 fail |
| `25_spec_arithmetic_mediation.md` | draft | does the arithmetic decide the both-regime choice |
| `26_spec_overnight_choice_scan.md` | spec (run) | choice4/8 banks and the E15 scan |
| `26_results_overnight_choice_scan.md` | result | choice banks complete; scan 34 of 40 cells, screen |
| `27_spec_phrase_number_patch.md` | spec (run as E16) | what the phrase carries, by donor patch |
| `28_spec_choice4_side.md` | spec (run as E17) | are forced closes a think-budget artefact |
| `29_spec_implied_horizon_side.md` | spec (run as E18) | implied horizons |
| `30_results_overnight_patch_budget_implied.md` | result | E16, E17, E18 (overnight 2026-11-07), with an addendum on the manifold figures |

Number clashes (names kept so old references still resolve):

- `15_spec_crossing.md` and `15_spec_same_subspace_sweep.md` are two different specs; both are E12.
- `18_results_overnight.md` here and `18_spec_multichoice.md` in stage 1 share a number.
- `26_*` is a spec and its results.

Specs `04`-`06`, `10`-`11`, `14`, `16`-`20` are in `../../s1_data_vectors/docs/INDEX.md`. Root docs
(`12`, `13`, `23`, `24`, `31`) are listed in `00_README.md`.
