# Stage 1 docs: index

One line per doc. Label: spec (criteria fixed before a run), result, draft (not approved).
Findings are in `01_plan.md` §4.

| Doc | Label | What it is |
|---|---|---|
| `04_spec_probes.md` | spec (done) | E7: linear vs MLP probes on the top-k PCs |
| `06_results_probes_full.md` | result | E7 on the full bank, Qwen3-8B |
| `10_spec_scale_14b.md` | spec (done) | E11: the 8B map on Qwen3-14B |
| `11_results_14b.md` | result | E11 results |
| `14_spec_unit_overlap.md` | spec (done) | unit-overlap bank, horizon vs unit word |
| `16_spec_pristine_manifold.md` | spec (done) | E13: a manifold free of unit word, regime, forced closes, order |
| `17_results_pristine.md` | result | E13 prototype on the 8B units bank |
| `18_spec_multichoice.md` | spec (run in stage 2 doc `26`) | 4- and 8-choice banks; header still says draft |
| `19_results_fill.md` | result | fill bank pooled with the old 8B runs |
| `20_results_overnight_fill.md` | result | second overnight, post hoc on saved activations |

`18_spec_multichoice.md` shares its number with `../../s2_steering/docs/18_results_overnight.md`.
Stage 2 docs are in `../../s2_steering/docs/INDEX.md`. Root docs (`12`, `13`, `23`, `24`, `31`) are
listed in `00_README.md`.
