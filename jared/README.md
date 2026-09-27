# Jared — Temporal horizon robustness on Qwen3-8B

Snapshot of my working directory as of 2026-09-26. Provenance in `SNAPSHOT.yaml`.
The long-form docs are `00_README.md` (index, lineage, environment, how to run),
`01_plan.md` (research question, tests R1–R5), `02_execution_plan.md` (build
log, decisions), `03_baseline_qwen3.5-0.8b.md` (the starter's 0.8B baseline).

## Question

Is the linearly readable time-horizon direction in the residual stream a stable,
prompt-independent quantity, and does the model's choice depend on it? Prompts
are the starter's intertemporal choice: small near payout vs. large far payout
"for this time horizon: X", X log-swept from 30 seconds to 500 years (96 horizon
prompts + 20 no-horizon controls). Lineage: `temporal-awareness` (Rios-Sialer et
al. 2026, Qwen3-4B). Model here: `Qwen/Qwen3-8B`, bf16, one RTX 4090, thinking
on and off via the native toggle.

## What is here

- `spar_horizon/` — package: prompts and template/unit/perturbation registries,
  model I/O (chat template, thinking toggle, batched greedy generation with a
  force-closed think block, assistant prefill), activation extraction at the
  turn-suffix and answer positions, PCA sweep + ridge horizon probe, three
  behavior coherence tests, attribution patching, plots, run manifests.
- `experiments/exp0..exp5_*.py` + `run_all.sh` — one CLI per experiment (see
  table). Every run writes `manifest.json`; a run without one is a failure.
- `tests/` — 48 pytest tests, tokenizer-only, no GPU. Green on 2026-09-26.
- `analysis/` — post-hoc on synced results: cross-mode (think on/off) probe
  transfer, feature dimensionality, PCA-3 export, an interactive 3D explorer
  (`pca3_explorer.ipynb`, ~10 MB with outputs, and its HTML template), and
  `snapshot.py`, which freezes a run's derived data into `snapshots/`.
- `snapshots/v1_noprefill/` — the only full Qwen3-8B run so far: manifests,
  PCA coordinates (`pca3.json.gz`), sweeps, transfer CSVs, figures. ~16 MB.
  **Activations are not in the repo** (1.7 GB, local only, see `SNAPSHOT.yaml`).
- `starter/` — untouched copies of the original starter script, its 116
  prompts, and its figure.
- `horizon_geometry.py` / `.ipynb` — notebook-shaped wrapper around E0.

| | Experiment | Asks | Status |
|---|---|---|---|
| E0 | baseline | peak layer, probe layer, does the model read the content | run, both modes |
| E1 | nuisance | is PC1 horizon rather than reward/delay; probe survives swap/relabel | run, think-off |
| E2 | paraphrase | probe transfer across 6 constraint wordings | run, think-off |
| E3 | units | does "1 year" land with "12 months" and "365 days" | run, think-off |
| E4 | perturb | distractor, horizon-first, implicit horizon: probe transfer + behavior | run, think-off |
| E5 | attribution | which layer/position/component carries horizon to the answer | run, think-off |

## Headline results (v1_noprefill, Qwen3-8B, 116 prompts, 2026-09-24)

Geometry and probe numbers are valid. **Thinking-off behavior numbers are not**
(see next section).

- **Geometry replicates.** PC1 at the turn-suffix newline rank-correlates with
  log horizon at |ρ| = 0.95, peak layer 16, in both thinking modes. A ridge
  probe on the residual stream reads log horizon with R² = 0.9995 at layer 11.
  The 3D PCA panels are in `snapshots/v1_noprefill/exp0_*/geometry_3d*.png`.
- **Cross-mode transfer.** A probe trained on thinking-off activations reads
  thinking-on activations at ρ ≈ 0.98–0.998 across the five shared suffix
  tokens (`cross_mode_transfer.json`). The direction is the same in both modes.
- **R4 nuisance.** Probe transfer to swapped-option prompts 0.986, to 1/2
  relabeled prompts 0.978.
- **R1 paraphrase.** Best off-diagonal transfer across the 6 constraint
  templates 0.99 at layer 10 (`exp2_*/transfer_L*.csv`).
- **R3 units.** Within-duration spread / between-duration spread along the
  probe axis = 0.13 (pass threshold was < 0.20). The axis tracks duration, not
  the unit word.
- **E4 perturbations.** Probe transfer |ρ|: clean 0.998, distractor 0.997,
  horizon-first 0.90, implicit horizon 0.88. Order/label stability columns in
  that table are from the invalid think-off answers; ignore them.
- **E5 attribution (think-off).** 48 clean/corrupt pairs, 33 flip the a/b logit
  difference. Attribution concentrates on the horizon unit token ("months",
  layers 13–15) and the final position (layers 23–35). Heatmaps in
  `exp5_*/{noising,denoising}_*.png`.
- **Thinking-on behavior (E0 only).** Median think length 2,733 tokens; 46 of
  116 think blocks force-closed at 3,072 (short horizons loop). Temporal
  reasoning 27/27, order stability 38/76, label stability 61/80.

## Known problems and what is next

1. **Think-off answers were prose, not labels.** Without a prefill Qwen3-8B
   opens with "To determine the best investment..." so the 6-token answer
   window held no label and every think-off behavior number is meaningless.
   Fixed in code on 2026-09-24 by prefilling the assistant turn with
   `I choose:` and skipping formatting tokens before the label. **The whole
   think-off run must be rerun** and snapshotted as `v2_prefill`; the
   suffix-position geometry from v1 stands.
2. `git` is `null` in every v1 manifest because the tree was not yet under
   version control. Later runs will record the commit.
3. Not started: R2 domain transfer (new scenario text), R5 steering (uses the
   E5 map to choose where), and `04_results_qwen3-8b.md` with the R1–R5
   pass/fail rows filled in.

## Reproducing

```
~/miniconda3/envs/mp/bin/python -m pytest -q tests                      # 48 tests, no GPU
MODEL=Qwen/Qwen3.5-0.8B LIMIT=6 MODES=off bash experiments/run_all.sh   # CPU smoke, ~8 min
MAX_THINK=3072 bash experiments/run_all.sh                              # on the 4090: all six, both modes, ~2 GPU-h
python analysis/snapshot.py results/horizon v2_prefill --note "..."     # freeze derived data
```

Outputs go to `out/<exp>_<model>_think-<on|off>/` locally, or to
`$WORKSPACE/results/horizon/` on the box. `00_README.md` has the vast.ai
commands, the model-choice table, and the smoke-test log.
