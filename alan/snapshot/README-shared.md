# Alan — Planning Temporal Manifolds snapshot

Snapshot of my working repo (`CodeReclaimers/SPAR-2026`, private) at the commit named in
`SNAPSHOT.yaml`. **Second snapshot (2026-10-01).** Start with `notes/update-2026-10-01.md`: a plain-language
account of everything added since the first snapshot (forward curve model, Guttman check, LEACE, the
polynomial ladder, stakes, the relevance-decoder generalization and late-readout results, and the
Qwen3.5-27B run). The 2026-09-23 note covers the first snapshot. The Qwen3-14B runs were regenerated
from committed code on 2026-09-25 (`runs/reproduce_all.log`); the Qwen3.5-27B runs were produced on
2026-09-29/30 on a Vast.ai A100 by `scripts/vast_full_chain.sh` (`runs/chain.log`, checksums in `snapshot/`).

## What is here

- `ptm/`, `scripts/`, `tests/` — capture and analysis code (Python; `uv sync --extra dev`; 59 tests).
- `CLAUDE.md` — project conventions, capture facts, and commands. `progress-*.md` — dated logs with
  every number and the command that produced it. `notes/` — weekly-update notes (`update-2026-10-01.md` is the current one).
- `data/prompts/` — all prompt datasets (regenerable: `gen_prompts.py`, `gen_matrix.py`).
- `runs/<run>/index.parquet` — one row per prompt: parameters, generated text, parsed choice, label
  logits; `meta.json` — model, layer convention, positions.
- **Activations** are hosted separately to keep this repo small:
  `https://huggingface.co/datasets/CodeReclaimers/spar-planning-temporal-manifolds-activations`
  (public). Per run it holds `index.parquet`, `meta.json`, `subset.json` and bf16
  `acts_subset_L*_c*.safetensors` (rows in `index.parquet` order). Qwen3-14B: the canonical run at
  layers 22 and 37 for T1, T3, T8, R0, R3; the matrix run at layers 22 and 26 for T1, T3, R0; the extended
  run at layers 22 and 37 for T3, R0 (464 MB). Qwen3.5-27B: all eight runs at 13 layers (16, 20, 22, 23,
  24, 28, 29, 31, 35, 42, 46, 53, 59 — the sweep peaks plus the standard fractional depths) × all
  positions (37 GB; uploading at snapshot time, run by run). Load with
  `ptm.subset.load_subset(run_dir, layer, position)` → float32 `[n, d]`, or read the safetensors
  directly. Stored layer `l` is the residual stream after decoder layer `l−1`; positions are named in
  `CLAUDE.md` (T0–T8 = the 9-token transition window ending in the empty think block, identical for Qwen3
  and Qwen3.5; R0–R7 = the first generated tokens, R3 = the choice label; the `_long` runs add M0 = first
  reasoning token, E0 = last token, MEAN = mean over the reasoning). Full activations are not hosted:
  `scripts/reproduce_all.sh` (14B, local) and `scripts/vast_full_chain.sh` (any model, rented GPU)
  regenerate every dataset, capture, and analysis; each run's `SHA256SUMS.shards` lets a recapture be
  checked for byte identity.
- `figures/<run>/` — sweep heatmaps, cell scatters, centroid-path figures, interactive 3-D views
  (`view3d_*.html`, open in a browser), and result tables (`sweep.csv`, `transfer/*.csv`, …).

## Runs

Qwen/Qwen3-14B (bf16, no-thinking template, 2 × RTX 4080) and Qwen/Qwen3.5-27B (bf16, same template,
1 × A100 80 GB; run names `qwen3.5-27b_*`). Both models have every run below.

| run | prompts | purpose |
|---|---|---|
| `*_investment_n2000_s0` | 2,000 | replication of the preprint's intertemporal choice, 17 horizons + null |
| `*_matrix_s0` | 3,240 | 30 configs × 2 domains × 12 horizons × 3 renderings + relevance/factual/scaling controls |
| `*_extended_n2000_s0` | 2,000 | 25-horizon grid, 1 minute to 10,000 years |
| `*_phrasings_n2000_s0` | 2,000 | four phrasings of the constraint line |
| `*_units_paired_n2000_s1` | 2,000 | same horizon rendered in different units, matched option parameters |
| `*_families_s0` | 960 | four held-out distractor families (option age, future event, frequency, elapsed planning) |
| `*_matrix_s0_long`, `*_families_s0_long` | 3,240 / 960 | the same prompts with 160 generated tokens and anchored reasoning positions (M0, E0, MEAN) |

Also `figures/stakes_pilot*/` (stakes pilot on existing captures) and, for the 27B,
`figures/qwen3.5-27b_matrix_s0/peak/` (cell analyses at the sweep's peak layers, L23 T3 / L20 R0 / L31 T6).

## Headline results

For everything since 2026-09-25 — curve model, Guttman check (the "vertex near one year" is withdrawn), LEACE,
polynomial ladder, stakes, relevance-decoder generalization, late readout, and the 27B comparison — read
`notes/update-2026-10-01.md`; the numbers and commands are in `progress-202609{27,28,29,30}.md`.
The list below is the first snapshot's (Qwen3-14B) summary and still holds, with one correction: the
quadratic "vertex at 0.6–0.65 years" is the midpoint of the horizon grid, not a landmark (see the note).

- **Replication.** PC1 of the residual stream rank-correlates with log horizon at |ρ| = 0.96 at the
  `assistant` token, layers 36–37 (reward-ratio control |ρ| ≤ 0.01). Behavior: short option chosen
  92% at 1 day, 43% at 10 years, 25% at 100 years, 10% with no stated horizon.
- **Geometry.** The 17 horizon-bin centroids lie 88% in a plane (top-2 centroid PCs), with 5
  components for 95% of centroid variance; in that plane PC1 is linear in log horizon (R² 0.97) and
  PC2 quadratic (R² 0.87–0.89) with vertex at 0.6–0.65 years. Split-half reliability 0.99.
  Interactive 3-D views: `figures/*/view3d_*.html`.
- **Extended grid (1 min – 10,000 y).** Preference collapses toward chance below the shortest option
  delay (62% at 1 minute, 51% at 10 minutes), plateaus at ~27–29% short from 100 to 1,000 years,
  then drops; the extremes leave the standard-range plane (centroid PC3 0.03 → 0.11) and the top end
  saturates beyond ~1,000 years.
- **Phrasing / units.** Within each of four constraint phrasings PC1 ordinality is 0.94–0.96, but
  pooled PCA is dominated by phrasing clusters; a ridge trained on one phrasing transfers to the
  others at R² 0.82–0.89. Rendering the same horizon in different units moves samples < 1 grid step
  along the path, yet "decades" renderings get 11 points more long-option choices than "years" at
  matched inputs, and the along-path shift does not account for it.
- **Transfer (matched matrix, frozen ridge).** Training on structured prompts only: 0.98 within a
  factor of 2 on new scenarios, but 0.34–0.39 on prose renderings. Training on all three renderings:
  0.99 / 0.97 / 0.91 within 2× on new scenarios in structured / plain / varied prose, 0.93 on withheld
  horizons (0.79 on new scenarios), climate from investment-only training 0.57. A regex on the prompt
  text is perfect on every explicit suite.
- **Relevance controls (the central open problem).** The activation decoder's estimate is pulled by
  durations that are irrelevant to the decision (residual–distractor correlation +0.58 "established
  20 years ago", +0.74 "you have 2 hours to prepare"), while the model's own choices ignore them
  (standardized coefficient on the distractor −0.03 / −0.11 vs −1.97 / −1.72 on the real horizon).
  No cell among 70 (7 layers × 10 positions) is free of the pull (min +0.14 / +0.30). Scaling horizon
  and delays together moves the estimate by 0.8 decades per decade: the readout is mostly absolute
  duration, not horizon relative to the options.

Details, commands, and intermediate numbers: `progress-2026091{9}.md`, `progress-202609{22,23,24,25,27,28,29,30}.md`.
Note: the 2026-09-19 log's behavior table came from an earlier capture with 95 unparsed responses;
the tables in `figures/qwen3-14b_investment_n2000_s0/behavior_by_horizon.csv` supersede it.

## Reproducing

```
uv sync --extra dev
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q tests
nohup scripts/reproduce_all.sh &          # ~1 h 45 on 2 × RTX 4080 SUPER (16 GB); needs Qwen/Qwen3-14B cached
# any other model on a rented GPU (inside tmux):  MODEL=Qwen/Qwen3.5-27B TAG=qwen3.5-27b scripts/vast_full_chain.sh
# then locally: scripts/vast_fetch.sh HOST PORT ; scripts/peak_cells.sh TAG
```
Capture facts that matter for comparison: activations are taken by forward hooks in an unpadded
per-sample pass (bit-exact against transformers' hidden states); stored layer `l` is the residual
stream after decoder layer `l−1`; label logits are float32; Qwen3-14B needs the explicit 2-GPU split. Layer specs in every script accept
fractional depth (`0.55L`), resolved against the run's layer count (`ptm/depth.py`). For Qwen3.5 the
gated-DeltaNet kernels (`flash-linear-attention`) and their reference fallback are each deterministic but not
bit-identical; each run's `meta.json` records which was used.
