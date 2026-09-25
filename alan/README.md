# Alan — Planning Temporal Manifolds snapshot

Snapshot of my working repo (`CodeReclaimers/SPAR-2026`, private) at the commit named in
`SNAPSHOT.yaml`. Everything here was regenerated from that commit in one run of
`scripts/reproduce_all.sh` on 2026-09-25 (log: `runs/reproduce_all.log`); the datasets, run indices,
figures, and tables are therefore consistent with the code by construction.

## What is here

- `ptm/`, `scripts/`, `tests/` — capture and analysis code (Python; `uv sync --extra dev`; 33 tests).
- `CLAUDE.md` — project conventions, capture facts, and commands. `progress-*.md` — dated logs with
  every number and the command that produced it. `notes/` — weekly-update drafts.
- `data/prompts/` — all prompt datasets (regenerable: `gen_prompts.py`, `gen_matrix.py`).
- `runs/<run>/index.parquet` — one row per prompt: parameters, generated text, parsed choice, label
  logits; `meta.json` — model, layer convention, positions.
- **Activations** are hosted separately to keep this repo small:
  `https://huggingface.co/datasets/CodeReclaimers/spar-planning-temporal-manifolds-activations`
  (public). It holds, per run, `index.parquet`, `meta.json`, `subset.json`,
  and bf16 `acts_subset_L*_c*.safetensors` (rows in `index.parquet` order): the canonical run at layers
  22 and 37 for positions T1, T3, T8, R0, R3; the matrix run at layers 22 and 26 for T1, T3, R0; the
  extended run at layers 22 and 37 for T3, R0 (464 MB total). Load with
  `ptm.subset.load_subset(run_dir, layer, position)` → float32 `[n, 5120]`, or read the safetensors
  directly. Stored layer `l` is the residual stream after decoder layer `l−1`; positions are named in
  `CLAUDE.md` (T0–T8 = the 9-token Qwen3 transition window ending in the empty think block; R0–R3 = the
  first generated tokens, R3 = the choice label). Full activations (41 layers × 17 positions, ~14 GB per
  run) are not hosted: `scripts/reproduce_all.sh` regenerates every dataset, capture, and analysis.
- `figures/<run>/` — sweep heatmaps, cell scatters, centroid-path figures, interactive 3-D views
  (`view3d_*.html`, open in a browser), and result tables (`sweep.csv`, `transfer/*.csv`, …).

## Runs (all Qwen/Qwen3-14B, bf16, no-thinking template)

| run | prompts | purpose |
|---|---|---|
| `qwen3-14b_investment_n2000_s0` | 2,000 | replication of the preprint's intertemporal choice, 17 horizons + null |
| `qwen3-14b_matrix_s0` | 3,240 | 30 configs × 2 domains × 12 horizons × 3 renderings + relevance/factual/scaling controls |
| `qwen3-14b_extended_n2000_s0` | 2,000 | 25-horizon grid, 1 minute to 10,000 years |
| `qwen3-14b_phrasings_n2000_s0` | 2,000 | four phrasings of the constraint line |
| `qwen3-14b_units_paired_n2000_s1` | 2,000 | same horizon rendered in different units, matched option parameters |

## Headline results

All numbers below are read from the fresh analysis logs in `runs/*.analysis.log` (Qwen3-14B).

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

Details, commands, and intermediate numbers: `progress-2026091{9}.md`, `progress-202609{22,23,24,25}.md`.
Note: the 2026-09-19 log's behavior table came from an earlier capture with 95 unparsed responses;
the tables in `figures/qwen3-14b_investment_n2000_s0/behavior_by_horizon.csv` supersede it.

## Reproducing

```
uv sync --extra dev
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q tests
nohup scripts/reproduce_all.sh &          # ~1 h 45 on 2 × RTX 4080 SUPER (16 GB); needs Qwen/Qwen3-14B cached
```
Capture facts that matter for comparison: activations are taken by forward hooks in an unpadded
per-sample pass (bit-exact against transformers' hidden states); stored layer `l` is the residual
stream after decoder layer `l−1`; label logits are float32; Qwen3-14B needs the explicit 2-GPU split.
