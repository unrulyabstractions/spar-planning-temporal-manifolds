# Task prompt: add mid-thinking geometry to the thinking-on experiments

Paste everything below this line into a new session, run from
`spar-planning-temporal-manifolds/jared/`.

---

## Goal

The thinking-on experiments currently keep hidden states only at the turn
suffix (before the model starts reasoning) and at the answer (after
`</think>`). Nothing is kept from *inside* the think block. Extend E0 with
thinking on so that PCA, the horizon probe, and the 3D explorer also cover
positions in the middle of the model's reasoning, and answer: does the
horizon axis persist, drift, or dissolve while the model reasons, and does
the geometry at the end of thinking predict the answer?

## Where things are

Working directory: `spar-planning-temporal-manifolds/jared/`.
Read `00_README.md` first, then `02_execution_plan.md` sections 6 and 7.

| File | Role |
|---|---|
| `spar_horizon/config.py` | `Settings` dataclass: `model`, `thinking`, `n_response=6`, `max_think_tokens=3072`, `batch_size=8`, `prefill="I choose:"`, `limit` |
| `spar_horizon/model_io.py` | `generate_batch()` returns a `Generation` per prompt: `ids` (full unpadded token sequence), `prompt_len`, `answer_start` (index of the label token), `think_len`, `forced`, `pre_len` (tokens between the think close and the label: `</think>`, newline, prefill, any leading markdown) |
| `spar_horizon/extract.py` | `extract()` builds the kept-position index `keep` per prompt and runs one forward pass with `output_hidden_states=True`. Returns `Extraction(activations, tokens, answers, kept, n_suffix, think_lens, forced)`; `activations[layer]` is `[n_kept_prompts, n_positions, d_model]` |
| `spar_horizon/geometry.py` | `sweep()` PC1 Spearman per (layer, position); `horizon_probe()` RidgeCV probe; `probe_sweep()`; `display_layer()` |
| `spar_horizon/plots.py` | `pca_panels()`, `pca3_stats()`, `pca3_panels()`, `heatmap()`; `_region(pos, n_suffix)` labels positions |
| `spar_horizon/runs.py` | `save_activations()` / `load_activations()` (npz with `L0..L36`, `tokens`, `n_suffix`, `kept`, `horizons`, `rho`, `r2`), `Manifest` |
| `experiments/exp0_baseline.py` | E0 CLI: extract, sweep, probe, 2D + 3D panels, `activations.npz`, `answers.json`, `manifest.json` |
| `analysis/export_pca3.py` | PC1-3 per (mode, layer, position) to `pca3.json` for the explorer |
| `analysis/feature_dimensionality.py` | fraction of horizon signal in top-k PCs per (mode, layer, position) |
| `analysis/pca3_explorer_nb.py` | source of `pca3_explorer.ipynb` (rebuild with the nbformat snippet in the README); reads snapshots |
| `analysis/snapshot.py` | freezes derived data under `snapshots/<name>/` |
| `tests/` | 48 pytest tests; `test_generation.py` and `test_model_io.py` use the cached Qwen3.5-0.8B on CPU |

Current position layout, thinking on (Qwen3-8B, prefill on):

```
[ suffix: <|im_end|> \n <|im_start|> assistant \n <think> \n ]   n_suffix = 7, ends at prompt_len
[ ... think block, 1.5k-3k tokens, NOT KEPT ... ]
[ pre: </think> \n\n I choose: (+ any ' **') ]                    pre_len tokens before answer_start
[ answer: label ) <|im_end|> ... ]                                 n_response tokens from answer_start
```

`Extraction.n_suffix` is overloaded: it is the index of the answer token in
the kept layout (suffix + pre), which downstream code uses to label regions.

## Facts that constrain the design

- Think length varies per prompt (median 2733 tokens on Qwen3-8B, forced
  closed at 3072 for 46 of 116 prompts). Absolute think positions do not
  align across prompts; positions must be defined relative to each prompt's
  own think span.
- Short horizons (seconds, minutes) make the model loop and hit the budget.
  The `forced` flag marks those; keep it in the output so forced prompts can
  be excluded or highlighted.
- The forward pass for extraction is per prompt over the full sequence, up
  to ~3.2k tokens; that already fits a 4090. Keeping more positions costs no
  extra compute, only npz size: each extra position adds
  116 × 37 × 4096 × 4 bytes ≈ 70 MB.
- With thinking off there is no think block. The new positions must be
  absent in that mode without breaking E1-E5, which read E0's
  `activations.npz` and `manifest.json` (`probe_layer`, `probe_pos`).
- The 0.8B model used for local tests never closes its think block; the
  forced-close path is the only thinking-on path testable locally
  (`--max-think-tokens 32`).
- Every extraction prints the answer-position token distribution and warns
  if it is not a label. Keep that.

## What to build

1. **Think-span positions.** Add a setting `think_positions` (default 9).
   For each prompt with thinking on, take positions at fixed fractions
   f = 0.0, 0.125, ..., 1.0 of its think span `[prompt_len, prompt_len + think_len)`,
   i.e. `prompt_len + round(f * (think_len - 1))`. f = 0 is the first
   reasoning token, f = 1 is the last before the close. Insert these
   between the suffix and pre regions in `keep`.
2. **Region bookkeeping.** Replace the overloaded `n_suffix` with explicit
   region boundaries stored in `Extraction` and the npz: `suffix`, `think`,
   `pre`, `answer`, each `(start, end)` in the kept layout. Update
   `_region()` in `plots.py` and the region labels in `export_pca3.py` and
   the notebook chips so think positions read "think 0.25" etc. Keep a
   backward-compatible `n_suffix` (= answer index) so E1-E5 still run.
3. **Token labels.** Think positions have a different token in every prompt;
   `tokens` will show the majority token with `*`. Give them the fraction
   label instead, e.g. `"think@0.25*"`, so tables and chips are readable.
4. **Analysis per think position.** E0's existing sweep and probe loop over
   positions, so the new columns come for free. Add to the manifest: probe
   R² and PC1 Spearman at each think fraction at the probe layer, and the
   same restricted to non-forced prompts.
5. **Trajectory view.** New figure in E0 (thinking on only): at the probe
   layer, project every kept position onto the suffix-fit probe direction
   and plot the projection against the position index, one line per
   horizon, colored by horizon. Shows whether the horizon reading holds,
   drifts, or collapses through the think block. Save the projections in
   the npz as `probe_track [n_prompts, n_positions]`.
6. **Answer prediction from the think end.** At the f = 1.0 position, fit a
   logistic probe for the chosen label (a vs b) with 5-fold CV and report
   accuracy per layer; compare to the same probe at the suffix and at the
   answer token. Put the three curves in one figure and the peak accuracies
   in the manifest.
7. **Explorer.** `export_pca3.py` and the notebook pick up the new positions
   automatically once regions are exported; check the chips group them
   under a "think" heading and that the 3D title shows the fraction.
8. **Tests.** Extend `tests/test_generation.py` with a forced-close case
   asserting the think positions are strictly increasing, lie within
   `[prompt_len, answer_start - pre_len)`, and that f = 0 and f = 1 map to
   the first and last think tokens. Add a test that `extract()` with
   thinking off yields an empty think region and the same layout as today.

## Constraints

- Do not change the prompts, the prefill, or the horizon bank.
- Keep `generate_batch()` output token-identical (there is a parity test).
- Keep every experiment's CLI and `run_all.sh` working; run the local smoke
  (`MODEL=Qwen/Qwen3.5-0.8B LIMIT=6 MODES=off ... bash experiments/run_all.sh`)
  plus a thinking-on E0 smoke with `--max-think-tokens 32 --limit 6`.
- Follow the repo's style: imports at top, no dead code, docstrings that say
  what a function returns.
- Nothing is committed and nothing is pushed to the GPU box without Jared.

## Definition of done

- 48 existing tests plus the new ones pass.
- Local thinking-on smoke on the 0.8B writes `activations.npz` with think
  positions, the trajectory figure, and a manifest carrying the per-fraction
  probe numbers.
- `export_pca3.py` and the notebook show think positions for a thinking-on
  snapshot.
- A short section in `02_execution_plan.md` records the design and the
  layout change, and `00_README.md`'s position diagram is updated.

## Open questions to raise with Jared before starting, if unsure

- Nine fractions or fewer? Nine adds ~630 MB to the thinking-on npz.
- Should forced-close prompts be excluded from the think-position analysis
  by default, or included with a flag? (Suggested: included, flagged, with
  the manifest reporting both.)
