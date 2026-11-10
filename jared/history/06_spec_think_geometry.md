# Spec: mid-thinking geometry in E0

Status: draft for Jared's approval, 2026-09-29. Companion to
`05_prompt_think_geometry.md` (the task) and `02_execution_plan.md`.

## 1. Intent

Keep hidden states from inside the think block so PCA, the horizon probe,
and the explorer can answer: does the horizon axis persist, drift, or
dissolve while Qwen3-8B reasons, and does the state at the end of thinking
predict the answer? Add a dense per-token track of the probe reading so the
*where* of any change is visible, not only the *whether*.

Decisions already taken with Jared:

- Four think fractions, `0.25, 0.5, 0.75, 1.0`. f = 0 is dropped because it
  is one token past the `<think>\n` already kept in the suffix.
- The turn suffix stays intact. The current probe position is the newline
  right after `<|im_end|>` (R² 0.9996 at L11 in both modes), so trimming the
  early suffix tokens would delete the best position and save no compute.
- Force-closed prompts are included by default, flagged per prompt, and
  every think-position number is reported twice: all prompts and non-forced.
- A dense track (probe projection at every think token) is in scope.
- Progress bars on every loop longer than a few seconds.

## 2. Position layout (thinking on)

```
[ suffix: <|im_end|> \n <|im_start|> assistant \n <think> \n ]   region suffix = (0, S)
[ think@0.25  think@0.5  think@0.75  think@1.0 ]                  region think  = (S, S+4)
[ pre: </think> \n\n I choose: (+ any ' **') ]                     region pre    = (S+4, S+4+P)
[ answer: label ) <|im_end|> ... ]                                 region answer = (S+4+P, S+4+P+R)
```

Think position for fraction f in prompt g:
`g.prompt_len + round(f * (g.think_len - 1))`. f = 1.0 is the last token
before the close (`</think>` itself is the first pre token). If
`think_len < 4` two fractions can land on the same token; the columns are
then duplicates, which is harmless and only happens on toy budgets.

Thinking off: the think region is `(S, S)`, empty, and the layout is
byte-identical to today's.

`n_suffix` stays in `Extraction`, the npz, and the manifest with its current
meaning, the index of the answer token, so E1 to E5 run unchanged.

## 3. Changes by file

### `config.py`

- `think_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)`. Validated
  in `__post_init__`: strictly increasing, all in `(0, 1]`.
- `common.py` gets `--think-fractions 0.25,0.5,0.75,1.0`.

### `extract.py`

- New pure function `think_positions(prompt_len, think_len, fractions)` →
  list of absolute indices. Tested directly.
- `keep` becomes `suffix ++ think ++ pre ++ answer`. Thinking off passes an
  empty `fractions` so nothing else changes.
- `Extraction` gains:
  - `regions: dict[str, tuple[int, int]]` with keys `suffix, think, pre, answer`
  - `think_fractions: tuple`
  - per kept prompt: `forced_mask [n]` bool, `think_len [n]`, `pre_len [n]`,
    `keep_idx [n, n_positions]` absolute token index of every kept position,
    `keep_ids [n, n_positions]` token id at every kept position,
    `label_logits [n, 2]` logits for the `a` and `b` label ids at the
    position that predicts the label (`answer_start - 1`), read from the
    same forward pass.
  - `think_text [n]` decoded think block per kept prompt (empty with
    thinking off); goes to `answers.json`, not the npz.
- `tokens` for think positions is `f"think@{f:g}*"`.
- The answer-position token distribution print and the not-a-label warning
  stay exactly as they are.
- Progress: `tqdm` over prompts in the extraction loop, `desc="extract"`,
  `mininterval=2`, `dynamic_ncols=True`. The `log_every` argument goes away.

### `model_io.py`

- `generate_batch` gets a `tqdm` over batches, `desc="generate"`. Output
  tokens are untouched; the parity test guards this. `log_every` goes away.
- Nothing else. The unused `regions` field on `Generation` is removed
  (no dead code).

### `geometry.py`

- `horizon_probe` already returns the fitted RidgeCV; keep it. New:
  - `probe_directions(activations, has_h, log_h, pos)` → `coef [L, d]`,
    `intercept [L]`, one probe per layer fit on all horizon prompts at `pos`.
  - `probe_track(activations, coef, intercept)` → `[L, n, n_positions]`,
    the predicted log10 horizon at every kept position. In log-year units,
    so the figure's y axis reads directly against the colorbar ticks.
  - `label_probe(X, y, folds=5, seed=0)` → 5-fold stratified CV accuracy of
    a logistic regression (standardised features, `C=1`) predicting the
    chosen label. Returns `nan` when a class has fewer than `folds`
    members. Also returns the majority-class baseline.
  - `label_probe_sweep(activations, y, positions)` → `acc [L, len(positions)]`.
- `sweep` and `probe_sweep` get a `tqdm` over layers, `desc="pc1 sweep"` and
  `desc="probe sweep"`. The existing per-layer table print stays.

### `plots.py`

- `_region(pos, regions, tokens)` replaces `_region(pos, n_suffix)`:
  `"turn"`, `"think 0.25"`, `"pre"`, `"answer"`, `"response"`.
  `pca_panels` and `pca3_panels` take `regions` instead of `n_suffix`.
- New `track_panel(track, tokens, regions, has_h, log_h, forced, path, title)`:
  x = kept position index with region bands shaded and labelled, y =
  projected log horizon, one line per prompt, turbo-coloured by horizon,
  forced prompts dashed, no-horizon prompts grey. Horizontal colorbar
  ticks reuse `_TICKS`.
- New `dense_panel(dense, think_len, has_h, log_h, forced, path, title)`:
  x = position within the think block as a fraction 0 to 1 (each prompt
  rescaled to its own length), y = projected log horizon, same colouring,
  lines drawn with a 32-token running mean so 3k-token traces are legible.
  A second axis below shows the same lines against absolute token index
  so the forced-close budget is visible as a hard right edge.
- New `label_probe_panel(acc, baseline, labels, path, title)`: accuracy per
  layer, one curve per compared position, dashed line at the baseline.

### `runs.py`

- `save_activations` writes, in addition to today's keys:
  `region_suffix, region_think, region_pre, region_answer` (each `[2]`),
  `think_fractions`, `forced_mask`, `think_len`, `pre_len`, `keep_idx`,
  `keep_ids`, `label_logits`, `rho_nf`, `r2_nf`, `probe_coef`,
  `probe_intercept`, `probe_track`, `dense_track` (`[n, max_think_len]`,
  nan-padded), `dense_layer`, `label_acc [L, 4]`, `label_acc_nf [L, 4]`.
  Optional arrays default to empty so `test_activations_roundtrip` and
  callers that pass only `rho` keep working.
- `load_activations` returns every key present and, for an npz written
  before this change, derives `regions` from `n_suffix`:
  `suffix=(0, n_suffix)`, `think=(n_suffix, n_suffix)`, `pre=(n_suffix,
  n_suffix)`, `answer=(n_suffix, n_positions)`. This keeps
  `export_pca3.py` working on `snapshots/v1_noprefill`.

### `experiments/exp0_baseline.py`

After the existing sweep and probe (unchanged), and only when
`cfg.thinking == "on"`:

1. `rho_nf, r2_nf` = the same two sweeps restricted to non-forced prompts.
2. `coef, intercept` = `probe_directions(...)` at `probe_pos`;
   `track = probe_track(...)`; write `probe_track.png`.
3. Dense track: second forward pass per kept prompt, `tqdm`
   `desc="dense track"`, hidden state at `probe_layer` over
   `[prompt_len, prompt_len + think_len)` projected with
   `coef[probe_layer]`; write `dense_track.png`.
4. Label probe at four positions: suffix `probe_pos`, `think@1.0`, last
   pre token (the state that *predicts* the label), answer token (a
   ceiling: its state has already seen the label). Curves on
   `label_probe.png`; all-prompt and non-forced versions.
5. Manifest additions:
   ```
   regions, think_fractions,
   think_positions: {"0.25": {rho, r2, rho_nf, r2_nf}, ...}   # at probe_layer
   label_probe: {position: {peak_acc, peak_layer, peak_acc_nf, peak_layer_nf}}
   label_baseline, dense_layer, forced (count, unchanged)
   ```
6. `answers.json` rows gain `forced`, `think_len`, `label`, `think_text`.

Thinking off: the script runs exactly as today plus the new npz metadata
arrays; no new figures.

### `analysis/export_pca3.py`, `analysis/pca3_explorer_nb.py`

- Export `regions` and `think_fractions` per mode; keep `n_suffix`.
- Chips: grouped headings `suffix`, `think`, `pre`, `answer`, `response`
  from `regions`; think chips read `think@0.25`.
- 3D title uses the same region label helper, so it shows `think 0.25`.
- Rebuild `pca3_explorer.ipynb` with the README's nbformat snippet.

### Docs

- `02_execution_plan.md`: a short "Mid-thinking positions" note under
  build notes: layout, the suffix decision with the R² evidence, storage
  list, the dense-track second pass.
- `00_README.md`: position diagram updated to the layout in section 2.

## 4. Storage cost

Per extra kept position on the 8B: 116 × 37 × 4096 × 4 B ≈ 70 MB. Four
think positions add ≈ 280 MB to the thinking-on npz. Everything else added
is under 10 MB combined (`probe_track` ≈ 0.3 MB, `dense_track` < 2 MB,
`probe_coef` ≈ 0.6 MB, `label_logits` and index arrays negligible).

## 5. Runtime cost

- Extraction: unchanged; the forward pass covers the full sequence already.
- Dense track: one extra forward pass per prompt at the same length,
  roughly the extraction time again, a few minutes on the 4090 for 116
  prompts.
- Non-forced sweeps: one more `sweep` and one more `probe_sweep`, same
  cost as the existing ones.
- Label probe: 4 positions × 37 layers × 5 folds of a 116 × 4096 logistic
  fit, well under a minute.

## 6. Tests

Existing 48 stay green. New:

- `test_generation.py`
  - `test_think_positions_forced_close`: 0.8B, `max_think_tokens=32`,
    fractions `(0.25, 0.5, 0.75, 1.0)`: positions strictly increasing, all
    in `[prompt_len, answer_start - pre_len)`, last equals
    `prompt_len + think_len - 1`; with fractions `(0.0, 1.0)` the first
    equals `prompt_len`.
  - `test_extract_thinking_off_layout`: `extract()` on 3 prompts, thinking
    off: `regions["think"]` is empty, `n_positions == n_suffix + n_pre +
    n_response`, tokens contain no `think@`, `keep_idx` rows are strictly
    increasing, `label_logits` has shape `[n, 2]`.
  - `test_extract_thinking_on_layout`: `max_think_tokens=32`: four
    `think@` tokens between suffix and pre, `forced_mask.all()`.
- `test_geometry.py`
  - `test_label_probe_separable`: synthetic linearly separable data →
    accuracy 1.0; a single-class `y` → `nan`.
  - `test_probe_track_is_prediction`: `probe_track` equals `X @ coef +
    intercept` on random data.
- `test_runs.py`
  - `test_activations_roundtrip_regions`: new keys round-trip; an npz
    without them loads with derived regions.
- `test_config.py` (new, tiny): bad `think_fractions` raise.

## 7. Smoke runs (local, 0.8B, CPU)

```
MODEL=Qwen/Qwen3.5-0.8B LIMIT=6 MODES=off bash experiments/run_all.sh
python experiments/exp0_baseline.py --model Qwen/Qwen3.5-0.8B --thinking on \
    --max-think-tokens 32 --limit 6 --skip-behavior
python analysis/export_pca3.py out    # then open the notebook on it
```

The thinking-on smoke must write `activations.npz` with four `think@`
columns, `probe_track.png`, `dense_track.png`, `label_probe.png`, and a
manifest with `think_positions` and `label_probe`.

## 8. Task decomposition (draft, for after spec approval)

Each task is one commit-sized change, verified before the next.

1. `config.py`: `think_fractions` + validation + CLI flag. Test.
2. `extract.py`: `think_positions()` pure function. Test.
3. `extract.py`: regions, keep layout, think token labels, per-prompt
   arrays, `label_logits`, tqdm. Thinking-off and thinking-on layout tests.
4. `model_io.py`: tqdm in `generate_batch`, drop `log_every` and the unused
   field. Parity test.
5. `runs.py`: save/load new keys, legacy regions. Roundtrip test.
6. `plots.py`: `_region` on regions; update `pca_panels`, `pca3_panels`,
   E0 call sites. Run thinking-off smoke, compare figures to today's.
7. `geometry.py`: `probe_directions`, `probe_track`, tqdm on sweeps. Test.
8. `plots.py` + E0: `track_panel`, `probe_track.png`, non-forced sweeps,
   `think_positions` in the manifest.
9. E0: dense second pass, `dense_panel`, `dense_track.png`.
10. `geometry.py` + `plots.py` + E0: `label_probe`, sweep, panel, manifest.
11. `answers.json` extras.
12. `export_pca3.py` + notebook chips/titles; rebuild ipynb; check on
    `snapshots/v1_noprefill` and on the local thinking-on smoke.
13. Full local `run_all.sh` off smoke + thinking-on E0 smoke; full pytest.
14. Docs: execution plan note, README diagram.

## 9. Open points for Jared

- The answer-token label probe is a ceiling (the state has seen the
  label). I've added the last pre token as the honest "predicts the label"
  position. Keep all four, or drop the answer token?
- Dense track at the probe layer only. Adding the display layer too would
  double a small cost. Fine to start with one?

## 10. Amendments after the 2026-09-30 box runs (approved: "yes build")

Defaults taken: four label-probe positions with the answer token marked as a
ceiling; dense track at the probe layer only; behavior skipped for thinking
on; generation cache in.

- **No more skipping on pre-length.** The thinking-on E0 dropped 55 of 116
  prompts because forced closes (`\n</think>\n\n`) and natural closes
  (`</think>` + one whitespace token) differ by a token, and a markdown lead
  before the label adds another. New fixed-width layout:
  `suffix | think | pre | pre-label | answer` where `pre` is the
  `2 + len(prefill_ids)` tokens starting at `</think>` (thinking on only),
  `pre-label` is the single token at `answer_start - 1` (the state that
  predicts the label, whatever the lead was), and `answer` starts at the
  label. Lead tokens are no longer kept as columns. Thinking off:
  `suffix | pre-label | answer`. `Generation` gains `lead`.
- **Generation cache.** `Settings.cache_dir` (CLI default
  `<out base>/gen_cache`, `--no-cache` disables, tests pass `None`). Key =
  sha256 of model, thinking, max_think_tokens, prefill, n_response, system,
  and the prompt's input ids. One `.npz` per prompt holding the
  `Generation` fields. Hits skip generation; misses are batched as before.
- **Phase timing.** `Manifest.phase(name)` context manager prints
  `[HH:MM:SS] <name> done, <s> s` and records `phases: {name: seconds}`.
  `code_hash` (sha256 over `spar_horizon/*.py` + `experiments/*.py`) is
  written to the manifest since the box has no git.
- **run_all.sh** passes `--skip-behavior` to E0 when the mode is `on`.
- **Manifest per prompt.** `forced_mask`, `think_len`, `lead` are saved per
  prompt in the npz; `answers.json` gets `forced`, `think_len`, `label`,
  `think_text`.
