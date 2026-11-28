# evaluation/: code for the evaluation pipeline

Code side of [`evaluation_design/`](../evaluation_design/README.md) (the design proposal). Owner: Evaluations
(random-vector baselines: Augusto). Drafted with Claude, 2026-11-18.

It holds the **shared random draws** (below) and a first **evaluation runner** (replay readout, last section). The draws are, the random directions that every steering method is compared
against (`[eval §4.1]` item 6, `[eval §6]`; asked for in `extraction_design/README.md` §8.2 item 6 and §13).

## Shared random draws: `shared_draws.py`

One file of random directions per (model, layer, site), made once from the extraction activations and used by both
teams, so "random" means the same thing in every report.

| Family | Draw | Rules out |
|---|---|---|
| `isotropic` (N1) | unit(z), z ~ N(0, I_d) | "any push of this size" |
| `covariance` (N2) | unit(g @ (X − mean X) / √(m−1)), g ~ N(0, I_m): N(0, Σ) of the activations | "any push along directions the model uses" |
| `real_diff` (N2') | unit(h_i − h_j), random pairs of distinct prompts | same, from real activation differences |

- **K = 99** per family (replay readout: smallest p = 0.01). The **generation readout uses rows `[:19]`** of the same
  arrays (smallest p = 0.05), not a separate set, so both readouts refer to the same directions.
- **Rows used:** bank split `extraction`, regime ≠ `no_horizon` (the prompts the methods are fitted on, "none" prompts
  included). The verification grid and the no-horizon twins are never used, even if they were extracted.
- Directions are unit vectors, as in `steering/` (an edit scales them by `x × typical_norm`).

### Running it

When the full activations exist (`steering/extract.py` output), one command per site:

```
cd evaluation
python shared_draws.py make --activations ../steering/activations/<run> \
    --bank ../prompt_bank/data/t0_s0/prompts.jsonl \
    --model google/gemma-4-31B-it --layer 38 --site label --out draws/
# same with --site prelabel
```

It writes `draws/<model>/L38_<site>/`:

```
draws.npz      isotropic, covariance, real_diff: [99, d] float32; real_diff_pairs: [99, 2] row indices into uids.txt
uids.txt       the prompts used, sorted (row i of the activations matrix)
manifest.json  seed keys, definitions, input SHA-256s (activations, index, bank, uid list), numpy/python versions
SHA256SUMS     hashes of the three files above
```

Check a copy (hashes, then regenerate every family and compare):

```
python shared_draws.py verify --draws draws/gemma-4-31B-it/L38_label \
    --activations ../steering/activations/<run> --bank ../prompt_bank/data/t0_s0/prompts.jsonl
```

The isotropic family needs only the hidden size, so it can be made before any activations exist; it is identical to
the one in the full file: `python shared_draws.py make --iso-only --d <hidden size> --model … --layer 38 --site label --out draws/`.

### Why the draws are the same everywhere

- **Seeds are not chosen by hand.** Each (model, layer, site, family) gets its seed from the SHA-256 of
  `ptm-shared-draws/v1|<model>|L<layer>|<site>|<family>`, so families and sites are independent and nobody picks numbers.
- **Stable across numpy versions.** numpy guarantees PCG64's raw stream and `SeedSequence`, not its samplers
  (`Generator.standard_normal` may change between versions). Normals are made by our own Box–Muller from the raw stream;
  `test_golden_values` pins the first values.
- **Row order does not matter:** rows are sorted by uid.
- **One step can differ in the last bits:** the matrix product for `covariance` (BLAS). Draws are stored in float32 and
  `verify` allows 1e-6, so the published file and its SHA-256 are the reference.

### Refusals

- fewer than 100 eligible prompts (a covariance from m prompts spans at most m−1 directions; this stops a smoke run
  from becoming the shared draws; `--min-rows` overrides for tests);
- an output directory that is not empty (draws are never overwritten);
- `index.jsonl` and the `.npy` with different row counts, or rows without a uid.

### Assumptions about extraction's files

Written from `extraction_design/README.md` only (not from the `steering/` code): `L<layer>_<site>.npy` holds one row per
line of `index.jsonl`, in the same order, and each line has the prompt's uid under the key `uid` (`--uid-key` if not).

### When to make new draws

A new baseline run or a re-extraction gives new activations, so new draws: the manifest records the activations' hash,
and `verify` fails against other activations.

### Tests

```
cd evaluation && python -m pytest -q test_shared_draws.py
```

Synthetic bank and activations: reproducibility, golden values, row-order invariance, iso-only = full, rows used, that
covariance draws follow the activations' variance, refusals, and `verify` catching tampered draws or other activations.

## Evaluation runner (replay readout): `replay_eval.py`, `analyze.py`, `run_local.sh`

First version of the pipeline in `evaluation_design/` §9, stages 0–5, on the replay readout plus a small generation check.
Built only on extraction's documented interface (`steer.methods.load`, `Method.refit`, `steer.edits.LinearEdit`,
`steer.hooks.replay_logits` / `steering`, `pbank.local_model.LocalChatModel`), not on its internals.

```
evaluation/run_local.sh                                          # Qwen3-1.7B, thinking off: the whole loop in ~15 min
THINK=1 CAP=4096 BATCH=4 TAG=think GEN_PAIRS=8 evaluation/run_local.sh   # thinking on
```

`run_local.sh` chains baseline → extract (all splits) → fit → bundle → hook gate (`check_hook.py --method` on CAA) →
shared draws → `replay_eval.py items calibrate n6 replay gencheck` → `analyze.py`. Every stage skips what exists.
With a methods directory that holds each method at several sites (the Gemma run: label and prelabel), pass
`--site`; two bundles of one method are refused, not silently merged.

- **items:** the prompts that can be read out. The answer label is located here independently (regex after the
  end-of-thinking token), then compared with extract.py's `label_index` (`label_crosscheck.json`). Prompts that pbank
  could not parse, or whose thinking never closed, are listed in `skipped.jsonl`, never dropped silently.
- **calibrate (P4):** CAA refitted without fold 0 of the extraction split; x swept on that fold's scored prompts; the
  smallest x whose order-balanced |mean odd effect| reaches 1 logit while the even part of the label mass,
  (mass(+x) + mass(−x))/2, stays ≥ 95% of unsteered (P2: a mass change that reverses with the sign is steering).
  The sign is a result, not a criterion. If no healthy x reaches the target, the largest healthy x is used and
  `calibration.json` says so.
- **n6 (damage-matched null):** for each of `--n6-methods` (default CAA and curve-displace), the method's damage at
  its replay request: mean next-token KL(unedited ‖ edited) over every token of 8 fixed neutral texts (`NEUTRAL_TEXTS`:
  plain prose, no amounts, options or durations), averaged over ±x. Then each of the first `--n6-k` isotropic draws is
  rescaled (bracket, then bisection on log strength) until its damage is within ±10% of the method's. `n6.json`
  records every draw's strength and KL; matched draws become the replay family `n6:<method>`, and a method is ranked
  only against its own. Cheap: forward passes on ~40-token texts.
- **replay:** every arm at +x and −x on the verification grid: the bundles (`curve-snap` also at x = 0; the
  within-regime CAAs `caa-none`, `caa-near_only`, `caa-both`, `caa-within` as comparison arms when present), and K
  draws of each null family: isotropic, covariance, real differences (the shared draws), in-span (random directions in
  the curve's subspace) and label-shuffled refits (`Method.refit`, horizons permuted within option order × regime) of
  CAA and both curves. Readout: y = logit(far) − logit(near) at the answer slot, and the label mass. Also the N0 check:
  zero edit vs a hook-free forward pass; its "replay matches the generated choice" rate is exact only for a greedy
  baseline (a sampled letter need not be the top prediction).
- **gencheck:** steered generation on a subset of verification pairs (none, CAA ±x, isotropic[0] ±x, curve-displace
  ±shift), in the baseline's thinking mode, cap and decoding (`--gen-samples N` passes on a sampled baseline):
  choice, parse failures, cap hits, reply length.
- **analyze.py → `report.md`:** per arm, the odd (steering) and even (disruption) parts; meaning vs letter by
  verification pair (C1); the switch point of P(far) vs log10(H/D) and its shift Δlog10 H* (P1); plateaus and slope
  (C2); label mass. Each method is ranked against each null family of its kind (push vs curve), and against its own
  N6 family. 95% intervals for methods, controls and comparisons: paired bootstrap over verification pairs (both
  option orders resampled together; `--boot`, default 1000; the same resamples for every arm). Readout agreement on
  the generation-check items: per arm, the share of generated choices equal to the replay prediction (y > 0 = far)
  and both readouts' P(far); per ± pair, the odd effect on P(far) in generation (bootstrap CI over items) vs replay.

- **Push methods:** `caa`, `trend` and `probe` bundles each become a `method` arm at x_push, ranked against the push
  null families; `reward` and `choice` are controls. A methods directory may hold any bundles at one layer (no `caa`
  needed) when `calibration.json` already exists: the nulls then take their typical norm from the first bundle (a
  curve bundle's from its `fit.json`). That is how a confirm run replays `screen.py`'s winner alone.
- **`--write-span {all,prompt}`** (default `all`): which tokens every edit writes to, in calibrate, replay (every arm,
  nulls and N6 draws included) and gencheck. `all`: prompt and reasoning. `prompt`: each row's own prompt tokens only
  (the patch reading), masked per row in the right-padded batch (`prompt_tokens`); in generation, the prefill pass
  (`steer.hooks` "prompt"). N6's damage is still measured on the neutral texts at every token. `calibration.json`
  records the span, and a calibration made at another span is refused.
- **`--tf32`** (default off): TF32 matmuls on CUDA. The confirm stage stays in plain float32; `screen.py` uses TF32
  only after its own check. Span, TF32 and replay dtype are in `settings.json` and the report header.

Not yet: concept-control comparisons beyond reporting, C3–C8 except the C2 signatures, readout sites other than the
answer label (prompt_end, think_end).
Tests: `python -m pytest -q test_replay_eval.py` (label location on the real Qwen3 tokenizer, strata shuffle, the
meaning/letter and switch-point arithmetic, the loop detector).

The calibration health bound was first the smaller of the ±x mean label masses. On Qwen3-1.7B (thinking off, fp32)
CAA at +x lowered the mass (0.845 at x = 0.4) while −x raised it (1.000), so that rule stopped at x = 0.02, where
the odd effect is noise. The even part stays at the unsteered 0.923 up to x = 0.4 and drops only at 0.8 (0.631).
`calibration.json` keeps both signs' masses (`label_mass_plus/minus`).
With thinking on (Qwen3-1.7B, bf16, under the old rule) the sweep got CAA no further than +0.27 logits even at 75%
of the unsteered mass, so there the rule was not what held calibration back.

## Screen and select: `screen.py`

Picks the one config (bundle = method at its layer and site, × write span) that the confirm run then tests once
against its nulls (`jared/autoresearch/GOALS_2026-11-27.md`, Design). The model loads once; each bundle is hooked
at its own layer.

```
python screen.py screen --bank B --run R --items-from <replay_eval out with items.jsonl> \
    --uids ../prompt_bank/data/t0_s0/splits/reserved_horizons.jsonl --uids-role test \
    --bundles <bundle dir> ... --write-span all prompt --out S [--tf32] [--deadline <unix time>] [--replay-batch 4]
python screen.py select --out S --confirm-out <confirm run's replay_eval --out>
```

- **screen:** items are the readable extraction prompts whose uid has the role (`test`: 9 horizon values the fit never
  saw), scored regimes only. Unsteered y once. Per config, the strength grid (`--push-grid` 0.1 0.2 0.4 0.8 × typical
  norm, `--shift-grid` 0.5 1 2 4 decades) at ±s, each step computed as calibrate does (`sweep_point`); the sweep stops
  at the first unhealthy strength or sign change. S = `choose_strength` "peak". One line per config in `screen.jsonl`
  (sweep, chosen strength, S, mass ratio, odd by regime and option order, letter part, seconds); a restart skips the
  configs there, and no config starts past `--deadline`. With `--tf32`, unsteered y is computed with and without TF32
  first; over 0.05 logits on any item, the whole screen runs without. `screen_settings.json` keeps each invocation's
  check and unsteered values.
- **select:** highest S among non-control configs with a healthy chosen strength; within 0.25 logits of the best, the
  smaller strength (by position in its own grid, so push and curve strengths compare). Writes `winner.json`,
  `screen.md` (every config by S) and `<confirm-out>/calibration.json` with the winner's strength and span, which
  replay_eval's calibrate stage returns unchanged. A curve winner has `x_push: null`: push nulls need a strength the
  screen did not choose.

Tests: `python -m pytest -q test_screen.py` (stop rule, peak score, regime and order split, resume, deadline, TF32
check, selection rule, the calibration accepted by replay_eval).

## Transfer to the eval types: `transfer_eval.py`

Stage 6 of `evaluation_design/` §9 (transfer), on the replay readout: do T0's methods, at the strength frozen on T0,
also steer the horizon on Jared's eval types, realistic variations of the T0 prompts (system prompts, multi-turn,
other domains, implied horizons, distractors; `jared/EvalsPromptTypes.md` on branch `jared-eval-types`)?

```
# in jared/ on jared-eval-types: one bank per eval type
python -m pipeline.s1_data_vectors.analysis.export_evaltype_banks     # -> ~/SPAR/results/horizon/evaltype_banks/
# then, with the T0 run
TRANSFER=1 TRANSFER_SET=transfer_sets/pilot_3types.json evaluation/run_local.sh   # the 300-prompt pilot
TRANSFER=1 evaluation/run_local.sh                                    # everything: 19 types, 2,630 prompts, every arm
```

- **Check sets: what to run, in one file.** `transfer_sets/<name>.json` picks the types (each with the result you
  expect, printed beside its numbers), the regimes and variants, a cap on rows per type, the arms (names, families or
  methods, e.g. `["caa", "curve-displace", "isotropic", "shuffled:caa"]`) and K. Rows are taken as whole items (base,
  swap and twin together) in bank order, which alternates near and both items. Outputs go to `transfer/<name>/`, so a
  pilot and the full run never share a baseline. Unknown keys are refused. `pilot_3types.json`: `system_prompt`
  (expected to work), `prose_natural` (might work), `coding_agent` (not expected to work), 100 rows each: 20 near and
  20 both items with their swaps, and twins for the first 10 per regime.

- **Banks.** pbank's schema, tier `evaltype`, split `transfer`. Fitting, calibration and the shared draws take only split
  `extraction`, and the T0 replay only `verification`, so these prompts never reach them. The eval-type record (type,
  variant, twin, its own regime and truth) is kept in `notes`. Conversation-rendered types (`system_prompt`,
  `system_prompt_clear`, `multiturn_analyst`) keep their messages in `notes["messages"]`; their `text` is a transcript,
  for reading and hashing only.
- **baseline:** the unsteered reply to every `near_only`/`both` prompt, with the T0 run's model, thinking mode, cap and
  batch size, in baseline.py's file format (`transfer/<type>/run/`). It is generated here, not by
  `prompt_bank/baseline.py`, which sends `text` as one user turn: a conversation is rendered from `notes["messages"]`
  (a single user text renders exactly as pbank's `chat_text`, checked in the tests). Resumes; refuses a run with other
  settings; the run's hash covers every turn of a conversation.
- **items, replay:** replay_eval's own stages, per type in `transfer/<type>/` (`stage_replay(..., split="transfer")`):
  label location and `skipped.jsonl`, the unsteered pass and the N0 check, every arm at ±x with x read from the T0 run's
  `calibration.json` (never recalibrated). The shuffled-label refits use the T0 bank and activations, as on T0.
- **report → `transfer/transfer_report.{md,json}`:** per type and group (`prompts`: the type's base and swap prompts;
  `twins`: the same options in the T0 template with the horizon stated), next to T0's verification grid: per method and
  control the meaning (order-balanced, on the 40 base/swap pairs of a type), the odd part over all items, Δlog10 H*, the
  label mass at ±x, the unsteered label mass and accuracy, and each statistic ranked against each null family
  (analyze.py's `arm_results` and `compare_to_nulls`). Types without a bank or results are listed as not run.
- **Prompting baseline** (`evaluation_design` §9 stage 5, "vs prompting baseline"): one sentence appended as the last
  line of the last user turn, read like an edit (the generated reasoning kept, no hook; `Replayer.run_prompted`). Arms,
  family `prompting`: `prompt:disposition` (+ "be more long-term-minded", − "short-term-minded"), `prompt:shift`
  (+ "as if the horizon were 100 times longer", − "shorter"), `prompt:neutral` ("Think carefully about this decision.",
  one side: the floor for adding any sentence). Wording from Jared's E22 (Qwen3-8B, regenerated reasoning): both pairs
  moved the choice as far as the best steering, the disposition pair also moved rows it should not. Reported beside the
  methods, not ranked against the random nulls (no matched null; compare with `prompt:neutral`). On by default in the
  transfer stage (drop with a check set's `arms`), opt-in on T0 (`PROMPTING=1`, `replay_eval.py --prompting`). Not yet
  in `gencheck`: with regenerated reasoning the sentence can act on the reasoning too, which this readout cannot see.
- **The switch point only where it applies:** a group needs ≥ 5 readable items with each correct option. `city_no_values`
  gives no values, so its `both` prompts have no correct answer and no switch point is fitted. Eval-type horizons sit
  ≥ 0.25 decades from both delays, so even where it is fitted the location is less pinned down than on the T0 grid: read
  meaning and odd first.
- **Cost.** The 19 exported types hold 2,630 `near_only`/`both` prompts, about 13 × T0's grid, and every arm is replayed
  on each; the shuffled refits are redone per type. With thinking on, the baseline dominates (at the 75 s per prompt
  measured for Gemma 4 31B in `prompt_bank/README.md`, about 55 GPU-hours); the pilot set is about a ninth of that.

Tests: `python -m pytest -q test_transfer_eval.py` (conversation rendering on the real Qwen3 tokenizer, the baseline
with a fake model, the replay stage on the transfer split with fake edits, the report on hand-made arm results, and
analyze.py's report card after its refactor into `arm_results` / `compare_to_nulls`, check-set loading, row and arm
selection, and the pilot set's row counts on the exported banks when they are present).

## Loop check: `check_loops.py`

Counts how many token-cap hits in a generation run are repetition loops (a 64-token block repeated ≥ 3× in the last 1024
tokens) rather than long reasoning. Reads `responses.jsonl` of any prompt_bank run; `--bank` adds a breakdown by regime.

```
python check_loops.py --run ../prompt_bank/runs/<model>__<bank> --bank ../prompt_bank/data/<bank>
```

On Qwen3-1.7B with thinking on (greedy, cap 4096), 323 of 337 cap hits are loops, so the 330 missing answers are mostly a
decoding problem. The Qwen3 model cards advise against greedy decoding in thinking mode (recommended: temperature 0.6,
top-p 0.95, top-k 20).

## Sampling test: `sampling_test.py`

Re-generates a subset of a greedy pbank run with Qwen's recommended sampling (seeded per batch; same cap and batch
size): looping cap hits, stratified by regime, plus answered controls. Uses pbank's own model and parser; only
`generate` is wrapped to sample.

```
python sampling_test.py --bank ../prompt_bank/data/t0_s0 --greedy ../prompt_bank/runs/<run> --out results/<run>__sampling
```

On Qwen3-1.7B with thinking on (cap 4096): of 100 prompts that looped under greedy, 62 answer and only 5 still loop. The
other 33 are long reasoning cut at the cap (median reply 3,504 tokens). On 31 controls, sampling keeps the greedy choice
on 21 of 26 and loses 5 answers. Speed is unchanged (~36 s/prompt): one long reply still holds its whole batch.

## Example reports: `reports/`

Saved outputs of plumbing runs, as examples of what the runner produces (`results/` itself is gitignored). These are
small models, so the numbers are not evidence about steering.

- `Qwen3-1.7B__t0_s0__think_report.md`: report card of the full thinking-on run (2026-11-19, ~9–10 h on an 8 GB GPU).
- `Qwen3-1.7B__t0_s0__think_loops.txt`: `check_loops.py` on the same run.
- `Qwen3-1.7B__t0_s0__think_sampling.json`: `sampling_test.py` summary on the same run (100 looping + 31 controls).

## Diagnostics report: `diagnose.py`

CPU only (numpy and json), seconds; reports and never gates. Spec: `jared/EvalDiagnosticsSpec.md`. `run_local.sh` runs it
as its last step; it also works right after fitting, before any replay (sections whose inputs are missing say
"skipped: <file>").

```
python diagnose.py <run dir> --artifacts <dir with activations/ and methods/> --bank prompt_bank/data/t0_s0 \
    --baseline <dir with responses.jsonl> [--out DIR]     # --out defaults to the run dir
```

It writes `diagnostics.html` (one self-contained page: summary of every "look" flag at the top, then sections A data and
labels, B baseline behaviour, C each fitted direction, Cd curve dimensionality, D strength, E outcomes, F readout noise), `diagnostics.json`
(every number) and `diagnostics_items.csv` (one row per prompt). Every check shows its value, a threshold and a flag;
the thresholds are constants at the top of `diagnose.py`. Directions are refitted from each bundle's `fit_uids.jsonl`
and the activations (`caa`, `caa-within`, `choice`, `regime`, `reward`), so the report does not trust the saved
cosines. Tests: `python -m pytest -q test_diagnose.py`.

### Curve dimensionality (diagnose.py section Cd) and `layer_sweep.py`

How many dimensions does the curve need, and is `k = 4` enough? Section C2 answers it for every site extracted at the
run's layer, on the fit prompts of the curve bundle (of any bundle, if there is no curve bundle), with and without
`none`. Per site it prints one line, e.g. "curve at k=4, held-out horizons: rho 0.60 (ridge 0.96); k=8 adds +0.09; 0 of
10 components tested above the shuffled-horizon null", and a table: for k = 1..8 the k-th eigenvalue of the per-horizon
means over the 95th percentile of the same with the horizon labels shuffled across prompts (above 1 is above chance;
50 shuffles, seed 0), and for k in {1, 2, 3, 4, 6, 8} the held-out-horizon Spearman and median |t-hat - t| (decades) of
the curve, with the ridge probe of log10 horizon beside it. Folds hold out whole horizons (5 folds, horizon names dealt
in turn); prompt-level folds leak. The rank-matched null cannot reject a nuisance dimension that ranks below a real
component, so read the held-out columns before the count. It reports and never gates, and the numbers are in
`diagnostics.json` (`sections.Cd`). Cost: about 45 s for the 4 sites of the Qwen3-8B quick run (CPU, 8 cores; the curve
cross-validation and the shuffles dominate). The computation is in `curve_dim.py`.

`layer_sweep.py <activations dir> --bank DIR [--uids splits/reserved_horizons.jsonl --uids-role fit]` does the layer by
site version offline, for every `L<layer>_<site>.npy`: ridge-probe and curve (k = 4) held-out-horizon rho, CAA split-half
reliability (100 halvings) and the CAA classes' far-rate gap, on prompts with a horizon and no `none`. It writes
`layer_sweep.json` and `layer_sweep.html` (one shaded table per metric, no external dependencies, light and dark).
Tests: `python -m pytest -q test_curve_dim.py test_layer_sweep.py`.
