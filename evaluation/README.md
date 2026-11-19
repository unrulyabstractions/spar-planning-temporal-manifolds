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
shared draws → `replay_eval.py items calibrate replay gencheck` → `analyze.py`. Every stage skips what exists.

- **items:** the prompts that can be read out. The answer label is located here independently (regex after the
  end-of-thinking token), then compared with extract.py's `label_index` (`label_crosscheck.json`). Prompts that pbank
  could not parse, or whose thinking never closed, are listed in `skipped.jsonl`, never dropped silently.
- **calibrate (P4):** CAA refitted without fold 0 of the extraction split; x swept on that fold's scored prompts; the
  smallest x whose order-balanced |mean odd effect| reaches 1 logit while the label mass stays ≥ 95% of unsteered.
  The sign is a result, not a criterion. If no healthy x reaches the target, the largest healthy x is used and
  `calibration.json` says so.
- **replay:** every arm at +x and −x on the verification grid: the 5 bundles (`curve-snap` also at x = 0), and K
  draws of each null family: isotropic, covariance, real differences (the shared draws), in-span (random directions in
  the curve's subspace) and label-shuffled refits (`Method.refit`, horizons permuted within option order × regime) of
  CAA and both curves. Readout: y = logit(far) − logit(near) at the answer slot, and the label mass. Also the N0 check:
  zero edit vs a hook-free forward pass.
- **gencheck:** steered generation on a subset of verification pairs (none, CAA ±x, isotropic[0] ±x, curve-displace
  ±shift), in the baseline's thinking mode and cap: choice, parse failures, cap hits, reply length.
- **analyze.py → `report.md`:** per arm, the odd (steering) and even (disruption) parts; meaning vs letter by
  verification pair (C1); the switch point of P(far) vs log10(H/D) and its shift Δlog10 H* (P1); plateaus and slope
  (C2); label mass. Each method is ranked against each null family of its kind (push vs curve).

Not yet: damage-matched null (N6), concept-control comparisons beyond reporting, C3–C8 except the C2 signatures,
bootstrap CIs, the replay-vs-generation agreement statistic (the raw generation table is there).
Tests: `python -m pytest -q test_replay_eval.py` (label location on the real Qwen3 tokenizer, strata shuffle, the
meaning/letter and switch-point arithmetic, the loop detector).

The calibration health bound (the worst held-out item keeps ≥ 95% of its unsteered label mass, `--min_mass_ratio`) is an
untested default and probably too strict: on Qwen3-1.7B with thinking on, it stopped calibration at x = 0.02.

## Loop check: `check_loops.py`

Counts how many token-cap hits in a generation run are repetition loops (a 64-token block repeated ≥ 3× in the last 1024
tokens) rather than long reasoning. Reads `responses.jsonl` of any prompt_bank run; `--bank` adds a breakdown by regime.

```
python check_loops.py --run ../prompt_bank/runs/<model>__<bank> --bank ../prompt_bank/data/<bank>
```

On Qwen3-1.7B with thinking on (greedy, cap 4096), 323 of 337 cap hits are loops, so the 330 missing answers are mostly a
decoding problem. The Qwen3 model cards advise against greedy decoding in thinking mode (recommended: temperature 0.6,
top-p 0.95, top-k 20).

## Example reports: `reports/`

Saved outputs of plumbing runs, as examples of what the runner produces (`results/` itself is gitignored). These are
small models, so the numbers are not evidence about steering.

- `Qwen3-1.7B__t0_s0__think_report.md`: report card of the full thinking-on run (2026-11-19, ~9–10 h on an 8 GB GPU).
- `Qwen3-1.7B__t0_s0__think_loops.txt`: `check_loops.py` on the same run.
