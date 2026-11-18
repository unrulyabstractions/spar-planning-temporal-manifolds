# evaluation/: code for the evaluation pipeline

Code side of [`evaluation_design/`](../evaluation_design/README.md) (the design proposal). Owner: Evaluations
(random-vector baselines: Augusto). Drafted with Claude, 2026-11-18.

For now it holds one thing: the **shared random draws**, the random directions that every steering method is compared
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
