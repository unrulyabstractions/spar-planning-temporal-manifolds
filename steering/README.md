# Steering vectors

Computes the time-horizon steering directions of the report (Appendix A.3) for one model, from a prompt bank and
an unsteered baseline run of that model. It produces:

- the **linear direction (CAA)**;
- the **manifold curve**;
- the **controls**: random directions and a reward-contrast direction;
- **diagnostics** for auditing all of the above.

It is roadmap step 2's first half ("extract"). Verification and calibration (switch points, the gate, alpha_k)
are not here yet.

Prompt generation stays in `../prompt_bank`, and nothing in this directory builds or changes prompts. This
package reads the bank's records and reuses pbank's answer parser and model runner. It imports them rather than
copying them, so the two packages can't drift apart on what a regime, a label or a parsed choice is.

## Steps

```
cd ../prompt_bank
uv run --extra local python baseline.py --bank data/t0_s0              # 1. generate (Gemma 4 31B, thinking on)
cd ../steering
uv sync --extra dev
uv run python extract.py --bank ../prompt_bank/data/t0_s0 \
    --run ../prompt_bank/runs/gemma-4-31B-it__t0_s0 --layers all      # 2. activations, every layer and site
uv run python fit.py --activations activations/gemma-4-31B-it__t0_s0 \
    --bank ../prompt_bank/data/t0_s0 --layer 38                        # 3. directions, curve, diagnostics
uv run python bundle.py --fit fits/gemma-4-31B-it__t0_s0/default/L38_label \
    --activations activations/gemma-4-31B-it__t0_s0                    # 4. one bundle per method, for hand-over
uv run python tools/check_hook.py --method methods/gemma-4-31B-it/caa__gemma-4-31B-it__t0_s0__default__L38_label \
    --bank ../prompt_bank/data/t0_s0                                   # gate: the hook and the method on this model
```

1. **Generate.** `baseline.py` runs the model once over the bank and stores every reply's exact generated token
   ids. This is the only generation step.
2. **Extract.** `extract.py` generates nothing. For each prompt it replays the stored ids up to and including the
   answer-label token as one teacher-forced forward pass, and keeps the residual stream at the requested layers.
   It is cheap: minutes even for 31B, against hours for generation. So other layers can be extracted later
   without regenerating.
3. **Fit.** `fit.py` runs on the CPU, takes seconds, and needs no GPU.
4. **Bundle.** `bundle.py` packages the fit's methods: one directory per method, with what it is, how to apply
   it and what it was fitted on (below). `tools/check_hook.py --method` then checks on the GPU that the hook
   and the method behave in that model's real generation setup.

Each step writes its own directory with its settings, inputs and their SHA-256 hashes. Each refuses to overwrite
an earlier output, and refuses inputs that don't match: a different bank, different weights, or an altered array.

## Definitions and conventions

- **Layer L** is the residual stream after decoder block L, counting from 1. It equals `hidden_states[L]`, and is
  read or edited by hooking `decoder_layers(model)[L-1]` (`steer/layers.py`). The report's "layer 38 of 60" on
  Gemma 4 31B is read this way. `tests/test_layers.py` checks it exactly on a tiny Qwen3 and on real Qwen3.5-0.8B
  weights; Gemma 4's block list is found at `model.language_model.layers`.
- **Sites** (`steer/sites.py`):
  - `label` is the token carrying the label character of the model's own "I choose: b)". It is the report's
    "answer-label token" and fit.py's default.
  - `prelabel` is the token before it, whose state predicts the label.
  - `prompt_end` is the last token of the model input, whose state predicts the first generated token.
  - `think_end` is the end-of-thinking token, whose state predicts the first token of the answer. Thinking-on runs
    only.

  All are extracted, as rows of the same prompts, and `fit.py --site` takes any of them. `extract.py --layers all`
  keeps every layer a hook can steer (1 .. n_layers − 1), so a fit at another layer or site needs no GPU. At the
  prelabel site, extract.py also keeps the 64 largest next-token logits with their token ids
  (`prelabel_top_logits.npz`, read by `steer.activations.top_logits`): the unsteered scores of the answer letters.
- **H/D** is the stated horizon over the **far** option's delay (report A.1), i.e. the bank's `h_over_far_delay`.
- **Log-time t** is log10 of the horizon in days, taken at its grid value. A rewritten "100 weeks" sits with
  "23 months".
- **Strength.** Every direction is stored as a unit vector. An edit adds `strength × typical_norm × direction`,
  where `typical_norm` is the mean activation norm at that layer and site. Every direction applied at the same
  strength is therefore matched in norm, as the report requires of the controls.

## Choices made where the report is silent or ambiguous

All of these are command-line options, recorded in each output's JSON. Defaults follow the report where it says
something.

1. **"None" prompts are included by default.** The report's Methods section says they are "used for extraction".
   `--no-include-none` leaves them out of everything: CAA, the controls and the curve. Evidence from Jared's Qwen3
   runs:
   - the model's choices on "none" prompts are confident but follow option order and model-specific habits;
   - the 8B's long-minus-short difference of means steered backwards because of the regime mix
     (`origin/jared`, `jared/08_results_steering.md`).

   What leaving them out costs, on the t0_s0 extraction split:
   - **The curve's range.** All 24 grid horizons below 10 days are pure "none", so the curve starts at 10 days, and
     the next ten horizons rest on 1–9 prompts each. The verification grid (0.88–124 years) stays inside the range.
   - **CAA's short class** shrinks from 350 prompts (279 none + 71 near_only) to 71. The two classes then differ
     perfectly in which answer is correct, so CAA gets closer to a "choose far" direction. On the Qwen3.5-2B smoke run
     its cosine with the choice direction rose from 0.862 to 0.904.

   The default follows the written spec for the first pass (Alan, 2026-11-14). Switching it needs only a refit: the
   activations of "none" prompts are always extracted.

   Two selection rules apply whatever the options:
   - verification-grid prompts are never fitted on, even if `extract.py` was run on them;
   - no-horizon prompts are never used.

   `--exclude-horizons` leaves out given stated horizons, e.g. to hold them out for calibration. fit.json counts the
   prompts left out for each reason.
   `--uids FILE [--uids-role ROLE]` fits only on the prompts listed in a jsonl file (rows with `uid`, and `role` for
   `--uids-role`), e.g. `prompt_bank/data/t0_s0/splits/reserved_horizons.jsonl` with role `fit`. It combines with the
   rules above; fit.json records the file, its sha256 and role under `uids_file`, and counts the prompts it leaves out as
   `excluded.not_in_uids` (only prompts the other rules keep). `--draws` is not restricted: the shared draws come from
   every fitted-split prompt of the extraction and enter the diagnostics only (see 4).
2. **The CAA classes are not balanced by regime.** With D the far delay, H/D >= 4 holds only "both" prompts (the
   ones where "far" is the answer). H/D <= 1/4 holds "none" and "near_only" prompts. On the t0_s0 bank that is
   57 against 279 + 71.

   fit.py reports this composition. It also reports the cosine between CAA and a *choice* direction (replies
   that chose far, minus those that chose near). That measures how much of CAA is "which option was chosen"
   rather than "how long the horizon is".

   **Within-regime CAA** removes that difference: inside one regime the correct answer is the same for every prompt.
   - In each regime, long is the third of its prompts with the longest horizons and short the third with the
     shortest, split on log-horizon (`--within-quantile`). H/D cannot do this: inside "none" and "near_only" it is
     always below 1.
   - `caa_none`, `caa_near_only`, `caa_both` are long minus short inside that regime. `caa_within` averages the
     "near_only" and "both" differences, weighted by regime size (`--within-pool`); "none" is left out because the
     horizon cannot change the answer there.
   - They are bundled as the comparison methods `caa-none`, `caa-near_only`, `caa-both`, `caa-within`.
   - fit.py prints, and `fit.json` records under `diagnostics.caa_within`, each one's class sizes and cosines, and
     the cosines between the regimes' directions. A regime with fewer than two distinct horizons gets none, and
     bundle.py bundles what the fit has.
   **Trend and probe** are horizon directions with the regime part removed, bundled as `trend` and `probe`.
   - Within each regime the activations and log10 H are demeaned. `trend` is the sum over prompts of activation x log10 H,
     unit-normalized: the covariance (activation pattern) of a horizon decoder (Haufe et al. 2014), the right vector for
     an additive edit. A mean difference like CAA also carries the regime offset.
   - `probe` is the ridge decoder's weight vector for the same target, alpha from `logspace(0, 6, 13)` by 5-fold CV over
     whole horizons (sorted distinct horizons dealt to folds in turn), implemented in numpy by SVD (`steer/horizon_decoders.py`).
   - Both point toward longer horizons. `fit.json` records, under `diagnostics.horizon_decoders`, each one's held-out-horizon
     Spearman, probe's alpha, cosines with CAA, choice and reward, and the prompt counts.
   - They use the regimes of the fitted prompts, so `--no-include-none` leaves "none" out of them too.
3. **Reward-contrast control.** The report names this control but doesn't define it. Here it is the mean of
   prompts with reward multiplier >= 25 minus those with multiplier <= 5 (`--reward-high-min`,
   `--reward-low-max`).
4. **Random controls.** 100 isotropic random unit vectors (`--n-random`, `--seed`), or, with `--draws DIR`, the
   shared draws the evaluation team's `evaluation/shared_draws.py` writes for these activations:

   ```
   python ../evaluation/shared_draws.py make --activations activations/<run> --bank ../prompt_bank/data/t0_s0/prompts.jsonl \
       --model <model id> --layer 38 --site label --out draws/
   python fit.py --activations activations/<run> --bank ../prompt_bank/data/t0_s0 --layer 38 --draws draws/<model>/L38_label
   ```

   - The fit's random baseline is then the same vectors as the evaluation's nulls: 99 each of isotropic,
     covariance-matched and real-difference directions. `fit.json` reports CAA's largest and 95th-percentile |cosine|
     per family, and records the draws' hashes under `random_source`.
   - `fit.py` refuses draws made for another model, layer, site or activations file, or changed since they were
     written (`steer/draws.py`).
   - The draws enter the diagnostics only. A fit with `--draws` has the same methods, bit for bit.
   - They are drawn from all 471 fitted-split prompts, including any that a fit leaves out with
     `--exclude-horizons`.
5. **Curve.**
   - PCA keeps 4 components (`--components`; the report says 3-8).
   - The smoother is Gaussian-kernel local-linear with a 0.3-decade bandwidth (`--bandwidth`; grid points are
     0.185 decade apart).
   - The curve is stored as a 2001-point polyline over the observed range, and t-hat is read from the nearest
     point on it.
   - Local-linear rather than a plain kernel average, because the plain average is biased inwards at both ends
     of the grid.
6. **The curve edit has two readings**, both implemented (`Manifold.steer`, `mode=`):
   - `snap` ("moved to the curve point at t-hat + Delta") replaces the subspace part with the curve point. Its new
     t-hat is exactly t-hat + Delta, up to the polyline's resolution. At Delta = 0 it is **not** the identity: it
     removes the off-curve offset.
   - `displace` adds c(t-hat + Delta) - c(t-hat). Delta = 0 is the identity, but where the curve turns, the new
     t-hat misses t-hat + Delta by about (offset from the curve) × (turning). In `tests/test_manifold.py` that is
     a 0.028-decade median against 0.0017 for snap.

   Both leave the part orthogonal to the subspace unchanged. Choosing between them affects the Delta = 0 control
   and the size of the realized shift.
7. **Replay check.** For each replayed prompt, extract.py records the label margin: the logit of the label the
   model generated minus the best other logit, at the token before it. A faithful replay of a greedy generation
   gives a margin >= 0, except on near-ties, which can land one bf16 step (0.125) below zero. extract.json reports
   the smallest margin. A clearly negative one would mean the replay does not reproduce the generation.

   A sampled baseline run (baseline.py's default with thinking on) need not write its most likely label, so its
   margins can be honestly negative. For such a run the check is that the label is among the run's top-k tokens
   (`replay_label_rank` < top_k), which is necessary but weaker. `replay_consistent` in index.jsonl is the check that
   applies to the run, and extract.json says which one it was.

## Applying an edit inside the model

```python
from steer.edits import LinearEdit, CurveEdit
from steer.hooks import steering, replay_logits

caa = LinearEdit(direction, typical_norm)          # kind "push":   x = strength, a fraction of the typical norm
slide = CurveEdit(manifold, mode="snap")           # kind "target": x = shift along the curve, in decades

with steering(local_chat_model.model, caa, x=0.1, layer=38, positions="generated"):
    replies = local_chat_model.generate(texts, thinking=True, max_new_tokens=8192)

logits = replay_logits(model, prompt_ids, answer_ids, slide, x=0.5, layer=38)   # one teacher-forced pass
```

- **Edits** (`steer/edits.py`) are torch counterparts of the numpy references. An edit returns a displacement
  `delta(h, x, ctx)` at every position.
  - `x` is a number, or a tensor with one value per row.
  - `ctx` (an `EditContext`) carries per-token roles (pad / prompt / generated), positions and per-row facts. CAA and
    the curve ignore it; it is in the signature for edits that need it.
- **The hook** (`steer/hooks.py`) decides where the displacement is applied:
  - `positions="all"`, `"prompt"`, `"generated"`, or a callable on the context;
  - `at_generated(index)` builds a callable for one generated token per row (the label or prelabel token);
  - `at_prompt(indices)` builds one for chosen prompt tokens per row (the horizon's tokens, below);
  - padding is never edited.

  It reads each pass's token positions from the model's forward call (`cache_position`, else `attention_mask`,
  else `position_ids`), so it works inside batched, left-padded `generate()` with the key-value cache, and at
  batch 1, where transformers 5.22.2 passes neither a cache position nor a mask. Unedited positions keep their bits, and a zero edit
  reproduces the unhooked model exactly.
- **`record=[]`** collects each pass's context and the per-token norm of the applied displacement, for audits and for
  matching random controls' sizes token by token.
- **Before steering a new model**, or after a transformers upgrade, run `tools/check_hook.py`. It checks the zero edit,
  the roles and the effect in that model's real generation setup, and exits 1 on failure. `--positions` applies the
  edit at `all` (the bundles' default), `prompt`, `generated`, or one token per row: `label` (the answer-label token,
  which can no longer change the choice) or `prelabel` (the token whose state predicts the label); for the last two
  it also checks that exactly that token was edited (`steer.hooks.at_generated`).
  - **The tests** cover a tiny Qwen3 and Qwen3.5-0.8B.
  - **On Gemma 4 31B, layer 38, thinking on,** the gate has passed twice:
    - locally, with CPU offload, on 2 prompts;
    - on a rented A100 80 GB, on 8 prompts.

    `progress-20261114.md` and `progress-20261115.md` have the outputs.
  - **Not yet run on Gemma:** `check_hook.py --method` with a real bundle.

## The horizon site

`extract_horizon.py` reads the residual stream at the prompt tokens that state the horizon ("7 years"), for methods
that are fitted on those tokens or edit only them:

```
uv run python extract_horizon.py --bank ../prompt_bank/data/t0_s0 --model google/gemma-4-31B-it
```

- **No baseline run is needed.** The horizon is in the prompt, so this is one forward pass per prompt, with no
  generation. On Qwen3.5-2B the 700 prompts at all 25 layers took 32 s (`progress-20261122.md`).
- **Every token of the horizon is kept separately,** with its string. Gemma 4 and Qwen3.5 split numbers into digits:
  "15 minutes" is `'1'`, `'5'`, `' minutes'`. On `t0_s0` a horizon has 2 to 7 tokens, 2,600 in all.
- **All layers by default,** 0 to the number of blocks. Layer 0 is the embeddings. The last layer is the last block's
  output before the final norm, which is not `hidden_states[n]` (`steer.layers.capture_all`).
- **The audit is part of the run.** Every horizon's tokens must decode back to the stated horizon, unit rewrites
  included, or nothing is written.
- **Thinking is part of the model input.** Gemma 4's template puts a system turn before the prompt when thinking is
  on, which shifts the token indices and changes the activations. Extract with the setting of the runs the result
  will be used with (`--thinking`, the default, or `--no-thinking`).
- **Output:** `activations/<model>[-nothink]__<bank>__horizon/`, one float32 file per layer, `index.jsonl` (one row
  per token) and `horizon_sites.jsonl` (one row per prompt). `steer/activations.py` lists the fields.

Editing only the horizon's tokens during generation:

```python
from steer.activations import horizon_sites
from steer.hooks import at_prompt, steering

sites = horizon_sites("activations/gemma-4-31B-it__t0_s0__horizon")
rows = [sites[uid] for uid in uids]                      # per prompt: token_indices, log10_days, near_first, label_style, ...
with steering(lcm.model, edit, x, layer, positions=at_prompt([r["token_indices"] for r in rows]), rows=rows):
    replies = lcm.generate(texts, thinking=True, max_new_tokens=8192)
```

`rows` reaches the edit as `ctx.rows`. A prompt without a horizon has an empty `token_indices` and is not edited.

## Handing a method over: bundles

```
python bundle.py --fit fits/<run>/default/L38_label --activations activations/<run>
    -> methods/<model>/{caa,reward,choice,curve-snap,curve-displace}__<run>__default__L38_label/
```

```python
from steer.methods import load
m = load("methods/gemma-4-31B-it/caa__gemma-4-31B-it__t0_s0__default__L38_label")   # checks every file's SHA-256
with steering(model, m.edit, x=0.1, layer=m.layer, positions=m.positions): ...
null = m.refit("activations/<run>", "../prompt_bank/data/t0_s0", horizons=shuffled)  # same recipe, other labels
```

Each bundle is one method, self-describing (format `steer-method/1`, see `steer/methods.py`). `method.json` carries:
- the kind (push or target) and the edit class;
- the layer and its convention, the site, the default positions;
- the request x: name, unit, sign, and the typical norm or the curve's range;
- what the method was fitted on: model and revision, baseline run, bank and activations hashes, settings,
  include-none, prompts left out per reason;
- how it was trained: closed form, not selected by search;
- the code commits of extraction, fit and bundling;
- the SHA-256 of every other file.

The other files:
- `arrays.npz`: the parameters;
- `fit.json` and `fit_uids.jsonl`: the fit's diagnostics, and the prompts it was fitted on with their classes;
- `search.jsonl`: a snapshot of every fit run on those activations.

`reward` and `choice` are concept controls fitted by the same recipe.

`Method.refit(activations, bank, uids=, horizons=)` reruns the same fit on other prompts or relabelled horizons. It
is how the evaluation team builds label-shuffled nulls and held-out-horizon refits. It refuses verification-grid and
no-horizon prompts, and activations other than the ones fitted from. `tools/check_hook.py --method <bundle>` runs the
hook gate with the bundled method itself.

`methods/` is git-ignored like `activations/` and `fits/`: bundles are published with the upload scripts, not committed.

## Publishing outputs

Outputs are hosted on Hugging Face (agreed with the evaluation team, 2026-11-19). There is one script per
destination, and the GCS one remains available. They take the same paths:

```
python tools/upload_hf.py  ../prompt_bank/runs/<run> activations/<run> fits/<run> methods/<model>     # dry run
python tools/upload_hf.py  ... --upload --repo-id ORG/NAME [--prefix v1] [--public]
python tools/upload_gcs.py ... --upload --dest BUCKET/PREFIX --key KEY.json                           # or $GCS_KEY
```

Both work the same way:
- **Dry run by default:** nothing leaves the machine without `--upload`.
- **Names mirror the repository** (`steering/methods/...`, `prompt_bank/runs/...`), so a download unpacked at a
  checkout's root sits where the code expects it.
- **`SHA256SUMS`** is uploaded too: check a download with `sha256sum -c SHA256SUMS`.
- **A destination that already has files is refused,** never overwritten.
- **Credential-looking files** (`.env*`, `*.pem`, `*.key`, names with credential / secret / token), `.git` contents
  and symbolic links are refused.
- **Uploads are verified afterwards:** rclone's MD5 check plus a file count for GCS; each file's remote hash for
  Hugging Face.

They follow the team's earlier uploads:
- GCS through rclone with a service-account key file, as in `phrasing_study/scripts/run_pipeline.sh`;
- Hugging Face through `huggingface_hub`, as in `multiturn_planning/scripts/upload_hf.py`, with new repos private.

Credentials are the tools' own (`$GCS_KEY` or application-default credentials; `HF_TOKEN` or `hf auth login`).
The scripts never read or print a key.

## Not here yet

- switch points, the gate and calibration (alpha_k);
- the linear-vs-manifold comparison on held-out *horizons*. fit.py's held-out check holds out prompts; every horizon
  stays in training unless `--exclude-horizons` is given;
- repeated cross-validation.

## Files

```
activations/<run>/extract.json         settings, model, software, bank and run identity, counts, array SHA-256s
activations/<run>/index.jsonl          one row per extracted prompt (steer/activations.py lists the fields)
activations/<run>/skipped.jsonl        prompts not extracted, with the reason (no answer, format failure, ...)
activations/<run>/L38_label.npy        float32 [n, d]; also L38_prelabel.npy, L38_prompt_end.npy, L38_think_end.npy
activations/<run>/prelabel_top_logits.npz   token_ids int32 [n, k], logits float32 [n, k], largest first
fits/<run>/search.jsonl               append-only: one line per fit.py run on these activations (written, refused,
                                      error), with settings, code commit and headline diagnostics
fits/<run>/<name>/L38_label/fit.json   settings, inputs, class sizes, prompts left out per reason, diagnostics
fits/<run>/<name>/L38_label/directions.npz   caa, reward, choice, random [n, d], typical_norm, raw norms
fits/<run>/<name>/L38_label/manifold.npz     the curve (Manifold.load)
fits/<run>/<name>/L38_label/per_horizon.jsonl  per stated horizon: count, regimes, choices, CAA projection, t-hat
fits/<run>/<name>/L38_label/fit_uids.jsonl     per fitted prompt: uid, regime, horizon, H/D, reward multiplier,
                                              option order, label style, far label, choice, CAA/reward class, fold
```

`activations/` and `fits/` are git-ignored: both can be regenerated from a baseline run.

## Code map

| file | what it does |
|---|---|
| `bundle.py` | stage 3: package a fit's methods as bundles for the evaluation team |
| `extract.py` | stage 1: replay the baseline's replies, read activations at the label |
| `extract_horizon.py` | read activations at the prompt tokens that state the horizon, all layers; needs no baseline run |
| `fit.py` | stage 2: directions, curve, controls, diagnostics (command-line wrapper around `steer/fitting.py`) |
| `steer/layers.py` | layer convention; `capture`, `capture_all` and `edit` hooks |
| `steer/sites.py` | locate the answer-label token, and the horizon's tokens in the prompt |
| `steer/draws.py` | read and check the evaluation team's shared random draws |
| `steer/activations.py` | on-disk format and loader (checks hashes) |
| `steer/fitting.py` | the fit as a function, `fit(X, records, chosen, FitSettings)`, for refits on other prompts or labels |
| `steer/directions.py` | difference of means, thresholds, random controls, linear edit |
| `steer/manifold.py` | per-horizon means, PCA, local-linear curve, projection, curve edit |
| `steer/stats.py` | Spearman, grouped folds |
| `steer/edits.py` | the edits as torch functions: `LinearEdit`, `CurveEdit` (snap / displace), `EditContext` |
| `steer/hooks.py` | `steering(...)`: apply an edit during generation; `replay_logits(...)`: the replay readout |
| `steer/methods.py` | the bundle format: `write_bundle`, `load`, `Method.refit` |
| `steer/provenance.py` | the code commit outputs record (dirty if steering/ or prompt_bank/ has edits) |
| `tools/check_hook.py` | gate before steering a model: zero edit exact, roles right, strong edit has an effect |
| `tools/upload_hf.py`, `tools/upload_gcs.py` | publish outputs (dry run by default; checksummed, verified, never overwrite) |
| `tools/compare_fits.py` | compare two fit output trees exactly (arrays bit for bit), e.g. before and after a refactor |
| `tools/vast_smoke.sh` | the pipeline on a rented GPU box, one gated stage at a time (code version, setup, weights, tests, hook, horizon site, baseline, extract at all layers and sites, label audit, then shared draws, fit, bundle and method gate at the label and prelabel sites, checksums) |

## Tests

`uv run pytest -s`. The `-s` shows each test's diagnostic numbers. What the tests cover:

- the layer convention: tiny Qwen3, the Gemma 4 module tree, and real Qwen3.5-0.8B on a GPU;
- the label locator, on the real Gemma 4 and Qwen3.5 tokenizers;
- CAA recovering a planted direction;
- the curve recovering a planted helix, and both edit modes' properties;
- bundles: every method loads with the fit's exact parameters, refit on the same prompts reproduces them bit for
  bit, a tampered file is refused, and a within-order horizon shuffle gives a label-shuffled null;
- uploads: what is collected and refused, GCS through real rclone into a local directory (then `sha256sum -c`),
  Hugging Face through a fake hub that records commits (private by default, no overwrite, corruption caught);
- the torch edits against the numpy references (float64 to 1e-10; float32 to one polyline vertex), and zero
  requests giving exactly zero displacements;
- the hook in real `generate()` calls: roles, position masks, padding, exact zero edit, generated-only leaving the
  first token alone, replay. On a tiny Qwen3, and on Qwen3.5-0.8B in bf16 on a GPU;
- `fit.py` end to end on synthetic activations over the real bank (`tests/fixtures.py`), and `tools/compare_fits.py`
  telling identical trees from ones that differ by one ulp; with `--draws`, the methods unchanged and draws for other
  inputs refused;
- the horizon site: its tokens decode back for all 671 horizons on the Gemma 4 and Qwen3.5 tokenizers;
  `extract_horizon.py` on Qwen3.5-0.8B against `hidden_states`; a horizon-only edit in a padded batch.

Tests that need a cached model or a GPU skip without one.
