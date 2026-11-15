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
    --run ../prompt_bank/runs/gemma-4-31B-it__t0_s0 --layers 38       # 2. activations at the answer label
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

  Both are extracted.
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
2. **The CAA classes are not balanced by regime.** With D the far delay, H/D >= 4 holds only "both" prompts (the
   ones where "far" is the answer). H/D <= 1/4 holds "none" and "near_only" prompts. On the t0_s0 bank that is
   57 against 279 + 71.

   fit.py reports this composition. It also reports the cosine between CAA and a *choice* direction (replies
   that chose far, minus those that chose near). That measures how much of CAA is "which option was chosen"
   rather than "how long the horizon is". A within-regime contrast is **not implemented**, pending a team
   decision.
3. **Reward-contrast control.** The report names this control but doesn't define it. Here it is the mean of
   prompts with reward multiplier >= 25 minus those with multiplier <= 5 (`--reward-high-min`,
   `--reward-low-max`).
4. **Random controls.** 100 isotropic random unit vectors (`--n-random`, `--seed`).
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
  - padding is never edited.

  It reads each pass's `cache_position` and `attention_mask` from the model's forward call, so it works inside
  batched, left-padded `generate()` with the key-value cache. Unedited positions keep their bits, and a zero edit
  reproduces the unhooked model exactly.
- **`record=[]`** collects each pass's context and the per-token norm of the applied displacement, for audits and for
  matching random controls' sizes token by token.
- **Before steering a new model**, or after a transformers upgrade, run `tools/check_hook.py`. It checks the zero edit,
  the roles and the effect in that model's real generation setup, and exits 1 on failure. The tests cover only a tiny
  Qwen3 and Qwen3.5-0.8B; Gemma 4 31B has not been run yet.

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
activations/<run>/L38_label.npy        float32 [n, d]; also L38_prelabel.npy
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
| `fit.py` | stage 2: directions, curve, controls, diagnostics (command-line wrapper around `steer/fitting.py`) |
| `steer/layers.py` | layer convention; `capture` and `edit` hooks |
| `steer/sites.py` | locate the answer-label token |
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
| `tools/compare_fits.py` | compare two fit output trees exactly (arrays bit for bit), e.g. before and after a refactor |

## Tests

`uv run pytest -s`. The `-s` shows each test's diagnostic numbers. What the tests cover:

- the layer convention: tiny Qwen3, the Gemma 4 module tree, and real Qwen3.5-0.8B on a GPU;
- the label locator, on the real Gemma 4 and Qwen3.5 tokenizers;
- CAA recovering a planted direction;
- the curve recovering a planted helix, and both edit modes' properties;
- bundles: every method loads with the fit's exact parameters, refit on the same prompts reproduces them bit for
  bit, a tampered file is refused, and a within-order horizon shuffle gives a label-shuffled null;
- the torch edits against the numpy references (float64 to 1e-10; float32 to one polyline vertex), and zero
  requests giving exactly zero displacements;
- the hook in real `generate()` calls: roles, position masks, padding, exact zero edit, generated-only leaving the
  first token alone, replay. On a tiny Qwen3, and on Qwen3.5-0.8B in bf16 on a GPU;
- `fit.py` end to end on synthetic activations over the real bank (`tests/fixtures.py`), and `tools/compare_fits.py`
  telling identical trees from ones that differ by one ulp.

Tests that need a cached model or a GPU skip without one.
