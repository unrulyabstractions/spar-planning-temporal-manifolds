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
```

1. **Generate.** `baseline.py` runs the model once over the bank and stores every reply's exact generated token
   ids. This is the only generation step.
2. **Extract.** `extract.py` generates nothing. For each prompt it replays the stored ids up to and including the
   answer-label token as one teacher-forced forward pass, and keeps the residual stream at the requested layers.
   It is cheap: minutes even for 31B, against hours for generation. So other layers can be extracted later
   without regenerating.
3. **Fit.** `fit.py` runs on the CPU, takes seconds, and needs no GPU.

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

1. **"None" prompts are included by default.** The report's Methods section says they are "used for extraction";
   `--regimes near_only both` leaves them out. Evidence from Jared's Qwen3 runs:
   - the model's choices on "none" prompts are confident but follow option order and model-specific habits;
   - the 8B's long-minus-short difference of means steered backwards because of the regime mix
     (`origin/jared`, `jared/08_results_steering.md`).

   The team should decide this.
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

## Not here yet

- applying the edits inside the model during generation (the numpy edits in `steer/directions.py` and
  `steer/manifold.py` are the reference; `steer.layers.edit` is the hook they plug into);
- switch points, the gate and calibration (alpha_k);
- the linear-vs-manifold comparison on held-out *horizons* (fit.py's held-out check holds out prompts; every
  horizon stays in training);
- repeated cross-validation.

## Files

```
activations/<run>/extract.json         settings, model, software, bank and run identity, counts, array SHA-256s
activations/<run>/index.jsonl          one row per extracted prompt (steer/activations.py lists the fields)
activations/<run>/skipped.jsonl        prompts not extracted, with the reason (no answer, format failure, ...)
activations/<run>/L38_label.npy        float32 [n, d]; also L38_prelabel.npy
fits/<run>/<name>/L38_label/fit.json   settings, inputs, class sizes, diagnostics
fits/<run>/<name>/L38_label/directions.npz   caa, reward, choice, random [n, d], typical_norm, raw norms
fits/<run>/<name>/L38_label/manifold.npz     the curve (Manifold.load)
fits/<run>/<name>/L38_label/per_horizon.jsonl  per stated horizon: count, regimes, choices, CAA projection, t-hat
```

`activations/` and `fits/` are git-ignored: both can be regenerated from a baseline run.

## Code map

| file | what it does |
|---|---|
| `extract.py` | stage 1: replay the baseline's replies, read activations at the label |
| `fit.py` | stage 2: directions, curve, controls, diagnostics |
| `steer/layers.py` | layer convention; `capture` and `edit` hooks |
| `steer/sites.py` | locate the answer-label token |
| `steer/activations.py` | on-disk format and loader (checks hashes) |
| `steer/directions.py` | difference of means, thresholds, random controls, linear edit |
| `steer/manifold.py` | per-horizon means, PCA, local-linear curve, projection, curve edit |
| `steer/stats.py` | Spearman, grouped folds |

## Tests

`uv run pytest -s`. The `-s` shows each test's diagnostic numbers. What the tests cover:

- the layer convention: tiny Qwen3, the Gemma 4 module tree, and real Qwen3.5-0.8B on a GPU;
- the label locator, on the real Gemma 4 and Qwen3.5 tokenizers;
- CAA recovering a planted direction;
- the curve recovering a planted helix, and both edit modes' properties.

Tests that need a cached model or a GPU skip without one.
