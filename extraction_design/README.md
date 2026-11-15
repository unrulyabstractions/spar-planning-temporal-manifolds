# Extraction pipeline for horizon steering: what it hands over, and how to use it

**Status:** built and tested, but not merged: branch `steering-handover`, not yet on `main`. On Gemma 4 31B it has run
only at smoke scale (16 prompts). No full-run methods exist yet. **Owner:** Extraction (prompt bank and steering
methods: Alan). Drafted with Claude, 2026-11-15.
**Human-readable version:** [`explainer/index.html`](explainer/index.html), with the same content plus figures.
**This file** is the reference version: plain structure, every term defined, every claim sourced. It can be read by
people and by LLM assistants working on the evaluation pipeline.

## How to read this document

- **Answering the evaluation design.** It answers the evaluation team's design proposal
  ([`evaluation_design/README.md`](../evaluation_design/README.md), cited as `[eval §n]`) and uses its numbering:
  items I1–I9, the agreement points of §4.1, null families N0–N6 and confound checks C1–C8.
- **Order.** Section 1 says what extraction produces and what state it is in. Section 2 defines the terms. Sections
  3–7 describe the pipeline, its conventions, the methods, how to apply them and how they are packaged.
- **Answers to the design.** Section 8 maps all of this onto I1–I9 and §4.1. Section 9 lists the fitting choices that
  matter for evaluation.
- **Evidence and practicalities.** Section 10 lists what has been checked, and on what. Section 11 is about running on a
  rented GPU, section 12 about getting the outputs, and section 13 lists open items.
- **Claim sources** are file paths in this repository, or the progress logs that record each command and its output:
  `steering/progress-20261114.md` and `steering/progress-20261115.md`.

---

## 1. Purpose and status

The extraction side turns the T0 prompt bank and one model's activations into **steering methods**, and hands each
one to the evaluation team in a form the evaluation pipeline can run without knowing how it was built. Two packages:

- **`prompt_bank/`:** builds the bank, has a second model check every prompt, and runs the model unsteered once over
  the bank (the **baseline run**). This is the official generator for the experiments.
- **`steering/`:** reads activations from the baseline run and fits the methods:
  - a linear direction (CAA);
  - a manifold curve, used two ways: `snap` and `displace`;
  - two concept controls: reward contrast and choice;
  - random directions.

  It packages each method as a self-describing directory (a **bundle**). It also provides the code that applies a
  method inside the model during generation (the **hook**).

What exists now (branch `steering-handover`):

| Part | State | Evidence |
|---|---|---|
| Bank `t0_s0` (700 prompts), checker, baseline runner | on `main` | `prompt_bank/README.md` |
| Extraction, fitting, diagnostics | on `main` | `steering/README.md` |
| Torch edits, generation-time hook, replay readout | branch | tests; Qwen3.5-0.8B/2B; Gemma 4 31B (§10) |
| Bundles (`steer-method/1`), `Method.refit`, search log | branch | tests; Qwen3.5-2B bundles checked in-model (§10) |
| Hook gate `tools/check_hook.py` | branch | passed on Qwen3.5-0.8B, Qwen3.5-2B, Gemma 4 31B |
| Upload scripts (Hugging Face, GCS) | branch | tests; real Hugging Face upload verified (§10) |
| Rented-GPU run script `tools/vast_smoke.sh` | branch | one A100 run (§11) |
| Full Gemma 4 31B baseline, fit and bundles | **not run** | throughput problem, §11.3 |

## 2. Glossary

Terms already defined in `[eval §2]` (residual stream, layer, site, typical norm, CAA, curve slide, push vs target
method, replay readout, and so on) mean the same here. Additions and precise meanings:

**Data**

- **Bank `t0_s0`:** the T0 prompt bank at seed 0, `prompt_bank/data/t0_s0/prompts.jsonl`. It has two splits:
  - **extraction** (500 prompts): 471 with a horizon, plus 29 no-horizon twins;
  - **verification** (200 prompts): 100 boundary items, each in both option orders.
- **uid:** a prompt's identifier: `x0042` (extraction), `x0042-null` (its no-horizon twin), `v017-far-first`
  (verification).
- **Regime:** which options pay off within the stated horizon:
  - `none`: neither does;
  - `near_only`: the near option does;
  - `both`: both do;
  - `no_horizon`: there is no CONSTRAINT line.

  A payout exactly at the horizon counts as paying off [`prompt_bank/README.md`, "The decision rule"].
- **H/D:** stated horizon divided by the **far** option's delay (`h_over_far_delay`).
- **Grid value:** a horizon's value on the 48-point grid, from 30 s to 500 y, before any unit rewrite ("100 weeks" has
  grid value "23 months").
- **Log-time t:** log10 of the grid value in days.
- **Baseline run:** one unsteered pass of a model over the bank, in `prompt_bank/runs/<model>__<bank>/`:
  - thinking on, greedy decoding;
  - every reply's exact generated token ids are stored (`generated_ids`).

**Extraction**

- **Label site:** the generated token that carries the label character of the model's own "I choose: b)". **Prelabel
  site:** the token before it. Both are extracted [`steering/steer/sites.py`].
- **Replay:** one teacher-forced forward pass over the prompt plus the stored reply, up to and including the label
  token. Extraction reads activations this way and generates nothing.
- **Replay margin:** at the token before the label, the logit of the label the model generated minus the best other
  logit.
  - A faithful replay of greedy generation gives ≥ 0, except on near-ties of one bf16 step (0.125).
  - `extract.json` records the smallest margin.
- **bf16:** the 16-bit floating-point format the model runs in.

**Methods and hand-over**

- **Method:** one way of editing the residual stream, with a request `x`:
  - `caa`, `reward` and `choice` are **push** methods: x is a strength, a fraction of the typical norm;
  - `curve-snap` and `curve-displace` are **target** methods: x is a shift in decades of log-time.
- **Bundle:** one directory per method (§7), format `steer-method/1`.
- **Edit:** the torch object that computes a method's displacement, `edit.delta(h, x, ctx)`.
- **EditContext (`ctx`):** what the hook knows about each token of a forward pass:
  - its role (pad / prompt / generated);
  - its position;
  - whether this pass edits it;
  - optional per-row facts supplied by the caller [`steering/steer/edits.py`].
- **Positions:** which tokens an edit is applied to: `all`, `prompt`, `generated`, or a callable on the context.
  Padding is never edited.
- **Hook gate:** `tools/check_hook.py`, a pass/fail check that the hook behaves in one model's real generation setup.
- **Fit name:** the label of one fit configuration (`default` unless `--name` is given).
- **Search log:** `fits/<activations>/search.jsonl`, one line per `fit.py` run on those activations, whatever its
  outcome.
- **Code commit:** the git commit recorded in every output, with `-dirty` if `steering/` or `prompt_bank/` had
  uncommitted edits [`steering/steer/provenance.py`].

## 3. The pipeline at a glance

```
prompt_bank/                                   steering/
  generate.py   build + check the bank           extract.py   replay replies, read layer-38 activations (minutes)
  baseline.py   run the model once (hours)  ->   fit.py       directions, curve, controls, diagnostics (CPU, seconds)
  review.py     hand review in a browser         bundle.py    one bundle per method
                                                 tools/check_hook.py --method   gate on the GPU
                                                 tools/upload_hf.py | upload_gcs.py   publish, checksummed
```

| Stage | Command (from the package directory) | Writes | Needs | Refuses |
|---|---|---|---|---|
| Baseline | `baseline.py --bank data/t0_s0` | `runs/<model>__<bank>/` run.json, responses.jsonl, summary.json | GPU holding the model in bf16 | resuming with different settings, or a different bank |
| Extract | `extract.py --bank … --run … --layers 38` | `activations/<run>/` extract.json, index.jsonl, skipped.jsonl, `L38_label.npy`, `L38_prelabel.npy` | same GPU, minutes | a run on another bank; replies without `generated_ids`; overwriting |
| Fit | `fit.py --activations … --bank … --layer 38` | `fits/<run>/<name>/L38_label/`; appends `fits/<run>/search.jsonl` | CPU, seconds | overwriting; activations whose hashes don't match |
| Bundle | `bundle.py --fit … --activations …` | `methods/<model>/<method>__<run>__<name>__L38_label/` | CPU | overwriting; activations other than the fit's |
| Gate | `tools/check_hook.py --method <bundle> --bank …` | nothing (exit code 0/1) | GPU | — |
| Publish | `tools/upload_hf.py` / `upload_gcs.py` | remote copy + `SHA256SUMS` | network | a non-empty destination; credential-like files; symlinks |

Every stage records its settings, its inputs' SHA-256 hashes and the code commit [`steering/README.md`, "Steps"].
`activations/`, `fits/`, `methods/` and `runs/` are git-ignored: they are published with the upload scripts, not
committed.

## 4. Conventions

- **Layer L** is the residual stream after decoder block L, counting blocks from 1: `hidden_states[L]`. It is read or
  edited by hooking block L−1 of the model's decoder list. `decoder_layers(model)` finds that list without knowing the
  model class: it is the one `nn.ModuleList` whose length is the number of hidden layers. In Gemma 4 that is
  `…language_model.layers`; code that hard-codes `model.model.layers` (as in Qwen) misses it
  [`steering/steer/layers.py`]. The headline is layer 38 of Gemma 4 31B's 60.
- **Sites.**
  - `label` is the token carrying the label character. It is fit.py's default and the site every bundle so far uses.
  - `prelabel` is the token before it. Its identity depends on the label style, because the tokenizer merges the space
    into letters but not into digits. On Gemma 4 the label token is `' a'` after `':'`, but `'1'` after `' '`
    (x0001 and x0002 in the A100 run, §10).

  So the prelabel site sits on a different token for each label style, which matters for C1.
- **Thinking.** Gemma 4 ends its thinking with `<channel|>`, Qwen3/3.5 with `</think>`. pbank finds the marker in the
  tokenizer and refuses a model family with neither [`prompt_bank/README.md`, "How a model's answer is read"].
  The label site is in the answer, after the marker.
- **Push strength.** An edit adds `x × typical_norm × direction`:
  - directions are unit vectors;
  - `typical_norm` is the mean activation norm at that layer and site, stored in the bundle.

  Every push method at the same x therefore has the same norm.
- **Target shift.** `x` is in decades of log10(horizon in days), measured from each token's own t̂, and `t̂ + x` is
  clamped to the curve's range (stored in the bundle).
- **Sign.** +x moves toward longer horizons for `caa` and both curve methods, toward larger reward multipliers for
  `reward`, and toward "chose far" for `choice`. Each bundle states its own sign in `method.json` → `request.sign`.
- **Default positions:** `all`, following the report, which steers all tokens. The hook also takes `prompt`,
  `generated`, or a callable.
- **Arithmetic.** Edits compute in float32 by default, and the result is cast back to the model's dtype once. A zero
  request gives an exactly zero displacement [`steering/steer/edits.py`].

## 5. The methods

| Method | Kind | Edit | What it is | Request x |
|---|---|---|---|---|
| `caa` | push | `LinearEdit` | mean of H/D ≥ 4 prompts minus mean of H/D ≤ 1/4 prompts, unit-normalised | strength (× typical norm) |
| `curve-snap` | target | `CurveEdit(mode="snap")` | replace the activation's subspace part with the curve point c(t̂ + x) | shift, decades |
| `curve-displace` | target | `CurveEdit(mode="displace")` | add c(t̂ + x) − c(t̂) | shift, decades |
| `reward` | push | `LinearEdit` | concept control: reward multiplier ≥ 25 minus ≤ 5 | strength |
| `choice` | push | `LinearEdit` | concept control: replies that chose far minus those that chose near | strength |

- **Random directions:** the fit also stores 100 isotropic random unit vectors (seed 0) in `directions.npz`.
  These are not bundled.
- **The curve** (`steering/steer/manifold.py`):
  - built from the per-horizon means in their top 4 principal components;
  - smoothed with a Gaussian-kernel local-linear smoother (bandwidth 0.3 decade);
  - stored as a 2001-point polyline over the observed range;
  - t̂ is the nearest polyline point.
- **Both curve edits leave the part of h outside the subspace unchanged.** They differ in what they do inside it
  (figure f2):
  - `snap` lands exactly on the curve, so its new t̂ is t̂ + x up to the polyline's resolution. At x = 0 it is **not**
    the identity: it removes the activation's offset from the curve.
  - `displace` keeps the offset. x = 0 is the identity, but where the curve turns, the realised t̂ misses t̂ + x. In
    `tests/test_manifold.py` the median miss is 0.028 decade, against 0.0017 for snap.
- **Δ = 0 for snap matters in practice.** On Qwen3.5-2B (thinking off, 8 prompts) `curve-snap` at x = 0 changed 7 of
  8 replies; `curve-displace` and `caa` at x = 0 changed none [`steering/progress-20261114.md`]. This is plumbing
  scale, but the Δ = 0 control `[eval §6.3]` is not a formality.

## 6. Applying a method inside the model

```python
from pbank.local_model import LocalChatModel
from steer.methods import load
from steer.hooks import steering, replay_logits

lcm = LocalChatModel("google/gemma-4-31B-it")
m = load("methods/gemma-4-31B-it/caa__gemma-4-31B-it__t0_s0__default__L38_label")    # checks every file's SHA-256

with steering(lcm.model, m.edit, x=0.1, layer=m.layer, positions=m.positions):
    replies = lcm.generate(texts, thinking=True, max_new_tokens=8192)

logits = replay_logits(lcm.model, prompt_ids, answer_ids, m.edit, x=0.1, layer=m.layer)  # one teacher-forced pass
```

**What the hook does** [`steering/steer/hooks.py`]:
- **Per-pass context.** A forward pre-hook on the model reads each pass's `cache_position` and `attention_mask`:
  - the pass that starts at position 0 is the prompt;
  - every later pass is generated tokens;
  - attention-mask zeros are padding.

  This is what makes it work inside batched, left-padded `generate()` with the key-value cache.
- **Edit and mask.** The edit's displacement is computed at every position and applied only where the mask allows,
  with `torch.where`. Unedited positions keep their exact bits.
- **Per-row requests.** `x` may be one number, or one value per row (a different strength per prompt in a batch).
- **Per-row facts.** `rows=[{...}, ...]` passes per-row facts into `ctx.rows`, e.g. uid and stated horizon. CAA and the
  curve ignore them. They are there so that edits that need them fit the same signature: centroid substitution needs
  each prompt's source horizon, and some edits touch only a prompt's horizon tokens.
- **Recording.** `record=[]` collects each pass's context and the L2 norm of the applied displacement per token, so a
  random control can be matched to a method's size token by token (N1, N6).
- **Replay readout.** `replay_logits(...)` is the replay readout of `[eval §2]`: next-token logits after the prompt
  plus the model's own reply up to the token before its label, in one pass with the edit applied.

**What is guaranteed, and how we know** (§10 has the numbers):
- a zero request reproduces the unhooked token ids exactly (null N0);
- a `generated`-only edit leaves the prompt pass, and so the first generated token, bit-identical;
- the torch edits match the numpy references to 4.4e-16 in float64.

**Before steering any model, method or transformers version:** run `tools/check_hook.py`, which exits 1 on failure:
1. zero edit exact;
2. roles and positions correct;
3. a strong edit has an effect;
4. with `--method`, the bundled method at x = 0 reproduces the unhooked ids. curve-snap is exempt and only reported.

## 7. The bundle

```
methods/<model>/<method>__<activations>__<fit name>__L<layer>_<site>/
  method.json      what the method is, how to apply it, what it was fitted on, hashes of the other files
  arrays.npz       the parameters: direction + typical_norm, or the curve (mu, U, t_grid, curve, ...)
  fit.json         the fit's settings, inputs, class sizes, prompts left out per reason, diagnostics
  fit_uids.jsonl   one row per fitted prompt: uid, regime, horizon, grid value, H/D, reward multiplier,
                   near_first, label style, far label, baseline choice, CAA class, reward class, fold
  search.jsonl     every fit.py run on these activations up to bundling (written, refused, error)
```

A real `method.json`, from the 16-prompt Gemma smoke run of §11, abridged:

```json
{
  "format": "steer-method/1",
  "method_id": "caa__gemma-4-31B-it__smoke__default__L38_label",
  "method": "caa", "kind": "push", "edit": {"class": "LinearEdit"},
  "layer": 38,
  "layer_convention": "residual stream after decoder block `layer`, counting blocks from 1 (= hidden_states[layer]); see steering/steer/layers.py",
  "site": "label", "default_positions": "all",
  "request": {"name": "strength",
              "unit": "fraction of typical_norm (the mean activation norm at this layer and site)",
              "sign": "+x moves toward longer horizons (mean of H/D >= caa_long_min minus mean of H/D <= caa_short_max)",
              "typical_norm": 305.83},
  "training": {"procedure": "difference of class means, unit-normalized", "objective": null,
               "selected_by_search": false, "search_log": "search.jsonl lists every fit.py run on these activations up to bundling"},
  "fitted_on": {"model": {"model_id": "google/gemma-4-31B-it", "model_revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
                          "dtype": "torch.bfloat16", "torch": "2.11.0+cu128", "transformers": "5.22.2", "...": "..."},
                "thinking": true, "baseline_run": "gemma-4-31B-it__smoke", "bank": "t0_s0",
                "bank_prompts_sha256": "75901a42…", "activations_array_sha256": "2f9c796a…",
                "n_prompts": 13, "regimes": {"both": 4, "none": 8, "near_only": 1},
                "excluded": {"not_extraction_split": 0, "no_horizon": 0, "none_regime": 0, "excluded_horizon": 0},
                "include_none": true},
  "fit_settings": {"caa_long_min": 4.0, "caa_short_max": 0.25, "reward_high_min": 25, "reward_low_max": 5,
                   "n_random": 100, "components": 4, "bandwidth": 0.3, "folds": 5, "seed": 0},
  "code_commit": {"extract": "unknown", "fit": "unknown", "bundle": "6b640738…"},
  "files": {"arrays.npz": "c3fe5dd8…", "fit.json": "4d254aa1…", "fit_uids.jsonl": "248baf66…", "search.jsonl": "be29c819…"}
}
```

The curve bundles' `request` reads:

```json
{"name": "shift", "unit": "decades of log10(horizon in days), from each token's t-hat", "sign": "+x moves toward longer horizons",
 "t_range_log10_days": [-3.46, 5.26], "clamped": "t-hat + x is clamped to t_range"}
```

The two `"unknown"` commits are a real gap, described in §11.2. The code was shipped to the rented machine without
its `.git` directory, so `extract.py` and `fit.py` could not name their commit.

**Loading and refitting** [`steering/steer/methods.py`]:

```python
m = load(path)                  # refuses a bundle whose files don't match their recorded SHA-256
m.edit, m.layer, m.site, m.positions, m.kind, m.structure      # structure: the direction, or the Manifold
m.fit_uids()                    # the rows of fit_uids.jsonl
m.refit(activations_dir, bank_dir, uids=None, horizons=None)   # same recipe and settings, other prompts or labels
```

`refit` reruns the same closed-form fit. It refuses:
- activations other than the ones the method was fitted from (by hash), and a different bank;
- verification-grid prompts and no-horizon prompts.

What it is for:
- **N4, label-shuffled null:** `horizons={uid: grid value}` relabels prompts, for example a shuffle within option
  order. Refitting on a prompt's own horizon reproduces the parameters bit for bit (tested).
- **C8, held-out horizons:** `uids=` without the held-out horizons' prompts, or fit with `--exclude-horizons`.
- **C1, within-order refits:** `uids=` restricted to one option order or label style. `fit_uids.jsonl` has both
  columns.

## 8. Answers to the evaluation design

### 8.1 Items I1–I9 `[eval §3.1]`

| Item | Status | Where |
|---|---|---|
| I1 the edit during generation | done | `steer.hooks.steering(model, edit, x, layer, positions)`; `replay_logits` |
| I2 conventions | done, machine-readable | `method.json`: layer + convention, site, positions, request name/unit/sign, typical norm or t range |
| I3 fitting data | done | `fit_uids.jsonl` per bundle; `fit.json` counts what was left out and why |
| I4 refit as a function | done | `Method.refit(activations, bank, uids=, horizons=)` |
| I5 activations | exist per run | `activations/<run>/L38_{label,prelabel}.npy`, published with the upload scripts |
| I6 structure | done | `Method.structure`: the unit direction, or the `Manifold` (`arrays.npz`) |
| I7 objective | not applicable | all methods are closed form; `method.json` → `training.selected_by_search: false` |
| I8 search history | done | `search.jsonl`, append-only, written by `fit.py` itself, including refused and failed runs |
| I9 diagnostics | done | `fit.json`: class composition, CAA–choice and CAA–reward cosines, random baseline, held-out check |

### 8.2 The §4.1 agreement points (extraction's answers)

1. **Hand-over format.** One bundle per method (§7), loaded by `steer.methods.load`.
   - The interface is the one sketched in `[eval §4.1]`, with one change: the request comes before the context,
     `edit.delta(h, x, ctx)`, because ctx is optional.
   - The controls `reward` and `choice` are bundled the same way, as the concept controls of N5.
2. **Generation-time hook.** Extraction built it (§6), and it is gated per model (§10).
3. **Data splits.**
   - `fit.py` and `refit` fit only on the 500-prompt extraction split. Verification-grid prompts are dropped even if
     they were extracted, so the evaluation team can extract the grid for its own nulls without risk of leakage.
   - `--exclude-horizons` holds out horizons. **The list of held-out horizons is the evaluation team's call (C8).**
   - Layer 38 is fixed by the spec.
   - k = 4 components and the 0.3-decade bandwidth come from the report's range (3–8) and the grid spacing (0.185
     decade), not from tuning. They stay frozen for the Gemma fit.
4. **Operating point.** Extraction hands over a strength unit, not a frozen strength. Calibrating α_k on the evaluation
   team's own split keeps the strength choice inside its nulls (P4).
5. **Search disclosure.** `search.jsonl` (I8).
6. **Shared random draws.** Not built yet.
   - Each fit stores 100 isotropic unit vectors (seed 0).
   - Seeded isotropic (N1) and covariance-matched (N2) draws per (model, layer, site), with hashes, can be published
     from our activations once the evaluation team picks the families and K `[eval §6]`.
7. **Versioning.**
   - Every output records the SHA-256 of its inputs and the code commit; bundles record a hash for every file.
   - A gap was found on the rented GPU: with the code shipped without `.git`, the commit is recorded as `"unknown"`
     (§11.2).

## 9. Fitting choices that matter for evaluation

All are command-line options of `fit.py`, recorded in `fit.json` and `method.json` [`steering/README.md`,
"Choices made where the report is silent or ambiguous"].

1. **"None" prompts are included by default**, following the written spec ("used for extraction").
   - One recorded switch, `--include-none` / `--no-include-none`, applies to CAA, the controls and the curve together.
     Switching later is a refit; activations of "none" prompts are always extracted.
   - Leaving them out has real costs on `t0_s0`:
     - the curve then starts at 10 days instead of 30 s, because all 24 grid horizons below 10 days are pure "none"
       (figure f1, left);
     - CAA's short class shrinks from 350 prompts to 71.
2. **CAA's classes are not balanced by regime.** They are set by H/D, with D the far delay:
   - the long class (H/D ≥ 4) is 57 `both` prompts, so far is correct in all of them;
   - the short class (H/D ≤ 1/4) is 279 `none` + 71 `near_only`.

   CAA therefore partly encodes "which answer is correct". `fit.json` reports its cosine with the `choice` direction.
   On a Qwen3.5-2B smoke run that cosine was 0.862 with "none" kept and 0.904 without. This is the C2 confound in
   numbers. A within-regime contrast is not implemented, pending a team decision.
3. **Concept controls.** `reward` (multiplier ≥ 25 minus ≤ 5) and `choice` (chose far minus chose near), built by the
   same recipe. The report names a reward-contrast control but doesn't define it; this definition is ours.
4. **The prelabel token depends on label style** (§4). This matters for any C1 analysis at that site.
5. **`fit.py` has no minimum class size.** On the 16-prompt smoke run it wrote a fit whose CAA long class held one
   prompt. Whether it should refuse below a minimum is an open decision (§13). Until then, check `fit.json`'s class
   sizes.

## 10. What has been checked, and on what

| Check | Model / data | Result | Source |
|---|---|---|---|
| Test suites | CPU + local GPU | steering 81 passed; prompt_bank 64 passed | `progress-20261114.md` |
| Layer convention | tiny Qwen3; Gemma 4 module tree; Qwen3.5-0.8B weights | exact | `tests/test_layers.py` |
| Torch edits vs numpy references | synthetic | max diff 4.44e-16 (float64) | `progress-20261114.md` |
| Hook in real `generate()` | Qwen3.5-0.8B bf16, 6 prompts | zero edit identical 6/6; strength 4 changed 6/6 | `progress-20261114.md` |
| Hook gate, each bundled method | Qwen3.5-2B, thinking off, 8 prompts | checks 1–2 pass; snap at 0 changed 7/8 (reported); caa / displace / reward at 0 changed 0/8 | `progress-20261114.md` |
| Refit on own prompts | synthetic activations over the real bank, 471 prompts | caa and curve-snap parameters identical; a within-order horizon shuffle gives cosine +0.442 with the real CAA | `tests/test_methods.py`; `progress-20261114.md` |
| Hook gate | Gemma 4 31B, local, CPU offload, 2 prompts | zero edit 2/2 identical; roles ok; strong edit changed 2/2 | `progress-20261114.md` |
| Hook gate | Gemma 4 31B, A100 80 GB, 8 prompts | zero edit 8/8 identical; roles ok (left padding [2, 4, 0, 3, 4, 8, 4, 6]); strong edit changed 8/8 | `progress-20261115.md` |
| Extraction replay | Gemma 4 31B, A100, 13 answered prompts | 0 replay mismatches; smallest label margin +11.56 | `progress-20261115.md` |
| Label-site hand audit | same 13 | all 13 are the label right after `<channel|>I choose:`; a shifted-index negative control fails 13/13 | `progress-20261115.md` |
| Hugging Face upload | the A100 outputs, 14 files | created private; 14/14 remote hashes verified | `progress-20261115.md` |

**Not yet checked on Gemma:**
- `check_hook.py --method` with a real bundle;
- any full-size fit;
- any steered generation beyond the gate's strong-edit check.

## 11. Running on a rented GPU

### 11.1 The run script

`steering/tools/vast_smoke.sh STAGE …` (from the repository root, on the box) runs the pipeline one stage at a time
and stops at the first failure:

```
crlf  setup  weights  tests  hook  baseline  extract  audit  fit  bundle  method  sums
```

- **Settings** come from the environment: `MODEL` (default Gemma 4 31B-it), `LAYER` (38), `N` (64 prompts), `BATCH`
  (8), `MAX_NEW` (4096), `RUN` (smoke).
- **The `audit` stage** prints every label site with its decoded neighbours and fails if any decoded token differs
  from the parsed choice.
- **The HF token** reaches the box only through ssh's stdin, never on disk or on a command line:

  ```
  printf '%s\n' "$TOKEN" | ssh BOX 'read -r HF_TOKEN; export HF_TOKEN; …'
  ```

### 11.2 Gotchas found on the first run (A100 SXM4 80 GB, vast.ai, 2026-11-15)

1. **Python extras.** `uv sync --frozen` alone installs no extras: prompt_bank had no torch, and neither project had
   pytest. Each README's command is needed: `--all-extras` for prompt_bank, `--extra dev` for steering. The script now
   uses them.
2. **Uncached test models.** Tests that need a cached model **skip** on a fresh machine, silently reducing coverage.
   The `weights` stage now also downloads Qwen3.5-0.8B and the Qwen3.5-2B/27B tokenizers.
3. **Batch size and memory.** Batch 16 runs out of memory with Gemma 4 31B at 4,096 new tokens on 80 GB; batch 8
   works.
   - About 21 GiB is free after the weights.
   - The key-value cache needs about 1.1 GB per sequence at about 3k tokens.
   - On top of that, SDPA's `repeat_kv` makes temporary copies.
4. **Settings are locked once a run starts.** A run directory remembers its settings. After a failed first batch,
   retrying with a different batch size is refused until the empty directory is removed. That is deliberate: greedy
   outputs depend on padding, so batch size is part of a run's identity.
5. **No `/workspace`.** The image `vastai/pytorch:cuda-12.8.1-auto` has no `/workspace`, which the team justfile
   assumes.
6. **Shipping without `.git` loses provenance.** `just push`-style rsync leaves out dotfiles, including `.git`, so
   every output records its code commit as `"unknown"`. **Not fixed yet.** Options: ship `.git`, or pass the commit in
   an environment variable that `provenance.py` reads.
7. **Bandwidth is billed by some hosts.** The host billed $40/TB in both directions, so the 64 GB download (weights
   plus packages) cost $2.50, more than the GPU time. Filter offers on download cost.
8. **Destroy, don't stop.** A stopped instance keeps billing its disk: $0.28/h for 200 GB.

### 11.3 Throughput: the open problem

Measured on the A100 (batch 8, thinking on, cap 4,096 new tokens):

```
[8/64]  x0001..x0008: 7/8 parsed, max 4096 tokens, 75.6 s/prompt
[16/64] x0009..x0016: 6/8 parsed, max 4096 tokens, 74.7 s/prompt
```

- **Speed.** Each batch took about 605 s for 4,096 decoding steps, about 148 ms per step.
- **Reply lengths.** The 16 replies averaged 1,661 generated tokens. 3 of the 16 hit the cap without answering, and
  every batch had one, so every batch cost the full cap.
- **Projection.** At that rate the 700-prompt baseline is about 88 batches × 10 min ≈ **15 hours** at cap 4,096. At the
  spec's 8,192 cap, batches with a runaway reply cost up to twice as much.
- **Earlier estimate.** `prompt_bank/README.md` quotes a 2026-10-17 pilot: about 13 s per prompt at batch 4, about
  2.5 GPU-hours for 700 prompts. That has not been reconciled with this measurement, and the README needs updating.
- **Not diagnosed.** 148 ms per step is well above the memory-bandwidth floor (about 31 ms: 62 GB of weights read per
  step at about 2 TB/s).

This is the main thing standing between the current state and a full Gemma run. It also sets the cost of every
generation readout the evaluation pipeline plans `[eval §9]`.

## 12. Getting the outputs

```
python tools/upload_hf.py  ../prompt_bank/runs/<run> activations/<run> fits/<run> methods/<model>   # dry run
python tools/upload_hf.py  ... --upload --repo-id ORG/NAME [--prefix v1]        # new repos are private
python tools/upload_gcs.py ... --upload --dest BUCKET/PREFIX --key KEY.json
```

- **Names mirror the repository**, so a download unpacked at a checkout's root sits where the code expects it.
- **Verifying a download:**

  ```
  cd <download>/<prefix> && sha256sum -c SHA256SUMS
  ```

  `steer.methods.load` checks each bundle's own hashes again.
- **Refusals:** both scripts refuse a destination that already has files. Nothing is ever overwritten.
- **Hosting is not decided.** The Hugging Face path has been exercised for real, into a private repo under Alan's
  personal account. The GCS path has been tested only against a local directory.

## 13. Open items

**Needed from the evaluation team:**
- the held-out horizon list for C8;
- the null families and K to precompute as shared draws (§8.2 item 6);
- whether `EditContext` covers the comparison methods' needs, before anything depends on it;
- the preferred host: Hugging Face or GCS.

**Decisions on the extraction side:**
- a minimum class size in `fit.py` (§9.5);
- a within-regime CAA contrast (§9.2);
- recording the code commit when shipping without `.git` (§11.2.6);
- the throughput problem (§11.3);
- stale lines in the READMEs:
  - `prompt_bank/README.md`'s 2.5-hour estimate;
  - `steering/README.md`'s "Gemma 4 31B has not been run yet".

**Not built:**
- shared random-draw files;
- an answer-letter control direction;
- the switch-point, gate and α_k calibration code (roadmap step 2's second half).

## 14. Sources

- `prompt_bank/README.md`: the bank, the decision rule, the baseline runner, the answer parser.
- `steering/README.md`: the steps, conventions, fitting choices, hook, bundles, uploads.
- `steering/progress-20261114.md`: Steps 1–6, the Qwen3.5 checks, the local Gemma smoke test.
- `steering/progress-20261115.md`: the vast.ai run, its gotchas and its numbers.
- Code: `steering/steer/{edits,hooks,methods,manifold,fitting,layers,sites}.py`, `steering/tools/check_hook.py`,
  `steering/tools/vast_smoke.sh`, `steering/tools/upload_{hf,gcs,common}.py`.
- Figures: `explainer/make_figs.py`.
  - f1 is computed from the committed bank, and asserts the counts quoted in §9.
  - f2 is an illustration.
- The evaluation design: `evaluation_design/README.md`.
