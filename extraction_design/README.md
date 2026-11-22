# Extraction pipeline for horizon steering: what it hands over, and how to use it

**Status:** built, tested and on `main`, including the hook fix and single-token edits of 2026-11-19 (§6). Two
later additions are also on `main`: the horizon site (§6) and `fit.py --draws` (§8.2 item 6), the within-regime CAA
methods (§5, §9.2) and seeded sampling (§11.3). On Gemma 4 31B the
pipeline has run only at smoke scale (16 prompts, most recently on 2026-11-22 with sampling), plus the horizon
extraction for all 700 prompts. No full-run methods exist yet. A complete practice hand-over from a
small model is published (§12). **Owner:** Extraction (prompt bank and steering methods: Alan). Drafted with Claude,
2026-11-15; updated 2026-11-22.
**Human-readable version:** [`explainer/index.html`](explainer/index.html), with figures. It was written on
2026-11-15. Statements in it that had become false were corrected on 2026-11-22, but what was added here after
2026-11-15 is only in this file.
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
  `steering/progress-20261114.md`, `-20261115.md`, `-20261117.md` and `-20261119.md`.
- **Evaluation-side code** is cited as `evaluation/…`. It is on `main`.

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

What exists now:

| Part | State | Evidence |
|---|---|---|
| Bank `t0_s0` (700 prompts), checker, baseline runner | on `main` | `prompt_bank/README.md` |
| Extraction, fitting, diagnostics | on `main` | `steering/README.md` |
| Torch edits, generation-time hook, replay readout | on `main` | tests; Qwen3.5-0.8B/2B; Gemma 4 31B (§10) |
| Bundles (`steer-method/1`), `Method.refit`, search log | on `main` | tests; Qwen3.5-2B bundles checked in-model (§10) |
| Hook gate `tools/check_hook.py` | on `main` | passed on Qwen3.5-0.8B, Qwen3.5-2B, Gemma 4 31B |
| Upload scripts (Hugging Face, GCS) | on `main` | tests; two real Hugging Face uploads verified (§10); the one real GCS upload was refused by the account (§12) |
| Rented-GPU run script `tools/vast_smoke.sh` | on `main` | one A100 run (§11) |
| Hook fix for generation without padding; single-token edits (`at_generated`); gate option `--positions` | on `main` | tests; Qwen3.5-2B (§6, §10) |
| Practice hand-over: Qwen3.5-2B, thinking off, all 700 prompts | published on Hugging Face (§12) | `steering/progress-20261117.md` |
| Shared random draws | built by the evaluation team, on `main`; `fit.py --draws` reads them | `evaluation/shared_draws.py` (§8.2 item 6) |
| Horizon site: activations at the tokens that state the horizon, all layers; `at_prompt` | on `main` | tests; Qwen3.5-0.8B/2B (§6, §10) |
| Within-regime CAA: `caa-none`, `caa-near_only`, `caa-both`, `caa-within` | on `main` | tests; Qwen3.5-2B (§5, §9.2) |
| Full Gemma 4 31B baseline, fit and bundles | **not run** | throughput and decoding, §11.3 |

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
  - thinking on unless `--no-thinking` is given;
  - decoding: seeded sampling when thinking is on, greedy (always the most likely next token) when it is off;
    `--decoding` overrides (§11.3). Every run before 2026-11-22 was greedy;
  - every reply's exact generated token ids are stored (`generated_ids`).

**Extraction**

- **Label site:** the generated token that carries the label character of the model's own "I choose: b)". **Prelabel
  site:** the token before it. Both are extracted [`steering/steer/sites.py`].
- **Replay:** one teacher-forced forward pass over the prompt plus the stored reply, up to and including the label
  token. Extraction reads activations this way and generates nothing.
- **Replay margin:** at the token before the label, the logit of the label the model generated minus the best other
  logit.
  - A faithful replay of greedy generation gives ≥ 0, except on near-ties of one bf16 step (0.125).
  - A sampled run can write a label that is not the most likely token, so its margins can be negative. Its check is
    the label's rank instead: it must be among the run's top-k tokens.
  - `extract.json` records the smallest margin, the largest rank, and which check applied.
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
  `steer.hooks.at_generated(index)` builds a callable that edits one token per row, for example only the label token.
  Padding is never edited.
- **Hook gate:** `tools/check_hook.py`, a pass/fail check that the hook behaves in one model's real generation setup.
- **Fit name:** the label of one fit configuration (`default` unless `--name` is given).
- **Search log:** `fits/<activations>/search.jsonl`, one line per `fit.py` run on those activations, whatever its
  outcome.
- **Code commit:** the git commit recorded in every output, with `-dirty` if tracked files under `steering/` or
  `prompt_bank/` had uncommitted changes [`steering/steer/provenance.py`].

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
  - `label` is the token carrying the label character. It is fit.py's default and the site of every published bundle.
    Its state is computed after the label was written, so an edit at this token alone cannot change the choice (§6).
  - `prelabel` is the token before it, the one whose state predicts the label. `fit.py --site prelabel` fits there
    (§9.6). Its identity depends on the label style, because the tokenizer merges the space
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
  `generated`, or a callable. **This default is under review.**
  - The methods are fitted at one site, but `all` applies them to every prompt and reply token, and those tokens'
    activations are unlike the fitted site's. For the curve methods, t̂ is then read at tokens the curve was never
    fitted on (raised by the evaluation team, 2026-11-18).
  - For the curve methods `all` was an oversight, not a decision.
  - No new default is chosen yet. §6 has the first single-token results.
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
| `caa-none`, `caa-near_only`, `caa-both` | push | `LinearEdit` | comparison method: CAA inside one regime (§9.2) | strength |
| `caa-within` | push | `LinearEdit` | comparison method: the `near_only` and `both` within-regime differences, pooled (§9.2) | strength |

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
  8 replies; `curve-displace` and `caa` at x = 0 changed none [`steering/progress-20261114.md`]. Edited at the label
  token only, snap at x = 0 still changed 4 of 8 [`steering/progress-20261119.md`], so editing every token explains
  only part of the 7. This is plumbing scale, but the Δ = 0 control `[eval §6.3]` is not a formality.

## 6. Applying a method inside the model

```python
from pbank.local_model import LocalChatModel
from steer.methods import load
from steer.hooks import steering, replay_logits

lcm = LocalChatModel("google/gemma-4-31B-it")
m = load("methods/gemma-4-31B-it/caa__gemma-4-31B-it__t0_s0__default__L38_label")    # checks every file's SHA-256

with steering(lcm.model, m.edit, x=0.1, layer=m.layer, positions=m.positions):
    replies = lcm.generate(texts, thinking=True, max_new_tokens=4096)   # greedy; pass decoding= to sample (§11.3)

logits = replay_logits(lcm.model, prompt_ids, answer_ids, m.edit, x=0.1, layer=m.layer)  # one teacher-forced pass
```

**What the hook does** [`steering/steer/hooks.py`]:
- **Per-pass context.** A forward pre-hook on the model works out the position of every token in each pass. It uses
  the first of `cache_position`, `attention_mask` and `position_ids` that `generate()` passes:
  - the pass that starts at position 0 is the prompt;
  - every later pass is generated tokens;
  - attention-mask zeros are padding.

  This is what makes it work inside batched, left-padded `generate()` with the key-value cache, and at batch 1.
- **Bug fixed on 2026-11-19.**
  - transformers 5.22.2 passes no `cache_position`, and at batch 1 no attention mask either. The hook then took every
    generated token for a prompt token. On Qwen3.5-2B at batch 1, `positions="generated"` edited nothing.
  - Only generation without padding (batch 1, or a batch of equal-length prompts) with `prompt`, `generated` or a
    callable was affected. `positions="all"` and replay were not, so no result reported so far changes.
  - The fix reads `position_ids` when there is no attention mask [`steering/progress-20261119.md`].
- **Single-token edits.** `at_generated(index)` returns a `positions` callable that edits one token per row:
  generated token `index[i]` of row i, with −1 meaning the last prompt token. With the label's index from
  `steer.sites.locate_label` it edits the label token; one less gives the prelabel token. It works in generation and in
  replay.
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

`--positions {all,prompt,generated,label,prelabel}` chooses where checks 3 and 4 apply the edit. For
`label` and `prelabel` the gate also checks that exactly one token per row was edited and that each reply matches the
unhooked one up to that token. Check 3 then reports changed choices beside changed replies.

**Which token to edit: first results.** Qwen3.5-2B, thinking off, layer 15, 8 prompts, strength 0.5 for the push
methods and a 2-decade shift for the curve methods [`steering/progress-20261119.md`]:

| Method | label-site bundle, label token only: replies / choices changed | prelabel-site bundle, prelabel token only: replies / choices changed | label-site bundle, all tokens: replies changed |
|---|---|---|---|
| `caa` | 2/8, 0/8 | 7/8, 5/8 | 8/8 |
| `reward` | 2/8, 0/8 | 1/8, 1/8 | 8/8 |
| `choice` | 3/8, 0/8 | 5/8, 4/8 | 8/8 |
| `curve-snap` | 5/8, 0/8 | 4/8, 2/8 | 8/8 |
| `curve-displace` | 3/8, 0/8 | 2/8, 1/8 | 7/8 |

- **An edit at the label token alone never changes the choice.** The label is already written when that token's state
  is computed; only the text after it can change.
- **An edit at the prelabel token can.** That is the candidate for a single-token default, with a prelabel fit.
- **This is a plumbing check, not an effect size:** 8 prompts, one strength, a model that barely represents the
  horizon (§12).

**The horizon site** (requested by the evaluation team, for comparison methods that are
fitted on, or edit only, the tokens that state the horizon):

- **`extract_horizon.py`** reads the residual stream at each token of the stated horizon, separately, at every layer.
  - It needs no baseline run: one forward pass per prompt, no generation.
  - Numbers are split into digits, so "15 minutes" is `'1'`, `'5'`, `' minutes'`. On `t0_s0` a horizon has 2 to 7
    tokens on both Gemma 4 and Qwen3.5, 2,600 tokens over the 671 prompts that state one.
  - The run audits itself: every horizon's tokens must decode back to the stated horizon, or nothing is written.
- **Output** (`steering/steer/activations.py` has the fields):
  - `L<layer>_horizon.npy`, one float32 file per layer, one row per horizon token;
  - `index.jsonl`: per row, the uid, which token of the horizon, its index in the model input, its string;
  - `horizon_sites.jsonl`: per prompt, the horizon's character span, token indices and strings, plus option order,
    label style and log-time. All 700 prompts have a row; the 29 without a horizon have empty lists.
- **Layers** 0 to the number of blocks. Layer 0 is the embeddings. The last layer is the last block's output before
  the final norm, which is not `hidden_states[n]`.
- **Thinking changes the model input on Gemma 4** (a system turn is added before the prompt), so the token indices
  and activations depend on it. The setting is recorded; extract with the one the runs will use.
- **Editing only those tokens:** pass a prompt's rows of `horizon_sites.jsonl` as the hook's `rows`, and their
  `token_indices` to `steer.hooks.at_prompt`, which accounts for left padding in a batch
  [`steering/README.md`, "The horizon site"].
- **Run on Gemma 4 31B on 2026-11-22** (thinking on): 700 prompts, 671 with a horizon, 2,600 horizon tokens, 61
  layers, no decode mismatch, under two minutes on an A100 with the model load. The output is 3.2 GB (2,600 tokens × 61
  layers × a hidden size of 5,376 × 4 bytes). It is not uploaded yet.

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

The two `"unknown"` commits were a real gap, described in §11.2 and since closed. The code was shipped to the rented
machine without its `.git` directory, so `extract.py` and `fit.py` could not name their commit.

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
- **Strength calibration:** `uids=` without one fold (`fit_uids.jsonl` has a `fold` column); the strength is then
  chosen on the held-out fold. The evaluation runner does this [`evaluation/replay_eval.py`].
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
     Its first proposal (2026-11-18): 5 weeks, 15 months, 16 years and 210 years (grid points 28, 34, 40 and 46 of
     48). That takes 39 of the 471 fitted prompts out of the C8 refit, 12 of them from CAA's 57 long-class prompts.
     The list is not final.
   - Layer 38 is fixed by the spec.
   - k = 4 components and the 0.3-decade bandwidth come from the report's range (3–8) and the grid spacing (0.185
     decade), not from tuning. They stay frozen for the Gemma fit.
4. **Operating point.** Extraction hands over a strength unit, not a frozen strength. Calibrating α_k on the evaluation
   team's own split keeps the strength choice inside its nulls (P4).
   - The bank has no separate split for this. The evaluation runner holds out one fold of the extraction split, refits
     without it, and calibrates on that fold (§7).
5. **Search disclosure.** `search.jsonl` (I8).
6. **Shared random draws.** Built by the evaluation team: `evaluation/shared_draws.py`.
   - **Input:** what `extract.py` writes (`L<layer>_<site>.npy`, and `index.jsonl` with each row's `uid`), plus the
     bank.
   - **Output,** per (model, layer, site): three families of K = 99 directions each, from the extraction-split prompts
     that have a horizon, with a manifest of seeds and input hashes:
     - N1 isotropic;
     - N2 covariance-matched;
     - N2′ real differences (the activation of one prompt minus another's).

     `shared_draws.py verify` regenerates a file and compares.
   - **Plan:** whoever extracts the Gemma activations runs it right after `extract.py` and publishes the draws beside
     them.
   - **`fit.py --draws DIR`** uses these files as the fit's random baseline, as
     the evaluation team proposed.
     - It refuses draws made for another model, layer, site or activations file.
     - The methods do not change, bit for bit; only the diagnostic does. `fit.json` records the draws' hashes.
     - Without `--draws`, a fit still stores its own 100 isotropic unit vectors (seed 0).
   - **What the families show.** On the Qwen3.5-2B practice activations (layer 15, label site), CAA's largest
     |cosine| with 99 random directions is 0.044 for isotropic, 0.879 for covariance-matched and 0.716 for real
     differences [`steering/progress-20261122.md`]. Against isotropic directions CAA's cosine of 0.862 with the
     `choice` direction looks far from chance; against directions with the activations' own covariance it does not.
     The isotropic baseline alone overstates how special a direction is.
   - **File names:** `shared_draws.py` reads `L<layer>_<site>.npy`, and `extract.py` writes the layer with two
     digits. The names agree for layers 10 and above.
7. **Versioning.**
   - Every output records the SHA-256 of its inputs and the code commit; bundles record a hash for every file.
   - On a machine without `.git` the commit comes from a stamp written before the code is copied (§11.2 item 6).
     Outputs of the 2026-11-15 rented-GPU run predate it and say `"unknown"`.

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
   numbers.

   **Within-regime CAA** (asked for by the evaluation team on 2026-11-18, to separate
   "longer horizon" from "pick the far option"). Inside one regime the correct answer is the same for every prompt.
   - **Classes.** H/D cannot define them: within `none` or `near_only` the far option does not pay off, so H/D < 1
     for every prompt. Instead, inside a regime, long is the third of its prompts with the longest horizons and short
     the third with the shortest, split on log-horizon (`--within-quantile`, default 1/3). Whole grid horizons fall on
     one side, so a class is about, not exactly, a third.
   - **Four methods,** bundled like `caa`:
     - `caa-none`, `caa-near_only`, `caa-both`: long minus short inside that regime;
     - `caa-within`: the `near_only` and `both` differences averaged, weighted by regime size (`--within-pool`).
       `none` is left out of the pool by decision (Alan, 2026-11-22): there the horizon cannot change the answer, and
       the pool then covers 10 days to 500 years, the range the choice can test.
   - **`fit.json` → `diagnostics.caa_within`** has, per regime, the class sizes, the horizon cut-offs, each class's
     share of far choices, and cosines with `caa`, `choice` and `reward`; and the cosines between the regimes'
     directions.
   - **`fit_uids.jsonl`** has a `caa_within_class` column.
   - **On Qwen3.5-2B** (practice activations, layer 15, label site) [`steering/progress-20261122.md`]:

   | Direction | Long / short prompts | Horizons: long from, short up to | Cosine with `choice` | Cosine with `caa` |
   |---|---|---|---|---|
   | `caa` (for comparison) | 57 / 350 | set by H/D | +0.862 | 1 |
   | `caa-none` | 94 / 97 | 1.2 days / 23 minutes | +0.369 | +0.440 |
   | `caa-near_only` | 43 / 36 | 3 years / 6 months | +0.427 | +0.212 |
   | `caa-both` | 30 / 36 | 210 years / 59 years | −0.333 | −0.362 |
   | `caa-within` (pool of `near_only` and `both`) | 73 / 72 | | +0.011 | −0.140 |

     - The choice contamination drops from 0.86 to at most 0.43 in size, and to 0.01 for the pooled direction.
     - **The regimes' directions do not agree with each other:** their cosines are +0.112 (`both`, `near_only`),
       +0.008 (`both`, `none`) and −0.176 (`near_only`, `none`). One shared "horizon length" direction would make them
       point the same way. On this small model, which barely represents the horizon, that may only mean there is
       little to find; the same numbers on Gemma are the ones that matter.
   - **What a within-regime contrast does not remove:**
     - inside `near_only` and `both`, longer horizons come with longer far delays (rank correlation 0.40 and 0.35 on
       `t0_s0`, computed once and not by committed code), because of how the bank is built;
     - the model's own choice still varies inside a regime (far choices: 0.63 of the long class and 0.81 of the short
       class in `both`).
   - **For a label-shuffled null (N4):** `refit(horizons=...)` does not recompute regimes, so shuffle horizons within
     a regime for these methods. A shuffle across regimes would put a prompt's new horizon in its old regime's
     ranking.
3. **Concept controls.** `reward` (multiplier ≥ 25 minus ≤ 5) and `choice` (chose far minus chose near), built by the
   same recipe. The report names a reward-contrast control but doesn't define it; this definition is ours.
4. **The prelabel token depends on label style** (§4). This matters for any C1 analysis at that site.
5. **`fit.py` has no minimum class size.** On the 16-prompt smoke run it wrote a fit whose CAA long class held one
   prompt. Whether it should refuse below a minimum is an open decision (§13). Until then, check `fit.json`'s class
   sizes.
6. **The fitting site.** `--site prelabel` fits the same methods at the prelabel token, the one an edit must reach to
   change the choice (§6). On Qwen3.5-2B, layer 15, the prelabel fit is weaker than the label fit: cos(CAA, choice) is
   +0.551 against +0.862, and the curve's held-out Spearman is +0.121 against +0.222
   [`steering/progress-20261119.md`]. These are practice-model numbers. Which site the Gemma methods are fitted at is
   open (§13).

## 10. What has been checked, and on what

| Check | Model / data | Result | Source |
|---|---|---|---|
| Test suites | CPU + local GPU | steering 98 passed (2026-11-22, branch); prompt_bank 64 passed (2026-11-14) | `progress-20261122.md`; `progress-20261114.md` |
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
| Practice baseline | Qwen3.5-2B, thinking off, all 700 prompts | 700/700 parsed; none at the token cap; longest reply 147 tokens | `progress-20261117.md` |
| Extraction replay | same run, 700 prompts | 18 margins at −0.125 (one bf16 step), none lower | `progress-20261117.md` |
| Label-site audit | same 700 | the label token is the parsed label for 700/700 | `progress-20261117.md` |
| Hook gate, each of the 5 bundles | same run's bundles, 8 prompts | all pass, same counts as on 2026-11-14; `choice` gated for the first time | `progress-20261117.md` |
| Hugging Face upload | the practice hand-over, 42 files | 42/42 remote hashes verified; later made public | `progress-20261117.md` |
| Hook at batch 1 (no padding) | tiny Qwen3; Qwen3.5-2B | before the fix no generated token was edited; after it, one per decoding step | `tests/test_hooks.py`; `progress-20261119.md` |
| Hook gate at a single token | Qwen3.5-2B, 8 prompts: 5 label-site and 5 prelabel-site bundles | all 10 pass; exactly one token edited per row; results in §6 | `progress-20261119.md` |
| Horizon tokens found | Gemma 4 and Qwen3.5 tokenizers, thinking on and off, 671 horizons | all decode back to the stated horizon; shifted by one token, none does | `tests/test_sites.py` |
| Horizon extraction | Qwen3.5-0.8B, 21 prompts | stored rows equal `hidden_states` at those tokens exactly (layers 0 and 12) | `tests/test_extract_horizon.py` |
| Horizon-only edit in a padded batch | Qwen3.5-0.8B, 21 prompts | the edited tokens are each prompt's horizon; the 2 prompts without one are unchanged | `tests/test_extract_horizon.py` |
| Hook gate, fixed hook, sampled | Gemma 4 31B, A100 80 GB, 8 prompts, thinking on | zero edit 8/8 identical; roles ok (left padding [2, 4, 0, 3, 4, 8, 4, 6]); strong edit changed 8/8 | `progress-20261122.md` |
| Horizon extraction, full bank | Gemma 4 31B, thinking on, 700 prompts, 61 layers | 671 horizons, 2,600 tokens, 0 decode mismatches | `progress-20261122.md` |
| Sampled baseline | Gemma 4 31B, A100, first 16 extraction prompts, cap 4,096 | 16/16 parsed, 0 at the cap; the greedy run of 2026-11-15 had 13/16 and 3 (§11.3) | `progress-20261122.md` |
| Extraction replay and label audit, sampled run | same 16 | label is the top prediction for 16/16, margins +11.56 to +19.44; 16 label sites, 0 not matching | `progress-20261122.md` |
| Shared draws, fit with draws, bundles, hook gate per bundle | same 16 (plumbing: too few prompts for a meaningful fit) | draws verified; 8 bundles, all pass the gate; no `caa-near_only` (one prompt in that regime) | `progress-20261122.md` |
| Code commit without `.git` | the same rented run | every output records the stamped commit `cabc33d…` | `progress-20261122.md` |
| Copy-back | the same run, 122 files | 122/122 SHA-256 verified | `progress-20261122.md` |
| Horizon extraction, full bank | Qwen3.5-2B, thinking off, 700 prompts, 25 layers | 671 horizons, 2,600 tokens, 0 decode mismatches; 32 s | `progress-20261122.md` |
| Within-regime CAA | synthetic activations over the real bank; Qwen3.5-2B practice activations | class sizes 94/97, 43/36, 30/36; directions equal the stated differences of means; existing directions and curve unchanged; the 4 new bundles pass the hook gate | `tests/test_fit.py`; `progress-20261122.md` |
| `fit.py --draws` | Qwen3.5-2B practice activations | methods identical to the fit without draws; draws for other inputs refused | `tests/test_fit.py`; `progress-20261122.md` |

**Not yet checked on Gemma:**
- the hook gate at any `--positions` other than `all`;
- the steering test suite on a rented machine: it stalled there on 2026-11-22 in a test that loads Qwen3.5-0.8B, was
  stopped, and is not diagnosed (it passes locally);
- any full-size fit;
- any steered generation beyond the gate's strong-edit check.

## 11. Running on a rented GPU

### 11.1 The run script

`steering/tools/vast_smoke.sh STAGE …` (from the repository root, on the box) runs the pipeline one stage at a time
and stops at the first failure:

```
crlf  version  setup  weights  tests  hook  horizon  baseline  extract  audit  draws  fit  bundle  method  sums
```

- **Before copying the code to the box,** stamp it in the git checkout (§11.2 item 6):
  `cd prompt_bank && python -m pbank.provenance stamp`. The `version` stage fails if the outputs would record
  `"unknown"`.
- **`horizon`** runs `extract_horizon.py` on all 700 prompts; it needs no baseline run.
- **`draws`** runs the evaluation team's `shared_draws.py` and verifies its output; `fit` then uses the draws.
- **`method`** runs the hook gate on every bundle of the run.

- **Settings** come from the environment: `MODEL` (default Gemma 4 31B-it), `LAYER` (38), `N` (64 prompts), `BATCH`
  (8), `MAX_NEW` (4096), `RUN` (smoke), `THINKING` (on), `DECODING` (auto, as `baseline.py`), `SAMPLING` (extra
  sampling options for a model without a generation config) and `DRAWS_MIN_ROWS` (100).
- **Rehearsed locally on 2026-11-22** in a copy without `.git`, on Qwen3.5-0.8B: every stage but `setup`, `weights`
  and `tests` ran and passed, the hook gate on all nine bundles included [`steering/progress-20261122.md`].
- **Run on a rented A100 the same day** on Gemma 4 31B: every stage passed except `tests`, whose steering half
  stalled and was stopped (§10). The run cost about $2.15 for 50 minutes.
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
   retrying with a different batch size is refused until the empty directory is removed. That is deliberate: a reply
   depends on the batch it is generated in, so batch size is part of a run's identity.
5. **No `/workspace`.** The image `vastai/pytorch:cuda-12.8.1-auto` has no `/workspace`, which the team justfile
   assumes.
6. **Shipping without `.git` loses provenance.** `just push`-style rsync leaves out dotfiles, including `.git`, so
   every output recorded its code commit as `"unknown"`. **Fixed on 2026-11-22:**
   - `python -m pbank.provenance stamp`, run in the git checkout before copying, writes `CODE_STAMP.json`: the commit
     and the SHA-256 of every tracked file (dotfiles left out, since such a copy drops them).
   - Without git, outputs record the stamped commit if the code files still match their hashes, and the commit with
     `-modified` if any was edited on the box.
   - Rejected: passing the commit in an environment variable, which would keep vouching for the commit after the code
     was edited; and shipping `.git`, which is large and changes how the team's `just push` works.
7. **Bandwidth is billed by some hosts.** The host billed $40/TB in both directions, so the 64 GB download (weights
   plus packages) cost $2.50, more than the GPU time. Filter offers on download cost.
8. **Destroy, don't stop.** A stopped instance keeps billing its disk: $0.28/h for 200 GB.

### 11.3 Throughput: the open problem

Measured on the A100 (batch 8, thinking on, cap 4,096 new tokens):

```
[8/64]  x0001..x0008: 7/8 parsed, max 4096 tokens, 75.6 s/prompt
[16/64] x0009..x0016: 6/8 parsed, max 4096 tokens, 74.7 s/prompt
```

A batch decodes until its longest reply stops. So the time per prompt is (ms per decoding step) × (the batch's longest
reply) / (batch size). `prompt_bank/tools/step_times.py` computes both quantities from a run's records. The
2026-10-17 pilot's figures come from its committed logs:

| Per A100 80 GB | 2026-10-17 pilot (`alan/` pilot prompts) | 2026-11-15 run (`t0_s0`, first 16 prompts) |
|---|---|---|
| GPUs, batch, token cap | 2 × A100 (one process each), batch 4, cap 8,192 | 1 × A100 SXM4, batch 8, cap 4,096 |
| replies: median / mean / at the cap | 716 / 747 tokens / 0 of 384 | 1,306 / 1,661 tokens / 3 of 16 |
| decoding steps per prompt | 270 | 512 (every batch ran to the cap) |
| ms per decoding step (median) | 91.9 | 146.7 |
| **seconds per prompt per GPU** | **25.4** | **75.1** |
| 700 prompts | 4.9 GPU-hours | 14.6 GPU-hours |

- **The often-quoted "13 s per prompt" was the pilot's wall time with two GPUs in parallel.** Per GPU it was 25.4 s,
  4.9 GPU-hours for 700 prompts, not 2.5. `prompt_bank/README.md` now says so.
- **Why the per-GPU figures differ by about 3** (1.9 × 1.6):
  - `t0_s0` draws longer replies, and a runaway reply holds its whole batch to the cap, so there are 1.9 × the
    decoding steps per prompt;
  - each step is 1.6 × slower, at batch 8 and longer contexts. In the pilot, steps slow from about 91 ms (longest reply
    ≤ 1,500) to about 109 ms (> 3,000) at batch 4.

  The runs also differ in transformers version (5.17 vs 5.22.2) and generation code, so the 1.6 is not attributed
  further. The 2026-11-15 figures rest on 16 prompts.
- **Planning figure:** about 15 GPU-hours for this bank at cap 4,096 under greedy decoding, and more at the spec's
  8,192. With sampling the 2026-11-22 run suggests about 10 (below).
- **The largest lever is the runaway reply,** not the GPU: one long reply makes the other seven slots in its batch wait.
  Options include a lower cap, smaller batches, or generation that refills finished slots. None is decided, and each
  has trade-offs: a lower cap changes which replies get parsed.
- **Not yet explained:** even the pilot's 92 ms per step is about 3 × the memory-bandwidth floor (62 GB of weights read
  per step at about 2 TB/s, about 31 ms). That needs a profile of `generate()` on Gemma 4's layer layout, which can be
  done locally on a model with the real layer layout but fewer layers.

**Decoding: replies that loop under greedy decoding.** Small Qwen models with thinking on often repeat the
same block of reasoning until the token cap, so the reply never reaches an answer:

- **Our probes, cap 4,096:** Qwen3.5-2B parsed 0 of 16 replies and Qwen3.5-4B 3 of 16
  [`steering/progress-20261117.md`]. This is why the practice hand-over has thinking off.
- **The evaluation team's full run** (Qwen3-1.7B, thinking on, greedy, cap 4,096): 370 of 700 prompts answered. 337
  hit the cap, 323 of them in a loop (a 64-token block repeated at least 3 times)
  [`evaluation/reports/Qwen3-1.7B__t0_s0__think_loops.txt`].
- **Sampling instead of greedy** (reported by the evaluation team on 2026-11-19, with Qwen's recommended settings and
  a fixed seed; code on branch `eval-runner`, `evaluation/sampling_test.py`; not rechecked here):
  - of 100 prompts that looped, 5 still loop and 62 now answer;
  - the other 33 are long, not looping, and still hit the cap;
  - on prompts that greedy had answered, sampling changed 5 of 26 choices, so a readout from generated answers would
    need several samples per prompt.
- **Gemma 4 31B:** 3 of the first 16 replies hit the cap (table above). They are not loops (below).
- **Decided (Alan, 2026-11-22): seeded sampling for thinking-on runs, and a default cap of 4,096 tokens.** Built:
  - `baseline.py --decoding {auto,greedy,sample}`. `auto`, the default, samples when thinking is on and is greedy
    when it is off. The cap is `--max-new-tokens`, default 4,096 (it was 8,192).
  - Sampling values come from the model's own generation config (Gemma 4: temperature 1.0, top-p 0.95, top-k 64)
    unless `--temperature`, `--top-p` or `--top-k` are given. A model without such a file (Qwen3.5) must be given all
    three. The values and where each came from are recorded in `run.json`.
  - `--seed` (default 0). Each batch is seeded from it and from the batch's own prompts, so the same batch repeats
    exactly whatever ran before it. On Qwen3.5-2B, two runs gave identical token ids for 16 of 16 prompts.
  - **The replay check** for a sampled run is that the label is among the top-k tokens, since it need not be the most
    likely one (§2, "Replay margin"). On a sampled Qwen3.5-2B run the label was the second most likely token for 35
    of 96 prompts; the greedy check would have flagged all 35.
  - **The hook gate** takes the same options. A zero edit still reproduces a sampled batch exactly (6 of 6 prompts
    on Qwen3.5-0.8B), because the unhooked and hooked runs are seeded alike.
  - Bundles record the baseline's decoding (`method.json` → `fitted_on.decoding`).
  - New runs are not comparable with the earlier greedy ones. `--decoding greedy` reproduces their setting.
- **What sampling does not fix** [`steering/progress-20261122.md`]:
  - **Qwen3.5-2B with thinking on still answers nothing.** With the sampling values above for Qwen, all 16 of 16
    replies ran to the 4,096 cap, as under greedy. None is a loop by the evaluation team's definition: the model
    deliberates in circles ("Wait, …") without repeating itself exactly.
  - **Gemma's capped replies were not loops either.** The 3 of 16 replies at the cap in the 2026-11-15 greedy run
    are long deliberations (all three on horizons under a week, where neither option pays off), not repeated
    blocks. Sampling helped them all the same (next item).
- **Sampling on Gemma 4 31B: the same 16 prompts, greedy (2026-11-15) against sampled (2026-11-22)**, both at batch
  8 and cap 4,096 on an A100; sampled with Gemma's own settings and seed 0 [`steering/progress-20261122.md`]:

  | | greedy | sampled |
  |---|---|---|
  | answered | 13 of 16 | 16 of 16 |
  | at the cap | 3 | 0 |
  | tokens per reply: median / mean / max | 1,306 / 1,661 / 4,096 | 1,062 / 1,383 / 3,091 |
  | seconds per prompt | 75.1 | 50.6 and 53.5 (two batches) |

  - The 13 prompts both runs answered have the same choice in both.
  - The three that hit the cap under greedy now answer, at 2,337 to 3,091 tokens: 57 to 75% of the cap.
  - **Limits:** 16 prompts and one seed. Other prompts or seeds may still hit the cap.
  - **Planning figure with sampling:** about 10 GPU-hours for the 700 prompts at 52 s per prompt.
  - **One reply is one draw.** Readouts that count generated answers need several seeds per prompt.
  - **The evaluation runner** calls `generate` without a decoding argument, so its generation check is greedy until
    it passes the run's decoding (`evaluation/replay_eval.py`; theirs to change).

Throughput and decoding together are the main thing standing between the current state and a full Gemma run. They
also set the cost of every generation readout the evaluation pipeline plans `[eval §9]`. Note that the evaluation
design's cost estimate quotes the old 13 s figure.

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
- **Hosting: Hugging Face.** The evaluation team agreed on 2026-11-19 to use it for everything.
  - The Hugging Face path has run for real twice (§10).
  - The one real GCS upload (2026-11-17) was refused, because the project's billing account is closed. Nothing was
    written [`steering/progress-20261117.md`].

**The practice hand-over.** A complete small-model hand-over is public, for building and debugging evaluation code
against the real interfaces: <https://huggingface.co/datasets/CodeReclaimers/spar-steering-practice-qwen35-2b>.

- **Contents:** the baseline run (Qwen3.5-2B, thinking off, all 700 prompts), activations at layers 12 and 15 (label
  and prelabel), the layer-15 label-site fit, and the five bundles. Its dataset card lists the commands and checks.
- **It is for plumbing only.** Thinking is off, the model barely represents the horizon (the curve's held-out
  Spearman is +0.222), and answers flip easily (the median replay margin is 0.25 logits). Draw no conclusion about
  steering from it.
- **Not in it:** the prelabel-site fit and bundles of §6, which exist only locally.

## 13. Open items

**Answered by the evaluation team (2026-11-18 and 11-19):**
- **Held-out horizons for C8:** a first list (§8.2 item 3), not final.
- **Shared draws:** built on their side (§8.2 item 6).
- **`EditContext`:** it covers their comparison methods. What is missing is upstream: the `horizon` site below.
- **Host:** Hugging Face (§12).

**Requested by the evaluation team, built:**
- **The horizon site** (§6): activations at the tokens that state the horizon, every token separately, all layers,
  with the audit and the per-prompt rows for the hook. Still to do: run it on Gemma 4 31B.
- **`fit.py`'s random baseline from the shared draw files** (§8.2 item 6).
- **A within-regime CAA** as comparison methods (§9.2).

**Decisions on the extraction side:**
- what to do with a prompt whose reply hits the cap under sampling: another seed, a higher cap, or accept the loss
  (§11.3);
- the throughput problem (§11.3);
- the bundles' default positions, and whether to fit the Gemma methods at the prelabel site (§4, §6, §9.6);
- a minimum class size in `fit.py` (§9.5).

**To do:**
- the full 700-prompt Gemma baseline, then `extract.py`, `shared_draws.py`, fit and bundles;
- before it, `tools/vast_smoke.sh` must run all 700 prompts and extract both splits: today it takes the first `N`
  extraction prompts and extracts the extraction split only;
- upload the Gemma horizon activations;
- diagnose the steering test suite's stall on a rented machine (§10).

**Not built:**
- an answer-letter control direction;
- the switch-point and gate code (roadmap step 2's second half). The evaluation runner now calibrates the strength
  (§8.2 item 4); whether anything of this remains for extraction is to be confirmed with the evaluation team.

## 14. Sources

- `prompt_bank/README.md`: the bank, the decision rule, the baseline runner, the answer parser.
- `steering/README.md`: the steps, conventions, fitting choices, hook, bundles, uploads.
- `steering/progress-20261114.md`: Steps 1–6, the Qwen3.5 checks, the local Gemma smoke test.
- `steering/progress-20261115.md`: the vast.ai run, its gotchas and its numbers.
- `steering/progress-20261117.md`: the Qwen3.5-2B practice hand-over, its checks and its uploads.
- `steering/progress-20261119.md`: the label-only and prelabel-only gate, and the batch-1 hook fix.
- `evaluation/`: the shared draws (`shared_draws.py`), the evaluation runner (`replay_eval.py`) and its first reports.
- The evaluation team's review and requests: Slack, #planning-temporal-manifolds, 2026-11-18 and 2026-11-19.
- Code: `steering/steer/{edits,hooks,methods,manifold,fitting,layers,sites}.py`, `steering/tools/check_hook.py`,
  `steering/tools/vast_smoke.sh`, `steering/tools/upload_{hf,gcs,common}.py`.
- Figures: `explainer/make_figs.py`.
  - f1 is computed from the committed bank, and asserts the counts quoted in §9.
  - f2 is an illustration.
- The evaluation design: `evaluation_design/README.md`.
