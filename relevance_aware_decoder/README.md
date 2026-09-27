# Relevance-aware horizon decoder

**Status: design for group review. Nothing has been run yet** (only unit tests and a synthetic dry run).
Branch `relevance-aware-decoder`. Builds directly on Alan's matrix run and code in `../alan/` (used read-only).

## 1. The question

Alan found that a linear horizon decoder at boundary tokens is **pulled by irrelevant durations**
(`alan/progress-20260925.md`). The sentence "The household's savings account was established 20 years ago"
moves the decoder's readout toward 20 years, even though the model's *choice* ignores it:

| | effect of the irrelevant duration |
|---|---|
| model's choice (logistic coefficient on log D; the real horizon's is −1.7 to −2.0) | −0.03 (mention), −0.11 (role) |
| decoder readout, ρ(error, D − H) at his selected cell L26 T1 | +0.58 (mention), +0.74 (role); every one of 70 cells > +0.14 |

That decoder was trained only on prompts that contain **one** duration, so it never had a reason to tell
"the horizon" apart from "a duration". The result therefore does not yet say whether the *model* keeps the two
apart. This experiment asks:

> **If we train the decoder on prompts that contain irrelevant durations (labelled with the real horizon),
> does the pull go away, and does that hold for wording and for kinds of distractor it never saw?**

It matters for project goal 2 (a per-turn horizon monitor): a monitor that reports "durations mentioned in
context" instead of "the horizon the model is planning with" is not useful.

## 2. Possible outcomes and what each would mean

| Outcome | What we would see | Interpretation |
|---|---|---|
| **Separable** | pull ≈ 0 even for a *kind* of distractor never seen in training (tier C) | The model represents "my planning horizon" separately from other durations, and a linear readout can reach it. Alan's pull was a training artefact. |
| **Role-level only** | pull ≈ 0 for trained kinds with new wording (A, B), but not for a new kind (C) | The separation is learned per distractor type; a monitor would need to see every kind of distractor in training. |
| **Wording only** | pull ≈ 0 for trained sentences (A) but not for new wording (B) | The decoder memorised sentences. The question stays open. |
| **Entangled** | pull stays even for trained sentences (A) | At that layer and token, the two durations are mixed in a way no linear readout can undo. A finding about the representation, and bad news for a linear monitor. |

Each outcome is decided per distractor family, with the rules in §5. Proposed thresholds are listed there
so the group can agree on them **before** anything runs.

## 3. Design

### 3.1 Prompts (5,520 in total)

**Clean prompts: Alan's matrix, unchanged** (`ptm.matrix.build_matrix`, seed 0; a test checks the texts are
identical to the matrix file in his snapshot). That is 30 option configurations × 2 domains (investment, climate) × 12 horizons
(1 day to 100 years) × 3 renderings (structured, plain prose, varied prose) = 2,160 prompts. His configuration-level
splits are kept: 20 train, 4 dev, 6 test configurations. Recapturing them (instead of reusing his activations) keeps
everything on one machine and one software stack.

**Distractor prompts: a plain-prose matrix prompt plus one extra sentence** with an irrelevant duration D.
Every distractor prompt has a **clean twin**: the same configuration, domain and horizon without the sentence.
The twin is what makes the paired metric (§4) possible.

Five **families**, one temporal role each. Each has 4 templates: 2 **seen** (allowed in training) and 2 **unseen**
(test only).

| Family | Seen templates (train + tier A) | Unseen templates (tier B only) | D drawn from |
|---|---|---|---|
| `entity_age` (Alan's "mention", generalised) | {Entity} was established {D} ago. · {Entity} was set up {D} ago. | {Entity} has existed for {D}. · {Entity} first opened {D} ago. | 1 day – 100 years (12-point grid) |
| `prep_time` (Alan's "role", generalised) | You have {D} to prepare your recommendation. · Your recommendation is due in {D}. | The meeting where you present your choice is in {D}. · You were given {D} to study the two options. | 1 hour – 1 month |
| `tenure` | You have held this role for {D}. · You took on this responsibility {D} ago. | Your predecessor held the role for {D}. · You last reviewed {role}'s plans {D} ago. | grid up to 30 years |
| `background_event` | The office building was last renovated {D} ago. · The local school finished building its new gym {D} ago. | The town library changed its opening hours {D} ago. · {Project} was completed {D} ago. | grid up to 30 years |
| `other_horizon` | For an unrelated decision last year, {role} used a time horizon of {D}. · {Neighbour} plans its {thing} over a horizon of {D}. | The previous plan, which has now ended, covered a period of {D}. · A separate long-term project of {role} runs for {D}. | 1 day – 100 years |

`{Entity}` is "the household's savings account" / "the city's climate office" (Alan's wording); the other
placeholders are filled per domain the same way (`rad/families.py`).

`other_horizon` is the hardest family on purpose. It uses horizon language for a *different* decision, and it
defeats the text baseline that reads "the sentence containing *horizon*". That is the case where an activation
monitor could beat a regex.

Other design choices:
- **Where the sentence goes.** It is inserted at one of three places: after "You are …", before the options
  (Alan's mention slot), or after the horizon sentence (Alan's role slot). Slot and template are assigned from
  a balanced, shuffled list, so they are not confounded with horizon, configuration or domain.
- **D is drawn uniformly** from the family's set and is never equal to the prompt's own horizon.

**Counts** (distractor prompts):

| Split | Configurations | Horizons | Families × template groups | Prompts |
|---|---|---|---|---|
| train | 20 × 2 domains | 8 training horizons | 5 × seen | 1,600 |
| dev | 4 × 2 | 8 | 5 × seen | 320 |
| test | 6 × 2 | all 12 | 5 × (seen + unseen) | 1,440 |

Total: 3,360 distractor prompts + 2,160 clean = 5,520.

### 3.2 Decoders

A ridge regression from one (layer, token) cell to log₁₀ horizon, the same model as Alan's. The ridge penalty
is chosen on dev, then the decoder is frozen.

| Decoder | Trained on |
|---|---|
| `baseline` | clean prompts only: train configurations, training horizons, all three renderings. This is Alan's condition 2. |
| `rad_all` | baseline rows + distractor rows of **all five** families (seen templates) |
| `lofo_<f>` | baseline rows + distractor rows of every family **except f** (leave one family out; 5 decoders) |

### 3.3 Test tiers

All tiers use the 6 **held-out test configurations**, so every scenario is new to every decoder.

| Tier | Decoder | Test prompts | Passing shows |
|---|---|---|---|
| pre | `baseline` | all distractor test prompts | how much pull there is to remove (replicates Alan on 5 families) |
| A | `rad_all` | seen templates | it can ignore sentences it was trained on, in new scenarios |
| B | `rad_all` | unseen templates of trained families | it ignores the *role*, not the exact words |
| C | `lofo_f` | family f, all four templates | it ignores irrelevant durations in general, including a role never seen |

Per family there are 144 prompts in tiers A and B and 288 in tier C (12 scenario clusters).

### 3.4 Where in the network

The sweep covers 7 layers {14, 18, 22, 26, 29, 33, 37} × 10 positions {T0–T8, R0} = 70 cells, reporting point
estimates for every decoder and tier. Full results with bootstrap intervals are reported at two cells:
Alan's selected cell **L26 T1**, and the cell with the best `rad_all` dev error.

## 4. Metrics

**Primary: paired pull fraction.** For each distractor prompt and its clean twin:

    delta = readout(distractor prompt) − readout(twin)        (log10 years)
    gap   = log10 D − log10 H
    pull  = slope of delta regressed on gap

- pull = 0 means the readout ignores the distractor.
- pull = 1 means it moves all the way to D.
- 95% intervals come from a bootstrap over scenarios (configuration × domain).

**Why paired instead of Alan's ρ(error, D − H).** The gap D − H is itself correlated with −H, and ordinary
decoders over-predict short horizons and under-predict long ones. So the unpaired metric can be clearly
positive when there is **no** pull at all. `tests/test_analysis.py` shows this on synthetic data: true pull 0,
unpaired ρ > 0.3. Pairing subtracts everything the two prompts share. We still report Alan's metric next to it
for continuity.

**Behavior gate: is the distractor really irrelevant to the model?** Using the fp32 choice logits:

    behavior pull = slope(Δ short log-odds on gap) / slope(short log-odds on log10 H)

This puts the model's own shift on the same scale as the readout's. A family counts as **irrelevant** if its
behavior pull's 95% interval lies within ±0.10.

A family that fails the gate (plausible for `other_horizon`) is not scored as "should be ignored". It is
reported separately. For such a family the interesting question flips: does the readout move by the *same*
amount as behavior? That links to Alan's second candidate, a probe for the behaviorally defined horizon.

**Also reported:**
- clean-test accuracy (within 2×, RMSE), so a decoder can't "ignore" distractors by getting worse overall;
- text baselines on every tier: first-duration regex, "horizon"-sentence regex, and a TF-IDF ridge trained on
  the same rows as each decoder.

## 5. Decision rules (proposed; please agree before running)

These are applied per family, at each headline cell. Only families that pass the behavior gate are scored.

1. **Nothing to remove:** baseline pull < **0.20**.
2. **Pull removed:** the 95% interval's upper bound is < **0.10**.
3. **Verdict:**
   - SEPARABLE if removed in C;
   - ROLE-LEVEL ONLY if removed in A and B but not C;
   - WORDING ONLY if removed in A only;
   - ENTANGLED if pull in A ≥ half the baseline pull;
   - PARTIAL otherwise.
4. **Accuracy guard:** the verdict counts only if `rad_all` loses ≤ **0.05** clean-test within-2× compared with
   `baseline`.

Headline claim: SEPARABLE at the headline cell in at least 4 of the eligible families (or in all of them, if
fewer than 4 are eligible). All thresholds are command-line flags of `scripts/evaluate.py`.

## 6. Limitations and open choices for the group

- **Explicit horizons only**, and still single-turn. A pass here is necessary for a multi-turn monitor, not sufficient.
- **Distractor prompts are plain prose only** (the clean prompts cover all three renderings). Crossing distractors
  with renderings would triple their count.
- **Linear decoders only.** If the verdict is ENTANGLED, a small non-linear probe on the same activations is the
  obvious follow-up (no new capture needed).
- **One model** (Qwen3-14B, thinking off) and one seed. A second seed only re-draws D, slots and templates. It would
  cost about half a run (§8).
- **Six test configurations** (12 scenario clusters) keep the intervals honest, but wide-ish.
- **Open for discussion:**
  - Are these five families the right ones?
  - Should `other_horizon` be split into "plainly irrelevant" and "arguably relevant" versions?
  - Are the thresholds right?

## 7. How to run

**Local, no GPU (a few seconds):**

    cd relevance_aware_decoder
    python -m pytest -q tests          # 13 tests, including a synthetic end-to-end run with a known answer
    python scripts/gen_prompts.py data --smoke   # writes data/prompts_s0.parquet (+ smoke slice) and prints examples

Use any Python ≥ 3.12 with `requirements.txt` installed. Alan's `ptm` package is imported from `../alan`
automatically; nothing is installed into or written to that folder.

**Local plumbing test on the 8 GB GPU (recommended before renting):**

    bash scripts/smoke_local.sh        # Qwen3-1.7B on ~600 prompts; checks capture → twins → evaluation → summary

**Full run (one GPU with ≥ 40 GB; see `COST.md` for the vast.ai procedure and guardrails):**

    bash scripts/run_pipeline.sh       # prompts → tests → capture check → capture → evaluate → subset export

**Outputs** go to `results/qwen3-14b_relevance_s0/`:
- `summary.md`: the verdict tables;
- `behavior_gate.csv`, `headline_cells.csv`: pulls with intervals;
- `sweep.csv`, `sweep_pull.png`: per-cell pulls;
- `text_baselines.csv`, `selection.csv`, `predictions.npz`.

A shareable activation subset (layers 22, 26, 29 × T0–T8, R0; ≈ 1.7 GB) is also written. The full activations
(≈ 39 GB) stay on the instance.

## 8. Cost

On one 48 GB GPU from vast.ai (RTX A6000 or RTX 6000 Ada), the whole run is about **1.5–2 hours of rental,
roughly $1–2**. On an H100 it is about 1 hour for roughly $3–4. The estimate, the price snapshot and the
step-by-step procedure are in [`COST.md`](COST.md). Nothing is rented without explicit confirmation.

## Files

    rad/families.py      distractor families, templates, duration sets
    rad/build.py         prompt set (Alan's matrix + distractors, twins, balanced slots/templates)
    rad/analysis.py      ridge (one SVD, many penalties), paired pull, behavior pull, text baselines
    scripts/gen_prompts.py   scripts/evaluate.py   scripts/run_pipeline.sh   scripts/smoke_local.sh
    tests/               prompt-set invariants, metric checks, synthetic end-to-end run
    data/                generated prompt files (deterministic; regenerate with gen_prompts.py)
