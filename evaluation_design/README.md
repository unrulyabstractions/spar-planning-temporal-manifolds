# Evaluation pipeline for horizon steering: design proposal

**Status:** proposal for team discussion, nothing decided. **Owner:** Evaluations (random baselines and non-linear methods:
Augusto Nicola; prompt generation: Jared). Drafted with Claude, 2026-11-14.
**Human-readable version:** [`explainer/index.html`](explainer/index.html) (same content, with figures).
**This file** is the reference version: plain structure, every term defined, every claim sourced, so that it can be read
by people and by LLM assistants working on the pipeline.

## How to read this document

- Section 1 says what the pipeline is for. Section 2 defines every term and acronym used later.
- Section 3 lists the information about a steering method that the pipeline can use, and what each item makes possible.
- Section 4 separates what the extraction and evaluation teams must agree on from what the evaluation team decides itself.
- Sections 5–9 are the technical design: principles, random baselines, comparison methods, confound checks, stages.
- Sections 10–12: the report each method gets, the open decisions, and the references.
- Claim sources are given in square brackets: `[R3]` is reference 3 in section 12; `[spec A.4]` is section A.4 of the
  experiment spec; file paths point into this repository.

---

## 1. Purpose

The project asks whether the planning horizon a language model acts on can be read from its activations and **steered**
(made more myopic or more long-term) [spec, Hypotheses]. Prior work found that the horizon forms an ordered, curved
structure in activation space for single decisions [R4]. The extraction team (`steering/`) turns the prompt bank
(`prompt_bank/`) and the model's activations into steering methods: so far a linear direction (CAA) and a manifold
curve, with random and reward-contrast controls [`steering/README.md`].

The evaluation pipeline takes any such method and answers two questions:

1. **Does it steer the horizon?** Not just "does it change answers", but "does it change them the way a different
   horizon would, more than a matched random edit would, and not through some other variable?"
2. **How well?** How large the effect is, whether it lands where it was aimed, and what it costs in answer quality.

It sits in roadmap step 2 ("Extract and verify: gate and calibration on structured prompts") and is reused in steps 3–5
(linear vs manifold, transfer to realistic prompts, replication) [spec A.6].

Why it needs designing now: the team's earlier steering runs record as "not decided and blocking" what counts as a
success for a steering vector, the number of random draws, seeds, and multiple-comparison handling
[`jared/pipeline/s2_steering/DECISIONS.md`, line 24]. Random controls have also looked as good as real methods: in the
E15 scan, random cells averaged 2.9 label flips against 2.5 for the manifold step [same file, line 144].

## 2. Glossary

**Prompts and data**

- **T0 bank / structured prompts:** 700 templated two-option investment prompts (`prompt_bank/`, tier T0). Each states a
  horizon H in a CONSTRAINT line, e.g. "…for this time horizon: 8 years." Realistic tiers T1–T3 come later [spec A.2].
- **Near / far option:** the sooner, smaller payout and the later, larger one. D = the far option's delay.
- **H/D:** stated horizon divided by the far delay. **Regime:** which options pay off within H: `none`, `near_only`,
  `both`, or no horizon given (`null`) [spec A.1].
- **Verification grid:** 100 held-out items near the boundary where the correct answer flips (H/D ∈ [0.35, 2.8]), each
  in both option orders; never used for fitting [spec A.1; `prompt_bank/README.md`].
- **Option order / label style:** whether near is listed first; labels `a)/b)` or `1)/2)`. Both are randomised in the
  bank [spec, Methods].
- **Unit rewrite:** the same duration in another unit ("100 weeks" for "23 months") [spec A.1].
- **Null-control prompt:** the same prompt with the CONSTRAINT line removed [spec A.1].

**Model internals**

- **Residual stream / activations / h:** the vector a transformer layer passes to the next, one per token. Width d
  (thousands of numbers).
- **Layer L:** the residual stream after block L, counting from 1 (`steering/` convention). Headline: layer 38 of 60 on
  Gemma 4 31B [spec A.3].
- **Site / position:** which tokens are read or edited. `label` = the token of the model's own answer label
  ("I choose: **b**)"); `prelabel` = the token before it [`steering/README.md`].
- **Typical norm:** mean length of activation vectors at a layer and site. Strength is a fraction of it.
- **Edit / steering / intervention:** changing h during the forward pass, then letting the model continue.
- **Thinking on:** the model writes a reasoning trace before answering.
- **bf16:** the 16-bit number format models run in; about 3 significant digits.
- **PCA, PC:** principal component analysis; a PC is one of the directions along which a set of vectors varies most.
- **Per-horizon means:** the average activation of all prompts stating the same horizon (48 horizons from 30 s to
  500 y [spec A.1]).
- **Manifold curve:** a smooth curve through the per-horizon means in their top 3–8 PCs, indexed by log-time t; **t̂**
  is the point on the curve nearest to an activation [spec A.3; `steering/steer/manifold.py`].

**Steering methods**

- **CAA (contrastive activation addition):** add a fixed direction, the mean activation of one class minus another
  [R1; the related ActAdd is R3]. Here: long-horizon prompts (H/D ≥ 4) minus short-horizon ones (H/D ≤ 1/4) [spec A.3].
- **Curve slide:** move an activation along the manifold curve by Δ decades of log-time. `snap` replaces the subspace
  part with the curve point at t̂ + Δ; `displace` adds c(t̂ + Δ) − c(t̂) [`steering/README.md`, choice 6].
- **Reward-contrast direction:** CAA built from reward size instead of horizon; a control [spec A.3].
- **Donor patching:** run the same prompt with another horizon (the donor), copy its activations at a site, paste them
  into the original run.
- **Centroid substitution:** h − μ_source + μ_target, with μ the per-horizon means [R21].
- **Spline replacement:** replace the subspace coordinates with a point on a spline through the means [R13].
- **Mean-AcT / Linear-AcT (activation transport):** per-dimension shift, or shift plus scale, mapping one class's
  activations onto another's [R15]. **MiMiC:** the same idea matching mean and covariance [R16].
- **Slerp / spherical steering:** rotate h toward a target direction while keeping its length [R17].
- **DAS / Boundless DAS (distributed alignment search):** learn a small subspace in which swapping in a donor's
  coordinates reproduces the donor's behaviour [R18]. **ReFT / LoReFT:** trained low-rank edits [R19].
- **Push vs targeting method:** a push method has a strength knob ("more long-term"); a targeting method takes a
  requested horizon or shift ("act as if 2× longer").

**Measurement**

- **Switch point s:** the H/D at which the model's choice flips from near to far. **Acted-on horizon** H* = H/s; the
  steering effect is **Δ log H*** = −Δ log s, in decades (factors of 10) [spec A.4].
- **α_k:** the strength at which the model acts on a horizon k times longer (default k = 2) [spec A.4].
- **S/R (transfer efficiency):** S = shift from steering at α_k; R = shift from rewriting the prompt with a k-times
  longer horizon [spec A.5].
- **Replay readout:** re-run the model's own (unsteered) answer as a fixed input, with the edit applied, and read the
  probabilities of the two labels at the answer slot. One forward pass; no new generation [`steering/extract.py` does
  the same replay without an edit].
- **Generation readout:** let the edited model write a full thinking-on answer and parse it.
- **Odd / even part:** for a ± pair of edits with mean effects Δ(+x) and Δ(−x): odd = (Δ(+x) − Δ(−x))/2 (reverses with
  the sign: steering), even = (Δ(+x) + Δ(−x))/2 (same either way: disruption).
- **Null / null family:** a set of random edits that a method must beat. **K:** number of random draws.
- **Rank p-value:** p = (1 + b)/(1 + K), where b = number of random draws at least as large as the method [R8].
- **KL divergence:** a measure of how much the model's next-token probabilities change.
- **Health metrics:** coherence (LLM judge), refusal, format failure, truncation [spec A.5].
- **LLM judge:** another model asked to grade outputs, validated against hand labels.

## 3. What information about a method the pipeline can use

The pipeline should evaluate methods it didn't build and may not fully understand. Instead of requiring a fixed
"package" for each method, this section lists the pieces of information that matter, why each matters, and what it makes
possible. The more of them the extraction team can share for a method, the more controls the pipeline can run. The
report card (section 10) says which controls were possible and which weren't, and why.

### 3.1 The items

**I1. The edit, applicable during generation.** *Why:* without it nothing can be evaluated. The `steering/` package has
numpy reference edits but, as its README notes, "applying the edits inside the model during generation" is "not here
yet" [`steering/README.md`, "Not here yet"]. *Enables:* every readout, the black-box nulls (isotropic N1, damage-matched
N6), the ± split, all confound checks that only need outputs. *Example:* CAA as "add strength × typical_norm × caa at
every position of layer 38"; the curve slide as `Manifold.steer(mode, Δ)`.

**I2. Conventions.** Layer index, positions edited, strength units, sign, and for targeting methods the unit of the
request (decades of log-time). *Why:* random controls must match the method's size token by token and run at the same
layer and positions; a mismatch silently changes the comparison. *Enables:* matched nulls, comparable numbers across
methods. *Example:* `steering/` already fixes these (layer after block L, `strength × typical_norm`, unit vectors).

**I3. The fitting data: which prompts, which labels, which classes.** *Why:* (a) to make sure no evaluation prompt was
used for fitting; (b) because the composition of the classes can create confounds: CAA's long class contains only `both`
prompts and its short class only `none` and `near_only` ones [`steering/README.md`, choice 2], and whether `none` prompts
belong at all is open [choice 1]. *Enables:* split hygiene; diagnosis of class confounds (option-order and label balance
per class, section 8 C1–C2). *Example:* `fit.json` records class sizes; `per_horizon.jsonl` records regimes per horizon.

**I4. The fitting procedure as callable code.** A function that refits the method on a given set of prompts and labels.
*Why:* several controls are "the same method, fitted differently". *Enables:* the label-shuffled null (N4: refit with
horizon labels shuffled); refits with some horizons held out (calibration, C8); refits within one option order (C1);
bootstrap over prompts (stability of the method itself). *Example:* `fit.py` is a deterministic command-line tool with
its options recorded in `fit.json`; wrapping it as `refit(prompt_ids, labels)` would cover this.

**I5. The activations at the site.** *Why:* realistic random directions must look like real activation differences.
*Enables:* the covariance-matched null (N2) and real-difference null; concept-control directions built the same way as
CAA (chosen option, answer letter); the comparison methods of section 7, which are computed from per-horizon means.
*Example:* `activations/<run>/L38_label.npy`. If not shared, the evaluation team can extract them itself with
`extract.py`, so this item costs little.

**I6. The method's internal structure.** The subspace basis, the curve, the t̂ reader, the learned parameters.
*Why:* some failure modes are specific to the structure. *Enables:* in-subspace nulls and random curves (N3); the Δ = 0
control for `snap`, which is not the identity (it removes the off-curve part of h) [`steering/README.md`, choice 6];
calibration in the method's own units (C8); interpretation of rescaling vs replacing (C6). *Example:* `manifold.npz`.

**I7. For trained or optimised methods: the objective and the training set.** *Why:* a method chosen by its effect beats
unchosen random edits almost by construction. *Enables:* selection-matched nulls (retrain on shuffled targets; or run the
same search in random subspaces). *Example:* a future DAS or ReFT run; also "the best direction in the curve's
subspace", or a direction fitted from donor differences, as proposed in `jared/31_goals_next_specs.md` (item 2).

**I8. The search history.** Which layers, sites, strengths and variants were tried before this one was chosen, and on
which prompts. *Why:* picking the best of many settings and then testing it on the same prompts inflates its apparent
effect. *Enables:* a correct null for the selection (a max-statistic null, section 5 P4), or a clean hand-over of
"frozen" settings. *Example:* the E15 scan ran 34 layer × position × direction cells [`DECISIONS.md`, line 144].

**I9. Diagnostics already computed.** *Why:* avoids duplicated work and conflicting numbers. *Example:* `fit.json`
reports the cosine between CAA and a choice direction, class composition, and a held-out check.

### 3.2 Examples: likely hand-overs and what the pipeline can run for each

| Hand-over | Items likely available | What can run | What can't, and why |
|---|---|---|---|
| CAA from `fit.py` | I1 (once the generation hook exists), I2, I3, I4, I5, I9 | All nulls except in-subspace (N3); all confound checks | Calibration only via α_k (CAA is a push method) |
| Manifold curve slide from `fit.py` | I1–I6, I9 | All of the above, plus N3 (random curves), the Δ = 0 `snap` control, calibration in decades (C8), rescale vs replace (C6) | — |
| A direction fitted from donor differences (proposed in `jared/31_goals_next_specs.md`) | I1–I5, I7 | All nulls, plus a selection-matched null | If I7 is missing, its nulls understate the bar |
| A learned subspace (DAS) | I1, I2, I6, I7 | Random-rotation null, shuffled-target retraining, all confound checks | Needs gradients through the model: rented GPU at 8B+ |
| A black-box edit (e.g. from another library, no fit details) | I1, I2 | Isotropic and damage-matched nulls; covariance-matched null from our own activations (I5); all output-based confound checks | Label-shuffled null, within-order refits, held-out-horizon refits |

## 4. What needs agreement, and what the evaluation team decides

### 4.1 Needs agreement between the extraction and evaluation teams

These affect both teams' code or data, so neither side should settle them alone.

1. **The hand-over format.** Which of items I1–I9 are shared for each method, in what form (files, functions), and
   where they live. Proposal: a thin common wrapper (below) that both teams use.
2. **The generation-time edit hook.** Who implements applying an edit inside the model during generation (currently
   missing from `steering/`), so that the same code serves extraction checks and evaluation.
3. **Data splits.** The verification grid stays untouched by fitting and selection; which horizons are reserved for
   held-out calibration (C8); which prompts the extraction team may use for choosing layers and strengths.
4. **Who freezes the operating point, and on which data.** The spec fixes layer 38 in advance and α_k "separately for
   each steering direction" [spec A.3–A.4]; it should be agreed whether the extraction team hands over a frozen
   strength or the evaluation team calibrates it on a separate split.
5. **Search disclosure.** The extraction team records what was tried before a method is handed over (item I8).
6. **Shared random draws.** One set of random directions per (model, layer, site), with seeds and hashes, used by both
   teams, so that "random" means the same thing in every report.
7. **Versioning.** Methods carry the hashes `steering/` already writes, so a report card refers to one exact method.

A possible wrapper (illustrative, not a decided interface):

```python
class Edit:
    layer: int                    # residual stream after block `layer` (steering/ convention)
    positions: str                # "all" | "prompt" | "generated" | a callable
    kind: str                     # "push" (takes a strength) or "target" (takes a shift in decades)
    def __call__(self, h, ctx, x):  # h: [batch, tokens, d]; x: strength or target shift
        ...                       # returns the edited h; ctx: token roles, option order, ...
    fit: "FitInfo | None"         # items I3, I4, I8: prompt ids, labels, refit(...), search log
    structure: "Structure | None" # items I6, I7: basis, curve, learned parameters, objective
```

### 4.2 Decided by the evaluation team

These are internal to evaluation. They should be shared for review, but don't require the extraction team to change
anything.

- The principles every evaluation follows (section 5).
- Which null families run for which method, and K (section 6).
- The statistics: odd/even split, rank test, confidence intervals, multiplicity handling (sections 5–6).
- Readouts: the replay readout for screening, generation for verification, and how their agreement is checked (section 9).
- The confound battery, and which checks are pass criteria vs reported only (section 8).
- The detailed pass rule, within the spec's criteria [spec A.4].
- The report card format (section 10), the implementation, and the compute plan.

## 5. Principles

Rules for how every test is run. Each closes one way to get a false positive or an uninterpretable result.

- **P1. Signed effects, in horizon units.** The headline number is Δ log H*, with its sign [spec A.4]. Counts of flipped
  answers are reported but not used to decide: random edits push models toward their defaults and flip answers near a
  decision boundary in either direction, so a count cannot separate steering from disruption [R5, R6].
- **P2. Push both ways.** Every edit, method or random, is applied at +x and −x. The odd part is the steering; the even
  part is disruption. Only the odd part is compared with the nulls. (Jared's "opposite step" condition is this idea for
  one prompt.)
- **P3. Paired, identical runs.** All arms use the same prompts, batches, batch size, seeds and hardware; only the edit
  differs. GPU arithmetic in bf16 depends on the batch shape, and greedy generation can turn a tiny difference into a
  different trace: re-batching alone changed 12 of 96 labels in E10 [`DECISIONS.md`, line 63], and the E16 noise floor
  was 5 of 24 flips [line 102]. Every evaluation starts with a zero-edit rerun (null N0).
- **P4. Freeze, then test.** Layer, site, strength and α_k are chosen on a calibration split; controls and the
  verification grid run once on the frozen setting [R2]. If a sweep is reported instead, its null is each random
  direction's best result over the same sweep (a max-statistic null).
- **P5. Hold out horizons, not only prompts.** Splits are grouped by prompt configuration [spec A.3]. For targeting
  claims, some horizons are left out of the fit entirely; `fit.py`'s held-out check currently holds out prompts, not
  horizons [`steering/README.md`, "Not here yet"].
- **P6. Scale every effect.** Between a floor (no edit) and ceilings: rewriting the prompt to the target horizon (the R of
  S/R [spec A.5]) and, at a given site, donor patching. Plus a practical bar: prompting "focus on the long term"
  [spec A.5]; in AxBench, prompting beat every steering method tested [R10].
- **P7. Health is an outcome.** Coherence, refusals, format failures, truncation and reasoning length at every strength;
  broken outputs count as outcomes, not dropped [spec A.5].

## 6. Random baselines

### 6.1 Why several null families

Activations use some directions far more than others; a few dimensions can carry values thousands of times the median
[R11]. A uniformly random ("isotropic") direction puts most of its length where the model barely reads, so it is the
weakest opponent [R7]. In the most careful study found, CAA beat isotropic random directions at 96% of operating points
but label-shuffled ones at only 72% (53% on Llama 3.1) [R2]. Random directions are also not harmless: over 1,000 draws,
random steering raised harmful compliance from 0% to 1–13% [R5]. Most steering papers, including all the curved-manifold
ones reviewed, report no random-direction control at all [R9].

The team's runs so far used 1, 2 or 3 random draws per condition, isotropic or inside the 8-dim span, with no recorded
rationale [`DECISIONS.md`, lines 70, 231]; `fit.py` draws 100 isotropic directions [`steering/README.md`, choice 4].

### 6.2 The families

All are matched to the method's size **token by token** (a method whose size varies over tokens, like the curve slide,
gets random twins with the same size profile), at the same layer, positions and strengths.

| Null | Built how | Rules out | Needs items |
|---|---|---|---|
| N0 zero edit | Rerun with no edit, identical batches | Numerical noise | — |
| N1 isotropic | Uniform random unit directions | "Any push of this size" | I1, I2 |
| N2 covariance-matched | Random directions with the covariance of the activations at the site, or differences of two random prompts' activations [R7] | "Any push along directions the model uses" | I5 |
| N3 in-subspace | Random directions, or random curves of the same length, inside the method's subspace; a curve fitted to shuffled horizons [R12, R20] | "Any move in that subspace", without the horizon order | I6 |
| N4 label-shuffled | The method refitted with horizon labels shuffled within strata (option order, regime) [R2] | Artefacts of the recipe | I4 |
| N5 concept controls | Same recipe for other variables: reward size [spec A.3], chosen option, answer letter; competing targets [R22] | Steering a correlate of horizon | I4 or I5 |
| N6 damage-matched | Random directions rescaled until next-token KL on neutral text matches the method's (±10%) [R2] | "This much disruption", whatever its direction | I1 |

### 6.3 Defaults per method type

| Method type | Nulls | Extra control |
|---|---|---|
| Fixed direction (CAA, probe direction) | N1, N2, N4, N5, N6 | — |
| Curve or subspace edit (curve slide, spline, centroid substitution) | N1, N2, N3, N4, N6 | Δ = 0 run for edits that replace coordinates (`snap`) |
| Transport map (Mean-/Linear-AcT, MiMiC) | N2, N4 (shuffled bins), N6 | — |
| Trained intervention (DAS, ReFT) | Random subspace or rotation [R18]; retraining on shuffled targets | Selection-matched null |
| Black box | N1, N2, N6 | — |

### 6.4 The test and the number of draws

- Each random direction is one draw of a **direction-level statistic**: the mean odd-part shift over all prompts. The
  method is compared with the K draws of each family on the same prompts: p = (1 + b)/(1 + K) [R8].
- The effect's confidence interval comes from a paired bootstrap over prompt clusters [R2].
- With K draws the smallest possible p is 1/(1 + K): with 3 draws, 0.25, so "beats the 95th percentile" is undefined;
  with 20, the maximum falls below the true 95th percentile 36% of the time (0.95²⁰) [R9].
- Proposal: **K = 99** on the replay readout, **K ≥ 19** on generation. Draws are made once per (model, layer, site),
  saved with seed and hash, and shared across methods.
- **Selection-matched nulls.** If a method searched for its direction, the null searches too: retrain on shuffled
  targets, or run the same search in random subspaces of the same size. When the effect is linear in the direction, the
  best direction inside a set of K measured random directions V with effects y has effect √(yᵀ(VVᵀ)⁻¹y), so this null
  needs no extra model runs.

Schematic: [`explainer/figs/f4_rank_test.png`] shows three rows of 99 dots (one per random direction) with each row's
95th percentile and a vertical line for the method.

## 7. Comparison methods and bounds

### 7.1 Why these

Across the literature, every intervention that moved a curved or ordinal feature to a chosen value **wrote the target
value's coordinates** (a target mean, a point on a fitted curve or circle, a donor's subspace coordinates) rather than
adding a fixed vector [R12, R13, R20, R21; summary in R9]. The team's results fit this: donor patching at the number
tokens moved the written horizon in 24 of 24 pairs, the fitted manifold step in 3 of 24 [`DECISIONS.md`, lines 19, 102].
For a many-level concept, a method should also be able to aim at a value. The arms below can aim, fit the data size
(about 10 prompts per horizon), and except DAS are closed-form edits from the per-horizon means and their PCs.

| Arm | The edit | Question it answers | Source |
|---|---|---|---|
| Centroid substitution (full, and top-k PCs) | h − μ_source + μ_target, optionally projected on the top k PCs of the means | Does writing the target's average state steer, and in how many dimensions? | [R12, R21] |
| Spline replacement | Replace the coordinates in 32–64 PCs with the spline point at the target; keep the rest. Matched linear twin: the straight chord | Was the curve the wrong object, or the slide the wrong implementation? | [R13] |
| Mean-AcT vs Linear-AcT; MiMiC in PCs | Per-dimension shift; shift + scale (1-D optimal transport); mean + covariance matching | Must the edit depend on the input, not just shift it? | [R15, R16] |
| Slerp to the target mean | Rotate h toward the target horizon's mean direction, keeping its length | Did changing the activation's size sink the slide? | [R17] |
| Boundless DAS, then mean patch | Learn a small subspace where swapping in a donor's coordinates reproduces the donor; write the target horizon's mean coordinates there | Smallest subspace that carries the horizon causally (rented GPU at 8B+) | [R18] |

Not shortlisted: methods built for binary concepts that can't aim at a value or need thousands of examples or large
trained networks (Curveball, INNSteer, FLAS) [R9]. If closed-form curve methods fail, a geodesic under a metric learned
from the model's outputs is the stretch option [R14].

### 7.2 Bounds and baselines

- **Floor:** zero edit (N0).
- **Ceiling, behaviour:** rewrite the prompt with the target horizon: the R of S/R [spec A.5].
- **Ceiling, site:** donor patch of all activations at the site; the most any edit at that site can be expected to do.
- **Practical bar:** prompting "focus on the long term" [spec A.5; R10].

Two cheap diagnostics for a fixed direction that fails: re-express it in the unembedding's "causal inner product" [R23],
and split it into the parts the next layer can and cannot read, patching each separately [R24].

## 8. The confound battery

A method can beat every null and still not steer the horizon: other variables in these prompts move the choice the same
way a longer horizon does. Each check names one, says why it is plausible here, and gives the test and the pass rule.

### C1. Answer letter and option position

- **Why plausible.** Directions are extracted at the answer-label token, which is the letter itself [spec A.3]. If the
  long option is labelled `a)` more often than `b)` within a CAA class, CAA carries part of "a)". Surface features of
  this kind moved choices in our phrasing study: on Qwen3-14B, listing the short option first raised P(short) by 0.16
  and `1)/2)` labels by 0.15 (both about 0 on Qwen3-32B) [`phrasing_study/explainer-32b/`, branch `phrasing-study`];
  the bank randomises order and labels for this reason [spec, Methods]. Averaging over both orders can make a letter
  direction look inert while it flips many answers.
- **Test.** Measure every effect separately on prompts where far = `a)` and far = `b)` (the verification grid has each
  item in both orders). Report **meaning** = mean of the two (push toward far whatever its letter) and **letter** = half
  their difference (push toward the same letter). Same split for label styles. Also: letter balance inside each CAA
  class; cosine between the method and a letter direction (answered `a)` minus answered `b)`); refits within one order.
  For a site in doubt: patch in activations from a prompt with the opposite option order; if the answer follows the
  donor's letter, the site holds the answer, not the horizon.
- **Pass.** Meaning beats the nulls' meaning distribution; letter is small next to meaning. This makes spec criterion 2
  ("same sign in both option orders") quantitative.
- Schematic: [`explainer/figs/f3_meaning_letter.png`]: a horizon steer moves both orders the same way; a letter steer
  moves them in opposite directions.

### C2. Decision vs horizon

- **Why plausible.** CAA's classes differ in the correct answer as well as in horizon: the long class is all `both`
  prompts (far correct), the short class `none` and `near_only` prompts [`steering/README.md`, choice 2]. So CAA is close
  to a "choose far" direction; `fit.py` reports its cosine with a choice direction for this reason. On Qwen3-8B, a
  long-minus-short mean difference steered backwards because of the regime mix [`steering/README.md`, choice 1, citing
  `jared/08_results_steering.md`].
- **Test.** Fit the choice curve, P(far) against log H/D, on the verification grid, steered and unsteered, with a
  logistic that has a location, a slope and two plateaus. A horizon steer moves the location (the switch point); a
  decision bias lifts the low plateau (far chosen even where it can't pay off); disruption lowers the slope. Add the
  chosen-option direction as a concept control (N5) and report results within each regime.
- **Pass.** The location moves; the plateaus stay near 0 and 1; the slope doesn't drop much.
- Schematic: [`explainer/figs/f1_signatures.png`]: the three signatures.

### C3. Reward sensitivity

- **Why plausible.** Choosing far also means choosing the bigger reward (2–50× the near one [spec A.1]). A direction that
  increases sensitivity to reward size looks like a longer horizon on most prompts.
- **Test.** On prompts where both options pay off within the horizon, find the reward multiplier at which the choice
  switches; it must not move [spec A.4, criterion 3]. The reward-contrast direction is the matched concept control;
  results are broken down by multiplier.
- **Pass.** As spec criterion 3, and the method beats the reward direction [criterion 1].

### C4. Disruption

- **Why plausible.** Disturbed models fall back on defaults and flip borderline answers either way [R6]; with thinking on,
  disturbed reasoning can also run long, truncate or lose the format.
- **Test.** The even part (P2); the slope of the choice curve (C2); next-token KL on neutral text; health metrics and
  reasoning length per strength (P7); the damage-matched null (N6) [R2, R25].
- **Pass.** The odd part beats N6; health stays within agreed bounds at the operating strength.

### C5. Surface form

- **Why plausible.** The horizon arrives as a few tokens; a method can latch onto the unit word or the wording of the
  CONSTRAINT line rather than the duration. In Jared's E16, the number tokens carried the number (24/24) and the unit
  token did not (0/24) [`DECISIONS.md`, line 102].
- **Test.** Fit on part of the bank, evaluate on another: across the four CONSTRAINT phrasings, unit rewrites and label
  styles [spec A.1], in Δ log H* units.
- **Pass.** At least 0.8 of the in-template effect on template variants, as the spec requires before realistic prompts
  [spec A.6, step 4].

### C6. Rescale vs replace

- **Why plausible.** "Act as if every horizon were 2× longer" and "act as if the horizon were 1 year" can give the same
  average shift on a grid, but are different interventions. The curve slide is built as a rescaling (both modes move by
  Δ decades from t̂); centroid substitution, spline replacement and DAS mean patching are written as replacements (move
  to horizon k). The model need not respond as designed.
- **Test.** Plot the realised Δ log H* against the stated horizon across the grid: flat for rescaling, falling through
  zero at the target for replacing. On null-control prompts (no CONSTRAINT line), a rescaling edit has nothing to
  rescale; a replacing one still imposes its value.
- **Pass.** Not a pass criterion: a description of what the method does, needed to interpret calibration (C8) and
  transfer.
- Schematic: [`explainer/figs/f2_rescale_replace.png`].

### C7. The stated horizon

- **Why plausible.** A method can change the choice through a side route (C2, C3) while the model still "believes" the
  stated horizon. With thinking on, the reasoning usually restates the horizon. The team already reads written horizons
  from traces (E14: 47 of 128 moved, 16 landed on target [`DECISIONS.md`, line 19]); the phrasing study had a "name the
  horizon" protocol [`phrasing_study/`].
- **Test.** On steered generations: (1) extract the horizon the reasoning uses (LLM judge validated on hand labels, as for
  coherence [spec A.5]); (2) ask a short follow-up, "What time horizon were you planning for?", with the edit still on.
- **Pass.** The stated horizon moves in the same direction as the choice, by a consistent amount; few answers contradict
  their own reasoning.

### C8. Calibration

- **Why plausible.** Hypothesis 2 claims manifold steering is more *accurate*, not only stronger [spec, Hypotheses]. A
  method can push the right way without control of how far. No paper has reported requested-vs-achieved values for a
  many-level ordinal concept [R9].
- **Test.** For targeting methods: request shifts of several sizes, including toward horizons left out of the fit (P5);
  report the calibration slope (achieved over requested), exact-bin and ±1-bin hit rates, and whether small requests give
  proportionally small effects [spec A.5]. For push methods, α_k plays the role of the request. Compare with the model's
  own response to prompt rewrites, which need not be monotone, rather than an ideal straight line.
- **Pass.** For H2: slope and hit rates better than the linear method's at matched norm [spec, Methods].

### Also checked

- **Positions.** The spec edits all tokens and checks "generated tokens only" [spec A.3]; add "prompt tokens only".
- **Selection.** Effects are reported on the fixed verification grid, never on items chosen afterwards for being
  flippable.
- **Readout agreement.** The replay readout is used for screening only after its agreement with generation is measured on
  a subset (section 9).

## 9. Stages and cost

Generation with thinking on Gemma 4 31B took about 13 s per prompt (A100, batch 4) [`prompt_bank/README.md`], so the full
random battery cannot run on generation. The cheap replay readout is used for screening and nulls; generation is saved
for the frozen setting.

| Stage | What runs | Readout | Size |
|---|---|---|---|
| 0 Sanity | Zero-edit reruns in the exact batch setup; replay margin check (exists in `extract.py`); unsteered health | both | 1–2 passes over the grid |
| 1 Where (optional) | Donor-patch ceiling and opposite-order donor test per candidate site × layer | replay | 2 passes per pair |
| 2 Screen | Strength sweep, α_k; nulls N1–N6 at K = 99 | replay | ~600 directions × few strengths × 200 prompts |
| 3 Freeze | Record layer, site, strength / α_k per method | — | — |
| 4 Confounds | C1–C6 and C8 at the frozen setting; readout-agreement check on a subset | replay (+ generation subset) | a few × the grid |
| 5 Verify | Switch-point shift in both orders vs ≥ 19 random per key family, reward control, prompting baseline; C7; health | generation | ~25 arms × 200 prompts |
| 6 Transfer | Template variants, then T1 → T3, α_k frozen [spec A.6] | generation | per tier |

Limitation of the replay readout: it measures the edit's effect on the final decision given the model's original
reasoning, not effects that act through new reasoning. Hence the agreement check, and verification on generation.

## 10. The report card

| Block | Contents |
|---|---|
| Identity | Method, items available (I1–I9), layer, positions, frozen strength / α_k, data and code hashes |
| Effect | Δ log H* with prompt-cluster bootstrap CI; S/R; vs prompting baseline; noise floor |
| Nulls | Per family run: K, 95th percentile, rank p; families not run, and why |
| Confounds | C1 meaning/letter; C2 location, slope, plateaus; C3 reward-switch shift; C4 even part, KL; C5 transfer ratios; C6 shift vs stated horizon; C7 stated-horizon shift; C8 calibration slope, hit rates |
| Health | Coherence, refusal, format failure, truncation, reasoning length, per strength |
| Verdict | Pass / fail per spec criterion [spec A.4]; which confound checks passed |

## 11. Open decisions

**With the extraction team** (section 4.1): the hand-over format and wrapper; who builds the generation-time hook; the
data splits and reserved horizons; who freezes the operating point; search disclosure; shared random draws.

**Within the evaluation team** (section 4.2):

1. Pass rule. Proposal: beat N2 and N4 (not only N1) at p ≤ 0.05 on generation, on the meaning part (C1), and beat the
   reward direction [spec A.4].
2. Strata for label shuffling (N4): within option order at least; within regime too?
3. Which confound checks gate a pass. Proposal: C1–C3 gate; C4–C8 reported.
4. Whether the replay readout is acceptable for screening, given the agreement check.
5. Damage bounds for C4 (which coherence and format-failure rates are acceptable).

## 12. References

Papers (arXiv IDs checked against abstract pages during the literature review; see
`reports/Nonlinear steering methods.md` and its notes in the authors' workspace for details):

- [R1] Rimsky (Panickssery) et al. 2023, *Steering Llama 2 via Contrastive Activation Addition*, arXiv:2312.06681.
- [R2] Raj et al. 2026, *How Far Can You Steer?*, arXiv:2611.06001. Three null families (isotropic, label-shuffled,
  KL-matched), 20 draws each at frozen operating points.
- [R3] Turner et al. 2023, *Activation Addition (ActAdd)*, arXiv:2308.10248.
- [R4] Rios-Sialer et al. 2026, prior work on horizon geometry, arXiv:2606.05194.
- [R5] Korznikov et al. 2025, random steering directions are not inert, arXiv:2509.22067.
- [R6] Malla et al. 2026, random pushes converge and reproduce side-effect rankings, arXiv:2609.06951.
- [R7] Lee & Heimersheim 2024, covariance-matched random perturbations, arXiv:2410.12555.
- [R8] Phipson & Smyth 2010, permutation p-values should never be zero, arXiv:1603.05766 (seen via secondary sources).
- [R9] Our literature review: `reports/Nonlinear steering methods.md` (Augusto's workspace; summary of controls and
  methods, with every citation).
- [R10] Wu et al. 2025, *AxBench*, arXiv:2501.17148.
- [R11] Sun et al. 2024, massive activations, arXiv:2402.17762.
- [R12] Engels et al. 2024, *Not all language model features are linear*, arXiv:2405.14860.
- [R13] Wurgaft et al. 2026, manifold steering with splines, arXiv:2605.05115.
- [R14] Oozeer et al. 2026, Riemannian steering, arXiv:2605.24942.
- [R15] Rodriguez et al. 2024, *Controlling language and diffusion models by transporting activations* (Linear-AcT),
  arXiv:2410.23054.
- [R16] Singh et al. 2024, *MiMiC*, arXiv:2402.09631.
- [R17] You et al. 2026, spherical steering, arXiv:2602.08169.
- [R18] Wu et al. 2023, *Boundless DAS*, arXiv:2305.08809.
- [R19] Wu et al. 2024, *ReFT / LoReFT*, arXiv:2404.03592.
- [R20] Kantamneni & Tegmark 2025, number helices, arXiv:2502.00873.
- [R21] Gurnee et al. 2025 (Anthropic), line-break counting, transformer-circuits.pub/2025/linebreaks.
- [R22] Xu & Sun 2026, competing-target controls, arXiv:2610.19359.
- [R23] Park et al. 2023, *The linear representation hypothesis*, arXiv:2311.03658.
- [R24] Makelov et al. 2023, subspace activation patching illusions, arXiv:2311.17030.
- [R25] Riegler et al. 2026, distortion-matched random controls, arXiv:2605.03160.

Team documents in this repository:

- Experiment spec: `jared/pipeline/EXPERIMENT_SPEC.md` (branch `pipeline`; from the midterm report).
- Extraction code and its choices: `steering/README.md` (branch `steering-vectors`).
- Prompt bank: `prompt_bank/README.md` (branch `prompt-bank`).
- Earlier steering experiments and open decisions: `jared/pipeline/s2_steering/DECISIONS.md` (`main`).
- Phrasing study (surface-feature effects, Qwen3-14B and 32B): `phrasing_study/` (branch `phrasing-study`).
