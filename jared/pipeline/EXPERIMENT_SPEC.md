# Experiment spec: Planning Temporal Manifolds

Copied from the midterm report (`report.pdf`, SPAR Fall 2026, compiled 2026-11-02):
section 3 (Hypotheses), section 4 (Methods) and Appendix A (Experimental Details and
Roadmap). Appendix A is the full spec; sections 3–4 summarise it. Not to be confused with
`SPEC.md` in this folder, which is the contract for the code reorganisation.

## Hypotheses

1. **Steering vectors obtained from long-term prompt responses should reproduce a long-term
   preference in other contexts.** Possible evidence includes using activation vectors from one
   task to steer in another, transferring the steering effect. Other comparisons, such as high
   cosine similarity between vectors extracted from different tasks, could also support this.

2. **Steering along a nonlinear temporal manifold shifts temporal behaviour more accurately than
   linear techniques.** Following previous work, we believe modelling temporal preferences using
   manifolds will yield clearer behavioural shifts than traditional steering counterparts. Evidence
   here is based on comparing the effect on model answers after applying steering over the fitted
   temporal manifold. Furthermore, confirming that small changes in position along the fitted
   manifold produce proportionally small behavioural changes supports this.

## Methods

**Tasks and data.** All prompts are investment decisions, at increasing levels of realism.
Structured prompts present a templated binary choice between a near and a far option, each of
which pays off within a stated planning horizon or doesn't. The correct answer is known whenever
at least one option pays off: the near one if only it does, the larger far one if both do. When
neither does (every horizon under 1 week, and longer ones up to the near option's delay), no
answer is correct; these prompts are used for extraction but not to score behaviour, which is
verified only on prompts near the boundary where the answer flips. Horizons span 48 log-spaced
values from 30 seconds to 500 years. Realistic prompts move toward how human users ask for advice
in everyday model usage. First, they state the horizon in free-text scenarios ("I need the money
for a house deposit in 2 years"). Then, the horizon is implied rather than explicit ("I'm retiring
next spring"). Finally, they ask for open-ended answers such as a portfolio allocation or a bond
duration. Based on early experimental results, option order and labels are randomized throughout.
Experiments run on open-weight reasoning models from the Qwen3 and Gemma 4 families, with
Gemma 4 31B as the headline model.

**Steering methods.** We record activations at the answer position and average them per horizon.
A linear direction (CAA) is the mean of long-horizon activations minus the mean of short-horizon
ones. A nonlinear method fits a smooth manifold curve through the per-horizon means in a
low-dimensional principal subspace. It steers by sliding an activation along this curve by a
target shift in log-time, so the activation stays on the manifold. Controls are random directions
and a reward-contrast direction, both matched in norm.

**Verification.** On held-out structured prompts, each method must shift the horizon the model
acts on, read from where its choice switches, in both option orders and beyond the controls.

**Linear vs manifold (H2).** At matched norms, we compare how closely the realized horizon shift
tracks the intended one. We also test whether small moves along the curve produce proportionally
small behavioural changes, and how coherence degrades with distance from the manifold.

**Transfer (H1).** We apply vectors fitted on structured prompts to each realism level. We also
measure cosine similarity with vectors extracted at that level. The readout is the horizon of the
recommended action, compared against rewriting the prompt with a longer stated horizon.

We report coherence, refusal and format-failure rates at every strength.

## A. Experimental details and roadmap

### A.1 Structured prompts

- **Bank:** 500 investment prompts (seed 0): 471 state a horizon, 29 don't (null control).
- **Horizon grid** (table below): 48 values evenly spaced in log time from 30 seconds to 500
  years, each 1.53 times the previous (about 5.4 per decade). Each value is written in the largest
  unit in which it is at least 2, rounded to a whole number (two significant figures above 100),
  so no value moves by more than 10% ("84 minutes", "15 months").
- **Options:** the near option pays $1k–100k after 1 week to 1 year. The far option pays the near
  reward × {2, 5, 10, 25, 50} after 2–50 years. Option order is randomized.
- **Regimes:** neither option pays within the horizon, only the near one does, both do, or no
  horizon is given.
- **Template variation:** 4 phrasings of the constraint line (below), unit rewrites (the same
  duration in another unit, such as 23 months as 100 weeks), and two label styles (`a)/b)` and
  `1)/2)`).
- **Null control:** the same prompt with the CONSTRAINT line removed.
- **Verification grid:** a separate set of prompts with matched units, around the boundary where
  the correct answer flips (H/D ∈ [0.35, 2.8], where H is the stated horizon and D is the far
  option's delay). It does not overlap with the extraction prompts.

Sample prompt (H/D = 0.8, correct answer a):

```
SITUATION: Plan for the future of the household.
TASK: You, the head of the household, are tasked to choose the best investment:
 a) 10,000 dollars in 1 year.
 b) 50,000 dollars in 10 years.
OBJECTIVE: Think deeply about which option is preferable.
CONSTRAINT: You must select the option that provides the greatest benefit for
this time horizon: 8 years.
ACTION: Select one of the two options. Provide reasoning.
FORMAT: I choose: <a) or b)>. My reasoning: <1-3 sentences>
```

The four phrasings of the CONSTRAINT line, with `{H}` the rendered horizon:

1. You must select the option that provides the greatest benefit for this time horizon: {H}.
2. Your planning horizon is {H}. Choose the option that is best over that horizon.
3. Only outcomes within the next {H} matter for this decision.
4. Evaluate the two options over a time frame of {H} and pick the better one.

Horizon grid: the 48 stated horizons, grouped by the unit they are written in.

| Unit    | n  | Values                                          |
|---------|----|-------------------------------------------------|
| seconds | 4  | 30, 46, 71, 110                                 |
| minutes | 9  | 3, 4, 6, 10, 15, 23, 36, 55, 84                 |
| hours   | 8  | 2, 3, 5, 8, 12, 18, 28, 43                      |
| days    | 4  | 3, 4, 6, 10                                     |
| weeks   | 4  | 2, 3, 5, 8                                      |
| months  | 6  | 3, 4, 6, 10, 15, 23                             |
| years   | 13 | 3, 5, 7, 11, 16, 25, 39, 59, 91, 140, 210, 330, 500 |

### A.2 Realism tiers (all investment)

- T1–T2: about 200 items each, generated with Haiku 5.5, checked by GPT-5.6 Luna, with a sample
  hand-audited.
- Investment risk is crossed with horizon, so that "long" does not always mean "safe".
- Every option in T1–T2 carries a duration h_i, which the expected-horizon readout uses (A.5).

| Tier            | Horizon cue              | Answer                | Readout                                       | Example |
|-----------------|--------------------------|-----------------------|-----------------------------------------------|---------|
| T0 Structured   | Explicit constraint line | a/b                   | Switch point                                  | A.1 |
| T1 Free-text    | Explicit, in prose       | a/b                   | Choice probability                            | "I need the money for a house deposit in 2 years. 2-year CD or 20-year index fund?" |
| T2 Implicit     | Inferred                 | a/b, then 4/8 options | Expected horizon                              | "This is for my grandkids' education." |
| T3 Open-ended   | Inferred                 | Free-form             | Scalar from answer (equity share, bond duration) | "How should I allocate my savings? I'm retiring next spring." |

Sample prompts for T1–T3 follow. They were written by us to show the format; the real items are
generated as described above.

T1, free-text (stated horizon 2 years; expected answer a):

```
I'm saving for a house deposit and I'll need the money in 2 years. I have
$30,000 to put away. Which is the better choice for me?
 a) A 2-year certificate of deposit paying 4.5% a year.
 b) A global stock index fund, held for 20 years.
Start your answer with a) or b), then explain briefly.
```

T2, implicit, two options (horizon implied: about 18 years; expected answer b):

```
My daughter was born last week, and I want to set aside $10,000 for her
university education. Which should I choose?
 a) A 1-year high-yield savings account, renewed each year.
 b) A global stock index fund, held until she starts university.
Start your answer with a) or b), then explain briefly.
```

T2, implicit, four options (horizon implied: under 1 year; h_i = 1 month, 1 year, 5 years,
20 years):

```
We're getting married next summer, and we want the $15,000 we've saved for the
wedding to grow a little until then. Which option fits best?
 a) A high-yield savings account (withdraw at any time).
 b) A 1-year certificate of deposit.
 c) A 5-year bond fund.
 d) A stock index fund, held for 20 years.
Start your answer with a), b), c) or d), then explain briefly.
```

T3, open-ended (horizon implied; readout: equity share):

```
I'm 64 and I'm retiring next spring. I have $400,000 in savings. How should I
split it between stocks, bonds and cash? Please give percentages.
```

### A.3 Activations and steering

- **Extraction site:** residual stream at the answer-label token, at a layer fixed before
  evaluation. We will choose the layers based on results from prior work, locating the causal
  effect peaks at 0.61–0.69 of depth: layer 38 of 60 on Gemma 4 31B (the headline model, where the
  horizon geometry also peaks) and L24 on Qwen3-8B (the pilot).
- **CAA:** mean of long-horizon activations (H/D ≥ 4) minus mean of short-horizon ones
  (H/D ≤ 1/4), normalized. Strength is set as a fraction of the typical activation norm at that
  layer.
- **Manifold curve:** PCA on the per-horizon mean activations (top 3–8 components). We fit a
  smooth curve through them, indexed by log-time t. To steer, each token is projected onto the
  curve (t̂) and moved to the curve point at t̂ + Δ (clamped to the observed range), leaving the
  rest of the activation unchanged. Δ is measured in decades.
- **Positions:** all tokens, as in prior work. Steering only generated tokens is run as a check.
- **Statistics:** repeated k-fold cross-validation grouped by prompt configuration, so paraphrases
  of one item never fall on both sides of a split.

### A.4 Verification and calibration

- **Switch point s:** the stated H/D at which the model's choice flips from the near option to the
  far one. The acted-on horizon is H* = H/s, so the steering effect is Δ log H* = −Δ log s.
- **Pass criteria:**
  1. The shift in Δ log H* exceeds the 95th percentile of the random directions and also exceeds
     the reward-contrast direction.
  2. The shift has the same sign in both option orders.
  3. Steering does not move the model's reward-indifference point on prompts where both options
     pay off within the horizon.
- **Calibration:** α_k is the strength at which the model acts on a horizon k times longer
  (default k = 2). It is fixed here, separately for each steering direction, and never retuned on
  realistic prompts.

### A.5 Metrics

- **Expected horizon:** E[log h] = Σ_i p_i log h_i over the options. This is graded and works for
  2, 4 and 8 options.
- **Transfer efficiency:** S/R, with a cluster-bootstrap 95% CI. S is the shift produced by
  steering at α_k. R is the shift produced by rewriting the prompt with a horizon k times longer.
- **H2 precision:** realized vs intended shift at matched norm. We also measure whether behavioural
  change is monotonic in Δ, and coherence as a function of distance to the curve.
- **Health:** coherence (LLM judge, validated on ~50 hand labels), refusal, format failure and
  truncation at every strength. Broken outputs are counted as outcomes, not dropped.
- **Baseline:** prompting the model to "focus on the long term".

### A.6 Roadmap

1. Pipeline: debug locally on a small Qwen3.
2. Extract and verify: gate and calibration on structured prompts. If the gate fails, stop before
   building realistic data.
3. H2: compare linear and manifold steering on held-out horizons.
4. H1: apply to T0 template variants (gate: ≥ 0.8 of the in-template effect), then
   T1 → T2 → T3.
5. Replicate: second model family, same rules, no new sweep.
6. Stretch: multi-turn robustness (steer only the reply to "are you sure?"); duration outside
   planning ("the movie lasted 3 hours").

**Models:** the headline model is Gemma 4 31B (instruction-tuned, `google/gemma-4-31B-it`; 60
layers, steered at layer 38). In a pilot with thinking on, it chose correctly in all 16 unit
combinations tested, showed no option-order bias and never hit the 8,192-token cap. Replication:
Qwen3-8B, Qwen3-14B, Qwen3.5-27B and Qwen3-32B. Thinking is on, with a token cap.
