# E14 results: steering the horizon at the horizon phrase (Qwen3-8B, thinking on)

2026-11-01. Spec `21_spec_phrase_steering.md`; run `SPAR/results/horizon/exp14_Qwen3-8B_phrase/`.

## Setup in brief

- **The task.** The model chooses between a small payout soon and a large one later, "for this time horizon: 39
  years", and reasons before answering. The horizon's *regime* says which payouts arrive in time:
  - **none:** neither arrives in time;
  - **near:** only the near payout does;
  - **both:** both do.
- **The intervention.** We added a vector to the layer-16 hidden states of the horizon phrase (the number tokens and
  the unit word). It was added on every prefill pass: the pass that starts the reasoning, and again the pass that
  scores the answer. So the vector is present when the choice is read out.
  - Each prompt got a target horizon across one payout delay, in four transitions: none→near, both→near, near→both
    and near→none.
  - A *decade* is a factor of 10 in the horizon.
- **The vectors.**
  - **The steer:** the move from the prompt's horizon to the target along a curve of fitted horizon effects. The curve
    comes from 1,386 prompts, fitted with the unit word, regime, option order and delays as separate terms, so that
    it holds them at their average (the "pristine" fit of `16`).
  - **Invariant part:** the steer minus its component along those terms.
  - **Controls:**
    - a random vector of the same length;
    - the unedited prompt run again (the *rerun*);
    - an *opposite* step, the same number of decades the other way. It was cut short at the curve's ends (30 seconds,
      500 years) on 42 of 64 prompts.
    - a *donor patch*, which copies the phrase states of the same prompt rendered with the target horizon.
- **Measures.**
  - **Stated horizon:** the first duration the trace states as its horizon ("over a 13-year time horizon"). It
    *lands* when it falls in the target regime.
  - **Choice:** the change in near log-odds against the rerun, signed + toward the direction the unsteered model's
    behaviour predicts for the transition.
  - **Scale:** C = 19.2, the unsteered model's mean near log-odds in the near regime minus that in the both regime.
  - **Behaviour map:** a fit of the unsteered model's near log-odds on where each delay sits relative to the horizon:
    a jump at each delay plus slopes (`pipeline/s3_analysis/by_label/behaviour_step.py`). We use it to *predict* the choice change a trace
    should show if the model simply used the horizon it wrote.
  - **Natural close:** the reasoning ended on its own rather than being cut at 3,072 tokens. Cut traces pull near
    log-odds toward 0, so §2 uses natural closes.

## The answer

- **The pre-registered tests fail (H1–H3, §4).**
  - On the 104 prompts with a donor, the steer moved near log-odds +1.4 and the donor patch +9.9.
  - The spec's reading of this outcome: the phrase carries the horizon the reasoning uses, and the steer does not
    reach it.
- **What the steer does reach (post hoc).** It changed the first stated horizon on 55 prompts (steer at L14–18),
  always toward the target. Random vectors never moved it (0 of 384), and the opposite step moved it its own way (all
  11 moved traces on both→near).
  - **Within the prompt's unit the move is about one decade, whatever the step size.** It is often the digit written
    twice: 6 → 66 months, a factor of 11, is the median shift of 1.04 decades.
  - **Larger moves came only from swapping to a longer unit word,** on up-steps from horizons of 4 months or less,
    often keeping the number: 10 minutes → 10 years.
  - So the stated horizon rarely crossed a delay: 16 of 128 traces landed at L16.
- **Where it landed, the choice went with it, on none→near only, and all through unit swaps.**
  - Landed traces (natural closes) moved +11.5, close to the +10.9 the behaviour map predicts for their stated
    horizons, against random +0.1 on the same prompts.
  - At the pre-registered condition alone (steer at L16, all closes) it was +4.9 against a predicted +9.4.
  - Because the steer is present when the choice is read out, this does not show that the written horizon causes the
    choice.
- **A lesson for the fit.** The pristine curve is close to a written-number direction by construction. The unit word
  accounts for 88% of the variance of log duration in the fit's prompts. With the unit as its own term, the curve is
  fitted from the variation within a unit, where duration is the written number.

## 1. What the steer does to the stated horizon

![E14: stated horizon and choice](../../results/horizon/exp14_Qwen3-8B_phrase/summary.png)

At L16, 32 prompts per transition:

| transition | moved toward the target | landed | unit word changed |
|---|---|---|---|
| none → near (up) | 15 | 10 | 11 |
| near → both (up) | 14 | 5 | 2 |
| near → none (down) | 11 | 0 | 0 |
| both → near (down) | 7 | 1 | 0 |
| all | **47 / 128** | **16 / 128** | 13 |
| random vector | 0 / 128 (0 / 384 at L14–18) | 0 | 0 |
| rerun | 0 / 128 | 0 | 0 |

**Within the unit** (steer at L14–18: 46 prompts, 97 traces):
- The median shift is 1.04 decades (IQR 0.70–1.10).
- How far it moves does not grow with the step (r = +0.06 over steps of 1.4–6.6 decades). Whether it moves does:
  0–16% of steps under 2 decades move it, against 54–61% over 2 decades (`geometry.png`, right). Step size also goes
  with transition and starting horizon.
- 58 traces (28 prompts) contain the prompt's digits or are contained in them:
  - 27 write the digit twice (6 → 66, 2 → 22);
  - the rest add or drop digits (213 → 13, 4 → 48);
  - only 6 are an exact factor of 10.
- The other 39 traces are rewrites of 0.1–2.1 decades (7 → 3 years, 11 → 5 years).
- The invariant part made essentially no same-unit moves (1 trace).

**With a new unit word** (steer at L14–18: 15 prompts, 31 traces):
- All are up-steps from horizons of 30 seconds to 4 months: 12 none→near prompts and 3 near→both.
- Steered traces (L14–18 and the invariant part) swapped units 45 times; 21 of the swaps kept the exact number.
- The swaps follow the up-direction. On none→near at L16 the steer swapped in 11 of 32 traces and the invariant part
  in 14. The opposite step swapped in 1; 3 more were the model stating a payout delay as its horizon. Random never
  swapped.
- They may partly repair a horizon the model already doubts. The reruns of these prompts flag a "typo" or "mistake"
  in 9 of 12, against 10 of 20 for other none→near prompts. That is one rerun per prompt, so only suggestive.
- None of the 46 moved down-step traces swapped unit, and 23 of the 32 near→none prompts need a new unit to land. So
  no near→none trace landed.

**Caveats on the measure.**
- **Parser check:** a separate model instance read 80 traces blind to the parser's output (40 of them moved). It
  agreed with the parser on 79; the miss is fixed. Files are in `parser_audit/`.
- **Retracted horizons:** 20% of landed traces later name the prompt's own horizon. The single both→near landing ("91
  years → 3 years") is retracted at once ("Wait, the time horizon mentioned is 91 years?").
- **Delays:** the measure keeps a stated horizon that equals a payout delay. 13 of 128 moved traces state one, and 5
  of them landed.

## 2. What the choice does (natural closes)

Change in near log-odds against the rerun, natural closes in both runs. 95% CIs resample prompts; the number of
prompts is in brackets. "Steered" pools the steer at L14–18 and the invariant part at L16. "Predicted" applies the
behaviour map to the move of the stated horizon alone. The all-traces version is in `summary.txt`.

| transition | random | steered, unmoved | moved, not landed (predicted) | **landed** (predicted) | landed prompts: random / donor | donor, all |
|---|---|---|---|---|---|---|
| none → near | −1.3 [−4.3, 1.6] (25) | −2.0 [−5.1, 1.2] (17) | −9.3 [−20.8, 1.7] (6) (+0.2) | **+11.5 [4.8, 15.4] (13)** (+10.9) | +0.1 / +15.3 | +8.8 [4.7, 12.7] |
| near → both | −0.2 [−1.3, 0.8] (30) | +1.2 [0.0, 2.5] (29) | **+4.3 [2.8, 5.9] (12)** (+4.3) | +6.3 [1.6, 12.8] (6) (+18.9) | −2.0 / 1 prompt | +14.5 [5.6, 23.2] |
| near → none | −0.3 [−1.2, 0.8] (28) | +0.4 [−0.5, 1.3] (28) | +1.1 [−0.4, 2.6] (8) (−2.1) | none | — | +7.8 [5.6, 10.5] |
| both → near | −0.8 [−3.9, 2.4] (22) | +2.9 [−0.4, 6.4] (23) | −0.1 [−10.7, 10.5] (4) | 1 prompt, retracted | — | +15.2 [10.4, 19.8] |

- **Where the up-steps move the stated horizon, the choice matches the map's prediction.**
  - none → near, landed: +11.5 against +10.9.
  - near → both, moved within the near regime: +4.3 against +4.3. The map falls about 5 log-odds per decade there.
  - Within prompts (all closes), landed traces exceed non-landed ones by +9.5 [6.1, 12.5] on none → near (10 prompts)
    and +6.5 [1.8, 12.3] on near → both (9).
- **Where it does not match.**
  - near → both, landed: +6.3 against a predicted +18.9.
  - near → none, moved: +1.1 against a predicted −2.1.
- **The none → near result rests on unit swaps.** All 38 landed traces there are swaps, from the horizons the model
  already doubts. By condition (all closes), the steer gave:

  | condition | n | observed | predicted |
  |---|---|---|---|
  | L14 | 6 | +13.3 | +11.4 |
  | **L16** | **10** | **+4.9** | **+9.4** |
  | L18 | 8 | +12.8 | +11.4 |
  | invariant part | 14 | +10.1 | +10.0 |

- **H2's +1.8 is mostly not landing.** On natural closes H2 is +1.86 [0.26, 3.43]. The 16 landed L16 traces average
  +5.2 and the other 112 +1.3. By transition, the non-landed share of the total is near → both 39%, both → near 26%,
  near → none 14% and none → near −14%.
  - The near → both part is the graded within-regime shift above.
  - The both → near part is not specific to the direction: the layer-matched random step gives +0.0 there and the
    opposite step +2.1.
- **What these comparisons do and do not control.**
  - The same-prompt and within-prompt comparisons control which prompts land, and how far their choice could move.
  - They do not control which traces land, since landing and the choice come from the same generation.
  - They also cannot separate the written horizon from the steer itself, which is present when the answer is scored.

## 3. Geometry of the fit at the phrase

![E14: fit geometry](../../results/horizon/exp14_Qwen3-8B_phrase/geometry.png)

**Method.** For each prompt with a donor we estimate the real change of the phrase state when the horizon text
changes from the prompt's to the donor's:
- it is the difference of the mean states of pooled prompts with each text;
- regime, option order and delays are removed;
- the fit is made on one half of the prompts and the change measured on the other, both ways.

Medians at L16, cross-fitted (104 prompts, 208 half-measurements):

| | unit token | number tokens |
|---|---|---|
| R² of the real change explained by the steer, all donors | 0.02 | −0.05 |
| R² by the steer, the 9 donors that keep the prompt's unit | 0.44 | 0.57 |
| R² by the steer plus the fit's unit term (at the number tokens, a number-size offset at fixed duration) | 0.78 | 0.45 |
| cosine of the steer with the fit's unit term | −0.47 | −0.82 |

- **This describes the vector, not the model's mechanism.**
  - 95 of the 104 donors change unit, and the steer, which holds the unit fixed, misses those changes by
    construction. Within a unit it covers about half of the change.
  - With log duration = log number + log unit size, a feature that reads the written number appears as +curve and
    −unit term; collinearity (η² 0.88) also anti-correlates the two estimates. Either way the steer is close to "same
    unit, another number".
  - It does not explain why moves stop near one decade, nor the swaps.
- **Invariant part.**
  - It is orthogonal to the unit term, yet it swapped units on none → near as often as the steer, and made no
    same-unit moves.
  - So the steer's digit edits came from its component along the fitted terms (the in-span part).
  - The spec's check (`21` §5) remains inconclusive.
- **Probe.** A unit-word probe on the unit-token states (accuracy 1.00, essentially token identity) still reads the
  prompt's own unit after every steer step. The swaps do not cross this probe's boundary.

## 4. Pre-registered verdict (`21` §5)

| criterion | result | bar | |
|---|---|---|---|
| H1: share of traces writing a target-regime duration (steer, L16) | 0.32 vs random 0.20; difference +0.12 [0.05, 0.19] | ≥ 0.30, and random ≤ 0.10 | fail |
| H2: change in near log-odds toward the prediction | +1.80 [0.58, 2.98] | ≥ 2.88 (0.15 C), CI above 0 | fail |
| H3: steer minus opposite step, none and both starts | +0.35 [−2.47, 3.16]; prompts not cut short +1.57 [−1.93, 5.35] (22) | ≥ 2.88, CI above 0 | fail |
| H3: both reverses lower near (near starts) | near→both −4.24 [−6.06, −2.59]; near→none −1.00 [−2.26, 0.19] | both below 0 | fail |

**Reported, not gating:**
- **Other layers:** H2 at L14 is +1.55 and at L18 +2.17.
- **In-span share of the steer at L16:** number tokens 0.73, unit token 0.50.
- **Ratio to the donor:** 0.32 on H1's share, 0.14 on H2's change.
- **H1's measure** counts any duration written in the trace ("that leaves 3 years to grow"). The rerun already scores
  0.25–0.38 on the none and both starts. On the stated-horizon measure of §1, random scores 0 and the steer 0.13, so
  H1 fails either way.

## 5. What this means, and what to run next

**Supported (8B, thinking on, one run):**
- At L16 the phrase carries the horizon the reasoning uses: the donor patch moves the stated horizon on every prompt,
  and the choice with it.
- The steer moves the stated horizon in a signed direction (random never, the opposite step its own way). But it moves
  it about one decade within the unit, often by writing a digit twice, or swaps to a longer unit from short horizons.
  So it rarely crosses a delay.

**Observed (correlational):** where an up-step moved the stated horizon (none → near landed, near → both within the
regime), the choice change matched what the unsteered model does at that horizon.

**Open:**
- what stops the move within a unit at about a digit;
- whether the written horizon, or the steer itself, moves the choice;
- why near → both landings and near → none respond against prediction.

**Next, each separating two readings:**
1. **Magnitude or digit length?** (box, about 2 h)
   - Fix the direction and scale it. Report each step's norm, not only its decades: norm per decade ranges from 9
     to 165 across prompts (median 13 for the steer).
   - Arms:
     - the steer;
     - its in-span part alone;
     - its invariant part alone;
     - the first digit token only;
     - the last digit token only.
   - Score same-unit outcomes only. A magnitude code gives a stated shift that grows with α; a digit-length code
     gives x → xx past a threshold.
2. **Does the written horizon cause the choice?** (box, about 1 h; natural closes)
   - Continue each prompt's own reasoning from a copy whose stated horizon is replaced by the target, with no steer.
     Control: a sham edit that rewrites the prompt's own horizon.
   - Reverse arm: in landed steered traces, restore the prompt's horizon in the text, keep the steer, and see whether
     the choice reverts. This separates the text from the steer's direct effect on the answer pass.

## Appendix

**A. Before E14.**
- **Stage A** (`exp8_Qwen3-8B_pristine/results.json`): steering at the answer slot along the pooled curve, with α = ±1
  or ±2 decades for every prompt.
  - Thinking off, pre-label L24, |α| = 2: the choice moves 0.139 C toward the rational answer; invariant part 0.092.
  - Prompts that cross a delay move more, by +0.060 [0.046, 0.073] C. With step length controlled the crossing term
    is +0.013 [−0.005, 0.032], so the effect is graded (`analysis/crossing_split_pristine.txt`).
  - Thinking on: up to 0.03 at every cell. At L24 the length-controlled crossing term is +0.025 [0.010, 0.040].
- **Stage B** (`exp13_Qwen3-8B_pristine_suffix`): a ±1 decade step at the suffix newline with thinking on. Traces named
  their own horizon in 99–100% of cases, and nothing beat random.
- **Behaviour** (`analysis/behaviour_step.txt`, thinking on, hinge fit within duration):
  - Near log-odds drop by 17.5 (0.91 C) where the far delay falls inside the horizon.
  - With the far delay outside the horizon, they fall by about 5 per decade of longer horizon.
  - In the none regime they fall by about 8 per decade as the horizon shortens.
  - Spec `21` §1 called the slopes flat; it carries a correction.

**B. The run.**
- **Prompts:** 32 per transition, drawn from 122–426 eligible.
- **Fit:** the pooled fit per layer and role (number tokens, unit token) on 1,386 prompts; k = 8.
- **Conditions:**
  - rerun;
  - steer at L14, L16 and L18;
  - random step at the same three layers;
  - invariant part at L16;
  - donor patch at L16, with donors found for 25, 32, 15 and 32 prompts (both → near, none → near, near → both,
    near → none);
  - opposite step for none and both starts.
- **Size:** 38 conditions, 1,192 generations, about 4 h on one RTX 5090 at batch 12.
- **Crash:** a crash on a one-prompt batch was fixed and tested, and the run was repeated.

**C. Other observations.**
- **Label churn:** on the none and both starts, any edit flips 22–50% of labels against the rerun, random included.
- **Rerun choices:** the model picks near on 0.53 of none → near prompts and 0.38 of both → near prompts.
- **Forced closes:**
  - The steer at L16 raises them on both → near (0.56 vs a rerun's 0.25 and random's 0.28) and none → near (0.44 vs
    0.16 and 0.28).
  - Landed traces are 31% forced, against 0% for their reruns.

**D. Reproduce** (CPU, saved results; the geometry needs `states.npz` and `steering/phrase_prompts_Qwen3-8B.json`;
the predictions need `analysis/behaviour_step.json`):

```
python -m pipeline.s3_analysis.by_label.exp14_analysis  /home/Jared/SPAR/results/horizon/exp14_Qwen3-8B_phrase   # §1, §2, §4: summary.{txt,json,png}
python -m pipeline.s1_data_vectors.analysis.exp14_geometry  /home/Jared/SPAR/results/horizon/exp14_Qwen3-8B_phrase   # §1, §3: geometry.{txt,json,png}
```

§1–§3 are post hoc (added 2026-11-01) and gate nothing. The pre-registered criteria are in §4, unchanged.
