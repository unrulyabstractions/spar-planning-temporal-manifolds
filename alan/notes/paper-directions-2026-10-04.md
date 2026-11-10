# Paper directions — draft for group discussion (Alan, 2026-10-04)

Produced with two independent reviewers (Claude and Astra via Codex, two rounds; reviews/astra-20261004/). Every claim
below was checked against the progress logs and code; numbers are quoted from the record, nothing was recomputed.
Mentor interests on record (2026-10-01 meeting): using the horizon for monitoring; quantifying the manifold geometry;
longer term, intrinsic manifold coordinates. Open design choices: single- vs multi-turn, thinking vs non-thinking.

## Where the week-1 hypotheses stand (notes/update-2026-09-23.md)

| | Hypothesis | Status | Key evidence |
|---|---|---|---|
| H1 | Along-path coordinate is magnitude; unit effects on behaviour are off-path | Magnitude part supported; behavioural part reframed; mediator untested | Paired unit rewrites: ridge transfer R² 0.78–0.89, < 1 grid step along the path (09-22). Crossed years/months × delay-unit design: short-choice rate 0.99 when horizon and option-delay units match vs 0.73–0.78 when not (10-03, Qwen3-14B, one scenario). |
| H2 | ≈ 2-D response with a Guttman arch (revised 09-29) | Revised claim supported; "PC3 not cubic" not universal | Vertex tracks the range midpoint (slope 0.82–1.47 across four models and two grids); PC1 share above a banded null. Exceptions: 14B L29 R0 PC3 cubic 0.57; 27B L31 T6 0.72 (null 0.57–0.79). |
| H3 | The decision reads the coordinate | Narrow predictive result only | Prompt-parameter interactions log-loss 0.231 vs 0.149 with PCs (09-23). The degree-3 kill test was never run; no causal test. |
| H4 | Extremes over-resolved | Unrestricted claim fails | Extended grid saturates (spacing 1.3 for 1000→3000 y, 0.3 for 3000→10000 y at L22 R0; 09-22). |

## Three options

### A. Readout reliability and behavioural relevance (monitoring)
- **Supported now:** across Qwen3-14B, Qwen3.5-27B, Qwen3.8-27B and Gemma 4 31B, a readout averaged over the model's
  explanation tokens (MEAN) recovers the assigned horizon with low sensitivity to irrelevant durations (Qwen3.8 slope
  +0.030–0.042, within-2× 0.81–0.87; Gemma −0.004–0.015, 0.84–0.91), and transition-token probes trained with
  distractor prompts become robust too (slopes ≤ 0.05 at most cells) and generalise to unseen distractor families.
- **Caveats:** MEAN is computed over text written after the answer ("I choose: a). My reasoning: …"), so it cannot
  show what drove that answer. A regex on the prompt reads explicit horizons at 1.00 within-2× on every explicit suite
  tested (09-25), so monitoring an explicitly stated horizon needs a use beyond reading the instruction.
- **Hypothesis / kill:** MEAN's robustness survives controls for pooling, output length and semantic restatement, and a
  pre-decision readout predicts an independent subsequent choice beyond the instruction and prompt parameters. Kill the
  late-readout explanation if the advantage disappears under those controls; kill the monitoring extension if it adds
  no held-out predictive value (a narrower readout-reliability result would remain).
- **Decisive next experiment (local, Qwen3-14B):** cross answer-only / answer-then-explanation / explanation-then-answer
  formats with equal-length pooling and a semantic restatement audit; add a small continuation battery that reads
  activations before new choices and tests prediction on withheld decisions.
- **Fit:** monitoring. Overlaps Augusto's relevance-aware decoder; needs a division of ownership.

### B. Multiscale temporal geometry
- **Supported now (Qwen3-14B, one fixed scenario):** a smooth curve in log horizon carries 58–89 % of the variance of a
  dense horizon sweep; the remainder is high-dimensional and partly tied to the rendered digits. Horizon is more
  curve-dominated than a matched reward amount (spline R² +0.20 to +0.27). Each time unit forms its own neighbourhoods
  (95–100 % of nearest neighbours share the unit), mostly a translated copy of one curve. Shared-curve approximations
  depend on site and model (Gemma T3: 0.359 shared-curve + offsets vs 0.466 per-style curves).
- **Caveats:** dense-set results are one model and one scenario; local dimension estimates depend on sampling density
  and neighbourhood size; horizon (a task-wide constraint) and an option's reward play different roles.
- **Hypothesis / kill:** the smooth component and unit-associated neighbourhood structure reproduce across independent
  scenarios after matching sampling and number format, and a matched non-temporal global constraint stays less smooth.
  Kill the generalisation if independent scenarios fail; kill temporal specificity if role- and format-matching removes
  the difference.
- **Decisive next experiment (local, Qwen3-14B):** a compact crossed design over new option configurations, label
  orders and prose formats with matched unit renderings and a non-temporal global constraint; held-out smooth fits and
  neighbourhood structure across sampling densities. Intrinsic coordinates as an evaluated extension (learned on
  training contexts, tested against ridge on held-out ones).
- **Fit:** quantifying manifold geometry; Shantanu's intrinsic-coordinate interest.

### C. Unit compatibility and decisions
- **Supported now (Qwen3-14B, one crossed years/months design):** matching the horizon's unit to the option delays'
  unit changes choices (0.99 vs 0.73–0.78 short at 3–10 y), while the years/months local-geometry ordering persists in
  both conditions — the geometry ordering does not explain the behavioural reversal.
- **Hypothesis / kill:** unit compatibility produces a reproducible choice effect at equivalent durations across
  scenarios, and a pre-answer compatibility feature predicts paired choice changes beyond decoded magnitude. Kill the
  general matching claim if it fails across configurations, option order or paraphrases; kill a mediator claim if the
  feature does not predict held-out paired changes.
- **Decisive next experiment (local, Qwen3-14B):** replicate the crossing across reward/delay configurations with
  conversion controls; fit magnitude and compatibility features on training scenarios and predict held-out paired
  changes; only then pre-answer patching with positive and unrelated-feature controls.
- **Fit:** mapping behaviour to geometry; possible circuit-tracing extension.

## Is more analysis needed before proposing these?
No. The shelved peak-cell reruns for Qwen3.8/Gemma, the degree-3 H3 baseline, the semantic restatement audit and further
dimension diagnostics can follow the choice of direction. (Gemma's weak T3 result is already at its T3 peak, L33; only
its R0 peak, L37, differs from the fixed cells.) What should come first are decisions: the monitoring target and
timing, the main contribution, and ownership relative to Augusto and Jared.

## Questions for the group
1. Which contribution should anchor the paper: readout reliability (monitoring), geometric characterisation, or a
   behavioural mechanism?
2. For monitoring, what target and timing matter: the assigned horizon or a behaviourally inferred one; before the
   answer, between decisions, or after a response?
3. Can the main experiments stay single-turn and non-thinking, with a small continuation test where needed?
4. How should ownership divide with Augusto (relevance-aware decoder) and Jared (mid-thinking geometry), and which
   decisive replication should get rented compute?
