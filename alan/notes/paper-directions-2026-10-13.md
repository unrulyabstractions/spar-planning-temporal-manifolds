# Paper hypotheses — draft for the 2026-10-13 meeting (Alan, 2026-10-11)

Supersedes notes/paper-directions-2026-10-04.md for discussion purposes (that note is kept as the record of the 10-04
proposal). Written in the format Ian asked for on 10-08: for each hypothesis, the evidence to gather and evaluate and how
confounds are handled. Numbers are quoted from the progress logs and the committed per-model reports; the one number
computed for this note (the factor table below) is read from `figures/<model>_unit_boundary_n24/unit_boundary.md`,
produced by `scripts/unit_boundary.py`.

## What changed since 10-04

- **Unit compatibility replicated across models (10-07/08).** 18,128 prompts, behaviour only: horizon unit × delay unit
  (weeks / months / quarters / years) × long delay D ∈ {2, 5, 10} y × reward ratio {5, 20} × option order × 4 phrasings
  × 24 horizons at H/D ∈ [1/1.5, 1.5]. Short = $10,000 in 1 year; the short option is correct iff H < D. Format
  adherence 1.000 in all three models. Below-boundary violation rate (long chosen though H < D), matched / mismatched:

  | model | short option first | long option first |
  |---|---|---|
  | Qwen3-14B | 0.300 / 0.717 | 0.912 / 0.968 |
  | Qwen3.8-27B | 0.011 / 0.254 | 0.212 / 0.655 |
  | Gemma 4 31B-it | 0.100 / 0.467 | 0.086 / 0.604 |

  Mismatched violations exceed matched at every level of every factor (D, ratio, order, phrasing, horizon unit, delay
  unit; 19 levels) in all three models. Smallest gaps: Qwen3-14B long first 0.912 / 0.968; Qwen3.8 phrasing 2
  0.002 / 0.242; Gemma phrasing 2 0.000 / 0.272.
- **The 10-03 headline was re-read (10-07).** "0.99 vs 0.73–0.78 short at 3–10 y" was a bin average; the effect is a
  boundary shift (Qwen3-14B switches at 10.00 y with matched units, 8.37 / 8.09 y mismatched). It reproduces in the new
  design and is the mildest corner of the effect (months ↔ years).
- **10-08 meeting:** Ian — evaluate Qwen3-14B in thinking mode or stop using it; try thinking mode on the larger models
  to see whether they resolve mismatched units. Shantanu — average activations over the two option orders.
- **Thinking-mode prep (10-11):** the answer parser now reads only text after the model closes its thinking (commit
  dff9d8f, tested on 6 tokenizers); a 384-prompt Qwen3-14B pilot set pairs prompt-for-prompt with the non-thinking run.
  Not yet run (local GPU driver mismatch, reboot pending).
- **Team ownership (Actionables tab):** Alan — unit compatibility; Jared — prompt types; Augusto — phrasings;
  Soham — beyond PCA / PLS / Guttman, intrinsic dimension. Readout reliability is unassigned.
- **Comments added to the brainstorming doc since 10-06** (authors not identifiable from the text export): Monitors —
  "End goal: having monitor for temporal-related misalignment (e.g myopia or stalling)"; "Maybe we can't pull out the
  time horizon, but something weaker: indication that it changed". Geometry — "Okay, nice plot, but what's the point?"
  Stakes — "Not so promising as main research direction: more like 'future work'". Implicit horizons — "Not basing our
  work on this; not the core!"

## Proposed anchor: unit compatibility and decisions

### C1 (behaviour). Models apply an explicit horizon constraint reliably only when the horizon and the option delays are stated in the same unit.

- **Evidence so far:** the table above; three models, two families. Qwen3.8 and Gemma follow the constraint well with
  matched units in all four units (matched switch H/D: Qwen3.8 0.95–1.22, Gemma 1.01–1.04 for weeks/months/quarters,
  0.88 for years) and lose it when units differ. The larger models cut matched-unit violations far more than mismatched
  ones (short first: Qwen3-14B 0.300 / 0.717 → Qwen3.8 0.011 / 0.254), so the gap is not a weak-model artifact.
- **Evidence to gather:**
  1. Thinking mode (Ian): Qwen3-14B pilot first, then the larger models if warranted. Measure below-boundary violations
     matched vs mismatched, paired by prompt with the non-thinking runs; count unclosed or truncated thinking as format
     failures.
  2. A second decision domain (all runs so far are one investment scenario).
  3. Jared's prompt types in place of the four current phrasings.
  4. A wider H/D range for cells whose curves are flat over [0.67, 1.5] (switch points censored there).
- **Confounds and how they are handled:**
  - *Option order and label bias* (P(choose "a)") short / long first: Qwen3-14B 0.244 / 0.977, Qwen3.8 0.674 / 0.684,
    Gemma 0.373 / 0.661): order is crossed; report by order and order-averaged.
  - *Liquidity preference:* above the boundary, choosing the short option is often a stated preference after a correct
    conversion, not an error (reasoning audit, 10-08). Use below-boundary violations as the metric.
  - *Phrasing:* four phrasings crossed; the effect holds in each (above). Jared's types extend this.
  - *D and reward ratio:* crossed; the effect holds at every level.
  - *Rendering format is confounded with horizon unit:* week and month horizons are integers, quarter and year horizons
    have two decimals (delays are integers in every unit). Control: integer-quarter and integer-year horizons at the
    durations where they exist.
  - *Unit familiarity* (quarters are rare for multi-year durations) cannot be separated from unit identity with these
    four units; a further unit (e.g. days) would help.
  - *Post-hoc reasoning:* in the current format the explanation follows the answer, so quoted reasoning is suggestive
    only; thinking mode puts the reasoning first.
- **How it could fail / what each outcome means:** if thinking mode brings mismatched violations down to the matched
  level in the larger models, the claim narrows to single-pass (non-thinking) comparison — still a result, and it says
  the conversion has to be written out. If thinking does not fix it, it is a robustness failure that survives deliberation.
  If a second domain or Jared's prompt types remove the matched/mismatched gap, the claim is scoped to this prompt family.

### C2 (representation vs use). Pre-answer activations encode the duration comparison correctly even on mismatched-unit prompts where the choice is wrong.

- **Evidence so far (indirect, earlier prompt sets):** a ridge readout trained on canonical renderings transfers to unit
  rewrites at R² 0.78–0.89 (Qwen3-14B, 09-22); readouts track absolute duration (slope vs log scale factor 0.81 / 0.98 /
  0.77 / 0.79 for Qwen3-14B / Qwen3.5-27B / Qwen3.8-27B / Gemma 4, 10-04); each unit forms its own neighbourhood on a
  shared magnitude curve (95–100 % of nearest neighbours share the unit; shared curve + per-unit offset recovers most of
  the fit, Qwen3-14B). None of this is on the unit-boundary prompts.
- **Evidence to gather:** activations on a subset of the unit-boundary prompts for one or two larger models. At
  pre-answer positions, decode log H and log D (or the sign of H − D) with probes trained on matched-unit prompts; test
  on mismatched prompts, split by whether the model chose correctly.
- **Key design point:** decodability alone is uninformative — a regular expression on the prompt always knows the right
  answer. The test is on the model's errors: on mismatched prompts where it chooses the long option although H < D, does
  the readout say "delay exceeds horizon" (represented, not used) or "fits" (not represented either)?
- **Confounds and how they are handled:**
  - *Option order:* average activations over the two orders (Shantanu), and also report per order.
  - *Unit identity is trivially decodable:* hold out unit pairs between training and test, so a probe cannot succeed by
    reading the unit token.
  - *Post-answer information:* read only positions before the answer token.
  - *Probe capacity:* compare against the same probe on shuffled labels and on a non-temporal quantity (reward).
- **How it could fail:** if the readout's comparison is at chance on the mismatched prompts the model gets wrong, the
  claim flips to "no unit-invariant comparison is represented at these positions" — the stronger version of the
  no-unified-understanding question raised on 10-08.

### C3 (exploratory, no new compute). Mismatch failures are fixed per-unit-pair biases, not noisy conversion.

- **Evidence so far:** in Qwen3.8, short first, cells with delays in quarters sit at P(short) 0.86–1.00 at every H/D,
  while cells with a horizon in quarters sit at 0.04–0.61 (10-08). Quarters ↔ years (conversion factor 4) loses the
  boundary, while months ↔ years (factor 12) shifts it by only 2.5–5 % in Qwen3.8 (−0.048 and −0.025 in ln H/D) — so
  difficulty does not track the size of the conversion factor.
- **Evidence to gather:** fit candidate decision rules per cell on the existing three runs (e.g. compare the raw
  numerals ignoring units; a constant per-pair bias; correct comparison plus noise) and compare their likelihoods. Not
  computed yet.
- **Confounds:** the same rendering confound as C1 (decimal vs integer horizons by unit); option order (fit per order).

## Secondary: readout reliability as a monitor (option A of 10-04)

- **Status:** no new evidence since 10-04. The mean-over-explanation readout recovers the assigned horizon robustly in
  four models (Qwen3.8 slope +0.030–0.042, within-2× 0.81–0.87; Gemma −0.004–0.015, 0.84–0.91), but it is computed after
  the answer, and a regex on the prompt reads explicit horizons at 1.00 within-2×.
- **Link to C:** the unit-mismatch prompts remove the regex objection. There the instruction implies the short option
  and the model chooses the long one, so a readout that predicts the violation does better than any parser of the
  instruction. This matches the doc's "monitor for temporal-related misalignment" end goal and its weaker
  "indication that it changed" target.
- **Hypothesis:** a pre-answer readout predicts constraint violations on held-out prompts beyond what the prompt
  parameters (units, order, phrasing, H/D) predict.
- **Confounds:** prompt parameters (include them as baseline predictors); order (as in C2); post-answer tokens (excluded);
  pooling and output length (equal-length pooling if explanation tokens are used).
- **Ownership:** unassigned in the Actionables tab; overlaps Augusto's relevance-aware decoder. Ian's "reasoning then
  answer" comment overlaps thinking mode, where the reasoning comes first.

## Demoted to support

- **Multiscale geometry (option B of 10-04).** The doc comment "Okay, nice plot, but what's the point?" and Soham's
  assignment (beyond PCA, intrinsic dimension) argue against making it a main direction. Its useful part — unit
  neighbourhoods on a shared magnitude curve — is the representation side of C2. Open thread 1 (lower-dimensional local
  scatter for years renderings) stays parked.
- **Stakes.** Decodable in all four large models; a choice effect only in Qwen3.5-27B (+0.298 [+0.153, +0.447]; Qwen3.8
  +0.052 [−0.092, +0.213]; Gemma +0.078 [−0.070, +0.227]). Future work, as the doc says.

## Scope that applies to every hypothesis above

One decision domain (investment); an explicit constraint sentence; single turn; greedy decoding; non-thinking (until the
pilot); behaviour only for the unit-boundary runs. Qwen3-14B: Ian (10-08) — thinking mode or stop using it; the thinking pilot decides.

## Questions for the group

1. Is unit compatibility (C1 with C2 as the mechanism question) a good anchor, with the monitor framing as the
   application?
2. If thinking mode fixes the mismatch, is "single-pass comparison fails" still interesting enough to lead with?
3. Which second decision domain, and should it wait for Jared's prompt types?
4. Who takes readout reliability, given the overlap with Augusto's decoder?
