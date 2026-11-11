# Results: overnight autoresearch (2026-11-03)

Post hoc, on saved artifacts only (activations, traces, answers of the existing runs); no
model was run and no box was used. Every analysis here was decided tonight, after the data
existed: read the numbers as exploratory, not pre-registered. Literature: `23_literature.md`.
Outputs: `SPAR/results/horizon/2026-11-03_autoresearch/`. Scripts: `analysis/` (named per
section); tests: `tests/test_autoresearch_2026_11_03.py` (fast suite 333 passed). An independent
code review of the new scripts (one agent, read-only) found nothing that changes a headline number.
Its fixes are applied and the outputs rerun:
- the "converts" flag compared a plural canonical unit with a singular one;
- E-F's decoded-ingredient baseline is now fully nested;
- the break-even fits group folds by bank-prefixed pair.
It also notes two conservative biases: the "best surface" baseline in §2 is picked on the
same rows, and the inner CV splits used for hyperparameters are ungrouped.

## 1. Summary

- **The 8B's "irrational" choices are arithmetic failures, not horizon misreads** (§3, §4).
  - The unsteered traces restate the prompt's horizon correctly: it moved > 0.05 decades in 3 of
    753 natural closes on the 8B and 6 of 445 on the 14B.
  - In the both regime (both payouts arrive in time) the 8B picks near on half the old-bank
    prompts. Every both-regime trace (105, coded by two agents, 31 of 32 agreeing with an
    independent earlier coding) reframes the task as "reinvest the earlier payout".
  - Where far wins at any normal rate (break-even r* ≥ 10%/yr), a correct comparison went
    far 45 of 45 times. An erroneous, overridden or absent one went near 30 of 32 times.
    The default without a finished calculation is "earlier money compounds longer".
  - The errors echo the representation findings: "(1+r)² > 25 → r > 4" read as "4%", a
    $9.97M product written $99.7M.
- **Choice follows the break-even rate** (§4).
  - Both regime: near 82% (8B) / 77% (14B) when r* < 10%/yr, 4% on the fill bank (r* past
    10⁴ %/yr).
  - The fill-vs-old-bank behaviour gap of `19`/`20` is mostly r*.
  - The r* dependence is made by the reasoning. With thinking off, the both-regime choice
    is flat in r* (14B 0.20 / 0.20 / 0.12 near; 8B fill 0.35 near off vs 0.04 on).
  - The 14B is a cleaner break-even chooser (out-of-fold log loss 0.48 vs null 0.65). On the
    old banks the 8B's r* fit is no better than the base rate (0.695 vs 0.693): it is
    near-flat at ~40% near from 10% to 1000%/yr, as its arithmetic errors predict.
- **Before reasoning, the residual stream reads the choice no better than the prompt's surface** (§2).
  - At the suffix the eventual choice is readable at AUC 0.82–0.95 (both models, none and
    both regimes), and a reader of the prompt record or text does as well: activations exceed
    the best surface reader by at most +0.03 at any suffix cell (and fall up to 0.19 below it
    at the earliest layers).
  - Positive control: with thinking off, the same reader at the same tokens does find
    choice information beyond the surface (+0.20 to +0.22 AUC at the 8B's last suffix token,
    both regime). So the thinking-on null is not insensitivity.
  - Mid-reasoning (think@0.25, 0.5) the stream reads it worse than the surface.
  - Information beyond the prompt appears only at the end of the block (think@1 from L16–20
    on the 8B, L20 on the 14B, AUC ≈ 1).
  - Nor is the break-even rate itself computed at the prompt end: a probe reads r* exactly
    as well as a linear map of the ingredients it encodes (§4.3; low power, r* is 97% linear
    in them).
  - The arithmetic shortcut is not readable before reasoning either (§2.2: AUC 0.39–0.58,
    all CIs covering 0.5, n 77).
  - It is fragile: in E10's unpatched rerun of identical prompts, 12 of 50 both-regime
    choices flipped under bf16 batch noise, against 0 of 46 near-regime (p 0.0003).
- **The prompt-end geometry is a unit-word and numeral code with a coarse duration order** (§5).
  - Over the whole residual stream (E9 renderings, cross-validated distances), "60 minutes"
    vs "1 hour" is 2.5–6× farther apart than "2 years" vs "3 years" at every layer.
  - Against other-unit pairs ~0.3 decades apart ("3 days" vs "1 week") the ratio is 0.97–1.10
    from L9 on: an exact duration match earns nothing over a 0.3-decade neighbour in another
    unit.
  - A numeral code is shared across units ("60 minutes" ~ "60 months": numeral ratio
    0.35–0.89 across layers, ~0.8 from L15 on). It is the magnitude of the written number, not a per-digit code: digit count,
    leading and last digit add ≤ 0.09 of the mean pair beyond L3.
  - The integrated duration that E9's probe read (unit share 0.02) is a minority subspace in
    a text code.
- **The CAA "wrong way" steer is the model's own none-regime gradient, not the forced-close
  confound** (§6).
  - At the L24 answer slot the long-minus-short direction has cosine +0.51 with the
    within-group chose-near direction and −0.05 with forced-vs-natural. Removing the forced
    direction leaves the push unchanged.
  - Within the none regime the 8B's near share rises from 0.36 to 1.00 as the horizon
    approaches the near delay. The short half of the horizons is none-regime, so "longer"
    really does mean "more near" there.

## 2. Is the choice set before reasoning? (`decision_emergence.py`)

**Question.** "Reasoning theater" (2603.05488) and answer-before-CoT probes (2504.05419) find
answers decodable before a chain ends. Here: within the regimes where the choice varies,
is the eventual choice readable from the residual stream before or during reasoning,
beyond what the prompt itself predicts?

**Method.**
- Rows: natural closes with a horizon. 8B: full + units + fill pooled (anchors out); 14B: full.
- Readers, all out of fold over 5 GroupKFolds of base prompts, scored by AUC within a regime:
  - surface: logistic on the metadata (delays, rewards, ratio, order, unit size, horizon,
    bank, domain, phrasing);
  - text: TF-IDF of the prompt;
  - acts: logistic on the top-64 PCs of a cell, dimensions z-scored first;
  - acts+surface.
- CIs resample base prompts.
- Limitation, found by the test: the reader keeps 64 PCs after z-scoring, so a choice signal
  of low variance is missed. The acts AUCs are lower bounds.

**Results** (`decision_emergence.{txt,png}`):

| | 8B none (n 453) | 8B both (n 228) | 14B none (n 276) | 14B both (n 86) |
|---|---|---|---|---|
| surface / text | 0.82 / 0.89 | 0.85 / 0.83 | 0.79 / 0.80 | 0.95 / 0.88 |
| suffix newline, best layer | 0.88 (L12/L24) | 0.84 (L16–28) | 0.82 (L15–30) | 0.94 (L20) |
| acts − best surface, suffix (range over layers) | −0.04…−0.01 | −0.06…−0.01 | −0.08…+0.03 | −0.19…+0.00 |
| think@0.25, best | 0.57 | 0.79 | 0.71 | 0.85 |
| think@0.5, best | 0.70 | 0.73 | 0.67 | 0.80 |
| think@1 from L20 | 1.00 | 0.99 | 0.99 | 0.99 |
| acts − best surface, pre-label L24+ | +0.11 | +0.14 | +0.20 | +0.05 |

- **No pre-reasoning disposition beyond the prompt.** The suffix carries the
  choice-relevant prompt features linearly, and nothing more that a linear probe on 64 PCs
  can find.
- **Mid-reasoning the choice is less readable than from the prompt.** At a single token
  25% or 50% of the way through the block the residual stream holds the local text, not a
  running decision. This matches the horizon's own dip there (`06`).
- **The extra information appears where the reasoning ends** (think@1 L16–20 onward, and
  the answer slot from L20–22, as the patching map found).
- AUCs below 0.5 at early think layers are the usual out-of-fold anticorrelation when a
  reader has no signal.

**Positive control: thinking off** (`decision_emergence_off.txt`, same code, `off` mode).
- Without a think block the choice is made in the forward pass from the prompt. So a reader
  that works should find information beyond the surface by the last suffix token (here the
  token after the prefilled empty think block, where the answer starts).
- It does:
  - 8B both regime (n 355): surface 0.49 and text 0.64; acts at the last suffix token 0.84
    (L28) and 0.86 (L36), +0.20 [+0.12, +0.31] and +0.22 [+0.15, +0.30] over the best
    surface; answer slot L24+ +0.30.
  - 8B none regime (n 671): +0.02 to +0.04 at the suffix with CIs excluding 0; +0.07 at the
    answer slot.
  - 14B: none +0.01 to +0.03 at the suffix, +0.11 at the slot. Both regime (n 93, near share
    0.18): +0.11 to +0.17 at the slot, CIs mostly touching 0.
- So the thinking-on null at the same tokens is not insensitivity of the reader. With
  thinking on, the stream at the prompt end does not carry the decision that the reasoning
  will reach.
- Quirk: with thinking off, acts+surface scores below acts alone on the 8B both rows (0.56
  vs 0.86 at L36). The many sparse surface columns appear to push the inner CV toward strong
  regularization. The acts column is the primary measure throughout.

### 2.2 Is the arithmetic shortcut set before reasoning? (`heuristic_disposition.py`)

**Question.** On the old-bank both prompts where far wins at any normal rate (r* ≥ 10%/yr),
"chose near" almost exactly marks "the trace did not complete a correct comparison"
(§3.3: 45/45 vs 30/32). Is that shortcut readable before reasoning? The fill bank is left
out, so the surface cannot win by separating banks.

**Result** (`heuristic_disposition.txt`). No.

| rows | n | near share | surface | text | acts, suffix newline L8–36 | acts, last suffix token L8–36 |
|---|---|---|---|---|---|---|
| r* ≥ 10%/yr | 77 | 0.39 | 0.44 [0.32, 0.60] | 0.61 [0.45, 0.74] | 0.39–0.50 | 0.51–0.58 |
| all old-bank both | 105 | 0.50 | 0.64 [0.54, 0.73] | 0.43 | 0.36–0.59 | 0.48–0.63 |

Every activation CI covers 0.5 on the clean row set. Power is limited (n 77; CI half-width
about ±0.15), so a weak disposition (AUC ≲ 0.65) could be missed.

**Corroboration: the both-regime choice is fragile under a rerun** (`rerun_stability.py`).
- E10 re-ran 96 prompts unpatched in another batch; bf16 batching perturbs the greedy trace
  slightly.
- All 12 label changes are in the both regime: 12 of 50 prompts (7 of 31 natural closes)
  against 0 of 46 near-regime prompts, Fisher p 0.0003.
- Flips go both ways (8 far→near, 4 near→far).
- A choice that flips under numerical noise in the trace is a property of the trace's path
  (whether the arithmetic gets done), not of a state fixed at the prompt end. This is the
  prediction of P2, now partly observed.

## 3. What the reasoning says (`trace_audit.py`, `trace_coding_summary.py`)

**3.1 Regex audit, all thinking-on traces.**
- The first duration a trace states as its horizon (`exp14_analysis.first_stated_horizon`)
  matches the prompt's: it moved > 0.05 decades in 3 of 753 natural closes (8B old banks),
  0 of 244 (8B fill) and 6 of 445 (14B).
- So the text-mediation test (does the written horizon predict the choice beyond the
  prompt's?) has nothing to work with on unsteered traces. E14's steered traces remain the
  only place it can be run.
- Both-regime near-choosers write about half as much: median think length 881 vs 1,788
  tokens (8B), 589 vs 1,402 (14B).
- The "conflict" regex (`typo|wait, the horizon|mistake|inconsistent`) splits the choosers.
  Theme flags are post-treatment, so these are descriptive:
  - none regime: far-choosers 0.72 vs near-choosers 0.30 (8B, Fisher p 8e-20); 0.68 vs
    0.39 on the 14B (p 3e-6). A far choice in the none regime goes with declaring the
    horizon a typo and dropping it.
  - both regime: far 0.19 vs near 0.02 (8B, p 0.004), 0.29 vs 0.06 (14B, p 0.01). Here
    "mistake" likely flags self-correction of the arithmetic, the path to far in §3.3. This
    is not checked against the traces: the regex mixes horizon doubts with arithmetic slips.
- Other themes (`trace_audit.txt`): in the none regime far-choosers argue reinvestment
  (0.86 vs 0.59) and risk (0.40 vs 0.23); near-choosers argue immediacy (0.96 vs 0.85).

**3.2 Open coding of 48 traces** (one Sonnet agent; `trace_coding.json`; 44 of 48 quotes
checked verbatim against the traces; one arithmetic claim checked by hand):
- Both regime (32): every trace reads the horizon correctly (in-time judgement right 32 of
  32) and reframes the task as reinvestment.
- None regime (16): the horizon is mostly not processed as stated. Minutes-to-days horizons
  are called a typo or "time to decide". Only 5 of 16 say neither payout arrives in time.
  Near-picks argue "sooner", far-picks "larger sum".

**3.3 Arithmetic coding, all 105 natural-close both-regime 8B traces** (two Sonnet agents,
one codebook, arithmetic recomputed in Python; `both_b8{a,b}_coded.json`). The 14B batch
was stopped by a safeguard false positive and is not coded.

| r* | computes | n | near | near share [95% Wilson] |
|---|---|---|---|---|
| ≥ 10%/yr | correct | 45 | 0 | 0.00 [0.00, 0.08] |
| ≥ 10%/yr | error | 9 | 9 | 1.00 [0.70, 1.00] |
| ≥ 10%/yr | overridden | 8 | 8 | 1.00 [0.68, 1.00] |
| ≥ 10%/yr | none | 15 | 13 | 0.87 [0.62, 0.96] |
| < 10%/yr | correct | 11 | 7 | 0.64 [0.35, 0.85] |
| < 10%/yr | error / overridden / none | 17 | 16 | |

- All 7 "correct, chose near" traces are low-r* prompts where near does win at the rate the
  trace assumes (5–7%): those choices are right under the model's own framing.
- Agreement with the independent earlier coding on "a correct comparison decides": 31 of 32.
- The agent flagged an ambiguity: unevaluated compound formulas with a stated rate were
  coded as numeric (about 10 traces). Under the strict reading they become "none", which
  does not change the near/far split.
- **Reading.** With thinking on, the 8B's choice in the both regime is decided by whether
  the trace completes a correct comparison. The horizon is read correctly. The "truth" label
  (far) is the right answer only under the reinvestment framing at r* ≥ the assumed rate;
  for 28 low-r* prompts near is defensible.

## 4. The break-even rate

**4.1 Behaviour** (`breakeven_rate.{txt,png}`): r* = (R_far/R_near)^(1/(d_far − d_near)) − 1,
capped at 10⁴/yr (fill delays of seconds overflow otherwise).

| near share by log10 r* | < −1 (≤ 10%) | −1…0 | 0…1 | 1…2 | ≥ 2 |
|---|---|---|---|---|---|
| 8B both, old banks | 0.82 (28) | 0.40 (50) | 0.40 (25) | 0.00 (2) | |
| 8B both, fill | | | | | 0.04 (122) |
| 14B both | 0.77 (22) | 0.30 (46) | 0.00 (16) | 0.00 (2) | |

- Out-of-fold log loss, both regime: 14B r* alone 0.481 (null 0.654), plus horizon / near
  delay 0.433; 8B old banks 0.695 (null 0.693: no better than the base rate); 8B pooled with the fill 0.395 (null 0.567).
- None regime: the strongest term is log(horizon / near delay), +1.05 to +1.11 per SD on
  both models. Near grows as the horizon approaches the near delay, with r* adding −0.5 to
  −0.8 per SD.

**4.2 Thinking off: no r* dependence** (`breakeven_rate_off.txt`; same code with the
thinking-off answers).

| near share, both regime | r* ≤ 10% | 10–100% | 100–1000% | fill (≥ 10⁴%) |
|---|---|---|---|---|
| 8B off | 0.49 (37) | 0.21 (84) | 0.28 (32) | 0.35 (197) |
| 8B on | 0.82 (28) | 0.40 (50) | 0.40 (25) | 0.04 (122) |
| 14B off | 0.20 (25) | 0.20 (50) | 0.12 (16) | |
| 14B on | 0.77 (22) | 0.30 (46) | 0.00 (16) | |

- Without a think block, r* does not predict the both-regime choice. 14B off: log loss 0.505
  vs null 0.476, worse than the base rate. 8B off pooled: 0.654 vs 0.626.
- The break-even sensitivity, both the near choices at low r* and the far choices at
  astronomical r*, is produced by the reasoning: by the arithmetic the trace does or fails
  to do.
- In the none regime, off-mode choice does fall with r* (0.81 → 0.43 near on the 8B),
  and so does on-mode.
- Caveat: thinking-off choices carry a large option-order effect (+6.3 log-odds, `01` §6),
  not modelled here.

**4.3 Is r* computed at the prompt end?** (`breakeven_decode.txt`; 8B full + units, n 986)
- A quadratic surface of the ingredients predicts log r* at R² 0.998, a linear one at 0.973.
- At suffix L20, a ridge probe on the top-64 PCs reads r* at 0.831. A linear map of the four
  ingredients as the same cell encodes them (each probed out of fold: ratio 0.86, delays
  0.92–0.94; fully nested) reads it at 0.832.
- The same holds at L8–32 and at the last suffix token: differences ≤ 0.002.
- Nothing beyond the ingredients; low power, since only 3% of r* is nonlinear in them.

## 5. Duration, unit word or numeral (`unit_number_rsa.py`)

**Method.**
- E9 activations: 62 renderings (22 durations × 2–4 units) in 24 contexts; suffix newline
  and last suffix token; odd layers.
- Dimensions z-scored. Distances are cross-validated (dot product of a pair's difference in
  one half of the contexts with the other half; unbiased, tested), scaled so the mean pair
  is 1.
- Contrasts:
  - integration ratio: same duration in another unit, against same-unit pairs 0.1–0.5
    decades apart;
  - unit-neutral version: against other-unit pairs 0.1–0.5 decades apart;
  - numeral ratio: same numeral, other duration, against gap-matched other-numeral,
    other-unit pairs.
- CIs resample contexts within each half. They are very tight because they reflect context
  sampling only, not the fixed set of 61 same-duration pairs.

| layer (newline) | 1 | 5 | 9 | 13 | 17 | 21 | 25 | 29 | 33 |
|---|---|---|---|---|---|---|---|---|---|
| integration ratio (same-unit ref) | 4.9 | 5.4 | 2.8 | 3.1 | 2.7 | 2.6 | 2.6 | 2.6 | 2.6 |
| integration ratio (other-unit ref) | 1.15 | 1.15 | 0.97 | 1.01 | 1.01 | 1.03 | 1.02 | 1.03 | 1.05 |
| numeral ratio | 0.35 | 0.85 | 0.59 | 0.69 | 0.85 | 0.89 | 0.84 | 0.81 | 0.81 |
| same duration, other unit (1 = mean pair) | 1.21 | 1.08 | 0.85 | 0.72 | 0.59 | 0.55 | 0.56 | 0.53 | 0.51 |

The last suffix token is similar (same-unit ratio 6.3 → 3.0, other-unit ratio 1.04–1.12).

- Across units the code does order durations: same-duration pairs fall from 1.2 to ~0.55
  of the mean pair with depth. But a 0.3-decade neighbour in another unit is as close as the
  same duration, and a same-unit neighbour is 2.5× closer.
- This resolution matches E9's unit shifts (~0.30 decades) and the unit-lookup ceiling
  (R² 0.96). It is the representational side of E14's digit edits and E10's unit swap.
- **The numeral code is the written number's magnitude, not its digits** (`numeral_digits.txt`).
  Pair distances were regressed on the gap bins, same unit, same numeral, |Δ log numeral|,
  and same digit count / leading digit / last digit (Levy & Geva's per-digit code would put
  weight on the last three).
  - Beyond L3 the digit terms are small and mixed in sign (|coef| ≤ 0.09 of the mean pair).
    "Same numeral" adds nothing consistent once |Δ log numeral| is in.
  - |Δ log numeral| carries +0.28 per decade at L9 and +0.15–0.20 at L15–33. "Same unit"
    carries −0.37 (L3) to −0.14 (L21+).
  - So the prompt end holds two roughly additive text codes, log written number and unit
    word. Their sum, log duration, is formed only coarsely.
  - At L3, digit count and last digit do matter (−0.18 each): token-level similarity before
    the number is read as a value.
  - E14's digit edits ("6 months" → "66 months") fit a steer that moves the log-numeral code
    by about a decade, which the reasoning then writes as a number of that size.
- Implication for steering: a "duration" direction is not a privileged axis of this code.
  A manipulation that targets the unit-word and numeral subspaces separately is closer to
  what the model stores (P4).

## 6. What the CAA direction is made of (`caa_confound.py`)

8B thinking on, full bank. Directions: long-minus-short halves (CAA); forced-minus-natural
within duration; chose-near-minus-far within duration × regime; and long-minus-short within
each regime.

| cell | cos(CAA, near) | cos(CAA, forced) | CAA·near after removing forced | within-regime CAA·near (none / near / both) |
|---|---|---|---|---|
| suffix newline L16 | +0.12 | −0.09 | +0.06 | +0.05 / −0.06 / +0.34 |
| think close L24 | +0.19 | +0.22 | +0.36 | −0.40 / −0.83 / +0.29 |
| pre-label L22 | +0.47 | −0.11 | +0.42 | +0.14 / −0.71 / +0.46 |
| pre-label L24 | +0.51 | −0.05 | +0.50 | +0.40 / −0.55 / +0.34 |

Null |cos| 95th percentile 0.033.

- The forced-close confound (`01` §4, suspected) does not explain the sign at the answer
  slot.
- Behaviour within the none regime: near share 0.36, 0.36, 0.67, 0.89, 1.00 across log10
  horizon bins from < −4 to −1…0 (8B); 0.37 → 0.94 on the 14B.
- The CAA push toward near is therefore the behaviour of the prompts that make up the short
  half, not an artefact.
- What drives the none gradient (`trace_audit.txt`, regex, descriptive):
  - On the 8B, the share of none traces that question the horizon ("typo", "mistake",
    "doesn't make sense") falls from 0.77 to 0.41 as the horizon nears the near delay.
    Questioning goes with far (near 0.26 vs 0.56 in the furthest bin).
  - But near share also rises within each group (questions: 0.26 → 1.00; does not: 0.56 →
    0.97), so dismissing the horizon accounts for only part of the gradient.
  - On the 14B the flag fires on ~0.9 of none traces in every bin: the regex is too broad
    there to separate anything. Within the near regime, longer does point toward far (−0.55 to
  −0.83), as expected.
- Correction for `01` §4 / `08`: "CAA steers the wrong way" should read "CAA follows the
  none-regime gradient (near rises as the horizon nears the near delay)". The
  `08` §9 alternative ("the short half is all none-regime") is the right one, and the
  mechanism is the gradient within none, not only the composition.

## 7. Corrections and additions for `01_plan.md` / `13_evidence.md` (proposed, not applied)

1. §4 "Mean difference (CAA) ... Suspected cause: long horizons are force-closed more often":
   replace with §6.
2. §4 / R5: the both-regime near choices are arithmetic failures (§3.3). The "truth" label
   is not the model's target under its reinvestment framing; for r* < ~10%/yr near is
   defensible. Effects in units of C mix the horizon comparison with arithmetic success.
3. R3: add §5. The full-space geometry is a unit-word + numeral code; duration ordering
   across units has ~0.3-decade resolution.
4. New: no pre-reasoning choice information beyond the prompt with thinking on (§2), though
   the same reader finds it with thinking off; the excess appears at the end of the think block.

## 8. Proposed next experiments (each needs a spec and Jared's approval; box unless noted)

| # | experiment | question | prediction from tonight | cost |
|---|---|---|---|---|
| P1 | **Arithmetic insertion** at the first commitment in both-regime traces (8B): insert a correct worked comparison ("reinvesting $X at 7% for N years gives $Y < $Z"); controls: a neutral sentence of the same length, and a wrong comparison favouring near inserted into far-choosers; continue greedily | is the computation the causal mediator of the both-regime choice? | correct insertion flips near → far on r* ≥ 10% prompts in most cases; wrong insertion flips far → near less often (the model checks). Off-policy edits are weak per Thought Branches, so pair with P2 | ~200 continuations, ~30 min |
| P2 | **On-policy resampling** (Thought Anchors / Branches): 8 samples at T = 0.6 per both-regime prompt, record choice and whether the trace computes | is "skips the arithmetic" a property of the prompt or of the sample? | §2.2 found no pre-reasoning disposition and 24% of both-regime choices flip under batch noise, so near share per prompt should be intermediate (most prompts between 0.2 and 0.8), with P(near) falling in r* | 105 × 8 traces, ~2 h on one 4090 |
| P3 | **Rate-anchored prompt**: add "assume a 5% annual return" to the both-regime prompts | does the model use the horizon at all once the rate is fixed? | choice follows r* vs 5% almost perfectly; the horizon adds nothing within the both regime | one E0-style run on 105 prompts, ~20 min |
| P4 | **Numeral vs unit patching** within one unit (`01` §5.4 follow-up), now with E-C's prediction: patch the numeral tokens between "6 months" and "66 months" | is within-unit magnitude carried by the numeral tokens? | the reasoning writes the donor numeral with the recipient's unit; a duration-curve steer of the same norm does not | ~30 min on 2 GPUs |
| P5 | **DAS vs PCA vs size-matched random** for E10's unit swap (Makelov 2311.17030) | is the 8-dim unit subspace the carrier or a dormant-pathway illusion? | PCA ≈ DAS ≫ random of matched size | ~1 h |
| P6 | **None-regime gradient at the answer slot**: patch the answer slot between none prompts far from and close to the near delay | is the none gradient computed in the reasoning (text) or read at the slot? | transfers like the regime contrast (0.5 at L24) if it is a decision, not a horizon | ~15 min |
| P7 | The 14B arithmetic coding (86 traces, files ready in the scratchpad; stopped tonight by a safeguard flag on the coding agent, request `req_011CghDfH1cveikwH2bEN6Zg`). Jared decides whether and how to retry | does the 14B's cleaner r* behaviour come from correct arithmetic? | most 14B both-near traces sit at r* < 10% with correct comparisons | agent time only |
| P8 | Window-averaged think positions (box re-extraction): mean of the residual over ±32 tokens around think@0.25/0.5 | is the mid-block dip in horizon and choice readability (§2, `06`) a single-token artefact? | readability rises but stays below the surface reader | ~1 h extraction |

## 9. Files

`SPAR/results/horizon/2026-11-03_autoresearch/`:
- `decision_emergence.{txt,json,png}`, `decision_emergence_off.{txt,json,png}`, `heuristic_disposition.{txt,json}`, `rerun_stability.txt`;
- `trace_audit.{txt,json}`, `trace_audit_rows.json`, `trace_sample.jsonl`,
  `trace_coding.json` (open coding), `both_b8{a,b}.jsonl` and `both_b8{a,b}_coded.json`,
  `trace_coding_summary.txt`;
- `breakeven_rate.{txt,json,png}`, `breakeven_rate_off.{txt,json,png}`, `breakeven_decode.{txt,json}`;
- `unit_number_rsa.{txt,json,png}`, `numeral_digits.{txt,json}`, `caa_confound.{txt,json}`;
- `lit_geometry.md`, `lit_causal_cot.md` (the raw reviews; synthesis in `23_literature.md`).
