# Spec: implied horizons, side experiment (draft 2026-11-07, not approved)

Status: **draft, not approved.** Qwen3-8B, thinking on, one 4090 box. A side question, not on the main path of `01` §5.
Builds on `22` (E14), `08` §10-11 (E9, E10), `24` §3-5 and the fill bank's accuracies (`19` §7).

## 1. Question

Every result so far uses a written duration, and the prompt-end code is mostly that text: unit word plus written number
(`24` §5; unit alone R² 0.96, `01` §4). When the horizon is only implied ("until the lease expires"): (a) does the
choice track it as it tracks the written one; (b) does it reach the states where the written one sits, before or after
reasoning; (c) does the trace first turn the anchor into a duration, and how accurately? Prediction from `24`: (a) yes
where (c) succeeds; (b) absent before reasoning, present at the end of the think block (resolved in text).

## 2. Candidates

| # | experiment | tests | cost | box? | result that would change our beliefs |
|---|---|---|---|---|---|
| 1 | Twin-matched implied bank: generate implied + explicit-twin + no-horizon prompts, score choice and stated horizon, read the saved activations with the explicit-trained probe | a, c fully; b by readout | ~4 h, ~$1.6 | yes | implied accuracy far below the twin (horizon use depends on the written format), or an implied duration readable at the suffix (resolved before reasoning) |
| 2 | Donor patch, explicit twin -> implied phrase (E14 machinery) | b causally | ~3 h, needs bank of 1 | yes | a patch that moves the stated horizon of an implied prompt as it does an explicit one |
| 3 | Forward pass only: probe readout and RSA of implied vs explicit at phrase end and suffix | b before reasoning | ~15 min | yes (or Jared's xpu if 8B fits) | an implied anchor landing near its twin at the suffix |
| 4 | Trace-only conversion check on a few anchors, no truth design | c | ~30 min | yes | anchors the model cannot convert at all |

Pick **1**. Rejected: **2**, E14 needs identical phrase positions and an implied phrase has another token length, so
any alignment is arbitrary, and the last-token patch carries the unit word (E10), which the implied phrase lacks; run
it only if 1 finds the implied duration at the suffix. **3** is folded into 1 for free; alone, its predicted null says
little. **4** is a subset of 1 without the behaviour truth.

## 3. The bank

Built on the crossed generator (`full_bank.delays_for`, `draw_crossed_delays`): its delays give explicit thinking-on
accuracy 98.5% near / 97.5% both (`19` §7; high break-even rate, so both-regime choices do not hinge on `24` §3's
reinvestment arithmetic). Each anchor has an interval [lo, hi] in years; the nearer delay sits **≥ 0.3 decades outside
it** (MARGIN raised from 0.2), so any reading inside gives the same regime and truth (`near` -> near, `both` -> far;
`none` has no truth, near share graded in log(h / near delay), `24` §4.1). Infeasible regimes are skipped.

**Families** (2-3 of 8 anchors each shown; years in brackets):
- **A, computed from stated dates.** Context sentence states "now"; the phrase names an end point. "It is 9:15 am. Only
  payouts received before 4:30 pm today count." [7.25 h = 8.3e-4]. "Today is 1 February 2026. ... by the end of the
  fiscal year, which closes on 30 June 2026." [149 d = 0.408]. "... before the next US presidential election (5 November
  2028)" [2.76]. Exact, but numbers stay in the prompt.
- **B, world-knowledge anchors, no numbers anywhere.** Lexical durations: "over the coming fortnight" [2 w = 0.038],
  "over a generation" [20-30]. Life events: "before your newborn daughter starts kindergarten" [4-6], "by the time
  your newborn grandchild retires" [60-70]. The only number-free family; date-independent by choice.
- **C, event anchors with the duration stated elsewhere.** "The household's lease, signed today, runs for 18 months."
  in SITUATION; constraint "... for this time horizon: until the lease expires." [1.5]. "The board meets every quarter
  and met today." / "before the next board review" [0.25]. Same grid durations as the bank, unit word present but
  displaced from the phrase.
- **D, vague quantifiers (descriptive, no truth).** "in the short term", "in the long run", "very soon", "for the
  foreseeable future". Measures only what duration the trace states.

**Twins.** Each base is rendered (i) implied, (ii) explicit twin: the same text, including A/C's context sentence,
with the constraint "for this time horizon: X", X = the interval's log-midpoint via `natural_horizon`, and, once per
anchor, (iii) null: the constraint line removed. Twins differ only in the constraint phrase. Records carry `family`,
`anchor_id`, `anchor_lo/hi`, `rendering`, `twin_id`, plus the usual `regime`, `truth`, r*.

Size: 24 anchors (A, B, C x 8) x (2 near + 2 both + 1 none) x (implied, twin) = 240, + 24 null + 24 vague = **288**.

## 4. Measures

- **(a) Behaviour:** accuracy on near + both per family, implied vs twin (paired by base, McNemar); near share in
  none against the twin's; near share of the null twin as the "anchor ignored" reference. Natural closes first; forced
  rate per arm reported.
- **(c) Conversion:** first duration the trace states as its horizon. `exp14_analysis.first_stated_horizon` needs a
  horizon word nearby, so it will miss "the lease ends in 18 months"; extend it for anchor phrasing, then audit 40
  implied traces blind (as `22` §1) before scoring. Report: converted (states any duration), error |log10 stated −
  log10 true| (A, C) or inside [lo, hi] (B), and whether the stated duration flips the regime.
- **(b) Representation, readout:** ridge probe on the saved explicit full + fill activations (suffix newline L16; think@1
  natural closes and pre-label L16-24), applied to implied prompts. Report MAE in decades and Spearman ρ with the true
  log duration, against the twins' MAE in the same run.
- **Text mediation (new):** untestable on unsteered explicit traces (3 of 753 restate a moved horizon, `24` §3.1);
  implied traces misconvert naturally. Where the stated regime differs from the true one, which does the choice follow?

## 5. Criteria (fixed before running)

- C0 (reference valid): twin accuracy ≥ 90% on near and on both. Fail -> delay design is wrong, stop and fix.
- C1 (behaviour tracks), per family: implied accuracy ≥ 85% and ≥ twin − 10 points. "Ignored" if the implied near
  share in near and in both are each within 15 points of the null twin's.
- C2 (converts): ≥ 80% of implied traces state a duration; A and C median error ≤ 0.1 decades; B ≥ 85% inside [lo, hi].
- C3 (same states): suffix-readout MAE of implied ≤ 1.5x the twins' MAE and ρ ≥ 0.8 -> present before reasoning;
  ρ < 0.3 -> absent. Same thresholds at think@1 / pre-label on natural closes.
- C4 (mediation): with ≥ 10 regime-flipping misconversions, choice follows the stated regime in ≥ 75%. Fewer ->
  reported as underpowered, no verdict.

Readings: C1+C2 pass, C3 absent at suffix and present at think@1 -> the implied horizon is resolved in the trace and
then used like a written one; the phrase-position code is a text code, consistent with `24` §5, and candidate 2 is
pointless. C3 present at suffix -> resolved before reasoning; run candidate 2. C1 fails as "ignored" -> anchors are not
read as constraints; prior horizon claims are format-specific. C1 fails with C2 failing and C4 passing -> choice follows
the converted duration, errors are conversion errors (the analogue of `24`'s arithmetic failures). C2 passes, C1 fails
-> the stated duration is not what decides, which contradicts E14's landed traces and needs a look. C passing where B
fails -> the model uses a duration present in the text, not world knowledge.

## 6. Confounds and what is not tested

- **Ambiguity / date dependence:** handled by intervals, margins and stating "now" in A; residual cultural defaults in
  B (US school and retirement ages). The model's own sense of today's date is not tested.
- **Prompt length:** the implied phrase is longer, A and C add a sentence; twins carry the same sentence, so only the
  phrase differs. Positions after the phrase shift, so no position-wise phrase comparison (why 2 is rejected).
- **Unit-word shortcut gone** in A/B by design, so a suffix-readout null there is predicted and weak evidence; C (unit
  word displaced) is the informative contrast for (b). Only B is number-free: A tests date arithmetic, C binding.
- **Mediation is post-treatment:** stated duration and choice come from one generation; descriptive, not causal.
- Not tested: causal role of phrase states, rewording (one wording per anchor), domains, 14B, thinking off, choice4/8.

## 7. Cost and stopping

288 generations x ~45 s at batch 12 = 3.6 h, plus ~0.4 h provisioning: **~4 h, ~$1.6 at $0.40/h** (cap $2.5).
Activations at kept positions for 4 layers only (L12/16/20/24); readout and coding run locally afterwards. Modular, one
unit saved as it finishes, sync every unit: M1 A + C near/both, twins, null (144, 1.8 h; decides C0-C2 with exact
truth); M2 B near/both, twins, null (72, 0.9 h); M3 none regime (48, 0.6 h); M4 vague (24, 0.3 h). A run stopped
after M1 answers (a) and (c) on exact anchors.

## 8. Open questions for Jared

1. Budget and stop time; run M1 alone first?
2. Does C (duration stated elsewhere) count as "implied" for you, or only B?
3. B's cultural anchors acceptable, or restrict B to lexical durations (fortnight, semester, generation)?
4. Keep the none regime (M3)? Long horizons force-close ~89% there (`19` §7).
5. Pre-approve candidate 2 conditional on C3 "present at suffix", or a new spec then?

Housekeeping: `build_implied_prompts` beside `build_fill_prompts`, `--bank implied` in `exp0_baseline.py`, the parser
extension; tests first (twins differ only in the constraint line; each regime holds over [lo, hi]).

## Decisions (Jared, 2026-11-07, overnight)

Approved for an overnight run on one RTX 4090. The open questions of §8 are settled as follows.

1. **Order and budget.** Modules run M1 first, then M2 and M3, M4 last. The budget is a shared $10 hard cap across three
   experiments, so the whole run must fit about 4 h of one 4090, and M1 (~1.8 h) must be usable on its own. Every unit is
   saved as it finishes, so any prefix is analysable.
2. **Family C counts as implied** (the duration is stated elsewhere in the prompt; the constraint phrase names the event).
3. **B's cultural anchors stay** (US kindergarten, high-school and retirement ages), but their rows carry
   `us_convention` = true (B5, B6, B7) so they can be dropped in the analysis.
4. **The none-regime module is dropped from the criteria** (89% forced closes). The code still builds it as M3, run after
   M1 and M2; it is cut first if the time runs short, and no criterion needs it.
5. **No donor patch** (candidate 2): it needs a new spec.

Implementation: `spar_horizon/implied_bank.py` (`--bank implied`), `experiments/exp18_implied.py`,
`analysis/implied_summary.py`, `queue_implied.sh`. Sizes: M1 144 (A and C, 8 anchors each, 2 near + 2 both bases, implied +
twin, plus a null per anchor), M2 72, M3 48, M4 24 = 288 records. Where the criteria are computed:
C0-C2 and C4 from the summary of M1 (and M2); C3 needs the explicit E0 activations (`--probe-runs`).
