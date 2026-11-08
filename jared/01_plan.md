# Plan and status: the time-horizon representation in Qwen3-8B, thinking on

Last revised 2026-10-26 (audit: §4 corrected, §6 lists what was struck). Qwen3-14B
results are in `11`, the local checks in `12`, the graded claims in `13`.
The question, what is done (§3: do not rerun it),
what is established (§4), what is next (§5), and the dated decisions (§6).
The previous version is `history/01_plan_2026-10-11.md`.

## 1. Question

On Qwen3-8B with **thinking on**: is the stated time horizon represented in a
stable, prompt-independent form, does the model's choice causally use that
representation, and what happens to it while the model reasons?

| # | Sub-question | Status |
|---|---|---|
| R1 | Survives rewording (Alan's 4 constraint phrasings) | open on the full bank; starter bank, thinking off: transfer 0.99 |
| R2 | Survives a domain change | deferred |
| R3 | Encodes duration, not the unit word | passes at the suffix (unit share 0.02), with unit shifts of ~0.3 decades (§4) |
| R4 | Invariant to option order, labels, rewards | thinking on: option order adds +2.5 near log-odds |
| R5 | Causally used | partly: the decision reaches the answer slot from L24; curve steering at L24 moves it, the probe line does not; the horizon-vs-decision question is open (§4) |
| R6 | Persists through reasoning | weak in the middle of the think block, re-formed at the close (§4) |

## 2. Fixed setup

- **Model:** `Qwen/Qwen3-8B`, bf16, one vast.ai RTX 4090. Thinking on only (§6, 2026-10-14).
- **Bank:** the full bank (Alan's generator, prior project's structured
  format), 500 prompts, 471 with a horizon over 48 log-spaced horizons (30 s
  to 500 y), delays, rewards and option order drawn per prompt. Each record
  carries `regime`: `none` (horizon shorter than the near delay), `near`
  (between the delays), `both` (past the far delay), `null`.
- **Reasoning:** think block force-closed at 3,072 tokens; 103 of 500
  forced, mostly long horizons. Forced prompts are flagged, not dropped.
- **Outcome for causal tests:** near log-odds `logit(near label) - logit(far
  label)` at the token before the label, on the cached reasoning (no
  regeneration). Effects are in units of the regime contrast
  `C = mean LO(near) - mean LO(both)` = 12.9 log-odds on the 176 near/both prompts.
- **Data on disk** (`SPAR/results/horizon/`): the 500 reasoning traces in
  `gen_cache/`, activations at every layer for the suffix, think@0.25/0.5/1,
  `</think>` and pre-label positions in `exp0_Qwen3-8B_think-on_full/activations.npz`.
  New analyses at these positions need no box time.

## 3. Done: do not rerun

| what | date | output (`SPAR/results/horizon/`) | write-up |
|---|---|---|---|
| E0 full bank, thinking on (generation, activations, behavior) | 2026-10-04 | `exp0_Qwen3-8B_think-on_full/`, `gen_cache/` | `pipeline/s1_data_vectors/docs/06_results_probes_full.md` |
| E0 full bank, thinking off (record only) | 2026-10-04 | `exp0_Qwen3-8B_think-off_full/` | same |
| E7 linear vs MLP probes, both modes | 2026-10-04 | `analysis/probe_*` | `pipeline/s1_data_vectors/docs/06_results_probes_full.md`, spec `pipeline/s1_data_vectors/docs/04_spec_probes.md` |
| E8 run 1: patching map (thinking off), 4 decodability-chosen cells: all null | 2026-10-11 | `exp8_Qwen3-8B_full/` | `pipeline/s2_steering/docs/08_results_steering.md` §1-3, spec `pipeline/s2_steering/docs/07_spec_steering.md` |
| E8 run 2: answer-slot ablation and steering at L24/27/30, answer-slot patching across prompts, both modes | 2026-10-12 | `exp8_Qwen3-8B_answer_slot/` | `pipeline/s2_steering/docs/08_results_steering.md` §5 |
| Horizon-vs-decision check on the answer-slot subspace (local) | 2026-10-12 | `analysis/steering_decision_check.*` | `pipeline/s2_steering/docs/08_results_steering.md` §6 |
| E8 run 3: L24 curve split into decision direction and rest, natural closes | 2026-10-14 | `exp8_Qwen3-8B_decision_L24/` | `pipeline/s2_steering/docs/08_results_steering.md` §9 |
| E9: unit rewrites at the suffix, 22 durations x 2-4 units x 24 contexts | 2026-10-14 | `exp9_Qwen3-8B_units/`, `analysis/units.*` | `pipeline/s2_steering/docs/08_results_steering.md` §10 |
| E8 at L22, decision split | 2026-10-18 | `exp8_Qwen3-8B_decision_L22/` | `pipeline/s2_steering/docs/08_results_steering.md` §12 |
| E10: patch before reasoning, 96 pairs | 2026-10-18 | `exp10_Qwen3-8B_reasoning/` | `pipeline/s2_steering/docs/08_results_steering.md` §11 |
| E11: Qwen3-14B E0 both modes, E8 answer slot, probe line, coherence | 2026-10-20/21, rest 2026-10-24 | `exp0_Qwen3-14B_*`, `exp8_Qwen3-14B_*`, `coherence_Qwen3-14B_think-on_full/` | `pipeline/s1_data_vectors/docs/11_results_14b.md`, `13_evidence.md` |
| Local checks: natural closes and within-unit R², horizon vs delay | 2026-10-21 | `analysis/` | `12_results_local_checks.md` |
| Unit-overlap bank E0, 8B both modes | 2026-10-25 | `exp0_Qwen3-8B_think-{on,off}_units/` | spec `pipeline/s1_data_vectors/docs/14_spec_unit_overlap.md`; no results doc yet |
| Pristine steering stage A (answer slot, fixed α) and stage B (suffix, thinking on) | 2026-10-30 | `exp8_Qwen3-8B_pristine/`, `exp13_Qwen3-8B_pristine_suffix/`, `analysis/crossing_split_pristine.*` | `pipeline/s2_steering/docs/22_results_phrase_steering.md` §1, spec `16` §5 |
| E14: steering at the horizon phrase, thinking on, 4 transitions x 38 conditions | 2026-10-31 | `exp14_Qwen3-8B_phrase/` | `pipeline/s2_steering/docs/22_results_phrase_steering.md`, spec `pipeline/s2_steering/docs/21_spec_phrase_steering.md` |
| 4- and 8-choice banks (implicit ROI, 246 prompts each), behavior plus reduced activations and structure probes, thinking on | 2026-11-06 | `exp0_Qwen3-8B_think-on_choice{4,8}/` | `pipeline/s2_steering/docs/26_results_overnight_choice_scan.md` §1, spec `pipeline/s2_steering/docs/26_spec_overnight_choice_scan.md` |
| Phrase-steering scan (L10/14/18/22/26 x number, unit, suffix, answer x manifold, random; 10 prompts per cell): 34 of 40 cells, L26 incomplete; no cell reaches excess ≥ 4 | 2026-11-06 | `exp15_Qwen3-8B_scan/` | `pipeline/s2_steering/docs/26_results_overnight_choice_scan.md` §2 |
| Starter bank: v1 (thinking off, E0-E5), v3 (thinking on, E0, E6) | 2026-09 | `snapshots/v1_noprefill`, `snapshots/v3_think` (removed 2026-10-27; in git at `a93acf9`) | `history/` |

## 4. Established (Qwen3-8B, thinking on, full bank unless noted)

**Representation**
- Before reasoning the horizon is readable almost perfectly: suffix newline
  after `<|im_end|>` R² 1.00 (L16), leave-one-horizon-out.
- **But the unit word alone scores R² 0.96 under the same protocol** (each
  held-out horizon predicted by the mean of training horizons with the same
  unit; no activations; 2026-10-13). Only the margin above 0.96 shows
  magnitude within a unit. The k = 1 linear probe at the suffix (0.90) is
  below a unit lookup. Within-unit R² (`12` §1): on the 48 horizon averages
  the suffix k = 1 direction orders horizons within a unit at 0.50
  (borderline, 95% CI 0.20–0.67, driven by the year horizons; MLP 0.75); per
  prompt it is mostly the unit word. A leave-one-horizon-out unit lookup
  scores about -0.25 within unit (averages and per prompt), not 0, so that is
  the floor for "unit only".
- In the middle of the think block the horizon is barely readable (R² 0.32 at
  think@0.25, 0.28 at 0.5). On natural closes it is already back at the last
  think token (0.96; 0.82 over all prompts, where forced think@1 is a
  mid-sentence cut), and stays at the `</think>` (0.96) and the pre-label
  token (0.98), in 2-4 curved dimensions plus a long tail.
- Forced closes confound every thinking-on cell after the block: the top
  horizon-average component at `</think>` separates forced from natural
  closes with AUC 1.00 (0.88 at the pre-label token).
- Starter bank (v3, `history/03_results_think_geometry_starter.md`): a commit
  phrase appears at a median 34% of the block and nothing in the horizon
  reading changes there; the think-block probe direction is nearly orthogonal
  to the suffix one (cosine 0.09-0.15); the chosen label is readable from the
  suffix before reasoning at 0.91 (fixed delays, so label and horizon were confounded).

- **Units (E9):** before reasoning the readout tracks duration across units
  (unit share 0.02 at the suffix, L16), but a rewrite moves it by 0.30 decades
  on average ("1 century" read as ~16 years, "3600 seconds" as ~5 hours).

**Causal path**
- The decision reaches the answer slot (pre-label `:`) at L22-24: patching
  that one position from a prompt in the other regime transfers 0.22 at L22,
  0.56 at L24, 0.82 at L34. Reasoning does not move where the decision is
  read from (thinking off: 0.58 at L24).
- The positions where the horizon decodes best are not on the causal path
  (run 1: all 12 verdicts null; the suffix newline decodes at 1.00 and
  carries 0.00 of the effect in thinking-off patching).
- **Probe line:** moving along the ridge probe direction shifts its readout
  by 18-29 decades and the choice by at most 0.01 C. Inert at every cell.
- **Curve at L24:** moving along the curve of horizon averages by 2 decades
  shifts the choice 0.15 C toward the rational answer (both-regime prompts
  +0.15 at -2 decades). Two caveats from the 2026-10-13 review:
  near-regime prompts move toward far at both -2 and +2 decades (-0.14 each),
  so part of the effect is non-specific; and the L24 subspace decodes the
  near-vs-far log-odds at R² 0.65 (the a-vs-b label only at 0.38 with per-fold PCs), so the
  curve may carry the decision rather than the horizon.
- **Mean difference (CAA)** at L24 steers the wrong way (+0.13 C toward near
  at +1 decade). Suspected cause: long horizons are force-closed more often,
  so long-minus-short partly reads "was cut off".
- **Run 3 (L24, natural closes):** the curve without the decision direction
  still moves the choice (0.07 C), so the pre-registered criteria pass. The
  ablation split (0.11 without the direction, 0.09 for it alone) is not a
  split: only 0.25 of the direction lies in the ablated top 4 (`08` §9). The
  subspace without that direction still predicts the decision at R² 0.52.
  The post-hoc within-edit check (0.03 of a real regime change vs 0.02 for a
  pure decision push, overlapping intervals) cannot separate "horizon
  compared with delays" from "average decision per horizon"; both readings
  stay open. The mean difference steers backwards
  because the short half of the horizons is all `none`-regime.
- **L22 (2026-10-18):** the same tests one layer earlier fail all three
  criteria; ablating the horizon subspace removes 0.09 C (L24: 0.30).
- **Before reasoning (E10, 2026-10-18):** swapping the top-8 horizon
  subspace at the phrase's last token (the unit word) at L12/L16 makes the
  reasoning write the recipient's number with the donor's unit in 68-70 of
  the 71 pairs whose units differ ("59 years" read as "59 days"; random swap
  2 of 71), and the choice follows: 29-30 of 46 label flips where units
  differ (random 5-8), 0 of 6 where they match. (The earlier 71-73% divided by
  all 96 pairs, including 25 that share a unit.) The subspace is causally the
  unit, reasoning reads it back as a written horizon, and nothing here tests
  magnitude within a unit. Two limits: the random swap is not size-matched to
  the 8-dim subspace swap, and re-batching the unpatched prompts alone changes
  12 of 96 labels against the E0 cache.
- **Ablation at L27/L30** removes 0.74-0.83 C, but the subspace there decodes
  the a-vs-b decision at 0.95, so this says nothing about the horizon.

## 5. Next (proposed, thinking on only; each needs an approved spec before code)

1. ~~Local: natural closes only, within-unit R², grouped inner CV~~ done
   2026-10-21 (`12` §1).
2. **Local, saved E8 results:** split the per-prompt curve effects
   (`toward` in `exp8_Qwen3-8B_answer_slot/results.json`) by whether the
   shifted horizon crosses a delay. A horizon representation should change
   the choice only when it does.
3. ~~Box: L24 curve split into decision and rest~~ done 2026-10-14 (`08` §9).
4. ~~E10, patch before reasoning~~ done 2026-10-18 (`08` §11): the
   last-token patch swaps the unit. Follow-up (box, ~30 min on 2 GPUs):
   same-unit pairs only (e.g. "3 years" vs "39 years"), patching the number
   tokens, and the top-8 subspace over the whole phrase. This is the causal
   version of the within-unit question.
5. ~~E9 unit rewrites~~ done 2026-10-14 (`08` §10). Still open at the
   suffix: E2 with Alan's 4 phrasings (forward pass only, minutes).
6. **Box, from the 2026-10-26 audit** (each needs a spec):
   - the decision split with the decision direction projected into the
     ablated top k, so the three ablation rows split one subspace (8B L24,
     14B off L27 first);
   - T1 rerun with the fixed horizon span (whole phrase), to check the
     `encode_end` / `spread_peak` markers behind S1;
   - size-matched random controls for the E10 swap and for ablation.

## 6. Decisions, dated

- **2026-09-21.** One model, Qwen3-8B. Standalone package in `jared/`.
- **2026-09-23.** Think block force-closed at `max_think_tokens` (3,072);
  generation batched (left-padded greedy).
- **2026-10-02.** The full bank (Alan's generator, `spar_horizon/full_bank.py`);
  the label is located after the model's own `I choose:`, no prefill.
  Full-bank results are reported as a replication next to the paper
  (Qwen3-4B), Alan and Augusto (Qwen3-14B, thinking off).
- **2026-10-03.** 48 log-spaced horizons; three think fractions (0.25, 0.5, 1);
  E7 probes leave-one-horizon-out on saved activations.
- **2026-10-11.** E8 success criteria fixed before each run (`07` §5, §9).
- **2026-10-14. Thinking on only for Qwen3-8B.** With thinking off the
  model separates the regimes (contrast 11.4 log-odds vs 12.9 on), but option
  order adds +6.3 log-odds (vs +2.5 on) and the dependence on the exact
  horizon within a regime flips sign with option order, so there is no
  coherent per-horizon choice to explain. Thinking-off results stay as a
  record; nothing new runs in that mode. Cost: the prompt-side patching map
  (T1) exists only for thinking off, since patching prompt positions with
  thinking on changes the reasoning that follows.
- **2026-10-20.** E11 (`10`): the 8B map on Qwen3-14B, both modes.
- **2026-10-25.** Unit-overlap bank (`14`): 8B only, 14B dropped for now.
- **2026-10-26. Audit of docs and code; struck or corrected:**
  - "The suffix code is shared across models" (`11` §2, `13` C1): the null
    permuted horizons across prompts, so it only tested same vs different
    prompts. Against a null that keeps prompts grouped in both models it is
    not significant.
  - The decision-split ablation readings (`08` §9, `11` §3, `13` B2): the
    decision direction lies mostly outside the ablated top k. Verdicts kept.
  - "14B binds the label four layers after the hand-off, thinking off" (`11`
    §1): 53 rows scored the wrong label tokens (`**b)**`); without them bind
    is one layer after the hand-off, as on 8B.
  - E10 "71-73%": wrong denominator; 96-99% over pairs whose units differ.
  - The `08` §9 "within-edit" check as decisive: post hoc, inconclusive;
    "follows the pattern of real behavior": not on both->none.
  - think@1 "recovers" (0.81): 0.96 on natural closes.
  - "Unit only" means within-unit R² near -0.25, not 0.
  - Code fixed: label logits for unspaced labels, write-once reasoning cache,
    E10 rates, T1 horizon span, per-fold refit in the decision check,
    prompt-preserving alignment null on shared natural closes,
    within-unit grouping by (value, unit).
- **2026-10-30.** E14 (`21`): thinking-on steering moves to the horizon phrase,
  with a target per prompt that crosses a delay (log-midpoint), along the
  pooled manifold, with the donor patch as the ceiling.
- **2026-11-01.** E14 (`22`): H1-H3 fail. Post hoc measures (first stated
  horizon, choice by where it lands) were added to `exp14_analysis.py`, labelled
  post hoc and gating nothing. The pre-registered H1 measure was kept as written.
- **2026-11-06.** Overnight run (`26`): choice4 and choice8 banks run with reduced activations. The scan
  stopped at 34 of 40 cells (L26 has only the number-token pair); the screen found no cell at excess ≥ 4,
  and random controls matched the manifold cells. Extras were not run. Follow-ups, each needing a spec:
  the L26 gap, explicit-ROI banks, a probe of the choice banks with the regime held fixed.

## 7. Facts that constrain the design

- Think length on the full bank: median 1,748 tokens; E0 thinking on took
  124 min at batch 12. On the cached reasoning (batch 8, last-position
  logits only), one thinking-on E8 cell took 25 min (ablation 7, steering
  19) and answer-slot patching over all layers 12 min.
- bf16 batched greedy is not bit-stable across batch composition (a few
  prompts change between identical runs).
- Activations: about 6 GB per mode, float32 (Qwen outlier dimensions exceed
  float16); load per layer.
- The local Qwen3.5-0.8B never closes its think block and answers by
  reflex: plumbing checks only. Model-backed checks go to Jared via
  `local_checks.py` (see `../../CLAUDE.md`).
- The unit word grows with the duration by design, so unit and log horizon
  are confounded on the bank (unit explains 0.97 of log horizon).
  Consequence (30, addendum): the pooled fit's unit adjustment creates the PC1 slope of the manifold (raw per-horizon
  means are flat, fitted slope -9.7 per decade); same-unit steps are unaffected.

## 8. Deferred

R2 (domain), the multiturn design (`history/08_spec_multiturn_draft.md`),
E5 attribution on the full bank (superseded by patching), Alan's extended
grid, and the replication against Alan's public 14B activations.

## History

| file | what it was |
|---|---|
| `history/01_plan_2026-10-11.md`, `history/00_README_2026-10-12.md` | previous plan and index, both modes |
| `history/02_spec_full_bank.md` | spec for the full bank and its first box run; built and run |
| `history/03_results_think_geometry_starter.md` | thinking-on results on the starter bank (v3); numbers superseded by `06` |
| `history/05_results_probes_v3.md` | E7 on v3; superseded by `06` |
| `history/01_plan_2026-09-21.md`, `history/02_execution_plan_2026-09-23.md` | original plan and E0-E5 build plan |
| `history/03_baseline_qwen3.5-0.8b.md` | the starter on the 0.8B |
| `history/05_prompt_think_geometry.md`, `history/06_spec_think_geometry.md` | mid-thinking positions, built |
| `history/08_spec_multiturn_draft.md` | multiturn draft, never run |
