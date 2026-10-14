# Plan and status: the time-horizon representation in Qwen3-8B, thinking on

Last revised 2026-10-14. The question, what is done (§3: do not rerun it),
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
| R3 | Encodes duration, not the unit word | open, and now the main threat to the probe results (§4) |
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
| E0 full bank, thinking on (generation, activations, behavior) | 2026-10-04 | `exp0_Qwen3-8B_think-on_full/`, `gen_cache/` | `06_results_probes_full.md` |
| E0 full bank, thinking off (record only) | 2026-10-04 | `exp0_Qwen3-8B_think-off_full/` | same |
| E7 linear vs MLP probes, both modes | 2026-10-04 | `analysis/probe_*` | `06_results_probes_full.md`, spec `04_spec_probes.md` |
| E8 run 1: patching map (thinking off), 4 decodability-chosen cells: all null | 2026-10-11 | `exp8_Qwen3-8B_full/` | `08_results_steering.md` §1-3, spec `07_spec_steering.md` |
| E8 run 2: answer-slot ablation and steering at L24/27/30, answer-slot patching across prompts, both modes | 2026-10-12 | `exp8_Qwen3-8B_answer_slot/` | `08_results_steering.md` §5 |
| Horizon-vs-decision check on the answer-slot subspace (local) | 2026-10-12 | `analysis/steering_decision_check.*` | `08_results_steering.md` §6 |
| Starter bank: v1 (thinking off, E0-E5), v3 (thinking on, E0, E6) | 2026-09 | `snapshots/v1_noprefill`, `snapshots/v3_think` | `history/` |

## 4. Established (Qwen3-8B, thinking on, full bank unless noted)

**Representation**
- Before reasoning the horizon is readable almost perfectly: suffix newline
  after `<|im_end|>` R² 1.00 (L16), leave-one-horizon-out.
- **But the unit word alone scores R² 0.96 under the same protocol** (each
  held-out horizon predicted by the mean of training horizons with the same
  unit; no activations; 2026-10-13). Only the margin above 0.96 shows
  magnitude within a unit. The k = 1 linear probe at the suffix (0.90) is
  below a unit lookup. The "one curved direction" reading in `06` is open
  until within-unit R² is computed.
- In the middle of the think block the horizon is barely readable (R² 0.32 at
  think@0.25, 0.28 at 0.5); it recovers at the last think token (0.81), the
  `</think>` (0.96) and the pre-label token (0.98), in 2-4 curved
  dimensions plus a long tail.
- Forced closes confound every thinking-on cell after the block: the top
  horizon-average component at `</think>` separates forced from natural
  closes with AUC 1.00 (0.88 at the pre-label token).
- Starter bank (v3, `history/03_results_think_geometry_starter.md`): a commit
  phrase appears at a median 34% of the block and nothing in the horizon
  reading changes there; the think-block probe direction is nearly orthogonal
  to the suffix one (cosine 0.09-0.15); the chosen label is readable from the
  suffix before reasoning at 0.91 (fixed delays, so label and horizon were confounded).

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
  near-vs-far log-odds at R² 0.65 (the a-vs-b label only at 0.23), so the
  curve may carry the decision rather than the horizon.
- **Mean difference (CAA)** at L24 steers the wrong way (+0.13 C toward near
  at +1 decade). Suspected cause: long horizons are force-closed more often,
  so long-minus-short partly reads "was cut off".
- **Ablation at L27/L30** removes 0.74-0.83 C, but the subspace there decodes
  the a-vs-b decision at 0.95, so this says nothing about the horizon.

## 5. Next (proposed, thinking on only; each needs an approved spec before code)

1. **Local, saved activations:** probes on natural closes only
   (`probe_dimensionality.py --natural-only`, built, not run) and
   within-unit R² at the suffix, `</think>` and pre-label cells, with the
   ridge's inner CV grouped by horizon.
2. **Local, saved E8 results:** split the per-prompt curve effects
   (`toward` in `exp8_Qwen3-8B_answer_slot/results.json`) by whether the
   shifted horizon crosses a delay. A horizon representation should change
   the choice only when it does.
3. **Box, cached reasoning:** at L24, steer with the curve minus its
   near-vs-far component, and with that component alone at matched norm;
   directions refit on natural closes. Code gap: `exp8_steering.py --tests
   answer` also builds the thinking-off set; it needs a mode filter.
4. **Box, regenerates reasoning (~1 h):** steer before reasoning, at the
   prompt positions after the horizon around L15-19 where the horizon flows
   in thinking-off patching, then let the model reason. This is the only test
   of whether reasoning uses the horizon code. (Called T5 in `07` §8; T5 in
   `07` §9 is the answer-slot patch. Rename it when specced.)
5. **Representation tests at the suffix (forward pass only, no reasoning):**
   E3 with Alan's unit rewrites (same duration, different unit), which is the
   direct test of R3, and E2 with his 4 phrasings.

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
