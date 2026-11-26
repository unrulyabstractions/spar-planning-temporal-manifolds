# Plan: improving the evaluation pipeline

Started 2026-11-19. Replaces the Qwen3-8B plan, now `history/01_plan_2026-11-19.md`. Docs dated before
2026-11-19 that cite "`01_plan.md` §N" mean that archived file.

## 1. Goal

Make the team's evaluation pipeline (`../evaluation/`, code; `../evaluation_design/README.md`, design; owner
Augusto) give a pass/fail that means something, and add eval prompt types beyond the T0 bank.

Not in scope here: steering extraction and fitting (`../steering/`, Alan).

## 2. Fixed setup (decided 2026-11-19)

- **Where changes go:** `../evaluation/` directly, on branches off `main`, as PRs for Augusto to review.
- **Development model:** `Qwen/Qwen3-1.7B`, thinking on, as in Augusto's example report
  (`../evaluation/reports/Qwen3-1.7B__t0_s0__think_report.md`), so changes can be compared against it. The
  target stays Gemma 4 31B, layer 38 of 60, thinking on.
- **Focus areas:** (a) statistical power and validity, (b) new eval prompt types. Speed and the §11 open
  decisions of the design are worked on only where (a) or (b) need them.

## 3. Starting point

Augusto's example run (Qwen3-1.7B, thinking on, about 10 h on an 8 GB GPU) is plumbing, not evidence:

- 79 of 200 verification items are readable.
- Calibration does not reach its 1-logit target, because the mean-difference (CAA) vector tops out at
  +0.27 logits.
- Only the `choice` control beats the nulls (p = 0.05).
- With greedy decoding, 323 of 337 runs that hit the cap are loops. Sampling (now the default for
  thinking-on baselines in `prompt_bank/baseline.py`; `--decoding greedy` restores greedy) answers 62 of 100
  of the looping prompts.

So a method that fails the current pipeline cannot be told apart from a test that is too weak.

## 4. Candidate work (not approved; each needs a spec before code)

| # | Item | Area |
|---|---|---|
| P1 | Positive control with a known large effect (e.g. a direction along the answer-label logits, or patching a long-horizon activation into a short-horizon prompt), and a power check: the smallest effect the design detects with 19 and 99 draws | (a) |
| P2 | Continuous outcome: near-minus-far label log-odds at the token before the label, after force-closing the think block at a budget (the Qwen3-8B method), so no item is lost to parsing. Effects reported as a fraction of the regime contrast, or as decades of implied horizon | (a) |
| P3 | Dose response: a few edit strengths α, test the slope against the nulls' slopes (one test instead of one per α) | (a) |
| P4 | Edit position vs thinking: also edit at the think-block close, since the horizon representation is weak mid-think and re-forms at the close (archived plan §4, R6) | (a) |
| P5 | Eval prompt types from `EvalsSpec.md` and `EvalsPromptTypes.md` (implicit horizons, domains, roles, distractors) as the surface-form check C5 and a held-out-domain test | (b) |
| P6 | Settle the design's §11 decisions that P1-P3 depend on: pass rule, label-shuffle strata, which confound checks gate a pass | (a) |

## 5. Reviews (2026-11-19)

- **Speed of `evaluation/`** (estimates for Gemma 31B, not measured). Generation dominates: about 700 GPU-h
  per (layer, site) at 19 draws, against about 33 GPU-h for replay at K = 99.
  - Pure-engineering fixes, roughly 2-3× each:
    - stop looping rows early and refill the batch;
    - cache the layer-38 residual for replay;
    - `logits_to_keep=1`.
  - These change the statistics and need a decision:
    - generate only for what passes the replay screen;
    - use the same sampling seed across arms.
- **Correctness of `evaluation/`:** 22 of 22 tests pass. The tests don't cover `rank_p`, calibration or
  `analyze`. These findings bear on (a):
  - **Sign handling in `rank_p`** (`nulls.py:104`): it compares the signed statistic with |null|. A negative
    effect always gets p = 1, and a positive one gets a two-sided p. In the example report, CAA gets p = 1.00
    in every family. Ask Augusto whether the test is meant to be one-sided.
  - **Number of draws:** replay runs with K = 19 (`run_local.sh`, `--k` default), not the K = 99 of design
    §6.4, so the smallest possible p is 0.05.
  - **Draws depend on K:** the first rows of the shared draws change with K (Box-Muller layout), and
    `in_span` draws are regenerated rather than stored.
  - **Null routing differs from design §6.3:** curves are never compared with N1/N2, `in_span` is compared
    only with push methods, and N6 is missing.
  - **Calibration:** the health bound uses the mean held-out mass, not the worst item as the README says.
    The example run never reached the 1-logit target (x = 0.02, `reached: False`).
  - **Stale caches on resume:** the per-arm caches and `calibration.json` are not keyed by x, K or the item
    set, so a resumed run with different settings can reuse old results.

## 6. Decisions, dated

- 2026-11-19: focus moves from the Qwen3-8B representation study to the evaluation pipeline. Areas (a) and
  (b); changes go to `../evaluation/` as PRs; development model Qwen3-1.7B, thinking on.
