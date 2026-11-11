# Spec: overnight multichoice behavior and phrase-steering scan (2026-11-06)

Status: **approved 2026-11-06 with the amendments in §6.** Qwen3-8B, thinking on, one rented
RTX 4090/5090 at <= $0.60/hr. Budget: **$8 hard cap**, expected $2-3. Target: `just down` by $6, with no hard stop there (Jared, 2026-11-06); I may go past $6 to finish a unit of work, never past $8 (down by $7.50 to leave margin).

## 1. Part A: 4- and 8-choice banks, behavior only

Spec `18_spec_multichoice.md` unchanged, with these cuts (Jared, 2026-11-06):
- banks `choice4` and `choice8` (implicit ROI), **~250 prompts each**: `per_duration` 2 instead of 4
  (1 `near` + 1 `crossed` base per duration, ~86 bases x ~2.8 renderings);
- **behavior plus a reduced activation set** (amended, §6): no all-layer dump;
- tasks 0-4 and 6 of spec 18 (dependency, bank, wiring, labels, `multichoice_behavior`,
  `queue_multichoice.sh`); task 5, the 0.8B smoke, runs **on the box** and is plumbing only
  (the 0.8B never closes its think block);
- results go to `26_results_overnight_choice_scan.md`, not `19_...`.

Tests: the fast suite (`pytest -q tests --ignore=tests/test_generation.py`, mp env) locally.

## 2. Part B: 2-choice investment steering scan, thinking on

Built from `exp14_phrase_steer.py`. A screen, not a test: no H1-H3.

**Cells (40):** layers L10/14/18/22/26 x positions x directions.
- positions: number tokens, unit token (phrase, prefill), suffix newline (prefill), answer slot
  (pre-label `:`, scoring pass only);
- directions: the manifold step to the per-prompt target (spec 21 §2), and a random vector
  matched in norm to that cell's step.

**Prompts:** 10 per cell, the same 10 in every cell (paired): 5 none->near and 5 both->near, drawn
from the E14 eligible set (seed 0). Targets as in spec 21 §2.

**Noise:** the unsteered baseline plus 2 no-op reruns on the same 10 prompts (batch composition
changes between them; E10 found 12/96 labels change from batching alone).

**Measure:** a flip is a label different from the baseline label. Per cell: flips toward the target
direction (more near), flips in total, and the change in near log-odds. Excess = cell flips minus
the mean of the two no-op reruns' flips. **A cell is reported if excess toward-target flips >= 4**,
and next to its random twin and the noise flips. With 10 prompts and ~12% noise (~1.2 flips), this
catches large effects only; nothing in it is a significance claim.

**Manifold fits.**
- phrase positions: refit for L10/22/26 on the box with `exp14_phrase_states.py` (add
  `--layers`, writing to a new states file; the existing one has L12/14/16/18 and its skip rule
  would otherwise skip the run). L14/L18 reuse the existing states.
- suffix newline and answer slot: fitted on CPU from the saved E0 activations in
  `SPAR/results/horizon/` (all layers kept there), then shipped in the inputs file.
- Answer-slot cells run on the baseline reasoning without regenerating (E8 style), so they are
  cheap; all other cells regenerate with the steer on every prefill pass.

**Extras (in order, only if the remaining budget to the $6 down-line is > $2):**
1. `choice4_roi`, `choice8_roi`, ~250 each; 2. decision-direction and CAA at the top 3-5 cells;
3. alpha sweep 0.5/1/2 at the top cells with full controls; 4. a second prompt seed for the top cells.

## 3. Cost

At ~$0.50/hr (a 4090; a 5090 is faster):
| step | hours |
|---|---|
| provision, push, 0.8B smoke | 0.4 |
| Part A, 2 banks x ~250 prompts (~1.1 h each, cap-bound batches of 12) | 2.2 |
| Part B, ~430 generations (36 regenerating cells x 10 + baseline + 2 reruns) at ~13 s | 1.6 |
| phrase states at 3 new layers, sync | 0.3 |
| **total** | **~4.5 h = $2.3-3** |

`just burn` hourly; `just down` before $6, report what was spent. The hard cap is applied by me
against `just status`'s real $/hr, not `BUDGET_USD` in `.env` ($50, which only feeds the warnings).

## 4. Order

Local code and tests -> `just up` -> push, smoke on the box -> Part A (tmux) -> Part B (tmux) ->
extras -> `just sync` -> `just down`. Then write `26_results_overnight_choice_scan.md` and update
`01_plan.md`.

## 5. Where this departs from your message or the specs (need your OK)

1. **10 prompts per cell = 5 + 5** across the two transitions, the same in all 40 cells. If you
   meant 10 per transition (20 per cell), Part B doubles to ~3 h and the total goes to ~$4-5.
2. **"Flip" and "beyond noise"** as defined in §2 (excess over the mean of 2 no-op reruns, toward
   the target). Answer-slot cells measure only the score on fixed reasoning, so their flips are not
   comparable in kind to the regenerating cells.
3. **Positions beyond spec 21:** suffix and answer slot manifolds come from the saved E0
   activations (CPU fit), not from the box.
4. **"> $2 remaining"** for extras means $6 minus spend, not the `.env` budget.
5. **Running model-backed work from the session** (smoke and runs, all via the box) is as you
   instructed; it overrides the "hand Jared a script" rule in `SPAR/CLAUDE.md` for this session only.
6. **Housekeeping:** `git status` in this checkout shows staged deletions of dotfiles under `jared/`
   (`.bashrc`, `.gitconfig`, ...) that are not mine. I will not touch or commit them.

## 6. Amendments (Jared, 2026-11-06)

**Activations for the choice banks (reduced set), analysed on the box.**
- Layers 10, 14, 16, 18, 22, 24, 26, 30, float32. About 1.2 GB per ~250-prompt bank.
- Positions: the suffix tokens, think@0.25/0.5/1, `</think>` and following tokens, pre-label and
  answer (as E0), plus **new prompt-side positions**: the horizon phrase (number tokens mean, unit
  token) and, for every option, its label token (`a)`) and its line-end `.`. Those option positions
  are my guess at where per-option value could be built (return, delay, rank in half).
- `pipeline/s3_analysis/by_label/choice_structure.py` runs on the box on the saved chunks and writes small json/png:
  - grouped CV probes (held out by base, so renderings never straddle folds) for: half chosen,
    chosen option's position, truth option, log horizon, regime;
  - at option positions: probes for that option's return %, log delay, half, rank in half, and
    "is the chosen option"; whether it is readable before reasoning (suffix / option positions) vs
    after (pre-label);
  - horizon PCA at the suffix and phrase, unit share of variance, rendering invariance;
  - forced vs natural closes, and position bias of the chosen label.
- I bring back the summaries and, if under ~3 GB, the activation chunks for later local analysis.

**Modular, stoppable runs.**
- Part A runs E0 in chunks of 48 prompts, written as `chunk_NNN.npz`/`.json` as each finishes,
  skipped on a rerun. Prompts are shuffled with a fixed seed, so any prefix of chunks is balanced
  over duration and regime and a run stopped midway is usable. The generation cache stays too.
- Part B's unit is one cell (a manifold cell and its random twin run back to back; L18, L14, L22,
  L10, L26 order, so the most-motivated layers finish first). Each cell is saved when it ends and
  skipped on a rerun.
- Analysis scripts take whatever chunks/cells exist and report n.
