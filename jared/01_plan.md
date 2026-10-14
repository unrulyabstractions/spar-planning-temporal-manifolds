# Plan and status: the time-horizon representation in Qwen3-8B, thinking on

Last revised 2026-10-02. One file for the question, the experiments, the
decisions, and what is done. Superseded drafts are in `history/` and do not
need reading; the facts that still matter from them are summarized here.

## 1. Research question

**Is the linearly-readable time-horizon representation a stable, prompt-
independent quantity, and does the model's choice causally depend on it?**
And, new to this project: what happens to it while the model reasons?

| # | Sub-question | Manipulation | Metric | Pass |
|---|---|---|---|---|
| R1 | Survives rewording | Constraint phrasings | Probe trained on phrasing A, tested on B: rho, R² | rho > 0.8 |
| R2 | Survives domain change | Investment vs. climate (and others) | Cross-domain transfer | rho > 0.7 |
| R3 | Encodes duration, not unit | "1 year" / "12 months" / "365 days" | Within-duration vs between-duration spread on the axis | unit variance < 20% |
| R4 | Invariant to nuisance | Option order, labels, rewards | PC1 rho with reward near 0; horizon rho stable | reward rho < 0.2 |
| R5 | Causally used | Add ±alpha × horizon direction at the probe layer | Choice flip rate vs alpha, random-direction control | monotone, control flat |
| R6 | Persists through reasoning | Thinking on: positions inside the think block | Probe R² and rho per think fraction; label readable at the think end | see `03_results_think_geometry.md` |

R1-R4 and R6 are representational. R5 turns it from descriptive into a finding.

## 2. Where this comes from, and what it must be compared to

The starter (`starter/spar_starter_geometry.py`) distills the prior project
(`../../temporal-awareness/`, Rios-Sialer et al. 2026, Qwen3-4B-Instruct-2507):
the constraint sentence, PCA per (layer, token) against log horizon, turn-suffix
and response positions, and the three coherence tests. The paper localized the
horizon signal to late-layer attention (L21, L24) and steered it with CAA at
L19-22.

Two sibling efforts in this repo use the prior project's **structured prompt
format** on Qwen3-14B with thinking off:

- **Alan** (`../alan/`): 2,000-prompt runs with randomized option delays and
  rewards, 17-point horizon grid, four constraint phrasings, unit rewrites, a
  climate domain, and relevance controls. PC1 |rho| 0.96 at the `assistant`
  token (L36-37); choice of the short option falls smoothly from 92% at 1 day
  to 15% at 50 years.
- **Augusto** (`../relevance_aware_decoder/`): Alan's matrix plus distractor
  durations, to ask whether a linear decoder can separate "the horizon" from
  "a duration".

**Replication note.** From the full bank onward our prompts are in that same
format (`spar_horizon/full_bank.py` is Alan's generator, copied). Every
full-bank result should therefore be reported next to the matching number in
the paper (Qwen3-4B), Alan's runs (Qwen3-14B, thinking off), and Augusto's
where applicable, as a replication on Qwen3-8B with thinking on and off. The
starter-bank results (snapshots `v1_noprefill`, `v3_think`) are not comparable
to theirs: different format, fixed delays, near option always first.

## 3. Model

Qwen3-8B (`Qwen/Qwen3-8B`): native `enable_thinking` toggle, bf16 fits the
vast.ai 4090, ungated, same family as the paper's model. Llama-3.1-8B and
Gemma-2-9B have no native think block; Qwen3-14B needs 8-bit on a 4090;
Qwen3.5-27B's hybrid attention has no plain residual stream. The local
Qwen3.5-0.8B is for tests and smoke runs only: it never closes its think
block and its answers are position reflexes (see §6).

## 4. Prompt banks

| bank | `--bank` | prompts | format | answer slot | status |
|---|---|---|---|---|---|
| starter | `starter` (default) | 116: 16 horizons × 3 reward pairs × 2 fixed delay pairs + 20 no-horizon | bare 5 lines | prefill `I choose:` | frozen; the v1/v3 snapshots |
| full | `full` | 500 by default from Alan's generator: delays and rewards drawn per prompt, option order randomized, 48 log-spaced horizons (30 s to 500 y) in natural units | prior project's SITUATION/TASK/OBJECTIVE/CONSTRAINT/ACTION/FORMAT | none; the model writes `I choose:` and the label is located after it | built 2026-10-02, not yet run on the box |

Why the full bank (details in `02_spec_full_bank.md`): with fixed delays the
sensible answer is a step function of the horizon, so the label probe and the
horizon probe read the same variable; the near option was always a), so
position bias and the rational choice coincided; the bare format is comparable
to nobody; and three of the 16 horizons were rendered ungrammatically.

Each full-bank record carries `regime` (`none`: horizon shorter than the near
delay; `near`: between the delays; `both`: past the far delay; `null`) and
`truth` (`near` / `far` where one is defined). The question is ill-posed in
the `none` regime and open-ended in `both` (no interest rate is given); the
model says so in its reasoning. That is inherited from the paper and from
Alan's design, and is a reason to report behavior per regime.

## 5. Experiments and status

| | Name | Question | starter bank | full bank |
|---|---|---|---|---|
| E0 | baseline | Peak and probe layer; PCA 2D/3D; with thinking on the think-span positions, probe track, dense track, label probe | done, both modes (`v3_think`) | done 2026-10-04, both modes, `results/horizon/` (thinking on 124 min, 103/500 forced) |
| E1 | nuisance | Is PC1 horizon, not reward or delay? | done, off | Phase 2 |
| E2 | paraphrase | Probe transfer across constraint wordings | done, off (6 wordings) | Phase 2, with Alan's 4 phrasings |
| E3 | units | Same duration, different units | done, off | Phase 2, with Alan's unit rewrites |
| E4 | perturb | Distractor, horizon first, implicit horizon | done, off | Phase 2; perturbations need the six-line layout |
| E5 | attribution | Which layer/position/component carries the horizon to the answer | done, off | not planned yet |
| E6 | think PCA | PCA of the whole think block | done, on (0.8B and 4B logs) | Phase 1 if time allows |
| R5 / E8 | patching, ablation, steering | Where is the horizon causally carried; is the low-dimensional code necessary and sufficient; does the curve beat the line; does the end-of-reasoning code drive the answer | not planned | run 2026-10-11, `08_results_steering.md`: pre-registered cells null; patching locates the path (answer slot from L24) |
| E7 | probes | Is the horizon extractable, linearly or only nonlinearly, and in how many dimensions? Linear and MLP probes on the top-k PCs, k increased step by step, leave-one-horizon-out (requirement from Jared, 2026-10-03; spec `04_spec_probes.md`) | done on v3, `05_results_probes_v3.md` | done, `06_results_probes_full.md` |
| multiturn | horizon across turns | override and carry-over | draft only (`history/08_spec_multiturn_draft.md`) | deferred |

Results so far: `03_results_think_geometry.md` (Qwen3-8B, thinking on, starter
bank). Earlier runs: `snapshots/`.

## 6. Decisions, dated

- **2026-09-21.** One model, Qwen3-8B. Thinking on is the primary condition,
  off is the control. Standalone package in `jared/`, not the prior repo.
- **2026-09-23.** E2 uses constraint paraphrases only. E4 adds implicit
  horizons and two structural perturbations. E5 is attribution patching with
  our own hooks, thinking off only. Layers for E2/E3: E0's probe layer ± 2.
- **2026-09-23, box.** Every 8B prompt overran a 1024-token think budget and
  short horizons loop ("neither is feasible, maybe a typo"), so the think
  block is **force-closed** at `max_think_tokens` (3072) and the forced
  count is recorded; generation is **batched** (left-padded greedy, verified
  token-identical to single-prompt).
- **2026-09-24.** Qwen3-8B thinking off writes a paragraph before its choice,
  so the starter bank uses the assistant **prefill** `I choose:`.
- **2026-09-30.** Mid-thinking positions: kept layout `suffix | think | pre |
  prelabel | answer`, four think fractions, explicit `regions` everywhere;
  forced prompts are flagged, not skipped. Spec: `history/06_spec_think_geometry.md`.
- **2026-10-02.** The **full bank** (Alan's generator, copied into
  `spar_horizon/full_bank.py` and `horizons.py`). No prefill: the label is
  located after the model's own `I choose:` as Alan parses it. 500 prompts
  for the first thinking-on run. Our 16 horizons kept, in natural units.
  E1-E4 move to the full bank after Phase 1. Results to be framed as a
  replication (§2).
- **2026-10-03.** E7 probes (`04_spec_probes.md`): log-horizon regression,
  linear vs MLP on the top-k PCs, leave-one-horizon-out, run after the fact
  on saved activations. For the box run: 48 log-spaced horizons instead of
  16, and three think fractions (0.25, 0.5, 1) instead of four.

## 7. Facts that constrain the design

- Think length on the 8B: median ~2.8k tokens, 46-48 of 116 forced at 3072;
  the forced prompts are mostly the long horizons (22 of 30 at ≥ 10 years).
- bf16 batched greedy is not bit-stable across batch composition (two prompts
  changed between identical runs).
- The 0.8B reproduces the geometry (|rho| 0.955) but its answers are fixed
  reflexes ("a)" for letters, "2)" for digits), so its behavior numbers mean
  nothing; it is a plumbing check only (`history/03_baseline_qwen3.5-0.8b.md`).
- Each kept position costs 116 × 37 × 4096 × 4 bytes ≈ 70 MB in the npz on
  the starter bank; scale by prompt count for the full bank.
- Run time (2026-10-03): behavior regeneration was 96% of the v3 wall
  time. Now position bias and per-regime accuracy come from the base
  answers (`base_behavior`), the stability tests take `--behavior-n`, and
  the default batch is 12 with an out-of-memory step-down. Details in
  `02_spec_full_bank.md` §3.1.
- Qwen3-4B on the full bank (8 prompts, 2026-10-03): think lengths and the
  forced-at-long-horizons pattern match the 8B on the starter bank, so the
  3072 budget and the 500-prompt estimate in `02_spec_full_bank.md` stand.
- Alan found that rendering the same horizon in a new unit shifts both
  geometry and behavior (11 points more long-option choices for "decades"),
  which is why the full bank keeps one rendering per horizon by default.

## 8. Deferred

R2 beyond the climate domain, R5 steering, the multiturn design, E5 on the
full bank, and Alan's extended grid (1 minute to 10,000 years).

## History

| file | what it was | why it is in history |
|---|---|---|
| `history/01_plan_2026-09-21.md` | original research plan | merged into this file |
| `history/02_execution_plan_2026-09-23.md` | build plan for E0-E5 with task checklists and build notes | all tasks done; notes merged into §6-7 |
| `history/03_baseline_qwen3.5-0.8b.md` | the starter on the 0.8B | one-line summary in §7 |
| `history/05_prompt_think_geometry.md`, `history/06_spec_think_geometry.md` | task prompt and spec for the mid-thinking positions | built; results in `03_results_think_geometry.md` |
| `history/08_spec_multiturn_draft.md` | multiturn spec, draft | never run; its counterbalancing point is addressed by the full bank |
