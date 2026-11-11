# Spec: the full prompt bank (Alan's generator, no prefill) and the Phase 1 run

Date: 2026-10-02. Decisions from Jared are recorded in §6; Phase 1 code is
built (§5), the box run is not. Companion to `01_plan.md` (status, decisions)
and `03_results_think_geometry.md` (the starter-bank results this replaces).

## 1. Why

The 116-prompt starter bank (`spar_horizon/prompts.py`) has four problems
that Alan's generator (`../alan/ptm/prompts.py`) already solves:

| Problem in the starter bank | Alan's generator |
|---|---|
| Option delays fixed (6 mo/10 y, 1 mo/5 y), so the sensible answer is a step function of the horizon and the label probe is a horizon probe | Delays drawn per sample: 5 short (1 week to 1 year) × 6 long (2 to 50 years); rewards 7 bases × 5 multipliers |
| Near option always a), so position bias, label bias and the rational choice coincide | Order randomized 50/50, stored as `short_first` |
| Bare 5-line prompt that no one else uses; needs the `I choose:` prefill because nothing tells the model how to answer | The prior project's SITUATION/TASK/OBJECTIVE/CONSTRAINT/ACTION/FORMAT layout, shared with Augusto and the paper; the FORMAT line asks for `I choose: <a) or b)>. My reasoning: <1-3 sentences>` |
| "1 years", "1 months", "300 seconds", "3600 seconds" | Correct singulars, natural units |

It also carries the four constraint phrasings, the unit rewrites (tested to
10%), a second domain (climate) and every parameter as a column. Adopting it
makes the 8B results comparable with Alan's and Augusto's 14B thinking-off
results and with the paper: **every full-bank number is reported as a
replication next to theirs** (see `01_plan.md` §2).

## 2. What was built

### 2.1 The bank

- `spar_horizon/horizons.py`: Alan's `horizons.py`, copied unchanged.
- `spar_horizon/full_bank.py`: Alan's `prompts.py` minus the pandas helpers,
  with our grid (`HORIZONS`: since 2026-10-03, 48 log-spaced horizons, 30 seconds to 500 years in natural
  units) as the default instead of his 17-point one. `tests/test_full_bank.py`
  checks the copy against a fixture dumped from his code
  (`tests/fixtures/alan_prompts.json`: three configs, including unit rewrites,
  phrasings and the paired mode).
- `record(sample)` turns a `PromptSample` into the dict the experiments
  consume: the starter keys (`prompt`, `rewards`, `delays`, `horizon`,
  `near_delay_years`, `far_delay_years`) plus

  | field | meaning |
  |---|---|
  | `near_label` | `"a"` or `"b"`: which label is the near option |
  | `regime` | `null` (no horizon), `none` (horizon shorter than the near delay), `near` (between the delays), `both` (past the far delay) |
  | `truth` | `near` in regime `near`, `far` in regime `both`, else `None`: the rational choice under "a payout counts only if its date is within the horizon" |
  | `sample_uid`, `prompt_id`, `domain`, `horizon_text`, `short_first`, `phrasing_id` | passed through; enough to rebuild the prompt |

- `render_variant(rec, swap=, labels=)` rebuilds the prompt with the options
  swapped and/or relabeled for the order and label stability tests.
- `build_full_prompts(seed, n, limit)`; `spar_horizon/banks.py` dispatches on
  `Settings.bank` (`build_records`, `variant_builder`).
- The starter bank is untouched; its records now also carry the two delay
  years so `behavior.only_near_delivers` no longer needs the delay table.

### 2.2 Settings and CLI

`Settings` gains `bank` (`starter` | `full`), `n_prompts` (500),
`choice_prefix` ("" | `"I choose:"`) and `answer_window` (48). `prefill` and
`choice_prefix` are exclusive. `--bank full` sets `prefill=""` and
`choice_prefix="I choose:"` unless `--prefill` is given. The run folder gets
a `_full` suffix; starter paths are unchanged. `run_all.sh` takes `BANK` and
`N_PROMPTS` and runs E0 only for the full bank.

### 2.3 Answer slot: located, not pinned

With `choice_prefix` set, generation after the turn (thinking off) or after
the (possibly forced) `</think>` (thinking on) runs for `answer_window`
tokens and `model_io.locate_label` returns the first letter-or-digit token
after the model's own `I choose:`, markdown in between allowed. This is
Alan's parser (`ptm.chat.find_choice`) in token space. Without the phrase it
falls back to the first letter-or-digit token, and the existing answer-token
WARNING reports it. In the kept layout the model's `I choose:` is part of
the variable-length lead between `pre` (`</think>` + whitespace) and the
pre-label token (`:`), so the fixed-width regions are unchanged.

### 2.4 Behavior

`behavior()` takes the bank's variant builder. The temporal-reasoning test
scores against `near_label`, so a randomized bank is scored correctly.
Order and label stability are unchanged. Per-regime accuracy against `truth`
is an analysis step on `answers.json` (which now carries `near_label`,
`regime`, `truth`, `sample_uid`), not part of the run.

## 3. Phase 1 run (the box)

Connect and ship from `spar-planning-temporal-manifolds/` with `just new <console ssh line>`
and `just push` (`00_README.md`, Shipping); run these on the box from `jared/`:

```
# memory check for batch 12 (the default) and think lengths: one batch
python experiments/think_budget.py --model Qwen/Qwen3-8B --bank full --limit 12 --max-think-tokens 3072
python experiments/exp0_baseline.py --model Qwen/Qwen3-8B --thinking on  --bank full --skip-behavior
python experiments/exp0_baseline.py --model Qwen/Qwen3-8B --thinking off --bank full
# optional: per-prompt stability with thinking on, on a 100-prompt subset (~200 extra thinking generations)
python experiments/exp0_baseline.py --model Qwen/Qwen3-8B --thinking on  --bank full --behavior-n 100
```

Estimate for the thinking-on E0 at batch 12: about 60 min of generation
(42 batches of up to 3,072 decode steps) plus about 10 min for extraction,
the dense pass and the probes. The 4B think-budget check (2026-10-03) found
think lengths unchanged by the structured format.

To report, per mode: peak and probe layer, probe R² at the suffix; the think
fraction table; the label probe; and from `base_behavior` in the manifest,
`near_rate` by horizon against Alan's `behavior_by_horizon.csv`, the
position bias, and accuracy against `truth` by regime, with the `near`
regime reported on its own.

### 3.1 Run-time changes (2026-10-03, approved)

The v3 thinking-on run spent 3,793 of its 3,956 s regenerating the swapped
and relabeled banks for the stability tests. Two changes:

- **Behavior from the base answers.** Because the full bank randomizes option
  order, position bias is read from the base answers with no extra
  generation: per horizon, the near rate when the near option is a) minus
  when it is b), averaged over horizons (`behavior.base_behavior`, always in
  the manifest as `base_behavior`, both banks). This is a population-level
  bias; the per-prompt stability tests are stricter and stay available,
  on all prompts or an evenly spaced subset (`--behavior-n`, recorded as
  `behavior_n`).
- **Batch 12.** The 8B's KV cache is ~147 KB per token, so 12 sequences at
  3.3k tokens take ~5.9 GB next to 16.4 GB of weights on the 24 GB 4090.
  A simulation on the v3 think lengths gives 30,720 decode steps at batch 12
  against 46,058 at batch 8; per-step cost grows ~9% from the larger cache
  reads, so ~1.35× faster. If a batch runs out of memory, `generate_batch`
  retries it and continues at two thirds of the size, so a wrong estimate
  costs one batch. Batch 16 does not fit. Batch composition changes bf16
  greedy outputs on a few prompts, as already noted for v3.

## 4. Phase 2 (separate spec once Phase 1 numbers are in)

E1-E4 on the full bank. E2 uses Alan's `CONSTRAINT_PHRASINGS` (4) instead of
`CONSTRAINT_TEMPLATES` (6); E3 uses `UNIT_REWRITES` with `paired=True`; E4's
`distractor` and `horizon_first` perturbations must be rewritten for the
eight-line layout, and the implicit-horizon bank needs a structured rendering.

## 5. Definition of done for Phase 1 code

- All tests pass (fast suite plus `test_generation.py` on the 0.8B).
- `exp0_baseline.py --model Qwen/Qwen3.5-0.8B --thinking off --bank full --limit 6`
  and the thinking-on smoke with `--max-think-tokens 32` complete and write
  a manifest with the full-bank tag.
- `00_README.md` and `01_plan.md` describe both banks.

## 6. Decisions (Jared, 2026-10-02)

- Locate the label after the model's own `I choose:`, as Alan does; no prefill on the full bank.
- 500 prompts for the first thinking-on run.
- E1-E4 move to the full bank after Phase 1.
- Keep our 16 horizons (rendered by Alan's formatter, so "5 minutes", "1 hour", "1 year").
  Superseded 2026-10-03: 48 log-spaced horizons over the same range, for the E7 probes.
  Each is an integer in the largest unit where it is at least 2, so units only grow with
  the duration. Side effect: half the log range is below the shortest near delay (1 week),
  so in a 500-prompt draw the `near` regime falls from ~150 to ~83 prompts.
- Name: the full bank, not "Alan's bank".
- Frame full-bank results as a replication against the paper, Alan and Augusto.

## 7. Out of scope

The climate domain, Alan's extended grid and matrix renderings (reachable
through `DatasetConfig`, not run); Alan's activation format and HF uploads;
any change to `../alan/`, which stays read-only.
