# Multi-turn planning: does the single-turn horizon geometry survive a multi-step plan?

Exploratory experiment for the project's core question (proposal: *Planning Temporal Manifolds*, RQ1–RQ2):
when a model builds a plan over several turns, can the horizon it is planning over be read from its
activations at turn boundaries? Author: Augusto Nicola. Status: **design + local pilots; nothing run on the
target model yet.**

## Why the design looks like this

Alan's experiments and our relevance-aware decoder run both showed that when the horizon is written in the
prompt, **a regex on the text beats any activation decoder**. In a multi-turn chat the target horizon sits in
turn 1 and is visible (through attention) from every later token, so "H_target is readable at turn 5" can be
plain copying. To learn something about *planning*, we read horizons that are **not in the text yet** at the
moment of capture, and every activation result is compared with a **text-only baseline** that sees everything
written so far.

## Protocol (the proposal's "Continue" protocol, made concrete)

- **System prompt** (identical for every conversation): the format rules. First reply = an outline with the
  titles of exactly 5 steps and no timings; each user `Continue` → one step as
  `Step <n>: <title>` / `Time horizon: <how far from today this step will be complete, one duration>` /
  `Details: …`; after step 5, `Continue` → `Plan Completed`. 7 assistant turns per conversation.
- **First user turn**: scenario + goal + (optionally) a horizon sentence, e.g. *"Our company wants to cut its
  carbon emissions. Help me make a plan. The time horizon for this plan is 25 years."*
- **Content-matched horizons**: 12 scenarios chosen to make sense from a week to fifty years, each at all
  8 H_target values (1 week, 1 month, 3 months, 1 year, 3 years, 10 years, 25 years, 50 years; 3.4 decades).
- **Paraphrase control**: 3 wordings of the horizon sentence ("should achieve this within {h}", "the time
  horizon for this plan is {h}", "all of it needs to happen within {h} from today").
- **No-horizon condition**: the same scenarios without a horizon sentence; the model picks every horizon
  itself. One prompt per scenario, so plans are sampled (Qwen3 non-thinking settings: T 0.7, top-p 0.8,
  top-k 20), 16 per scenario. The horizon condition is greedy.
- Thinking off (the proposal's main condition): Qwen3's template adds an empty think block to the current turn.
- Guard: a step reply that starts a second step or appends "Plan Completed" is cut there and closed with
  `<|im_end|>` (logged as `cut`; the proposal's "constrained decoding" backup).

Size: 288 horizon + 192 no-horizon = **480 conversations**, 3,360 assistant turns.

## Measurement points (13 per turn)

| Name | Tokens | In context at that moment |
|---|---|---|
| **P0–P8** | the boundary window before each reply: `<\|im_end\|> \n <\|im_start\|> assistant \n <think> \n\n </think> \n\n` | everything up to the user's turn; **not** the reply being planned |
| U | last token of the user's turn (`Continue`, or the end of the first message) | same |
| H | `Time horizon:` inside a step reply | the step's title |
| V | the horizon value's last token | the value itself (positive control) |
| E | `<\|im_end\|>` closing the reply | the whole reply |

Every turn is captured on exactly the sequence it was generated from (Qwen3 drops the empty think block from
earlier turns in the history, so a re-tokenised transcript would not match; checked in the tests). Residual
stream, layer 0 = embeddings, layer l = after decoder layer l (Alan's convention); every 4th layer plus the
proposal's 0.4 / 0.6 / 0.8 L; float16.

## Analyses (`mtp/analysis.py`, `scripts/evaluate.py`)

All probes are ridge regressions to log10 years, CV **grouped by scenario** (a probe is always scored on
scenarios it never saw). Metrics: R², Spearman ρ, mean error in decades, within-2× (from the RAD run), and
PC1 ρ for comparison with the paper.

| | Question | Target | Compared with |
|---|---|---|---|
| **A** persistence | Does the turn-1 H_target probe still read H_target at turn t? | H_target | the same probe at turn 1; a probe refit at turn t |
| **B** next step | At the boundary before a step, can we read the horizon the model is about to write? | H_step of the upcoming step | text baseline: H_target, previous step's H_step, step index; **ΔR² = (text + activations) − text** |
| **C** first turn | Does the turn-1 boundary already carry step j's horizon, beyond H_target? | H_step(j), j = 1…5 | baseline on H_target only |
| **D** no horizon | Can we read horizons the model chose itself (no horizon phrase anywhere)? | H_step, no-horizon conversations | text baseline without H_target; transfer of a probe trained on horizon conversations |

Controls: labels shuffled within scenario (noise floor for ΔR²); a behavior check that the plans respond to
H_target at all (slope of the last step's log H_step on log H_target, monotone plans, steps within target).
Exploratory: no pre-registered thresholds; the summary reports the best cells *with* the shuffled floor.

## Layout

```
mtp/prompts.py     scenarios, horizons, wordings, system prompt, specs; parsing of step horizons
mtp/chat.py        chat encoding, named positions, the step cut guard
mtp/run.py         batched generation turn by turn + teacher-forced capture (safetensors shards + index.parquet)
mtp/analysis.py    analyses A-D and behavior
scripts/capture.py generate + capture   (--smoke: 15-conversation slice; --gpu-gib: CPU offload for local tests)
scripts/adherence.py  format-adherence report
scripts/evaluate.py   A-D -> results/<run>/{A,B,C,D}_*.csv, behavior.json, summary.md  (--quick: sanity pass)
scripts/run_pipeline.sh  tests -> capture -> adherence -> quick evaluate -> checksums (QUICK_EVAL=0 for the full
                      analysis on the box; by default it runs locally after copy-back, so no GPU idles on CPU work)
tests/             prompts/parsing/positions on the Qwen3 tokenizer; synthetic end-to-end check of A-D
```

## Pilots (local RTX 5060 Ti, 8 GB; 15-conversation `--smoke` slice)

- **Qwen3-1.7B, rules in the first user turn**: writes all 5 steps with details in the outline, then repeats
  itself or answers "Plan Completed" to every `Continue`. 0/15 usable. Moving the rules into a system prompt
  and adding the cut guard did not change that. Too small to follow the protocol.
- **Qwen3-4B, rules in a system prompt** (CPU offload): outline titles-only 15/15, "Plan Completed" at the end
  15/15, but the first `Continue` got **Step 2**: the model counted the outline as step 1, so every step was
  shifted and the plan ended a turn early. A few steps had "Time horizon: Ongoing" (unparseable; excluded and
  counted). Fix: rule 2 now says the first `Continue` gets step 1.
- **Qwen3-4B after the fix** (`results/pilot_Qwen3-4B/`): outline titles-only 15/15, step numbers right 100%,
  horizons parsed 97%, "Plan Completed" 15/15; **13/15 conversations fully usable** (the 2 others end with
  "Time horizon: Ongoing", both no-horizon). Plans respond to H_target: last step vs target slope 1.03,
  ρ 0.97, every step within the target. Often a counting pattern ("1 day, 2 days, 3 days…") or a last step
  equal to the target, so the text baseline for B will be strong. The analysis runs end to end (4 s);
  its probe numbers mean nothing at 3 scenarios.
- **Qwen3-14B smoke on the rented box** (RTX PRO 5000, 48 s for 15 conversations): outline and "Plan Completed"
  100%, but only 6/15 fully usable: in 8 conversations the model answered the 5th `Continue` with "Plan
  Completed" (steps 1-4 fine, step 5 skipped): an off-by-one on rule 3 ("After step 5, when the user says
  Continue again"). Fix: rule 3 now enumerates "steps 1, 2, 3, 4 and 5, one per Continue; never skip a step".
- Local 4B runs need `--batch 4 --gpu-gib 4.5` (turn 6 contexts no longer fit at batch 8 on 8 GB).
