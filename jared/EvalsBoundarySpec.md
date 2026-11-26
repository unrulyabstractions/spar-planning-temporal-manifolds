# Spec: where the horizon structure holds, and where it stops (2026-11-21)

Status: **approved 2026-11-21** with the tier changes in §5. S0 done (results in
`results/horizon/evaltypes_Qwen3-8B_think-on/boundary/`). S1 code ready (`exp21_boundary_off.py`,
`queues/queue_boundary_off.sh`); waiting for Jared's 0.8B smoke (`local_smoke_exp21.py`) and a free box. Qwen3-8B. Thinking off first, thinking on only where thinking off
cannot answer. Budget **$2.00 hard cap** for the box. Branch `jared-boundary`, off `jared-eval-types` (81e2498).

An offshoot of `01_plan.md`: a check that the representation later work leans on still holds outside Alan's
template. "Holds" means **decodable in the same manner**: the probe fitted on the full bank reads the horizon. This
spec does not test causal use; E19 (`EvalsPromptsV2.md` §8) covers that for the existing types.

## 1. Questions

1. **Boundary.** As prompts move away from Alan's template (multi-turn, LLM-written harder prompts, no explicit
   horizon, role or persona only), at which rung does the full-bank horizon structure stop reading the horizon?
2. **Moved or gone.** Where the fixed probe fails, is the structure still there in another direction (a probe refit
   on that type works), or is it gone (the refit fails too)?
3. **Off predicts on.** Does the cheap thinking-off reading predict where the thinking-on reading holds?

## 2. What is already known (no rerun)

- **The suffix is the same in both modes.** The five tokens from `<|im_end|>` to `assistant\n` match in the saved
  thinking-off and thinking-on full-bank activations: cosine ≥ 0.997 at L10/16/22/30 (checked 2026-11-21). The
  `suffix_end` position (index 4) is one of them. So the thinking-on eval-type activations at that position already
  are the thinking-off result.
- **Fixed-probe transfer on the 20 existing types** (`EvalsPromptsV2.md` §2): it holds where the horizon is near the
  end of one user turn (L18 Spearman 0.91–0.97). It fails for `system_prompt` 0.22, `multiturn_analyst` 0.36 and
  `prose_natural` 0.45, and `json_io` is biased by 8 decades. After thinking, at the label slot, these transfer
  better (0.72 / 0.65 / 0.67 / 0.92).
- **Distance from Alan's prompt** (§7) ranks the types. The order follows how much of the template a type rewrites:
  one phrase < the task line < the speaker < the whole layout.
- **The unit word alone scores R² 0.96 on the full bank** (`history/01_plan_2026-11-19.md` §4). On explicit prompts a
  probe can "read the horizon" by reading the unit word. Prompts with no explicit horizon remove that shortcut, which
  is why they are the most informative rung.
- **E18 implied bank** (`pipeline/s2_steering/docs/29_spec_implied_horizon_side.md`, 288 rows): it labels an implied
  horizon as an interval [lo, hi] in years. Rung N1 reuses those prompts and labels.

## 3. The ladder

Each new type has 48 base items. The horizons are spread over the full-bank grid (10 minutes to 30 years), about one
per grid point, with near and both regimes balanced. Rows that need a truth use the crossed delays (`full_bank`), so a
row's regime does not depend on the exact reading of an interval.

| Rung | Type | What changes from Alan's template | Label | Source |
|---|---|---|---|---|
| R0 | `t0_w0` | nothing (reference) | explicit | existing |
| R1 | 20 existing eval types | one phrase up to the whole layout | explicit / implied | existing activations |
| R2 | M1 `mt_first` | horizon in user turn 1 of 3; the choice asked in turn 3 | explicit | templated |
| R2 | M2 `mt_last` | horizon in the last of 3 turns, which also asks the choice | explicit | templated |
| R2 | M3 `mt_revised` | horizon set in turn 1 and changed in turn 3; the label is the new one | explicit | templated |
| R2 | M4 `mt_long` | about 8 turns (~1.5k tokens) of unrelated chat between the horizon and the question | explicit | templated |
| R2 | M5 `mt_assistant` | the horizon appears only in an earlier assistant turn ("So you need this within 3 months.") | explicit | templated |
| R3 | H1 `llm_prose` | a realistic advice request, horizon mid-paragraph | explicit | LLM-written |
| R3 | H2 `llm_competing` | several durations in the text (loan term, past events, the delays); only one is the horizon | explicit | LLM-written |
| R3 | H3 `llm_document` | an email or memo of ~400 tokens with the decision question at the end | explicit | LLM-written |
| R4 | N1 `implied_event` | horizon implied by an event or date ("until the lease ends in March") | interval | E18 bank |
| R4 | N2 `persona_user` | who the user is sets the horizon ("I'm 81 and want this for my grandkids' wedding" vs "a 24-year-old saving for retirement"); no duration in the text | interval | LLM-written |
| R4 | N3 `persona_model` | only the system prompt's role for the model sets it ("You are a day-trading desk assistant" vs "You advise a 50-year endowment"); the user turn is the same for every row | interval | LLM-written |
| R4 | N0 `no_horizon` | no horizon and no persona cue (control) | none | templated |

- **LLM-written prompts (H1–H3, N2, N3)** are written by Claude subagents from a brief that fixes the label, the
  options and the regime of each item. A script then checks every item: the options and delays are present and
  parse, and for N2/N3 the text has no duration. Jared hand-checks 20 random items (4 per type) before the box run.
- **Labels with no explicit horizon.** Each N2/N3 persona gets an interval [lo, hi] in years, written down before
  any activations are read (E18's scheme). As a second label, the model states the horizon it assumes: one short
  thinking-off generation, at most 16 tokens, prompted "What time horizon applies here? Answer with a duration."
  The persona interval and the model's own stated horizon can disagree, and that difference is a result in itself.
- **N0 is the floor.** On N0 the probe should give the same reading for every row, near the bank mean. A readout that
  still spreads out on N0 rows is reading something other than the horizon.

## 4. Positions and layers

- **Layers:** 10, 14, 16, 18, 22, 24, 26 and 30, as in the eval-type runs. The tables show 14/18/22/26.
- **Positions, thinking off:**
  - `suffix_end`, the `\n` after `assistant`. Free for R1, identical in both modes.
  - `prelabel`, the `:` of the `I choose:` prefill after an empty think block. One forward pass, no generation.
  - `phrase_end`, the last token of the horizon phrase, or of the persona sentence for N2/N3. Multi-turn and
    persona types only, to tell "never encoded" apart from "encoded, then lost by the end of the prompt".
- **Thinking on (stage S2 only):** `think_end` and `prelabel` after reasoning, as in the existing runs.

## 5. Measures

All of these are per (type, layer, position), with a 95% bootstrap CI over base items.

1. **Fixed probe:** the full-bank ridge probe from `eval_types_probe.py`, applied unchanged. It is fitted on the
   thinking-off bank at `prelabel`, and on either bank at `suffix_end`, where the two are identical. Reported as
   Spearman, MAE and bias, in decades.
2. **Refit probe:** ridge fitted within the type, GroupKFold(5) by base item. For interval labels the target is the
   log-midpoint.
3. **Unit lookup** (explicit rungs only): each held-out row is predicted by the mean training horizon of its unit
   word, with no activations. A probe that does no better than this is reading the unit word.
4. **PCA, two cheap checks:**
   - project the type's rows onto the full bank's top PCs (k = 1, 2, 4) and report the Spearman of PC1 with log
     horizon;
   - plot PC1–PC2 of the full bank, with the type's horizon averages drawn over it, at `suffix_end` and `prelabel`
     for L18 and L22. That is one figure per rung.
5. **No-horizon rungs (N1–N3):**
   - Spearman against the interval midpoint;
   - AUC for short-persona vs long-persona rows;
   - the same two against the model's stated horizon;
   - and against the N0 floor.

6. **Random baseline:** a label-permutation null. Horizons (or interval midpoints) are shuffled across base items,
   with all rows of one item kept together, 200 times. The fixed probe is rescored on each shuffle, and the refit
   probe is refitted on each. The null's p95 is the "above random" bar.

**Report everything.** Every measure goes in the tables for every cell; Jared reads them afterwards. Each cell also
gets a tier, from the fixed probe and the refit probe separately (approved 2026-11-21):
- **strong:** Spearman lower CI ≥ 0.8, |bias| ≤ 1 decade, and above the unit lookup on explicit rungs;
- **weak:** Spearman lower CI ≥ 0.5;
- **above random:** Spearman lower CI above the permutation null's p95;
- **fail:** not above random, meaning no readable structure.

"Moved" means the fixed probe's tier is below the refit probe's. The boundary is the first rung, and the first
property of a rung, where a type drops a tier, at each position.

## 6. Stages

| Stage | What | Where | Cost |
|---|---|---|---|
| S0 | §5 measures on the 20 saved R1 types at `suffix_end` (thinking off = on there), plus `prelabel` thinking on as the comparison | local, saved activations | $0 |
| S1 | build R2–R4 (≈ 13 new types × 48 = ~620 prompts, plus 48 for N0); thinking-off forward passes at the three positions, plus the 16-token stated-horizon generation for R4; R1 `prelabel` thinking off | box | est. $0.30–0.50 |
| S2 | thinking on, generation, for a stratified subset: the types S1 calls **moved** or **gone**, plus 2 that hold as controls. Up to ~250 prompts | box | est. ~$0.11 per 32 prompts (E-types rate: 13–15 min per 32 at $0.45/hr), so ≤ $1.0 |

- **Spending rule:** run `just burn` after S1. S2 runs only if at least $1.2 is left. Its size is set so that the
  box goes down with ≥ $0.25 to spare, under the $2.00 cap.
- **When to stop:** if S1 calls every new type **holds** at `prelabel`, S2 is skipped; thinking on adds nothing for
  this question.

## 7. Tasks

1. **S0 analysis** (local). Extend `pipeline/s3_analysis/eval_types_probe.py` with the refit probe, the unit lookup,
   the PCA checks and the verdicts. Test on synthetic arrays.
2. **Templated generators** for M1–M5 and N0 in `spar_horizon/eval_types/`, with tests: a label and regime for every
   row, the horizon phrase position found, the token positions resolved under the Qwen3 template.
3. **LLM-written items** for H1–H3, N2 and N3. Write the briefs, generate the items with subagents, run the validator
   script, and send Jared 20 items for review.
4. **Thinking-off runner:** forward passes only. Reuse E19's prefill path, add `phrase_end` capture and the
   stated-horizon generation. Smoke-test on the 0.8B locally (Jared runs it on xpu).
5. **S1 box run** (`just up`, tmux, `just down`), then the analysis and `EvalsBoundaryResults.md`.
6. **S2**, only if the spending rule allows and §6 does not stop it.

## 8. Not in scope

- Causal tests on the new rungs (patching, steering).
- Other models.
- Changing the full-bank probe. It stays the fixed reference.
