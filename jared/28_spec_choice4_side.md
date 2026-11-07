# Spec: are the choice4 errors a think-budget artefact? (draft 2026-11-07, not approved)

Status: **draft, not approved.** Qwen3-8B, thinking on, `choice4` (implicit ROI). Follows `26` §1. Behavioural, one
short box session plus a CPU stage. No probes, no patching.

## 1. Question

In `choice4`, 105/246 think blocks hit the 3,072-token cap, and they hold 40 of the 44 errors (natural closes:
137/141 right; forced: 65/105). Read from `answers.json` (CPU, 2026-11-07), the forced closes also carry the
position bias and the wrong-half errors:

| | pick rate a/b/c/d | truth rate a/b/c/d | errors, wrong half |
|---|---|---|---|
| forced (105) | .15 .15 .29 .41 | .36 .28 .17 .19 | 23 of 40 |
| natural (141) | .20 .26 .29 .26 | .19 .27 .30 .24 | 3 of 4 |

A forced close appends `</think>` and asks for the label mid-thought (`model_io._generate_rows`). In the trace read
for this spec the reasoning had settled on the truth (b, `near`) and the answer pass wrote d, "the highest return,
even though it exceeds the 336-hour" horizon. A crude repetition check (a 150-character block three or more times in
the last 4,000 characters) flags about half the forced traces as looping.

So: **if the model gets more room, does it finish, and does it then choose like a natural close?** The answer decides
how every later choice4 result is read: whether the 77%/87% accuracy and the late-position bias describe the model's
choice or the cut, and what budget any causal run on this bank needs (`27` open question 3, half-switch patches).

## 2. Candidates considered

| | tests | cost | box? | result that would change beliefs |
|---|---|---|---|---|
| **A. Continue the 105 forced traces** (chosen) | errors and position bias come from the cut, not the choice | CPU stage + ~2 h, ~$0.9 | yes | forced prompts reach ~natural accuracy and lose the late bias -> the `26` accuracy and bias are budget artefacts |
| B. Order-reversed rerun of all 246 (`choice_variant(swap=True)`) | position bias is about position, not content | ~2.3 h, ~$1 | yes | the CPU split already puts the bias in the forced closes; A answers that first, at under half the cost |
| C. `choice4_roi` bank (explicit principal, `18` §2) | within-half errors come from computing returns from payouts | ~2.3 h, ~$1 | yes | within-half errors are 17/40 and sit in the forced closes; uninterpretable until A says whether they survive the cut |
| D. Cross-regime phrase donor patch on choice4 | half switches with the patched horizon while the within-half best is kept (regime vs half, causally) | ~3 h, ~$1.3 | yes | the most mechanistic, but with 43% forced closes the label noise swamps a 24-pair screen; needs A's budget first |

A text-only trace audit (CPU, free) is stage 0 of A, not a candidate: it fixes per-trace predictions before the box.

## 3. Design

**Stage 0 (CPU, mp env, saved artifacts only).** For each of the 105 forced prompts, from `answers.json` `think_text`:
- `loop`: the repetition flag above; the threshold is fixed here, before any continuation;
- `tentative`: the last option label the trace commits to in its final 1,500 characters ("so the best is b"), by
  regex; I hand-check 20 against the text and report agreement;
- check that every forced prompt's token ids are in the local `gen_cache` (the head must be the generated ids, not
  `think_text` re-tokenised). Missing ids -> stop and report.

These go in a predictions file written before the box run.

**Stage 1 (box).** Three units, each saved when it ends:
1. *Answer-pass rerun*: the original forced prefix (prompt + 3,072 think tokens + `</think>` + prefill) re-scored in
   new batches. This gives the batching noise floor on the forced labels (E10: 12/96 labels move from batching alone).
2. *Continuation*: resume greedy decoding from prompt + the 3,072 cached think tokens, for up to 3,072 more (total
   6,144), then the existing close and answer logic. Batch 8 (KV at 6k context x 8 is ~7 GB). Prompts shuffled with
   a fixed seed, stratified by regime, in chunks of 16.
3. *Optional, if budget is left*: prompts still open at 6,144 continue to 8,192 at batch 4.

Reuses `exp0_Qwen3-8B_think-on_choice4/answers.json` and `gen_cache`. No new bank, no activations.

## 4. Measures

- Close rate within the extension, overall, by regime, and by `loop`.
- Exact accuracy, and half / within-half accuracy, on continued prompts that close naturally, against their own
  forced labels (paired) and against the unit 1 noise floor. Transitions: wrong->right, right->wrong.
- Position bias on the continued answers: pick rate against truth rate for the same prompts. Excess-late is the sum
  over c and d of (pick rate - truth rate). Forced now: (.29 - .17) + (.41 - .19) = +0.34.
- Agreement of the forced and the continued answer with `tentative`; tokens used past 3,072.

## 5. Criteria (fixed before running)

- P0 (plumbing): unit 1 reproduces at least 85% of the forced labels. Otherwise the noise floor is too high to read
  unit 2: stop and report.
- C1 (the cut causes the errors): at least 60% of the 105 close within 6,144, and on those, exact accuracy is at
  least 85% with net fixes (wrong->right minus right->wrong) at least 10.
- C2 (the cut causes the late bias): excess-late on the closed continuations is at most +0.17 (half of +0.34).
- C3 (loops): of the `loop` traces, at most 30% close within 6,144, and of the others at least 70%.

## 6. Readings

- **C1 and C2 pass:** the choice4 errors and position bias are mostly budget artefacts. `26` gets an amendment
  note. Later choice4 runs use a 6,144 cap (about 1.5x the generation time) or report natural closes only. The
  forced answer pass is a separate default ("last listed / largest payout"), worth one line, not a finding about
  horizon.
- **Closes but C1 fails (closed and still wrong):** these are real reasoning errors on the harder prompts. If the
  remaining errors are mostly within-half, C (explicit ROI) is the next test; if wrong-half, the trace audit says
  whether it misconverts units or misreads the task (reinvestment, rate per time).
- **Most do not close, or C3 passes:** the model loops on these prompts, and a bigger cap only costs time.
  Forced closes are then missing data, and causal runs on this bank filter them or add a loop stop.
- **P0 fails:** the forced answer is unstable under batching, so the `26` forced-close accuracy is partly noise.

## 7. Confounds and what is not tested

- **Forced prompts are not a random sample:** 75/123 `both` vs 30/123 `near`, 33/51 days-weeks delays, 18 bases
  forced in every rendering, and truth early (a: .36). Natural-close accuracy (97%) is a ceiling for easier prompts.
  The target is the 85% in C1, not parity.
- **Position and forced close are entangled:** truth-early prompts force more often. C2 compares against the
  same prompts' truth rates, but it cannot say why truth-early prompts run long.
- **Continuation differs from a fresh 6,144 run:** the first 3,072 tokens are fixed. This tests "more room after
  the same start", not "the model under a larger budget".
- Not tested: regime vs half chosen, delay vs return in the decision, ROI matching (options are not ROI-matched by
  design, so "return" here is per-payout, not per-year), any representation or causal claim, choice8 (30% forced,
  a later rerun with the same code).

## 8. Cost and stopping

Provision, push and a 4-prompt continuation smoke 0.4 h; unit 1 0.1 h; unit 2 1.4 h (14 batches of 8, up to 3,072
tokens at 3-6k context, ~6 min each); unit 3 (optional, at most ~40 prompts at batch 4) 0.6 h. **Total 1.9-2.5 h,
$0.8-1.0 at $0.40/hr.** Stage 0 is CPU only. Any prefix of chunks is usable (balanced shuffle; analysis reports n);
`just sync` after every two chunks.

New code: a continuation entry point that resumes `_greedy` from each forced prompt's cached head, writing to a
**separate cache dir** (the cache keeps the first stored generation). Fast-suite tests with a stub model: a row that
closes mid-extension gives `forced=False` with the head unchanged; unit 1 builds the same row as `_generate_rows`.

## 9. Open questions for Jared

1. Budget and stop time; is unit 3 worth it?
2. Should I also save the reduced activation set (think@1, `</think>`, pre-label) on the continued traces? It
   would show whether `chosen_pos` at think@1 jumps to the natural-close 0.96 once the trace closes. It adds about
   0.3 h and makes the run a little like a probe repeat.
3. Should choice8 (74 forced) run in the same session? About +1 h.

## Decisions (Jared, 2026-11-07, overnight)

Answers to §9, approved for an overnight run. Code: `analysis/think_budget_stage0.py`,
`experiments/exp17_think_budget.py`, `analysis/think_budget_summary.py`, `queue_think_budget.sh`.

1. Units 1 and 2 run on choice4. Unit 3 (to 8,192 tokens, batch 4) is included but ordered last and optional.
2. No activations are saved on the continued traces.
3. The same experiment also runs on choice8's 74 forced traces, after choice4 (same code, `--bank choice8`).
4. Budget: a shared $10 hard cap across three experiments, so choice4 must fit about 2.5 h and choice8 about
   1 h on one RTX 4090.

Implementation notes that change the spec's text:
- The box has no `results/`, so stage 0 (CPU, local) extracts each forced trace's prompt ids and exact
  generated think ids from the local `gen_cache` into `steering/exp17_Qwen3-8B_think_budget/` (ships with
  `just push`). Chunks are 24 traces (not 16): three batches of 8.
- Choice8 has no pre-fixed C2 value: its threshold is half of its own forced excess-late (second-half
  positions, e to h); choice4 keeps +0.17. The C1 net-fix threshold of 10 scales as 10 x n / 105.
