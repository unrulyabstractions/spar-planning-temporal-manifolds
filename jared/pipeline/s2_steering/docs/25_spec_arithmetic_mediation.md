# Spec (draft, 2026-11-03, not approved): does the arithmetic decide the both-regime choice?

Draft from the overnight session (`24` §2.2, §3.3, §4). Needs Jared's approval before code.
Qwen3-8B, thinking on, box. Three parts. Part B is the cheapest and answers the most, so
it goes first.

## 1. Why

- In the both regime the 8B picks near on half the old-bank prompts.
- The traces show why (`24` §3.3, 105 coded traces). Where far wins at any normal rate
  (r* ≥ 10%/yr), a correct reinvestment comparison went far 45 of 45 times; an erroneous,
  overridden or absent one went near 30 of 32 times.
- The choice is not readable before reasoning (`24` §2.2), and it flips under bf16 batch
  noise in 12 of 50 both-regime prompts (0 of 46 near-regime).
- All of this is correlational. The coding shows that the arithmetic and the choice go
  together, not that one causes the other: a trace already leaning near could skip the
  sums. This spec tests the causal direction and measures how much of the both-regime
  choice is sampling noise.

## 2. Parts

**A. On-policy resampling** (Thought Anchors / Thought Branches style).
- Prompts: the 105 natural-close both-regime prompts of the full and units banks.
- Sample 8 traces per prompt at Qwen's recommended thinking settings (T = 0.6, top-p 0.95,
  top-k 20), max 3,072 think tokens; score the label as E0 does.
- Per prompt: near share and the share of traces that compute (regex proxy, validated
  against the 105 coded traces before use).
- Needs: a sampling path in `model_io` (today greedy only, `do_sample=False`).

**B. Arithmetic insertion** (off-policy, on the cached greedy traces).
- Rows: both-regime natural closes with r* ≥ 10%/yr (77 prompts: 32 near, 45 far).
- Cut each cached trace at a fixed point: the end of the first paragraph that names the
  reinvestment idea (regex "reinvest|compound|grow"), or at 30% of the block when none
  does. Then insert one of:
  - **correct**: "Let me compute it: reinvesting $R_near at 7% per year for Δ years gives
    $X, which is less than $R_far, so b) gives more." (numbers and labels filled per
    prompt);
  - **wrong**: the same sentence with X set to 1.5 × R_far and the conclusion reversed;
  - **neutral**: a sentence of the same token length restating the options.
- Continue greedily from the insertion and score the label.
- Needs: the E10 continuation path (`exp10_patch_reasoning.generate` on prefilled ids).

**C. Rate-anchored prompt** (one E0-style run).
- The 105 prompts with "Assume any money received can be invested at 5% per year." added
  to the OBJECTIVE line. Greedy, thinking on.
- Choice against "r* < 5%": does the horizon still matter within the both regime once the
  rate is fixed?

## 3. Success criteria (fixed before the run)

| part | measure | passes if |
|---|---|---|
| A | per-prompt near share, r* ≥ 10% rows | at least half of the 77 prompts have a near share strictly between 0 and 1 (the choice is a property of the sample, not of the prompt) |
| A | near share against the trace computing | within prompt, traces that compute go far more often than traces that do not (paired over prompts, sign test p < 0.01) |
| B | correct insertion on near-choosers (32) | ≥ 50% switch to far, against ≤ 10% under the neutral insertion |
| B | wrong insertion on far-choosers (45) | reported, no threshold: does the model check the numbers? |
| C | choice vs r* < 5% | ≥ 85% of choices match; the horizon term adds < 0.02 to out-of-fold log loss within the both regime |

If B passes, the arithmetic is a causal mediator of the both-regime choice. If B fails but
A shows within-prompt variance, the choice is fragile but the off-policy sentence is not
read as the model's own (the Thought Branches caution). Then A's computing/non-computing
split is the better evidence.

## 4. Cost (4090, cached prefixes where possible)

- B: 77 × 3 continuations of ≤ 2,000 tokens, ~30 min.
- C: 105 traces, ~25 min.
- A: 105 × 8 = 840 traces, ~2–2.5 h at batch 12. Can run with 4 samples first (~1 h).

## 5. Out of scope

The 14B (the arithmetic coding of its traces is not done, `24` P7); the none regime (its
choice turns on reinterpreting the horizon, a different mechanism, `24` §3.2); any
activation patching.
