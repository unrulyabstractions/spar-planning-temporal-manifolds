# Spec: 4-choice and 8-choice investment banks, unit-crossed (2026-10-27)

Status: **draft, awaiting approval.** Decisions (Jared, 2026-10-27):
- near / both regimes 50/50 at every duration, no `none`;
- the same principal for every option, options shown as payouts;
- ROI implicit first (Alan's TASK line word for word), then explicit (the principal stated),
  as separate banks built from the same draws;
- one delay per half;
- horizons and delays crossed with units as in the crossed bank
  (`16_spec_pristine_manifold.md` §4);
- bank, E0 and behavior only (probes and causal tests get a later spec).

Numbering: the main checkout already has `15`, `16` and `17`.

**Depends on** the crossed-bank code in `spar_horizon/full_bank.py`:
- `crossed_renderings`;
- `draw_crossed_delays`;
- `feasible_regimes`;
- `realism_labels`;
- the `MARGIN`/`GAP`/`DELAY_LIMITS`/`MIN_DISTANCE` constants.

On 2026-10-27 that code is uncommitted in the main checkout, and this worktree doesn't have it.
Before task 1, it is committed on `jared` and this branch is rebased onto it. These banks reuse it
unchanged, with no copies.

## 1. Why

On the full bank, three things move with the horizon and with the decision:
- **Unit word:** the unit explains 0.97 of log horizon.
- **Regime:** short horizons are all `none`, long ones mostly `both` (`01_plan.md` §4).
- **"Which half" and "which option":** with two options they are the same thing.

So the open question in `01_plan.md` §4, whether the L24 curve carries the horizon or the
decision, can't be settled there.

These banks separate four things:
1. **The horizon from the correct half.** Every duration appears equally in the `near` regime
   (the best near option is correct) and the `both` regime (the best far option is correct). Only
   H compared with the delays decides.
2. **The duration from the unit word.** Each duration is rendered in every unit that fits, as
   in the crossed bank, so E13's fit (`16` §2) can strip the unit from these activations too.
3. **The half decision.** Near or far, set by the horizon.
4. **The within-half decision.** Which option in the half, set by ROI alone, since the delay is
   shared within a half.

There are no `none` prompts, so every prompt has a correct answer. `none` is 59% of the full
bank and has no correct answer there (docs Q3).

## 2. The banks (`choice4`, `choice8`, `choice4_roi`, `choice8_roi`)

Each prompt has k = 4 or 8 options. Half are near options with delay `d_near`, and half are far
options with delay `d_far`, where `d_near < d_far`.

**Horizons, as the crossed bank.**
- The 48 durations, each rendered in every unit whose value lies in 0.1–1000
  (`crossed_renderings`). That is the exact integer where there is one, else 2 significant
  figures ("0.5 years", "1.4 hours").
- A base prompt is emitted once per rendering. Its renderings share everything else, and carry
  `pair_id`, `duration_id`, `rendered_unit` and `natural`.

**Durations kept.** Only those where both `near` and `both` are feasible under the crossed
limits (`feasible_regimes`):
- 30–71 s have neither, and 108 s and 3 min have `near` only.
- Those are dropped, so that regime never moves with duration at the short end.
- That leaves about 5 minutes to 500 years. The exact list comes from task 1.

**Regimes and delays.**
- Each duration's bases alternate `near` / `both`, starting with the other one at the next
  duration.
- Delays come from `draw_crossed_delays`, unchanged:
  - log margins `MARGIN` (0.2–1.5 decades) from H, and `GAP` (0.3–2 decades) between the delays;
  - limits of 1 minute to 1,000 years;
  - each delay rendered in a randomly drawn unit (so delay units are crossed with horizon units);
  - every delay at least 0.2 decades from every rendering of H.
- The regimes are:
  - `near`: `d_near < H < d_far`. The truth is the best near option.
  - `both`: `d_near < d_far < H`. The truth is the best far option.

The truth rule is the one `TRUTH` uses now: a payout counts only if it arrives within the
horizon, and among those the larger payout wins.

**Realism.**
- At short durations the payouts are in minutes or hours ("10,400 dollars in 40 minutes").
- Each record carries `realism_labels`: `delay_scale` and `alan_range`.
- As in `16`, these are labels only.
- Results are ours, not a replication, since the delays don't follow Alan's grids.

**Payouts.**
- Principal P is drawn from Alan's `SHORT_REWARDS` (1,000 to 100,000 dollars) and shared by
  all options in the prompt.
- Each option pays P × (1 + r):
  - near returns r are drawn without replacement from {2, 4, 6, 8, 10, 12, 15, 20}%;
  - far returns r are drawn without replacement from {50, 75, 100, 150, 200, 300, 400, 500}%.
- So every far payout beats every near payout. Within a half the steps are fine:
  - 4-choice draws 2 per half;
  - 8-choice draws 4 per half.
- P and r are whole numbers, so payouts are whole dollars.
- `gap_in_half` is the ratio of the best to the second-best payout in the truth's half, kept as
  a difficulty covariate.

**Order and labels.**
- Labels are `a)` … `d)` or `a)` … `h)`.
- Within each regime, the truth's position cycles through the k labels over the bases, so truth
  positions are balanced to within 1 base. The other options are shuffled into the remaining
  slots.
- The renderings of a base share the order.

**Template.** Format `formatted_v1`, domain `investment`, constraint phrasing 0. Only the ACTION
count, the FORMAT label list and the number of option lines differ from Alan's wording. The two
ROI variants differ only in the TASK line:

| bank | ROI | TASK line |
|---|---|---|
| `choice4`, `choice8` | implicit | `TASK: You, the head of the household, are tasked to choose the best investment:` (Alan's, word for word) |
| `choice4_roi`, `choice8_roi` | explicit | `TASK: You, the head of the household, have 10,000 dollars to invest and are tasked to choose the best investment:` |

In the implicit banks the returns can only be read from the payouts. The principal is shared by
all options, so it never changes the truth.

**ROI pairing.** Each explicit bank is built from the same draws as its implicit bank, so record
i of `choice4_roi` is record i of `choice4` with the TASK line swapped. Twins share `sample_uid`
and `pair_id`, and carry `roi_explicit`. Each bank is a separate E0 run in its own folder,
implicit first.

Example (4-choice, implicit, `near` regime, truth `d`; one of the renderings of 3 years):

```
SITUATION: Plan for the future of the household.
TASK: You, the head of the household, are tasked to choose the best investment:
a) 20,000 dollars in 10 years.
b) 10,400 dollars in 0.5 years.
c) 25,000 dollars in 10 years.
d) 10,800 dollars in 0.5 years.
OBJECTIVE: Think deeply about which option is preferable.
CONSTRAINT: You must select the option that provides the greatest benefit for this time horizon: 36 months.
ACTION: Select one of the four options. Provide reasoning.
FORMAT: I choose: <a), b), c) or d)>. My reasoning: <1-3 sentences>
```

**Size.**
- 4 bases per duration (2 `near`, 2 `both`) over about 43 durations gives about 172 bases.
- At about 2.8 renderings per base, that is about 500 prompts per bank.
- Task 1 sets `per_duration` so each bank lands near 500.
- That is about 86 bases per regime, so for k = 8 there are about 11 per truth position.
- Statistics resample by base, as in `16`.
- The bank uses an RNG of its own, so the full, units and crossed banks are untouched.

**Record fields.**
- Kept from the full bank:
  - `prompt`, `horizon`, `horizon_text`;
  - `near_delay_years`, `far_delay_years`;
  - `regime`, `truth` (`"near"` / `"far"`);
  - `sample_uid`, `prompt_id`, `domain`, `phrasing_id`.
- From the crossed bank:
  - `pair_id`, `duration_id`, `rendered_unit`, `natural`;
  - `delay_scale`, `alan_range`.
- New:
  - `n_options`, `principal`, `roi_explicit`;
  - `options`: a list of `{label, payout, return_pct, delay_text, delay_years, half, rank_in_half}`;
  - `labels`, `near_labels`, `far_labels`, `truth_label`, `gap_in_half`.
- `near_label` is not set, because there is no single near label. The code that needs it
  (exp8, exp10, `steering.answer_pairs`, the analysis scripts) is out of scope (§5).

## 3. Pipeline changes (E0 and behavior)

- **Bank names.** Add `choice4`, `choice8`, `choice4_roi` and `choice8_roi` in:
  - `config.py` (validation and the run tag, giving folders `..._think-on_choice4` …
    `..._think-on_choice8_roi`);
  - `pipeline/s1_data_vectors/experiments/common.py` (`--bank` choices, and the rule that generated prompts get no
    prefill and `choice_prefix = "I choose:"`);
  - `banks.py`, both `build_records` and `variant_builder`. For these banks the variant is
    reversed order and labels 1..k, so the existing hook has a meaning.
- **Label logits.** `extract.py` takes the label set from the record, not the default pair.
  - `label_logits` becomes `[n, k]`; the full and crossed banks stay `[n, 2]`.
  - The warning check covers a–h and 1–8.
  - `label_ids` already takes any labels. A tokenizer test checks that a–h are single tokens,
    with and without a leading space.
- **`parse_choice`** gets the record's labels.
- **exp0 label probe.** The target becomes "chose a near-half option", so it stays binary and
  `geometry.label_probe` is unchanged.
- **`answers.json`** carries the new fields and the crossed ones.
- **Behavior.** A new function, `multichoice_behavior(records, answers, label_logits)`. It
  leaves `base_behavior`, which is the 2-slot design, alone. Per regime and overall:
  - accuracy: exact option, correct half, and correct within half given the chosen half;
  - **half log-odds:** logsumexp(near labels) − logsumexp(far labels), the k-option version of
    the near log-odds, and the regime contrast C on it;
  - **within-half log-odds:** the truth's logit minus logsumexp of the rest of its half;
  - pick rate per label position (position bias), and accuracy against `gap_in_half`;
  - **rendering consistency:** the share of bases whose renderings give the same choice, and the
    spread of half log-odds across renderings;
  - heuristic baselines:
    - "largest payout" (always correct in `both`, always wrong-half in `near`);
    - "earliest delay, then largest";
    - uniform.

## 4. Runs

E0 on each bank: Qwen3-8B, thinking on, `max_think_tokens` 3,072 (unchanged, so forced-close
rates compare with the full bank), activations at all layers, `--skip-behavior` as for the full
bank.

- **Order:** the implicit banks (`choice4`, `choice8`) first, then the explicit ones
  (`choice4_roi`, `choice8_roi`). The explicit runs can be a separate box session.
- **Time:** the full bank took 124 min at batch 12 on a 4090. Batches run in prompt order
  until their longest member stops. With 103/500 prompts forced, 94% of batches of 12
  (1 − (397/500)^12) already ran to the 3,072-token cap, so 124 min is close to the cap-bound
  time. Longer reasoning can't add much; the prompts are only ~50–100 tokens longer. Expect
  about 2–2.5 h per bank on a 4090:
  - `choice4` + `choice8`: about 4–5 h, or 2–2.5 h on two GPUs in parallel lanes;
  - all four banks: about 8–10 h.

  The estimate fails if out-of-memory errors step the batch size down. It also fails if the
  box is not a 4090: `16` §4 puts the 532-prompt crossed bank at about 1 h on its box.
- **Storage:** about 6 GB of activations per bank.
- **Report:** the forced-close rate per bank against the full bank's 103/500.
- **Queue:** `queue_multichoice.sh`, with the same `step`/`.done` resume as `queue_units.sh`.

Local, before the box (you run it via `local_checks.py`): a 0.8B smoke on 4 prompts per bank.
It checks the rendering, that the label slot is found after `I choose:`, and that
`label_logits` has width k.

The implicit-vs-explicit comparison belongs to the results doc. Over the ROI twins it reports:
- the share of twins with the same choice;
- accuracy (exact, half, within half) in each variant;
- the paired difference in half and within-half log-odds (bootstrap CI by base).

## 5. Out of scope

- E7 probes, the E13 fit, and E8/E10 patching and steering on these banks. That includes
  `near_label` consumers and N-option `near_log_odds` in `steering.py`. They get their own spec
  after the behavior is in.
- Thinking off, Qwen3-14B, other domains, other constraint phrasings, null horizons.
- Order and relabel stability reruns. They are extra generation; the full bank skipped them.
- Any change to the full, units or crossed banks or their results.

## 6. Tasks

0. **Dependency:** the crossed-bank code is committed on `jared`, and this branch is rebased
   onto it.
1. **Bank:** `build_choice_prompts(k, roi_explicit, seed, per_duration, limit)`, the renderer,
   and the record fields. Tests:
   - only durations with both regimes feasible, and per duration `near`/`both` counts equal;
   - truth positions within 1 per regime;
   - every delay at least 0.2 decades from every horizon rendering, and the regime matches the
     delays for every rendering;
   - every far payout above every near one, and payouts distinct within a half;
   - the truth matches the rule;
   - the renderings of a base differ only in the horizon phrase;
   - the implicit TASK line equals Alan's word for word, and ROI twins differ only in the TASK
     line;
   - a golden rendering of one prompt;
   - the full and crossed banks are unchanged.
2. **Wiring:** bank names in `config.py`, `common.py`, `banks.py` (records and variant).
   Tests: routing and the run tag.
3. **Labels:** `extract.py` takes the record's label set and writes `[n, k]` logits;
   `parse_choice` takes N labels. Tests:
   - a–h single tokens on the 0.8B tokenizer;
   - `parse_choice` with 8 labels;
   - a record-path test of exp0's `answers.json` fields, with no model run.
4. **`multichoice_behavior`,** with tests on synthetic answers and logits that have known
   accuracies, log-odds and rendering consistency.
5. **The 0.8B smoke** in `local_checks.py`, which you run.
6. **`queue_multichoice.sh`,** dry-run with stubs.
7. **After the box run:** `19_results_multichoice.md`, behavior only, with the
   implicit-vs-explicit comparison (§4), and a §5 entry in `01_plan.md` for the probe/causal
   spec.
