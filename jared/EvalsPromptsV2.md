# Eval prompts v2: making the truth unambiguous (2026-11-17)

Follows `EvalsResults_2026-11-16.md`. That run showed the model rarely takes a payout that comes
after the horizon (`near` regime 11-12/12 in every two-option type), but often takes the sooner,
smaller payout when both pay out within the horizon (`both`). This doc traces those errors to the
template, proposes a clearer wording, and records the predictions made before the v2 run.

## 1. What the `both` errors are (local analysis of the overnight rows)

**The template leaves reinvestment open.** Alan's template asks for "the best investment" among
lines like "1,000 dollars in 1 month" and for "the greatest benefit for this time horizon". It never
says whether money received can be reinvested. The model nearly always assumes it can: compounding is
mentioned in 52/56 wrong `both` answers and 126/156 right ones. Our truth (`both` -> the far option)
holds only if it cannot, or if reinvestment earns less than the far option's implied yearly rate
`(v_far / v_near) ** (1 / gap) - 1`.

| implied yearly rate of the far option | `both` rows correct (overnight, 2-option types) |
|---|---|
| < 8% | 2/9 |
| 8-15% | 7/11 |
| 15-30% | 8/12 |
| 30-100% | 19/29 |
| > 100% | 69/92 |

17 overnight `both` items are lost to the near option reinvested at 10%/yr; the model is "right" on
5 of them. These items have no defensible truth under the original wording.

**Not all errors are disputable.** Above a 30%/yr implied rate the model still takes the near
option often, and more so at long horizons: correct on 32/35 with a horizon under a year, 53/64 at
1-10 years, 51/75 at 10 years or more. In the traces this is a heuristic, not arithmetic: "the
$1,000 now has 20 years to compound, so it beats $4,000 in a year".

**The horizon is read correctly in wrong answers.** A ridge probe for log horizon, trained on the
full bank (Qwen3-8B, thinking on) and applied to the eval types, has the same error on wrong and
right `both` rows (median 0.74 vs 0.75 decades at L18, end of prompt; p = 0.85). The decision fails
after the horizon is encoded.

**The main bank has the same problem.** In the full bank (Alan's replication, 500 prompts),
25 of 93 `both` items have an implied rate under 10%/yr. On the overnight Qwen3-8B think-on run of
that bank the model takes the far option in 50/93 `both` items (8/25 below 10%/yr, 27/44 at 25%/yr
or more) against 81/83 correct in `near`. The crossed bank is nearly clean (4/166 under 10%/yr).

## 2. Probe transfer (local, saved activations)

The full-bank probe reads the horizon well on 8 of 20 eval types at the end of the prompt (L18:
Spearman 0.91-0.97, error 0.2-0.6 decades); the two roi types rank well (0.91, 0.95) but sit 2 decades
low. It fails where the horizon is not near the end
of a single user turn: `system_prompt` 0.22, `multiturn_analyst` 0.36, `prose_natural` 0.45,
`json_io` biased by 8 decades. Adding the other eval types to the training set
(leave-one-type-out) barely helps them at the end of the prompt (0.51, 0.35, 0.65, 0.74), but at the
label slot after thinking (L22, prelabel) they transfer much better (0.72, 0.65, 0.67, 0.92). The
horizon direction at the end of the prompt is tied to the template; after thinking it is closer to
format-free. Script: `pipeline/s3_analysis/eval_types_probe.py`; table in the run directory's
`probe_transfer.md`.

## 3. The v2 wording

Three changes, each one sentence (`spar_horizon/eval_types/wording.py`):

| part | original | v2 |
|---|---|---|
| SITUATION | Plan for the future of the household. | ... Any money the household receives is kept as cash and earns no interest. |
| TASK | ... tasked to choose the best investment: | ... tasked to choose which payment the household receives: |
| CONSTRAINT | ... the greatest benefit for this time horizon: X. | ... leaves the household with the most money at the end of this time horizon: X. |

With all three the truth is the largest payout received within the horizon, with no assumption
about rates. The horizon still decides which options count, so the eval still tests horizon use.
The option lines are unchanged, so the tokens of each option are the same in both wordings.

## 4. The v2 run

Types (each pairs prompt by prompt with its counterpart, via `EvalType.item_seed`):

- `t0_w0`, `t0_w1_cash`, `t0_w3_clear`: T0 in the original wording, with CASH, and with all three
  changes; the same 40 items (12 near, 24 both, 4 none) and 8 swaps.
- `roi_unequal_clear`, `system_prompt_clear`, `sycophancy_lean_clear`, `horizon_revised_clear`: four
  overnight types with all three changes, on their overnight items; the overnight rows are the
  comparison.
- `free_form_anchor`: the free-form term with the horizon given by an anchor ("Your partner is 8
  weeks pregnant"), so the term cannot copy the horizon string. 21 prompts, 7 anchors.

Analysis: `pipeline/s3_analysis/eval_types_paired.py` (paired accuracy, exact McNemar on discordant
pairs) and `eval_types_score.py`.

### Predictions (written before the run)

1. CASH raises `both` accuracy (`t0_w0` -> `t0_w1_cash`, 28 paired rows) from about 0.7 to 0.9 or
   more, with `near` unchanged; the gain is largest below a 25%/yr implied rate.
2. All three changes (`t0_w3_clear`) do at least as well as CASH alone, with shorter thinking.
3. `roi_unequal_clear`: force-closed rows drop from 24/32 to under 12/32, and accuracy rises from 12/24.
4. `system_prompt_clear` `both` accuracy rises from 5/12; `sycophancy_lean_clear` follows the lean less
   than the overnight 5/24; `horizon_revised_clear` still follows the stale horizon about as often
   (6/24), since that error is about which horizon, not about reinvestment.
5. `free_form_anchor`: the term is within a factor 1.5 of the anchored horizon on at least 15/21.

## 5. Results (run 2026-11-17, Qwen3-8B think-on, RTX 4090 in France)

Rows in `$RESULTS/horizon/evaltypes_Qwen3-8B_think-on/<type>/`; `summary.md` and `paired.md` there
cover the overnight and v2 types together (the overnight types rescore identically; the overnight
summary and logs are kept in `overnight_2026-11-16/`). Paired rows, exact McNemar on discordant
pairs. Base + swap rows with a truth.

| A -> B | all | `near` | `both` | forced A / B | think median A / B |
|---|---|---|---|---|---|
| `t0_w0` -> `t0_w1_cash` | 33/44 -> 43/44 (p=0.006) | 16/16 -> 16/16 | 17/28 -> 27/28 (p=0.006) | 5 / 5 of 48 | 1256 / 1393 |
| `t0_w0` -> `t0_w3_clear` | 33/44 -> 43/44 (p=0.006) | 16/16 -> 16/16 | 17/28 -> 27/28 (p=0.006) | 5 / 2 of 48 | 1256 / 826 |
| `roi_unequal` -> `_clear` | 12/24 -> 24/24 (p=0.0005) | 7/16 -> 16/16 | 5/8 -> 8/8 | 21 / 6 of 28 | 3072 / 1965 |
| `system_prompt` -> `_clear` | 17/24 -> 23/24 (p=0.03) | 12/12 -> 12/12 | 5/12 -> 11/12 (p=0.03) | 2 / 4 of 28 | 747 / 779 |
| `horizon_revised` -> `_clear` | 18/24 -> 24/24 (p=0.03) | 12/12 -> 12/12 | 6/12 -> 12/12 (p=0.03) | 6 / 4 of 28 | 923 / 1122 |
| `sycophancy_lean` -> `_clear` | 19/24 -> 17/24 (p=0.73) | 12/12 -> 11/12 | 7/12 -> 6/12 | 5 / 7 of 28 | 1059 / 1688 |

`free_form_anchor`: term within a factor 1.5 of the anchored horizon on 19/21 (Spearman 0.96).

### Predictions, scored

1. **Held, and more broadly than predicted.** CASH alone takes `both` from 17/28 to 27/28 with
   `near` unchanged. The gain is not confined to low implied rates: above 100%/yr it is 13/19 ->
   19/19. Those were the "early money compounds longer" errors; once the prompt rules out interest,
   the heuristic goes too.
2. **Held.** All three changes match CASH alone (27/28) and think 40% less (median 826 vs 1393
   tokens), with fewer forced closes (2 vs 5) and full order agreement (8/8).
3. **Held.** `roi_unequal_clear` goes from 12/24 to 24/24, forced closes from 24/32 to 9/32 (all
   rows), order agreement from 3/8 to 8/8.
4. Mixed.
   - `system_prompt_clear`: held (`both` 5/12 -> 11/12).
   - `horizon_revised_clear`: **wrong.** I predicted the stale-horizon errors would stay (6/24);
     they went to 0/24. All 6 overnight "stale" errors were `both` rows, and the scorer's
     `chose_stale_truth` counts any wrong answer there, so it was measuring the reinvestment error,
     not stale-horizon following. With the clear wording the model uses the updated horizon every
     time.
   - `sycophancy_lean_clear`: **wrong, and the type was built wrong.** Its TASK line kept the
     assistant wording "choose the best investment" (`ASSISTANT_TASK`), so it has only CASH and
     END_CONSTRAINT, not RECEIVE_TASK. The lean is followed slightly more (11/28 vs 8/28, n.s.). All
     7 wrong answers follow the lean, and the traces invent compounding to justify it ("the user is
     leaning towards b), so the reasoning would support that ... investing $2k now gives more time
     to grow"), despite the no-interest sentence. With the reinvestment excuse removed, this type
     now measures lean-following cleanly: every error is one.
5. **Held.** 19/21 within a factor 1.5. The two misses are anchor arithmetic, not copying: "8 weeks
   pregnant" answered as 8 weeks (truth ~32), and a passport anchor answered with the 10-year
   validity (truth 3 years).

### Remaining errors under the clear wording

- `t0_w1_cash` (1 row): ignores CASH and argues the $1,000 "has 5 years to compound".
- `t0_w3_clear` (1 row): reads "2,000 dollars in 2 weeks" as a recurring payment ("otherwise the
  time horizon is irrelevant"). A sentence "Each option is a single, one-time payment." would close
  this.

### Cost

$0.64 on two failed China hosts overnight, $0.16 on a Texas host that never left "loading", $1.20
on the France host that ran it (2.8 h at $0.43/hr). $2.00 in all (local `just burn` estimate).

### To rerun

About 290 prompts, about 2.5 hours on one RTX 4090 at batch 5 plus a 2-minute model download on a
good host. `SPAR/.env` now restricts offers to North America and Europe (`GEO_FILTER`).

```
just search                      # make sure the cheapest offer is not one of the failing hosts
just up 3h
just push
just ssh "tmux new -d -s evaltypes2 'bash -l /workspace/SPAR/spar-planning-temporal-manifolds/jared/pipeline/s1_data_vectors/queues/queue_evaltypes_v2.sh; bash'"
just hold                        # the parallel model download keeps the GPU idle; just unhold once generation starts
# ... when $LOGS/evaltypes.done exists:
just down
~/miniconda3/envs/mp/bin/python -m pipeline.s3_analysis.eval_types_score  ~/SPAR/results/horizon/evaltypes_Qwen3-8B_think-on
~/miniconda3/envs/mp/bin/python -m pipeline.s3_analysis.eval_types_paired ~/SPAR/results/horizon/evaltypes_Qwen3-8B_think-on
```

`bash -l` matters: the HF token is set in `/etc/profile.d/spar.sh`, which only login shells read, and
the unauthenticated download stalled.

## 6. Recommendations

1. Build new banks with the clear wording (CASH, RECEIVE_TASK, END_CONSTRAINT). The run shows CASH
   alone removes nearly all `both` errors; the other two cut thinking by 40% and forced closes from 5 to 2. Without
   CASH the `both` truth depends on a reinvestment rate the prompt never states.
2. For the full bank (Alan's replication), report `both` accuracy separately for items whose far
   option's implied rate is at least 25%/yr, or drop the others: 25 of its 93 `both` items have an
   implied rate under 10%/yr, and the model takes the far option on only 8 of them.
3. Read horizon probes at the label slot (after thinking) when comparing formats. At the end of the
   prompt the full-bank direction does not transfer to system-message, multi-turn or prose prompts.
4. Retire `free_form` in favour of `free_form_anchor`: with a stated horizon the term is a copy.
5. Measure stale-horizon following with `horizon_revised_clear`, not `horizon_revised`: under the
   original wording its "stale" errors were reinvestment errors.
6. For v3: give `sycophancy_lean_clear` a receive-payment TASK line in the assistant voice ("Help
   me, the head of the household, choose which payment the household receives:"), and add "Each
   option is a single, one-time payment." to the clear wording.

## 7. Distance from Alan's prompt (2026-11-20, local, saved activations)

Goal: rank the existing eval types by how far Qwen3-8B's internal state for them is from its state
for Alan's prompt, and label every row, so the steering runs (E8 answer slot, E13 suffix, E14
horizon phrase) can be read against distance. Script:
`python -m pipeline.s3_analysis.eval_types_distance $RESULTS/evaltypes_Qwen3-8B_think-on
$RESULTS/exp0_Qwen3-8B_think-on_full` (the full bank supplies the subspaces below). It writes
`eval_distance.md` (the full table), `eval_distance.json` (per type, every layer and position, CIs)
and `eval_distance_rows.json` (per row: type, family, variant, regime, horizon, matched original,
pairing, distances) into the run directory. CPU only, about a minute.

**Distance.** Each row is compared with the same decision in Alan's template: a `base` row with its
item's twin (same options, order and horizon; 4 pairs per type), and the wording ladder with the
`t0_w0` row of the same item (48 pairs). Cosine distance of the hidden states after subtracting the
mean over all rows, divided by the median distance between two *different* Alan prompts of the same
regime. So 0 = same state as Alan's prompt, 1 = as far as another Alan prompt with other numbers.
`free_form` and `free_form_anchor` have no twin; they only get the unpaired distance (to the mean Alan
prompt; Alan prompts sit near 1), which ranks the paired types similarly (Spearman 0.77).

**Ranking** (L18, last prompt token; median [bootstrap 90%]):

| rank | type | distance | | rank | type | distance |
|---|---|---|---|---|---|---|
| 1 | roi_unequal (text = twin) | 0.00 | | 14 | role_fund_manager | 4.11 |
| 2 | t0_w0 (reference) | 0.00 | | 15 | public_health | 4.18 |
| 3 | unit_mismatch | 1.27 [1.09, 1.44] | | 16 | t0_w3_clear | 4.29 [4.23, 4.35] |
| 4 | roi_unequal_clear | 1.64 | | 17 | horizon_revised_clear | 5.19 |
| 5 | implicit_vague | 1.78 | | 18 | coding_agent | 5.58 |
| 6 | distractor_temporal | 2.19 | | 19 | role_assistant | 6.13 |
| 7 | roi_equal_10 | 2.31 | | 20 | system_prompt | 6.26 |
| 8 | implicit_concrete | 2.40 | | 21 | sycophancy_lean | 6.78 |
| 9 | t0_w1_cash | 2.59 [2.54, 2.65] | | 22 | prose_natural | 7.22 |
| 10 | city_no_values | 2.93 | | 23 | system_prompt_clear | 7.24 |
| 11 | horizon_revised | 3.71 | | 24 | sycophancy_lean_clear | 7.36 |
| 12 | loss_framing | 3.72 | | 25 | multiturn_analyst | 7.77 |
| 13 | city_values | 4.05 | | 26 | json_io | 7.84 |

The order follows how much of the template a type rewrites: one phrase (unit, distractor, implicit
horizon) < the TASK/SITUATION line (domains, fund manager) < the speaker (assistant, sycophancy,
system message) < the whole layout (prose, multi-turn, JSON). The clear wording moves every type
further out: +0.6 (sycophancy_lean), +1.0 (system_prompt), +1.5 (horizon_revised), +1.6
(roi_unequal); on Alan's own prompt the cash sentence alone is 2.6 and all three changes 4.3.

**Checks.**
- The ranking does not depend on the layer: Spearman 0.99 between L14, L18 and L22 at the last
  prompt token, 0.97 at the label slot.
- `roi_unequal` rows are word for word their twins: distance 0.00 at both positions.
- The same twin prompt run on two different boxes (a v2 type and its original): median 0.11, max
  0.16 at the last prompt token. That is the noise floor; every non-identical type is above 1.2.
- At the label slot (after thinking) the same check gives median 0.47, max 1.66, as large as most
  type distances there (0.4-2.8): one flipped token changes the whole trace. **Use the last prompt
  token for ranking; label-slot distances mostly measure the trace, not the prompt.**

**Preview against behaviour** (Spearman over the 25 paired types, no correction for the 4 tests):
- Distance vs the full-bank probe's transfer (Spearman of its readout, L18 last prompt token): -0.41, p = 0.04.
  Types further from Alan's prompt are read worse by the probe trained on Alan's prompts. This is
  the first hint that a direction found on the bank, so also steering along it, will weaken with
  distance. (`probe_transfer` was rerun to cover the v2 types; the 20 overnight types are unchanged,
  and the old file is in `overnight_2026-11-16/`.)
- Distance vs accuracy: -0.05 (p = 0.83). How far a prompt is from Alan's says nothing about
  whether the model gets it right; accuracy is set by the wording (cash sentence), not the format.

**The same distance in the subspaces steering uses** (L18). The full-space cosine is dominated by
format directions, so it is repeated in two subspaces fitted on the full bank:
- *horizon*: the 8-d steering subspace (top PCs of the bank's per-horizon averages, `fit_cell`'s
  basis, the one E8/E13 steer in);
- *bank PCA*: the bank's top PCs covering 90% of its variance (k = 20 at the last prompt token, 56 at
  the label slot).
Plus two direct measures: *probe shift*, |bank probe readout - its Alan twin's| in decades, and
*outside*, the share of a row's variance outside the bank PCs. Noise floors (same prompt, two
boxes) at the last prompt token are small: horizon 0.03, bank PCA 0.02, probe shift 0.04 decades.

| type | full | horizon | bank PCA | outside | probe shift (decades) |
|---|---|---|---|---|---|
| Alan's prompts (t0_w0, twins) | 0 | 0 | 0 | 0.16 | 0 |
| unit_mismatch | 1.27 | 1.90 | 0.90 | 0.31 | 0.45 |
| distractor_temporal | 2.19 | 1.30 | 1.29 | 0.59 | 0.30 |
| t0_w1_cash | 2.59 | 1.57 | 1.43 | 0.60 | 0.68 |
| city_values | 4.05 | 5.18 | 2.44 | 0.75 | 0.22 |
| role_fund_manager | 4.11 | 6.40 | 2.18 | 0.69 | 0.11 |
| public_health | 4.18 | 3.83 | 2.25 | 0.78 | 1.70 |
| coding_agent | 5.58 | 8.78 | 3.11 | 0.74 | 1.06 |
| system_prompt | 6.26 | 7.59 | 3.85 | 0.79 | 2.34 |
| prose_natural | 7.22 | 10.52 | 4.74 | 0.87 | 4.85 |
| multiturn_analyst | 7.77 | 9.37 | 4.68 | 0.78 | 6.14 |
| json_io | 7.84 | 11.98 | 5.19 | 0.88 | 9.93 |

(All types in `eval_distance.md`.) What it shows:
- The three spaces rank the types alike (Spearman 0.92-0.97), so the ranking is not an artefact of
  the full space.
- In the steering subspace, almost every type is further from its Alan twin than two Alan prompts
  with *different horizons* are from each other (distance > 1). Format moves the state along the
  steering axes more than the horizon itself does. Role and domain changes (fund manager, coding
  agent, city) move it more there than their full-space rank suggests; the distractor and the cash
  sentence less.
- Most of each eval type's state lies outside the bank's top PCs (30-88%, against 16% for Alan's
  prompts). `roi_unequal` is word for word its twin and still 84% outside: the bank's PCs do not
  cover the 8-option layout, so for multi-option types even the matched original is off-bank.
- The bank probe reads the horizon decades off on the reformatted prompts: 9.9 decades on JSON, 6.1
  on multi-turn, 4.9 on prose, 2.3 on system message; under 1 on the template-preserving types.
- Against behaviour (25 types, 20 uncorrected tests): horizon-subspace distance predicts probe
  transfer slightly better than the full space (-0.49, p = 0.01, against -0.41); probe shift does
  most (-0.68, p < 0.001), but it is partly the same measurement. No space predicts accuracy at the
  last prompt token (|ρ| <= 0.22). At the label slot horizon distance correlates with accuracy
  (+0.52, p = 0.008), but the label-slot noise floor there is as large as the signal (median 0.45,
  max 2.7), so I would not read anything into it.

**For the steering runs.** Join results on `sample_uid` with `eval_distance_rows.json` and plot the
steering effect against `distance.L18_suffix_end.paired` (or the type's median). Which modality
applies to each type is in `eval_distance.json` (`steering`): E8 needs a label answer (not the two
free-form types); E14 steers a life-event phrase in the implicit and anchor types, and has two
horizon phrases (stale, updated) in `horizon_revised*`. E14's position is not saved, so its own
distance needs a short GPU pass that saves the horizon-phrase token; until then the last prompt
token stands in. Only 4 pairs per twin type: per-type medians are tight here, but a steering
comparison per type will have few rows; the wording ladder (48 pairs) is the well-powered rung.

## 8. E19: the bank's answer-slot steering on the eval types (2026-11-21, RTX 4090 in Estonia, $0.18)

Question: do the steering vectors fitted on Alan's prompts (full bank, E8's L24 pre-label cells) still
move the choice on the new prompt types, and does that weaken with distance (§7)?

Run: `evaltypes_steering_prepare` (local) -> `exp19_evaltypes_steer` (box, 9 min) ->
`eval_types_steer` (local; writes `eval_steer.{json,md}` into `results/horizon/exp19_Qwen3-8B_evaltypes/`).
21 two-option label types, near and both rows (base, swap, twin), 636 rows thinking off and 625
thinking on. Thinking off feeds the prompt with "I choose:" prefilled. Thinking on feeds E0's own
trace, from gen_cache, cut before the label. Its unsteered log-odds reproduce E0's saved ones (median
|gap| 0.25 logits, max 1.0, all 625 choices agree). The 11 rows dropped are twins whose cached
generation is the v2 rerun, not the one the activations came from. Per row: unsteered, curve ±2 decades,
random direction of the same length ±2. Per type: within-type answer-slot patching, all layers (ceiling).
Left out: `json_io`, `loss_framing`, the multi-option and free-form types.

Gate, fixed before the run (accuracy >= 0.65, order agreement >= 0.75, near/both contrast >= 2
logits): thinking off 13 of 21 types pass, thinking on 19 of 21.

**Results** (toward = shift toward the answer the steered horizon makes rational, on the rows the
step crosses; specific = curve minus random, in units of the type's own contrast; raw = curve shift in logits):

| | thinking off | thinking on |
|---|---|---|
| bank (E8) | 0.38-0.39 | 0.14-0.15 |
| `t0_w0` (Alan's wording, this run) | specific 0.30, raw 3.4 logits | 0.12, 2.2 logits |
| gated types, range of specific | 0.12-0.51, except coding_agent 0.00 | 0.04-0.24, except coding_agent -0.01 |
| gated types, range of raw shift | 3.4-8.8 logits, prose 2.0, coding_agent 0.6 (CI spans 0) | 1.2-3.1 logits, coding_agent -0.2 (CI spans 0) |
| random direction, same length | <= 0.05 everywhere | <= 0.03 everywhere |
| label flips (rows that switch answer) | 0-13 per type | 0-3 per type |

- **Steering transfers to almost every format.** The bank's curve moves the choice on prompts
  far from Alan's: system message (specific 0.34 off / 0.16 on), prose, multi-turn, fund manager (0.51
  off), sycophancy (0.24 on). Every gated type except one is clearly above its random control.
- **Distance does not predict how well it transfers.** Across gated types, no distance (full, horizon
  subspace, probe shift; L18 suffix or L24 pre-label) correlates with the effect beyond chance (10 tests per
  mode; the strongest, L24 pre-label full distance vs effect over ceiling thinking on, is -0.47,
  p = 0.04, uncorrected). The raw shift does not fall with distance either (Spearman +0.17 off, -0.14 on).
  Contrast-scaled effects are smaller on types the model is very sure of (city, multi-turn), because
  the raw shift is similar and the contrast larger.
- **`coding_agent` is the exception: steering does nothing in either mode.** Its decision is in
  the slot as usual (patch transfer 0.52 at L24 off, 0.46 on, onset L24-25), its gate passes, its
  contrast is large (30). So the decision sits where steering acts, but the bank's horizon curve is
  not the direction it reads there. It is the only type whose options are not money or public
  goods (engineering projects that deliver a benefit after a delay). Why is untested.
- **The patching ceiling is format-independent.** Onset at L24-25 in almost every type and both
  modes (L24 transfer 0.3-0.8). Exceptions: `implicit_vague` thinking on (onset L27), and sycophancy
  thinking off, where patching barely transfers before L28-30: the decision there follows the user's
  stated lean, not the horizon.
- **Thinking off is unreliable on the eval types, as feared.** 8 of 21 fail the gate, mostly
  on order agreement (0.25-0.62), and `t0_w0` itself fails (order 0.62). Sycophancy thinking off
  follows the user's lean: accuracy 0.25 (`sycophancy_lean`) and 0.46 (`_clear`), negative contrast.
  The thinking-on gate is the one to trust.
- **Thinking-on steering moves log-odds but almost never the answer.** A 1-3 logit shift against a
  12-35 logit contrast: 0-3 flips per type. As on the bank, a ±2-decade step at the answer slot is
  too small to flip a reasoned decision.

**What to do next** (each needs a spec):
1. `coding_agent`: check whether its L24 pre-label states lie on the bank's curve (the probe readout
   and the curve coordinates of near vs both rows). This tells whether the horizon is encoded
   differently there or not read at all.
2. A dose curve (±1, ±2, ±4 decades) thinking on, on 4-5 types, to see whether larger steps flip
   reasoned answers, and whether the dose needed grows with distance.
3. Distance as a predictor is not supported at the answer slot. If it matters, it should show where
   format changes the input: steering before the reasoning (E13-style, at the suffix), where §7's
   distances are measured. That costs about 10x more (generation).

## 9. E21: cheap variants of the answer-slot edit (2026-11-22, RTX 4090 in Quebec, $0.63)

**Why.** E19's L24 curve step transfers to the eval types but barely moves a reasoned answer. Five
cheap variants at the same site (the pre-label token, same rows and saved reasoning as E19, one-token
passes on a cached prefix, no generation) ask how to make it stronger and whether the answer needs the
horizon there at all. Code: `exp21_evaltypes_variants.py`, analysis `pipeline/s3_analysis/eval_types_variants.py`,
results `~/SPAR/results/horizon/exp21_Qwen3-8B_evaltypes/eval_variants.{md,json}`. Unsteered scores
match E19 exactly; steered ones differ by 0.01 logits on average (bf16 rounding of a refitted basis).

**Readouts, fixed before the run** (gated types, effect toward the newly rational answer in units of
each type's contrast, mean over types with 95% CI over types):

| test | thinking off (13 gated types) | thinking on (19 gated) |
|---|---|---|
| dose: L24 curve 2 decades / 4 decades | +0.28 / +0.19 (ratio 0.68) | +0.10 / +0.05 (ratio 0.47) |
| random 0 at 4 decades, largest type | 0.05 (clean) | 0.03 (clean) |
| curve / mean difference (CAA) / probe line, 2 decades | +0.28 / −0.04 / +0.01 | +0.10 / −0.07 / −0.00 |
| curve minus CAA | +0.33 [+0.23, +0.43] | +0.17 [+0.12, +0.23] |
| curve by layer, L16 / 20 / 22 / 24 / 26 / 28 / 32 | .00 / .01 / .09 / **.28** / .14 / .06 / .06 | .00 / .01 / .04 / **.10** / .07 / .05 / .04 |
| stacked L22+24+26+28, 2 decades each | **+0.47** [+0.36, +0.59] | **+0.25** [+0.21, +0.30] |
| removal (k = 8): types where the horizon subspace removes more than 3 random ones | 13/13 | 19/19 |

Flips on gated rows that chose the old rational answer: thinking off 71/308 at L24 ±2, 37/308 at ±4,
133/308 stacked; thinking on **8/456 at L24 ±2, 2/456 at ±4, 50/456 stacked** (median shift 2.0, 1.0
and 5.8 logits).

**Findings**
- **Bigger steps along the curve do worse, not better.** At ±4 decades the effect falls to 0.5-0.7 of
  the ±2 effect, in both modes and every type but `coding_agent` thinking off, while random ±4 stays at the noise floor (so it is
  not damage). The steps that rows actually get at ±4 average 3.0-3.7 decades (the curve ends at the
  bank's shortest and longest horizons). A likely cause, untested: the curve's far ends are not just
  "more horizon". In the bank every very short horizon is a `both` prompt, so its average there also
  carries "neither option arrives", which the model answers like a long horizon (the same confound
  that turns CAA around, E20 page).
- **The curve's shape matters; a straight line does not work.** CAA (long minus short average) pushes
  the wrong way in all 19 gated types thinking on, as on the bank. The probe line is inert, as on the
  bank. The curve beats CAA by +0.33 (off) and +0.17 (on), CIs well clear of 0.
- **L24 is the layer, in both modes and every type.** Steering at L16 and L20 does nothing, L22 a third
  of L24, and the effect decays after L26. This matches where answer-slot patching starts to transfer
  (E19 onset L24-25).
- **Stacking L22-L28 is the strongest variant, and the only one that flips reasoned answers.** It
  makes the thinking-off effect 1.7x and the thinking-on effect 2.5x stronger (+0.25), flipping 11% of
  rows (50/456) against 2% at L24 alone. Still small against the E20 phrase donor patch (27-100% of
  rows flipped).
- **The answer token needs the horizon subspace.** Mean-ablating the top 8 horizon directions at L24
  removes 0.25-0.50 of the near-minus-both contrast thinking on in every gated type, and 0.11-0.49
  thinking off; three random 8-dim subspaces remove at most 0.02. With k = 2 it is already 0.16-0.30
  thinking on in 18 of 19 types (`unit_mismatch`: -0.01). Caveat: E8 found part of the decision direction inside this subspace, so this shows the
  subspace carries what the answer reads, not that the horizon alone does.
- **`coding_agent` needs the subspace but does not follow the curve.** Removal takes 0.39 of its
  contrast thinking on (k = 8), like other types, yet the curve does nothing at any layer (stacked:
  +0.09). So its answer reads these directions, but moving along the bank's curve from the row's stated
  horizon does not move it toward the rational answer. A guess, untested: its states sit elsewhere in
  the subspace than the bank curve predicts for its horizon.
- **Gate failures behave as expected.** Thinking-off sycophancy moves the wrong way (the answer follows
  the user's lean); `unit_mismatch` thinking off shows huge contrast-scaled effects only because its
  contrast is tiny (it fails the gate).

**Cost.** $0.63 for 1.55 h at $0.401/hr, of which the run used about 20 min. About an hour was lost to
a failed launch (`python` is not on the box's login PATH; use `/venv/main/bin/python`) that my wait
loop did not catch. Session total with E19: $0.81 of $1.50.

**What to do next** (each needs a spec)
1. A stacked dose at ±1 and over more layers (L22-L30) thinking on, since stacking is the one lever that
   grows, and ±1 per layer may avoid the curve's far end.
2. `coding_agent`: read its L24 states in curve coordinates (saved activations, CPU) to check whether
   they sit where the bank curve predicts for their horizon.
3. Separate horizon from decision in the removal: ablate the horizon subspace without E8's decision
   direction, as E8's T2 did on the bank.
