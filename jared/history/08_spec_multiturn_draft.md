# Spec: E6, the horizon across turns (DRAFT, waiting on Jared's answers)

Date: 2026-10-01. Companion to `02_execution_plan.md` (E0-E5) and
`07_results_think_geometry.md`. Decision points are marked **[Q1]**..**[Q6]**
and listed at the end.

## 1. Question

E0-E5 measure the horizon inside one user turn. Real use is a conversation:
a horizon stated earlier, a decision asked later. Three things we do not
know:

1. **Override.** Turn 1 states horizon H1, turn 2 states a different H2.
   Does the turn-2 representation read H2 cleanly, or a mixture of H1 and
   H2? Does the turn-2 choice follow H2?
2. **Carry-over.** Turn 1 states H1, turn 2 states no horizon. Does the
   model re-instantiate H1 at turn 2 (on the same axis, in the same
   layers), and does its choice track H1?
3. **Context cost.** Does any prior turn, horizon or not, move the
   geometry relative to the single-turn baseline?

The prefilled turn-1 answer is authored by us, so turn 1 is a fixed,
clean context: one question, one committed choice, no reasoning text.

## 2. Fixing the prompt bank first: counterbalanced option order

In every run so far option a) is the near payout. In the v3 8B runs the
answer is "a)" on all 10 horizons under 5 years, so **label = horizon** for
most of the bank. Consequences we have been carrying:

- the label probe (1.00 at pre-label) is partly a horizon probe;
- "picked a)" conflates position bias, label bias and the rational choice;
- the order-stability test is the only thing separating them, after the fact.

E6 (and a counterbalanced E0 rerun, which E6 produces as its `single`
condition) uses a bank where **the near option is a) for exactly half of
the prompts at each horizon** (3 of the 6 reward×delay combos, assignment
seeded and recorded as `near_label`). Labels stay a)/b). Every behavior
number is then reported in terms of **near vs far**, not a vs b, and the
label probe becomes two probes: one for the label written, one for the
option chosen. With counterbalancing these are decorrelated by
construction. Turn 1 and turn 2 are counterbalanced independently, so the
turn-1 label never predicts the turn-2 label.

The rational rule (from `only_near_delivers`): near if `near_delay <= H <
far_delay`, far if `H >= far_delay`, undefined if `H < near_delay`
(nothing pays in time; thinking-on 8B says "neither is feasible" and loops).

## 3. Conditions

Turn 2 is always the measured turn. Turn 1 varies.

| cond | turn 1 (user, prefilled assistant) | turn 2 (user) | what it tests |
|---|---|---|---|
| `single` | none | horizon H2, counterbalanced | baseline; the counterbalanced E0 |
| `unrelated` | a non-temporal a/b question, answered | horizon H2 | context cost of any prior turn |
| `same` | horizon H1 = H2, other reward/delay combo, answered | horizon H2 | priming/consolidation |
| `conflict` | horizon H1 != H2, answered | horizon H2 | override, interference vs |log H1 - log H2| |
| `carry_silent` | horizon H1, answered | no horizon sentence (the E0 control wording) | spontaneous carry-over |
| `carry_ref` | horizon H1, answered | "Same time horizon as before." | retrieval on request **[Q2]** |

Bank sizes per thinking mode: `single` 96 + 20 no-horizon controls,
every other condition 96 (16 horizons × 3 rewards × 2 delays for the
turn-2 slot). About 600 generations per mode. Thinking off: minutes.
Thinking on at a 3072 budget: ~2.5 h on the 4090 at E0's rate, so the
thinking-on half runs `conflict` and `carry_*` first **[Q5]**.

### Turn 1 details

- **H1 choice.** `same`: H1 = H2. `carry_*`: H1 sweeps the 16 horizons
  (H2 is absent). `conflict`: H1 = horizon at index `i2 + d` for
  `d` cycled over {-8, -4, -2, +2, +4, +8} across the 6 combos per H2,
  clipped to the list ends and the realized `log10(H1/H2)` recorded. This
  gives a spread of distances including sign, without wrapping 500 years
  onto 30 seconds.
- **Turn-1 reward/delay combo** is the next one in the 6-cycle after
  turn 2's, so turn 1 is never a verbatim repeat of turn 2.
- **Prefilled answer** is the rational rule's choice, written exactly as
  the model writes its own answers: `I choose: a)`. For H1 below the near
  delay the rule is undefined; those H1 are **excluded from turn 1**
  (turn 2 keeps the whole sweep) **[Q3]**. Alternative: prefill the model's
  own E0 answer, which makes turn 1 self-consistent but imports the
  position bias and the forced-close noise.
- **Unrelated turn 1:** same two-option shape, same answer format, no
  time: "Which is heavier? a) 2 kilograms of iron b) 1 kilogram of
  feathers. Answer with a) or b)." with the correct answer prefilled,
  order counterbalanced.

### Turn 2 wording

The constraint and option lines are the E0 starter template, unchanged,
so the turn-2 suffix is directly comparable to E0. No connective ("Now a
second decision:") by default; `--turn2-lead` adds one as a variant
**[Q4]**.

## 4. Positions and measurement

Causal attention means every turn-1 position has the same hidden state as
in a single-turn run, so nothing new is kept there. The kept layout is
E0's, applied to the turn-2 assistant turn: `suffix | think | pre |
prelabel | answer`. The probe position is the `\n` after turn 2's
`<|im_end|>` as before.

## 5. Analyses (per thinking mode, per condition)

1. **Fixed-probe transfer.** E0's suffix probe (trained on `single`)
   read at the turn-2 suffix: Spearman rho with H2 (`same`, `conflict`,
   `unrelated`) and with H1 (`carry_*`). Reports whether the single-turn
   axis is the one used in context.
2. **Mixing weights.** Fresh regression at every layer at the probe
   position: `reading ~ a·log H1 + b·log H2`, with 5-fold CV R². `b ≈ 1,
   a ≈ 0` is a clean override; `a > 0` is leakage. Reported per layer and
   at the think/pre/prelabel positions with thinking on.
3. **Interference curve.** `conflict` only: probe error and near/far flip
   rate (vs the same turn-2 prompt in `single`) binned by `log10(H1/H2)`,
   signed. Does a longer H1 pull the reading up and the choice toward far?
4. **Carry-over geometry.** `carry_*`: PCA at the display layer colored by
   H1; fresh-probe R² for H1 per layer; compared against the 20 no-horizon
   controls in `single`, which show where "no horizon" sits.
5. **Behavior.** Per condition: picked-near rate vs H2 (or H1 for carry),
   agreement with the rational rule, and a-rate (position bias, now
   measurable directly). The three E0 coherence tests run on `single` only.
6. **Thinking on.** Fraction of turn-2 think texts that quote the turn-1
   horizon string; dense track at the think-fit layer (L24-30 per
   `07_results`) rather than L11 **[Q6]**.
7. **Label vs choice probes** at the four E0 label positions, on `single`,
   now that they are separable.

Outputs: `exp6_<model>_think-<mode>/<condition>/activations.npz` +
`answers.json`, one `mixing.csv` and `interference.csv`, figures
`mixing_by_layer.png`, `interference.png`, `carry_pca.png`,
`behavior_by_horizon.png`, a manifest with every number above.

## 6. Code changes (small, additive)

| file | change |
|---|---|
| `prompts.py` | `choice_prompt(..., near_first=True)`; `build_prompts(counterbalance=True)` adds `near_label`; `rational_choice(rec)`; `build_multiturn(condition, seed, limit)` returning records with `turns=[(user1, assistant1)]`, `prompt`, `h1`, `h2`, `near_label1/2`; `UNRELATED_TURN` |
| `model_io.py` | `chat_text` and `suffix_length` accept `history` (list of (user, assistant) pairs); `generate_batch(prompts, ..., histories=None)`; the cache key already covers the full ids |
| `extract.py` | pass `histories` through; nothing else |
| `behavior.py` | `picked_near(rec, answer)`, rule agreement; existing tests unchanged |
| `geometry.py` | `mixing_weights(X, log_h1, log_h2)` |
| `experiments/exp6_multiturn.py` | conditions loop, analyses, figures, manifest; `--conditions` to select |
| `experiments/run_all.sh` | add E6 after E0 |
| tests | bank balance (3/3 per horizon), `conflict` distance spread, rational rule table, two-turn text contains both turns and ends with the same suffix tokens as single-turn, `near_label` round-trips from the prompt text, tiny-model smoke in both modes |

`swap=True` in `choice_prompt` stays for the E0 order-stability test;
`near_first` is the counterbalanced equivalent with metadata.

## 7. Tasks

Phase A (local, no model): A1 counterbalanced bank + `rational_choice` +
tests. A2 `build_multiturn` for all six conditions + tests. A3 `history`
in `chat_text`/`suffix_length`/`generate_batch` + tests on the 0.8B
tokenizer. A4 `mixing_weights` + planted-signal test.

Phase B (local 0.8B, thinking off, `--limit 12`): B1 `exp6` runs
`single` and writes a manifest. B2 all conditions, every figure renders.
B3 thinking-on smoke at a 32-token budget. B4 `run_all.sh`.

Phase C (box, Qwen3-8B): C1 thinking off, all conditions. C2 thinking on,
`conflict` + `carry_*` first, then the rest if time allows. C3
`09_results_multiturn.md`.

## 8. Open decisions

- **[Q1]** Counterbalance as described (3/3 per horizon, labels stay a/b)?
  Or additionally randomize labels (a/b vs 1/2) as a second factor?
- **[Q2]** Keep both carry conditions (`carry_silent`, `carry_ref`)?
  Silent tests what the model does by default, ref tests retrieval.
- **[Q3]** Turn-1 answer: rational rule (proposed) vs the model's own E0
  answer vs both? And exclude sub-near-delay H1 from turn 1?
- **[Q4]** Turn-2 wording: bare E0 template (proposed) or with a short
  connective?
- **[Q5]** Thinking on: full 600 generations (~2.5 h) or `conflict` +
  `carry_*` only (~1.3 h)?
- **[Q6]** A deliberately wrong turn-1 answer (consistency pressure) as a
  seventh condition now, or a later round?
