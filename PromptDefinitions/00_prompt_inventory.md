# Prompt inventory: every prompt used so far to locate the time horizon

Compiled 2026-10-12 from `jared/`, `alan/`, `relevance_aware_decoder/` (Augusto),
and the prior project `../../temporal-awareness/`. Read-only survey; nothing was
regenerated except where noted. Paths are relative to the SPAR root
(`/home/Jared/SPAR/`). Where a template is quoted it is verbatim from code.

## 1. Classification axes

Every prompt below can be placed on these axes. They are the candidate
vocabulary for defining new prompt types.

| Axis | Values seen so far |
|---|---|
| **A. Task frame** | A1 intertemporal choice (2 options, reward × delay, horizon given as a constraint); A2 temporal-scope preference (short-term vs long-term *approach*, no numbers); A3 temporal classification ("is this short- or long-term?"); A4 open-ended advice (steering evaluation only) |
| **B. How the horizon reaches the model** | B1 explicit, canonical wording; B2 explicit, paraphrased constraint; B3 explicit, unit-rewritten ("12 months" for "1 year"); B4 implicit life-event cue ("my rent is due next month"); B5 implicit semantics, no duration (CAA implicit, goal classification); B6 absent (null / no-horizon control); B7 carried from an earlier turn (drafted, never run) |
| **C. Other durations in context** | C0 none beyond the option delays; C1 irrelevant non-temporal sentence; C2 irrelevant duration (entity age, prep time, tenure, background event, other decision's horizon) |
| **D. Surface** | D1 bare 5-line starter; D2 SITUATION/TASK/… structured; D3 plain prose; D4 varied prose (8 orders × phrasings); plus domain, label style, option order, constraint position |
| **E. Regime** (A1 only: horizon vs option delays) | `none` (h < near delay: nothing pays in time), `near` (only near pays), `both` (h ≥ far delay), `null` |
| **F. Mode and readout** | thinking off / on (with forced close); prefill `I choose:` / model writes it / forced-choice two trajectories; label token, logit diff, or free-text parse |
| **G. Mechanistic role** | G1 geometry and probe bank; G2 invariance test (wording, units, order, labels); G3 relevance/specificity test (distractors); G4 matched clean/corrupt pairs for patching; G5 steering input or evaluation |

## 2. Summary table

| # | Family | Owner | Frame | Horizon | Distractor | Surface | N | Model | Role |
|---|---|---|---|---|---|---|---|---|---|
| J1 | Starter bank | Jared | A1 | B1 + B6 | C0 | D1 | 96 + 20 | Qwen3-8B (0.8B smoke) | G1 |
| J2 | Full bank | Jared (Alan's generator) | A1 | B1 + B6 | C0 | D2 | 500 | Qwen3-8B | G1, G5 |
| J3 | E1 swap / relabel | Jared | A1 | B1 | C0 | D1 | 2 × 96 | Qwen3-8B | G2 |
| J4 | E2 paraphrases (6) | Jared | A1 | B2 | C0 | D1 | 576 | Qwen3-8B | G2 |
| J5 | E3 unit spellings | Jared | A1 | B3 | C0 | D1 | 108 | Qwen3-8B | G2 |
| J6 | E4 perturbations | Jared | A1 | B1 / B4 | C0 / C1 | D1 | 96/96/96/60 | Qwen3-8B | G2, G3 |
| J7 | E5 attribution pairs | Jared | A1 | B1 | C0 | D1 | 48 pairs | Qwen3-8B | G4 |
| J8 | E8 crossing / answer-slot pairs | Jared | A1 | B1 | C0 | D2 | 176 recs, ≤96 pairs | Qwen3-8B | G4, G5 |
| J9 | Multiturn draft | Jared | A1 | B7 | C0 | D1 | ~600 | — (never run) | G2 |
| A1 | formatted_v1 replication | Alan | A1 | B1 + B6 | C0 | D2 | 2000 | Qwen3-14B | G1 |
| A2 | Extended grid | Alan | A1 | B1 + B6 | C0 | D2 | 2000 | Qwen3-14B | G1 |
| A3 | Constraint phrasings (4) | Alan | A1 | B2 | C0 | D2 | 2000 | Qwen3-14B | G2 |
| A4 | Unit rewrites, paired | Alan | A1 | B3 | C0 | D2 | 2000 (689 pairs) | Qwen3-14B | G2 |
| A5 | Matched matrix | Alan | A1 | B1 | C0 | D2/D3/D4 × 2 domains | 2160 | Qwen3-14B | G2 (transfer) |
| A6 | Controls: mention / role / factual / scaling | Alan | A1 (factual: no task) | B1 | C2 | D3 | 1080 | Qwen3-14B | G3 |
| R1 | Distractor families (5 × 4 templates × 3 slots) | Augusto | A1 | B1 | C2 | D3 | 3360 + 2160 clean | Qwen3-14B | G3 |
| P1 | Intertemporal choice (prior paper) | temporal-awareness | A1 | B1 + B6 | C0 | D2 (older) | up to 27k per domain, 9 domains | Qwen3-4B-Instruct-2507 | G1, G4 |
| P2 | Temporal-scope CAA pairs | temporal-awareness | A2 | B1 (explicit words) / B5 | C0 | (A)/(B) | 50–1000 per file | GPT-2, Qwen3-4B-I, Llama-3.1-8B | G1, G4, G5 |
| P3 | Goal classification | temporal-awareness | A3 | B5 | C0 | one sentence | 200 pairs | Qwen3-4B(-I) | G4 |
| P4 | Steering generation prompts | temporal-awareness | A4 | none | — | free | 7 | Qwen3-4B-I | G5 |

## 3. Shared mechanics

**Chat wrapping, everywhere in the SPAR repo.** One user turn, no system prompt,
`apply_chat_template(..., add_generation_prompt=True, enable_thinking=…)`.
Thinking off on Qwen3 gives the 9-token suffix
`<|im_end|> \n <|im_start|> assistant \n <think> \n\n </think> \n\n`
(T0–T8 in Alan's and Augusto's naming; suffix positions 0–8 in Jared's).
Thinking on gives the 5-token suffix, and the model writes `<think>` itself;
Jared force-closes at 3072 tokens with `\n</think>\n\n` (`jared/spar_horizon/model_io.py:34, 232-271`).

**Prefill.** Jared's starter runs since v2 append `I choose:` after the
suffix (12-token suffix). The full bank E0 has no prefill; the model writes
`I choose:` and the label is located within 48 tokens. E8 teacher-forces
`I choose:` on the full bank. Alan and Augusto never prefill; `find_choice`
(`alan/ptm/chat.py:62-78`) finds the label after the model's own `I choose:`,
including the bold `**a)` case.

**Prior paper.** Qwen3-4B-Instruct-2507 is treated as non-thinking (no think
block). The choice is read by two forced trajectories `"I choose: " + label`
(`temporal-awareness/src/binary_choice/binary_choice_runner.py:40-100`).

**Option grids shared by A1-frame families from Alan's generator** (`alan/ptm/prompts.py:176-180`, copied into `jared/spar_horizon/full_bank.py`):

```
SHORT_DELAYS  = 1 week, 1 month, 3 months, 6 months, 1 year
LONG_DELAYS   = 2, 5, 10, 20, 30, 50 years
SHORT_REWARDS = 1,000 2,500 5,000 10,000 25,000 50,000 100,000
REWARD_MULTIPLIERS = 2, 5, 10, 25, 50      # long reward = short × m
```

---

## 4. Jared (`spar-planning-temporal-manifolds/jared/`)

### J1. Starter bank (`spar_horizon/prompts.py:94-127`; original `SPAR/spar_starter_geometry.py`)

```
You must choose the best investment:
{a}) {first}
{b}) {second}
{constraint}
Answer with {a}) or {b}).
```

- Options: `"{reward:,} dollars in {delay}."`; near option is always first and labelled `a`.
- Constraint: `Select the option with the greatest benefit for this time horizon: {h}.`
- No-horizon control: `Select the option with the greatest benefit.`
- 16 horizons, rendered by `horizon_text`: 30 seconds, 300 seconds, 3600 seconds, 1 days, 7 days, 1 months, 3 months, 6 months, 1 years, 2 years, 5 years, 10 years, 25 years, 50 years, 100 years, 500 years.
- Reward pairs (near, far): (1,000, 50,000), (5,000, 200,000), (20,000, 500,000).
- Delay pairs: (6 months, 10 years), (1 month, 5 years).
- 16 × 3 × 2 = 96 prompts, plus 20 no-horizon controls.

Results:
- v1, Qwen3-8B, thinking off, no prefill: probe R² 0.9995 at L11; PC1 ρ 0.95 at L16.
- v3, thinking on: fresh-probe R² 0.46 / 0.31 / 0.09 at 25 / 50 / 75 % of the think block, 0.71 at its end; 0.96 on the prefill tokens.
- E7: at the suffix the horizon lies on a curve of about 2 dimensions.

Known issues:
- Near option is always `a`, so position bias, label bias and the rational answer coincide.
- Fixed delays make the rational answer a step function of the horizon, so a label probe is effectively a horizon probe.
- Ungrammatical renderings: "1 years", "1 days", "3600 seconds".
- Without the prefill, the 8B writes prose instead of a label 85 % of the time.
- Thinking on: 46-48 of 116 prompts were force-closed, mostly at long horizons.

### J2. Full bank (`spar_horizon/full_bank.py`; Alan's generator, copied 2026-09-25)

```
SITUATION: Plan for the future of the household.
TASK: You, the head of the household, are tasked to choose the best investment:
{label_a} {first option}
{label_b} {second option}
OBJECTIVE: Think deeply about which option is preferable.
CONSTRAINT: You must select the option that provides the greatest benefit for this time horizon: {horizon}.
ACTION: Select one of the two options. Provide reasoning.
FORMAT: I choose: <{label_a} or {label_b}>. My reasoning: <1-3 sentences>
```

- The CONSTRAINT line is dropped for null horizons (1/18 of prompts).
- 48 horizons, log-spaced from 30 s to 500 y and written in natural units: 30 seconds … 108 seconds, 3 minutes … 55 minutes, 84 minutes, 2 hours … 43 hours, 3 days … 10 days, 2 weeks … 8 weeks, 3 months … 23 months, 3 years … 500 years.
- Per prompt, random draws of delays, rewards and option order.
- seed 0, n = 500. Regimes (verified 2026-10-12): none 295, both 93, near 83, null 29.

Results:
- E0: R² 1.00 at the suffix newline (L16).
- The horizon is one curved direction before reasoning and diffuse during it.
- E8 (patching): the horizon reaches the answer slot only from L24. The probe-chosen cells are passengers: all 12 pre-registered ablation and steering verdicts failed.

Known issues:
- 59 % of prompts are in the `none` regime, where the question is ill-posed.
- The unit word explains 0.97 of log horizon. At the answer slot the direction is a staircase by unit.
- Forced versus natural think closes are linearly separable (AUC 1.00).
- "Think deeply" appears even with thinking off.

### J3. E1 swap and relabel (`experiments/exp1_nuisance.py:296-297`)

- Starter prompts with `swap=True` (far option written first as `a`), or with labels `1)` / `2)`.
- Probe transfer: 0.991 (swap) and 0.997 (relabel) on v3.

### J4. E2 paraphrases (`spar_horizon/prompts.py:25-32`)

The constraint sentence of J1 is replaced by one of:

```
starter:    Select the option with the greatest benefit for this time horizon: {h}.
deadline:   Your deadline is {h}. Select the option with the greatest benefit by then.
time_limit: You have a time limit of {h}. Select the option with the greatest benefit within it.
realized:   The benefit must be realized within {h}. Select the better option.
need_by:    You need the money within {h}. Select the option with the greatest benefit.
planning:   Your planning horizon is {h}. Select the option with the greatest benefit over it.
```

6 × 96 = 576 prompts. Mean off-diagonal transfer is 0.993 at L10.

### J5. E3 unit spellings (`prompts.py:72-79`)

Three spellings for each of six durations:
1 hour / 60 minutes / 3600 seconds; 1 day / 24 hours / 1440 minutes; 1 week / 7 days / 168 hours; 1 month / 30 days / 4 weeks; 1 year / 12 months / 365 days; 10 years / 120 months / 1 decade.

108 prompts. Within/between spread ratio 0.134 (pass is < 0.2).

### J6. E4 perturbations (`prompts.py:37-69`, `experiments/exp4_perturb.py`)

**Distractor.** This sentence is inserted after the options:
`The weather has been unusually mild this season, and the local market opens at nine.`

**horizon_first.** The constraint line moves above the options.

**implicit.** The constraint line is replaced by a life-event sentence (10 horizons, 60 prompts):

```
1 week:   I am moving abroad next week and need the money before I go.
1 month:  My rent is due next month and I need the money for it.
3 months: I am getting married in three months and need the money for the wedding.
6 months: My lease ends in six months and I need the money for a deposit then.
1 year:   I start graduate school in a year and need the money for tuition.
2 years:  My car lease ends in two years and I need the money to buy one.
5 years:  My daughter starts college in five years and I need the money for it.
10 years: I plan to buy a house in ten years and need the money for the down payment.
25 years: I retire in twenty-five years and need the money then.
50 years: I want to leave the money to my grandchildren in fifty years.
```

Probe transfer: clean 0.998, distractor 0.996, horizon_first 0.904, implicit 0.880.

Issues with the implicit set:
- Every sentence still states the duration ("next week", "in five years"), so it is implicit in framing only.
- Every sentence adds a "need the money" semantic.

### J7. E5 attribution pairs (`experiments/exp5_attribution.py:291-337`)

- Clean prompt: a horizon in the `near` regime. Corrupt prompt: a horizon in the `both` regime.
- Rewards and delays are the same within a pair, and the horizon number has the same digit count, e.g. "16 months" vs "25 years".
- 48 pairs. Metric: logit(" a") − logit(" b").
- Confound: unit words differ within a pair (months vs years), so the top attribution on the "months" token may be a unit effect.

### J8. E8 steering and patching inputs (`spar_horizon/steering.py:238-280`, `analysis/steering_prepare.py`)

- Uses the 176 near- and both-regime full-bank records.
- With thinking off, each prompt is paired with its swapped twin and the two log-odds are averaged.
- With thinking on, the cached reasoning is replayed up to the label.
- Crossing pairs: the donor horizon is in the other regime and has the same token length.
- Answer-slot pairs: the near option sits under the same label in both prompts.

### J9. Multiturn draft (`history/08_spec_multiturn_draft.md`, never run)

- Turn 1 is a choice whose answer we prefill. Turn 2 is the starter prompt, and turn 2 is where we measure.
- Conditions:
  - single
  - unrelated: turn 1 is `Which is heavier? a) 2 kilograms of iron b) 1 kilogram of feathers. Answer with a) or b).`
  - same (H1 = H2)
  - conflict (H1 ≠ H2, offset by ±2/4/8 grid steps)
  - carry_silent: turn 2 has no horizon
  - carry_ref: turn 2 says `Same time horizon as before.`

---

## 5. Alan (`spar-planning-temporal-manifolds/alan/`, Qwen3-14B, thinking off)

Generators:
- `ptm/prompts.py`, run by `scripts/gen_prompts.py`.
- `ptm/matrix.py`, run by `scripts/gen_matrix.py`.

Data lives in `alan/data/prompts/`.

Domains:
- investment: "Plan for the future of the household." / "the head of the household" / "choose the best investment" / dollars
- climate: "Plan for the climate policy of the city." / "the city administration" / "choose the best emissions program" / "tonnes of CO2 prevented"

### A1. formatted_v1 replication (`investment_n2000_s0`)

- Same template as J2.
- 17-point grid: 1 day, 3 days, 1 week, 2 weeks, 1 month, 2 months, 3 months, 6 months, 1 year, 2, 3, 5, 10, 20, 30, 50 and 100 years, plus null.

Results:
- PC1 vs log horizon |ρ| 0.96 at the `assistant` token, L35-37.
- The centroid path is planar: PC1 is linear and PC2 quadratic, with its vertex at about 0.6 y.
- Short-choice rate is 92-100 % up to 1 year, 43 % at 10 years, 15 % at 50 years.

### A2. Extended grid (`investment_extended_n2000_s0`)

- Adds 1 minute, 10 minutes, 1 hour and 6 hours, and 300, 1000, 3000 and 10000 years.
- Below the shortest option delay, preference falls toward chance (51-73 % short).
- The extremes leave the plane, and the representation saturates beyond about 1000 y.

### A3. Constraint phrasings (`investment_phrasings_n2000_s0`, `prompts.py:169-174`)

```
0: You must select the option that provides the greatest benefit for this time horizon: {horizon}.
1: Your planning horizon is {horizon}. Choose the option that is best over that horizon.
2: Only outcomes within the next {horizon} matter for this decision.
3: Evaluate the two options over a time frame of {horizon} and pick the better one.
```

- Within each phrasing, ordering holds (0.86-0.97).
- Pooled PCA instead clusters by phrasing (silhouette 0.77) at L14. The clusters merge by L29.
- A ridge regression trained on phrasing 0 transfers with R² 0.79-0.90.
- Phrasing 2 is a strict cutoff; the other three are soft.

### A4. Unit rewrites, paired (`investment_units_paired_n2000_s1`, `prompts.py:148-166`)

- Every rewrite is within 10 % of the canonical value, e.g. "1 year" becomes 12 months / 52 weeks / 365 days, and "100 years" becomes 1200 months / 10 decades / 1 century.
- Rewrites move the decoder by less than one grid step.
- Behavior does shift with decades and centuries (Δp_short −0.11 and −0.07). The model treats them as longer, but the linear readout treats them as shorter.

### A5. Matched matrix (`ptm/matrix.py`)

- Size: 30 option configs × 2 domains × 12 horizons × 3 renderings = 2160.
- The 12 horizons include 4 held out: 2 weeks, 3 months, 5 years, 20 years.

The plain rendering:

```
You are {role}. {situation} Two options are available: a) {r1} {unit} in {d1}, or b) {r2} {unit} in {d2}. Your time horizon for this decision is {H}: choose the option that provides the greatest benefit within it. Select one of the two options and give your reasoning. Respond in this format: I choose: <a) or b)>. My reasoning: <1-3 sentences>
```

The varied rendering has 8 variants: 4 sentence orders × 2 phrasings of the options and horizon. The alternative horizon phrasing is `Choose the option with the greatest benefit over a time horizon of {H}.`

Results:
- A decoder trained on the structured rendering only reaches 0.34-0.39 within a factor of 2 on prose.
- A decoder trained on all renderings reaches 0.91-0.99.
- Investment → climate transfer reaches 0.57.
- A regex for the first duration in the prompt scores 1.00 on every suite.

Issue: option order is tied to the config, not randomized.

### A6. Controls (`ptm/matrix.py:200-273`, 1080 prompts)

All controls are built on the plain rendering.
- **mention**: `{Entity} was established {D} ago.` is inserted before the options.
- **role**: `You have {D} to prepare your recommendation.`, with D ∈ {1 h, 2 h, 3 d, 1 w}.
- **factual**: no task, e.g. `The household holds two investments: one pays out 1,000 dollars after 6 months, the other pays out 10,000 dollars after 20 years. Its planning horizon is 1 day.`
- **scaling**: the horizon and both delays are multiplied by 0.1, 1 or 10.

Results:
- The decoder is pulled toward D: residual correlation +0.58 (mention) and +0.74 (role).
- Choices ignore D: coefficient on log D is −0.03 and −0.11, against about −1.9 on log H.
- Under scaling the readout is mostly absolute (slope 0.61-0.81).
- Alan calls distractor pull "the central open problem".

Gap Alan flagged and never filled: content-matched prompts with no stated horizon.

---

## 6. Augusto (`spar-planning-temporal-manifolds/relevance_aware_decoder/`, Qwen3-14B)

### R1. Distractor families (`rad/families.py:42-101`, `rad/build.py`)

Each prompt is Alan's plain rendering with exactly one irrelevant-duration sentence inserted. Every prompt has a `twin_uid` pointing to the clean prompt it came from; deleting the sentence recovers the twin exactly.

There are three insertion slots: after the role sentence, before the options, or after the horizon.

| Family | Seen templates | Unseen templates | D values |
|---|---|---|---|
| entity_age | `{Entity} was established {D} ago.` / `{Entity} was set up {D} ago.` | `{Entity} has existed for {D}.` / `{Entity} first opened {D} ago.` | 12-pt grid |
| prep_time | `You have {D} to prepare your recommendation.` / `Your recommendation is due in {D}.` | `The meeting where you present your choice is in {D}.` / `You were given {D} to study the two options.` | 1 h … 1 month |
| tenure | `You have held this role for {D}.` / `You took on this responsibility {D} ago.` | `Your predecessor held the role for {D}.` / `You last reviewed {role_np}'s plans {D} ago.` | grid ≤ 30 y |
| background_event | `The office building was last renovated {D} ago.` / `The local school finished building its new gym {D} ago.` | `The town library changed its opening hours {D} ago.` / `{Project} was completed {D} ago.` | grid ≤ 30 y |
| other_horizon | `For an unrelated decision last year, {role_np} used a time horizon of {D}.` / `{Neighbour} plans its {thing} over a horizon of {D}.` | `The previous plan, which has now ended, covered a period of {D}.` / `A separate long-term project of {role_np} runs for {D}.` | 12-pt grid |

Size: 3360 distractor prompts + 2160 clean prompts.

Results:
- The model's choices ignore all five families (behavior-pull gate passed).
- Training the decoder with distractors removes the pull on seen templates.
- On held-out families, removal is mixed: prep_time and other_horizon are "role-level only".
- The regex that reads the sentence containing "horizon" still gets 0.94 within 2×.

Issues:
- Distractors appear in plain prose only.
- prep_time uses hours, a unit never used for H, so it adds a constant offset.
- Some sentences read oddly ("A separate long-term project … runs for 1 day").

---

## 7. Prior project (`temporal-awareness/`)

### P1. Intertemporal choice (`src/intertemporal/formatting/configs/default_prompt_format.py`)

This is the ancestor of J2/A1, with a slightly different template:

```
SITUATION: {situation} {extra_situation}
TASK: You, {role}, are tasked to {task}:
{a} {reward} {unit} in {time}
{b} {reward} {unit} in {time}
OBJECTIVE: Think deeply about which option is preferable.
CONSTRAINT: You must select the option that provides the greatest benefit for this time horizon: {horizon}

ACTION: Select one of the two options. Provide reasoning on why this choice was made.

FORMAT: Respond in this format:
I choose: <a) or b)>.
My reasoning: <reasoning in 1-3 sentences>
```

- The horizon is padded to 20 characters and ends in ".".
- Label styles: 9 defined, e.g. `a)`, `x.`, `[i]`, `(AA)`, `{Option A}`.
- Context variations: 7, covering role, reasoning request and extra situation.
- Nine domains: investment, charity, climate, education, energy, health, research, startup, wellbeing, plus housing.

The geometry config (`GEOMETRY_CFG`):
- Horizons from 1 second to 5 centuries (17), plus None.
- About 4590 prompts.
- Both options draw rewards from the same 1k-100k range, so many samples are not a real tradeoff.

Results:
- Patching localized the decision to L21-35 attention and MLP, with L24 attention the top component.
- CAA steering works at L19-22.

### P2. Temporal-scope CAA pairs (`data/raw/temporal_scope*/`)

Schema: `{question, immediate: "(A) …", long_term: "(B) …"}`.

Explicit example:
- question: "When deciding on a solution to this problem, you should focus on:"
- (A) What works right now to fix the immediate issue
- (B) What will create lasting value over the next decade

Implicit example: (A) "Tactical responses that handle the situation at hand" vs (B) "Strategic frameworks that create institutional capacity".

Uses:
- GPT-2 probes: 92.5 % CV accuracy.
- Qwen3-4B-I probe: 0.992 on implicit, 0.884 transfer to explicit.
- Qwen3-4B-I binary-choice steering: `{question}\n(A) …\n(B) …\nAnswer with only (A) or (B):`, which goes from 68.6 % to 97.2 % long-term.
- EAP-IG with system prompt `Recommend an approach for the given scenario. Reply only with (A) or (B)` and prefill `(`.

Issue: in `implicit_expanded_1000`, (A) is the immediate option in all 1000 pairs, despite metadata claiming the order was randomized.

### P3. Goal classification (`data/temporal_classification/`)

Template: `The goal is to <goal>. Is this a short-term or long-term goal? The answer is:`

- 200 pairs with no temporal words, matched in length.
- Patching put the computation in L29-31 of Qwen3-4B.

Confounds listed in the project's own README: ambition and scale.

### P4. Steering generation prompts (`notebooks/08_steering_finalized.ipynb`)

Seven open-ended advice prompts, e.g. "A nonprofit is deciding whether to expand now or build reserves. What would you suggest?"

---

## 8. What the inventory shows

1. **Every prompt in the SPAR repo is one task.** All of them are a single-turn choice between two options, with the horizon given as an instruction. None asks the model to plan, i.e. produce or execute a sequence of steps. The "planning horizon" we have measured is an instructed constraint, not a horizon the model chooses for itself.
2. **The horizon is always written in the prompt** as a duration string, apart from the 60 J6 implicit prompts and the prior project's P2/P3 (which give no number at all). Regex baselines read it at 0.94-1.00, so decodability is guaranteed by the input. E8 found the probe cells are not causally used. The prompts cannot by themselves separate "the model read a duration" from "the model uses a horizon".
3. **The unit word confounds magnitude.** On the full bank, units explain 0.97 of log horizon. Unit rewrites (J5, A4) were tested only on small or paired sets, and decades/centuries shift behavior.
4. **The regime is unbalanced and partly ill-posed.** 59 % of the full bank is `none`, where no option pays in time. Only `near` and `both` have a defined rational answer.
5. **Order and label confounds remain.** The starter bank always puts near at `a`. Alan's matrix ties order to config. P2 implicit-1000 always puts immediate at (A).
6. **Distractor durations pull decoders but not behavior** (A6, R1). This is the cleanest evidence so far that "a duration in context" and "the operative horizon" are different things.
7. **Several cells of the axis grid are untested:**
   - an implicit horizon (B4/B5) inside the A1 frame, with balanced regimes and counterbalanced order;
   - a horizon carried across turns (B7);
   - content-matched prompts with no horizon;
   - any thinking-on run with distractors or paraphrases;
   - any task where the horizon has to be computed (e.g. "until the end of next quarter", or a date relative to a stated "today").
