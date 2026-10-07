# Phrasing study: which ways of phrasing a prompt change temporal-horizon behaviour?

**Question (group decision 2026-10-06).** Phrasing is not a central pillar of the project, but every experiment
has to account for it. This study answers: *what (if any) ways to phrase a prompt are relevant when studying
temporal horizons, because they change the behaviour?* **Deliverable:** a ranked list of phrasing factors with
effect sizes, so the main experiments know which factors to control or counterbalance.

All prompt text is in [`variants.yaml`](variants.yaml): edit there, then `python scripts/check_variants.py
[--show <variant>]` validates it and prints counts and an example prompt. Code: package `phr/`. Nothing here is a steering/causal experiment: steering is the
follow-up, once this run gives directions to steer along.

## Design

Every item = **content** x **intervention** x **protocol**.

- **Content** (held fixed within a comparison): scenario + true horizon (+ options).
- **Intervention**: how the content is said (a rewording, a unit, an implicit phrasing) or an added sentence
  that leaves the stated horizon unchanged (a cue). Each variant has a **reference** (default: the canonical
  rendering) on the *same* content cell, so every effect is a paired difference.
- **Protocol**: how the model is run and what is measured.

| Protocol | What the model does | Behaviour measure |
|---|---|---|
| `choice` | Alan's formatted two-option prompt, single turn | P(short): two-way softmax of the two label logits after a prefilled `I choose:` + the model's own separator (detected per run, e.g. ` **`) |
| `state` | same prompt, asked to name the horizon | stated duration (parsed) vs the true one |
| `multiturn` | the "Continue" planning protocol of `multiturn_planning/` (outline, 5 steps, "Plan Completed") | the plan's step horizons (plan end = largest step horizon) |

**Common currency: effective horizon** (log10 years). Choice: shift of the P(short)-vs-log-horizon curve, from a
fixed-effects fit of the model's log-odds on log10 horizon (shift = how much the variant moves the curve along the
horizon axis; multiplier 10^shift, < 1 = acts like a shorter horizon). Multi-turn: difference in log10 plan end.
Same unit in both protocols.

**Noise floor.** `nulls` variants should change nothing (punctuation, a synonym, "Please", "Thank you."). A variant
**matters** if |mean Δ P(short)| is above the largest null |mean Δ|, its 95% CI (bootstrap over the 16 scenarios =
8 option configs x 2 domains) excludes 0, and |Δ| ≥ 0.05.

### Families (choice protocol; cells = 8 configs x 2 domains x 10 horizons x 2 option orders = 320)

Every cell is run with **both option orders** (short option first and second; labels follow position), so every
comparison is order-balanced, and the order effect itself is reported as a contrast.

| Family | Variants | Reference |
|---|---|---|
| `nulls` | no final period, largest/greatest, gives/provides, carefully/deeply, "Thank you.", "Please" | canonical |
| `reword` | planning horizon / time frame / within the next / only outcomes within (flagged `meaning_shift`) | canonical |
| `unit` | days, weeks, months, years, decades, words ("six months"), named ("half a year", "a decade") | canonical |
| `option_unit` | options in months; options + horizon in months (Alan's unit-match effect) | canonical / opt_months |
| `labels` | A)/B), 1)/2) | canonical |
| `layout` | the whole prompt structure: Alan's `formatted` lines (canonical), `constraint_first`, `markdown`, `prose`, `task_brief` (a natural task assignment), `user_request` (a person asking for help, first person) | canonical |
| `layout_cross` | days / months / named units and one cue per direction, repeated in every non-canonical layout: does an effect found in Alan's template hold in other structures? | that layout's canonical |
| `cue` | 3 phrasings each of **hurry** ("I'm a bit hurried.", "I need this done quickly.", "We're in a bit of a rush.") and **relax** ("Take your time.", "There's no rush.", "We're not in a hurry at all."), agreement **sooner/later**, 2 **neutral** ("Thanks.", "Okay.") | canonical; directions also contrasted (all phrasings pooled) |
| `no_horizon` | no constraint, with and without one cue per direction | no cue |
| `implicit` / `twin` | 18 implicit horizons vs the same frame with the duration written out, in the `formatted` and `user_request` layouts. `determinacy`: **concrete** (facts + common knowledge fix the horizon, e.g. "before our baby is born; my partner is 8 weeks pregnant") vs **vague** (only a rough range, e.g. "before the end of this week" = Friday or Sunday; the twin is a central guess). `with_number`: the text holds a number that is not the horizon | twin |

### Multi-turn conditions (2,436 conversations, 13,308 generated turns; 12 scenarios, horizons 1 month – 20 years)

| Condition | n | Compared with |
|---|---|---|
| `base` canonical horizon sentence (greedy) | 60 | — (adherence: plan end vs target) |
| `unit` days / months / named in the first turn | 156 | base, same scenario + horizon |
| `implicit` / `twin` (5 personal scenarios x 12 items: 6 concrete, 6 vague) | 60 + 60 | twin |
| cue in the **first** turn: 3 hurry + 3 relax phrasings + 1 neutral | 420 base + 336 free (4 samples/scenario) | same conversation without cue |
| cue with the "Continue" before **step 3**: 3 hurry + 3 relax + 2 neutral | 480 base + 768 free | **branched** from the no-cue parent: turns 1–3 copied, so the comparison is exact |
| `free`: no horizon at all, the model picks (sampled, 8 per scenario) | 96 | — |

Cue effects are reported per phrasing and pooled per direction (hurry / relax / neutral), so a direction "works" only
if its phrasings agree.

Activations (residual stream, 9 layers at 0.2–0.9 depth + last) are stored for every choice item (transition
window T0–T8 + the readout position C) and every generated multi-turn turn (the 13 positions of
`multiturn_planning`), for analyses to be chosen with the team (PCA, probes, ...). Behaviour analysis does not
need them.

## Validation so far (local RTX 5060 Ti 8 GB, Qwen3-1.7B)

- 28 unit tests: items (pairing, references, only-the-variant-changes, no duration in no-number implicit items),
  multi-turn branching (identical turns before the cue), label tokenisation, and **right-padded batches = one
  prompt at a time** (fp32 on CPU: max |Δ log-odds| 1e-5; bf16 on GPU differs by up to ~0.35 log-odds from
  rounding alone, the same for every variant).
- Smoke slice end to end: readout separator detected as ` **`; readout mass on the two labels 1.00; free
  generations follow the format 100% and agree with the readout 96%; state answers parse 100%; step horizons
  parse 100%.

- Full-size dry run (`run_pipeline.sh`, Qwen3-1.7B, all 9,024 + 162 items and 1,212 conversations): 38 min on the
  local GPU (choice 2 min, state 3 s, multi-turn 36 min for 7,080 generated turns), evaluate 36 s, all gates OK,
  6.3 GB. It caught one bug (a family named `null` is read by YAML as None and vanished from the analysis; renamed
  `nulls`, and `check_variants.py` now rejects non-string names). Effective-horizon shifts are flagged `reliable`
  only where the reference curve falls (slope ≤ -0.5 log-odds per decade) on ≥ 4 levels.

- 2026-10-07 revision (layouts, 3 phrasings per cue direction, concrete/vague implicit items): smoke slice on
  Qwen3-1.7B end to end, all gates OK; 31 tests. Not re-run at full size.

**Expected on Qwen3-14B** (48 GB card; the multi-turn run did 3,360 turns in 19 min on an RTX PRO 5000):
choice 24,864 items ≈ 15–25 min, multi-turn 13,308 turns ≈ 75 min, smoke + tests + download ≈ 15 min:
**≈ 2 h of rental, ≈ $2**. Disk ≥ 100 GB (weights 30 GB + activations ≈ 23 GB choice + 16 GB multi-turn + smoke).
To shorten: fewer `later_cues` / `first_turn_cues` in `variants.yaml` (each cue costs ~160 conversations).

## Run it (vast.ai, one 48 GB GPU)

Single commands, run on the box from `/workspace` (Python at `/venv/main/bin/python` on the PyTorch image):

```
git clone -b phrasing-study https://github.com/unrulyabstractions/spar-planning-temporal-manifolds.git && cd spar-planning-temporal-manifolds
/venv/main/bin/pip install -r phrasing_study/requirements.txt && apt-get install -y rclone
nohup env PY=/venv/main/bin/python GCS_DEST=<bucket>/augusto/phrasing GCS_KEY=/root/gcs-key.json bash phrasing_study/scripts/run_pipeline.sh > /workspace/run.out 2>&1 &
tail -f phrasing_study/runs/pipeline.log
```

Without `GCS_DEST`/`GCS_KEY` the upload step is skipped and the run stays on the box. The pipeline stops after the
smoke slice if a gate fails (`scripts/gates.py`). Afterwards: `rm /root/gcs-key.json`, destroy the instance.
Fetch the small files locally (no activations) with
`rclone copy :gcs:<bucket>/augusto/phrasing/<run> runs/<run> --exclude "*_acts_*" --gcs-service-account-file <key> --gcs-bucket-policy-only`.

## Layout

```
variants.yaml           all prompt text: horizon renderings, families, implicit items, cues, multi-turn settings
phr/durations.py        duration text -> years
phr/items.py            choice + state items (content x variant), config validation
phr/model.py            loading, capture hooks, right-padded choice forward, batched generation
phr/single.py           choice (readout + gencheck) and state runners
phr/multiturn.py        multi-turn specs (cues, branching) and runner (imports ../multiturn_planning/mtp)
phr/analysis.py         paired deltas, null floor, effective-horizon shifts, contrasts, multi-turn comparisons
scripts/capture.py      build items + run protocols        scripts/evaluate.py   results/<run>/summary.md + CSVs
scripts/gates.py        sanity gates                       scripts/run_pipeline.sh  the whole run
scripts/check_variants.py  validate variants.yaml after editing (no model)
runs/<run>/             items_*.parquet, choice/state/multiturn parquet, gencheck, *_acts_*.safetensors, meta
```
