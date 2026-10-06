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

| Family | Variants | Reference |
|---|---|---|
| `nulls` | no final period, largest/greatest, gives/provides, carefully/deeply, "Thank you.", "Please" | canonical |
| `reword` | planning horizon / time frame / within the next / only outcomes within (flagged `meaning_shift`) | canonical |
| `unit` | days, weeks, months, years, decades, words ("six months"), named ("half a year", "a decade") | canonical |
| `option_unit` | options in months; options + horizon in months (Alan's unit-match effect) | canonical / opt_months |
| `labels` | A)/B), 1)/2) | canonical |
| `structure` | constraint line before the options; plain prose layout | canonical |
| `cue` | "leaning towards sooner/later" (agreement), "I'm a bit hurried" / "Take your time" (pressure) | canonical; pairs also contrasted |
| `no_horizon` | no constraint line, with and without each cue | no cue |
| `implicit` / `twin` | 16 implicit horizons ("before our baby is born; my partner is 8 weeks pregnant") vs the same frame with the duration written out; `with_number` marks items whose text holds a duration that is not the horizon | twin |

Option order (short first vs second) is crossed in every cell and reported as its own contrast.

### Multi-turn conditions (1,212 conversations; 12 scenarios, horizons 1 month – 20 years)

| Condition | n | Compared with |
|---|---|---|
| `base` canonical horizon sentence (greedy) | 60 | — (adherence: plan end vs target) |
| `unit` days / months / named in the first turn | 156 | base, same scenario + horizon |
| `implicit` / `twin` (5 personal scenarios x 12 items) | 60 + 60 | twin |
| cue in the **first** turn ("I'm a bit hurried." / "Take your time.") | 120 + 192 free | same conversation without cue |
| cue with the "Continue" before **step 3** (+ "Thanks." as null) | 180 + 288 free | **branched** from the no-cue parent: turns 1–3 copied, so the comparison is exact |
| `free`: no horizon at all, the model picks (sampled, 8 per scenario) | 96 | — |

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

**Expected on Qwen3-14B** (48 GB card; the multi-turn run did 3,360 turns in 19 min on an RTX PRO 5000): multi-turn
≈ 40 min, choice ≈ 10 min, smoke + tests ≈ 5 min, model download ≈ 5–10 min: **≈ 1.5 h of rental, ≈ $1.5**.
Disk ≥ 80 GB (weights 30 GB + activations ≈ 17 GB + smoke).

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
