# Results: two local checks on the saved activations (8B and 14B), 2026-10-21

Analysis only (numpy / sklearn on the E0 activations; no model runs). Outputs in
`results/horizon/analysis/`; scripts `analysis/probe_natural_within_unit.py`
(Check 1) and `analysis/regime_comparison.py` (Check 2). Each reading is stated
against the criteria fixed before the run. Where a criterion left a choice open,
the choice is flagged (§3).

**Short version.**
- **Check 1:** the verdicts depend on a choice the criteria did not fix: is
  within-unit R² scored per prompt or on the 48 horizon averages?
  - Per prompt, the low-k code at the suffix and at pre-label (thinking off)
    is mostly a unit staircase.
  - On the horizon averages, it carries magnitude within a unit at every k
    on 8B, from k = 1.
  - Natural closes lower thinking-on R² by more than 0.1 only at k = 2 (8B).
- **Check 2:** the pre-registered test cannot decide.
  - A post-hoc within-horizon test finds a horizon × far-delay signal at the
    suffix newline from L15/16 (0.42 depth), 9–11 layers before the hand-off (`11` §1: 8B L24, 14B L26 off / L27 on).
  - It follows the rational regime, not the choice.
  - Moving the threshold shows it is graded, not a step at h = far. The one
    exception is 14B thinking-on pre-label, which is a step from L1, probably
    read off the reasoning text.

## 1. Check 1: natural closes and within-unit R²

Setup:
- Probes, dimension ordering and the leave-one-horizon-out protocol are those
  of `06` §2, at the layers `06` §1 uses for 8B.
- The 14B layers come from a linear-only sweep. 14B suffix newline at L20, the
  layer `11` §2 aligns at, is added.
- The ridge's inner CV is grouped by horizon. That fix was already in the
  committed `probes.fit_ridge`, and the 8B all-prompt R² reproduces `06` §2
  to ±0.02, so `06` §5's concern about it changes nothing.
- 471 prompts with a horizon. Natural closes: 371 on 8B (100 forced), 445 on
  14B (26 forced).
- Shuffled-horizon controls are ≤ −0.02 in every cell.

### 1a. Within-unit R²: per prompt vs on horizon averages

"Within-unit R²" removes each unit word's mean from target and prediction
(`06` §5). Scored per prompt, the denominator is the small spread of log
horizon inside a unit. Each prompt's scatter around its horizon's mean
prediction counts as error. Scored on the 48 horizon averages
(`probes.horizon_within_unit_r2`, new), it asks only whether the averages are
ordered and scaled within a unit. That is the object `06` §2's shape claims are
about. All prompts, interior horizons; **per prompt / horizon averages**:

| cell | lin k=1 | lin k=2 | lin k=8 | MLP k=1 | MLP k=8 | lin k=64 | ceiling (per prompt) |
|---|---|---|---|---|---|---|---|
| 8B suffix newline L16 (on) | .24 / **.50** | .19 / **.52** | .24 / .66 | .38 / .75 | .55 / .79 | .70 / .87 | .87 |
| 8B suffix last L10 (on) | −.48 / −.01 | −.68 / .14 | .04 / .41 | −1.00 / .18 | .54 / .80 | .77 / .90 | .84 |
| 8B think close L19 | −.07 / −.06 | −7.3 / −.84 | −3.8 / .09 | −3.2 / −.73 | −3.5 / .24 | – | −.21 |
| 8B pre-label on L21 | −.47 / −.17 | −5.1 / −.37 | −2.3 / .39 | −1.7 / −.47 | −1.0 / .51 | – | .44 |
| 8B pre-label off L21 | .00 / **.75** | .16 / **.78** | .26 / .76 | −.05 / .74 | .48 / .79 | – | .83 |
| 14B suffix newline L20 (on) | −.63 / .11 | −.46 / .11 | .21 / .73 | −.15 / .54 | .54 / .81 | .60 / .81 | .75 |
| 14B suffix newline L7 (on, sweep pick) | −.13 / .08 | −.32 / −.14 | −2.8 / −.91 | −.25 / .10 | −3.3 / −1.2 | .73 / .85 | .81 |
| 14B suffix last L21 (on) | −2.2 / −.25 | −4.3 / −.13 | −.83 / .37 | −2.4 / −.26 | −.33 / .36 | .42 / .75 | .66 |
| 14B think close L27 | −6.5 / −1.8 | −5.3 / −1.0 | −3.8 / .03 | −7.2 / −2.0 | −1.3 / .66 | – | .28 |
| 14B pre-label on L22 | −6.1 / −1.7 | −6.4 / −2.1 | −2.9 / .11 | −6.4 / −1.8 | −1.2 / .51 | – | .36 |
| 14B pre-label off L22 | −.05 / −.05 | −.89 / .01 | −.20 / .66 | −.08 / −.07 | .50 / .83 | – | .75 |

For context, the ordinary R² at linear k = 1 is 0.90 (8B suffix newline), 0.83
(8B pre-label off) and 0.92 (14B suffix L20). Full tables, including k = 3, 4,
natural closes and the horizon-mean ordering, are in
`probe_natural_within_unit.txt` and `probe_within_levels.txt`.

### 1b. Natural closes vs all prompts (thinking-on cells)

Largest changes in interior R² (natural minus all):

| cell | k=2 lin | k=2 MLP | k=3 lin | k=3 MLP | k=8 lin | k=8 MLP |
|---|---|---|---|---|---|---|
| 8B think close L19 | +.01 | **−.17** | +.23 | +.11 | +.03 | −.01 |
| 8B pre-label on L21 | **−.13** | **−.17** | +.03 | .00 | +.01 | .00 |
| 8B suffix cells (\|Δ\|) | ≤ .03 | ≤ .01 | ≤ .02 | ≤ .01 | ≤ .01 | ≤ .01 |
| 14B think close / pre-label on | ≤ .03 | ≤ .02 | ≤ .01 | ≤ .01 | ≤ .03 | ≤ .01 |

### 1c. Verdicts

1. **Suffix within-unit R² ≥ 0.5: kept at the horizon-average level; at the
   prompt level only for the full code.**
   - Per prompt, it passes only with many components: k = 64 (0.70–0.77 on
     8B, 0.60–0.73 on 14B), the ceiling (0.84–0.87 / 0.66–0.81) and the 8B
     MLP at k = 8 (0.54–0.55).
   - On the horizon averages, the 8B suffix newline passes from k = 1
     (linear 0.50, MLP 0.75). The linear pass is borderline: 0.503, 95%
     bootstrap CI over horizons 0.20–0.67, carried by the year horizons
     (audit 2026-10-26). 14B L20 needs k = 8 for the linear probe (0.73);
     the MLP passes at k = 1 (0.54).
   - The last suffix token passes at both levels only from k = 8 (MLP).
2. **Pre-label thinking off, within-unit R² < 0.3 (staircase): met per prompt,
   not met on horizon averages.**
   - 8B, per prompt: linear 0.00–0.26 at k ≤ 8 (MLP 0.34–0.48 at k ≥ 3).
     On averages: 0.74–0.80 at every k.
   - 14B, per prompt: linear ≤ −0.05. On averages: 0.66–0.83 from k = 4, but
     ≈ 0 at k = 1–2.
   - Reference for "unit only" (audit 2026-10-26): a leave-one-horizon-out
     unit lookup scores about −0.25 within unit, per prompt and on averages,
     not 0. Values near 0 are slightly above the unit level; the < 0.3
     staircase threshold was set without this floor.
   - So on 8B the horizon-average path orders horizons within a unit from its
     first component, and `06` §2's curvature claims stand. On 14B the top two
     answer-slot components are a unit staircase.
   - The prompt-level reading would withdraw both. I recommend the
     horizon-average reading because `06` §2's shape claims are about the
     averages (judgment call 1).
3. **Thinking-on cells losing > 0.1 R² on natural closes: only at k = 2, on
   8B.**
   - On 8B, pre-label k = 2 drops 0.13 (linear) and 0.17 (MLP), and think
     close drops 0.17 (MLP, k = 2). At k ≥ 3 nothing drops, and think close
     rises (linear k = 3 +0.23).
   - The suffix cells move by at most 0.05 (8B suffix last, linear k = 4).
   - 14B changes ≤ 0.03, but has only 26 forced closes.
   - So the 8B k = 2 "arc" at the thinking-on answer slot (`06` §4) was partly
     reading "was cut off". The k ≥ 3 and ceiling numbers were not.
   - Natural-only also has 100 fewer prompts on 8B, so part of the k = 2 drop
     may be sample size.
4. **Not in the criteria, but the clearest result:** thinking-on think close
   and pre-label carry little within-unit magnitude at k ≤ 4 at either level.
   - Horizon averages are ≤ 0.29 linear, and only 0.24–0.66 with the MLP at
     k = 8; per prompt they are −1 to −8.
   - There the low-k horizon code is a unit code; magnitude within a unit
     needs the full space (8B ceiling 0.44).

## 2. Check 2: an explicit horizon-vs-delay comparison before the hand-off?

Setup:
- Near- and both-regime prompts: 176 thinking off; natural closes only
  thinking on (137 on 8B, 169 on 14B). The regime is sign(log h − log far);
  ties count as `both`.
- Per leave-one-horizon-out fold, fitted on training prompts only: project out
  the top 8 horizon-average PCs and the far-delay averages. Far delay takes 6
  values (2–50 years), so its 5 PCs remove its group means exactly.
- Then a logistic regime probe.
- Every layer, at the suffix newline and pre-label, 8B and 14B, both modes.
- Hand-off: L22 (8B), L25 (14B).

### 2a. Pre-registered test (pooled AUC, null = regime shuffled within far delay)

The projected pooled AUC (0.55–0.85) is above the null's 95th percentile
(0.58–0.61) at almost every layer **from L1** in all 8 series. The only
exceptions are L3 (8B off suffix) and L3–L7 (8B on suffix).

**This null cannot separate a comparison from leftover horizon information.**
Log horizon alone predicts the regime at AUC 0.93–0.95, 8 PCs do not remove all
of it, and shuffling within far delay breaks the label's link to h. By the
letter of reading 1, "no explicit comparison" is rejected. I do not read the
pooled numbers as a comparison from L1.

### 2b. Post-hoc: AUC inside each held-out horizon

Within one horizon, the horizon cannot rank prompts. The regime is then a
threshold on far delay, and far-delay means are projected out exactly. Only a
horizon × far-delay interaction can score. The null is the same shuffle (200).
5–7 horizons per cell have both regimes. Regime AUC / choice AUC (the same
score against what the model chose):

| series | L14 | L15 | L16 | L18 | L20 | L22 | L24 | L28 | last |
|---|---|---|---|---|---|---|---|---|---|
| 8B off suffix | .76/.53 | .97/.53 | .99/.59 | .93/.46 | .97/.50 | .97/.49 | .96/.57 | .90/.56 | .92/.58 |
| 8B off pre-label | .67/.47 | .86/.48 | .89/.54 | .91/.47 | .97/.53 | .97/.52 | .96/.58 | .99/.52 | .98/.54 |
| 8B on suffix | .49/.60 | .97/.67 | .96/.66 | .93/.62 | .96/.69 | .92/.67 | .96/.67 | .87/.67 | .85/.69 |
| 8B on pre-label | .68/.69 | .76/.70 | .71/.68 | .71/.74 | .77/.77 | .80/.77 | .89/.72 | .88/.77 | .75/.78 |
| 14B off suffix | .74/.69 | .92/.73 | .96/.74 | .99/.76 | .98/.74 | .95/.70 | .97/.64 | .99/.65 | .99/.79 |
| 14B off pre-label | .47/.57 | .62/.67 | .78/.63 | .81/.75 | .96/.68 | .99/.73 | .91/.65 | .96/.65 | .94/.65 |
| 14B on suffix | .71/.57 | .89/.67 | .94/.72 | .97/.68 | .96/.70 | .92/.64 | .95/.66 | .99/.69 | .97/.73 |
| 14B on pre-label | .95/.75 | .98/.76 | .99/.78 | .97/.76 | .95/.69 | .95/.67 | .93/.68 | .95/.75 | .90/.73 |

Null p95 is 0.60–0.65. The first layer at or above 0.9 is L15 (8B off and on
suffix, 14B off suffix), L16 (14B on suffix), L17 (8B off pre-label), L20 (14B
off pre-label) and L23 (8B on pre-label). For 14B on pre-label it is L1. Weak
onset: L9 on 8B, L13 on 14B.

**Threshold shift (control, added after the run).** Each probe is trained and
scored on sign(log h − log far − s), at shifts of −0.5 to +0.5 decade (4–20% of
labels change). A step at h = far, the output of a comparison, peaks at s = 0.
A graded interaction scores the shifted thresholds alike. From L15 on
(`regime_comparison_shift.json`):

| series | layers where s = 0 scores best | median (s=0 − best shifted) | median AUC at s = −.5 / −.25 / 0 / +.25 / +.5 |
|---|---|---|---|
| 8B off suffix | 6 / 22 | −.02 | .88 / .95 / .93 / .87 / .61 |
| 8B off pre-label | 11 / 22 | .00 | .95 / .98 / .98 / .84 / .69 |
| 8B on suffix | 0 / 22 | −.06 | .98 / .95 / .92 / .85 / .50 |
| 8B on pre-label | 8 / 22 | −.04 | .85 / .79 / .79 / .68 / .66 |
| 14B off suffix | 17 / 26 | +.01 | .97 / .89 / .98 / .88 / .84 |
| 14B off pre-label | 2 / 26 | −.04 | .98 / .92 / .94 / .79 / .71 |
| 14B on suffix | 19 / 26 | +.01 | .95 / .93 / .97 / .83 / .80 |
| **14B on pre-label** | **26 / 26** | **+.19** | .68 / .72 / **.95** / .73 / .61 |

**Irrational prompts** (chose far in `near`, or near in `both`): how many the
projected probe calls by the regime, with Wilson 95% CIs.
- 8B off: 29/33 at L22 [.73, .95], at both positions.
- 14B off: 15–18 of 21 at L25–27.
- With thinking on it is weaker: 8B 20–22 of 32; 14B pre-label 16–17 of 31,
  which spans chance.
- The sign call is crude: the same probe calls only 66–80% of rational prompts
  by their regime.
- The within-horizon regime-vs-choice AUC is the cleaner evidence. Regime
  beats choice by 0.2–0.5 from L15 on, except 8B on pre-label (0.80 vs 0.77
  at L22).

### 2c. Verdicts

1. **"Nothing above the null before L22 / L25" (no explicit comparison):
   rejected by the pre-registered metric, but that metric is uninformative.**
   It is above the null from L1, where no comparison is plausible (§2a).
2. **"An earlier signal at the suffix newline": supported, on the post-hoc
   metric.**
   - A horizon × far-delay signal reaches 0.92–0.97 within-horizon AUC at the
     suffix newline at L15 (8B off and on, 14B off) and L16 (14B on). That is
     9–11 layers before `11`'s hand-off markers (8B L24, 14B L26 off / L27 on), at 0.40–0.42 relative depth. (L22 / L25 in criterion 1 are the pre-registered first layers, not those markers.)
   - At the pre-label position, thinking off, it follows at L17 (8B) and L20
     (14B), also before the hand-off.
   - The shift control says it is **graded, not a step at h = far**. In 7 of 8
     series a threshold 0.25–0.5 decade lower scores as well or better. It is
     a horizon-dependent reading of the delay, not a binary "horizon exceeds
     delay" bit.
3. **"Only at pre-label from the hand-off on, following the choice": not
   supported.**
   - The signal starts before the hand-off.
   - On irrational prompts it follows the rational regime, not the choice.
4. **14B thinking-on pre-label is the one step-like comparison.**
   - It peaks at the true threshold at all 26 layers (+0.19 over any shift)
     and is at 0.93 from L1.
   - A binary verdict present from the first layer at the answer slot is most
     likely read from the reasoning text, which states it. That fits the
     working hypothesis: with thinking on, the text carries the result.
   - The probe's sign call follows the regime on only half of the irrational
     prompts (16–17 of 31). Whether this verdict is what the model then
     overrides is open.

## 3. Judgment calls

1. **Prompt vs horizon-average within-unit R².** The criterion did not say
   which. I report both and recommend the horizon-average one for shape
   claims about the averages. The prompt-level one is right for "can a probe
   read one prompt's horizon within its unit". The horizon-average version
   was computed without the full-dim ceiling, which is too slow at d = 5120.
2. **14B layers.** A linear-only sweep, at k = 2, 8, 64, on all prompts, with
   no MLP and no ceiling (a full sweep would take about 4.5 h).
   - The suffix newline sweep is a plateau (0.98–0.99 from L4 to L40), so its
     pick, L7, is noise-level. L7 is also a layer where k ≤ 4 reads nothing,
     so I added L20.
   - Natural-only runs reuse the all-prompt layer.
3. **The thinking-off pre-label cell has no natural-only column** (no forced
   closes there). The suffix cells use the thinking-on run.
4. **Check 2 probe.** Logistic, C = 100, balanced classes. C was tuned on a
   pilot that saw labels (AUC flat for C = 100–1000). Each fold's projected
   features are rescaled to unit mean squared norm.
   - X is rotated into the span of the centred data (n − 1 coordinates). This
     is lossless and uses no labels, not a PC cut.
5. **Shuffles.** Pooled: 200, extended to 1,000 when the AUC beats the null's
   90th percentile (285 of 312 cells). Within-horizon: 200.
6. **The within-horizon metric, the regime-vs-choice AUC, the far-threshold
   control and the shift control were all added after seeing the
   pre-registered test fail to decide.** Treat them as exploratory.
7. **The within-horizon AUC is noisy.** Only 5–7 horizons per cell have both
   regimes. Flipping the 3 tie prompts (h = far) to `near` moves 8B
   thinking-on pre-label L28 from 0.88 to 0.75. Read thinking-on values to
   about ±0.1.
8. **Not ruled out:** the near delay is not projected out. It is uncorrelated
   with the far delay (r = −0.02), and on its own it orders the regimes
   within a horizon at AUC 0.41, so it cannot produce 0.9+. A planted single
   comparison feature is partly removed by the projection (AUC about 0.75 at
   any amplitude), so the test under-detects a true comparison.

## 4. Against `06` and `11`

- **`06` §2, "the horizon is essentially one curved direction" at the
  suffix.** On 8B this holds on the horizon averages: that one direction
  orders horizons within a unit (0.50 linear, 0.75 MLP). Per prompt it is
  mostly the unit word (0.24).
  - On 14B it holds only at L20, not at the sweep's best layer L7, where
    k ≤ 4 reads nothing (R² ≤ 0.22) and the plateau is flat. At L20 the linear
    k = 1 direction is unit-only (0.11) and the MLP recovers magnitude (0.54).
- **`06` §2, pre-label thinking off "largely a staircase by unit"** (cubic
  plus unit-word F-test). On the horizon averages, within-unit ordering is
  0.75–0.80 on 8B, so there is a staircase and a substantial slope within each
  step. On 14B the first two components are the staircase alone.
- **`06` §4, thinking-on curvature.** The k = 2 arc at the 8B answer slot
  loses 0.13–0.17 R² on natural closes, so part of it was the forced-close
  signal. It also has no within-unit magnitude at either level.
- **`06` §5 is out of date.** The grouped inner CV and within-unit R² already
  exist in `probes.py`, and grouping changed R² by ≤ 0.02.
- **`11` §5, "the answer slot holds a decision attached to the horizon".** The
  non-horizon, non-delay variable readable at the answer slot follows the
  rational regime, not the model's choice, on irrational prompts. It is
  already present at the suffix newline 9–11 layers before the hand-off.
  - `11` §3's decision direction was fitted on the choice, a different
    variable, so the two are not in conflict. But "decision" should not be
    used for the regime signal.
  - The integration of horizon and delay starts in the text after the
    horizon (0.42 depth), around T1's `spread_peak` (8B 0.53, 14B 0.58), not
    at the hand-off.
- **No conflict** with `11` §1 (stage order) or the `11` §2 suffix alignment (that alignment was struck on 2026-10-26: it does not beat a null that keeps the shared prompts).
  The 14B suffix ceilings here (0.99) agree with its 0.94 readout.

## 5. Files

- New analysis code:
  - `analysis/probe_natural_within_unit.py`, with `--levels`;
  - `analysis/regime_comparison.py`, with `--within`, `--far-control` and
    `--shift-control`.
- Edited:
  - `spar_horizon/probes.py`: `horizon_within_unit_r2` and the
    `r2_within_horizon` keys;
  - `analysis/probe_dimensionality.py`: `run_file` and `--model`;
  - `analysis/horizon_shape.py`: `--model`. Its layers are still 8B's and its
    output name has no model tag, so a 14B run would overwrite
    `horizon_shape.png`.
- Tests: `tests/test_probe_natural_within_unit.py` and
  `tests/test_regime_comparison.py` are new; one test was added to
  `tests/test_probes.py`. Fast suite: 154 passed.
- Outputs (`results/horizon/analysis/`):
  - `probe_natural_within_unit.{json,txt,png}`,
    `probe_natural_within_unit_14b_suffix_L20.json` and
    `probe_within_levels.{json,txt}`;
  - `regime_comparison.{json,png}`, `regime_comparison_within.{json,png}`,
    `regime_comparison_far_control.json` and `regime_comparison_shift.json`.
- Not fixed: `runs.load_activations` loads every layer eagerly, about 7 GB per
  14B file, which is unsafe with 14 GB of RAM. The new scripts load one layer
  at a time.
