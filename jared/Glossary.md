# Glossary: short names used in the steering and evaluation pipeline

Sources: `evaluation_design/README.md` (§2 terms, §5 principles, §6 nulls, §8 confounds), `evaluation/analyze.py`
(readout arithmetic), `steering/README.md` (strength, sites), `prompt_bank/README.md` (bank, regimes).

## The prompt

Every T0 prompt offers two investments: a **near** option (sooner, smaller payout) and a **far** option (later,
larger payout). A CONSTRAINT line states a time horizon, e.g. "…for this time horizon: 8 years."

| Symbol | Meaning |
|---|---|
| **H** | The stated **horizon**: how long the investor has. |
| **D** | The **far option's delay**: when the larger payout arrives. |
| **H/D** | Horizon divided by far delay. H/D > 1: the far option pays off in time (`both`, far is correct). H/D < 1: it doesn't (`near_only` or `none`, far is wrong). Plotted as **log10(H/D)**, in decades; 0 means H = D. |
| **H\*** | The **switch point**: the horizon at which the model is 50/50 between near and far. It is the `loc` of a logistic P(far) = sigmoid(s · (log10 H/D − loc)) fitted over the verification grid. A model that answers by the rule has H\* = D, so loc = 0. Qwen3-8B unsteered: +0.38, meaning it needs a horizon about 2.4× the far delay before it picks far. |
| **Δlog10 H\*** | How far steering moves the switch point, in decades: (loc(−x) − loc(+x)) / 2. Positive means the model acts as if the horizon were longer. Overnight `choice`: +2.12, i.e. as if the horizon were ~130× longer. |
| **s** | Slope of that logistic: how sharply the choice flips at H\*. |
| **r** | In the minimal-pair bank `t0_mp0`: the two partners have horizons D/r and D·r, with r in [1.3, 1.6], so one is `near_only` and the other `both`, and nothing else differs. |

**Regimes** (which options pay off within H):

| Regime | Pays off within H | Correct answer |
|---|---|---|
| `none` | neither (H shorter than the near delay) | none; used for fitting only, never scored |
| `near_only` | only the near option | near |
| `both` | both | far |
| `no_horizon` (`null`) | no CONSTRAINT line | none; a control |

**CAA classes:** "long" = H/D ≥ 4, "short" = H/D ≤ 1/4.

## Steering and readout

| Symbol | Meaning |
|---|---|
| **L** (L24) | Layer: the residual stream after transformer block L. |
| **x** | Steering strength. An edit adds x × typical_norm × (unit direction), where typical_norm is the mean activation norm at that layer and site. So x = 0.4 means a push of 40% of a typical activation's size. |
| **±x** | Every edit is run at +x and −x (P2). |
| **y** | The readout: logit(far) − logit(near) at the answer-letter position. |
| **odd** | (y(+x) − y(−x)) / 2: the steering part. |
| **even** | (y(+x) + y(−x)) / 2 − y(0): the disruption part (both signs do the same thing, so it isn't steering). |
| **meaning / letter** | The odd part split over the two option orders of each verification item. Meaning is the push toward far whatever its letter; letter is the push toward the first-listed letter (C1). |
| **K** | Number of random draws per null family. With K = 19 the smallest possible p is 1/(K+1) = 0.05. |
| **p95** | The 95th percentile of the null family's statistic. |
| **α_k** | The strength that should shift the horizon by a factor k (targeting methods). |
| **S/R** | Transfer efficiency: S = shift from steering at α_k, R = shift from rewriting the prompt with a k-times longer horizon. |

Sites (token positions where a direction is read or applied): `label` = the answer letter itself; `prelabel` = the
token before it, whose state predicts the letter; `prompt_end` = the last prompt token; `think_end` = the end of the
reasoning.

## Numbered codes

| Code | Family | Members |
|---|---|---|
| **T0–T3** | Prompt tiers | T0 = templated two-option prompts (`prompt_bank/`). T1–T3 = realistic tiers, not built yet. |
| **t0_s0, t0_mp0** | Bank folders | Tier T0, seed 0; tier T0, minimal pairs, seed 0. |
| **N0–N6** | Null baselines | N0 zero edit (numerical noise), N1 isotropic random, N2 covariance-matched random, N3 in-subspace random, N4 label-shuffled refit, N5 concept controls (reward size, chosen option, answer letter), N6 damage-matched random. |
| **C1–C8** | Confounds | C1 answer letter / position, C2 decision vs horizon, C3 reward sensitivity, C4 disruption, C5 surface form (phrasing, label style, rewrites), C6 rescale vs replace, C7 stated horizon in the reasoning, C8 calibration. |
| **P1–P7** | Test principles | P1 signed effects in horizon units (Δlog H\*), P2 push both ways, P3 paired identical runs, P4 freeze then test, P5 hold out horizons, P6 scale every effect, P7 health is an outcome. |
| **I1–I8** | What a method owner hands over | I1 the edit, I2 conventions, I3 fitting data, I4 refit code, I5 activations, I6 internal structure, I7 training objective, I8 search history. |
| **A–F** | Sections of `diagnostics.html` | A data/labels, B baseline behaviour, C directions, D strength, E outcomes vs nulls, F readout noise. |
| **E*n*** | Jared's experiments | `jared/pipeline/.../exp<n>_*.py`, listed in `jared/01_plan.md`. |
| **S*n*** | Jared's pipeline stages | e.g. S1 = `pipeline/s1_data_vectors`. |
| **R*n*, spec A.*n*** | Citations | `evaluation_design/README.md` §12 references; appendix A of the project spec. |
