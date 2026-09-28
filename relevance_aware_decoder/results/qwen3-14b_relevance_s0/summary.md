# Relevance-aware decoder: results

Run `runs/qwen3-14b_relevance_s0` · 5520 prompts · generated 2026-09-28 20:09 · thresholds: behavior gate ±0.1, removed if CI upper < 0.1, min baseline pull 0.2, max clean drop 0.05

## Behavior gate

| family | n | behavior_pull | behavior_pull_lo | behavior_pull_hi | choice_flip_rate | irrelevant |
|---|---|---|---|---|---|---|
| entity_age | 672 | 0.061 | 0.028 | 0.097 | 0.079 | True |
| prep_time | 672 | -0.007 | -0.031 | 0.016 | 0.071 | True |
| tenure | 672 | -0.017 | -0.042 | 0.005 | 0.071 | True |
| background_event | 672 | 0.020 | -0.015 | 0.053 | 0.092 | True |
| other_horizon | 672 | -0.007 | -0.058 | 0.044 | 0.106 | True |

## Ridge alpha (grid 0.01 to 1e+06; cells whose dev-selected alpha is at an edge)

| decoder | n_cells | at_min | at_max |
|---|---|---|---|
| baseline | 70 | 2 | 0 |
| lofo_background_event | 70 | 0 | 0 |
| lofo_entity_age | 70 | 0 | 0 |
| lofo_other_horizon | 70 | 0 | 0 |
| lofo_prep_time | 70 | 1 | 0 |
| lofo_tenure | 70 | 1 | 0 |
| rad_all | 70 | 0 | 0 |

## Cell L26:T1

Clean test within-2x: baseline 0.891, rad_all 0.863 (drop +0.028; OK)

| family | pre (baseline) | A | B | C | verdict |
|---|---|---|---|---|---|
| entity_age | +0.10 [+0.08, +0.12] | -0.01 [-0.02, +0.00] | -0.00 [-0.02, +0.02] | +0.01 [-0.01, +0.02] | no pull to remove (baseline pull below threshold) |
| prep_time | +0.25 [+0.22, +0.27] | +0.01 [-0.00, +0.03] | +0.04 [+0.02, +0.05] | +0.18 [+0.16, +0.19] | ROLE-LEVEL ONLY: removed for trained roles (new wording too), not for a new role |
| tenure | +0.10 [+0.07, +0.12] | -0.01 [-0.02, +0.01] | +0.00 [-0.01, +0.01] | -0.01 [-0.03, -0.00] | no pull to remove (baseline pull below threshold) |
| background_event | +0.12 [+0.10, +0.14] | -0.00 [-0.01, +0.01] | -0.01 [-0.02, +0.01] | -0.01 [-0.02, +0.00] | no pull to remove (baseline pull below threshold) |
| other_horizon | +0.12 [+0.09, +0.13] | -0.00 [-0.01, +0.01] | +0.01 [-0.00, +0.03] | +0.04 [+0.03, +0.05] | no pull to remove (baseline pull below threshold) |

## Cell L18:T1

Clean test within-2x: baseline 0.845, rad_all 0.836 (drop +0.009; OK)

| family | pre (baseline) | A | B | C | verdict |
|---|---|---|---|---|---|
| entity_age | +0.20 [+0.18, +0.22] | +0.00 [-0.01, +0.01] | +0.02 [+0.01, +0.03] | +0.03 [+0.02, +0.04] | no pull to remove (baseline pull below threshold) |
| prep_time | +0.19 [+0.17, +0.21] | +0.00 [-0.01, +0.01] | -0.01 [-0.05, +0.03] | +0.04 [+0.01, +0.07] | no pull to remove (baseline pull below threshold) |
| tenure | +0.26 [+0.23, +0.28] | +0.00 [-0.00, +0.01] | +0.01 [-0.00, +0.02] | +0.00 [-0.01, +0.02] | SEPARABLE: pull removed for a role never seen in training |
| background_event | +0.20 [+0.16, +0.24] | -0.00 [-0.02, +0.01] | +0.02 [-0.00, +0.05] | +0.03 [+0.01, +0.04] | SEPARABLE: pull removed for a role never seen in training |
| other_horizon | +0.24 [+0.21, +0.27] | +0.00 [-0.01, +0.02] | +0.03 [+0.01, +0.04] | +0.15 [+0.13, +0.17] | ROLE-LEVEL ONLY: removed for trained roles (new wording too), not for a new role |

## Log

```
run runs/qwen3-14b_relevance_s0: 5520 rows ({'distractor': 3360, 'main': 2160}); prompts missing from run: 0
activations: full (87 shards)
format adherence by condition: {'distractor': 1.0, 'main': 1.0}
decoder baseline                 train rows   960  dev rows  192
decoder rad_all                  train rows  2560  dev rows  512
decoder lofo_entity_age          train rows  2240  dev rows  448
decoder lofo_prep_time           train rows  2240  dev rows  448
decoder lofo_tenure              train rows  2240  dev rows  448
decoder lofo_background_event    train rows  2240  dev rows  448
decoder lofo_other_horizon       train rows  2240  dev rows  448

behavior: beta_H = -4.975 short log-odds per decade of horizon (clean plain prompts)
  entity_age         n= 672  behavior pull +0.061 [+0.028, +0.097]  choice flips 0.079  -> irrelevant
  prep_time          n= 672  behavior pull -0.007 [-0.031, +0.016]  choice flips 0.071  -> irrelevant
  tenure             n= 672  behavior pull -0.017 [-0.042, +0.005]  choice flips 0.071  -> irrelevant
  background_event   n= 672  behavior pull +0.020 [-0.015, +0.053]  choice flips 0.092  -> irrelevant
  other_horizon      n= 672  behavior pull -0.007 [-0.058, +0.044]  choice flips 0.106  -> irrelevant
text baselines done (2s)
layer 14 done (56s)
layer 18 done (112s)
layer 22 done (166s)
layer 26 done (219s)
layer 29 done (271s)
layer 33 done (326s)
layer 37 done (382s)
ridge alpha grid [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]: cells choosing an edge value per decoder
  baseline                 at min   2/70  at max   0/70
  lofo_background_event    at min   0/70  at max   0/70
  lofo_entity_age          at min   0/70  at max   0/70
  lofo_other_horizon       at min   0/70  at max   0/70
  lofo_prep_time           at min   1/70  at max   0/70
  lofo_tenure              at min   1/70  at max   0/70
  rad_all                  at min   0/70  at max   0/70
```