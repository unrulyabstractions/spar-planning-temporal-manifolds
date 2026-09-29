# Prompts in `spar_starter_geometry.py`

Every prompt is built by one template, `choice_prompt()`, and sent as a single **user** turn through the Qwen chat template with thinking disabled. The model (`Qwen/Qwen3.5-0.8B`) answers greedily with at most 6 new tokens.

## The template

```
You must choose the best investment:
{label_a}) {first option}
{label_b}) {second option}
{constraint}
Answer with {label_a}) or {label_b}).
```

Where:

- **Options**: a *near* option `{small reward} dollars in {short delay}.` and a *far* option `{large reward} dollars in {long delay}.`
- **Constraint** (with a horizon): `Select the option with the greatest benefit for this time horizon: {horizon}.`
- **Constraint** (no horizon, control): `Select the option with the greatest benefit.`
- **Labels**: `a`/`b` normally, `1`/`2` for the label-stability check.
- **Order**: near first normally, far first for the order-stability check.

## The swept variables

| Variable | Values |
|---|---|
| Reward pairs (near, far) | $1,000 vs $50,000; $5,000 vs $200,000; $20,000 vs $500,000 |
| Delay pairs (near, far) | 6 months vs 10 years; 1 month vs 5 years |
| Horizons (16, log-spaced) | 30 seconds, 300 seconds, 3600 seconds, 1 days, 7 days, 1 months, 3 months, 6 months, 1 years, 2 years, 5 years, 10 years, 25 years, 50 years, 100 years, 500 years |
| No-horizon controls | 20 random (reward, delay) draws, seed 0 |

Total: 16 horizons x 3 reward pairs x 2 delay pairs = **96** horizon prompts + **20** controls = **116** prompts for the geometry pass.

## Three passes over the bank

1. **Geometry** (the 116 prompts below, as-is): generate the answer, then record hidden states at the 9 turn-transition tokens and the 6 response tokens. PCA per (layer, token), Spearman |rho| of PC1 vs log horizon.
2. **Order stability**: every prompt re-asked with the two options swapped (`a)` becomes the far option).
3. **Label stability**: every prompt re-asked with labels `1)` / `2)` instead of `a)` / `b)`.

The temporal-reasoning score needs no new prompts. It looks at the geometry answers on prompts where `near delay <= horizon < far delay`, so only the near option can pay out in time, and counts how often the model picked it.

## One example of each variant

**Base (horizon, a/b, near first):**
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 30 seconds.
Answer with a) or b).
```

**Swapped order:**
```
You must choose the best investment:
a) 50,000 dollars in 10 years.
b) 1,000 dollars in 6 months.
Select the option with the greatest benefit for this time horizon: 30 seconds.
Answer with a) or b).
```

**Relabeled 1/2:**
```
You must choose the best investment:
1) 1,000 dollars in 6 months.
2) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 30 seconds.
Answer with 1) or 2).
```

**No-horizon control:**
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

## All 116 geometry prompts

Grouped by horizon. Each horizon has 6 prompts (3 reward pairs x 2 delay pairs).

### Horizon: 30 seconds

**#1** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 30 seconds.
Answer with a) or b).
```

**#2** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 30 seconds.
Answer with a) or b).
```

**#3** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 30 seconds.
Answer with a) or b).
```

**#4** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 30 seconds.
Answer with a) or b).
```

**#5** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 30 seconds.
Answer with a) or b).
```

**#6** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 30 seconds.
Answer with a) or b).
```

### Horizon: 300 seconds

**#7** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 300 seconds.
Answer with a) or b).
```

**#8** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 300 seconds.
Answer with a) or b).
```

**#9** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 300 seconds.
Answer with a) or b).
```

**#10** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 300 seconds.
Answer with a) or b).
```

**#11** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 300 seconds.
Answer with a) or b).
```

**#12** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 300 seconds.
Answer with a) or b).
```

### Horizon: 3600 seconds

**#13** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 3600 seconds.
Answer with a) or b).
```

**#14** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 3600 seconds.
Answer with a) or b).
```

**#15** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 3600 seconds.
Answer with a) or b).
```

**#16** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 3600 seconds.
Answer with a) or b).
```

**#17** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 3600 seconds.
Answer with a) or b).
```

**#18** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 3600 seconds.
Answer with a) or b).
```

### Horizon: 1 days

**#19** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 1 days.
Answer with a) or b).
```

**#20** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 1 days.
Answer with a) or b).
```

**#21** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 1 days.
Answer with a) or b).
```

**#22** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 1 days.
Answer with a) or b).
```

**#23** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 1 days.
Answer with a) or b).
```

**#24** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 1 days.
Answer with a) or b).
```

### Horizon: 7 days

**#25** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 7 days.
Answer with a) or b).
```

**#26** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 7 days.
Answer with a) or b).
```

**#27** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 7 days.
Answer with a) or b).
```

**#28** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 7 days.
Answer with a) or b).
```

**#29** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 7 days.
Answer with a) or b).
```

**#30** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 7 days.
Answer with a) or b).
```

### Horizon: 1 months

**#31** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 1 months.
Answer with a) or b).
```

**#32** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 1 months.
Answer with a) or b).
```

**#33** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 1 months.
Answer with a) or b).
```

**#34** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 1 months.
Answer with a) or b).
```

**#35** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 1 months.
Answer with a) or b).
```

**#36** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 1 months.
Answer with a) or b).
```

### Horizon: 3 months

**#37** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 3 months.
Answer with a) or b).
```

**#38** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 3 months.
Answer with a) or b).
```

**#39** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 3 months.
Answer with a) or b).
```

**#40** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 3 months.
Answer with a) or b).
```

**#41** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 3 months.
Answer with a) or b).
```

**#42** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 3 months.
Answer with a) or b).
```

### Horizon: 6 months

**#43** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 6 months.
Answer with a) or b).
```

**#44** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 6 months.
Answer with a) or b).
```

**#45** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 6 months.
Answer with a) or b).
```

**#46** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 6 months.
Answer with a) or b).
```

**#47** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 6 months.
Answer with a) or b).
```

**#48** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 6 months.
Answer with a) or b).
```

### Horizon: 1 years

**#49** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 1 years.
Answer with a) or b).
```

**#50** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 1 years.
Answer with a) or b).
```

**#51** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 1 years.
Answer with a) or b).
```

**#52** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 1 years.
Answer with a) or b).
```

**#53** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 1 years.
Answer with a) or b).
```

**#54** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 1 years.
Answer with a) or b).
```

### Horizon: 2 years

**#55** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 2 years.
Answer with a) or b).
```

**#56** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 2 years.
Answer with a) or b).
```

**#57** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 2 years.
Answer with a) or b).
```

**#58** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 2 years.
Answer with a) or b).
```

**#59** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 2 years.
Answer with a) or b).
```

**#60** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 2 years.
Answer with a) or b).
```

### Horizon: 5 years

**#61** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 5 years.
Answer with a) or b).
```

**#62** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 5 years.
Answer with a) or b).
```

**#63** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 5 years.
Answer with a) or b).
```

**#64** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 5 years.
Answer with a) or b).
```

**#65** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 5 years.
Answer with a) or b).
```

**#66** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 5 years.
Answer with a) or b).
```

### Horizon: 10 years

**#67** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 10 years.
Answer with a) or b).
```

**#68** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 10 years.
Answer with a) or b).
```

**#69** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 10 years.
Answer with a) or b).
```

**#70** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 10 years.
Answer with a) or b).
```

**#71** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 10 years.
Answer with a) or b).
```

**#72** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 10 years.
Answer with a) or b).
```

### Horizon: 25 years

**#73** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 25 years.
Answer with a) or b).
```

**#74** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 25 years.
Answer with a) or b).
```

**#75** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 25 years.
Answer with a) or b).
```

**#76** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 25 years.
Answer with a) or b).
```

**#77** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 25 years.
Answer with a) or b).
```

**#78** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 25 years.
Answer with a) or b).
```

### Horizon: 50 years

**#79** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 50 years.
Answer with a) or b).
```

**#80** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 50 years.
Answer with a) or b).
```

**#81** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 50 years.
Answer with a) or b).
```

**#82** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 50 years.
Answer with a) or b).
```

**#83** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 50 years.
Answer with a) or b).
```

**#84** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 50 years.
Answer with a) or b).
```

### Horizon: 100 years

**#85** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 100 years.
Answer with a) or b).
```

**#86** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 100 years.
Answer with a) or b).
```

**#87** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 100 years.
Answer with a) or b).
```

**#88** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 100 years.
Answer with a) or b).
```

**#89** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 100 years.
Answer with a) or b).
```

**#90** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 100 years.
Answer with a) or b).
```

### Horizon: 500 years

**#91** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 500 years.
Answer with a) or b).
```

**#92** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 500 years.
Answer with a) or b).
```

**#93** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 500 years.
Answer with a) or b).
```

**#94** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 500 years.
Answer with a) or b).
```

**#95** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit for this time horizon: 500 years.
Answer with a) or b).
```

**#96** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit for this time horizon: 500 years.
Answer with a) or b).
```

### Horizon: none (control)

**#97** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#98** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#99** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#100** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#101** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#102** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#103** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#104** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#105** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#106** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#107** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#108** rewards $20,000/$500,000, delays 6 months/10 years
```
You must choose the best investment:
a) 20,000 dollars in 6 months.
b) 500,000 dollars in 10 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#109** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#110** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#111** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#112** rewards $5,000/$200,000, delays 6 months/10 years
```
You must choose the best investment:
a) 5,000 dollars in 6 months.
b) 200,000 dollars in 10 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#113** rewards $20,000/$500,000, delays 1 month/5 years
```
You must choose the best investment:
a) 20,000 dollars in 1 month.
b) 500,000 dollars in 5 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#114** rewards $5,000/$200,000, delays 1 month/5 years
```
You must choose the best investment:
a) 5,000 dollars in 1 month.
b) 200,000 dollars in 5 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#115** rewards $1,000/$50,000, delays 6 months/10 years
```
You must choose the best investment:
a) 1,000 dollars in 6 months.
b) 50,000 dollars in 10 years.
Select the option with the greatest benefit.
Answer with a) or b).
```

**#116** rewards $1,000/$50,000, delays 1 month/5 years
```
You must choose the best investment:
a) 1,000 dollars in 1 month.
b) 50,000 dollars in 5 years.
Select the option with the greatest benefit.
Answer with a) or b).
```
