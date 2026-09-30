# Multi-turn planning: pilot_Qwen3-4B (quick pass)

model `Qwen/Qwen3-4B` · 15 conversations · usable (titles-only outline, 5 numbered steps): 15 · layers [14, 22, 29]

## Behavior

- n_usable_conversations: 15
- n_usable_steps: 73
- frac_steps_unparsed: 0.027
- frac_steps_ongoing: 0.027
- usable_steps_per_k: {1: 15, 2: 15, 3: 15, 4: 15, 5: 13}
- slope_last_step_vs_target: 1.031
- rho_last_step_vs_target: 0.973
- frac_steps_within_target: 1.000
- frac_monotone_plans: 0.867
- none_last_step_median_years: 0.392
- none_last_step_iqr_decades: 0.279

## A · persistence of H_target (probe from turn 1, tested at turn t)

best turn-1 cell: layer 14, P0; that cell across turns:

```
 turn      r2   rho  within2x  r2_same_turn  pc1_rho
    1   0.989 0.949     1.000         0.989    0.053
    2 -11.595 0.211     0.000        -0.000    0.105
    3 -11.442 0.000     0.000        -0.011    0.369
    4  -7.290 0.105     0.000         0.307    0.053
    5  -6.103 0.158     0.000        -0.000    0.053
    6  -5.140 0.211     0.111        -0.126    0.158
    7  -4.614 0.158     0.111        -0.000    0.053
```

## B / D · the step about to be written (text baseline vs activations)

```
condition  layer pos  n  r2_text  r2_act  r2_text_plus_act  delta_r2  delta_r2_shuffled  within2x_text  within2x_text_plus_act
  horizon     14  P8 45    0.931   0.375             0.915    -0.015             -0.014          0.711                   0.667
  horizon     22  P4 45    0.931   0.223             0.915    -0.015             -0.014          0.711                   0.667
  horizon     14  P4 45    0.931   0.360             0.915    -0.016             -0.027          0.711                   0.667
     none     14  P0 28    0.551   0.782             0.375    -0.176             -0.080          0.714                   0.643
     none     14  P4 28    0.551   0.657             0.308    -0.243             -0.038          0.714                   0.607
     none     29  P0 28    0.551   0.761             0.206    -0.345             -0.057          0.714                   0.643
```

reference positions inside the reply (H = 'Time horizon:', V = the value itself):

```
condition pos  layer  r2_act  delta_r2
  horizon   V     14   0.984    -0.008
     none   V     22   0.953     0.258
  horizon   H     29   0.932    -0.023
     none   H     29   0.837    -0.070
```

## D · probe trained on horizon conversations, applied to no-horizon ones

```
 layer pos    r2   rho  within2x  n
    29  P4 0.480 0.798     0.679 28
    14   H 0.426 0.849     0.607 28
    22  P8 0.335 0.901     0.536 28
```

## C · turn-1 activations and later steps, beyond H_target

```
 step  layer pos  r2_target_only  r2_target_plus_act  delta_r2
    3     22  P8           0.889               0.887    -0.002
    2     22  P8           0.837               0.834    -0.004
    5     14  P4           0.991               0.987    -0.005
    1     29  P8           0.955               0.947    -0.008
    4     14   U           0.938               0.923    -0.015
```

_evaluate: 1s_