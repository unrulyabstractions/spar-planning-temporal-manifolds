# Multi-turn planning: pilot_Qwen3-4B

model `Qwen/Qwen3-4B` · 15 conversations · clean (all 5 steps parsed and numbered): 13 · layers [0, 4, 8, 12, 14, 16, 20, 22, 24, 28, 29, 32, 36]

## Behavior

- n_clean_conversations: 13
- slope_last_step_vs_target: 1.031
- rho_last_step_vs_target: 0.973
- frac_steps_within_target: 1.000
- frac_monotone_plans: 0.846
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
  horizon      0  P0 45    0.931  -0.009             0.917    -0.014             -0.014          0.711                   0.667
  horizon      0  P4 45    0.931  -0.009             0.917    -0.014             -0.014          0.711                   0.667
  horizon      0  P8 45    0.931  -0.009             0.917    -0.014             -0.014          0.711                   0.667
     none     12  P0 20    0.627   0.811             0.573    -0.054             -0.107          0.600                   0.600
     none     14  P0 20    0.627   0.804             0.551    -0.076             -0.115          0.600                   0.550
     none     12   U 20    0.627   0.332             0.540    -0.087             -0.097          0.600                   0.550
```

reference positions inside the reply (H = 'Time horizon:', V = the value itself):

```
condition pos  layer  r2_act  delta_r2
  horizon   V      4   0.993    -0.016
  horizon   H     29   0.932    -0.023
     none   V      4   0.895    -0.071
     none   H     32   0.698    -0.428
```

## D · probe trained on horizon conversations, applied to no-horizon ones

```
 layer pos    r2   rho  within2x  n
    12  P0 0.446 0.804     0.500 20
    14   H 0.436 0.826     0.650 20
     4   H 0.435 0.810     0.450 20
```

## C · turn-1 activations and later steps, beyond H_target

```
 step  layer pos  r2_target_only  r2_target_plus_act  delta_r2
    2     16  P8           0.837               0.849     0.012
    3     24  P4           0.889               0.893     0.004
    5     12  P4           0.991               0.989    -0.003
    1     36  P8           0.955               0.947    -0.008
    4     14   U           0.938               0.923    -0.015
```

_evaluate: 4s_