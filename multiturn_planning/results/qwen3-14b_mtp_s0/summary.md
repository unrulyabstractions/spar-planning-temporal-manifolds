# Multi-turn planning: qwen3-14b_mtp_s0

model `Qwen/Qwen3-14B` · 480 conversations · usable (titles-only outline, 5 numbered steps): 480 · layers [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]

## Behavior

- n_usable_conversations: 480
- n_usable_steps: 2323
- frac_steps_unparsed: 0.032
- frac_steps_ongoing: 0.032
- usable_steps_per_k: {1: 480, 2: 480, 3: 480, 4: 478, 5: 405}
- slope_last_step_vs_target: 0.928
- rho_last_step_vs_target: 0.893
- frac_steps_within_target: 1.000
- frac_monotone_plans: 0.635
- none_last_step_median_years: 0.096
- none_last_step_iqr_decades: 0.477

## A · persistence of H_target (probe from turn 1, tested at turn t)

best turn-1 cell: layer 12, U; that cell across turns:

```
 turn     r2   rho  within2x  r2_same_turn  pc1_rho
    1  0.993 0.992     1.000         0.993    0.153
    2 -0.368 0.198     0.128         0.536    0.024
    3 -0.352 0.121     0.125        -0.400    0.117
    4 -0.273 0.140     0.122         0.500    0.370
    5 -0.245 0.125     0.125         0.502    0.075
    6 -0.220 0.166     0.128         0.538    0.257
    7 -0.216 0.166     0.125         0.685    0.346
```

## B / D · the step about to be written (text baseline vs activations)

```
condition  layer pos    n  r2_text  r2_act  r2_text_plus_act  delta_r2  delta_r2_shuffled  within2x_text  within2x_text_plus_act
     none     28  P8  912    0.445   0.410             0.491     0.047             -0.051          0.666                   0.630
     none     28  P2  912    0.445   0.420             0.473     0.029             -0.028          0.666                   0.633
     none     32  P8  912    0.445   0.435             0.473     0.028             -0.045          0.666                   0.615
  horizon     28  P7 1411    0.860   0.835             0.887     0.028             -0.001          0.670                   0.716
  horizon     28   U 1411    0.860   0.851             0.886     0.026             -0.004          0.670                   0.721
  horizon     24  P8 1411    0.860   0.856             0.886     0.026             -0.003          0.670                   0.709
```

reference positions inside the reply (H = 'Time horizon:', V = the value itself):

```
condition pos  layer  r2_act  delta_r2
  horizon   V      4   1.000     0.087
     none   V      8   0.988     0.386
  horizon   H     32   0.945     0.079
     none   H     40   0.771     0.256
```

## D · probe trained on horizon conversations, applied to no-horizon ones

```
 layer pos    r2   rho  within2x   n
    40   H 0.828 0.942     0.879 912
    36   H 0.810 0.941     0.863 912
    32   H 0.804 0.930     0.843 912
```

## C · turn-1 activations and later steps, beyond H_target

```
 step  layer pos  r2_target_only  r2_target_plus_act  delta_r2  delta_r2_shuffled
    4     24  P1           0.704               0.816     0.112             -0.016
    2     36  P0           0.643               0.745     0.102              0.042
    3     28  P1           0.674               0.764     0.090             -0.008
    1     36  P4           0.738               0.816     0.078             -0.007
    5     24  P2           0.790               0.817     0.028             -0.003
```

noise floor: best shuffled delta per step (same cells, residuals permuted within scenario):

```
 step  layer pos  delta_r2_shuffled
    2     32  P7              0.052
    3     36  P7              0.034
    1     32  P8              0.029
    5     40  P2              0.008
    4      4  P3             -0.002
```

_evaluate: 762s_