# Phrasing study: qwen3-14b_phrasing_s0

Model `Qwen/Qwen3-14B` (rev 40c069824f4251a91eefaf281ebe4c544efd3e18), commit f84e676b84d1ec95ecda6e9b4fe91d13126e8ae7, GPU NVIDIA RTX PRO 5000 Blackwell. Choice readout separator ' '.

## Choice protocol

24864 items. Free-generation check: 100% follow the format, readout agrees on 94%. Readout mass on the two labels: median 0.836, min 0.000.

### Validity gate (canonical curve)

| spearman_logh_vs_p | p_short_lowest | p_short_highest | order_effect | null_floor | n_rho_cells |
|---|---|---|---|---|---|
| -0.669 | 0.786 | 0.137 | 0.152 | 0.00926 | 320 |

| level | years | True | False | mean |
|---|---|---|---|---|
| 1 week | 0.0192 | 1 | 0.572 | 0.786 |
| 1 month | 0.0833 | 1 | 0.938 | 0.969 |
| 3 months | 0.25 | 1 | 0.881 | 0.941 |
| 6 months | 0.5 | 1 | 0.89 | 0.945 |
| 1 year | 1 | 1 | 0.796 | 0.898 |
| 2 years | 2 | 0.75 | 0.678 | 0.714 |
| 5 years | 5 | 0.541 | 0.344 | 0.443 |
| 10 years | 10 | 0.327 | 0.142 | 0.234 |
| 20 years | 20 | 0.151 | 0.0683 | 0.11 |
| 50 years | 50 | 0.166 | 0.109 | 0.137 |

### Variants: Δ P(short) vs reference (95% CI over scenarios). `matters` = above the null floor, CI excludes 0, |Δ| ≥ 0.05

| family | variant | ref | n | mean_delta | lo | hi | mean_abs_delta | meaning_shift | matters |
|---|---|---|---|---|---|---|---|---|---|
| cue | later_1 | canonical | 320 | -0.114 | -0.165 | -0.0675 | 0.126 | False | True |
| cue | relax_3 | canonical | 320 | -0.0695 | -0.0957 | -0.045 | 0.0788 | False | True |
| cue | relax_2 | canonical | 320 | -0.0481 | -0.067 | -0.0313 | 0.0527 | False | False |
| cue | relax_1 | canonical | 320 | -0.0207 | -0.0305 | -0.0118 | 0.0255 | False | False |
| cue | neutral_1 | canonical | 320 | -0.000264 | -0.0133 | 0.0126 | 0.0191 | False | False |
| cue | neutral_2 | canonical | 320 | 0.0047 | -0.00687 | 0.0148 | 0.0236 | False | False |
| cue | hurry_2 | canonical | 320 | 0.00483 | -0.0147 | 0.0229 | 0.0353 | False | False |
| cue | hurry_1 | canonical | 320 | 0.00765 | -0.0109 | 0.0252 | 0.0414 | False | False |
| cue | hurry_3 | canonical | 320 | 0.0148 | 0.00173 | 0.0294 | 0.0269 | False | False |
| cue | sooner_1 | canonical | 320 | 0.245 | 0.21 | 0.278 | 0.246 | False | True |
| implicit | implicit:baby_8wk | twin:baby_8wk | 16 | -0.626 | -0.799 | -0.438 | 0.626 | False | True |
| implicit | implicit:baby_new | twin:baby_new | 16 | -0.601 | -0.795 | -0.391 | 0.601 | False | True |
| implicit | implicit:busy_season | twin:busy_season | 16 | -0.579 | -0.729 | -0.442 | 0.579 | False | True |
| implicit | implicit:retire_60_65@user_request | twin:retire_60_65@user_request | 16 | -0.517 | -0.754 | -0.238 | 0.574 | False | True |
| implicit | implicit:friday_monday | twin:friday_monday | 16 | -0.476 | -0.684 | -0.282 | 0.476 | False | True |
| implicit | implicit:retire_60_65 | twin:retire_60_65 | 16 | -0.452 | -0.781 | -0.101 | 0.503 | False | True |
| implicit | implicit:final_school_year@user_request | twin:final_school_year@user_request | 16 | -0.441 | -0.563 | -0.327 | 0.441 | False | True |
| implicit | implicit:toddler_school | twin:toddler_school | 16 | -0.409 | -0.636 | -0.157 | 0.482 | False | True |
| implicit | implicit:wedding_spring | twin:wedding_spring | 16 | -0.402 | -0.597 | -0.224 | 0.402 | False | True |
| implicit | implicit:lease@user_request | twin:lease@user_request | 16 | -0.397 | -0.446 | -0.344 | 0.397 | False | True |
| implicit | implicit:toddler_school@user_request | twin:toddler_school@user_request | 16 | -0.369 | -0.535 | -0.213 | 0.371 | False | True |
| implicit | implicit:newborn_school@user_request | twin:newborn_school@user_request | 16 | -0.36 | -0.568 | -0.12 | 0.425 | False | True |
| implicit | implicit:birthday_40 | twin:birthday_40 | 16 | -0.332 | -0.619 | -0.075 | 0.333 | False | True |
| implicit | implicit:final_school_year | twin:final_school_year | 16 | -0.328 | -0.568 | -0.122 | 0.328 | False | True |
| implicit | implicit:kids_grown@user_request | twin:kids_grown@user_request | 16 | -0.314 | -0.533 | -0.104 | 0.337 | False | True |
| implicit | implicit:baby_8wk@user_request | twin:baby_8wk@user_request | 16 | -0.3 | -0.445 | -0.129 | 0.3 | False | True |
| implicit | implicit:baby_new@user_request | twin:baby_new@user_request | 16 | -0.289 | -0.443 | -0.121 | 0.289 | False | True |
| implicit | implicit:friday_monday@user_request | twin:friday_monday@user_request | 16 | -0.265 | -0.391 | -0.138 | 0.265 | False | True |
| implicit | implicit:university_8 | twin:university_8 | 16 | -0.214 | -0.412 | -0.0346 | 0.231 | False | True |
| implicit | implicit:university_8@user_request | twin:university_8@user_request | 16 | -0.184 | -0.363 | -0.00773 | 0.248 | False | True |
| implicit | implicit:wedding_spring@user_request | twin:wedding_spring@user_request | 16 | -0.183 | -0.335 | -0.0604 | 0.183 | False | True |
| implicit | implicit:busy_season@user_request | twin:busy_season@user_request | 16 | -0.174 | -0.324 | -0.0461 | 0.174 | False | True |
| implicit | implicit:newborn_school | twin:newborn_school | 16 | -0.172 | -0.388 | 0.0722 | 0.321 | False | False |
| implicit | implicit:birthday_40@user_request | twin:birthday_40@user_request | 16 | -0.123 | -0.271 | 0.0302 | 0.216 | False | False |
| implicit | implicit:week_end@user_request | twin:week_end@user_request | 16 | -0.0947 | -0.208 | -0.0104 | 0.0947 | False | True |
| implicit | implicit:week_end | twin:week_end | 16 | -0.0795 | -0.167 | -0.00669 | 0.0824 | False | True |
| implicit | implicit:year_end_oct@user_request | twin:year_end_oct@user_request | 16 | -0.0707 | -0.175 | 0.000178 | 0.0711 | False | False |
| implicit | implicit:lease | twin:lease | 16 | -0.0453 | -0.119 | -0.00149 | 0.0453 | False | False |
| implicit | implicit:newborn_adult | twin:newborn_adult | 16 | -0.0445 | -0.202 | 0.104 | 0.179 | False | False |
| implicit | implicit:year_end_oct | twin:year_end_oct | 16 | -0.0033 | -0.00942 | 0.00038 | 0.00387 | False | False |
| implicit | implicit:newborn_adult@user_request | twin:newborn_adult@user_request | 16 | 0.00221 | -0.149 | 0.163 | 0.196 | False | False |
| implicit | implicit:first_job@user_request | twin:first_job@user_request | 16 | 0.00558 | -0.0458 | 0.0663 | 0.0688 | False | False |
| implicit | implicit:mortgage_30 | twin:mortgage_30 | 16 | 0.101 | -0.0891 | 0.269 | 0.212 | False | False |
| implicit | implicit:first_job | twin:first_job | 16 | 0.139 | 0.0345 | 0.257 | 0.139 | False | True |
| implicit | implicit:mortgage_30@user_request | twin:mortgage_30@user_request | 16 | 0.165 | 0.0372 | 0.308 | 0.228 | False | True |
| implicit | implicit:kids_grown | twin:kids_grown | 16 | 0.177 | 0.0336 | 0.357 | 0.196 | False | True |
| labels | labels_AB | canonical | 320 | -0.00247 | -0.026 | 0.0191 | 0.0508 | False | False |
| labels | labels_12 | canonical | 320 | 0.197 | 0.171 | 0.222 | 0.198 | False | True |
| layout | task_brief | canonical | 320 | -0.0772 | -0.133 | -0.0201 | 0.186 | False | True |
| layout | markdown | canonical | 320 | -0.0114 | -0.0346 | 0.0082 | 0.0382 | False | False |
| layout | prose | canonical | 320 | 0.0252 | -0.0145 | 0.0613 | 0.0722 | False | False |
| layout | user_request | canonical | 320 | 0.0264 | -0.0142 | 0.0618 | 0.113 | False | False |
| layout | constraint_first | canonical | 320 | 0.0819 | 0.043 | 0.121 | 0.129 | False | True |
| layout_cross | task_brief+later_1 | task_brief | 320 | -0.347 | -0.438 | -0.255 | 0.347 | False | True |
| layout_cross | prose+later_1 | prose | 320 | -0.25 | -0.297 | -0.203 | 0.251 | False | True |
| layout_cross | markdown+later_1 | markdown | 320 | -0.163 | -0.239 | -0.0889 | 0.208 | False | True |
| layout_cross | user_request+later_1 | user_request | 320 | -0.114 | -0.18 | -0.0518 | 0.159 | False | True |
| layout_cross | constraint_first+unit_days | constraint_first | 320 | -0.113 | -0.145 | -0.0862 | 0.113 | False | True |
| layout_cross | constraint_first+unit_months | constraint_first | 224 | -0.0992 | -0.162 | -0.0322 | 0.157 | False | True |
| layout_cross | constraint_first+later_1 | constraint_first | 320 | -0.0976 | -0.123 | -0.0733 | 0.0976 | False | True |
| layout_cross | markdown+unit_months | markdown | 224 | -0.0696 | -0.119 | -0.013 | 0.148 | False | True |
| layout_cross | prose+unit_months | prose | 224 | -0.0673 | -0.122 | -0.0106 | 0.137 | False | True |
| layout_cross | task_brief+named | task_brief | 320 | -0.063 | -0.0857 | -0.0416 | 0.0813 | False | True |
| layout_cross | markdown+unit_days | markdown | 320 | -0.0625 | -0.103 | -0.0204 | 0.118 | False | True |
| layout_cross | constraint_first+named | constraint_first | 320 | -0.0599 | -0.0814 | -0.0406 | 0.064 | False | True |
| layout_cross | prose+unit_days | prose | 320 | -0.0591 | -0.0937 | -0.0276 | 0.112 | False | True |
| layout_cross | markdown+named | markdown | 320 | -0.0513 | -0.0675 | -0.0332 | 0.0701 | False | True |
| layout_cross | prose+named | prose | 320 | -0.0507 | -0.0674 | -0.0344 | 0.0825 | False | True |
| layout_cross | markdown+relax_1 | markdown | 320 | -0.0376 | -0.056 | -0.0219 | 0.0389 | False | False |
| layout_cross | user_request+named | user_request | 320 | -0.0289 | -0.0443 | -0.0115 | 0.0639 | False | False |
| layout_cross | prose+neutral_1 | prose | 320 | -0.0264 | -0.0438 | -0.012 | 0.0364 | False | False |
| layout_cross | prose+relax_1 | prose | 320 | -0.0231 | -0.0484 | 0.00039 | 0.0487 | False | False |
| layout_cross | prose+hurry_1 | prose | 320 | -0.015 | -0.0367 | 0.00395 | 0.0474 | False | False |
| layout_cross | markdown+neutral_1 | markdown | 320 | -0.0128 | -0.0268 | -0.000407 | 0.0253 | False | False |
| layout_cross | user_request+relax_1 | user_request | 320 | -0.0127 | -0.0413 | 0.014 | 0.0664 | False | False |
| layout_cross | user_request+neutral_1 | user_request | 320 | -0.0119 | -0.0296 | 0.00333 | 0.0367 | False | False |
| layout_cross | user_request+hurry_1 | user_request | 320 | -0.0119 | -0.0365 | 0.0119 | 0.0636 | False | False |
| layout_cross | task_brief+unit_days | task_brief | 320 | -0.0115 | -0.047 | 0.0248 | 0.125 | False | False |
| layout_cross | markdown+hurry_1 | markdown | 320 | -0.0107 | -0.0292 | 0.00572 | 0.0306 | False | False |
| layout_cross | user_request+unit_months | user_request | 224 | -0.00545 | -0.0589 | 0.049 | 0.132 | False | False |
| layout_cross | constraint_first+neutral_1 | constraint_first | 320 | -0.00344 | -0.0133 | 0.00381 | 0.0109 | False | False |
| layout_cross | constraint_first+relax_1 | constraint_first | 320 | 0.00284 | -0.00759 | 0.0117 | 0.0143 | False | False |
| layout_cross | constraint_first+hurry_1 | constraint_first | 320 | 0.00308 | -0.00862 | 0.0133 | 0.0174 | False | False |
| layout_cross | user_request+unit_days | user_request | 320 | 0.0104 | -0.0186 | 0.0391 | 0.105 | False | False |
| layout_cross | task_brief+unit_months | task_brief | 224 | 0.0145 | -0.0307 | 0.0643 | 0.108 | False | False |
| layout_cross | task_brief+relax_1 | task_brief | 320 | 0.0189 | -0.00588 | 0.0509 | 0.046 | False | False |
| layout_cross | task_brief+neutral_1 | task_brief | 320 | 0.0193 | 0.00716 | 0.0358 | 0.0259 | False | False |
| layout_cross | task_brief+hurry_1 | task_brief | 320 | 0.0255 | 0.00712 | 0.0447 | 0.0441 | False | False |
| layout_cross | constraint_first+sooner_1 | constraint_first | 320 | 0.101 | 0.0792 | 0.123 | 0.108 | False | True |
| layout_cross | user_request+sooner_1 | user_request | 320 | 0.195 | 0.161 | 0.231 | 0.202 | False | True |
| layout_cross | prose+sooner_1 | prose | 320 | 0.225 | 0.178 | 0.272 | 0.225 | False | True |
| layout_cross | markdown+sooner_1 | markdown | 320 | 0.262 | 0.226 | 0.298 | 0.262 | False | True |
| layout_cross | task_brief+sooner_1 | task_brief | 320 | 0.316 | 0.263 | 0.37 | 0.316 | False | True |
| no_horizon | nohorizon_later_1 | nohorizon_none | 32 | -0.207 | -0.318 | -0.104 | 0.207 | False | True |
| no_horizon | nohorizon_relax_1 | nohorizon_none | 32 | -0.00741 | -0.0209 | 0.00543 | 0.0162 | False | False |
| no_horizon | nohorizon_hurry_1 | nohorizon_none | 32 | 0.0261 | 0.00235 | 0.0561 | 0.0281 | False | False |
| no_horizon | nohorizon_sooner_1 | nohorizon_none | 32 | 0.698 | 0.549 | 0.832 | 0.698 | False | True |
| nulls | null_largest | canonical | 320 | -0.00891 | -0.014 | -0.00394 | 0.0137 | False | False |
| nulls | null_no_period | canonical | 320 | -0.0043 | -0.00959 | 0.000691 | 0.012 | False | False |
| nulls | null_gives | canonical | 320 | 0.00294 | -0.00167 | 0.00729 | 0.00948 | False | False |
| nulls | null_thanks | canonical | 320 | 0.00371 | -0.0116 | 0.0176 | 0.0275 | False | False |
| nulls | null_please | canonical | 320 | 0.00547 | -0.0088 | 0.0209 | 0.0235 | False | False |
| nulls | null_carefully | canonical | 320 | 0.00926 | -6.03e-06 | 0.0198 | 0.0158 | False | False |
| option_unit | opt_months_h_months | opt_months | 224 | -0.0926 | -0.128 | -0.0549 | 0.135 | False | True |
| option_unit | opt_months | canonical | 320 | 0.0226 | -0.0168 | 0.0616 | 0.123 | False | False |
| reword | reword_time_frame | canonical | 320 | -0.147 | -0.188 | -0.107 | 0.206 | False | True |
| reword | reword_only_within | canonical | 320 | -0.0746 | -0.11 | -0.0415 | 0.146 | True | True |
| reword | reword_within_next | canonical | 320 | -0.0174 | -0.037 | 0.00268 | 0.0633 | False | False |
| reword | reword_planning_horizon | canonical | 320 | -0.0105 | -0.0271 | 0.00549 | 0.0633 | False | False |
| unit | unit_decades | canonical | 128 | -0.0727 | -0.132 | -0.0145 | 0.117 | False | True |
| unit | unit_months | canonical | 224 | -0.0574 | -0.112 | 0.00369 | 0.15 | False | False |
| unit | unit_days | canonical | 320 | -0.0546 | -0.0921 | -0.0156 | 0.131 | False | True |
| unit | unit_weeks | canonical | 288 | -0.0461 | -0.0871 | -0.00491 | 0.143 | False | False |
| unit | named | canonical | 320 | -0.0369 | -0.0527 | -0.0219 | 0.0555 | False | False |
| unit | words | canonical | 320 | -0.0265 | -0.041 | -0.0142 | 0.0392 | False | False |
| unit | unit_years | canonical | 128 | 0.0435 | 0.00847 | 0.0891 | 0.0758 | False | False |

### Effective-horizon shift (log10 years; multiplier = 10^shift; < 1 acts like a shorter horizon)

| variant | ref | shift_log10 | lo | hi | slope | n_levels | multiplier | reliable |
|---|---|---|---|---|---|---|---|---|
| null_no_period | canonical | 0.0274 | 0.0107 | 0.0458 | -5.65 | 10 | 1.07 | True |
| null_largest | canonical | 0.0375 | 0.0214 | 0.0557 | -5.61 | 10 | 1.09 | True |
| null_gives | canonical | 0.00302 | -0.00571 | 0.0114 | -5.6 | 10 | 1.01 | True |
| null_carefully | canonical | -0.0237 | -0.0447 | -0.00604 | -5.66 | 10 | 0.947 | True |
| null_thanks | canonical | -0.0216 | -0.0498 | -0.000743 | -5.7 | 10 | 0.951 | True |
| null_please | canonical | -0.0181 | -0.048 | 0.0141 | -5.54 | 10 | 0.959 | True |
| reword_planning_horizon | canonical | 0.026 | -0.0275 | 0.0808 | -5.37 | 10 | 1.06 | True |
| reword_time_frame | canonical | 0.595 | 0.448 | 0.745 | -4.68 | 10 | 3.93 | True |
| reword_within_next | canonical | 0.0849 | 0.0265 | 0.17 | -5.33 | 10 | 1.22 | True |
| reword_only_within | canonical | 0.267 | 0.169 | 0.384 | -4.84 | 10 | 1.85 | True |
| unit_days | canonical | 0.167 | 0.0681 | 0.276 | -5.82 | 10 | 1.47 | True |
| unit_weeks | canonical | 0.113 | 0.0307 | 0.202 | -7.09 | 9 | 1.3 | True |
| unit_months | canonical | 0.172 | -0.0393 | 0.344 | -5.08 | 7 | 1.49 | True |
| unit_years | canonical | -0.383 | -0.707 | -0.0342 | 1.62 | 4 | 0.414 | False |
| unit_decades | canonical | 0.175 | 0.0198 | 0.448 | -5.83 | 4 | 1.5 | True |
| words | canonical | 0.0841 | 0.0538 | 0.114 | -5.55 | 10 | 1.21 | True |
| named | canonical | 0.146 | 0.107 | 0.183 | -5.51 | 10 | 1.4 | True |
| opt_months | canonical | -0.0577 | -0.158 | 0.0378 | -5.39 | 10 | 0.876 | True |
| opt_months_h_months | opt_months | 0.375 | 0.283 | 0.486 | -4.56 | 7 | 2.37 | True |
| labels_AB | canonical | 0.0109 | -0.0297 | 0.0585 | -5.87 | 10 | 1.03 | True |
| labels_12 | canonical | -0.819 | -0.971 | -0.684 | -4.41 | 10 | 0.152 | True |
| constraint_first | canonical | -0.285 | -0.418 | -0.174 | -5.89 | 10 | 0.519 | True |
| markdown | canonical | 0.0142 | -0.0344 | 0.066 | -5.54 | 10 | 1.03 | True |
| prose | canonical | -0.0626 | -0.168 | 0.0629 | -5.47 | 10 | 0.866 | True |
| task_brief | canonical | 0.306 | 0.156 | 0.488 | -4.74 | 10 | 2.02 | True |
| user_request | canonical | -0.0535 | -0.167 | 0.0682 | -5.14 | 10 | 0.884 | True |
| hurry_1 | canonical | -0.0119 | -0.0468 | 0.0204 | -5.81 | 10 | 0.973 | True |
| hurry_2 | canonical | 0.00658 | -0.0378 | 0.055 | -5.78 | 10 | 1.02 | True |
| hurry_3 | canonical | -0.0391 | -0.0694 | -0.0145 | -5.64 | 10 | 0.914 | True |
| relax_1 | canonical | 0.0444 | 0.0232 | 0.0673 | -5.6 | 10 | 1.11 | True |
| relax_2 | canonical | 0.166 | 0.119 | 0.233 | -5.53 | 10 | 1.46 | True |
| relax_3 | canonical | 0.366 | 0.287 | 0.473 | -5.02 | 10 | 2.33 | True |
| sooner_1 | canonical | -0.986 | -1.2 | -0.784 | -4.33 | 10 | 0.103 | True |
| later_1 | canonical | 0.501 | 0.34 | 0.682 | -5.61 | 10 | 3.17 | True |
| neutral_1 | canonical | -0.0116 | -0.0325 | 0.00709 | -5.66 | 10 | 0.974 | True |
| neutral_2 | canonical | -0.0168 | -0.0377 | 0.00345 | -5.71 | 10 | 0.962 | True |

### Contrasts

| contrast | a | b | n | mean_diff | lo | hi |
|---|---|---|---|---|---|---|
| pressure cue (hurry - relax), all phrasings, canonical layout | hurry | relax | 320 | 0.0552 | 0.0322 | 0.0784 |
| pressure cue (hurry - relax), no horizon | hurry | relax | 32 | 0.0335 | 0.00676 | 0.0669 |
| pressure cue (hurry - relax), layout constraint_first | hurry | relax | 320 | 0.000241 | -0.0123 | 0.00977 |
| pressure cue (hurry - relax), layout markdown | hurry | relax | 320 | 0.0269 | 0.0151 | 0.0404 |
| pressure cue (hurry - relax), layout prose | hurry | relax | 320 | 0.00809 | -0.00572 | 0.0201 |
| pressure cue (hurry - relax), layout task_brief | hurry | relax | 320 | 0.00659 | -0.0109 | 0.0211 |
| pressure cue (hurry - relax), layout user_request | hurry | relax | 320 | 0.000865 | -0.00994 | 0.011 |
| agreement cue (sooner - later), all phrasings, canonical layout | sooner | later | 320 | 0.358 | 0.288 | 0.43 |
| agreement cue (sooner - later), no horizon | sooner | later | 32 | 0.905 | 0.83 | 0.973 |
| agreement cue (sooner - later), layout constraint_first | sooner | later | 320 | 0.198 | 0.171 | 0.227 |
| agreement cue (sooner - later), layout markdown | sooner | later | 320 | 0.425 | 0.33 | 0.522 |
| agreement cue (sooner - later), layout prose | sooner | later | 320 | 0.475 | 0.417 | 0.539 |
| agreement cue (sooner - later), layout task_brief | sooner | later | 320 | 0.662 | 0.553 | 0.774 |
| agreement cue (sooner - later), layout user_request | sooner | later | 320 | 0.309 | 0.25 | 0.372 |
| option order (short first - short second), all variants |  |  | 12432 | 0.218 | 0.13 | 0.305 |
| unit-match interaction (h months: with month options - with mixed options) |  |  | 224 | -0.0353 | -0.0835 | 0.0152 |

### Layout cross: the same variant's Δ P(short) in each layout (vs that layout's canonical)

| base_variant | layout | n | mean_delta | lo | hi |
|---|---|---|---|---|---|
| hurry_1 | constraint_first | 320 | 0.00308 | -0.00862 | 0.0133 |
| hurry_1 | formatted | 320 | 0.00765 | -0.0109 | 0.0252 |
| hurry_1 | markdown | 320 | -0.0107 | -0.0292 | 0.00572 |
| hurry_1 | prose | 320 | -0.015 | -0.0367 | 0.00395 |
| hurry_1 | task_brief | 320 | 0.0255 | 0.00712 | 0.0447 |
| hurry_1 | user_request | 320 | -0.0119 | -0.0365 | 0.0119 |
| later_1 | constraint_first | 320 | -0.0976 | -0.123 | -0.0733 |
| later_1 | formatted | 320 | -0.114 | -0.165 | -0.0675 |
| later_1 | markdown | 320 | -0.163 | -0.239 | -0.0889 |
| later_1 | prose | 320 | -0.25 | -0.297 | -0.203 |
| later_1 | task_brief | 320 | -0.347 | -0.438 | -0.255 |
| later_1 | user_request | 320 | -0.114 | -0.18 | -0.0518 |
| named | constraint_first | 320 | -0.0599 | -0.0814 | -0.0406 |
| named | formatted | 320 | -0.0369 | -0.0527 | -0.0219 |
| named | markdown | 320 | -0.0513 | -0.0675 | -0.0332 |
| named | prose | 320 | -0.0507 | -0.0674 | -0.0344 |
| named | task_brief | 320 | -0.063 | -0.0857 | -0.0416 |
| named | user_request | 320 | -0.0289 | -0.0443 | -0.0115 |
| neutral_1 | constraint_first | 320 | -0.00344 | -0.0133 | 0.00381 |
| neutral_1 | formatted | 320 | -0.000264 | -0.0133 | 0.0126 |
| neutral_1 | markdown | 320 | -0.0128 | -0.0268 | -0.000407 |
| neutral_1 | prose | 320 | -0.0264 | -0.0438 | -0.012 |
| neutral_1 | task_brief | 320 | 0.0193 | 0.00716 | 0.0358 |
| neutral_1 | user_request | 320 | -0.0119 | -0.0296 | 0.00333 |
| relax_1 | constraint_first | 320 | 0.00284 | -0.00759 | 0.0117 |
| relax_1 | formatted | 320 | -0.0207 | -0.0305 | -0.0118 |
| relax_1 | markdown | 320 | -0.0376 | -0.056 | -0.0219 |
| relax_1 | prose | 320 | -0.0231 | -0.0484 | 0.00039 |
| relax_1 | task_brief | 320 | 0.0189 | -0.00588 | 0.0509 |
| relax_1 | user_request | 320 | -0.0127 | -0.0413 | 0.014 |
| sooner_1 | constraint_first | 320 | 0.101 | 0.0792 | 0.123 |
| sooner_1 | formatted | 320 | 0.245 | 0.21 | 0.278 |
| sooner_1 | markdown | 320 | 0.262 | 0.226 | 0.298 |
| sooner_1 | prose | 320 | 0.225 | 0.178 | 0.272 |
| sooner_1 | task_brief | 320 | 0.316 | 0.263 | 0.37 |
| sooner_1 | user_request | 320 | 0.195 | 0.161 | 0.231 |
| unit_days | constraint_first | 320 | -0.113 | -0.145 | -0.0862 |
| unit_days | formatted | 320 | -0.0546 | -0.0921 | -0.0156 |
| unit_days | markdown | 320 | -0.0625 | -0.103 | -0.0204 |
| unit_days | prose | 320 | -0.0591 | -0.0937 | -0.0276 |
| unit_days | task_brief | 320 | -0.0115 | -0.047 | 0.0248 |
| unit_days | user_request | 320 | 0.0104 | -0.0186 | 0.0391 |
| unit_months | constraint_first | 224 | -0.0992 | -0.162 | -0.0322 |
| unit_months | formatted | 224 | -0.0574 | -0.112 | 0.00369 |
| unit_months | markdown | 224 | -0.0696 | -0.119 | -0.013 |
| unit_months | prose | 224 | -0.0673 | -0.122 | -0.0106 |
| unit_months | task_brief | 224 | 0.0145 | -0.0307 | 0.0643 |
| unit_months | user_request | 224 | -0.00545 | -0.0589 | 0.049 |

### Implicit vs explicit twin (Δ P(short)), by layout, determinacy and hidden number

| layout | determinacy | with_number | items | n | mean_delta | lo | hi | mean_abs_delta |
|---|---|---|---|---|---|---|---|---|
| formatted | concrete | False | 4 | 64 | -0.281 | -0.37 | -0.192 | 0.315 |
| formatted | concrete | True | 6 | 96 | -0.261 | -0.345 | -0.169 | 0.325 |
| formatted | vague | False | 7 | 112 | -0.178 | -0.27 | -0.101 | 0.293 |
| formatted | vague | True | 1 | 16 | -0.409 | -0.636 | -0.157 | 0.482 |
| user_request | concrete | False | 4 | 64 | -0.156 | -0.221 | -0.0818 | 0.205 |
| user_request | concrete | True | 6 | 96 | -0.226 | -0.32 | -0.137 | 0.327 |
| user_request | vague | False | 7 | 112 | -0.223 | -0.268 | -0.176 | 0.246 |
| user_request | vague | True | 1 | 16 | -0.369 | -0.535 | -0.213 | 0.371 |

## State protocol (the model names the horizon)

| kind | n | parsed | within_2x | within_10pct | median_abs_log_err |
|---|---|---|---|---|---|
| canonical | 20 | 1 | 1 | 1 | 0 |
| implicit concrete (no number) | 4 | 1 | 0.5 | 0.25 | 0.31 |
| implicit concrete (with number) | 6 | 1 | 0.833 | 0.833 | 0 |
| implicit vague (no number) | 7 | 1 | 0.571 | 0.143 | 0.301 |
| implicit vague (with number) | 1 | 1 | 1 | 0 | 0.0969 |
| no_horizon | 2 | 1 | 0 | 0 | nan |
| twin concrete (no number) | 4 | 1 | 1 | 1 | 0 |
| twin concrete (with number) | 6 | 1 | 1 | 1 | 0 |
| twin vague (no number) | 7 | 1 | 1 | 1 | 0 |
| twin vague (with number) | 1 | 1 | 1 | 1 | 0 |
| unit | 108 | 1 | 1 | 0.991 | 0 |

## Multi-turn protocol

| conversations | step_turns | steps_parsed | base_slope_end_vs_target | base_spearman | free_median_plan_end_years |
|---|---|---|---|---|---|
| 2436 | 12180 | 0.975 | 0.898 | 0.892 | 0.134 |

Differences in log10 years of the plan's horizons vs the reference conversation (multiplier = 10^mean; 95% CI over scenarios).

| comparison | n | mean_log10 | lo | hi | multiplier |
|---|---|---|---|---|---|
| unit form named vs canonical (plan end) | 60 | -0.143 | -0.229 | -0.0647 | 0.719 |
| unit form unit_days vs canonical (plan end) | 60 | -0.182 | -0.322 | -0.0538 | 0.658 |
| unit form unit_months vs canonical (plan end) | 36 | -0.168 | -0.364 | 0.00501 | 0.679 |
| implicit vs explicit twin (plan end) | 60 | -0.423 | -0.481 | -0.32 | 0.377 |
| implicit vs explicit twin, concrete items (plan end) | 30 | -0.356 | -0.446 | -0.279 | 0.44 |
| implicit vs explicit twin, vague items (plan end) | 30 | -0.491 | -0.621 | -0.33 | 0.323 |
| base: first-turn cue direction 'hurry' vs none (plan end) | 180 | -0.17 | -0.222 | -0.119 | 0.676 |
| base: first-turn cue direction 'neutral' vs none (plan end) | 60 | -0.102 | -0.167 | -0.0445 | 0.79 |
| base: first-turn cue direction 'relax' vs none (plan end) | 180 | -0.0158 | -0.0884 | 0.0504 | 0.964 |
| base: first-turn cue 'hurry_1' vs none (plan end) | 60 | -0.126 | -0.216 | -0.0352 | 0.747 |
| base: first-turn cue 'hurry_2' vs none (plan end) | 60 | -0.199 | -0.261 | -0.132 | 0.633 |
| base: first-turn cue 'hurry_3' vs none (plan end) | 60 | -0.185 | -0.242 | -0.137 | 0.653 |
| base: first-turn cue 'neutral_1' vs none (plan end) | 60 | -0.102 | -0.167 | -0.0445 | 0.79 |
| base: first-turn cue 'relax_1' vs none (plan end) | 60 | -0.0538 | -0.119 | 0.0115 | 0.884 |
| base: first-turn cue 'relax_2' vs none (plan end) | 60 | -0.0509 | -0.124 | 0.0223 | 0.889 |
| base: first-turn cue 'relax_3' vs none (plan end) | 60 | 0.0573 | -0.0464 | 0.15 | 1.14 |
| base: cue direction 'hurry' before step 3 vs none (steps >= 3) | 180 | -0.187 | -0.234 | -0.141 | 0.65 |
| base: cue direction 'neutral' before step 3 vs none (steps >= 3) | 120 | -0.0167 | -0.0382 | 0.00158 | 0.962 |
| base: cue direction 'relax' before step 3 vs none (steps >= 3) | 180 | 0.051 | 0.0108 | 0.0968 | 1.12 |
| base: cue 'hurry_1' before step 3 vs none (steps >= 3) | 60 | -0.101 | -0.138 | -0.0688 | 0.792 |
| base: cue 'hurry_2' before step 3 vs none (steps >= 3) | 60 | -0.21 | -0.272 | -0.151 | 0.616 |
| base: cue 'hurry_3' before step 3 vs none (steps >= 3) | 60 | -0.249 | -0.315 | -0.184 | 0.563 |
| base: cue 'neutral_1' before step 3 vs none (steps >= 3) | 60 | -0.0195 | -0.0482 | 0.00828 | 0.956 |
| base: cue 'neutral_2' before step 3 vs none (steps >= 3) | 60 | -0.014 | -0.0336 | 0.00182 | 0.968 |
| base: cue 'relax_1' before step 3 vs none (steps >= 3) | 60 | 0.0154 | -0.0166 | 0.0542 | 1.04 |
| base: cue 'relax_2' before step 3 vs none (steps >= 3) | 60 | 0.0309 | -0.0128 | 0.0776 | 1.07 |
| base: cue 'relax_3' before step 3 vs none (steps >= 3) | 60 | 0.107 | 0.0399 | 0.176 | 1.28 |
| free: first-turn cue direction 'hurry' vs none (plan end) | 144 | -0.403 | -0.554 | -0.249 | 0.395 |
| free: first-turn cue direction 'neutral' vs none (plan end) | 48 | 0.0372 | -0.0459 | 0.125 | 1.09 |
| free: first-turn cue direction 'relax' vs none (plan end) | 144 | 0.137 | 0.0578 | 0.22 | 1.37 |
| free: first-turn cue 'hurry_1' vs none (plan end) | 48 | -0.386 | -0.52 | -0.251 | 0.411 |
| free: first-turn cue 'hurry_2' vs none (plan end) | 48 | -0.454 | -0.648 | -0.25 | 0.351 |
| free: first-turn cue 'hurry_3' vs none (plan end) | 48 | -0.369 | -0.523 | -0.222 | 0.427 |
| free: first-turn cue 'neutral_1' vs none (plan end) | 48 | 0.0372 | -0.0459 | 0.125 | 1.09 |
| free: first-turn cue 'relax_1' vs none (plan end) | 48 | 0.00707 | -0.0717 | 0.0776 | 1.02 |
| free: first-turn cue 'relax_2' vs none (plan end) | 48 | 0.143 | 0.0458 | 0.244 | 1.39 |
| free: first-turn cue 'relax_3' vs none (plan end) | 48 | 0.262 | 0.145 | 0.394 | 1.83 |
| free: cue direction 'hurry' before step 3 vs none (steps >= 3) | 288 | -0.227 | -0.304 | -0.144 | 0.594 |
| free: cue direction 'neutral' before step 3 vs none (steps >= 3) | 192 | 0.00328 | -0.00576 | 0.0111 | 1.01 |
| free: cue direction 'relax' before step 3 vs none (steps >= 3) | 288 | 0.106 | 0.0574 | 0.172 | 1.27 |
| free: cue 'hurry_1' before step 3 vs none (steps >= 3) | 96 | -0.112 | -0.175 | -0.0551 | 0.772 |
| free: cue 'hurry_2' before step 3 vs none (steps >= 3) | 96 | -0.275 | -0.366 | -0.176 | 0.531 |
| free: cue 'hurry_3' before step 3 vs none (steps >= 3) | 96 | -0.292 | -0.401 | -0.183 | 0.51 |
| free: cue 'neutral_1' before step 3 vs none (steps >= 3) | 96 | 0.01 | 0.00257 | 0.0185 | 1.02 |
| free: cue 'neutral_2' before step 3 vs none (steps >= 3) | 96 | -0.00345 | -0.0198 | 0.00958 | 0.992 |
| free: cue 'relax_1' before step 3 vs none (steps >= 3) | 96 | 0.0495 | 0.00209 | 0.122 | 1.12 |
| free: cue 'relax_2' before step 3 vs none (steps >= 3) | 96 | 0.0626 | 0.014 | 0.132 | 1.16 |
| free: cue 'relax_3' before step 3 vs none (steps >= 3) | 96 | 0.204 | 0.132 | 0.275 | 1.6 |
