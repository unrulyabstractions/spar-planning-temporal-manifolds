# Phrasing study: qwen3-14b_choice_r2

Model `Qwen/Qwen3-14B` (rev 40c069824f4251a91eefaf281ebe4c544efd3e18), commit d642a784d0cdf3f7cde8c8a76bbcd4c186059581, GPU NVIDIA A100 80GB PCIe. Choice readout separator ' ', summed over [' ', ' **'].

## Choice protocol

24864 items. Free-generation check: 100% follow the format, readout agrees on 99%. Readout mass on the two labels: median 1.000, min 0.999; mass_space median 0.841; mass_bold median 0.159.

### Validity gate (canonical curve)

| spearman_logh_vs_p | p_short_lowest | p_short_highest | order_effect | null_floor | n_rho_cells |
|---|---|---|---|---|---|
| -0.657 | 0.852 | 0.15 | 0.0864 | 0.0136 | 320 |

| level | years | True | False | mean |
|---|---|---|---|---|
| 1 week | 0.0192 | 1 | 0.703 | 0.852 |
| 1 month | 0.0833 | 1 | 0.99 | 0.995 |
| 3 months | 0.25 | 1 | 0.931 | 0.965 |
| 6 months | 0.5 | 1 | 0.943 | 0.972 |
| 1 year | 1 | 1 | 0.875 | 0.937 |
| 2 years | 2 | 0.75 | 0.787 | 0.769 |
| 5 years | 5 | 0.556 | 0.479 | 0.517 |
| 10 years | 10 | 0.397 | 0.253 | 0.325 |
| 20 years | 20 | 0.175 | 0.126 | 0.15 |
| 50 years | 50 | 0.186 | 0.113 | 0.15 |

### Variants: Δ P(short) vs reference (95% CI over scenarios). `matters` = above the null floor, CI excludes 0, |Δ| ≥ 0.05

| family | variant | ref | n | mean_delta | lo | hi | mean_abs_delta | meaning_shift | matters |
|---|---|---|---|---|---|---|---|---|---|
| cue | relax_3 | canonical | 320 | -0.0714 | -0.0997 | -0.0435 | 0.0822 | False | True |
| cue | later_1 | canonical | 320 | -0.0653 | -0.122 | -0.0147 | 0.104 | False | True |
| cue | relax_2 | canonical | 320 | -0.0468 | -0.0694 | -0.0264 | 0.0524 | False | False |
| cue | relax_1 | canonical | 320 | -0.00877 | -0.02 | 0.002 | 0.0265 | False | False |
| cue | neutral_2 | canonical | 320 | -0.00725 | -0.0213 | 0.00443 | 0.0221 | False | False |
| cue | hurry_2 | canonical | 320 | 0.00426 | -0.0149 | 0.0195 | 0.0334 | False | False |
| cue | neutral_1 | canonical | 320 | 0.00669 | -0.00103 | 0.0144 | 0.02 | False | False |
| cue | hurry_1 | canonical | 320 | 0.00858 | -0.0103 | 0.0231 | 0.0333 | False | False |
| cue | hurry_3 | canonical | 320 | 0.0205 | 0.00998 | 0.0307 | 0.0253 | False | False |
| cue | sooner_1 | canonical | 320 | 0.218 | 0.172 | 0.265 | 0.218 | False | True |
| implicit | implicit:retire_60_65 | twin:retire_60_65 | 16 | -0.47 | -0.753 | -0.18 | 0.492 | False | True |
| implicit | implicit:retire_60_65@user_request | twin:retire_60_65@user_request | 16 | -0.459 | -0.688 | -0.197 | 0.515 | False | True |
| implicit | implicit:busy_season | twin:busy_season | 16 | -0.451 | -0.565 | -0.344 | 0.451 | False | True |
| implicit | implicit:baby_8wk | twin:baby_8wk | 16 | -0.447 | -0.594 | -0.295 | 0.447 | False | True |
| implicit | implicit:friday_monday | twin:friday_monday | 16 | -0.401 | -0.522 | -0.283 | 0.401 | False | True |
| implicit | implicit:baby_new | twin:baby_new | 16 | -0.4 | -0.549 | -0.25 | 0.4 | False | True |
| implicit | implicit:kids_grown@user_request | twin:kids_grown@user_request | 16 | -0.375 | -0.611 | -0.147 | 0.395 | False | True |
| implicit | implicit:wedding_spring | twin:wedding_spring | 16 | -0.327 | -0.509 | -0.158 | 0.327 | False | True |
| implicit | implicit:toddler_school | twin:toddler_school | 16 | -0.282 | -0.44 | -0.109 | 0.34 | False | True |
| implicit | implicit:birthday_40 | twin:birthday_40 | 16 | -0.223 | -0.48 | 0.0273 | 0.309 | False | False |
| implicit | implicit:toddler_school@user_request | twin:toddler_school@user_request | 16 | -0.218 | -0.343 | -0.0914 | 0.225 | False | True |
| implicit | implicit:birthday_40@user_request | twin:birthday_40@user_request | 16 | -0.207 | -0.422 | -0.00949 | 0.26 | False | True |
| implicit | implicit:university_8@user_request | twin:university_8@user_request | 16 | -0.182 | -0.343 | -0.0347 | 0.232 | False | True |
| implicit | implicit:final_school_year | twin:final_school_year | 16 | -0.16 | -0.301 | -0.0318 | 0.16 | False | True |
| implicit | implicit:newborn_school | twin:newborn_school | 16 | -0.157 | -0.34 | 0.0316 | 0.262 | False | False |
| implicit | implicit:baby_8wk@user_request | twin:baby_8wk@user_request | 16 | -0.131 | -0.23 | -0.0446 | 0.131 | False | True |
| implicit | implicit:baby_new@user_request | twin:baby_new@user_request | 16 | -0.13 | -0.254 | -0.0367 | 0.13 | False | True |
| implicit | implicit:final_school_year@user_request | twin:final_school_year@user_request | 16 | -0.128 | -0.246 | -0.0368 | 0.128 | False | True |
| implicit | implicit:lease@user_request | twin:lease@user_request | 16 | -0.116 | -0.187 | -0.0549 | 0.116 | False | True |
| implicit | implicit:first_job@user_request | twin:first_job@user_request | 16 | -0.109 | -0.229 | -0.0148 | 0.13 | False | True |
| implicit | implicit:newborn_school@user_request | twin:newborn_school@user_request | 16 | -0.0907 | -0.238 | 0.0518 | 0.169 | False | False |
| implicit | implicit:newborn_adult | twin:newborn_adult | 16 | -0.0867 | -0.256 | 0.0585 | 0.191 | False | False |
| implicit | implicit:newborn_adult@user_request | twin:newborn_adult@user_request | 16 | -0.0818 | -0.21 | 0.0408 | 0.187 | False | False |
| implicit | implicit:busy_season@user_request | twin:busy_season@user_request | 16 | -0.0664 | -0.177 | -0.00123 | 0.0671 | False | True |
| implicit | implicit:university_8 | twin:university_8 | 16 | -0.0659 | -0.222 | 0.0805 | 0.196 | False | False |
| implicit | implicit:friday_monday@user_request | twin:friday_monday@user_request | 16 | -0.0593 | -0.102 | -0.0201 | 0.0596 | False | True |
| implicit | implicit:week_end | twin:week_end | 16 | -0.0313 | -0.0864 | 0.024 | 0.0592 | False | False |
| implicit | implicit:lease | twin:lease | 16 | -0.0202 | -0.0503 | -0.00127 | 0.0205 | False | False |
| implicit | implicit:week_end@user_request | twin:week_end@user_request | 16 | -0.0142 | -0.0375 | -0.00128 | 0.0143 | False | False |
| implicit | implicit:wedding_spring@user_request | twin:wedding_spring@user_request | 16 | -0.0097 | -0.023 | -0.00122 | 0.0097 | False | False |
| implicit | implicit:year_end_oct@user_request | twin:year_end_oct@user_request | 16 | -0.00238 | -0.00817 | 0.0016 | 0.00383 | False | False |
| implicit | implicit:year_end_oct | twin:year_end_oct | 16 | -0.000888 | -0.00255 | 0.000435 | 0.00147 | False | False |
| implicit | implicit:first_job | twin:first_job | 16 | 0.244 | 0.0893 | 0.433 | 0.244 | False | True |
| implicit | implicit:mortgage_30@user_request | twin:mortgage_30@user_request | 16 | 0.26 | 0.102 | 0.435 | 0.271 | False | True |
| implicit | implicit:kids_grown | twin:kids_grown | 16 | 0.27 | 0.0974 | 0.455 | 0.272 | False | True |
| implicit | implicit:mortgage_30 | twin:mortgage_30 | 16 | 0.368 | 0.19 | 0.526 | 0.388 | False | True |
| labels | labels_AB | canonical | 320 | -0.044 | -0.0732 | -0.0158 | 0.055 | False | False |
| labels | labels_12 | canonical | 320 | 0.151 | 0.118 | 0.184 | 0.155 | False | True |
| layout | task_brief | canonical | 320 | -0.0702 | -0.121 | -0.0214 | 0.173 | False | True |
| layout | prose | canonical | 320 | -0.00911 | -0.0464 | 0.0217 | 0.0746 | False | False |
| layout | user_request | canonical | 320 | 0.0343 | -0.00647 | 0.0732 | 0.116 | False | False |
| layout | markdown | canonical | 320 | 0.0463 | 0.0309 | 0.0626 | 0.0497 | False | False |
| layout | constraint_first | canonical | 320 | 0.0481 | 0.00864 | 0.0865 | 0.118 | False | False |
| layout_cross | task_brief+later_1 | task_brief | 320 | -0.285 | -0.365 | -0.208 | 0.286 | False | True |
| layout_cross | prose+later_1 | prose | 320 | -0.222 | -0.268 | -0.176 | 0.223 | False | True |
| layout_cross | constraint_first+unit_days | constraint_first | 320 | -0.118 | -0.144 | -0.0929 | 0.118 | False | True |
| layout_cross | constraint_first+unit_months | constraint_first | 224 | -0.105 | -0.168 | -0.0316 | 0.167 | False | True |
| layout_cross | user_request+later_1 | user_request | 320 | -0.0753 | -0.137 | -0.0187 | 0.134 | False | True |
| layout_cross | markdown+later_1 | markdown | 320 | -0.0679 | -0.115 | -0.0275 | 0.0838 | False | True |
| layout_cross | constraint_first+later_1 | constraint_first | 320 | -0.0628 | -0.0817 | -0.0468 | 0.0631 | False | True |
| layout_cross | task_brief+named | task_brief | 320 | -0.0582 | -0.0834 | -0.0362 | 0.08 | False | True |
| layout_cross | constraint_first+named | constraint_first | 320 | -0.0575 | -0.0723 | -0.042 | 0.0616 | False | True |
| layout_cross | prose+unit_months | prose | 224 | -0.0512 | -0.106 | 0.00572 | 0.14 | False | False |
| layout_cross | prose+named | prose | 320 | -0.0481 | -0.0648 | -0.0312 | 0.089 | False | False |
| layout_cross | prose+unit_days | prose | 320 | -0.0464 | -0.0816 | -0.016 | 0.113 | False | False |
| layout_cross | markdown+unit_months | markdown | 224 | -0.0311 | -0.0901 | 0.029 | 0.147 | False | False |
| layout_cross | markdown+unit_days | markdown | 320 | -0.0243 | -0.0555 | 0.00538 | 0.0967 | False | False |
| layout_cross | prose+neutral_1 | prose | 320 | -0.0222 | -0.0388 | -0.00614 | 0.0383 | False | False |
| layout_cross | markdown+named | markdown | 320 | -0.0217 | -0.0411 | -0.00419 | 0.0435 | False | False |
| layout_cross | user_request+named | user_request | 320 | -0.0205 | -0.0378 | -0.00554 | 0.0584 | False | False |
| layout_cross | markdown+neutral_1 | markdown | 320 | -0.0139 | -0.0294 | 0.000388 | 0.0257 | False | False |
| layout_cross | markdown+relax_1 | markdown | 320 | -0.0112 | -0.026 | 0.00174 | 0.0251 | False | False |
| layout_cross | user_request+hurry_1 | user_request | 320 | -0.0102 | -0.0303 | 0.00911 | 0.0455 | False | False |
| layout_cross | task_brief+unit_days | task_brief | 320 | -0.0101 | -0.0371 | 0.0155 | 0.137 | False | False |
| layout_cross | user_request+neutral_1 | user_request | 320 | -0.00818 | -0.0236 | 0.00568 | 0.0297 | False | False |
| layout_cross | user_request+relax_1 | user_request | 320 | -0.00519 | -0.0274 | 0.0155 | 0.0488 | False | False |
| layout_cross | markdown+hurry_1 | markdown | 320 | -0.00499 | -0.02 | 0.00898 | 0.0272 | False | False |
| layout_cross | prose+relax_1 | prose | 320 | -0.00496 | -0.0325 | 0.023 | 0.0554 | False | False |
| layout_cross | constraint_first+neutral_1 | constraint_first | 320 | -0.004 | -0.0101 | 0.000959 | 0.00744 | False | False |
| layout_cross | prose+hurry_1 | prose | 320 | 0.00129 | -0.022 | 0.0243 | 0.0503 | False | False |
| layout_cross | constraint_first+hurry_1 | constraint_first | 320 | 0.00497 | 0.000328 | 0.00899 | 0.012 | False | False |
| layout_cross | user_request+unit_months | user_request | 224 | 0.00508 | -0.0347 | 0.0473 | 0.12 | False | False |
| layout_cross | constraint_first+relax_1 | constraint_first | 320 | 0.0088 | -0.000533 | 0.0192 | 0.0147 | False | False |
| layout_cross | user_request+unit_days | user_request | 320 | 0.0106 | -0.0177 | 0.0381 | 0.0985 | False | False |
| layout_cross | task_brief+unit_months | task_brief | 224 | 0.0124 | -0.0257 | 0.0535 | 0.117 | False | False |
| layout_cross | task_brief+neutral_1 | task_brief | 320 | 0.0263 | 0.0104 | 0.0449 | 0.0315 | False | False |
| layout_cross | task_brief+relax_1 | task_brief | 320 | 0.0408 | 0.0154 | 0.0703 | 0.0533 | False | False |
| layout_cross | task_brief+hurry_1 | task_brief | 320 | 0.0526 | 0.033 | 0.0732 | 0.0611 | False | True |
| layout_cross | constraint_first+sooner_1 | constraint_first | 320 | 0.0996 | 0.0773 | 0.121 | 0.103 | False | True |
| layout_cross | markdown+sooner_1 | markdown | 320 | 0.176 | 0.125 | 0.231 | 0.176 | False | True |
| layout_cross | user_request+sooner_1 | user_request | 320 | 0.191 | 0.159 | 0.224 | 0.191 | False | True |
| layout_cross | prose+sooner_1 | prose | 320 | 0.242 | 0.182 | 0.304 | 0.242 | False | True |
| layout_cross | task_brief+sooner_1 | task_brief | 320 | 0.305 | 0.256 | 0.359 | 0.305 | False | True |
| no_horizon | nohorizon_later_1 | nohorizon_none | 32 | -0.265 | -0.37 | -0.164 | 0.265 | False | True |
| no_horizon | nohorizon_relax_1 | nohorizon_none | 32 | -0.00201 | -0.0346 | 0.0351 | 0.0352 | False | False |
| no_horizon | nohorizon_hurry_1 | nohorizon_none | 32 | 0.00841 | -0.0138 | 0.0337 | 0.023 | False | False |
| no_horizon | nohorizon_sooner_1 | nohorizon_none | 32 | 0.625 | 0.456 | 0.787 | 0.625 | False | True |
| nulls | null_no_period | canonical | 320 | -0.0136 | -0.0236 | -0.00609 | 0.0149 | False | False |
| nulls | null_largest | canonical | 320 | -0.0133 | -0.0231 | -0.00589 | 0.0172 | False | False |
| nulls | null_carefully | canonical | 320 | 0.00342 | -0.00472 | 0.0108 | 0.0132 | False | False |
| nulls | null_gives | canonical | 320 | 0.00353 | 0.000271 | 0.00689 | 0.00953 | False | False |
| nulls | null_thanks | canonical | 320 | 0.00593 | -0.00447 | 0.0161 | 0.0283 | False | False |
| nulls | null_please | canonical | 320 | 0.00647 | -0.00809 | 0.0214 | 0.0272 | False | False |
| option_unit | opt_months_h_months | opt_months | 224 | -0.0752 | -0.109 | -0.0407 | 0.119 | False | True |
| option_unit | opt_months | canonical | 320 | 0.0327 | 0.00533 | 0.0602 | 0.117 | False | False |
| reword | reword_time_frame | canonical | 320 | -0.137 | -0.174 | -0.098 | 0.186 | False | True |
| reword | reword_only_within | canonical | 320 | -0.0704 | -0.0998 | -0.0401 | 0.158 | True | True |
| reword | reword_within_next | canonical | 320 | -0.0124 | -0.0355 | 0.00999 | 0.0621 | False | False |
| reword | reword_planning_horizon | canonical | 320 | -0.0113 | -0.0269 | 0.0019 | 0.0547 | False | False |
| unit | unit_decades | canonical | 128 | -0.0743 | -0.131 | -0.0201 | 0.119 | False | True |
| unit | unit_days | canonical | 320 | -0.05 | -0.0875 | -0.0162 | 0.112 | False | False |
| unit | unit_months | canonical | 224 | -0.0471 | -0.0977 | 0.00712 | 0.14 | False | False |
| unit | named | canonical | 320 | -0.0416 | -0.0567 | -0.0276 | 0.0573 | False | False |
| unit | unit_weeks | canonical | 288 | -0.0276 | -0.0719 | 0.013 | 0.143 | False | False |
| unit | words | canonical | 320 | -0.0242 | -0.0343 | -0.0155 | 0.0309 | False | False |
| unit | unit_years | canonical | 128 | 0.0412 | 0.0131 | 0.0727 | 0.051 | False | False |

### Effective-horizon shift (log10 years; multiplier = 10^shift; < 1 acts like a shorter horizon)

| variant | ref | shift_log10 | lo | hi | slope | n_levels | multiplier | reliable |
|---|---|---|---|---|---|---|---|---|
| null_no_period | canonical | 0.0453 | 0.0277 | 0.0656 | -5.28 | 10 | 1.11 | True |
| null_largest | canonical | 0.0384 | 0.0241 | 0.0541 | -5.26 | 10 | 1.09 | True |
| null_gives | canonical | 0.000664 | -0.00786 | 0.00992 | -5.21 | 10 | 1 | True |
| null_carefully | canonical | -0.0188 | -0.0355 | -0.00236 | -5.27 | 10 | 0.958 | True |
| null_thanks | canonical | -0.011 | -0.0461 | 0.0144 | -5.33 | 10 | 0.975 | True |
| null_please | canonical | -0.042 | -0.0805 | -0.00186 | -5.1 | 10 | 0.908 | True |
| reword_planning_horizon | canonical | 0.0245 | -0.026 | 0.0747 | -5 | 10 | 1.06 | True |
| reword_time_frame | canonical | 0.561 | 0.444 | 0.684 | -4.62 | 10 | 3.64 | True |
| reword_within_next | canonical | 0.0418 | -0.00924 | 0.105 | -5.01 | 10 | 1.1 | True |
| reword_only_within | canonical | 0.209 | 0.118 | 0.316 | -4.61 | 10 | 1.62 | True |
| unit_days | canonical | 0.147 | 0.0566 | 0.239 | -5.49 | 10 | 1.4 | True |
| unit_weeks | canonical | 0.0825 | 0.0074 | 0.165 | -6.65 | 9 | 1.21 | True |
| unit_months | canonical | 0.125 | -0.0728 | 0.293 | -4.77 | 7 | 1.33 | True |
| unit_years | canonical | -0.354 | -0.631 | -0.0794 | 1.5 | 4 | 0.442 | False |
| unit_decades | canonical | 0.22 | 0.0502 | 0.523 | -6.33 | 4 | 1.66 | True |
| words | canonical | 0.0677 | 0.0408 | 0.0936 | -5.23 | 10 | 1.17 | True |
| named | canonical | 0.158 | 0.119 | 0.203 | -5.26 | 10 | 1.44 | True |
| opt_months | canonical | -0.0748 | -0.167 | 0.0236 | -4.95 | 10 | 0.842 | True |
| opt_months_h_months | opt_months | 0.37 | 0.292 | 0.459 | -4.29 | 7 | 2.34 | True |
| labels_AB | canonical | 0.125 | 0.0796 | 0.189 | -5.63 | 10 | 1.33 | True |
| labels_12 | canonical | -0.677 | -0.843 | -0.539 | -4.22 | 10 | 0.21 | True |
| constraint_first | canonical | -0.21 | -0.335 | -0.103 | -5.61 | 10 | 0.617 | True |
| markdown | canonical | -0.18 | -0.245 | -0.126 | -5.03 | 10 | 0.661 | True |
| prose | canonical | 0.00716 | -0.0814 | 0.122 | -5.09 | 10 | 1.02 | True |
| task_brief | canonical | 0.306 | 0.17 | 0.47 | -4.53 | 10 | 2.02 | True |
| user_request | canonical | -0.0792 | -0.2 | 0.0365 | -4.61 | 10 | 0.833 | True |
| hurry_1 | canonical | -0.01 | -0.0555 | 0.0293 | -5.43 | 10 | 0.977 | True |
| hurry_2 | canonical | 0.00837 | -0.0348 | 0.0499 | -5.41 | 10 | 1.02 | True |
| hurry_3 | canonical | -0.0505 | -0.085 | -0.0197 | -5.25 | 10 | 0.89 | True |
| relax_1 | canonical | 0.0198 | -0.0141 | 0.047 | -5.23 | 10 | 1.05 | True |
| relax_2 | canonical | 0.129 | 0.0841 | 0.175 | -5.2 | 10 | 1.35 | True |
| relax_3 | canonical | 0.3 | 0.219 | 0.396 | -4.68 | 10 | 1.99 | True |
| sooner_1 | canonical | -0.994 | -1.29 | -0.739 | -3.99 | 10 | 0.101 | True |
| later_1 | canonical | 0.363 | 0.188 | 0.545 | -5.54 | 10 | 2.31 | True |
| neutral_1 | canonical | -0.0114 | -0.0393 | 0.00914 | -5.29 | 10 | 0.974 | True |
| neutral_2 | canonical | 0.00407 | -0.0189 | 0.0226 | -5.3 | 10 | 1.01 | True |

### Contrasts

| contrast | a | b | n | mean_diff | lo | hi |
|---|---|---|---|---|---|---|
| pressure cue (hurry - relax), all phrasings, canonical layout | hurry | relax | 320 | 0.0534 | 0.0314 | 0.0765 |
| pressure cue (hurry - relax), no horizon | hurry | relax | 32 | 0.0104 | -0.031 | 0.0525 |
| pressure cue (hurry - relax), layout constraint_first | hurry | relax | 320 | -0.00384 | -0.0126 | 0.00381 |
| pressure cue (hurry - relax), layout markdown | hurry | relax | 320 | 0.00624 | 0.000253 | 0.0114 |
| pressure cue (hurry - relax), layout prose | hurry | relax | 320 | 0.00624 | -0.00902 | 0.0204 |
| pressure cue (hurry - relax), layout task_brief | hurry | relax | 320 | 0.0118 | -0.00644 | 0.0287 |
| pressure cue (hurry - relax), layout user_request | hurry | relax | 320 | -0.00501 | -0.017 | 0.00619 |
| agreement cue (sooner - later), all phrasings, canonical layout | sooner | later | 320 | 0.283 | 0.203 | 0.362 |
| agreement cue (sooner - later), no horizon | sooner | later | 32 | 0.889 | 0.802 | 0.967 |
| agreement cue (sooner - later), layout constraint_first | sooner | later | 320 | 0.162 | 0.135 | 0.19 |
| agreement cue (sooner - later), layout markdown | sooner | later | 320 | 0.244 | 0.168 | 0.32 |
| agreement cue (sooner - later), layout prose | sooner | later | 320 | 0.465 | 0.392 | 0.541 |
| agreement cue (sooner - later), layout task_brief | sooner | later | 320 | 0.59 | 0.489 | 0.69 |
| agreement cue (sooner - later), layout user_request | sooner | later | 320 | 0.266 | 0.211 | 0.324 |
| option order (short first - short second), all variants |  |  | 12432 | 0.157 | 0.0613 | 0.248 |
| unit-match interaction (h months: with month options - with mixed options) |  |  | 224 | -0.0281 | -0.0727 | 0.0191 |

### Layout cross: the same variant's Δ P(short) in each layout (vs that layout's canonical)

| base_variant | layout | n | mean_delta | lo | hi |
|---|---|---|---|---|---|
| hurry_1 | constraint_first | 320 | 0.00497 | 0.000328 | 0.00899 |
| hurry_1 | formatted | 320 | 0.00858 | -0.0103 | 0.0231 |
| hurry_1 | markdown | 320 | -0.00499 | -0.02 | 0.00898 |
| hurry_1 | prose | 320 | 0.00129 | -0.022 | 0.0243 |
| hurry_1 | task_brief | 320 | 0.0526 | 0.033 | 0.0732 |
| hurry_1 | user_request | 320 | -0.0102 | -0.0303 | 0.00911 |
| later_1 | constraint_first | 320 | -0.0628 | -0.0817 | -0.0468 |
| later_1 | formatted | 320 | -0.0653 | -0.122 | -0.0147 |
| later_1 | markdown | 320 | -0.0679 | -0.115 | -0.0275 |
| later_1 | prose | 320 | -0.222 | -0.268 | -0.176 |
| later_1 | task_brief | 320 | -0.285 | -0.365 | -0.208 |
| later_1 | user_request | 320 | -0.0753 | -0.137 | -0.0187 |
| named | constraint_first | 320 | -0.0575 | -0.0723 | -0.042 |
| named | formatted | 320 | -0.0416 | -0.0567 | -0.0276 |
| named | markdown | 320 | -0.0217 | -0.0411 | -0.00419 |
| named | prose | 320 | -0.0481 | -0.0648 | -0.0312 |
| named | task_brief | 320 | -0.0582 | -0.0834 | -0.0362 |
| named | user_request | 320 | -0.0205 | -0.0378 | -0.00554 |
| neutral_1 | constraint_first | 320 | -0.004 | -0.0101 | 0.000959 |
| neutral_1 | formatted | 320 | 0.00669 | -0.00103 | 0.0144 |
| neutral_1 | markdown | 320 | -0.0139 | -0.0294 | 0.000388 |
| neutral_1 | prose | 320 | -0.0222 | -0.0388 | -0.00614 |
| neutral_1 | task_brief | 320 | 0.0263 | 0.0104 | 0.0449 |
| neutral_1 | user_request | 320 | -0.00818 | -0.0236 | 0.00568 |
| relax_1 | constraint_first | 320 | 0.0088 | -0.000533 | 0.0192 |
| relax_1 | formatted | 320 | -0.00877 | -0.02 | 0.002 |
| relax_1 | markdown | 320 | -0.0112 | -0.026 | 0.00174 |
| relax_1 | prose | 320 | -0.00496 | -0.0325 | 0.023 |
| relax_1 | task_brief | 320 | 0.0408 | 0.0154 | 0.0703 |
| relax_1 | user_request | 320 | -0.00519 | -0.0274 | 0.0155 |
| sooner_1 | constraint_first | 320 | 0.0996 | 0.0773 | 0.121 |
| sooner_1 | formatted | 320 | 0.218 | 0.172 | 0.265 |
| sooner_1 | markdown | 320 | 0.176 | 0.125 | 0.231 |
| sooner_1 | prose | 320 | 0.242 | 0.182 | 0.304 |
| sooner_1 | task_brief | 320 | 0.305 | 0.256 | 0.359 |
| sooner_1 | user_request | 320 | 0.191 | 0.159 | 0.224 |
| unit_days | constraint_first | 320 | -0.118 | -0.144 | -0.0929 |
| unit_days | formatted | 320 | -0.05 | -0.0875 | -0.0162 |
| unit_days | markdown | 320 | -0.0243 | -0.0555 | 0.00538 |
| unit_days | prose | 320 | -0.0464 | -0.0816 | -0.016 |
| unit_days | task_brief | 320 | -0.0101 | -0.0371 | 0.0155 |
| unit_days | user_request | 320 | 0.0106 | -0.0177 | 0.0381 |
| unit_months | constraint_first | 224 | -0.105 | -0.168 | -0.0316 |
| unit_months | formatted | 224 | -0.0471 | -0.0977 | 0.00712 |
| unit_months | markdown | 224 | -0.0311 | -0.0901 | 0.029 |
| unit_months | prose | 224 | -0.0512 | -0.106 | 0.00572 |
| unit_months | task_brief | 224 | 0.0124 | -0.0257 | 0.0535 |
| unit_months | user_request | 224 | 0.00508 | -0.0347 | 0.0473 |

### Implicit vs explicit twin (Δ P(short)), by layout, determinacy and hidden number

| layout | determinacy | with_number | items | n | mean_delta | lo | hi | mean_abs_delta |
|---|---|---|---|---|---|---|---|---|
| formatted | concrete | False | 4 | 64 | -0.222 | -0.268 | -0.187 | 0.248 |
| formatted | concrete | True | 6 | 96 | -0.143 | -0.212 | -0.065 | 0.309 |
| formatted | vague | False | 7 | 112 | -0.0874 | -0.146 | -0.0249 | 0.254 |
| formatted | vague | True | 1 | 16 | -0.282 | -0.44 | -0.109 | 0.34 |
| user_request | concrete | False | 4 | 64 | -0.0683 | -0.12 | -0.0198 | 0.0949 |
| user_request | concrete | True | 6 | 96 | -0.139 | -0.216 | -0.0633 | 0.254 |
| user_request | vague | False | 7 | 112 | -0.113 | -0.166 | -0.0638 | 0.13 |
| user_request | vague | True | 1 | 16 | -0.218 | -0.343 | -0.0914 | 0.225 |
