# Phrasing study: qwen3-32b_choice_r2

Model `Qwen/Qwen3-32B` (rev 9216db5781bf21249d130ec9da846c4624c16137), commit d642a784d0cdf3f7cde8c8a76bbcd4c186059581, GPU NVIDIA A100 80GB PCIe. Choice readout separator ' **', summed over [' **', ' '].

## Choice protocol

24864 items. Free-generation check: 94% follow the format, readout agrees on 99%. Readout mass on the two labels: median 0.994, min 0.011; mass_bold median 0.966; mass_space median 0.019.

### Validity gate (canonical curve)

| spearman_logh_vs_p | p_short_lowest | p_short_highest | order_effect | null_floor | n_rho_cells |
|---|---|---|---|---|---|
| -0.724 | 0.998 | 0.41 | 0.0214 | 0.0201 | 320 |

| level | years | True | False | mean |
|---|---|---|---|---|
| 1 week | 0.0192 | 0.997 | 0.998 | 0.998 |
| 1 month | 0.0833 | 0.999 | 0.999 | 0.999 |
| 3 months | 0.25 | 0.999 | 0.999 | 0.999 |
| 6 months | 0.5 | 0.999 | 0.997 | 0.998 |
| 1 year | 1 | 0.998 | 0.919 | 0.959 |
| 2 years | 2 | 0.757 | 0.776 | 0.767 |
| 5 years | 5 | 0.616 | 0.575 | 0.596 |
| 10 years | 10 | 0.442 | 0.421 | 0.431 |
| 20 years | 20 | 0.279 | 0.313 | 0.296 |
| 50 years | 50 | 0.473 | 0.346 | 0.41 |

### Variants: Δ P(short) vs reference (95% CI over scenarios). `matters` = above the null floor, CI excludes 0, |Δ| ≥ 0.05

| family | variant | ref | n | mean_delta | lo | hi | mean_abs_delta | meaning_shift | matters |
|---|---|---|---|---|---|---|---|---|---|
| cue | later_1 | canonical | 320 | -0.182 | -0.224 | -0.141 | 0.183 | False | True |
| cue | relax_3 | canonical | 320 | -0.12 | -0.162 | -0.0774 | 0.16 | False | True |
| cue | relax_2 | canonical | 320 | -0.0107 | -0.03 | 0.00736 | 0.0535 | False | False |
| cue | neutral_2 | canonical | 320 | 0.00129 | -0.0128 | 0.0147 | 0.0415 | False | False |
| cue | neutral_1 | canonical | 320 | 0.00529 | -0.00946 | 0.0191 | 0.0413 | False | False |
| cue | hurry_1 | canonical | 320 | 0.0089 | -0.00269 | 0.0206 | 0.0398 | False | False |
| cue | hurry_2 | canonical | 320 | 0.0131 | 0.00391 | 0.0237 | 0.0337 | False | False |
| cue | relax_1 | canonical | 320 | 0.0136 | 0.003 | 0.0241 | 0.033 | False | False |
| cue | hurry_3 | canonical | 320 | 0.014 | -0.000261 | 0.0279 | 0.0374 | False | False |
| cue | sooner_1 | canonical | 320 | 0.144 | 0.101 | 0.187 | 0.147 | False | True |
| implicit | implicit:toddler_school | twin:toddler_school | 16 | -0.47 | -0.697 | -0.251 | 0.499 | False | True |
| implicit | implicit:busy_season | twin:busy_season | 16 | -0.44 | -0.658 | -0.236 | 0.44 | False | True |
| implicit | implicit:university_8 | twin:university_8 | 16 | -0.395 | -0.582 | -0.21 | 0.411 | False | True |
| implicit | implicit:retire_60_65 | twin:retire_60_65 | 16 | -0.377 | -0.427 | -0.314 | 0.377 | False | True |
| implicit | implicit:friday_monday | twin:friday_monday | 16 | -0.305 | -0.483 | -0.128 | 0.306 | False | True |
| implicit | implicit:newborn_school | twin:newborn_school | 16 | -0.274 | -0.551 | -0.0162 | 0.386 | False | True |
| implicit | implicit:busy_season@user_request | twin:busy_season@user_request | 16 | -0.244 | -0.303 | -0.179 | 0.244 | False | True |
| implicit | implicit:final_school_year@user_request | twin:final_school_year@user_request | 16 | -0.238 | -0.325 | -0.158 | 0.238 | False | True |
| implicit | implicit:baby_8wk@user_request | twin:baby_8wk@user_request | 16 | -0.225 | -0.286 | -0.163 | 0.225 | False | True |
| implicit | implicit:baby_new@user_request | twin:baby_new@user_request | 16 | -0.224 | -0.292 | -0.158 | 0.224 | False | True |
| implicit | implicit:toddler_school@user_request | twin:toddler_school@user_request | 16 | -0.215 | -0.348 | -0.0888 | 0.253 | False | True |
| implicit | implicit:baby_new | twin:baby_new | 16 | -0.178 | -0.342 | -0.0571 | 0.178 | False | True |
| implicit | implicit:baby_8wk | twin:baby_8wk | 16 | -0.174 | -0.354 | -0.052 | 0.174 | False | True |
| implicit | implicit:final_school_year | twin:final_school_year | 16 | -0.144 | -0.267 | -0.0448 | 0.144 | False | True |
| implicit | implicit:newborn_adult | twin:newborn_adult | 16 | -0.12 | -0.245 | 0.0125 | 0.271 | False | False |
| implicit | implicit:wedding_spring@user_request | twin:wedding_spring@user_request | 16 | -0.117 | -0.21 | -0.0414 | 0.118 | False | True |
| implicit | implicit:birthday_40@user_request | twin:birthday_40@user_request | 16 | -0.108 | -0.257 | 0.0204 | 0.188 | False | False |
| implicit | implicit:university_8@user_request | twin:university_8@user_request | 16 | -0.108 | -0.282 | 0.0439 | 0.211 | False | False |
| implicit | implicit:week_end | twin:week_end | 16 | -0.0947 | -0.176 | -0.0251 | 0.0947 | False | True |
| implicit | implicit:wedding_spring | twin:wedding_spring | 16 | -0.0809 | -0.14 | -0.0254 | 0.0809 | False | True |
| implicit | implicit:birthday_40 | twin:birthday_40 | 16 | -0.0808 | -0.306 | 0.131 | 0.282 | False | False |
| implicit | implicit:friday_monday@user_request | twin:friday_monday@user_request | 16 | -0.0774 | -0.101 | -0.0506 | 0.078 | False | True |
| implicit | implicit:retire_60_65@user_request | twin:retire_60_65@user_request | 16 | -0.0652 | -0.194 | 0.0676 | 0.209 | False | False |
| implicit | implicit:mortgage_30 | twin:mortgage_30 | 16 | -0.0622 | -0.16 | 0.0377 | 0.18 | False | False |
| implicit | implicit:kids_grown | twin:kids_grown | 16 | -0.0556 | -0.274 | 0.143 | 0.302 | False | False |
| implicit | implicit:year_end_oct | twin:year_end_oct | 16 | -0.0544 | -0.141 | -0.00667 | 0.0621 | False | True |
| implicit | implicit:kids_grown@user_request | twin:kids_grown@user_request | 16 | -0.0442 | -0.198 | 0.0873 | 0.197 | False | False |
| implicit | implicit:newborn_adult@user_request | twin:newborn_adult@user_request | 16 | -0.0398 | -0.111 | 0.0252 | 0.108 | False | False |
| implicit | implicit:year_end_oct@user_request | twin:year_end_oct@user_request | 16 | -0.0386 | -0.064 | -0.0151 | 0.0558 | False | False |
| implicit | implicit:lease@user_request | twin:lease@user_request | 16 | -0.0384 | -0.079 | 0.00135 | 0.0852 | False | False |
| implicit | implicit:week_end@user_request | twin:week_end@user_request | 16 | -0.0376 | -0.0597 | -0.0177 | 0.0536 | False | False |
| implicit | implicit:newborn_school@user_request | twin:newborn_school@user_request | 16 | -0.0201 | -0.172 | 0.132 | 0.202 | False | False |
| implicit | implicit:lease | twin:lease | 16 | 0.0218 | -0.00314 | 0.0605 | 0.0267 | False | False |
| implicit | implicit:first_job@user_request | twin:first_job@user_request | 16 | 0.112 | 0.0627 | 0.162 | 0.118 | False | True |
| implicit | implicit:mortgage_30@user_request | twin:mortgage_30@user_request | 16 | 0.166 | 0.125 | 0.207 | 0.166 | False | True |
| implicit | implicit:first_job | twin:first_job | 16 | 0.239 | 0.144 | 0.353 | 0.24 | False | True |
| labels | labels_AB | canonical | 320 | -0.00769 | -0.0291 | 0.0114 | 0.049 | False | False |
| labels | labels_12 | canonical | 320 | 0.00354 | -0.0228 | 0.0292 | 0.107 | False | False |
| layout | prose | canonical | 320 | -0.0337 | -0.064 | -0.00246 | 0.105 | False | False |
| layout | constraint_first | canonical | 320 | -0.0129 | -0.0461 | 0.0213 | 0.12 | False | False |
| layout | user_request | canonical | 320 | -0.0116 | -0.0414 | 0.0189 | 0.11 | False | False |
| layout | markdown | canonical | 320 | -0.00204 | -0.0211 | 0.0167 | 0.0415 | False | False |
| layout | task_brief | canonical | 320 | 0.0258 | -0.0139 | 0.0697 | 0.113 | False | False |
| layout_cross | task_brief+later_1 | task_brief | 320 | -0.43 | -0.511 | -0.348 | 0.43 | False | True |
| layout_cross | user_request+later_1 | user_request | 320 | -0.409 | -0.442 | -0.374 | 0.409 | False | True |
| layout_cross | prose+later_1 | prose | 320 | -0.211 | -0.272 | -0.157 | 0.213 | False | True |
| layout_cross | markdown+later_1 | markdown | 320 | -0.163 | -0.209 | -0.118 | 0.166 | False | True |
| layout_cross | constraint_first+later_1 | constraint_first | 320 | -0.14 | -0.16 | -0.119 | 0.14 | False | True |
| layout_cross | constraint_first+unit_months | constraint_first | 224 | -0.0894 | -0.131 | -0.0499 | 0.145 | False | True |
| layout_cross | constraint_first+unit_days | constraint_first | 320 | -0.0764 | -0.107 | -0.0475 | 0.105 | False | True |
| layout_cross | markdown+named | markdown | 320 | -0.0371 | -0.0668 | -0.006 | 0.0969 | False | False |
| layout_cross | prose+named | prose | 320 | -0.037 | -0.0568 | -0.0184 | 0.0686 | False | False |
| layout_cross | constraint_first+named | constraint_first | 320 | -0.0336 | -0.0499 | -0.018 | 0.0502 | False | False |
| layout_cross | user_request+named | user_request | 320 | -0.0285 | -0.0416 | -0.0151 | 0.0684 | False | False |
| layout_cross | task_brief+named | task_brief | 320 | -0.0274 | -0.0543 | 0.00417 | 0.0752 | False | False |
| layout_cross | task_brief+unit_days | task_brief | 320 | -0.0172 | -0.0522 | 0.0199 | 0.0897 | False | False |
| layout_cross | constraint_first+neutral_1 | constraint_first | 320 | -0.0142 | -0.0239 | -0.00506 | 0.0246 | False | False |
| layout_cross | task_brief+unit_months | task_brief | 224 | -0.0117 | -0.0403 | 0.0209 | 0.0747 | False | False |
| layout_cross | task_brief+relax_1 | task_brief | 320 | -0.00828 | -0.0114 | -0.00539 | 0.0173 | False | False |
| layout_cross | constraint_first+hurry_1 | constraint_first | 320 | -0.00755 | -0.025 | 0.0105 | 0.0338 | False | False |
| layout_cross | user_request+hurry_1 | user_request | 320 | -0.0049 | -0.0188 | 0.00923 | 0.04 | False | False |
| layout_cross | markdown+hurry_1 | markdown | 320 | -0.00378 | -0.0262 | 0.0165 | 0.0571 | False | False |
| layout_cross | task_brief+hurry_1 | task_brief | 320 | 0.000945 | -0.00288 | 0.00473 | 0.0145 | False | False |
| layout_cross | constraint_first+relax_1 | constraint_first | 320 | 0.00433 | -0.0033 | 0.0104 | 0.0224 | False | False |
| layout_cross | task_brief+neutral_1 | task_brief | 320 | 0.00638 | 0.00232 | 0.0105 | 0.0143 | False | False |
| layout_cross | markdown+neutral_1 | markdown | 320 | 0.00775 | -0.00504 | 0.0205 | 0.0406 | False | False |
| layout_cross | markdown+relax_1 | markdown | 320 | 0.0131 | 0.00294 | 0.0242 | 0.0377 | False | False |
| layout_cross | user_request+unit_days | user_request | 320 | 0.0137 | -0.0121 | 0.0376 | 0.0778 | False | False |
| layout_cross | prose+neutral_1 | prose | 320 | 0.0162 | 0.000464 | 0.0326 | 0.0521 | False | False |
| layout_cross | user_request+unit_months | user_request | 224 | 0.0212 | -0.00742 | 0.0537 | 0.098 | False | False |
| layout_cross | prose+hurry_1 | prose | 320 | 0.0253 | 0.00522 | 0.0449 | 0.0675 | False | False |
| layout_cross | user_request+relax_1 | user_request | 320 | 0.0409 | 0.0232 | 0.0601 | 0.0601 | False | False |
| layout_cross | markdown+unit_months | markdown | 224 | 0.0485 | 0.0102 | 0.0872 | 0.143 | False | False |
| layout_cross | user_request+neutral_1 | user_request | 320 | 0.0549 | 0.041 | 0.0718 | 0.0573 | False | True |
| layout_cross | markdown+unit_days | markdown | 320 | 0.056 | 0.0137 | 0.097 | 0.131 | False | True |
| layout_cross | prose+relax_1 | prose | 320 | 0.0577 | 0.0358 | 0.0812 | 0.0692 | False | True |
| layout_cross | prose+unit_days | prose | 320 | 0.0729 | 0.0353 | 0.112 | 0.129 | False | True |
| layout_cross | prose+unit_months | prose | 224 | 0.0859 | 0.0501 | 0.124 | 0.137 | False | True |
| layout_cross | markdown+sooner_1 | markdown | 320 | 0.143 | 0.0962 | 0.192 | 0.153 | False | True |
| layout_cross | constraint_first+sooner_1 | constraint_first | 320 | 0.16 | 0.123 | 0.197 | 0.16 | False | True |
| layout_cross | task_brief+sooner_1 | task_brief | 320 | 0.196 | 0.164 | 0.23 | 0.196 | False | True |
| layout_cross | prose+sooner_1 | prose | 320 | 0.205 | 0.167 | 0.247 | 0.207 | False | True |
| layout_cross | user_request+sooner_1 | user_request | 320 | 0.234 | 0.198 | 0.269 | 0.234 | False | True |
| no_horizon | nohorizon_later_1 | nohorizon_none | 32 | -0.374 | -0.488 | -0.258 | 0.425 | False | True |
| no_horizon | nohorizon_relax_1 | nohorizon_none | 32 | -0.0172 | -0.032 | -0.00371 | 0.0298 | False | False |
| no_horizon | nohorizon_hurry_1 | nohorizon_none | 32 | 0.0639 | 0.0158 | 0.123 | 0.0851 | False | True |
| no_horizon | nohorizon_sooner_1 | nohorizon_none | 32 | 0.541 | 0.432 | 0.649 | 0.541 | False | True |
| nulls | null_carefully | canonical | 320 | -0.000121 | -0.00797 | 0.00763 | 0.0223 | False | False |
| nulls | null_thanks | canonical | 320 | 0.00603 | -0.00329 | 0.0156 | 0.0327 | False | False |
| nulls | null_no_period | canonical | 320 | 0.013 | 0.00166 | 0.0276 | 0.0278 | False | False |
| nulls | null_largest | canonical | 320 | 0.0136 | 0.000882 | 0.0282 | 0.03 | False | False |
| nulls | null_please | canonical | 320 | 0.0142 | -0.00523 | 0.0376 | 0.0451 | False | False |
| nulls | null_gives | canonical | 320 | 0.0201 | 0.00181 | 0.0416 | 0.0406 | False | False |
| option_unit | opt_months | canonical | 320 | -0.045 | -0.0709 | -0.0163 | 0.114 | False | False |
| option_unit | opt_months_h_months | opt_months | 224 | 0.0347 | 0.000819 | 0.0698 | 0.119 | False | False |
| reword | reword_planning_horizon | canonical | 320 | -0.0329 | -0.06 | -0.00784 | 0.0782 | False | False |
| reword | reword_time_frame | canonical | 320 | -0.00295 | -0.0644 | 0.0564 | 0.133 | False | False |
| reword | reword_within_next | canonical | 320 | 0.0419 | 0.0106 | 0.0732 | 0.0994 | False | False |
| reword | reword_only_within | canonical | 320 | 0.0482 | -0.0143 | 0.106 | 0.139 | True | False |
| unit | unit_decades | canonical | 128 | -0.0786 | -0.159 | -0.00777 | 0.196 | False | True |
| unit | named | canonical | 320 | -0.0362 | -0.0619 | -0.0101 | 0.0886 | False | False |
| unit | unit_years | canonical | 128 | -0.0113 | -0.0241 | -0.00438 | 0.0113 | False | False |
| unit | words | canonical | 320 | -0.00193 | -0.0133 | 0.00922 | 0.0428 | False | False |
| unit | unit_months | canonical | 224 | 0.0523 | 0.0139 | 0.092 | 0.137 | False | True |
| unit | unit_weeks | canonical | 288 | 0.0722 | 0.0323 | 0.113 | 0.154 | False | True |
| unit | unit_days | canonical | 320 | 0.0762 | 0.0382 | 0.111 | 0.124 | False | True |

### Effective-horizon shift (log10 years; multiplier = 10^shift; < 1 acts like a shorter horizon)

| variant | ref | shift_log10 | lo | hi | slope | n_levels | multiplier | reliable |
|---|---|---|---|---|---|---|---|---|
| null_no_period | canonical | -0.0412 | -0.0971 | 0.00959 | -3.63 | 10 | 0.91 | True |
| null_largest | canonical | -0.0615 | -0.11 | -0.0156 | -3.69 | 10 | 0.868 | True |
| null_gives | canonical | -0.046 | -0.0876 | -0.000797 | -3.61 | 10 | 0.899 | True |
| null_carefully | canonical | 0.00672 | -0.0187 | 0.0297 | -3.84 | 10 | 1.02 | True |
| null_thanks | canonical | -0.0233 | -0.0599 | 0.00859 | -3.77 | 10 | 0.948 | True |
| null_please | canonical | -0.0637 | -0.133 | -0.000509 | -3.67 | 10 | 0.864 | True |
| reword_planning_horizon | canonical | 0.0141 | -0.0716 | 0.124 | -3.87 | 10 | 1.03 | True |
| reword_time_frame | canonical | 0.29 | 0.0685 | 0.573 | -3.08 | 10 | 1.95 | True |
| reword_within_next | canonical | -0.115 | -0.222 | -0.0069 | -3.63 | 10 | 0.768 | True |
| reword_only_within | canonical | 0.0292 | -0.192 | 0.252 | -3.28 | 10 | 1.07 | True |
| unit_days | canonical | -0.291 | -0.394 | -0.19 | -3.36 | 10 | 0.511 | True |
| unit_weeks | canonical | -0.154 | -0.236 | -0.0752 | -4.07 | 9 | 0.701 | True |
| unit_months | canonical | -0.277 | -0.454 | -0.112 | -2.71 | 7 | 0.528 | True |
| unit_years | canonical | -3.12 | -6.07 | -2.11 | 0.771 | 4 | 0.000759 | False |
| unit_decades | canonical | 0.278 | -0.0182 | 0.666 | -2.87 | 4 | 1.9 | True |
| words | canonical | 0.0589 | 0.0236 | 0.0997 | -3.64 | 10 | 1.15 | True |
| named | canonical | 0.233 | 0.15 | 0.302 | -3.68 | 10 | 1.71 | True |
| opt_months | canonical | 0.154 | -0.0117 | 0.285 | -3.88 | 10 | 1.42 | True |
| opt_months_h_months | opt_months | 0.145 | 0.0321 | 0.268 | -2.96 | 7 | 1.4 | True |
| labels_AB | canonical | 0.228 | 0.148 | 0.328 | -3.53 | 10 | 1.69 | True |
| labels_12 | canonical | 0.11 | -0.00374 | 0.242 | -3.56 | 10 | 1.29 | True |
| constraint_first | canonical | -0.0638 | -0.181 | 0.0537 | -3.97 | 10 | 0.863 | True |
| markdown | canonical | 0.0617 | 0.0179 | 0.114 | -3.69 | 10 | 1.15 | True |
| prose | canonical | 0.413 | 0.278 | 0.557 | -3.5 | 10 | 2.59 | True |
| task_brief | canonical | 0.384 | 0.194 | 0.566 | -3.05 | 10 | 2.42 | True |
| user_request | canonical | 0.665 | 0.47 | 0.872 | -2.84 | 10 | 4.62 | True |
| hurry_1 | canonical | -0.0193 | -0.0551 | 0.0206 | -3.82 | 10 | 0.957 | True |
| hurry_2 | canonical | -0.0698 | -0.13 | -0.0145 | -3.78 | 10 | 0.852 | True |
| hurry_3 | canonical | -0.0652 | -0.122 | -0.00836 | -3.75 | 10 | 0.861 | True |
| relax_1 | canonical | 0.0202 | -0.0372 | 0.0793 | -3.56 | 10 | 1.05 | True |
| relax_2 | canonical | 0.196 | 0.121 | 0.275 | -3.48 | 10 | 1.57 | True |
| relax_3 | canonical | 1.08 | 0.922 | 1.25 | -3.03 | 10 | 11.9 | True |
| sooner_1 | canonical | -0.852 | -1.01 | -0.702 | -3.32 | 10 | 0.141 | True |
| later_1 | canonical | 1.27 | 1.07 | 1.51 | -3.58 | 10 | 18.8 | True |
| neutral_1 | canonical | 0.00343 | -0.0398 | 0.039 | -3.73 | 10 | 1.01 | True |
| neutral_2 | canonical | 0.0389 | -0.000884 | 0.0769 | -3.71 | 10 | 1.09 | True |

### Contrasts

| contrast | a | b | n | mean_diff | lo | hi |
|---|---|---|---|---|---|---|
| pressure cue (hurry - relax), all phrasings, canonical layout | hurry | relax | 320 | 0.051 | 0.0243 | 0.0787 |
| pressure cue (hurry - relax), no horizon | hurry | relax | 32 | 0.0811 | 0.0367 | 0.132 |
| pressure cue (hurry - relax), layout constraint_first | hurry | relax | 320 | -0.0119 | -0.0284 | 0.00541 |
| pressure cue (hurry - relax), layout markdown | hurry | relax | 320 | -0.0169 | -0.0422 | 0.00557 |
| pressure cue (hurry - relax), layout prose | hurry | relax | 320 | -0.0323 | -0.0623 | -0.00602 |
| pressure cue (hurry - relax), layout task_brief | hurry | relax | 320 | 0.00922 | 0.00447 | 0.0139 |
| pressure cue (hurry - relax), layout user_request | hurry | relax | 320 | -0.0458 | -0.0598 | -0.0325 |
| agreement cue (sooner - later), all phrasings, canonical layout | sooner | later | 320 | 0.326 | 0.261 | 0.387 |
| agreement cue (sooner - later), no horizon | sooner | later | 32 | 0.915 | 0.879 | 0.948 |
| agreement cue (sooner - later), layout constraint_first | sooner | later | 320 | 0.299 | 0.263 | 0.336 |
| agreement cue (sooner - later), layout markdown | sooner | later | 320 | 0.306 | 0.243 | 0.367 |
| agreement cue (sooner - later), layout prose | sooner | later | 320 | 0.416 | 0.348 | 0.483 |
| agreement cue (sooner - later), layout task_brief | sooner | later | 320 | 0.626 | 0.547 | 0.704 |
| agreement cue (sooner - later), layout user_request | sooner | later | 320 | 0.643 | 0.605 | 0.682 |
| option order (short first - short second), all variants |  |  | 12432 | 0.01 | -0.0635 | 0.0862 |
| unit-match interaction (h months: with month options - with mixed options) |  |  | 224 | -0.0176 | -0.0681 | 0.0322 |

### Layout cross: the same variant's Δ P(short) in each layout (vs that layout's canonical)

| base_variant | layout | n | mean_delta | lo | hi |
|---|---|---|---|---|---|
| hurry_1 | constraint_first | 320 | -0.00755 | -0.025 | 0.0105 |
| hurry_1 | formatted | 320 | 0.0089 | -0.00269 | 0.0206 |
| hurry_1 | markdown | 320 | -0.00378 | -0.0262 | 0.0165 |
| hurry_1 | prose | 320 | 0.0253 | 0.00522 | 0.0449 |
| hurry_1 | task_brief | 320 | 0.000945 | -0.00288 | 0.00473 |
| hurry_1 | user_request | 320 | -0.0049 | -0.0188 | 0.00923 |
| later_1 | constraint_first | 320 | -0.14 | -0.16 | -0.119 |
| later_1 | formatted | 320 | -0.182 | -0.224 | -0.141 |
| later_1 | markdown | 320 | -0.163 | -0.209 | -0.118 |
| later_1 | prose | 320 | -0.211 | -0.272 | -0.157 |
| later_1 | task_brief | 320 | -0.43 | -0.511 | -0.348 |
| later_1 | user_request | 320 | -0.409 | -0.442 | -0.374 |
| named | constraint_first | 320 | -0.0336 | -0.0499 | -0.018 |
| named | formatted | 320 | -0.0362 | -0.0619 | -0.0101 |
| named | markdown | 320 | -0.0371 | -0.0668 | -0.006 |
| named | prose | 320 | -0.037 | -0.0568 | -0.0184 |
| named | task_brief | 320 | -0.0274 | -0.0543 | 0.00417 |
| named | user_request | 320 | -0.0285 | -0.0416 | -0.0151 |
| neutral_1 | constraint_first | 320 | -0.0142 | -0.0239 | -0.00506 |
| neutral_1 | formatted | 320 | 0.00529 | -0.00946 | 0.0191 |
| neutral_1 | markdown | 320 | 0.00775 | -0.00504 | 0.0205 |
| neutral_1 | prose | 320 | 0.0162 | 0.000464 | 0.0326 |
| neutral_1 | task_brief | 320 | 0.00638 | 0.00232 | 0.0105 |
| neutral_1 | user_request | 320 | 0.0549 | 0.041 | 0.0718 |
| relax_1 | constraint_first | 320 | 0.00433 | -0.0033 | 0.0104 |
| relax_1 | formatted | 320 | 0.0136 | 0.003 | 0.0241 |
| relax_1 | markdown | 320 | 0.0131 | 0.00294 | 0.0242 |
| relax_1 | prose | 320 | 0.0577 | 0.0358 | 0.0812 |
| relax_1 | task_brief | 320 | -0.00828 | -0.0114 | -0.00539 |
| relax_1 | user_request | 320 | 0.0409 | 0.0232 | 0.0601 |
| sooner_1 | constraint_first | 320 | 0.16 | 0.123 | 0.197 |
| sooner_1 | formatted | 320 | 0.144 | 0.101 | 0.187 |
| sooner_1 | markdown | 320 | 0.143 | 0.0962 | 0.192 |
| sooner_1 | prose | 320 | 0.205 | 0.167 | 0.247 |
| sooner_1 | task_brief | 320 | 0.196 | 0.164 | 0.23 |
| sooner_1 | user_request | 320 | 0.234 | 0.198 | 0.269 |
| unit_days | constraint_first | 320 | -0.0764 | -0.107 | -0.0475 |
| unit_days | formatted | 320 | 0.0762 | 0.0382 | 0.111 |
| unit_days | markdown | 320 | 0.056 | 0.0137 | 0.097 |
| unit_days | prose | 320 | 0.0729 | 0.0353 | 0.112 |
| unit_days | task_brief | 320 | -0.0172 | -0.0522 | 0.0199 |
| unit_days | user_request | 320 | 0.0137 | -0.0121 | 0.0376 |
| unit_months | constraint_first | 224 | -0.0894 | -0.131 | -0.0499 |
| unit_months | formatted | 224 | 0.0523 | 0.0139 | 0.092 |
| unit_months | markdown | 224 | 0.0485 | 0.0102 | 0.0872 |
| unit_months | prose | 224 | 0.0859 | 0.0501 | 0.124 |
| unit_months | task_brief | 224 | -0.0117 | -0.0403 | 0.0209 |
| unit_months | user_request | 224 | 0.0212 | -0.00742 | 0.0537 |

### Implicit vs explicit twin (Δ P(short)), by layout, determinacy and hidden number

| layout | determinacy | with_number | items | n | mean_delta | lo | hi | mean_abs_delta |
|---|---|---|---|---|---|---|---|---|
| formatted | concrete | False | 4 | 64 | -0.164 | -0.241 | -0.0845 | 0.204 |
| formatted | concrete | True | 6 | 96 | -0.178 | -0.271 | -0.0921 | 0.242 |
| formatted | vague | False | 7 | 112 | -0.121 | -0.22 | -0.0338 | 0.241 |
| formatted | vague | True | 1 | 16 | -0.47 | -0.697 | -0.251 | 0.499 |
| user_request | concrete | False | 4 | 64 | -0.0949 | -0.128 | -0.0668 | 0.116 |
| user_request | concrete | True | 6 | 96 | -0.063 | -0.141 | 0.00689 | 0.181 |
| user_request | vague | False | 7 | 112 | -0.0841 | -0.134 | -0.0409 | 0.167 |
| user_request | vague | True | 1 | 16 | -0.215 | -0.348 | -0.0888 | 0.253 |
