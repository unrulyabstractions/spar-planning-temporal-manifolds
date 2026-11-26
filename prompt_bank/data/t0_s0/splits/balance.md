# Balance of nuisance factors

Each cell: count (share of the regime / correct-option column). `*` marks a share more than 0.10 from the factor level's overall share, in a column of at least 10 records.

| dataset | n | flags |
|---|---|---|
| t0_s0 extraction | 500 | 15 |
| t0_s0 verification | 200 | 4 |
| reserved_horizons/fit | 419 | 17 |
| reserved_horizons/test | 81 | 38 |
| c5_surface/fit | 71 | 6 |
| c5_surface/test | 198 | 24 |
| c1_balanced/fit | 80 | 8 |
| c1_balanced_with_none/fit | 80 | 4 |

## t0_s0 extraction (n = 500)

By regime

| factor level | overall | both | near_only | no_horizon | none |
|---|---|---|---|---|---|
| labels=1)/2) | 50% | 44 (51%) | 44 (42%) | 17 (59%) | 145 (52%) |
| labels=a)/b) | 50% | 42 (49%) | 62 (58%) | 12 (41%) | 134 (48%) |
| near_first=False | 53% | 51 (59%) | 58 (55%) | 14 (48%) | 144 (52%) |
| near_first=True | 47% | 35 (41%) | 48 (45%) | 15 (52%) | 135 (48%) |
| phrasing_id=0 | 23% | 25 (29%) | 21 (20%) | 0 (0%)* | 69 (25%) |
| phrasing_id=1 | 24% | 16 (19%) | 29 (27%) | 0 (0%)* | 74 (27%) |
| phrasing_id=2 | 23% | 22 (26%) | 31 (29%) | 0 (0%)* | 61 (22%) |
| phrasing_id=3 | 25% | 23 (27%) | 25 (24%) | 0 (0%)* | 75 (27%) |
| phrasing_id=none | 6% | 0 (0%) | 0 (0%) | 29 (100%)* | 0 (0%) |
| rewritten=False | 66% | 68 (79%)* | 34 (32%)* | 29 (100%)* | 197 (71%) |
| rewritten=True | 34% | 18 (21%)* | 72 (68%)* | 0 (0%)* | 82 (29%) |
| reward_multiplier=10 | 19% | 17 (20%) | 20 (19%) | 4 (14%) | 56 (20%) |
| reward_multiplier=2 | 21% | 13 (15%) | 21 (20%) | 7 (24%) | 62 (22%) |
| reward_multiplier=25 | 20% | 18 (21%) | 22 (21%) | 7 (24%) | 54 (19%) |
| reward_multiplier=5 | 22% | 19 (22%) | 20 (19%) | 7 (24%) | 63 (23%) |
| reward_multiplier=50 | 18% | 19 (22%) | 23 (22%) | 4 (14%) | 44 (16%) |

By correct_option

| factor level | overall | far | near | none |
|---|---|---|---|---|
| labels=1)/2) | 50% | 44 (51%) | 44 (42%) | 162 (53%) |
| labels=a)/b) | 50% | 42 (49%) | 62 (58%) | 146 (47%) |
| near_first=False | 53% | 51 (59%) | 58 (55%) | 158 (51%) |
| near_first=True | 47% | 35 (41%) | 48 (45%) | 150 (49%) |
| phrasing_id=0 | 23% | 25 (29%) | 21 (20%) | 69 (22%) |
| phrasing_id=1 | 24% | 16 (19%) | 29 (27%) | 74 (24%) |
| phrasing_id=2 | 23% | 22 (26%) | 31 (29%) | 61 (20%) |
| phrasing_id=3 | 25% | 23 (27%) | 25 (24%) | 75 (24%) |
| phrasing_id=none | 6% | 0 (0%) | 0 (0%) | 29 (9%) |
| rewritten=False | 66% | 68 (79%)* | 34 (32%)* | 226 (73%) |
| rewritten=True | 34% | 18 (21%)* | 72 (68%)* | 82 (27%) |
| reward_multiplier=10 | 19% | 17 (20%) | 20 (19%) | 60 (19%) |
| reward_multiplier=2 | 21% | 13 (15%) | 21 (20%) | 69 (22%) |
| reward_multiplier=25 | 20% | 18 (21%) | 22 (21%) | 61 (20%) |
| reward_multiplier=5 | 22% | 19 (22%) | 20 (19%) | 70 (23%) |
| reward_multiplier=50 | 18% | 19 (22%) | 23 (22%) | 48 (16%) |

## t0_s0 verification (n = 200)

By regime

| factor level | overall | both | near_only |
|---|---|---|---|
| labels=1)/2) | 50% | 42 (42%) | 58 (58%) |
| labels=a)/b) | 50% | 58 (58%) | 42 (42%) |
| near_first=False | 50% | 50 (50%) | 50 (50%) |
| near_first=True | 50% | 50 (50%) | 50 (50%) |
| phrasing_id=0 | 34% | 38 (38%) | 30 (30%) |
| phrasing_id=1 | 26% | 24 (24%) | 28 (28%) |
| phrasing_id=2 | 19% | 18 (18%) | 20 (20%) |
| phrasing_id=3 | 21% | 20 (20%) | 22 (22%) |
| rewritten=False | 100% | 100 (100%) | 100 (100%) |
| reward_multiplier=10 | 22% | 22 (22%) | 22 (22%) |
| reward_multiplier=2 | 24% | 36 (36%)* | 12 (12%)* |
| reward_multiplier=25 | 20% | 10 (10%) | 30 (30%) |
| reward_multiplier=5 | 16% | 16 (16%) | 16 (16%) |
| reward_multiplier=50 | 18% | 16 (16%) | 20 (20%) |

By correct_option

| factor level | overall | far | near |
|---|---|---|---|
| labels=1)/2) | 50% | 42 (42%) | 58 (58%) |
| labels=a)/b) | 50% | 58 (58%) | 42 (42%) |
| near_first=False | 50% | 50 (50%) | 50 (50%) |
| near_first=True | 50% | 50 (50%) | 50 (50%) |
| phrasing_id=0 | 34% | 38 (38%) | 30 (30%) |
| phrasing_id=1 | 26% | 24 (24%) | 28 (28%) |
| phrasing_id=2 | 19% | 18 (18%) | 20 (20%) |
| phrasing_id=3 | 21% | 20 (20%) | 22 (22%) |
| rewritten=False | 100% | 100 (100%) | 100 (100%) |
| reward_multiplier=10 | 22% | 22 (22%) | 22 (22%) |
| reward_multiplier=2 | 24% | 36 (36%)* | 12 (12%)* |
| reward_multiplier=25 | 20% | 10 (10%) | 30 (30%) |
| reward_multiplier=5 | 16% | 16 (16%) | 16 (16%) |
| reward_multiplier=50 | 18% | 16 (16%) | 20 (20%) |

## reserved_horizons/fit (n = 419)

By regime

| factor level | overall | both | near_only | no_horizon | none |
|---|---|---|---|---|---|
| labels=1)/2) | 48% | 35 (47%) | 36 (41%) | 16 (62%)* | 116 (50%) |
| labels=a)/b) | 52% | 40 (53%) | 51 (59%) | 10 (38%)* | 115 (50%) |
| near_first=False | 54% | 48 (64%) | 46 (53%) | 13 (50%) | 120 (52%) |
| near_first=True | 46% | 27 (36%) | 41 (47%) | 13 (50%) | 111 (48%) |
| phrasing_id=0 | 22% | 21 (28%) | 16 (18%) | 0 (0%)* | 57 (25%) |
| phrasing_id=1 | 24% | 14 (19%) | 25 (29%) | 0 (0%)* | 62 (27%) |
| phrasing_id=2 | 24% | 20 (27%) | 26 (30%) | 0 (0%)* | 54 (23%) |
| phrasing_id=3 | 23% | 20 (27%) | 20 (23%) | 0 (0%)* | 58 (25%) |
| phrasing_id=none | 6% | 0 (0%) | 0 (0%) | 26 (100%)* | 0 (0%) |
| rewritten=False | 66% | 58 (77%)* | 30 (34%)* | 26 (100%)* | 161 (70%) |
| rewritten=True | 34% | 17 (23%)* | 57 (66%)* | 0 (0%)* | 70 (30%) |
| reward_multiplier=10 | 20% | 15 (20%) | 17 (20%) | 4 (15%) | 47 (20%) |
| reward_multiplier=2 | 20% | 12 (16%) | 14 (16%) | 6 (23%) | 50 (22%) |
| reward_multiplier=25 | 19% | 14 (19%) | 16 (18%) | 6 (23%) | 44 (19%) |
| reward_multiplier=5 | 23% | 18 (24%) | 20 (23%) | 6 (23%) | 51 (22%) |
| reward_multiplier=50 | 19% | 16 (21%) | 20 (23%) | 4 (15%) | 39 (17%) |

By correct_option

| factor level | overall | far | near | none |
|---|---|---|---|---|
| labels=1)/2) | 48% | 35 (47%) | 36 (41%) | 132 (51%) |
| labels=a)/b) | 52% | 40 (53%) | 51 (59%) | 125 (49%) |
| near_first=False | 54% | 48 (64%) | 46 (53%) | 133 (52%) |
| near_first=True | 46% | 27 (36%) | 41 (47%) | 124 (48%) |
| phrasing_id=0 | 22% | 21 (28%) | 16 (18%) | 57 (22%) |
| phrasing_id=1 | 24% | 14 (19%) | 25 (29%) | 62 (24%) |
| phrasing_id=2 | 24% | 20 (27%) | 26 (30%) | 54 (21%) |
| phrasing_id=3 | 23% | 20 (27%) | 20 (23%) | 58 (23%) |
| phrasing_id=none | 6% | 0 (0%) | 0 (0%) | 26 (10%) |
| rewritten=False | 66% | 58 (77%)* | 30 (34%)* | 187 (73%) |
| rewritten=True | 34% | 17 (23%)* | 57 (66%)* | 70 (27%) |
| reward_multiplier=10 | 20% | 15 (20%) | 17 (20%) | 51 (20%) |
| reward_multiplier=2 | 20% | 12 (16%) | 14 (16%) | 56 (22%) |
| reward_multiplier=25 | 19% | 14 (19%) | 16 (18%) | 50 (19%) |
| reward_multiplier=5 | 23% | 18 (24%) | 20 (23%) | 57 (22%) |
| reward_multiplier=50 | 19% | 16 (21%) | 20 (23%) | 43 (17%) |

## reserved_horizons/test (n = 81)

By regime

| factor level | overall | both | near_only | no_horizon | none |
|---|---|---|---|---|---|
| labels=1)/2) | 58% | 9 (82%)* | 8 (42%)* | 1 (33%) | 29 (60%) |
| labels=a)/b) | 42% | 2 (18%)* | 11 (58%)* | 2 (67%) | 19 (40%) |
| near_first=False | 49% | 3 (27%)* | 12 (63%)* | 1 (33%) | 24 (50%) |
| near_first=True | 51% | 8 (73%)* | 7 (37%)* | 2 (67%) | 24 (50%) |
| phrasing_id=0 | 26% | 4 (36%)* | 5 (26%) | 0 (0%) | 12 (25%) |
| phrasing_id=1 | 22% | 2 (18%) | 4 (21%) | 0 (0%) | 12 (25%) |
| phrasing_id=2 | 17% | 2 (18%) | 5 (26%) | 0 (0%) | 7 (15%) |
| phrasing_id=3 | 31% | 3 (27%) | 5 (26%) | 0 (0%) | 17 (35%) |
| phrasing_id=none | 4% | 0 (0%) | 0 (0%) | 3 (100%) | 0 (0%) |
| rewritten=False | 65% | 10 (91%)* | 4 (21%)* | 3 (100%) | 36 (75%) |
| rewritten=True | 35% | 1 (9%)* | 15 (79%)* | 0 (0%) | 12 (25%) |
| reward_multiplier=10 | 17% | 2 (18%) | 3 (16%) | 0 (0%) | 9 (19%) |
| reward_multiplier=2 | 26% | 1 (9%)* | 7 (37%)* | 1 (33%) | 12 (25%) |
| reward_multiplier=25 | 26% | 4 (36%)* | 6 (32%) | 1 (33%) | 10 (21%) |
| reward_multiplier=5 | 17% | 1 (9%) | 0 (0%)* | 1 (33%) | 12 (25%) |
| reward_multiplier=50 | 14% | 3 (27%)* | 3 (16%) | 0 (0%) | 5 (10%) |

By correct_option

| factor level | overall | far | near | none |
|---|---|---|---|---|
| labels=1)/2) | 58% | 9 (82%)* | 8 (42%)* | 30 (59%) |
| labels=a)/b) | 42% | 2 (18%)* | 11 (58%)* | 21 (41%) |
| near_first=False | 49% | 3 (27%)* | 12 (63%)* | 25 (49%) |
| near_first=True | 51% | 8 (73%)* | 7 (37%)* | 26 (51%) |
| phrasing_id=0 | 26% | 4 (36%)* | 5 (26%) | 12 (24%) |
| phrasing_id=1 | 22% | 2 (18%) | 4 (21%) | 12 (24%) |
| phrasing_id=2 | 17% | 2 (18%) | 5 (26%) | 7 (14%) |
| phrasing_id=3 | 31% | 3 (27%) | 5 (26%) | 17 (33%) |
| phrasing_id=none | 4% | 0 (0%) | 0 (0%) | 3 (6%) |
| rewritten=False | 65% | 10 (91%)* | 4 (21%)* | 39 (76%)* |
| rewritten=True | 35% | 1 (9%)* | 15 (79%)* | 12 (24%)* |
| reward_multiplier=10 | 17% | 2 (18%) | 3 (16%) | 9 (18%) |
| reward_multiplier=2 | 26% | 1 (9%)* | 7 (37%)* | 13 (25%) |
| reward_multiplier=25 | 26% | 4 (36%)* | 6 (32%) | 11 (22%) |
| reward_multiplier=5 | 17% | 1 (9%) | 0 (0%)* | 13 (25%) |
| reward_multiplier=50 | 14% | 3 (27%)* | 3 (16%) | 5 (10%) |

## c5_surface/fit (n = 71)

By regime

| factor level | overall | both | near_only | none |
|---|---|---|---|---|
| labels=a)/b) | 100% | 17 (100%) | 13 (100%) | 41 (100%) |
| near_first=False | 68% | 13 (76%) | 10 (77%) | 25 (61%) |
| near_first=True | 32% | 4 (24%) | 3 (23%) | 16 (39%) |
| phrasing_id=0 | 56% | 10 (59%) | 6 (46%)* | 24 (59%) |
| phrasing_id=2 | 44% | 7 (41%) | 7 (54%)* | 17 (41%) |
| rewritten=False | 100% | 17 (100%) | 13 (100%) | 41 (100%) |
| reward_multiplier=10 | 18% | 2 (12%) | 4 (31%)* | 7 (17%) |
| reward_multiplier=2 | 17% | 3 (18%) | 2 (15%) | 7 (17%) |
| reward_multiplier=25 | 21% | 2 (12%) | 3 (23%) | 10 (24%) |
| reward_multiplier=5 | 24% | 5 (29%) | 2 (15%) | 10 (24%) |
| reward_multiplier=50 | 20% | 5 (29%) | 2 (15%) | 7 (17%) |

By correct_option

| factor level | overall | far | near | none |
|---|---|---|---|---|
| labels=a)/b) | 100% | 17 (100%) | 13 (100%) | 41 (100%) |
| near_first=False | 68% | 13 (76%) | 10 (77%) | 25 (61%) |
| near_first=True | 32% | 4 (24%) | 3 (23%) | 16 (39%) |
| phrasing_id=0 | 56% | 10 (59%) | 6 (46%)* | 24 (59%) |
| phrasing_id=2 | 44% | 7 (41%) | 7 (54%)* | 17 (41%) |
| rewritten=False | 100% | 17 (100%) | 13 (100%) | 41 (100%) |
| reward_multiplier=10 | 18% | 2 (12%) | 4 (31%)* | 7 (17%) |
| reward_multiplier=2 | 17% | 3 (18%) | 2 (15%) | 7 (17%) |
| reward_multiplier=25 | 21% | 2 (12%) | 3 (23%) | 10 (24%) |
| reward_multiplier=5 | 24% | 5 (29%) | 2 (15%) | 10 (24%) |
| reward_multiplier=50 | 20% | 5 (29%) | 2 (15%) | 7 (17%) |

## c5_surface/test (n = 198)

By regime

| factor level | overall | both | near_only | none |
|---|---|---|---|---|
| labels=1)/2) | 38% | 22 (51%)* | 3 (10%)* | 51 (41%) |
| labels=a)/b) | 62% | 21 (49%)* | 28 (90%)* | 73 (59%) |
| near_first=False | 51% | 24 (56%) | 16 (52%) | 61 (49%) |
| near_first=True | 49% | 19 (44%) | 15 (48%) | 63 (51%) |
| phrasing_id=0 | 29% | 12 (28%) | 8 (26%) | 37 (30%) |
| phrasing_id=1 | 19% | 6 (14%) | 5 (16%) | 27 (22%) |
| phrasing_id=2 | 32% | 14 (33%) | 15 (48%)* | 34 (27%) |
| phrasing_id=3 | 20% | 11 (26%) | 3 (10%)* | 26 (21%) |
| rewritten=False | 78% | 39 (91%)* | 11 (35%)* | 104 (84%) |
| rewritten=True | 22% | 4 (9%)* | 20 (65%)* | 20 (16%) |
| reward_multiplier=10 | 20% | 10 (23%) | 6 (19%) | 23 (19%) |
| reward_multiplier=2 | 24% | 7 (16%) | 7 (23%) | 33 (27%) |
| reward_multiplier=25 | 21% | 11 (26%) | 2 (6%)* | 28 (23%) |
| reward_multiplier=5 | 19% | 8 (19%) | 6 (19%) | 24 (19%) |
| reward_multiplier=50 | 17% | 7 (16%) | 10 (32%)* | 16 (13%) |

By correct_option

| factor level | overall | far | near | none |
|---|---|---|---|---|
| labels=1)/2) | 38% | 22 (51%)* | 3 (10%)* | 51 (41%) |
| labels=a)/b) | 62% | 21 (49%)* | 28 (90%)* | 73 (59%) |
| near_first=False | 51% | 24 (56%) | 16 (52%) | 61 (49%) |
| near_first=True | 49% | 19 (44%) | 15 (48%) | 63 (51%) |
| phrasing_id=0 | 29% | 12 (28%) | 8 (26%) | 37 (30%) |
| phrasing_id=1 | 19% | 6 (14%) | 5 (16%) | 27 (22%) |
| phrasing_id=2 | 32% | 14 (33%) | 15 (48%)* | 34 (27%) |
| phrasing_id=3 | 20% | 11 (26%) | 3 (10%)* | 26 (21%) |
| rewritten=False | 78% | 39 (91%)* | 11 (35%)* | 104 (84%) |
| rewritten=True | 22% | 4 (9%)* | 20 (65%)* | 20 (16%) |
| reward_multiplier=10 | 20% | 10 (23%) | 6 (19%) | 23 (19%) |
| reward_multiplier=2 | 24% | 7 (16%) | 7 (23%) | 33 (27%) |
| reward_multiplier=25 | 21% | 11 (26%) | 2 (6%)* | 28 (23%) |
| reward_multiplier=5 | 19% | 8 (19%) | 6 (19%) | 24 (19%) |
| reward_multiplier=50 | 17% | 7 (16%) | 10 (32%)* | 16 (13%) |

## c1_balanced/fit (n = 80)

By regime

| factor level | overall | both | near_only |
|---|---|---|---|
| labels=1)/2) | 50% | 20 (50%) | 20 (50%) |
| labels=a)/b) | 50% | 20 (50%) | 20 (50%) |
| near_first=False | 50% | 20 (50%) | 20 (50%) |
| near_first=True | 50% | 20 (50%) | 20 (50%) |
| phrasing_id=0 | 20% | 8 (20%) | 8 (20%) |
| phrasing_id=1 | 21% | 7 (18%) | 10 (25%) |
| phrasing_id=2 | 26% | 10 (25%) | 11 (28%) |
| phrasing_id=3 | 32% | 15 (38%) | 11 (28%) |
| rewritten=False | 59% | 34 (85%)* | 13 (32%)* |
| rewritten=True | 41% | 6 (15%)* | 27 (68%)* |
| reward_multiplier=10 | 14% | 8 (20%) | 3 (8%) |
| reward_multiplier=2 | 21% | 7 (18%) | 10 (25%) |
| reward_multiplier=25 | 20% | 7 (18%) | 9 (22%) |
| reward_multiplier=5 | 19% | 9 (22%) | 6 (15%) |
| reward_multiplier=50 | 26% | 9 (22%) | 12 (30%) |

By correct_option

| factor level | overall | far | near |
|---|---|---|---|
| labels=1)/2) | 50% | 20 (50%) | 20 (50%) |
| labels=a)/b) | 50% | 20 (50%) | 20 (50%) |
| near_first=False | 50% | 20 (50%) | 20 (50%) |
| near_first=True | 50% | 20 (50%) | 20 (50%) |
| phrasing_id=0 | 20% | 8 (20%) | 8 (20%) |
| phrasing_id=1 | 21% | 7 (18%) | 10 (25%) |
| phrasing_id=2 | 26% | 10 (25%) | 11 (28%) |
| phrasing_id=3 | 32% | 15 (38%) | 11 (28%) |
| rewritten=False | 59% | 34 (85%)* | 13 (32%)* |
| rewritten=True | 41% | 6 (15%)* | 27 (68%)* |
| reward_multiplier=10 | 14% | 8 (20%) | 3 (8%) |
| reward_multiplier=2 | 21% | 7 (18%) | 10 (25%) |
| reward_multiplier=25 | 20% | 7 (18%) | 9 (22%) |
| reward_multiplier=5 | 19% | 9 (22%) | 6 (15%) |
| reward_multiplier=50 | 26% | 9 (22%) | 12 (30%) |

## c1_balanced_with_none/fit (n = 80)

By regime

| factor level | overall | both | near_only | none |
|---|---|---|---|---|
| labels=1)/2) | 50% | 20 (50%) | 3 (43%) | 17 (52%) |
| labels=a)/b) | 50% | 20 (50%) | 4 (57%) | 16 (48%) |
| near_first=False | 50% | 20 (50%) | 4 (57%) | 16 (48%) |
| near_first=True | 50% | 20 (50%) | 3 (43%) | 17 (52%) |
| phrasing_id=0 | 25% | 10 (25%) | 0 (0%) | 10 (30%) |
| phrasing_id=1 | 22% | 8 (20%) | 3 (43%) | 7 (21%) |
| phrasing_id=2 | 30% | 12 (30%) | 3 (43%) | 9 (27%) |
| phrasing_id=3 | 22% | 10 (25%) | 1 (14%) | 7 (21%) |
| rewritten=False | 74% | 35 (88%)* | 1 (14%) | 23 (70%) |
| rewritten=True | 26% | 5 (12%)* | 6 (86%) | 10 (30%) |
| reward_multiplier=10 | 21% | 10 (25%) | 1 (14%) | 6 (18%) |
| reward_multiplier=2 | 22% | 7 (18%) | 1 (14%) | 10 (30%) |
| reward_multiplier=25 | 18% | 7 (18%) | 2 (29%) | 5 (15%) |
| reward_multiplier=5 | 19% | 6 (15%) | 1 (14%) | 8 (24%) |
| reward_multiplier=50 | 20% | 10 (25%) | 2 (29%) | 4 (12%) |

By correct_option

| factor level | overall | far | near | none |
|---|---|---|---|---|
| labels=1)/2) | 50% | 20 (50%) | 3 (43%) | 17 (52%) |
| labels=a)/b) | 50% | 20 (50%) | 4 (57%) | 16 (48%) |
| near_first=False | 50% | 20 (50%) | 4 (57%) | 16 (48%) |
| near_first=True | 50% | 20 (50%) | 3 (43%) | 17 (52%) |
| phrasing_id=0 | 25% | 10 (25%) | 0 (0%) | 10 (30%) |
| phrasing_id=1 | 22% | 8 (20%) | 3 (43%) | 7 (21%) |
| phrasing_id=2 | 30% | 12 (30%) | 3 (43%) | 9 (27%) |
| phrasing_id=3 | 22% | 10 (25%) | 1 (14%) | 7 (21%) |
| rewritten=False | 74% | 35 (88%)* | 1 (14%) | 23 (70%) |
| rewritten=True | 26% | 5 (12%)* | 6 (86%) | 10 (30%) |
| reward_multiplier=10 | 21% | 10 (25%) | 1 (14%) | 6 (18%) |
| reward_multiplier=2 | 22% | 7 (18%) | 1 (14%) | 10 (30%) |
| reward_multiplier=25 | 18% | 7 (18%) | 2 (29%) | 5 (15%) |
| reward_multiplier=5 | 19% | 6 (15%) | 1 (14%) | 8 (24%) |
| reward_multiplier=50 | 20% | 10 (25%) | 2 (29%) | 4 (12%) |
