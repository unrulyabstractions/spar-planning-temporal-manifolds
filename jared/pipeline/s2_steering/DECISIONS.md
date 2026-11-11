# Stage 2 decisions: evaluations, testing and steering (Qwen3-8B)

Extracted 2026-11-09 from the docs, nothing new decided here. Where a doc says something I could not
find a decision for, it is in section 7.

Citation keys (all relative to `jared/`; line numbers are the file's own):
`plan` = `01_plan.md`; `31` = `31_goals_next_specs.md`; `README` = `pipeline/s2_steering/README.md`;
`07, 08, 09, 18, 21, 22, 25, 27, 28, 29, 30` = `pipeline/s2_steering/docs/NN_*.md`;
`15c` = `15_spec_crossing.md`; `15s` = `15_spec_same_subspace_sweep.md`; `26s` / `26r` = `26_spec_overnight_choice_scan.md` / `26_results_overnight_choice_scan.md`;
`16s` = `pipeline/s1_data_vectors/docs/16_spec_pristine_manifold.md`; `CLAUDE` = `/home/Jared/SPAR/CLAUDE.md`.
Status tags: PRE-REG (criterion fixed before a run), DECIDED (Jared, dated), DEFAULT (in code/spec, no recorded decision),
DRAFT (not approved), SUPERSEDED/RETRACTED (and by what). C = the regime contrast (section 1).

## 0. If you only read one thing

1. Every Stage 2 result is Qwen3-8B, thinking on, greedy, 3,072-token think cap, full bank (or a derived bank); thinking off is closed for new work (plan:187-194), though thinking-off stage A steering still ran on 2026-10-30 (see section 7).
2. The outcome is the near log-odds at the token before the label, in units of C (12.9 on the 176 near/both prompts; 19.2 in E14's own scale, so effect sizes are not comparable across docs).
3. Passing criteria so far: E8 run 3 (L24, natural closes), E9 R3, E10 plumbing + L16 subspace read, E16 C0-C3. Failing: E8 run 1 (all 12), L22, E14 H1-H3, E17 C1/C3, E18 C0.
4. Steering along the pooled manifold has never moved the written horizon much (E14: 47/128 moved, 16/128 landed; E16 number-token step 3/24) while donor states do (E16 24/24). The direction is the weak part, not the position (30:25-27, 30:123-125).
5. Decodable is not used: the best-decoding positions carried 0.00 of the effect (08:47-54). The causal path reaches the answer slot at L22-24 (plan:106-110).
6. "Horizon vs decision" is unresolved at the answer slot; the pre-registered test passed but the reading was downgraded (plan:123-131, 08:172-209).
7. Number is read at the number tokens (L12/16/22, 24/24), unit at the unit token; neither shows where the choice is computed (30:25-32).
8. Nearly everything after E10 is a screen (10 prompts/cell, 24 pairs, 16 rows/cell) with no significance claims (26r:3-4, 30:7-8).
9. Not decided and blocking: what counts as a Stage 2 success for a steering vector at the phrase, which direction to use (31 A1-A2 are not approved), the number of random draws, seeds, multiple-comparison handling, and a power basis for n (section 7).
10. Model-backed code is run by Jared, not the Claude session, except in sessions that say otherwise (CLAUDE:5-13; 26s:85-86).

## 1. Fixed setup for all evals

- **Model.** `Qwen/Qwen3-8B`, bf16, one vast.ai RTX 4090. DECIDED 2026-09-21 (plan:175), restated plan:26. E14 ran on an RTX 5090 (21:108, 22:268) and the scan spec allows a 4090 or 5090 (26s:3): hardware differed across runs.
- **Thinking mode.** Thinking on only. DECIDED 2026-10-14 (plan:187-194). Reason: thinking off has an option-order effect of +6.3 log-odds vs +2.5 on, and the within-regime horizon dependence flips sign with option order (plan:188-191, 07:30-36). Cost accepted: the prompt-side patching map T1 exists only for thinking off, because patching prompt positions with thinking on changes the reasoning (plan:192-194, 09:136-138). Thinking-off results stay as a record (plan:192).
- **Bank.** The full bank: Alan's generator (`spar_horizon/full_bank.py`), 500 prompts, 471 with a horizon, 48 log-spaced horizons from 30 s to 500 y, delays, rewards and option order drawn per prompt. DECIDED 2026-10-02 and 2026-10-03 (plan:178-183), restated plan:27-29. Regimes: `none` (horizon shorter than the near delay), `near` (between the delays), `both` (past the far delay), `null` (plan:29-31, 09:35-39). Counts of the 471: none 295, near 83, both 93 (09:35-39).
- **Prompt format.** Situation / task (option a, option b with reward and delay) / objective / constraint "... for this time horizon: X." / action / format "I choose: <a) or b)>. My reasoning: ..." (09:21-30). DEFAULT.
- **Decoding.** Greedy, batched, left-padded. DECIDED 2026-09-23 (plan:176); `do_sample=False` in `spar_horizon/model_io.py:201`. bf16 batched greedy is not bit-stable across batch composition (plan:233, model_io.py:19-21).
- **Think cap and force-close.** Think block force-closed at 3,072 tokens (`max_think_tokens`). DECIDED 2026-09-23 (plan:176, plan:31). 103/500 forced on the full bank, mostly long horizons; forced prompts are flagged, not dropped (plan:31-32). On force-close the code appends `</think>` plus the answer prefill and generates (model_io.py:10-12; 07:155-158).
- **Label parsing.** The label is located after the model's own `I choose:`, no prefill. DECIDED 2026-10-02 (plan:178-179). Labels are recorded per prompt because a markdown lead changes the label token's form (07:158-159); the 2026-10-26 audit fixed unspaced-label logits (plan:210-211).
- **Outcome for causal tests.** Near log-odds = `logit(near label) - logit(far label)` at the token before the label, one forward pass on the cached reasoning, no regeneration (plan:34-36, 07:40-42). DECIDED 2026-10-11 with the spec (07:3-5).
- **Unit of effect.** `C = mean LO(near regime) - mean LO(both regime)` on unedited prompts; every E8 effect is `mean(LO_edited - LO)/C` (07:48-51). C = 12.9 thinking on (176 prompts, plan:36-37), 11.4 off. E14 uses its own `C = 19.2` (22:33-34), and its H2 bar of 2.88 is 0.15 x 19.2 (22:194).
- **Prompts in causal tests.** The 176 `near` and `both` prompts, since only there does the horizon decide the rational answer (07:46-47, 09:48-49). DEFAULT of the E8 design.
- **Positions kept in the saved activations.** Suffix tokens (newline after `<|im_end|>` is `suffix_1`), think@0.25/0.5/1, `</think>`, pre-label token; every layer, float32 (plan:39-42, plan:234-235). DECIDED 2026-10-03 for the three think fractions (plan:182-183).
- **Layer convention.** `layer` = hidden_states index; an edit at L hooks decoder layer L-1; edits allowed at 1..n_layers-1 only (07:54-55, 07:150-153, `steering.py` module docstring). DEFAULT.
- **Steering step grid.** alpha in {-2, -1, +1, +2} decades, positive = longer horizon (07:84-86; `ALPHAS = (-2, -1, 1, 2)` at `exp8_steering.py:44`). Every direction is scaled to move each prompt as far as the curve does, so directions differ only in direction (07:73-82). DEFAULT; reason recorded: a probe-readout-matched step would be 10-19x too short (07:75-82).
- **Subspace.** Top-k PCs of the 48 horizon averages, k = min(8, horizons-1); curve = piecewise-linear interpolation in log h, clamped at the ends (07:64-69). Pooled "pristine" fit instead of horizon averages for E13-E16 (16s:28-48, 21:57-62). DEFAULT (E8); DECIDED 2026-10-27/29 for the pooled fit (16s:187, 16s:236).
- **Patch semantics.** A patch replaces (or adds to) the output of decoder layer L-1 at prompt positions on every prefill pass (reasoning pass and answer pass), never at decode steps (08:240-242, `exp16_phrase_patch.py` docstring). DEFAULT from E10.
- **Prompt-side phrase positions.** The tokens of "<horizon text>" after "time horizon: "; roles: number tokens (mean for the fit, step added to each) and the unit token (last token) (07:316-317, 21:41-50). DEFAULT.
- **Replication framing.** Full-bank results are reported as a replication next to the paper (Qwen3-4B) and Alan and Augusto (Qwen3-14B, thinking off). DECIDED 2026-10-02 (plan:179-181). What counts as a replication is not defined (section 7).

## 2. Evaluation protocol

### 2.1 Measuring behaviour
- **Choice.** The label chosen, plus the near log-odds. Flips = label differs from the unsteered or rerun baseline (26s:35-37, 27:43-45). DEFAULT.
- **Truth.** `near` regime -> near option; `both` -> far option; `none` -> no rational answer (09:35-39; 29:30-32). For the 4/8-choice banks the truth is the best payout that arrives within the horizon (26r:13-15). DEFAULT.
- **Forced vs natural closes.**
  - Forced closes confound every thinking-on cell after the block; top horizon-average component at `</think>` separates forced from natural at AUC 1.00 (plan:90-91). Forced is flagged, not dropped (plan:32).
  - E8 run 3 refits on the 371 natural closes of 471 (07:229-231, DECIDED 2026-10-14). Runs 1-2 use all closes (09:195-199).
  - E14 reports natural closes in section 2 because cut traces pull near log-odds toward 0 (22:38-39). E16 reports natural closes first (27:43-44) but the "natural" column uses only pairs where neither trace was cut (30:29-31).
  - No single rule for all Stage 2 runs (section 7).
- **Stated horizon.** The first duration the trace states as its horizon (E14 post hoc measure, 22:30-32); E10 records which horizon the reasoning names (07:351). A parser audit by a separate model instance agreed on 79 of 80 traces (22:105-107). The pre-registered H1 measure counts any duration written in the trace and was kept as written (plan:216-218, 22:201-203).
- **Mediation caution.** Stated duration and choice come from one generation, so any link between them is post-treatment and descriptive (29:99, 22:149-153).

### 2.2 Baselines and controls (what each is for)
- **Unedited near log-odds per prompt** = zero point of every effect (09:81-90).
- **Noise floor / rerun.**
  - E10: re-batching the unpatched prompts changes 12 of 96 labels vs the E0 cache (08:272-274).
  - E14: no-op rerun, same prompts (21:96); label churn 22-50% on none/both starts for any edit incl. random (22:271-273).
  - E15: baseline plus 2 no-op reruns with other batch order and smaller batch; noise about 2 of 10 flips (26s:35-37, 26r:73-74, exp15_scan docstring).
  - E16: baseline in input order + rerun in another order; floor 5/24 flips (27:93-95, 30:29).
  - E17: unit 1 re-scores the forced prefix in new batches; plumbing P0 requires >= 85% of forced labels reproduced (28:70-71).
  - All DEFAULT or spec decisions; the reruns are one extra pass, not an average over many.
- **Self patch.** Recipient's own states put back; must change 0 labels (E10: at most 2 of 24, 07:360; E16 C0: 0 labels, 27:48). PRE-REG.
- **Random controls.** Subspace ablation: 3 random orthonormal k-dim subspaces toward the same mean (07:94, `N_RANDOM_SUBSPACES = 3`, `exp8_steering.py:55`). Steering: random direction of the curve's length (07:244-246: one in run 3, two in runs 1-2). E14: one random per layer, as long as the step (21:101). E16: norm-matched isotropic and norm-matched inside the 8-dim span (27:34-35). Scan: one random twin per manifold cell (26s:30-31). E10: random 8-dim swap, not size-matched to the subspace swap (08:292-295; plan:171-172).
- **Mean ablation.** `x <- x - ((x - mu)B^T)B`; `mu` = mean activation at the same cell over the prompts the subspace was fitted on (471 horizon prompts of that mode; 371 natural closes for run 3), not the 176 edited ones and not zero (09:183-203). DEFAULT; reason: zero-ablation would leave the distribution (mean norm 183 vs prompts a median 21 from it, 09:199-203).
- **Probe line** (ridge direction for log horizon, at the curve's length) as the straight-line comparison (07:70-71). Dropped from run 3 since inert at every cell (07:244-246).
- **Mean difference (CAA).** Long half minus short half at the median log horizon (07:72-73). Steers backwards because the short half is all `none` (08:199-203).
- **Opposite step.** Same number of decades the other way, for none/both starts (21:102). Cut short at the curve ends on 42 of 64 prompts (22:26-27).
- **Donor patch = ceiling.** What a real horizon change does (21:103, 07:337).
- **Unit lookup.** Mean log horizon of training horizons with the same unit; R² 0.96 at the suffix (plan:77-81). Floor for "unit only" within unit is about -0.25 (plan:84-88, plan:211).
- **Decision-alone and `u`-only pushes** as controls for horizon vs decision (09:90-91, 07:240-246).

### 2.3 Statistics used and not used
- Used: bootstrap 95% over prompts, 1,000 resamples (07:105), within regime for removed shares (09:211-213); 2,000 prompt resamples for 15c and E14 (15c:53-54, 21:139-140); McNemar on paired implied vs twin (plan:66, 30:60-63); Wilson interval for the mediation share (30:67-68); Lomb-Scargle with shuffle null for periodicity, three PCs tried (30:111-115).
- Screens with no significance claim: E15 (10 prompts/cell), E16 (24 pairs), E17 (105+74 traces), E18 (16 rows per cell) (26r:3, 30:7-8).
- Not used: any multiple-comparison correction on the 12 E8 verdicts, the 40 scan cells, or the E14 layer profile; any power calculation; any run-to-run repetition beyond one rerun; any significance test for E10/E14 criteria beyond the CIs the criteria name. Only 15s writes a multiple-layer rule (two adjacent layers, 15s:101-102, DRAFT).

## 3. By modality

### 3.1 `patch/`: donor patching

**E10: patch the phrase before reasoning** (spec `07` section 12).
- Question: does the reasoning use the horizon code in the prompt? (07:312-317)
- Intervention: 96 pairs (first 96 near/both prompts in bank order, donor from the other regime, equal token length, `crossing_pairs` seed 0); thinking on; layers L12 and L16 (L22 run separately); position = phrase tokens, `last` = unit word (07:319-333). Conditions: recipient, donor text, full phrase, full last, subspace last (top-8, `x + ((x_d - x)B^T)B`), random last (8-dim), self last 16 (07:335-342). Subspace fitted in the same run from 471 prompts (07:344-346).
- Transfer = `(LO_c - LO_recipient)/(LO_donor text - LO_recipient)` over pairs with gap above 1 log-odds, bootstrap 95% (07:351-353).
- Criteria PRE-REG (07:354-360): (a) subspace last transfers >= half of full last at the same layer, interval of the difference to random last excludes 0, random last <= 0.1; (b) no pass/fail on full phrase; (c) self last 16 changes the label on at most 2 of 24 pairs.
- Result: plumbing passes (0 of 24 labels); subspace read passes at L16 (0.54 vs full 0.59, random 0.08); fails at L12 only on "random <= 0.1" (random 0.20 [0.01, 0.41]; subspace 0.85, full 0.71) (08:247-254). Status caution: spec says the conditions were "built on defaults proposed 2026-10-17 ... open to change before the run" (07:308-310), so criteria are written before the run but the run was not separately approved.
- What it shows: the patch swaps the unit, not the number; 68-70 of 71 unit-differing pairs write the recipient's number with the donor's unit (random 2/71); choice flips 29-30 of 46 (random 5-8), 0 of 6 where units match (08:256-285, plan:134-145). RETRACTED: earlier "71-73%" used the wrong denominator (08:260-261, plan:206).

**E16: phrase number patch** (spec `27`, drafted then approved overnight 2026-11-07, 27:77-82).
- Question: where is the number carried, and is the manifold step the wrong direction, place or size? (27:9-11)
- Pairs: 24 same-unit crossing pairs (12 none->near, 12 both->near), same phrase token length, seed 0, stratified by bank; donor = same prompt with another horizon of the same unit nearest the log-midpoint of the delays (27:14-20, 27:84-92). DECIDED 24 pairs (27:79).
- Layer 16 for all conditions; number-token donor patch also at L12 and L22 (27:37, 27:99).
- Conditions (27:22-35): self, full-phrase donor, number-token donor, unit-token donor, top-8 subspace donor on number tokens, manifold step on number tokens, manifold step on number + unit tokens, norm-matched random, in-span random, number donor at L12/L22. Conditions 9-10 kept in, run last (27:80). Manifold step goes from the prompt's horizon to the donor's horizon, not to the delay midpoint (27:93-96).
- Criteria PRE-REG (27:48-52): C0 self patch changes 0 labels; C1 >= 50% of traces write a target-regime horizon in condition 3 and random <= 10%; C2 manifold-number step reaches < 25% of condition 3's rate of writing the donor's number; C3 unit-token donor changes the stated number in at most 1 pair.
- Result: C0-C3 all pass (30:25). Self 0/24; full phrase 24/24; number donor 24/24 (L12/16/22); unit donor 0/24; top-8 subspace 12/24; manifold number step 3/24 (lands 8/24); manifold number+unit 4/24; random 0/24; in-span random 0/24 (30:12-24). Noise floor 5/24 label flips (30:29).
- Status: screen, no significance claims (30:8). The +6.18 "natural" log-odds for the manifold-number step rests on few pairs and is not an effect (30:30-31).

### 3.2 `ablate_steer/`: ablate, steer, patch at the answer slot (E8)

- **Run 1** (spec 07 sections 3-5, DECIDED and approved 2026-10-11, 07:3-5). Cells chosen by decodability: off suffix newline L16, off pre-label L21, on `</think>` L19, on pre-label L21 (07:54-60). Tests T1 interchange patching (thinking off, 96 pairs, one position group at a time, layers 1..last), T2 subspace ablation (k in {1,2,4,8}, 3 random), T3 line/curve/mean-diff/random steering, T4 = T2/T3 thinking on on cached reasoning (07:90-101).
  - Criteria PRE-REG (07:103-115), per cell, bootstrap 95%: necessary = ablating k=2 (off) or k=4 (on) removes >= 0.5 of C and every random removes <= 0.1; sufficient (line) = near-regime effect <= -0.25 at alpha=+1, both-regime effect >= +0.25 at alpha=-1, mean random |effect| at |alpha|=2 <= 0.1; curve beats line = curve's effect toward rational exceeds line's by >= 0.1 at |alpha| in {1,2}, interval excludes 0. T1 no pass/fail.
  - Result: all 12 verdicts fail; ablation removes at most 0.06 (08:31-43). Reason: the cells are not on the causal path (suffix newline decodes R² 1.00 and carries 0.00).
- **Run 2: answer slot** (spec 07 section 9, fixed 2026-10-11 before the run). Cells: pre-label L24/L27/L30, both modes; T2-T4 and criteria unchanged; random subspace only at the judged k (k=2 off, k=4 on); new T5 answer-slot patching across prompts (96 pairs off, 64 on), no pass/fail (07:190-206).
  - T5 result: on L22 0.22, L24 0.56, L34 0.82; off 0.58 at L24 (08:78-82).
  - Verdicts (08:84-91): off L24 curve beats line (ablation 0.33/0.37 at k=2/4, random 0.00); off L27 none; off L30 none; on L24 curve beats line; on L27 necessary; on L30 necessary. Probe line at most 0.01 everywhere; mean difference small and backwards at on L24 (-0.13).
  - Downgrade: L27/L30 "necessary" passes do not show the horizon is used because the subspace decodes the a-vs-b decision at R² 0.95 there (08:119-121).
- **Run 3: horizon or decision at L24** (spec 07 section 10, fixed 2026-10-14 before the run). Thinking on, pre-label L24, 371 natural closes, directions refit. Decision direction `u` = ridge from the k=8 coordinates to near log-odds. Steering: curve, curve without `u`, `u` part, `u` itself at the curve's length, mean diff, one random; ablation at k=4: subspace, subspace without `u`, `u` alone, 3 random, plus k=1,2,8. L22 optional second run (07:222-250).
  - Criteria PRE-REG (07:252-262; all 176 prompts, |alpha|=2, both signs pooled, units of C): steering = curve without `u` moves the choice >= 0.05 toward rational, interval excludes 0, and >= one third of the full curve's effect; ablation = subspace without `u` removes >= 0.1 of C, interval excludes 0, every random <= 0.05. Reading if both fail: the L24 curve effect is the decision (07:261-262).
  - Result: both pass (without `u`: steering 0.07 [0.05, 0.09], ablation 0.11 [0.07, 0.15]; random 0.01 / 0.00) (08:165-172).
  - Reading downgraded: ablation rows are not a split (only 0.25 of `u` lies in the top 4); the subspace without `u` still decodes near log-odds at R² 0.52 (08:174-183, plan:123-131).
- **L22** (same tests, 2026-10-18, optional run taken): all three criteria fail; ablation removes 0.09 [0.07, 0.10] vs 0.30 at L24 (08:303-310).
- **Horizon vs decision decodability check** (local, 2026-10-12, corrected 2026-10-26): subspace adds 0.58-0.83 R² beyond the decision alone at every cell; a-vs-b decision from the subspace 0.38-0.39 at L24, 0.95-0.97 at L27/L30 (08:96-127). Not a pass/fail.
- **Post hoc within-edit check** (09 section 6.4, after run 3's criteria passed): slope of demeaned effect on demeaned real-regime-change effect; curve 0.030 [0.000, 0.061], pure decision push 0.020 [0.007, 0.036]; cannot separate the readings (09:372-421, 08:184-197). Status: post hoc, inconclusive (plan:196-198).
- **E12 same-subspace sweep** (`15s`). DRAFT, "awaiting approval", criteria "fixed on approval" (15s:3-4, 15s:100). Layers 8B L18-L30, 14B L21-L33; cross-fitted directions H8, d, label, think length, C (5 dims); edits A/S/I/M. Criteria drafted: E12.1 ablating C removes <= half of what B3 removes and the label alone removes >= 0.2 (14B L27); E12.2 in the window (8B L22-L24, 14B L25-L26) ablating C removes >= 0.10 with CI low > 0 and >= 0.05 more than the largest norm-matched random; E12.3 C interchange transfers >= 0.05 (CI low > 0) and >= a third of the full-residual T5 transfer, random <= 0.02; E12.4 mediated fraction descriptive, claimed only if M >= 0.05 with CI low > 0 (15s:100-119). Never run. The table in section 1 of this draft is post hoc (15s:148).
- **E12 crossing spec** (`15c`). On hold behind the pristine manifold (15c:6-10; INDEX). Part A on saved E8 results: A1 `G(curve) >= 0.05 C`, CI above 0; A2 `G(curve) - G(decision) >= 0.05 C`, CI above 0; A3, A4 reported only (15c:61-77). Part B ladder on 8B on L24: B1 oriented J >= 0.05 C with CI above 0 and J minus mean slope above 0; B2 same at the near ladder; B3 `J(curve) - J(decision) >= 0.05 C` (15c:115-123). Status: "Draft for Jared's approval" (15c:3), then on hold. No results doc exists: the `16_results_crossing.md` it names is not in the repo (15c:146).
- **Pristine steering stage A (answer slot) and stage B (suffix)** (spec `16s` section 5, 2026-10-27, no gate DECIDED 2026-10-29, 16s:236-239). Directions: effects curve, invariant part, in-span part, unit `U`, 2 random; alpha = +/-1, +/-2; stage A thinking off suffix L16 and pre-label L22/L24, thinking on pre-label L22/L24; stage B thinking-on suffix L16, 96 bases, 4 conditions at alpha = +/-1 (16s:187-234). `exp8_steering` docstring: "The spec fixes no pass/fail criterion for them". Result: stage A thinking off pre-label L24 |alpha|=2: 0.139 C toward rational, invariant part 0.092, crossing term with step length controlled +0.013 [-0.005, 0.032]; thinking on up to 0.03 (22:243-248). Stage B: traces named their own horizon 99-100%, nothing beat random (22:249-250). No criteria, so no pass/fail.

### 3.3 `manifold_steer/`: steering along the manifold

**E14: steer at the phrase** (spec `21`, DECIDED 2026-10-30/31, 21:3-13; results `22`).
- Question: steering at the phrase, toward a target horizon that crosses a delay, does the reasoning state that horizon and does the choice follow?
- Intervention: 128 prompts, 32 per transition (none->near, both->near, near->both, near->none), natural renderings, seed 0, stratified by bank (21:88-91). Target: log-midpoint of (near, far) for ->near; `log far + 1` for near->both; `log near - 1` for near->none (21:67-72). Eligibility: target inside 30 s-500 y; for midpoint targets `log far - log near >= 0.6` (21:79-83).
- Direction: pooled manifold fitted at the phrase tokens per role per layer (duration effects with unit, regime, order factors and log delays as covariates; 1,386 prompts, k=8) (21:57-62, 22:262-263); step = `curve(target) - curve(log h)`. Donor patch kept as the ceiling (21:8-9).
- Layers: L16 carries all conditions; L14 and L18 carry the effects step and random only (21:9-11, 21:51-55).
- Conditions: no-op rerun, effects step (L14/16/18), invariant part (L16), random (L14/16/18), opposite step (none/both, L16), donor patch (L16) (21:93-102). 38 conditions, 1,192 generations, about 4 h (22:268).
- DECIDED: one box session, no pilot, no go/no-go pause (21:12-13, 2026-10-31).
- Criteria PRE-REG (21:132-147), judged on the L16 effects step: H1 share writing a target-regime horizon >= 0.30 with CI above random's and random <= 0.10; H2 oriented change in near log-odds >= 0.15 C with CI above 0 (rerun baseline); H3 on none/both effects step minus opposite step >= 0.15 C with CI above 0, and on near prompts both reverses lower near log-odds (CIs below 0). L14/L18 reported on the same thresholds, not gating. 2,000 prompt resamples.
- Result (22:190-195): H1 fail (0.32 vs random 0.20: the 0.30 bar is met, the "random <= 0.10" bar is not); H2 fail (+1.80 [0.58, 2.98] vs bar 2.88); H3 first part fail (+0.35 [-2.47, 3.16]); H3 near reverses fail (near->none -1.00 [-2.26, 0.19]). Donor patch +9.9 vs steer +1.4 on the 104 prompts with a donor (22:43-45).
- Post hoc (labelled, gate nothing; plan:216-218, 22:287): first-stated-horizon, landing, within-unit one-decade moves, unit swaps, geometry. These show the steer moved the stated horizon on 47/128 traces at L16 and landed 16/128 (22:76-79); random 0/384 (22:78).

**E13 stage B** (spec `16s`; see 3.2): steer at the thinking-on suffix newline L16 before reasoning. No criteria.

**E15: layer x position x direction scan** (spec `26s`, approved 2026-11-06 with amendments in section 6, 26s:3, 26s:89-114).
- Question: a screen, "no H1-H3" (26s:19-20). Cells: layers L10/14/18/22/26 x {number, unit, suffix newline, answer slot} x {manifold step, norm-matched random} = 40 (26s:22-31). Run order of layers L18, L14, L22, L10, L26 (26s:108-110).
- Prompts: 10 per cell, the same 10 in every cell, 5 none->near and 5 both->near, seed 0 from the E14 eligible set (26s:33-34). Departure from the message flagged "need your OK" (26s:74-76); the doc's header says approved with amendments, and the item-level OK is not recorded separately.
- Measure: flip = label differs from baseline; excess = cell flips minus mean of the two no-op reruns' flips; reported if excess toward-target flips >= 4 (PRE-REG, 26s:35-39). Answer-slot cells re-score cached reasoning with no regeneration, so their flips are a different kind of event (26s:44-49, 26s:76-78).
- Result: no cell reached excess >= 4 (largest +3); random cells averaged 2.9 flips vs 2.5 for manifold over the 13 comparable cells; answer-slot cells 0 flips (26r:74-82). 34 of 40 cells ran; L26 missing six cells (26r:92-96). Reading: no layer or position to follow up (26r:88-90).
- Extras (4 listed, 26s:50-52) were not run (26r:97-98).

**25: arithmetic mediation** (draft 2026-11-03, "not approved", 25:1-3). Parts A resampling at T=0.6/top-p 0.95/top-k 20, 8 traces per prompt; B insertion of correct/wrong/neutral arithmetic sentences; C rate-anchored prompt "5% per year". Criteria written (25:51-62) but DRAFT; never run. Would need sampling in `model_io`, which is greedy only (25:27-28).

### 3.4 `think_budget/`: E17 (spec `28`, approved overnight 2026-11-07, 28:121-137)
- Question: are choice4/8 forced-close errors a think-budget artefact? (28:20-24)
- Design: Stage 0 on CPU fixes `loop` (a 150-character block repeated three or more times in the last 4,000 characters) and `tentative` before any continuation (28:42-47). Stage 1: unit 1 rerun of forced answer pass; unit 2 resume from the 3,072 cached think tokens for up to 3,072 more (total 6,144), batch 8; unit 3 optional to 8,192 at batch 4, ordered last (28:48-56, exp17 docstring: 2,048 more for unit 3). Chunks of 24 traces, seed 0 shuffle, stratified by regime (28:133-135, `exp17_think_budget.py:42`). DECIDED: choice8 included, no activations saved on continued traces (28:123-128).
- Criteria PRE-REG (28:67-74): P0 unit 1 reproduces >= 85% of forced labels (else stop and report); C1 >= 60% of the 105 close within 6,144, and on those exact accuracy >= 85% with net fixes >= 10; C2 excess-late <= +0.17 (half of +0.34); C3 of `loop` traces at most 30% close and of the others at least 70%. Choice8: C1 net fixes scale as 10 x n/105; C2 is half of its own forced excess-late (28:138-139).
- Result (30:35-52): P0 pass (103/105, 73/74). Close share 65% / 62% (>= 60%). C1 fails narrowly on accuracy (choice4 84% vs 85%; choice8 78%) and, for choice8, net fixes (7 vs 7.05 needed); choice4 net fixes 13 - 2 = 11. C2 passes. C3 fails (20/57 = 35%; choice8 11/35 = 31%), though looping traces still close less often than others (48/48, 35/39). +8,192 closed none of the 37 / 28 that stayed open (25 / 27 attempted).
- Reading in the doc: errors are partly a budget problem and partly loops; the forced-vs-natural error comparison is confounded with horizon and looping (30:50-52). Excess-late drop (+0.26 -> +0.03) is a subset comparison, approximate (30:46-49).

### 3.5 `units_implied/`

**E9: unit rewrites at the suffix** (spec `07` section 11, fixed 2026-10-14 before the run, 07:275).
- Question: does the readout before reasoning track duration or unit word? Intervention: none (forward pass only). 22 durations from 10 min to 100 y, each in 2-4 units (62 renderings incl. "decades", "century"), in 24 full-bank contexts, plus originals: 1,512 prompts. Kept: suffix newline and last suffix token, every layer (07:281-294).
- Criterion PRE-REG (07:295-301): at the suffix newline, L16, the unit share of readout variance (within durations, over units) is below 20% of the total.
- Result: R3 passes; unit share 0.02 (curve readout 0.03), at most 0.08 from L12 on at both suffix positions; slope 0.90 over 7 decades (08:218-222). Qualifier: mean |shift| of a rewrite is 0.30 decades, e.g. "1 century" -0.80, "3600 seconds" +0.74 (08:223-229). Plan carries "R3 passes at the suffix with unit shifts of ~0.3 decades" (plan:19, 98-100).

**E18: implied horizons** (spec `29`, approved overnight 2026-11-07, 29:122-139).
- Question: does the choice track an implied horizon as it does a written one; does it reach the same states; does the trace convert the anchor first and how accurately? (29:7-12)
- Bank: families A (dates), B (world-knowledge, number-free), C (event anchors, duration stated elsewhere), D (vague, descriptive). Each base in implied form, explicit twin (log-midpoint of the interval) and, once per anchor, a null twin; nearer delay >= 0.3 decades outside the anchor's interval (29:30-56). 288 rows: M1 144, M2 72, M3 48, M4 24 (29:57, 29:137-138).
- DECIDED 2026-11-07 (29:124-134): M1 first and usable alone; family C counts as implied; B's US-convention anchors stay with `us_convention` flag; none-regime dropped from the criteria (89% forced) but still built as M3; no donor patch.
- Criteria PRE-REG (29:73-90): C0 twin accuracy >= 90% on near and on both (fail -> "stop and fix"); C1 per family implied accuracy >= 85% and >= twin - 10 points; C2 >= 80% of implied traces state a duration, A/C median error <= 0.1 decades, B >= 85% inside [lo, hi]; C3 suffix-readout MAE of implied <= 1.5x twins' and rho >= 0.8 means present, rho < 0.3 absent; C4 with >= 10 regime-flipping misconversions, choice follows the stated regime in >= 75%.
- Result (30:53-79): C0 fails (explicit-twin accuracy 0.875 near, 0.94 both). C1 passes for C, fails for A and B (accuracy implied vs twin, near/both: A 62/88 vs 81/94, B 56/88 vs 94/88, C 94/81 vs 88/100; 16 rows each). McNemar p = 0.34 (A), 0.11 (B), 0.69 (C), 32 pairs each: no gap significant. C2: A and C 40/40 state a duration (median error 0.04, 0.001 decades), B 35/40 state one and 69% inside the target interval (the doc does not print a C2 verdict). C4: 7 qualifying misconversions (< 10), follow-stated 1 of 7, underpowered, no verdict. C3: 21 cells, 5 present, 1 absent, rest intermediate; no single claim made.
- Status: provisional because C0 failed (30:55-56). Stop rule in 29:76 ("stop and fix") was not applied before M2-M4 ran; see section 5.
- Candidate 2 (donor patch of explicit twin into implied phrase): rejected for this spec; "run it only if 1 finds the implied duration at the suffix" (29:23-26, 29:134). Not triggered.

## 4. Interpretation decisions (what nulls and passes do and do not show)

- **Decodable is not used.** A position where the horizon decodes at R² 1.00 can carry 0.00 of the causal effect (08:47-54, plan:107-110). DECIDED reading.
- **Where the decision arrives.** The decision reaches the pre-label `:` at L22-24 (T5 0.22 / 0.56), and reasoning does not move where it is read from (plan:106-109).
- **Probe line is inert** at every cell (at most 0.01 C) while shifting its own readout by 18-29 decades (plan:111-112).
- **Curve at L24** moves the choice 0.15 C toward the rational answer (both-regime +0.15 at -2 decades), but near-regime prompts move toward far at both signs (-0.14 each) and the subspace decodes near-vs-far at R² 0.65 (plan:113-118, 08:129-146). The reading is "consistent with the average decision per horizon; cannot be separated from a horizon compared with the delays" (08:205-209, 09:419-421).
- **Run 3 pass** does not mean horizon beyond the decision: `u` is a ridge direction, not the whole decision (R² 0.52 left) (08:184-187, plan:123-131).
- **Mean difference (CAA)** steers backwards because the short half is all `none`, which the model answers like `both`; forced closes are not the cause (08:199-203, plan:119-121). The earlier forced-close explanation is RETRACTED by run 3 (08:200-203).
- **L22 fails; L27/L30 ablation** (0.74-0.83 C) says nothing about the horizon since the subspace decodes a-vs-b at 0.95 (plan:132-133, plan:146-147).
- **E10 swap** is causally the unit, read back as a written horizon; it says nothing about magnitude within a unit (08:287-299, plan:139-143). E16 later: number tokens 24/24, unit token 0/24 (plan:141-143, 30:25-27).
- **E14 H1-H3 fail**: the phrase carries the horizon the reasoning uses (donor patch moves every prompt) and the steer does not reach it (22:44-45, 22:205-212). Where an up-step did move the stated horizon, the choice matched what the map predicts; correlational, since the steer is present when the answer is scored (22:214-216, 22:58-60).
- **E14 H2's +1.8 is mostly not landing** (22:143-148).
- **Scan null.** At 10 prompts per cell the screen catches only large effects; no cell supports a layer or position follow-up. The one cell worth a look (L10 unit manifold) has an equal-sized random twin (26r:88-92). In-span share of steps is 50-85%, about 0.5 at the unit token, so a hit there would be largely a unit/regime/delay push (26r:83-86).
- **E16.** The number is read from the number tokens at all three depths so depth does not localise it; it shows the number is read from the phrase positions, not that the choice is computed there (30:25-32). The manifold step's weakness is the direction, not placement (30:123-125).
- **Manifold figures.** The fitted PC1 slope (-9.7 per decade) is produced by the unit adjustment (raw means -0.18 per decade); same-unit patch steps are unaffected; anything using the absolute curve across units (the scan's targets) inherits the confound (30:93-110, plan:239-241).
- **E17.** Forced-close errors are partly a budget problem (closed traces become accurate and lose the late-slot bias) and partly loops that more tokens do not fix; not a clean causal decomposition (30:50-52).
- **E18.** Nothing supports "only family C tracks"; with C0 failed the data neither show A/B differ from their twins nor that C does not (30:60-64).
- **Retracted or struck (2026-10-26 audit, plan:195-213):**
  - "Suffix code shared across models" (11 section 2, 13 C1): null permuted horizons across prompts.
  - Decision-split ablation readings (08 section 9, 11 section 3, 13 B2): `u` lies mostly outside the ablated top k; verdicts kept, readings struck.
  - "14B binds the label four layers after the hand-off": 53 rows scored wrong label tokens.
  - E10 "71-73%": wrong denominator; 96-99% over pairs whose units differ (the 08 text now says 70/71 and 68/71).
  - 08 section 9 within-edit check as decisive: post hoc, inconclusive; "follows the pattern of real behavior": not on both->none.
  - "think@1 recovers (0.81)": 0.96 on natural closes.
  - 21 section 1 claim that behaviour slopes are flat: corrected 2026-11-01 (21:26-31, 22:251-255).
  - 08 section 6 decodability numbers corrected 2026-10-26 (per-fold PC refit, up to 0.16 lower on thinking on) (08:100-103); 09 section 6.1's table still prints the older values (see contradictions).

## 5. Pre-registration rules

- **Criteria fixed before the run** (PRE-REG): E8 runs 1-2 (07:103-115, 07:183-206, 2026-10-11, DECIDED plan:185); E8 run 3 (07:252-262, 2026-10-14); E9 R3 (07:295-301); E10 (07:354-360, drafted 2026-10-18, conditions "open to change"); E14 H1-H3 (21:132-147); E15 report threshold (26s:38-39); E16 C0-C3 (27:46-52); E17 P0, C1-C3 (28:67-74); E18 C0-C4 (29:73-90). Specs for E12 (15c, 15s), 16s stage A/B, 25 are not run or carry no criteria.
- **Post hoc, labelled as such:** 09 section 6.4 within-edit check (09:372-377); 08 sections 6-7 review (08:96-153); 22 sections 1-3 (22:287); `exp14_analysis.py` post hoc measures "label post hoc and gate nothing" DECIDED 2026-11-01 (plan:216-218); 15s section 1 table (15s:148); 18 confound sweep ("exploratory", 18:206); 30 addendum (30:93-126); E14 geometry and probe.
- **Criteria changed after seeing results:** none recorded. The pre-registered H1 measure was kept as written even though a better measure was added (plan:216-218, 22:201-203).
- **Stop rules written and what happened:**
  - 26s: down by $6 target with no hard stop, never past $8 (26s:3-4). Followed: spend about $4.5, Part B stopped at 37 units, extras not run (26r:108-111). The stop was about instance lifetime, and L26 was left incomplete; nothing pre-registered said to stop at 34/40.
  - 27/28/29: shared $10 hard cap across three experiments (27:79-80, 28:129-130, 29:126-128). Followed in the sense that spend was $6.33 by `just burn`; unreconciled with the time window (30:3-6).
  - 28 P0: "stop and report" if < 85% reproduced (28:70-71). P0 passed (30:41).
  - 29 C0: "Fail -> delay design is wrong, stop and fix" (29:76). C0 failed (30:55). All four modules ran and the results were labelled provisional; the stop rule was not applied as written (modules ran unattended overnight, 29:124-128).
  - 28 stage 0: stop if forced prompts' token ids are missing from `gen_cache` (28:51-54); no record in the docs of it triggering.
  - 21: no pilot, no go/no-go pause (21:12-13).
- **Multiplicity:** no rule outside the 15s draft (section 2.3).
- **Screens are not tests.** E15, E16, E17, E18 declare "no significance claims" (26r:3, 30:7-8); their criteria are rule-based thresholds on counts, not hypothesis tests.

## 6. Compute and process decisions

- **Who runs model-backed code.** Not the Claude session: model-backed pytest, E0 smokes and think-budget runs go to Jared as a self-contained script on the xpu env; the fast suite (`pytest -q tests --ignore=tests/test_generation.py`, mp env) is fine locally (CLAUDE:5-19; README:104-108). Exception, this session only: the overnight runs of 2026-11-06 and 2026-11-07 ran from the session on the box, as instructed (26s:85-86; "overnight" decisions 27:77, 28:121, 29:122). The memory note "Model runs go to Jared" applies otherwise.
- **The box.** Claude runs it end to end: `just up HOURS`, `just push`, `just ssh`, `just sync`, `just down` (CLAUDE:22-32). One box shared by every session; `just status` first; experiments in tmux; check `tmux ls; nvidia-smi` before `down` (CLAUDE:37-47). Watchdog stops the box after `IDLE_KILL_MIN` idle minutes (CLAUDE:45-47; 27:104 gives 60 minutes). `just push` excludes dotfiles (27:104).
- **Budget.** Jared states a dollar budget each session, a hard cap; no budget -> ask (CLAUDE:49-56). Recorded caps: $8 for the 2026-11-06 scan, expected $2-3, target `down` by $6 (26s:3-5); shared $10 for 2026-11-07's three experiments (27:79-80). Spend $4.5 (26r:110-111, unverified against billing) and $6.33 (30:3-6, unreconciled with the time window: $6.33 at $0.421/h is about 15 h vs about 10 h). Billing key lacks read permission (31:49-50).
- **Queues and resumability.** Box queues in `pipeline/s2_steering/queues/*.sh`, run from `jared/` in tmux; each skips finished work (README:87-102). Unit = one condition or cell, saved on finish, claimed with a `.lock` folder so copies share a queue (exp14/15/16/10 docstrings). Chunked runs (48 prompts for E0 banks, 24 traces for E17) written as soon as done; shuffles use fixed seeds so any prefix is balanced (26s:107-114, 28:133-135).
- **Sync.** Sync after every few units; the host stopped itself twice on 2026-11-06 and two syncs dropped on 2026-11-07 (27:61-62, 26r:101-108, 30:81-83).
- **Order of conditions.** Controls first, so a stopped run is usable: E16 order 1, 2, 3, 6, 7, 8, 4, 5, 9, 10 (27:60-62); E14 both->near first, L16 steer/random/donor before L14/L18 (21:110-113); E15 layers L18, L14, L22, L10, L26 (26s:108-110).
- **Compute facts.** Think length median 1,748 tokens; E0 thinking on 124 min at batch 12; E8 on-cell 25 min on cached reasoning (plan:227-231). Activations about 6 GB per mode float32 (plan:234-235). E14 about 4 h; E16 about 3 h (13 units of 24 generations, 27:100); E17 1.9-2.5 h choice4 (28:104-107).
- **Tests.** All tests must pass; never delete, weaken or skip (global CLAUDE.md). Fast suite list: README:109-113.
- **Reproduction scripts.** `exp14_analysis` and `exp14_geometry` run on CPU from synced results (22:279-285).

## 7. Not decided / open for Stage 2 (most blocking first)

1. **What a Stage 2 success criterion is for a phrase-position steering vector.** E14's H1-H3 failed; nothing replaces them. 31 A1 (linear steer at number tokens, held-out pairs) and A2 (direction fitted from donor differences) are "not approved or costed" (31:3-4, 31:8-16). No spec says which of E16's measures (writes the donor's number, lands in regime, choice change) is the gate, or the pass bar.
2. **Which direction to build.** Pooled manifold step reaches 3/24 against donor 24/24 (30:19). Options are in 31 A1-A2 and the open item in 30:90-91; none chosen. Unit adjustment as part of the "horizon direction" is open (31:29-32, 31 B8).
3. **Horizon vs decision.** No approved test separates "horizon compared with delays" from "average decision per horizon" at the answer slot or earlier. 15c is on hold, 15s is a draft, and 31 A5 (what carries the choice, as opposed to the written horizon) is only listed (08:205-209; 22:232-238; 31:19-22). 22's next steps (magnitude or digit length; written horizon causes choice, with a reverse arm) are proposed, not approved (22:222-238).
4. **Seeds and replication of random draws.** Seed 0 is used everywhere for pair/prompt selection (07:100, 08:240, 21:90, 27:91, `exp17_think_budget.py:42`), DEFAULT, never varied. Number of random directions is inconsistent (1, 2 or 3; section 2.2) and no rationale is recorded. A second prompt seed for top cells was an unrun extra (26s:52).
5. **Sample size.** 10 prompts/cell, 24 pairs, 32 per transition, 16 rows per cell, 105+74 traces; no power basis. 27:65-68 asked whether 24 or 48 pairs; DECIDED 24 as a screen, 48 not run.
6. **Multiple-comparison handling.** None for E8's 12 verdicts, the 40 scan cells or E14's layers. Only a draft rule in 15s (two adjacent layers, 15s:101-102).
7. **Effect-size cutoffs have no recorded rationale** and differ between docs: 0.05 C (15c, 15s E12.3, 07 §10), 0.1 C (07 §5 necessary random, 07 §10 ablation, 15s E12.2), 0.15 C (E14 H2/H3), 0.25 C (07 §5 line), 0.5 C (07 §5 necessary), 4 flips (E15), 25%/50% (E16), 85%/30%/60% (E17), 90/85/80/75% (E18). Which `C` an E8-style cutoff uses (12.9 vs E14's 19.2) is not stated in the specs.
8. **Forced-close handling across Stage 2.** Natural-only in some runs, all closes in others, flagged-not-dropped globally (plan:32). E17 C1 failed narrowly and no decision records what cap to use for later choice4/8 runs (28:77-80 conditions the choice on C1 passing); loop detection or resampling is only a goal (31:47-48).
9. **E18 C0 failure.** Rewrite the explicit twins or relax the check; "possibly drop M3" (30:90-91, 31:38-41). Open.
10. **Thinking-off role.** The 2026-10-14 rule says "nothing new runs in that mode" (plan:192), but 14B E11 (2026-10-20), unit-overlap E0 both modes (2026-10-25, plan:59) and stage A thinking-off steering (2026-10-30, 22:243-248) ran after it. No recorded decision says these are exceptions. 15c Part B optional 14B off and 21 section 8 (thinking off out of scope) are inconsistent with each other.
11. **Greedy only.** All Stage 2 is greedy; bf16 batch noise flips 12 of 96 labels (E10) or 12 of 50 both-regime prompts (25:13-14) and is measured by one rerun. Sampling (25) is DRAFT. No decision on averaging over reruns.
12. **Layer range for the phrase steer.** L14/16/18 (E14), L10-26 scan, L12/16/22 patch only (E16). Layers between and below L12 for the patch (31 A3) and L22+ for the step (31 A4) are proposed. Whether steering at layers other than L16 is needed is open (31:17-18).
13. **Size-matched random controls** for E10's swap and for ablation, and the decision split with `u` projected into the ablated top k (plan:165-172): proposed, no spec.
14. **Replication standard.** Full-bank results "reported as a replication" (plan:179-181); no definition of replicates, and the comparison with Alan's public 14B activations is deferred (plan:243-247).
15. **Domain change (R2), multiturn, E5 attribution** deferred (plan:243-247).
16. **Process items listed but not done:** commit/push/merge, spend check against vast.ai billing, watchdog text and log, a box-side finish marker that stops the box (31:46-54).
17. **Individual items of 26s section 5** ("need your OK" list: 10 prompts = 5+5, flip definition, positions beyond spec 21, "> $2" meaning, running model-backed work from the session) were approved only as part of the overall "approved with amendments" (26s:3, 26s:74-87).

## Contradictions and inconsistencies between docs

- `09:295-304` (sec 6.1 table: L24 horizon from subspace 0.77, a-vs-b 0.23) vs `08:105-112` (corrected 2026-10-26: 0.73 and 0.38). 09 acknowledges the 0.23 predates the per-fold refit (09:304) but the table was not updated.
- C scale: 12.9 (plan:36, 07:48) vs 19.2 (22:34); both are called "C".
- Hardware: 4090 (plan:26) vs RTX 5090 for E14 (21:108, 22:268).
- Thinking-off: plan:192 "nothing new runs" vs plan:59 (units bank both modes, 2026-10-25) and 22:243-248 (stage A thinking off, 2026-10-30). Also 15c:125 and 21:202-204.
- E14 H1: the plan:217-218 and 22 list it as a fail, but the 0.30 bar was met (0.32); it fails only on "random <= 0.10" (0.20) (22:192).
- 15c:146 names `16_results_crossing.md`, which is not in the repo; plan:154-157 §5 item 2 still lists the crossing split as "next" while 22:243-248 and plan:60 report `crossing_split_pristine`. Part A on the old 14B cells is not reported anywhere.
- 07:308-310 says E10 conditions are "open to change before the run", while 08:247 calls its criteria "pre-registered".
- 29:76 says stop on C0 failure; 30:55-56 shows all modules ran.
- Spend: `30:3-6` flags $6.33 vs about 10 h window as unreconciled; `26r:110-111` $4.5 unverified.
