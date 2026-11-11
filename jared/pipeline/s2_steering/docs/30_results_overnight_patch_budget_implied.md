# Overnight 2026-11-07: phrase patch (27), think-budget extension (28), implied horizons (29)

One RTX 4090 box (59546743, $0.421/h), tmux queues, 10:06 to ~20:00 UTC. Spend $6.33 by `just burn`
(estimate, not a bill: it is hours x rate and ignores bandwidth; billing API not readable with this key).
Unreconciled: $6.33 at $0.421/h is about 15 h, but the window above is about 10 h; the box was destroyed, so
the uptime cannot be rechecked. Treat the spend as unverified.
Box destroyed after a full sync. All three are screens (24 pairs / 105+74 traces / 288 rows), no significance claims.
Raw summaries: `results/horizon/exp16_*/patch_summary.txt`, `exp17_*/think_budget_summary.txt`, `exp18_*/implied_summary.json`.

## 27: where the number lives (phrase patch, 24 same-unit crossing pairs, L16)

| condition | states donor's number | lands in target regime | d near log-odds (natural) |
|---|---|---|---|
| self patch | 0/24 | 0/24 | -0.02 |
| full phrase donor | 24/24 | 24/24 | +5.74 |
| number-token donor | 24/24 | 24/24 | +5.16 |
| unit-token donor | 0/24 | 0/24 | +0.16 |
| top-8 subspace donor | 12/24 | 20/24 | +0.72 |
| manifold step, number tokens | 3/24 | 8/24 | +6.18 (n small, see below) |
| manifold step, number+unit | 4/24 | 9/24 | +2.10 |
| random, number tokens | 0/24 | 0/24 | -0.61 |
| random inside the 8-dim span | 0/24 | 0/24 | +1.03 |
| number donor at L12 / L22 | 24/24 / 24/24 | 24/24 / 24/24 | +4.33 / +5.92 |

Criteria C0-C3 all pass. Reading: the number is carried at the number tokens, at L12, L16 and L22 alike (so depth does not
localise it), and not at the unit token. The fitted manifold step is the weak tool: it reaches the donor's number in 3/24
against 24/24 for the real states, and the top-8 subspace alone gets half. The in-span random does as little as the isotropic
one, so the step's failure is not just a norm effect.
Caveats: the noise floor is 5/24 label flips (rerun vs baseline); the "natural" log-odds column uses only pairs where
neither trace was cut off, and the manifold-number +6.18 comes from few such pairs and should not be read as an effect
(its forced column differs a lot, 16 forced vs 4-10 elsewhere). Donors repeat (only 19 none-start and 35 both-start prompts
have a same-length donor). The patch writes the donor's number in the trace, so this shows the number is *read* from the
phrase positions, not that the choice is computed there.

## 28: more thinking tokens for force-closed choice traces

| bank | forced traces (looping) | closed within +3072 | wrong->right / right->wrong | closed loop / no-loop |
|---|---|---|---|---|
| choice4 | 105 (57) | 68 (65%) | 13 / 2 | 20/57, 48/48 |
| choice8 | 74 (35) | 46 (62%) | 7 / 0 | 11/35, 35/39 |

P0 (rerun reproduces the original label) passes: 103/105, 73/74. C2 (late-position bias disappears after closing) passes.
C1 fails narrowly on accuracy (choice4 exact 84% vs 85% needed; choice8 78%) and choice8 net fixes (7 vs 7.05 needed).
C3 fails: the bar was at most 30% of looping traces closing, and 20/57 = 35% (choice8 11/35 = 31%) close, above it;
looping traces still close far less often than non-looping ones (48/48, 35/39). Extending to 8,192 closed none of the 37 and 28 that stayed open
(25 and 27 attempted). So the forced-close errors are partly a budget problem (the closed traces become accurate and lose the
late-slot pick bias). The excess-late comparison is between subsets: the 105 forced traces have +0.33, the 68 that
closed later have +0.03, which is not a before/after on the same traces. On the same 68 prompts the original forced labels
had +0.26 and the continued labels +0.03 (`position_original_same_prompts`), so the drop holds within the subset; the 37
that stayed open are not in either 68 figure. Excess-late is a rate difference on 68 prompts with no interval, so
the size of the drop is approximate. Errors concentrating in the forced closes (4 of 141 natural, 40 of 105 forced) is confounded with horizon and looping:
forced traces can differ from natural closes in horizon and in looping (57 of 105 loop), so the table does not isolate the cut-off as the cause. The errors are also partly loops that more tokens do not fix.

## 29: implied horizons (288 rows: M1 144, M2 72, M3 48, M4 24; readout probes trained on E0 think-on activations)

- C0 (reference valid) **fails**: explicit-twin accuracy 0.875 (near), 0.94 (both), under the threshold. Treat everything
  below as provisional.
- Behaviour, implied vs explicit twin (accuracy, near/both): A 62/88 vs 81/94, B 56/88 vs 94/88, C 94/81 vs 88/100.
  Each cell is 16 rows. C1 (the rule-based criterion) passes for C and fails for A and B. Paired McNemar
  p = 0.34 (A), 0.11 (B), 0.69 (C), 32 pairs each, with C0 failed: none of the gaps is significant, so the data do not
  show that A or B differ from their twins, nor that C does not; B (p = 0.11, only-implied-right 2 vs only-explicit-right 8)
  is the closest to a difference. Nothing here supports "only C tracks" as a finding.
- Conversion (what horizon the trace writes): A 40/40 and C 40/40 state a duration (median error 0.04 and 0.001 decades);
  B states one in 35/40 and 69% fall inside the target interval. Misconversions that flip the regime: A 8, B 5, C 0 (13 in all; the bars in
  `implied_2_conversion.png` count these).
- Mediation: of 7 regime-flipping misconversions, the choice follows the stated regime in 1 (14%; Wilson 95% interval
  3-51%). Underpowered (7 against the criterion's minimum of 10): no verdict. Why 7 and not 13: the 13 count compares the stated
  regime with `base_regime` over all 40 implied rows per family, including the none-regime rows (M3) and stated
  regimes of "none". `mediation()` in `implied_summary.py` only takes implied rows whose true regime is near or both and whose
  stated regime is the other of near/both, because only these have a rational option to follow. The summary does not
  store how the other 6 split between the two exclusions.
- Null twins pick near 12% / 25% / 50% (A/B/C), so the no-horizon default is not clean either.
- Readout (probe MAE in decades): implied is worse than explicit at the suffix (e.g. B 0.48 vs 0.23, A 0.90 vs 0.39) and
  rank-correlation is mixed, not uniformly high. Implied rho by site at L16 (A/B/C): suffix_1 0.88/0.90/0.72 (explicit
  0.96/0.97/0.87), prelabel 0.85/0.74/0.68 (explicit 0.85/0.89/0.60), think@1 0.66/0.24/0.41 (explicit 0.64/0.66/0.52);
  n is 15-40 per cell (`implied_3_readout.png`). The implied horizon is weakly present at think@1 (B rho 0.24, "absent").
  C3 verdict per cell is in the printout (21 cells: present 5, absent 1, rest intermediate); I have not turned it into a
  single claim.
- M3 (none regime) ran although decision 4 dropped it from the criteria; 89% of its traces were force-closed.

## Process notes

- The host connection dropped during two syncs (rsync code 12/255); the box itself and tmux survived. Syncs resume,
  so I reran them. No host stop this time.
- Nothing monitored the box between my turns until I scheduled an hourly check partway through. The idle watchdog would
  not have stopped a finished queue.

## Open

- Promote C0 failure of 29 into a decision: rewrite the explicit twins or relax the reference check?
- Better direction for 27: fit from donor differences at the number tokens (the donor patch works, the pooled manifold does not).

## Addendum (2026-11-08): what the manifold figures show, and what 27 adds to E14

Figures: `results/horizon/exp16_Qwen3-8B_patch/figures/` (`pipeline/s3_analysis/figures/patch_figures.py`): where/layer grid, choice shift
per pair, the manifold curve, all 8 manifold directions.

**Worth noting: the manifold's PC1 slope is made by the unit adjustment, not by the states.**
The pooled L16 number-token fit (`steering.fit_duration_effects`) holds unit, regime, option order and delays at their
sample average. Per-horizon raw means on PC1 have a slope of -0.18 per decade; the fitted curve has -9.7. Of that slope,
-9.40 per decade comes from the unit factor alone (unit effects on PC1 run from -49 for seconds to +25 for years, against
a raw span of -13 to +9). Unit and horizon are confounded on the bank (`01_plan.md` §7: unit explains 0.97 of log
horizon), so "average the unit out" shifts short horizons up and long ones down by construction; the unit effects
are identified only from the few durations written under several unit wordings (up to four). The number tokens come before
the unit word, so a "unit effect" on them is something that tracks the unit (digits, magnitude), not a read of the word;
untested. Consequences: the plotted PC1 trend overstates what the model's states do. The patch steps are not affected:
both horizons of a pair share the unit, so the offset cancels (step length 18.5 fitted vs 22.8 raw means over the 24 pairs,
cosine median 0.99, min 0.71). Anything that uses the absolute curve across units (the scan's targets, cross-unit steps)
inherits the confound.

**Periodicity.** Lomb-Scargle with a shuffle null on the maximum power over periods 0.4-4.5 decades: PC2 shows no
significant period (p 0.97 fitted, 0.52 raw means) although it looks sinusoidal; PC5 has one at 0.95 decades
(p 0.009 fitted, 0.005 raw means; about 0.015-0.03 after correcting for the three PCs tried), present in the raw means,
so not an artefact of the adjustment. A period of one decade fits a state that depends on the number's leading digits;
not tested (next: label PC5 by leading digit, split-half refit).

**What 27 adds to E14's donor patch** (E14: whole-phrase donor patch at L16, 104/104 traces write a target-regime
horizon, +9.9 near log-odds; but 95 of 104 donors changed the unit, so unit and number moved together):
- It separates number from unit. Same-unit pairs: number tokens alone 24/24, unit token alone 0/24 (E10 had patched the
  unit-word position with the top-8 subspace only).
- It adds depth for the number tokens (L12, L16, L22 all 24/24); E14 was L16 only.
- It adds the controls E14 lacked: self patch, rerun noise floor, in-span random, top-8 subspace (20/24).
- It gives the scan (26) the positive control it was missing: the same manifold step at the number tokens writes the donor's
  number in 3/24 against 24/24 for the donor states, so the scan's null was the direction, not the placement.
It does not show where the choice is computed (a patch that makes the trace write another horizon moves the choice, as
in E14), and it is 24 pairs. 28 and 29 do not extend the patching: 28 is about forced closes, 29 about implied horizons.
